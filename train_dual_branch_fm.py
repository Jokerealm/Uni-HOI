#!/usr/bin/env python3
"""
Train the dual-branch co-generative Flow Matching model for 4D HOI reconstruction.

The training graph is unified:

- Video branch:
  defaults to a frozen Wan TI2V teacher that predicts RGB latent velocities.
- State branch:
  predicts latent velocities for HOI state tokens
  (human/object Gaussians, joints, object motion, contact).
- Cross-branch supervision:
  object render consistency, 3D->2D geometry consistency, and late teacher coupling.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model.dual_branch_cogenerative_fm import DecodedHOIState, DualBranchCoGenerativeFlowMatching
from model.dual_branch_wan_teacher_fm import DualBranchWanTeacherFlowMatching
from model.joint_renderer_loss import DiffRasterizationLayer

LOSS_NAMES = (
    "fm",
    "geo_joint_heat",
    "geo_object_silhouette",
    "geo_depth",
    "human_gaussian",
    "object_gaussian",
    "joints",
    "reg_pose",
    "reg_shape",
    "reg_translation",
    "reg_object_pose",
    "reg_contact",
    "depth",
    "temp_object",
    "temp_contact",
    "phys_contact",
    "phys_penetration",
)

VIDEO_TEACHER_STATE_PREFIXES = ("_video_teacher.",)


def filter_video_teacher_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(VIDEO_TEACHER_STATE_PREFIXES)
    }

STAGE0_LOSS_NAMES = (
    "fm",
    "geo_joint_heat",
    "geo_object_silhouette",
    "geo_depth",
    "human_gaussian",
    "object_gaussian",
    "joints",
)

STAGE1_LOSS_NAMES = STAGE0_LOSS_NAMES

STAGE2_EXTRA_LOSS_NAMES = (
    "reg_pose",
    "reg_shape",
    "reg_translation",
    "reg_object_pose",
    "reg_contact",
    "depth",
    "temp_object",
    "temp_contact",
    "phys_contact",
    "phys_penetration",
)

STAGE2_LOSS_NAMES = STAGE1_LOSS_NAMES + STAGE2_EXTRA_LOSS_NAMES

LOSS_PRESETS: dict[str, tuple[str, ...]] = {
    "stage0": STAGE0_LOSS_NAMES,
    "stage1": STAGE1_LOSS_NAMES,
    "stage2": STAGE2_LOSS_NAMES,
    "full": LOSS_NAMES,
}

_HONEST_RENDER_MODULE = None


def decode_gaussian_params(tokens: Tensor) -> Dict[str, Tensor]:
    return {
        "means": tokens[..., 0:3],
        "rotations": F.normalize(tokens[..., 3:7], dim=-1),
        "scales": tokens[..., 7:10].clamp(min=1e-6),
        "opacities": tokens[..., 10:11].clamp(0.0, 1.0),
        "shs": tokens[..., 11:14].clamp(0.0, 1.0),
    }


def resize_video_batch(video: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
    if video.ndim != 5:
        raise ValueError(f"`video` must have shape [B, T, C, H, W], got {tuple(video.shape)}.")
    batch_size, num_frames, channels, _, _ = video.shape
    video = video.reshape(batch_size * num_frames, channels, video.shape[-2], video.shape[-1])
    video = F.interpolate(
        video,
        size=size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return video.reshape(batch_size, num_frames, channels, size[0], size[1])


def downsample_spatial_map(video: Tensor, size: Tuple[int, int]) -> Tensor:
    return resize_video_batch(video, size=size, mode="bilinear")


def scale_camera_intrinsics(camera_intrinsics: Tensor, source_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tensor:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return camera_intrinsics
    scaled = camera_intrinsics.clone()
    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    scaled[..., 0, 0] = scaled[..., 0, 0] * scale_x
    scaled[..., 1, 1] = scaled[..., 1, 1] * scale_y
    scaled[..., 0, 2] = scaled[..., 0, 2] * scale_x
    scaled[..., 1, 2] = scaled[..., 1, 2] * scale_y
    return scaled


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _unwrap_scheduler(scheduler):
    return getattr(scheduler, "scheduler", scheduler)


def _realign_scheduler_to_step(
    scheduler,
    optimizer: AdamW,
    step: int,
) -> None:
    raw_scheduler = _unwrap_scheduler(scheduler)
    step = max(int(step), 0)

    # Keep scheduler progress tied to our explicit optimizer-step counter so
    # resume behavior stays stable even if the process count changes.
    if hasattr(raw_scheduler, "lr_lambdas") and hasattr(raw_scheduler, "base_lrs"):
        lr_lambdas = list(raw_scheduler.lr_lambdas)
        lrs = [
            float(base_lr) * float(lr_lambdas[idx](step))
            for idx, base_lr in enumerate(raw_scheduler.base_lrs)
        ]
        for param_group, lr in zip(optimizer.param_groups, lrs):
            param_group["lr"] = lr
        raw_scheduler.base_lrs = [float(base_lr) for base_lr in raw_scheduler.base_lrs]
        raw_scheduler.last_epoch = step
        raw_scheduler._step_count = step + 1
        raw_scheduler._last_lr = lrs
        return

    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = float(param_group.get("initial_lr", param_group["lr"]))
    raw_scheduler.last_epoch = step
    raw_scheduler._step_count = step + 1
    try:
        raw_scheduler.step(step)
    except TypeError:
        pass


def collect_trainable_parameters(*modules: nn.Module) -> Tuple[nn.Parameter, ...]:
    params = []
    for module in modules:
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    return tuple(params)


def count_parameters(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _json_safe(value):
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        return detached.reshape(-1).tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_rank_debug_status(
    status_path: Optional[Path],
    *,
    phase: str,
    rank: int,
    world_size: int,
    step: int,
    sequence_names: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    if status_path is None:
        return
    payload = {
        "timestamp_unix": time.time(),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": int(rank),
        "world_size": int(world_size),
        "phase": str(phase),
        "global_step": int(step),
    }
    if sequence_names is not None:
        payload["sequence_names"] = [str(name) for name in sequence_names]
    if extra:
        payload.update({str(key): _json_safe(value) for key, value in extra.items()})
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = status_path.with_suffix(f"{status_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, status_path)
    except OSError:
        return


def reduce_scalar_metrics(accelerator: Accelerator, metrics: Dict[str, Tensor]) -> Dict[str, float]:
    metric_items = list(metrics.items())
    if not metric_items:
        return {}

    metric_names = []
    metric_scalars = []
    for name, value in metric_items:
        scalar = value.detach()
        if scalar.numel() != 1:
            raise ValueError(f"Metric `{name}` must be scalar, got shape {tuple(scalar.shape)}.")
        metric_names.append(name)
        metric_scalars.append(scalar.to(device=accelerator.device, dtype=torch.float32).reshape(()))

    reduced = accelerator.reduce(torch.stack(metric_scalars), reduction="mean")
    return {
        name: float(reduced[idx].item())
        for idx, name in enumerate(metric_names)
    }


def build_wandb_loss_metrics(
    reduced_metrics: Dict[str, float],
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    metrics_to_log: Dict[str, float] = {}
    if "loss_total" in reduced_metrics:
        metrics_to_log["loss_total"] = float(reduced_metrics["loss_total"])
    if "loss_video_fm" in reduced_metrics:
        metrics_to_log["loss_video_fm"] = float(reduced_metrics["loss_video_fm"])

    for loss_name, weight in loss_weights.items():
        if float(weight) <= 0.0:
            continue
        metric_name = f"loss_{loss_name}"
        if metric_name in reduced_metrics and metric_name not in metrics_to_log:
            metrics_to_log[metric_name] = float(reduced_metrics[metric_name])

    return metrics_to_log


def infer_condition_channels(dataset: DualBranchHOIDataset) -> int:
    return int(dataset.condition_channels)


def model_uses_wan_teacher(model_or_args) -> bool:
    backend = getattr(model_or_args, "video_backend", None)
    if backend is None:
        backend = getattr(model_or_args, "video_backend", "legacy_codec")
    return str(backend) == "wan_ti2v_5b"


def resolve_wan_teacher_num_frames(*, clip_length: int, pad_to_compatible_frames: bool) -> int:
    clip_length = int(clip_length)
    if clip_length <= 0:
        raise ValueError(f"`clip_length` must be > 0, got {clip_length}.")
    if (clip_length - 1) % 4 == 0:
        return clip_length
    if not bool(pad_to_compatible_frames):
        raise ValueError(
            "`Wan-AI/Wan2.2-TI2V-5B` requires `clip_length = 4k + 1` unless "
            "`--wan_pad_to_compatible_frames` is enabled. "
            f"Got clip_length={clip_length}."
        )
    return ((clip_length - 1 + 3) // 4) * 4 + 1


def build_model_from_args(
    args: argparse.Namespace,
    *,
    condition_channels: int,
):
    if model_uses_wan_teacher(args):
        teacher_num_frames = resolve_wan_teacher_num_frames(
            clip_length=args.clip_length,
            pad_to_compatible_frames=bool(getattr(args, "wan_pad_to_compatible_frames", True)),
        )
        return DualBranchWanTeacherFlowMatching(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            condition_channels=condition_channels,
            patch_size=args.patch_size,
            condition_patch_size=args.condition_patch_size,
            num_frames=args.clip_length,
            image_height=args.image_height,
            image_width=args.image_width,
            num_human_gaussians=args.num_human_gaussians,
            num_object_gaussians=args.num_object_gaussians,
            num_joints=args.num_joints,
            contact_dim=args.contact_dim,
            human_shape_dim=args.human_shape_dim,
            human_pose_dim=args.human_pose_dim,
            wan_model_id=args.wan_model_id,
            wan_dtype=args.wan_dtype,
            wan_prompt_max_sequence_length=args.wan_prompt_max_sequence_length,
            wan_prompt_override=args.wan_prompt_override,
            wan_local_files_only=args.wan_local_files_only,
            wan_hidden_dim=args.wan_hidden_dim,
            teacher_num_frames=teacher_num_frames,
        )
    return DualBranchCoGenerativeFlowMatching(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        condition_channels=condition_channels,
        video_channels=args.video_channels,
        patch_size=args.patch_size,
        num_frames=args.clip_length,
        image_height=args.image_height,
        image_width=args.image_width,
        num_human_gaussians=args.num_human_gaussians,
        num_object_gaussians=args.num_object_gaussians,
        num_joints=args.num_joints,
        contact_dim=args.contact_dim,
        human_shape_dim=args.human_shape_dim,
        human_pose_dim=args.human_pose_dim,
    )


def ensure_video_teacher_ready(model, device: torch.device) -> None:
    if model_uses_wan_teacher(model):
        model.ensure_video_teacher(device)


def sample_video_prior_latents(
    model,
    *,
    batch_size: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if model_uses_wan_teacher(model):
        return model.sample_video_prior(
            batch_size,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    if dtype is None:
        dtype = next(model.parameters()).dtype
    return torch.randn(
        batch_size,
        model.video_codec.num_frames * model.video_codec.num_patches_per_frame,
        model.hidden_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def broadcast_timesteps_like(timesteps: Tensor, target: Tensor) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}.")
    if timesteps.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch size mismatch between timesteps and target, got {timesteps.shape[0]} and {target.shape[0]}."
        )
    return timesteps.float().view(timesteps.shape[0], *([1] * (target.ndim - 1)))


def flow_match_sample(target: Tensor, noise: Tensor, timesteps: Tensor) -> Tuple[Tensor, Tensor]:
    if target.shape != noise.shape:
        raise ValueError(f"`target` and `noise` must share shape, got {tuple(target.shape)} and {tuple(noise.shape)}.")
    t_view = broadcast_timesteps_like(timesteps, target).to(device=target.device, dtype=target.dtype)
    xt = t_view * target + (1.0 - t_view) * noise
    velocity = target - noise
    return xt, velocity


class SequenceBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: DualBranchHOIDataset,
        *,
        batch_size: int,
        drop_last: bool,
        seed: int,
        shuffle: bool = True,
        warm_start_short_sequences: bool = True,
        interleave_window: int = 1,
        max_batches_per_sequence: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.warm_start_short_sequences = bool(warm_start_short_sequences)
        self.interleave_window = max(int(interleave_window), 1)
        self.max_batches_per_sequence = int(max_batches_per_sequence)
        if self.max_batches_per_sequence <= 0:
            raise ValueError(
                "`max_batches_per_sequence` must be > 0, "
                f"got {self.max_batches_per_sequence}."
            )
        self._epoch = 0
        self._all_sequence_dirs = [
            sequence_dir
            for sequence_dir in dataset.sequence_dirs
            if dataset.sequence_sample_indices.get(sequence_dir)
        ]

    def _ordered_sequence_dirs_for_epoch(self, epoch: int) -> list[str]:
        sequence_dirs = list(self._all_sequence_dirs)
        if self.shuffle and epoch == 0 and self.warm_start_short_sequences:
            sequence_dirs.sort(
                key=lambda sequence_dir: (
                    self.dataset.sequence_frame_counts[sequence_dir],
                    sequence_dir,
                )
            )
        elif self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(sequence_dirs)
        return sequence_dirs

    def __len__(self) -> int:
        total = 0
        for sequence_dir in self._ordered_sequence_dirs_for_epoch(self._epoch):
            num_samples = len(self.dataset.sequence_sample_indices[sequence_dir])
            if self.drop_last:
                total += num_samples // self.batch_size
            else:
                total += (num_samples + self.batch_size - 1) // self.batch_size
        return total

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(int(epoch), 0)

    def _sequence_batches_for_epoch(
        self,
        epoch: int,
        rng: random.Random,
    ) -> list[tuple[str, list[list[int]]]]:
        sequence_batches: list[tuple[str, list[list[int]]]] = []
        for sequence_dir in self._ordered_sequence_dirs_for_epoch(epoch):
            sample_indices = list(self.dataset.sequence_sample_indices[sequence_dir])
            if self.shuffle:
                rng.shuffle(sample_indices)
            batches: list[list[int]] = []
            for batch_start in range(0, len(sample_indices), self.batch_size):
                batch_indices = sample_indices[batch_start : batch_start + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch_indices)
            if batches:
                sequence_batches.append((sequence_dir, batches))
        return sequence_batches

    def _iter_draining_sequence_batches(
        self,
        sequence_batches: Sequence[tuple[str, Sequence[list[int]]]],
    ):
        for _, batches in sequence_batches:
            for batch_indices in batches:
                yield batch_indices

    def _iter_interleaved_sequence_batches(
        self,
        sequence_batches: Sequence[tuple[str, Sequence[list[int]]]],
        rng: random.Random,
    ):
        # Keep only a bounded set of active sequences hot at once so we stagger
        # large sequence bundle loads across ranks without destroying cache locality.
        pending_entries = collections.deque(
            {
                "sequence_dir": sequence_dir,
                "batches": list(batches),
                "next_batch_index": 0,
            }
            for sequence_dir, batches in sequence_batches
        )
        active_entries: list[dict[str, object]] = []
        while pending_entries and len(active_entries) < self.interleave_window:
            active_entries.append(pending_entries.popleft())

        round_index = 0
        while active_entries:
            if len(active_entries) > 1:
                if self.shuffle:
                    start_offset = rng.randrange(len(active_entries))
                else:
                    start_offset = round_index % len(active_entries)
                ordered_entries = active_entries[start_offset:] + active_entries[:start_offset]
            else:
                ordered_entries = list(active_entries)

            next_active_entries: list[dict[str, object]] = []
            for entry in ordered_entries:
                batches = entry["batches"]
                next_batch_index = int(entry["next_batch_index"])
                burst_count = 0
                while next_batch_index < len(batches) and burst_count < self.max_batches_per_sequence:
                    yield batches[next_batch_index]
                    next_batch_index += 1
                    burst_count += 1
                if next_batch_index < len(batches):
                    entry["next_batch_index"] = next_batch_index
                    next_active_entries.append(entry)

            while pending_entries and len(next_active_entries) < self.interleave_window:
                next_active_entries.append(pending_entries.popleft())

            active_entries = next_active_entries
            round_index += 1

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        rng = random.Random(self.seed + epoch)
        sequence_batches = self._sequence_batches_for_epoch(epoch, rng)
        if self.interleave_window <= 1 or len(sequence_batches) <= 1:
            yield from self._iter_draining_sequence_batches(sequence_batches)
            return
        yield from self._iter_interleaved_sequence_batches(sequence_batches, rng)


def normalize_batch_sampler_mode(mode: str) -> str:
    text = str(mode).strip().lower().replace("-", "_")
    if text in {"global", "global_shuffle", "global_clip_shuffle", "clip_shuffle"}:
        return "global_clip_shuffle"
    if text in {"sequence", "sequence_grouped", "sequence_batch"}:
        return "sequence_grouped"
    raise ValueError(
        "`batch_sampler_mode` must be one of "
        "`global_clip_shuffle` or `sequence_grouped`, "
        f"got {mode!r}."
    )


def resolve_sequence_batch_interleave_window(
    requested_window: int,
    *,
    world_size: int,
    cache_sequences: int,
) -> int:
    requested_window = int(requested_window)
    if requested_window > 0:
        return requested_window
    if world_size <= 1:
        return 1
    return max(2, int(cache_sequences))


def configure_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def resolve_mixed_precision_dtype(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return None


def probe_flash_attention(
    *,
    device: torch.device,
    mixed_precision: str,
    hidden_dim: int,
    num_heads: int,
) -> tuple[bool, str]:
    if device.type != "cuda":
        return False, "cuda_required"
    if hidden_dim % num_heads != 0:
        return False, f"hidden_dim_not_divisible_by_num_heads:{hidden_dim}/{num_heads}"
    dtype = resolve_mixed_precision_dtype(mixed_precision)
    if dtype is None:
        return False, "mixed_precision=no"
    if not hasattr(torch.backends, "cuda"):
        return False, "torch.backends.cuda_unavailable"
    if hasattr(torch.backends.cuda, "is_flash_attention_available") and not torch.backends.cuda.is_flash_attention_available():
        return False, "flash_attention_backend_unavailable"
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        return False, "sdp_kernel_api_unavailable"

    head_dim = hidden_dim // num_heads
    q = torch.randn(2, num_heads, 256, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        torch.cuda.synchronize()
        return True, f"dtype={dtype} head_dim={head_dim}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def report_attention_runtime(args: argparse.Namespace, accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        return
    flash_available = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "is_flash_attention_available"):
        flash_available = bool(torch.backends.cuda.is_flash_attention_available())
    flash_ok, detail = probe_flash_attention(
        device=accelerator.device,
        mixed_precision=args.mixed_precision,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    print(
        "[train_dual_branch_fm] attention runtime "
        f"| mixed_precision={args.mixed_precision} "
        f"| flash_available={int(flash_available)} "
        f"| flash_probe={int(flash_ok)} "
        f"| detail={detail}",
        flush=True,
    )
    if args.mixed_precision == "no":
        print(
            "[train_dual_branch_fm] warning: mixed_precision=no disables FlashAttention-compatible dtypes.",
            flush=True,
        )


def render_object_branch(
    renderer: DiffRasterizationLayer,
    object_gaussians: Tensor,
    object_poses: Tensor,
    camera_intrinsics: Tensor,
) -> Tensor:
    device_type = object_gaussians.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return renderer(
            decode_gaussian_params(object_gaussians.float()),
            object_poses=object_poses.float(),
            camera_intrinsics=camera_intrinsics.float(),
        )


def _subsample_vertex_trajectories(vertices: Tensor, num_points: int) -> Tensor:
    if vertices.ndim != 4 or vertices.shape[-1] != 3:
        raise ValueError(f"`vertices` must have shape [B, T, V, 3], got {tuple(vertices.shape)}.")
    total_vertices = vertices.shape[2]
    if total_vertices <= num_points:
        return vertices
    indices = torch.linspace(0, total_vertices - 1, steps=num_points, device=vertices.device)
    indices = indices.round().long().clamp(min=0, max=total_vertices - 1)
    return vertices.index_select(2, indices)


def render_human_proxy_branch(
    renderer: DiffRasterizationLayer,
    human_vertices: Tensor,
    camera_intrinsics: Tensor,
    *,
    num_points: int,
    gaussian_scale: float,
    gray_value: float = 102.0 / 255.0,
    opacity_value: float = 0.95,
) -> Tensor:
    sampled_vertices = _subsample_vertex_trajectories(human_vertices.float(), max(int(num_points), 1))
    batch_size, num_frames, num_vertices = sampled_vertices.shape[:3]
    identity_poses = torch.eye(4, device=human_vertices.device, dtype=human_vertices.dtype).view(1, 1, 4, 4)
    identity_poses = identity_poses.expand(batch_size, num_frames, 4, 4)
    rotations = human_vertices.new_zeros(batch_size, num_frames, num_vertices, 4)
    rotations[..., 0] = 1.0
    scales = human_vertices.new_full((batch_size, num_frames, num_vertices, 3), float(gaussian_scale))
    opacities = human_vertices.new_full((batch_size, num_frames, num_vertices, 1), float(opacity_value))
    shs = human_vertices.new_full((batch_size, num_frames, num_vertices, 3), float(gray_value))
    device_type = human_vertices.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return renderer(
            {
                "xyz": sampled_vertices.float(),
                "rotation": rotations.float(),
                "scaling": scales.float(),
                "opacity": opacities.float(),
                "shs": shs.float(),
            },
            object_poses=identity_poses.float(),
            camera_intrinsics=camera_intrinsics.float(),
        )


def build_teacher_state(batch: Dict[str, Tensor]) -> DecodedHOIState:
    return DecodedHOIState(
        human_shape=batch["human_shape"],
        human_pose=batch["body_pose"],
        human_translation=batch["cam_t"],
        human_gaussians=batch["human_gaussians"],
        object_gaussians=batch["object_gaussians"],
        joints_3d=batch["joints_3d"],
        object_transforms=batch["object_poses"],
        contact_signature=batch["contact_signature"],
    )


def build_object_supervision_target(
    *,
    object_visible: Tensor,
    teacher_object_render: Tensor,
    m_primary: Tensor,
    m_secondary: Tensor,
    m_object_region: Tensor,
    visible_weight: float,
    primary_weight: float,
    secondary_weight: float,
) -> Tuple[Tensor, Tensor]:
    visible_mask = m_object_region.float().clamp(0.0, 1.0)
    primary_mask = m_primary.float().clamp(0.0, 1.0)
    secondary_mask = m_secondary.float().clamp(0.0, 1.0)
    occluded_mask = (primary_mask + secondary_mask).clamp(0.0, 1.0)
    supervision_target = object_visible.float() * visible_mask + teacher_object_render.float() * occluded_mask
    supervision_weights = (
        float(visible_weight) * visible_mask
        + float(primary_weight) * primary_mask
        + float(secondary_weight) * secondary_mask
    )
    return supervision_target, supervision_weights


def build_human_supervision_target(
    *,
    human_visible: Tensor,
    masks_human: Tensor,
    human_proxy_render: Tensor,
    visible_weight: float,
    completion_weight: float,
    proxy_gray_value: float = 102.0 / 255.0,
) -> Tuple[Tensor, Tensor]:
    visible_mask = masks_human.float().clamp(0.0, 1.0)
    proxy_strength = (1.0 - human_proxy_render.float().mean(dim=2, keepdim=True)) / max(1.0 - float(proxy_gray_value), 1e-4)
    proxy_mask = proxy_strength.clamp(0.0, 1.0)
    completion_mask = (proxy_mask - visible_mask).clamp(0.0, 1.0)
    supervision_target = human_visible.float() * visible_mask + human_proxy_render.float() * completion_mask
    supervision_weights = (
        float(visible_weight) * visible_mask
        + float(completion_weight) * completion_mask
    )
    return supervision_target, supervision_weights


def compute_masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.expand_as(target)
    denom = mask.sum().clamp(min=1.0)
    return ((prediction - target).abs() * mask).sum() / denom


def compute_weighted_l1(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    weights = weights.expand_as(target).clamp(min=0.0)
    denom = weights.sum().clamp(min=1.0)
    return ((prediction - target).abs() * weights).sum() / denom


def compute_weighted_temporal_l1(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    temporal_prediction = prediction[:, 1:] - prediction[:, :-1]
    temporal_target = target[:, 1:] - target[:, :-1]
    temporal_weights = torch.maximum(weights[:, 1:], weights[:, :-1])
    return compute_weighted_l1(temporal_prediction, temporal_target, temporal_weights)


def density_to_occupancy(density: Tensor) -> Tensor:
    return 1.0 - torch.exp(-density.clamp(min=0.0))


def transform_object_points(object_gaussians: Tensor, object_poses: Tensor) -> Tensor:
    object_xyz = object_gaussians[..., :3]
    object_h = torch.cat([object_xyz, torch.ones_like(object_xyz[..., :1])], dim=-1)
    world = torch.matmul(object_poses.unsqueeze(2), object_h.unsqueeze(1).unsqueeze(-1)).squeeze(-1)[..., :3]
    return world


def compute_second_order_smoothness(points: Tensor) -> Tensor:
    if points.shape[1] < 3:
        return points.new_zeros(())
    accel = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    return accel.abs().mean()


def contact_activation(contact_signature: Tensor) -> Tensor:
    if contact_signature.shape[-1] >= 4:
        return contact_signature[..., -2:].sigmoid().mean(dim=-1, keepdim=True)
    return contact_signature.sigmoid().mean(dim=-1, keepdim=True)


def compute_contact_relative_velocity_loss(
    joints_3d: Tensor,
    object_world_points: Tensor,
    contact_signature: Tensor,
) -> Tensor:
    if joints_3d.shape[1] < 2:
        return joints_3d.new_zeros(())
    hand_span = min(4, joints_3d.shape[2])
    hand_points = joints_3d[:, :, -hand_span:]
    hand_center = hand_points.mean(dim=2)
    object_center = object_world_points.mean(dim=2)
    relative_velocity = (hand_center[:, 1:] - hand_center[:, :-1]) - (object_center[:, 1:] - object_center[:, :-1])
    weights = torch.maximum(contact_activation(contact_signature)[:, 1:], contact_activation(contact_signature)[:, :-1])
    return (relative_velocity.norm(dim=-1, keepdim=True) * weights).sum() / weights.sum().clamp(min=1.0)


def compute_contact_distance_loss(
    joints_3d: Tensor,
    object_world_points: Tensor,
    teacher_contact_signature: Tensor,
) -> Tensor:
    hand_span = min(4, joints_3d.shape[2])
    hand_points = joints_3d[:, :, -hand_span:]
    distances = torch.cdist(
        hand_points.reshape(-1, hand_span, 3),
        object_world_points.reshape(-1, object_world_points.shape[2], 3),
    ).min(dim=-1).values
    distances = distances.view(joints_3d.shape[0], joints_3d.shape[1], hand_span)
    target_weights = contact_activation(teacher_contact_signature)
    return (distances.mean(dim=-1, keepdim=True) * target_weights).sum() / target_weights.sum().clamp(min=1.0)


def compute_penetration_loss(
    joints_3d: Tensor,
    object_world_points: Tensor,
    *,
    margin: float = 0.02,
) -> Tensor:
    non_hand_count = max(joints_3d.shape[2] - min(4, joints_3d.shape[2]), 1)
    body_points = joints_3d[:, :, :non_hand_count]
    distances = torch.cdist(
        body_points.reshape(-1, non_hand_count, 3),
        object_world_points.reshape(-1, object_world_points.shape[2], 3),
    ).min(dim=-1).values
    penetration = torch.relu(float(margin) - distances)
    return penetration.mean()


def compute_point_set_chamfer(prediction_xyz: Tensor, target_xyz: Tensor) -> Tensor:
    if prediction_xyz.ndim != 3 or target_xyz.ndim != 3:
        raise ValueError(
            f"`prediction_xyz` and `target_xyz` must have shape [B, N, 3] / [B, M, 3], "
            f"got {tuple(prediction_xyz.shape)} and {tuple(target_xyz.shape)}."
        )
    distances = torch.cdist(prediction_xyz, target_xyz)
    return distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean()


def compute_gaussian_attr_nn_loss(
    prediction_tokens: Tensor,
    target_tokens: Tensor,
    *,
    attr_slice: slice = slice(7, 11),
) -> Tensor:
    if prediction_tokens.ndim != 3 or target_tokens.ndim != 3:
        raise ValueError(
            f"`prediction_tokens` and `target_tokens` must have shape [B, N, C], got "
            f"{tuple(prediction_tokens.shape)} and {tuple(target_tokens.shape)}."
        )
    prediction_xyz = prediction_tokens[..., :3]
    target_xyz = target_tokens[..., :3]
    distances = torch.cdist(prediction_xyz, target_xyz)
    pred_nn = distances.argmin(dim=-1)
    tgt_nn = distances.argmin(dim=-2)

    pred_attr = prediction_tokens[..., attr_slice]
    target_attr = target_tokens[..., attr_slice]

    matched_target_attr = torch.gather(
        target_attr,
        1,
        pred_nn.unsqueeze(-1).expand(-1, -1, target_attr.shape[-1]),
    )
    matched_pred_attr = torch.gather(
        pred_attr,
        1,
        tgt_nn.unsqueeze(-1).expand(-1, -1, pred_attr.shape[-1]),
    )
    loss_pred_to_target = F.smooth_l1_loss(pred_attr, matched_target_attr)
    loss_target_to_pred = F.smooth_l1_loss(matched_pred_attr, target_attr)
    return loss_pred_to_target + loss_target_to_pred


def _gaussian_tokens_to_cloud(tokens: Tensor) -> np.ndarray:
    raw = tokens.detach().float().cpu()
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError(f"Expected batched Gaussian tokens with batch=1, got {tuple(raw.shape)}.")
        raw = raw.squeeze(0)
    if raw.ndim != 2 or raw.shape[-1] != 14:
        raise ValueError(f"Expected Gaussian tokens with shape [N, 14], got {tuple(raw.shape)}.")
    xyz = raw[:, 0:3].numpy().astype(np.float32)
    rgb = raw[:, 11:14].clamp(0.0, 1.0).numpy().astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def _transform_cloud(cloud: np.ndarray, transform: Tensor) -> np.ndarray:
    transform_np = transform.detach().cpu().numpy().astype(np.float32)
    xyz = cloud[:, :3]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_world = (transform_np @ xyz_h.T).T[:, :3]
    out = cloud.copy()
    out[:, :3] = xyz_world
    return out


def _load_honest_render_module():
    global _HONEST_RENDER_MODULE
    if _HONEST_RENDER_MODULE is not None:
        return _HONEST_RENDER_MODULE
    module_path = Path(__file__).resolve().parent / "scripts" / "render_dual_branch_pointclouds.py"
    spec = importlib.util.spec_from_file_location("dual_branch_honest_render", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load honest render module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HONEST_RENDER_MODULE = module
    return module


def _resolve_honest_validation_sequence_dir(
    requested_sequence: str,
    sequence_dirs: list[str],
) -> str:
    if not sequence_dirs:
        raise RuntimeError("No dataset sequences are available for honest 3D validation.")
    if not requested_sequence:
        return sequence_dirs[0]
    requested = requested_sequence.strip()
    for sequence_dir in sequence_dirs:
        if sequence_dir == requested or Path(sequence_dir).name == requested:
            return sequence_dir
    available = ", ".join(Path(path).name for path in sequence_dirs[:10])
    raise KeyError(
        f"Could not resolve honest validation sequence `{requested_sequence}`. "
        f"Available examples: {available}"
    )


def _save_validation_gaussian_tokens(tokens: Tensor, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = tokens.detach().cpu()
    if raw.ndim == 3:
        raw = raw.squeeze(0)
    payload = {
        "xyz": raw[:, 0:3],
        "rotation": raw[:, 3:7],
        "scaling": raw[:, 7:10],
        "opacity": raw[:, 10:11],
        "shs": raw[:, 11:14],
        "raw": raw,
        "metadata": metadata,
    }
    torch.save(payload, path)


def _save_validation_combined_state(
    decoded_state: DecodedHOIState,
    output_dir: Path,
    *,
    sequence_name: str,
    num_ode_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "smpl": {
            "shape": decoded_state.human_shape.squeeze(0).detach().cpu(),
            "pose": decoded_state.human_pose.squeeze(0).detach().cpu(),
            "translation": decoded_state.human_translation.squeeze(0).detach().cpu(),
        },
        "G_h": {"raw": decoded_state.human_gaussians.squeeze(0).detach().cpu()},
        "G_o": {"raw": decoded_state.object_gaussians.squeeze(0).detach().cpu()},
        "motion": {
            "joints_3d": decoded_state.joints_3d.squeeze(0).detach().cpu(),
            "object_poses": decoded_state.object_transforms.squeeze(0).detach().cpu(),
            "contact_signature": decoded_state.contact_signature.squeeze(0).detach().cpu(),
        },
        "metadata": {
            "sequence_name": sequence_name,
            "num_frames": int(decoded_state.joints_3d.shape[1]),
            "num_ode_steps": int(num_ode_steps),
        },
    }
    torch.save(payload, output_dir / "gs_init_combined.pt")


@torch.no_grad()
def run_honest_3d_validation(
    *,
    model,
    dataset: DualBranchHOIDataset,
    sequence_dir: str,
    device: torch.device,
    output_dir: str,
    step: int,
    args: argparse.Namespace,
    wandb_enabled: bool,
    wandb_run: Optional[object] = None,
) -> Dict[str, float]:
    render_module = _load_honest_render_module()
    render_args = argparse.Namespace(
        overwrite=True,
        image_size=512,
        radius=0.012,
        points_per_pixel=12,
        num_frames=24,
        fps=12,
        elev=10.0,
        dist=1.9,
        azim_start=180.0,
        focal_scale=3.0,
        scene_scale=0.85,
        quantile=0.02,
    )

    bundle = dataset.get_sequence_bundle(sequence_dir)
    sample = dataset.get_sample(sequence_dir, 0, bundle=bundle)
    sequence_name = str(sample["sequence_name"])
    step_dir = Path(output_dir) / "honest_validations" / f"step_{step:07d}" / sequence_name
    gs_dir = step_dir / "pred_state"
    render_dir = step_dir / "honest_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    rgb = sample["rgb"].unsqueeze(0).to(device)
    masks_human = sample["masks_human"].unsqueeze(0).to(device)
    masks_object = sample["masks_object"].unsqueeze(0).to(device)
    depth = sample["depth"].unsqueeze(0).to(device)
    m_primary = sample["m_primary"].unsqueeze(0).to(device)
    m_secondary = sample["m_secondary"].unsqueeze(0).to(device)
    m_object_region = sample["m_object_region"].unsqueeze(0).to(device)
    keypoint_heatmaps = sample["keypoint_heatmaps"].unsqueeze(0).to(device)
    camera_intrinsics = sample["camera_intrinsics"].unsqueeze(0).to(device)

    if rgb.shape[-2:] != (args.image_height, args.image_width):
        source_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        rgb = resize_video_batch(rgb, size=(args.image_height, args.image_width), mode="bilinear")
        masks_human = resize_video_batch(masks_human, size=(args.image_height, args.image_width), mode="nearest")
        masks_object = resize_video_batch(masks_object, size=(args.image_height, args.image_width), mode="nearest")
        depth = resize_video_batch(depth, size=(args.image_height, args.image_width), mode="bilinear")
        m_primary = resize_video_batch(m_primary, size=(args.image_height, args.image_width), mode="nearest")
        m_secondary = resize_video_batch(m_secondary, size=(args.image_height, args.image_width), mode="nearest")
        m_object_region = resize_video_batch(
            m_object_region,
            size=(args.image_height, args.image_width),
            mode="nearest",
        )
        keypoint_heatmaps = resize_video_batch(
            keypoint_heatmaps,
            size=(args.image_height, args.image_width),
            mode="bilinear",
        )
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics,
            source_size=source_hw,
            target_size=(args.image_height, args.image_width),
        )

    condition_video = torch.cat(
        [
            rgb,
            masks_human,
            masks_object,
            depth,
            m_primary,
            m_secondary,
            m_object_region,
            keypoint_heatmaps,
        ],
        dim=2,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))

    use_wan_teacher = model_uses_wan_teacher(model)
    condition_latents = None
    if use_wan_teacher:
        condition_latents = model.encode_video_target(rgb)
        human_video_latents = sample_video_prior_latents(
            model,
            batch_size=1,
            generator=generator,
            device=device,
        ) * float(args.honest_val_prior_noise_std)
        object_video_latents = sample_video_prior_latents(
            model,
            batch_size=1,
            generator=generator,
            device=device,
        ) * float(args.honest_val_prior_noise_std)
        video_latents = human_video_latents
    else:
        video_latents = sample_video_prior_latents(
            model,
            batch_size=1,
            generator=generator,
            device=device,
        ) * float(args.honest_val_prior_noise_std)
    state_latents = torch.randn(
        1,
        model.state_codec.total_tokens,
        args.hidden_dim,
        generator=generator,
        device=device,
    ) * float(args.honest_val_prior_noise_std)

    was_training = model.training
    model.eval()
    try:
        times = torch.linspace(
            0.0,
            1.0,
            int(args.honest_val_num_ode_steps) + 1,
            device=device,
            dtype=video_latents.dtype,
        )
        state_to_video_scale = resolve_state_to_video_scale(args, step)
        video_to_state_scale = resolve_video_to_state_scale(args, step)
        for ode_step in range(int(args.honest_val_num_ode_steps)):
            t_cur = times[ode_step].expand(1)
            dt = times[ode_step + 1] - times[ode_step]
            forward_kwargs = {}
            if use_wan_teacher:
                forward_kwargs = {
                    "condition_latents": condition_latents,
                    "video_xt_human": human_video_latents,
                    "video_xt_object": object_video_latents,
                }
            output = model(
                video_xt=video_latents,
                state_xt=state_latents,
                timesteps=t_cur,
                condition_video=condition_video,
                camera_intrinsics=camera_intrinsics,
                sequence_names=[sequence_name],
                object_categories=[str(sample.get("object_category", "object"))],
                state_to_video_scale=state_to_video_scale,
                video_to_state_scale=video_to_state_scale,
                **forward_kwargs,
            )
            if use_wan_teacher:
                human_velocity = output.human_video_velocity if output.human_video_velocity is not None else output.video_velocity
                object_velocity = (
                    output.object_video_velocity if output.object_video_velocity is not None else output.video_velocity
                )
                human_video_latents = human_video_latents + broadcast_timesteps_like(dt.view(1), human_video_latents) * human_velocity
                object_video_latents = object_video_latents + broadcast_timesteps_like(dt.view(1), object_video_latents) * object_velocity
                video_latents = human_video_latents
            else:
                video_latents = video_latents + broadcast_timesteps_like(dt.view(1), video_latents) * output.video_velocity
            state_latents = state_latents + dt.view(1, 1, 1) * output.state_velocity
        decoded_state = model.decode_state_tokens(state_latents)
    finally:
        if was_training:
            model.train()

    metadata = {
        "sequence_name": sequence_name,
        "step": int(step),
        "num_frames": int(decoded_state.joints_3d.shape[1]),
        "num_ode_steps": int(args.honest_val_num_ode_steps),
    }
    _save_validation_gaussian_tokens(decoded_state.human_gaussians, gs_dir / "G_h.pt", metadata)
    _save_validation_gaussian_tokens(decoded_state.object_gaussians, gs_dir / "G_o.pt", metadata)
    _save_validation_combined_state(
        decoded_state,
        gs_dir,
        sequence_name=sequence_name,
        num_ode_steps=int(args.honest_val_num_ode_steps),
    )

    pred_human = _gaussian_tokens_to_cloud(decoded_state.human_gaussians)
    gt_human = _gaussian_tokens_to_cloud(sample["human_gaussians"])
    pred_object_world = _transform_cloud(
        _gaussian_tokens_to_cloud(decoded_state.object_gaussians),
        decoded_state.object_transforms[0, 0],
    )
    gt_object_world = _transform_cloud(
        _gaussian_tokens_to_cloud(sample["object_gaussians"]),
        sample["object_poses"][0],
    )

    human_outputs = render_module.render_comparison(
        pair_key="human_canonical",
        title="human canonical: pred vs GT",
        pred_cloud=pred_human,
        gt_cloud=gt_human,
        render_dir=render_dir,
        args=render_args,
        device=device,
    )
    object_outputs = render_module.render_comparison(
        pair_key="object_world_frame0000",
        title="object world frame0: pred vs GT",
        pred_cloud=pred_object_world,
        gt_cloud=gt_object_world,
        render_dir=render_dir,
        args=render_args,
        device=device,
    )

    metrics = {
        "honest3d/human_canonical_pred_to_gt_mean_nn": render_module.compute_pair_metrics(pred_human, gt_human)[
            "pred_to_gt_mean_nn"
        ],
        "honest3d/human_canonical_gt_to_pred_mean_nn": render_module.compute_pair_metrics(pred_human, gt_human)[
            "gt_to_pred_mean_nn"
        ],
        "honest3d/object_world_frame0000_pred_to_gt_mean_nn": render_module.compute_pair_metrics(
            pred_object_world,
            gt_object_world,
        )["pred_to_gt_mean_nn"],
        "honest3d/object_world_frame0000_gt_to_pred_mean_nn": render_module.compute_pair_metrics(
            pred_object_world,
            gt_object_world,
        )["gt_to_pred_mean_nn"],
    }

    (render_dir / "honest_render_meta.json").write_text(
        json.dumps(
            {
                "sequence_name": sequence_name,
                "step": int(step),
                "num_ode_steps": int(args.honest_val_num_ode_steps),
                "metrics": metrics,
                "notes": [
                    "human is compared in canonical space",
                    "object is compared in frame-0 world space",
                    "no mixed-space merged visualization is produced",
                ],
            },
            indent=2,
        )
    )

    if wandb_enabled:
        try:
            import wandb

            active_wandb_run = wandb_run if wandb_run is not None else getattr(wandb, "run", None)
            if active_wandb_run is None:
                raise RuntimeError("No active wandb run is available for honest 3D validation upload.")

            active_wandb_run.log(
                {
                    "honest3d/sequence_name": sequence_name,
                    "honest3d/human_canonical_pred_vs_gt_video": wandb.Video(
                        str(human_outputs["video"]),
                        format="mp4",
                    ),
                    "honest3d/human_canonical_pred_vs_gt_preview": wandb.Image(str(human_outputs["preview"])),
                    "honest3d/object_world_frame0000_pred_vs_gt_video": wandb.Video(
                        str(object_outputs["video"]),
                        format="mp4",
                    ),
                    "honest3d/object_world_frame0000_pred_vs_gt_preview": wandb.Image(
                        str(object_outputs["preview"])
                    ),
                    "honest3d/pointcloud/human_canonical_pred": wandb.Object3D(
                        render_module.cloud_to_object3d(pred_human)
                    ),
                    "honest3d/pointcloud/human_canonical_gt": wandb.Object3D(
                        render_module.cloud_to_object3d(gt_human)
                    ),
                    "honest3d/pointcloud/object_world_frame0000_pred": wandb.Object3D(
                        render_module.cloud_to_object3d(pred_object_world)
                    ),
                    "honest3d/pointcloud/object_world_frame0000_gt": wandb.Object3D(
                        render_module.cloud_to_object3d(gt_object_world)
                    ),
                    **metrics,
                },
                step=step,
            )
        except Exception as exc:
            print(
                f"[train_dual_branch_fm] warning: failed to upload honest 3D validation "
                f"| step={step:07d} | error={exc}",
                flush=True,
            )

    print(
        f"[train_dual_branch_fm] honest 3D validation saved "
        f"| step={step:07d} "
        f"| sequence={sequence_name} "
        f"| object_pred_to_gt={metrics['honest3d/object_world_frame0000_pred_to_gt_mean_nn']:.4f} "
        f"| human_pred_to_gt={metrics['honest3d/human_canonical_pred_to_gt_mean_nn']:.4f} "
        f"| render_dir={render_dir}",
        flush=True,
    )
    return metrics


def rotation_geodesic_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError(
            f"`prediction` and `target` must have shape [..., 3, 3], got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    relative = torch.matmul(prediction.transpose(-1, -2), target)
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cosine = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    return torch.acos(cosine).mean() / math.pi


def resolve_curriculum_boundaries(args: argparse.Namespace) -> Tuple[int, int]:
    fusion_start = max(0, int(round(args.max_steps * args.curriculum_fusion_start_ratio)))
    full_start = max(fusion_start + 1, int(round(args.max_steps * args.curriculum_full_start_ratio)))
    full_start = min(full_start, max(args.max_steps, 1))
    return fusion_start, full_start


def resolve_reconstruction_warmup_step(args: argparse.Namespace) -> int:
    ratio = float(getattr(args, "reconstruction_warmup_ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"`reconstruction_warmup_ratio` must be in [0, 1], got {ratio}.")
    return int(round(args.max_steps * ratio))


def resolve_video_backbone_unfreeze_step(args: argparse.Namespace) -> int:
    ratio = args.video_unfreeze_start_ratio
    if ratio < 0.0:
        if args.loss_preset == "reconstruction_first":
            ratio = args.reconstruction_warmup_ratio
        elif args.loss_preset == "geometry_then_video":
            ratio = args.curriculum_fusion_start_ratio
        else:
            ratio = args.curriculum_fusion_start_ratio
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"`video_unfreeze_start_ratio` must be in [0, 1] or <0 to follow fusion stage, got {ratio}.")
    return int(round(args.max_steps * ratio))


def resolve_state_to_video_scale(args: argparse.Namespace, step: int) -> float:
    if model_uses_wan_teacher(args):
        return 0.0
    return 0.0 if bool(getattr(args, "freeze_video_backbone", False)) else 1.0


def resolve_video_to_state_scale(args: argparse.Namespace, step: int) -> float:
    return 1.0


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def set_named_modules_requires_grad(holder: nn.Module, module_names: tuple[str, ...], requires_grad: bool) -> None:
    for name in module_names:
        set_module_requires_grad(getattr(holder, name), requires_grad)


def set_named_modules_train_mode(holder: nn.Module, module_names: tuple[str, ...], training: bool) -> None:
    for name in module_names:
        getattr(holder, name).train(training)


VIDEO_TEACHER_ROOT_MODULE_NAMES = (
    "video_codec",
    "condition_encoder",
    "video_sampler",
    "video_norm",
    "video_velocity_head",
)
VIDEO_TEACHER_BLOCK_MODULE_NAMES = (
    "video_block",
    "video_from_condition",
    "video_from_geometry",
    "video_from_state",
)
VIDEO_STATE_CONSUMER_BLOCK_MODULE_NAMES = (
    "state_from_video",
    "global_gate",
    "dynamic_gate",
)


def set_video_teacher_requires_grad(model: DualBranchCoGenerativeFlowMatching, requires_grad: bool) -> None:
    set_named_modules_requires_grad(model, VIDEO_TEACHER_ROOT_MODULE_NAMES, requires_grad)


def set_video_teacher_train_mode(model: DualBranchCoGenerativeFlowMatching, training: bool) -> None:
    set_named_modules_train_mode(model, VIDEO_TEACHER_ROOT_MODULE_NAMES, training)


def set_fusion_block_video_teacher_requires_grad(block: nn.Module, requires_grad: bool) -> None:
    set_named_modules_requires_grad(block, VIDEO_TEACHER_BLOCK_MODULE_NAMES, requires_grad)


def set_fusion_block_video_teacher_train_mode(block: nn.Module, training: bool) -> None:
    set_named_modules_train_mode(block, VIDEO_TEACHER_BLOCK_MODULE_NAMES, training)


def set_fusion_block_video_state_consumer_requires_grad(block: nn.Module, requires_grad: bool) -> None:
    set_named_modules_requires_grad(block, VIDEO_STATE_CONSUMER_BLOCK_MODULE_NAMES, requires_grad)


def set_fusion_block_video_state_consumer_train_mode(block: nn.Module, training: bool) -> None:
    set_named_modules_train_mode(block, VIDEO_STATE_CONSUMER_BLOCK_MODULE_NAMES, training)


def set_fusion_block_video_requires_grad(block: nn.Module, requires_grad: bool) -> None:
    for name in (
        "video_block",
        "video_from_condition",
        "video_from_geometry",
        "video_from_state",
        "state_from_video",
        "global_gate",
        "dynamic_gate",
    ):
        set_module_requires_grad(getattr(block, name), requires_grad)


def apply_video_backbone_schedule(
    model,
    args: argparse.Namespace,
    step: int,
) -> Dict[str, float]:
    if model_uses_wan_teacher(model) or model_uses_wan_teacher(args):
        return {
            "video_optim_stage": 0.0,
            "video_unfrozen_blocks": 0.0,
            "video_unfreeze_progress": 0.0,
            "video_teacher_frozen": 1.0,
            "state_to_video_scale": 0.0,
            "video_to_state_scale": resolve_video_to_state_scale(args, step),
        }
    fixed_video_teacher = bool(args.freeze_video_backbone)
    total_blocks = len(model.blocks)
    if total_blocks == 0:
        return {
            "video_optim_stage": 1.0,
            "video_unfrozen_blocks": 0.0,
            "video_unfreeze_progress": 1.0,
            "video_teacher_frozen": 1.0 if fixed_video_teacher else 0.0,
            "state_to_video_scale": resolve_state_to_video_scale(args, step),
            "video_to_state_scale": resolve_video_to_state_scale(args, step),
        }

    if fixed_video_teacher:
        num_unfrozen_blocks = 0
        stage = 0.0
        progress = 0.0
    elif not args.freeze_video_backbone:
        num_unfrozen_blocks = total_blocks
        stage = 1.0
        progress = 1.0
    else:
        unfreeze_step = resolve_video_backbone_unfreeze_step(args)
        progress = linear_ramp(step, unfreeze_step, args.max_steps)
        if step < unfreeze_step:
            num_unfrozen_blocks = 0
            stage = 0.0
        else:
            num_unfrozen_blocks = min(max(args.video_stage2_num_top_blocks, 0), total_blocks)
            stage = 1.0
    teacher_trainable = num_unfrozen_blocks > 0
    set_video_teacher_requires_grad(model, teacher_trainable)
    set_video_teacher_train_mode(model, teacher_trainable)

    frozen_prefix = total_blocks - num_unfrozen_blocks
    for block_idx, block in enumerate(model.blocks):
        video_trainable = teacher_trainable and block_idx >= frozen_prefix
        set_fusion_block_video_teacher_requires_grad(block, video_trainable)
        set_fusion_block_video_teacher_train_mode(block, video_trainable)
        set_fusion_block_video_state_consumer_requires_grad(block, True)
        set_fusion_block_video_state_consumer_train_mode(block, True)

    return {
        "video_optim_stage": stage,
        "video_unfrozen_blocks": float(num_unfrozen_blocks),
        "video_unfreeze_progress": progress,
        "video_teacher_frozen": 1.0 if fixed_video_teacher else 0.0,
        "state_to_video_scale": resolve_state_to_video_scale(args, step),
        "video_to_state_scale": resolve_video_to_state_scale(args, step),
    }


def linear_ramp(step: int, start: int, end: int) -> float:
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return float(step - start) / float(max(end - start, 1))


def resolve_active_loss_names(args: argparse.Namespace) -> tuple[str, ...]:
    stage_key = str(getattr(args, "training_stage", "stage1")).strip().lower()
    if stage_key in LOSS_PRESETS:
        return LOSS_PRESETS[stage_key]
    return LOSS_PRESETS["stage1"]


def build_curriculum_loss_weights(args: argparse.Namespace, step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    active_loss_names = set(resolve_active_loss_names(args))
    base_weights = {
        name: float(getattr(args, f"lambda_{name}")) if name in active_loss_names else 0.0
        for name in LOSS_NAMES
    }
    weights = {name: base_weights[name] for name in LOSS_NAMES}
    stage = 1.0 if str(getattr(args, "training_stage", "stage1")).lower() == "stage1" else 2.0
    metrics = {
        "curriculum_stage": stage,
        "curriculum_fusion_progress": 0.0,
        "curriculum_full_progress": 1.0 if stage >= 2.0 else 0.0,
        "loss_active_count": float(sum(weight > 0.0 for weight in weights.values())),
    }
    return weights, metrics


def compute_losses(
    *,
    model,
    output,
    video_xt: Tensor,
    video_velocity_target: Tensor,
    state_xt: Tensor,
    state_velocity_target: Tensor,
    teacher_state: DecodedHOIState,
    human_supervision_target: Optional[Tensor],
    human_supervision_weights: Optional[Tensor],
    object_supervision_target: Optional[Tensor],
    object_supervision_weights: Optional[Tensor],
    masks_human: Tensor,
    masks_object: Tensor,
    keypoint_heatmaps: Tensor,
    depth: Tensor,
    camera_intrinsics_render: Tensor,
    renderer: DiffRasterizationLayer,
    timesteps: Tensor,
    video_target_tokens: Tensor,
    state_target_tokens: Tensor,
    weights: Dict[str, float],
    video_teacher_is_frozen: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    state_t_view = broadcast_timesteps_like(timesteps, state_xt)
    state_xt = state_xt.float()
    state_velocity_target = state_velocity_target.float()
    masks_human = masks_human.float()
    masks_object = masks_object.float()
    keypoint_heatmaps = keypoint_heatmaps.float()
    depth = depth.float()
    camera_intrinsics_render = camera_intrinsics_render.float()

    state_velocity = output.state_velocity.float()
    state_x1_hat = state_xt + (1.0 - state_t_view) * state_velocity
    decoded_state = model.decode_state_tokens(state_x1_hat)
    zero = state_xt.new_zeros(())
    pred_geometry = model.project_geometry(decoded_state, camera_intrinsics_render)
    teacher_geometry = model.project_geometry(teacher_state, camera_intrinsics_render)
    token_hw = pred_geometry["geometry_maps"].shape[-2:]
    target_keypoint_maps = downsample_spatial_map(keypoint_heatmaps, size=token_hw)
    target_depth = downsample_spatial_map(depth, size=token_hw)
    target_object_mask = downsample_spatial_map(masks_object, size=token_hw).clamp(min=0.0, max=1.0)
    target_joint_mask = density_to_occupancy(target_keypoint_maps)
    teacher_object_mask = teacher_geometry["geometry_maps"][:, :, 2:3].clamp(min=0.0, max=1.0)
    pred_joint_mask = density_to_occupancy(pred_geometry["geometry_maps"][:, :, 0:1])

    object_world_points = transform_object_points(decoded_state.object_gaussians, decoded_state.object_transforms)

    loss_fm = F.mse_loss(state_velocity, state_velocity_target)
    loss_video_fm = F.mse_loss(output.video_velocity.float(), video_velocity_target.float())
    loss_geo_joint_heat = F.l1_loss(
        pred_joint_mask,
        target_joint_mask,
    )
    loss_geo_object_silhouette = F.l1_loss(
        pred_geometry["geometry_maps"][:, :, 2:3],
        target_object_mask,
    )
    loss_geo_depth = compute_masked_l1(
        pred_geometry["geometry_maps"][:, :, 3:4],
        teacher_geometry["geometry_maps"][:, :, 3:4],
        teacher_object_mask,
    )
    loss_human_gaussian = compute_point_set_chamfer(
        decoded_state.human_gaussians[..., :3],
        teacher_state.human_gaussians.float()[..., :3],
    ) + 0.1 * compute_gaussian_attr_nn_loss(
        decoded_state.human_gaussians,
        teacher_state.human_gaussians.float(),
    )
    loss_object_gaussian = compute_point_set_chamfer(
        decoded_state.object_gaussians[..., :3],
        teacher_state.object_gaussians.float()[..., :3],
    ) + 0.1 * compute_gaussian_attr_nn_loss(
        decoded_state.object_gaussians,
        teacher_state.object_gaussians.float(),
    )
    loss_joints = F.smooth_l1_loss(decoded_state.joints_3d, teacher_state.joints_3d.float())
    loss_reg_pose = F.mse_loss(decoded_state.human_pose, teacher_state.human_pose.float())
    loss_reg_shape = F.mse_loss(decoded_state.human_shape, teacher_state.human_shape.float())
    loss_reg_translation = F.mse_loss(decoded_state.human_translation, teacher_state.human_translation.float())
    loss_reg_object_pose_translation = F.smooth_l1_loss(
        decoded_state.object_transforms[..., :3, 3],
        teacher_state.object_transforms[..., :3, 3].float(),
    )
    loss_reg_object_pose_rotation = rotation_geodesic_loss(
        decoded_state.object_transforms[..., :3, :3],
        teacher_state.object_transforms[..., :3, :3].float(),
    )
    loss_reg_object_pose = loss_reg_object_pose_translation + loss_reg_object_pose_rotation
    loss_reg_contact = F.smooth_l1_loss(decoded_state.contact_signature, teacher_state.contact_signature.float())
    loss_depth = 0.5 * (
        compute_masked_l1(pred_geometry["geometry_maps"][:, :, 1:2], target_depth, target_joint_mask)
        + compute_masked_l1(pred_geometry["geometry_maps"][:, :, 3:4], target_depth, target_object_mask)
    )
    loss_temp_object = compute_second_order_smoothness(object_world_points)
    loss_temp_contact = compute_contact_relative_velocity_loss(
        decoded_state.joints_3d,
        object_world_points,
        decoded_state.contact_signature,
    )
    loss_phys_contact = compute_contact_distance_loss(
        decoded_state.joints_3d,
        object_world_points,
        teacher_state.contact_signature.float(),
    )
    loss_phys_penetration = compute_penetration_loss(decoded_state.joints_3d, object_world_points)

    losses = {
        "fm": loss_fm,
        "geo_joint_heat": loss_geo_joint_heat,
        "geo_object_silhouette": loss_geo_object_silhouette,
        "geo_depth": loss_geo_depth,
        "human_gaussian": loss_human_gaussian,
        "object_gaussian": loss_object_gaussian,
        "joints": loss_joints,
        "reg_pose": loss_reg_pose,
        "reg_shape": loss_reg_shape,
        "reg_translation": loss_reg_translation,
        "reg_object_pose": loss_reg_object_pose,
        "reg_contact": loss_reg_contact,
        "depth": loss_depth,
        "temp_object": loss_temp_object,
        "temp_contact": loss_temp_contact,
        "phys_contact": loss_phys_contact,
        "phys_penetration": loss_phys_penetration,
    }
    total_loss = zero
    for name, loss_value in losses.items():
        total_loss = total_loss + float(weights.get(name, 0.0)) * loss_value

    metrics = {"loss_total": total_loss.detach()}
    metrics.update({f"loss_{name}": loss_value.detach() for name, loss_value in losses.items()})
    metrics["loss_geo_aux"] = (loss_geo_joint_heat + loss_geo_object_silhouette + loss_geo_depth).detach()
    metrics["loss_reg"] = (
        loss_reg_pose + loss_reg_shape + loss_reg_translation + loss_reg_object_pose + loss_reg_contact
    ).detach()
    metrics["loss_temp"] = (loss_temp_object + loss_temp_contact).detach()
    metrics["loss_phys"] = (loss_phys_contact + loss_phys_penetration).detach()
    metrics["loss_reg_object_pose_translation"] = loss_reg_object_pose_translation.detach()
    metrics["loss_reg_object_pose_rotation"] = loss_reg_object_pose_rotation.detach()
    metrics["loss_video_fm"] = loss_video_fm.detach()
    metrics["loss_state_fm"] = loss_fm.detach()
    metrics["loss_object_render"] = loss_geo_object_silhouette.detach()
    metrics["loss_joint_heat"] = loss_geo_joint_heat.detach()
    metrics["loss_joints_3d"] = loss_joints.detach()
    metrics["loss_joints"] = loss_joints.detach()
    return total_loss, metrics


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{step:07d}.pt"
    raw_scheduler = _unwrap_scheduler(scheduler)
    model_state = filter_video_teacher_state_dict(accelerator.unwrap_model(model).state_dict())
    accelerator.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "resume_meta": {
                "global_step": int(step),
                "scheduler_step": int(getattr(raw_scheduler, "last_epoch", step)),
                "num_processes": int(accelerator.num_processes),
            },
            "args": vars(args),
        },
        str(path),
    )


def resume_if_available(
    *,
    args: argparse.Namespace,
    model,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> int:
    if not args.resume_checkpoint:
        return 0
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    model_state = filter_video_teacher_state_dict(checkpoint["model"])
    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch detected while resuming. "
            f"Missing keys: {incompatible.missing_keys[:10]} "
            f"| Unexpected keys: {incompatible.unexpected_keys[:10]}"
        )
    resume_step = int(checkpoint.get("step", 0))
    if bool(getattr(args, "resume_model_only", False)):
        return resume_step

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    saved_scheduler_state = checkpoint.get("scheduler")
    saved_scheduler_step = None
    if isinstance(saved_scheduler_state, dict):
        try:
            saved_scheduler_step = int(saved_scheduler_state.get("last_epoch", resume_step))
        except (TypeError, ValueError):
            saved_scheduler_step = None
    if saved_scheduler_step is not None and abs(saved_scheduler_step - resume_step) > 1:
        print(
            "[train_dual_branch_fm] warning: checkpoint scheduler progress does not match "
            f"checkpoint step | step={resume_step} | scheduler_step={saved_scheduler_step}. "
            "Re-aligning LR schedule to optimizer-step progress for a stable resume.",
            flush=True,
        )
    _realign_scheduler_to_step(scheduler, optimizer, resume_step)
    return resume_step


def load_init_checkpoint_if_available(
    *,
    args: argparse.Namespace,
    model,
) -> int:
    init_checkpoint = str(getattr(args, "init_checkpoint", "")).strip()
    if not init_checkpoint:
        return 0
    checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    model_state = filter_video_teacher_state_dict(checkpoint["model"])
    incompatible = model.load_state_dict(model_state, strict=bool(getattr(args, "init_checkpoint_strict", False)))
    if bool(getattr(args, "init_checkpoint_strict", False)) and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Strict init checkpoint load failed. "
            f"Missing keys: {incompatible.missing_keys[:10]} "
            f"| Unexpected keys: {incompatible.unexpected_keys[:10]}"
        )
    return int(checkpoint.get("step", 0))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train dual-branch co-generative Flow Matching for 4D HOI.")
    parser.add_argument("--data_root", type=str, default="sample_data/behave_1pct/sequences")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument(
        "--human_gaussian_source",
        type=str,
        default="smpl_mesh",
        choices=("smpl_mesh", "teacher"),
    )
    parser.add_argument("--output_dir", type=str, default="outputs/dual_branch_fm")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--project_name", type=str, default="dual-branch-fm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_with", type=str, default="tensorboard", choices=("tensorboard", "wandb", "none"))
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_model_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint_strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training_stage", type=str, default="stage1", choices=("stage1", "stage2"))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--clip_length", type=int, default=20)
    parser.add_argument("--clip_stride", type=int, default=4)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--dataset_cache_sequences", type=int, default=2)
    parser.add_argument("--cache_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb_cache_max_frames", type=int, default=256)
    parser.add_argument("--index_progress_every", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--batch_sampler_mode",
        type=str,
        default="global_clip_shuffle",
        choices=("global_clip_shuffle", "sequence_grouped"),
    )
    parser.add_argument("--sequence_batch_interleave_window", type=int, default=0)
    parser.add_argument("--sequence_batch_burst", type=int, default=1)
    parser.add_argument("--background_value", type=float, default=1.0)

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--condition_patch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--video_backend", type=str, default="wan_ti2v_5b", choices=("wan_ti2v_5b", "legacy_codec"))
    parser.add_argument("--video_channels", type=int, default=3)
    parser.add_argument("--wan_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--wan_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--wan_hidden_dim", type=int, default=3072)
    parser.add_argument("--wan_prompt_max_sequence_length", type=int, default=512)
    parser.add_argument("--wan_prompt_override", type=str, default="")
    parser.add_argument("--wan_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wan_pad_to_compatible_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_human_gaussians", type=int, default=1024)
    parser.add_argument("--num_object_gaussians", type=int, default=1024)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--human_shape_dim", type=int, default=10)
    parser.add_argument("--human_pose_dim", type=int, default=72)

    parser.add_argument(
        "--loss_preset",
        type=str,
        default="stage1",
        choices=tuple(LOSS_PRESETS.keys()) + ("geometry_then_video", "reconstruction_first", "core"),
        help=(
            "Legacy compatibility flag. Uni-HOI training now follows `--training_stage stage1|stage2`; "
            "`--loss_preset` is ignored unless older tooling still populates it."
        ),
    )
    parser.add_argument("--lambda_fm", type=float, default=1.0)
    parser.add_argument("--lambda_geo_joint_heat", type=float, default=1.0)
    parser.add_argument("--lambda_geo_object_silhouette", type=float, default=0.5)
    parser.add_argument("--lambda_geo_depth", type=float, default=1.0)
    parser.add_argument("--lambda_reg_pose", type=float, default=1.0)
    parser.add_argument("--lambda_reg_shape", type=float, default=0.25)
    parser.add_argument("--lambda_reg_translation", type=float, default=1.0)
    parser.add_argument("--lambda_reg_object_pose", type=float, default=1.0)
    parser.add_argument("--lambda_reg_contact", type=float, default=0.25)
    parser.add_argument("--lambda_depth", type=float, default=1.0)
    parser.add_argument("--lambda_temp_object", type=float, default=0.1)
    parser.add_argument("--lambda_temp_contact", type=float, default=0.1)
    parser.add_argument("--lambda_phys_contact", type=float, default=0.05)
    parser.add_argument("--lambda_phys_penetration", type=float, default=0.05)
    parser.add_argument("--lambda_video_fm", type=float, default=0.0)
    parser.add_argument("--lambda_state_fm", type=float, default=1.0)
    parser.add_argument("--lambda_video_latent", type=float, default=0.0)
    parser.add_argument("--lambda_state_latent", type=float, default=0.0)
    parser.add_argument("--lambda_human_visible", type=float, default=0.0)
    parser.add_argument("--lambda_human_temporal", type=float, default=0.0)
    parser.add_argument("--lambda_object_video", type=float, default=0.0)
    parser.add_argument("--lambda_object_render", type=float, default=1.0)
    parser.add_argument("--lambda_branch_coupling", type=float, default=0.25)
    parser.add_argument("--lambda_human_gaussian", type=float, default=1.0)
    parser.add_argument("--lambda_object_gaussian", type=float, default=1.0)
    parser.add_argument("--lambda_joints", type=float, default=1.0)
    parser.add_argument("--lambda_object_motion", type=float, default=0.5)
    parser.add_argument("--lambda_contact", type=float, default=0.0)
    parser.add_argument("--lambda_joint_heat", type=float, default=0.0)
    parser.add_argument("--lambda_object_silhouette", type=float, default=0.5)
    parser.add_argument("--lambda_object_depth", type=float, default=0.0)
    parser.add_argument("--lambda_geometry_distill", type=float, default=0.0)
    parser.add_argument("--human_visible_region_weight", type=float, default=1.0)
    parser.add_argument("--human_completion_region_weight", type=float, default=1.5)
    parser.add_argument("--num_human_video_points", type=int, default=1024)
    parser.add_argument("--human_proxy_gaussian_scale", type=float, default=0.012)
    parser.add_argument("--object_visible_region_weight", type=float, default=1.0)
    parser.add_argument("--object_primary_region_weight", type=float, default=0.3)
    parser.add_argument("--object_secondary_region_weight", type=float, default=0.05)
    parser.add_argument("--reconstruction_warmup_ratio", type=float, default=0.35)
    parser.add_argument("--curriculum_fusion_start_ratio", type=float, default=0.6)
    parser.add_argument("--curriculum_full_start_ratio", type=float, default=0.8)
    parser.add_argument("--freeze_video_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_unfreeze_start_ratio", type=float, default=-1.0)
    parser.add_argument("--video_stage2_num_top_blocks", type=int, default=2)
    parser.add_argument("--honest_val_every", type=int, default=5000)
    parser.add_argument("--honest_val_num_ode_steps", type=int, default=50)
    parser.add_argument("--honest_val_prior_noise_std", type=float, default=1.0)
    parser.add_argument("--honest_val_sequence", type=str, default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.cpu:
        raise RuntimeError("CPU training is not supported because `DiffRasterizationLayer` requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training, but `torch.cuda.is_available()` is false in this environment.")
    if not 0.0 <= args.curriculum_fusion_start_ratio <= args.curriculum_full_start_ratio <= 1.0:
        raise ValueError(
            "Curriculum ratios must satisfy 0 <= fusion_start <= full_start <= 1. "
            f"Got {args.curriculum_fusion_start_ratio} and {args.curriculum_full_start_ratio}."
        )
    if not 0.0 <= args.reconstruction_warmup_ratio <= 1.0:
        raise ValueError(
            f"`reconstruction_warmup_ratio` must be in [0, 1], got {args.reconstruction_warmup_ratio}."
        )
    if args.reconstruction_warmup_ratio > args.curriculum_full_start_ratio:
        raise ValueError(
            "`reconstruction_warmup_ratio` must be <= `curriculum_full_start_ratio` so the staged "
            "handoff from reconstruction-first training to dual-branch training is well-defined. "
            f"Got {args.reconstruction_warmup_ratio} > {args.curriculum_full_start_ratio}."
        )
    if args.num_human_video_points <= 0:
        raise ValueError(f"`num_human_video_points` must be > 0, got {args.num_human_video_points}.")
    if args.human_proxy_gaussian_scale <= 0.0:
        raise ValueError(
            f"`human_proxy_gaussian_scale` must be > 0, got {args.human_proxy_gaussian_scale}."
        )
    if args.image_height % args.patch_size != 0 or args.image_width % args.patch_size != 0:
        raise ValueError(
            f"Image size {(args.image_height, args.image_width)} must be divisible by patch_size={args.patch_size}."
        )
    if args.image_height % args.condition_patch_size != 0 or args.image_width % args.condition_patch_size != 0:
        raise ValueError(
            "Image size must be divisible by condition_patch_size. "
            f"Got image={(args.image_height, args.image_width)} and condition_patch_size={args.condition_patch_size}."
        )
    if args.video_stage2_num_top_blocks < 0:
        raise ValueError(f"`video_stage2_num_top_blocks` must be >= 0, got {args.video_stage2_num_top_blocks}.")
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}.")
    if args.dataloader_num_workers < 0:
        raise ValueError(
            f"`dataloader_num_workers` must be >= 0, got {args.dataloader_num_workers}."
        )
    if args.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"`dataloader_prefetch_factor` must be > 0, got {args.dataloader_prefetch_factor}."
        )
    args.batch_sampler_mode = normalize_batch_sampler_mode(args.batch_sampler_mode)
    if args.sequence_batch_interleave_window < 0:
        raise ValueError(
            "`sequence_batch_interleave_window` must be >= 0 so 0 can mean auto. "
            f"Got {args.sequence_batch_interleave_window}."
        )
    if args.sequence_batch_burst <= 0:
        raise ValueError(f"`sequence_batch_burst` must be > 0, got {args.sequence_batch_burst}.")
    if args.clip_length <= 0:
        raise ValueError(f"`clip_length` must be > 0, got {args.clip_length}.")
    if args.max_steps <= 0:
        raise ValueError(f"`max_steps` must be > 0, got {args.max_steps}.")
    if args.save_every <= 0 or args.log_every <= 0 or args.print_every <= 0:
        raise ValueError("`save_every`, `log_every`, and `print_every` must all be > 0.")
    if args.honest_val_every < 0:
        raise ValueError(f"`honest_val_every` must be >= 0, got {args.honest_val_every}.")
    if args.honest_val_num_ode_steps <= 0:
        raise ValueError(f"`honest_val_num_ode_steps` must be > 0, got {args.honest_val_num_ode_steps}.")
    if model_uses_wan_teacher(args):
        if not args.freeze_video_backbone:
            raise ValueError("`--video_backend wan_ti2v_5b` requires `--freeze_video_backbone`.")
        resolve_wan_teacher_num_frames(
            clip_length=args.clip_length,
            pad_to_compatible_frames=bool(args.wan_pad_to_compatible_frames),
        )
        useless_teacher_only_losses = (
            "video_fm",
            "video_latent",
            "human_visible",
            "human_temporal",
            "object_video",
        )
        enabled_teacher_only_losses = [
            name for name in useless_teacher_only_losses if float(getattr(args, f"lambda_{name}")) > 0.0
        ]
        if enabled_teacher_only_losses:
            raise ValueError(
                "Frozen Wan teacher mode does not backpropagate through teacher-only video losses. "
                f"Disable these lambdas instead: {', '.join(enabled_teacher_only_losses)}."
            )
    resolve_video_backbone_unfreeze_step(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    log_with = None if args.log_with == "none" else args.log_with
    if log_with == "tensorboard" and importlib.util.find_spec("tensorboard") is None:
        print("[train_dual_branch_fm] tensorboard is not installed; falling back to log_with=none.", flush=True)
        log_with = None
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=args.cpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(static_graph=True)],
    )
    wandb_enabled = log_with == "wandb"
    wandb_run = None
    set_seed(args.seed, device_specific=True)
    report_attention_runtime(args, accelerator)
    if accelerator.is_main_process and log_with is not None:
        init_kwargs = {}
        if wandb_enabled:
            init_kwargs = {"wandb": {"name": Path(args.output_dir).name}}
            wandb_init_timeout = os.getenv("WANDB_INIT_TIMEOUT", "").strip()
            if wandb_init_timeout:
                try:
                    import wandb

                    init_kwargs["wandb"]["settings"] = wandb.Settings(init_timeout=float(wandb_init_timeout))
                except Exception as exc:
                    print(
                        "[train_dual_branch_fm] warning: failed to apply WANDB_INIT_TIMEOUT "
                        f"| value={wandb_init_timeout!r} | error={exc}",
                        flush=True,
                    )
        try:
            accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs=init_kwargs)
        except Exception as exc:
            if wandb_enabled:
                print(
                    "[train_dual_branch_fm] warning: wandb init failed; continuing with log_with=none "
                    f"| error={exc}",
                    flush=True,
                )
                log_with = None
                wandb_enabled = False
            else:
                raise
    if accelerator.is_main_process and wandb_enabled:
        try:
            wandb_run = accelerator.get_tracker("wandb", unwrap=True)
        except Exception as exc:
            print(
                "[train_dual_branch_fm] warning: failed to access active wandb run "
                f"| error={exc}",
                flush=True,
            )
    if accelerator.is_main_process:
        print(
            f"[train_dual_branch_fm] accelerator ready "
            f"| device={accelerator.device} "
            f"| world_size={accelerator.num_processes} "
            f"| log_with={log_with or 'none'} "
            f"| output_dir={args.output_dir}"
            ,
            flush=True,
        )
        print(
            f"[train_dual_branch_fm] building dataset index "
            f"| data_root={args.data_root} "
            f"| split={args.split_file or '<all>'}:{args.split_key} "
            f"| cache_sequences={args.dataset_cache_sequences}"
            ,
            flush=True,
        )
    dataset_index_start_time = time.time()

    def report_dataset_index_progress(sequence_idx: int, total_sequences: int, sequence_name: str, num_frames: int) -> None:
        if not accelerator.is_main_process:
            return
        elapsed = time.time() - dataset_index_start_time
        print(
            f"[train_dual_branch_fm] dataset index progress "
            f"| sequence={sequence_idx}/{total_sequences} "
            f"| name={sequence_name} "
            f"| frames={num_frames} "
            f"| elapsed={elapsed:.1f}s",
            flush=True,
        )

    require_human_vertices = args.lambda_human_visible > 0.0 or args.lambda_human_temporal > 0.0
    dataset = DualBranchHOIDataset(
        data_root=args.data_root,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        processed_subdir=args.processed_subdir,
        gs_subdir=args.gs_subdir,
        human_gaussian_source=args.human_gaussian_source,
        num_human_gaussians=args.num_human_gaussians,
        num_object_gaussians=args.num_object_gaussians,
        num_joints=args.num_joints,
        contact_dim=args.contact_dim,
        background_value=args.background_value,
        max_sequences=args.max_sequences,
        cache_sequences=args.dataset_cache_sequences,
        cache_rgb=args.cache_rgb,
        rgb_cache_max_frames=args.rgb_cache_max_frames,
        index_progress_every=args.index_progress_every,
        index_progress_callback=report_dataset_index_progress if accelerator.is_main_process else None,
        split_file=args.split_file,
        split_key=args.split_key,
        prefer_h5_cache=True,
        include_human_vertices=require_human_vertices,
        include_keypoint_heatmaps=False,
    )
    if accelerator.is_main_process:
        print(
            f"[train_dual_branch_fm] dataset index ready "
            f"| sequences={len(dataset.sequence_dirs)} "
            f"| clips={len(dataset)} "
            f"| clip_length={args.clip_length} "
            f"| clip_stride={args.clip_stride} "
            f"| cache_hit={int(dataset.loaded_from_disk_cache)} "
            f"| elapsed={time.time() - dataset_index_start_time:.1f}s"
            ,
            flush=True,
        )
        if dataset.loaded_from_disk_cache:
            print(
                f"[train_dual_branch_fm] dataset index cache hit "
                f"| path={dataset.index_cache_path}",
                flush=True,
            )
        print("[train_dual_branch_fm] inferring condition channels from first sample...", flush=True)
    condition_channels = infer_condition_channels(dataset)
    honest_val_sequence_dir = None
    if args.honest_val_every > 0:
        honest_val_sequence_dir = _resolve_honest_validation_sequence_dir(
            args.honest_val_sequence,
            list(dataset.sequence_dirs),
        )
    if accelerator.is_main_process:
        print(f"[train_dual_branch_fm] condition channels ready | channels={condition_channels}", flush=True)
        print(
            "[train_dual_branch_fm] Uni-HOI stage ready "
            f"| training_stage={args.training_stage} "
            f"| active={','.join(resolve_active_loss_names(args))}",
            flush=True,
        )
        if honest_val_sequence_dir is not None:
            print(
                "[train_dual_branch_fm] honest 3D validation ready "
                f"| every={args.honest_val_every} "
                f"| sequence={Path(honest_val_sequence_dir).name} "
                f"| ode_steps={args.honest_val_num_ode_steps}",
                flush=True,
            )
    model = build_model_from_args(args, condition_channels=condition_channels)
    init_checkpoint_step = load_init_checkpoint_if_available(args=args, model=model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    global_step = resume_if_available(args=args, model=model, optimizer=optimizer, scheduler=scheduler)
    video_schedule_metrics = apply_video_backbone_schedule(model, args, global_step)
    trainable_parameters = collect_trainable_parameters(model)
    resolved_sequence_batch_interleave_window = 1
    train_batch_sampler = None
    global_num_batches = len(dataset) // args.batch_size
    if global_num_batches == 0:
        raise ValueError(
            "Training dataloader would produce zero global batches. "
            f"clips={len(dataset)}, batch_size={args.batch_size}, drop_last=True. "
            "Lower the batch size or provide more training clips."
        )
    dataloader_kwargs = {
        "num_workers": int(args.dataloader_num_workers),
        "pin_memory": bool(args.dataloader_pin_memory and not args.cpu),
    }
    if args.dataloader_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
        dataloader_kwargs["prefetch_factor"] = int(args.dataloader_prefetch_factor)
    if args.batch_sampler_mode == "sequence_grouped":
        resolved_sequence_batch_interleave_window = resolve_sequence_batch_interleave_window(
            args.sequence_batch_interleave_window,
            world_size=accelerator.num_processes,
            cache_sequences=args.dataset_cache_sequences,
        )
        train_batch_sampler = SequenceBatchSampler(
            dataset,
            batch_size=args.batch_size,
            drop_last=True,
            seed=args.seed,
            shuffle=True,
            interleave_window=resolved_sequence_batch_interleave_window,
            max_batches_per_sequence=args.sequence_batch_burst,
        )
        global_num_batches = len(train_batch_sampler)
        dataloader_kwargs["batch_sampler"] = train_batch_sampler
    else:
        dataloader_kwargs["batch_size"] = int(args.batch_size)
        dataloader_kwargs["shuffle"] = True
        dataloader_kwargs["drop_last"] = True
    train_dataloader = DataLoader(dataset, **dataloader_kwargs)
    if accelerator.is_main_process:
        if init_checkpoint_step > 0:
            print(
                "[train_dual_branch_fm] initialized model weights "
                f"| init_checkpoint={args.init_checkpoint} "
                f"| checkpoint_step={init_checkpoint_step:07d} "
                f"| strict={int(bool(args.init_checkpoint_strict))}",
                flush=True,
            )
        if model_uses_wan_teacher(args):
            teacher_num_frames = resolve_wan_teacher_num_frames(
                clip_length=args.clip_length,
                pad_to_compatible_frames=bool(args.wan_pad_to_compatible_frames),
            )
            print(
                "[train_dual_branch_fm] Wan teacher frame setup "
                f"| clip_length={args.clip_length} "
                f"| teacher_frames={teacher_num_frames} "
                f"| pad_enabled={int(bool(args.wan_pad_to_compatible_frames))}",
                flush=True,
            )
        if args.batch_sampler_mode == "sequence_grouped":
            print(
                "[train_dual_branch_fm] sequence batch sampler ready "
                f"| global_batches={global_num_batches} "
                f"| batch_size_per_device={args.batch_size} "
                f"| cache_sequences={args.dataset_cache_sequences} "
                f"| rgb_cache_max_frames={args.rgb_cache_max_frames} "
                f"| interleave_window={resolved_sequence_batch_interleave_window} "
                f"| sequence_batch_burst={args.sequence_batch_burst} "
                f"| num_workers={args.dataloader_num_workers} "
                f"| pin_memory={int(bool(args.dataloader_pin_memory and not args.cpu))} "
                f"| world_size={accelerator.num_processes}",
                flush=True,
            )
        else:
            print(
                "[train_dual_branch_fm] global clip shuffle dataloader ready "
                f"| global_batches={global_num_batches} "
                f"| batch_size_per_device={args.batch_size} "
                f"| cache_sequences={args.dataset_cache_sequences} "
                f"| rgb_cache_max_frames={args.rgb_cache_max_frames} "
                f"| num_workers={args.dataloader_num_workers} "
                f"| pin_memory={int(bool(args.dataloader_pin_memory and not args.cpu))} "
                f"| world_size={accelerator.num_processes}",
                flush=True,
            )

    renderer = DiffRasterizationLayer(
        image_height=args.image_height,
        image_width=args.image_width,
    )
    renderer = renderer.to(accelerator.device)
    if accelerator.is_main_process:
        print("[train_dual_branch_fm] renderer ready, preparing accelerator-wrapped modules...", flush=True)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        scheduler,
    )
    num_batches = len(train_dataloader)
    if num_batches == 0:
        raise ValueError(
            "Prepared train dataloader has zero local batches. "
            f"global_batches={global_num_batches}, batch_size={args.batch_size}, "
            f"process_index={accelerator.process_index}, world_size={accelerator.num_processes}. "
            "Lower the batch size or provide more training clips."
        )
    raw_model = accelerator.unwrap_model(model)
    ensure_video_teacher_ready(raw_model, accelerator.device)
    if accelerator.is_main_process:
        print("[train_dual_branch_fm] accelerator.prepare complete, entering training setup...", flush=True)
        print(
            f"[train_dual_branch_fm] prepared dataloader ready "
            f"| local_batches_per_rank={num_batches} "
            f"| global_batches={global_num_batches}",
            flush=True,
        )
        if model_uses_wan_teacher(raw_model):
            print(
                f"[train_dual_branch_fm] frozen Wan teacher ready "
                f"| requested={args.wan_model_id} "
                f"| resolved={raw_model.resolved_wan_model_id}",
                flush=True,
            )
    all_parameters = tuple(model.parameters())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    slow_step_threshold = 0.0
    slow_step_threshold_raw = os.getenv("HDM_SLOW_STEP_LOG_THRESHOLD", "").strip()
    if slow_step_threshold_raw:
        try:
            slow_step_threshold = max(float(slow_step_threshold_raw), 0.0)
        except ValueError:
            if accelerator.is_main_process:
                print(
                    "[train_dual_branch_fm] warning: invalid HDM_SLOW_STEP_LOG_THRESHOLD "
                    f"| value={slow_step_threshold_raw!r}",
                    flush=True,
                )
    rank_status_enabled = os.getenv("HDM_WRITE_RANK_STATUS", "1").strip().lower() not in {"0", "false", "no"}
    rank_status_path = (
        Path(args.output_dir) / "debug_status" / f"rank_{accelerator.process_index:02d}.json"
        if rank_status_enabled
        else None
    )
    last_video_stage = None
    last_unfrozen_video_blocks = None
    sampler_label = "sequence-batch"
    if resolved_sequence_batch_interleave_window > 1:
        sampler_label = "sequence-batch-interleaved"
    if accelerator.num_processes == 1:
        loader_label = sampler_label
    else:
        loader_label = f"{sampler_label}-shard:{accelerator.process_index}/{accelerator.num_processes}"
    write_rank_debug_status(
        rank_status_path,
        phase="training_ready",
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        step=global_step,
        extra={
            "output_dir": args.output_dir,
            "device": str(accelerator.device),
            "loader": loader_label,
        },
    )
    checkpoint_progress_bar = None
    checkpoint_segment_end = None

    def create_checkpoint_progress_bar(step: int):
        if not accelerator.is_main_process:
            return None, None
        segment_size = int(args.honest_val_every) if int(args.honest_val_every) > 0 else 5000
        segment_start = (step // segment_size) * segment_size
        segment_end = min(segment_start + segment_size, args.max_steps)
        if segment_end <= step and step < args.max_steps:
            segment_start = step
            segment_end = min(step + segment_size, args.max_steps)
        segment_total = max(segment_end - segment_start, 1)
        segment_initial = min(max(step - segment_start, 0), segment_total)
        bar = tqdm(
            total=segment_total,
            initial=segment_initial,
            desc=f"progress {segment_start:07d}->{segment_end:07d}",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
            leave=True,
        )
        return bar, segment_end

    if accelerator.is_main_process:
        param_count = count_parameters(trainable_parameters)
        print(
            f"[train_dual_branch_fm] clips={len(dataset)} "
            f"| trainable={param_count / 1e6:.2f}M "
            f"| cond_channels={condition_channels} "
            f"| video_stage={int(video_schedule_metrics['video_optim_stage'])} "
            f"| unfrozen_video_blocks={int(video_schedule_metrics['video_unfrozen_blocks'])}"
            ,
            flush=True,
        )
        print(
            f"[train_dual_branch_fm] entering training loop "
            f"| start_step={global_step:07d} "
            f"| target_step={args.max_steps:07d} "
            f"| steps_remaining={max(args.max_steps - global_step, 0):07d} "
            f"| batch_size_per_device={args.batch_size} "
            f"| grad_accum={args.gradient_accumulation_steps} "
            f"| loader={loader_label} "
            f"| save_every={args.save_every} "
            f"| log_every={args.log_every} "
            f"| progress_window={int(args.honest_val_every) if int(args.honest_val_every) > 0 else 5000}"
            ,
            flush=True,
        )
        if rank_status_path is not None:
            print(
                f"[train_dual_branch_fm] per-rank debug status "
                f"| dir={rank_status_path.parent}",
                flush=True,
            )
        checkpoint_progress_bar, checkpoint_segment_end = create_checkpoint_progress_bar(global_step)
    last_video_stage = int(video_schedule_metrics["video_optim_stage"])
    last_unfrozen_video_blocks = int(video_schedule_metrics["video_unfrozen_blocks"])
    batch_wait_start = time.time()

    try:
        while global_step < args.max_steps:
            for batch in train_dataloader:
                batch_wait_sec = time.time() - batch_wait_start
                step_compute_start = time.time()
                raw_model = accelerator.unwrap_model(model)
                video_schedule_metrics = apply_video_backbone_schedule(raw_model, args, global_step)
                current_video_stage = int(video_schedule_metrics["video_optim_stage"])
                current_unfrozen_video_blocks = int(video_schedule_metrics["video_unfrozen_blocks"])
                if (
                    accelerator.is_main_process
                    and (
                        current_video_stage != last_video_stage
                        or current_unfrozen_video_blocks != last_unfrozen_video_blocks
                    )
                ):
                    current_trainable = count_parameters(collect_trainable_parameters(raw_model))
                    print(
                        f"[train_dual_branch_fm] video schedule -> stage={current_video_stage} "
                        f"| unfrozen_video_blocks={current_unfrozen_video_blocks} "
                        f"| trainable={current_trainable / 1e6:.2f}M "
                        f"| step={global_step:07d}"
                        ,
                        flush=True,
                    )
                last_video_stage = current_video_stage
                last_unfrozen_video_blocks = current_unfrozen_video_blocks

                with accelerator.accumulate(model):
                    loss_weights, curriculum_metrics = build_curriculum_loss_weights(args, global_step)
                    state_to_video_scale = resolve_state_to_video_scale(args, global_step)
                    video_to_state_scale = resolve_video_to_state_scale(args, global_step)
                    sequence_names = [str(name) for name in batch["sequence_name"]]
                    write_rank_debug_status(
                        rank_status_path,
                        phase="batch_ready",
                        rank=accelerator.process_index,
                        world_size=accelerator.num_processes,
                        step=global_step,
                        sequence_names=sequence_names,
                        extra={
                            "target_step": global_step + 1,
                            "video_stage": current_video_stage,
                            "unfrozen_video_blocks": current_unfrozen_video_blocks,
                        },
                    )
                    rgb = batch["rgb"].to(accelerator.device, non_blocking=True)
                    human_visible = batch["human_visible"].to(accelerator.device, non_blocking=True)
                    masks_human = batch["masks_human"].to(accelerator.device, non_blocking=True)
                    masks_object = batch["masks_object"].to(accelerator.device, non_blocking=True)
                    m_primary = batch["m_primary"].to(accelerator.device, non_blocking=True)
                    m_secondary = batch["m_secondary"].to(accelerator.device, non_blocking=True)
                    m_object_region = batch["m_object_region"].to(accelerator.device, non_blocking=True)
                    keypoint_heatmaps = batch["keypoint_heatmaps"].to(accelerator.device, non_blocking=True)
                    depth = batch["depth"].to(accelerator.device, non_blocking=True)
                    camera_intrinsics = batch["camera_intrinsics"].to(accelerator.device, non_blocking=True)
                    object_poses = batch["object_poses"].to(accelerator.device, non_blocking=True)
                    human_shape = batch["human_shape"].to(accelerator.device, non_blocking=True)
                    body_pose = batch["body_pose"].to(accelerator.device, non_blocking=True)
                    cam_t = batch["cam_t"].to(accelerator.device, non_blocking=True)
                    object_categories = [str(name) for name in batch["object_category"]]
                    human_gaussians = batch["human_gaussians"].to(accelerator.device, non_blocking=True)
                    object_gaussians = batch["object_gaussians"].to(accelerator.device, non_blocking=True)
                    joints_3d = batch["joints_3d"].to(accelerator.device, non_blocking=True)
                    contact_signature = batch["contact_signature"].to(accelerator.device, non_blocking=True)

                    if rgb.shape[-2:] != (args.image_height, args.image_width):
                        rgb = resize_video_batch(rgb, size=(args.image_height, args.image_width), mode="bilinear")
                        human_visible = resize_video_batch(
                            human_visible,
                            size=(args.image_height, args.image_width),
                            mode="bilinear",
                        )
                        masks_human = resize_video_batch(
                            masks_human,
                            size=(args.image_height, args.image_width),
                            mode="nearest",
                        )
                        masks_object = resize_video_batch(
                            masks_object,
                            size=(args.image_height, args.image_width),
                            mode="nearest",
                        )
                        m_primary = resize_video_batch(
                            m_primary,
                            size=(args.image_height, args.image_width),
                            mode="nearest",
                        )
                        m_secondary = resize_video_batch(
                            m_secondary,
                            size=(args.image_height, args.image_width),
                            mode="nearest",
                        )
                        m_object_region = resize_video_batch(
                            m_object_region,
                            size=(args.image_height, args.image_width),
                            mode="nearest",
                        )
                        keypoint_heatmaps = resize_video_batch(
                            keypoint_heatmaps,
                            size=(args.image_height, args.image_width),
                            mode="bilinear",
                        )
                        depth = resize_video_batch(depth, size=(args.image_height, args.image_width), mode="bilinear")
                        source_hw = batch["rgb"].shape[-2:]
                        camera_intrinsics_render = scale_camera_intrinsics(
                            camera_intrinsics,
                            source_size=(int(source_hw[0]), int(source_hw[1])),
                            target_size=(args.image_height, args.image_width),
                        )
                    else:
                        camera_intrinsics_render = camera_intrinsics
                    condition_video = torch.cat(
                        [
                            rgb,
                            masks_human,
                            masks_object,
                            depth,
                            m_primary,
                            m_secondary,
                            m_object_region,
                            keypoint_heatmaps,
                        ],
                        dim=2,
                    )
                    human_supervision_target = None
                    human_supervision_weights = None
                    object_supervision_target = None
                    object_supervision_weights = None
                    video_target = rgb
                    teacher_state = build_teacher_state(
                        {
                            "human_shape": human_shape,
                            "body_pose": body_pose,
                            "cam_t": cam_t,
                            "human_gaussians": human_gaussians,
                            "object_gaussians": object_gaussians,
                            "joints_3d": joints_3d,
                            "object_poses": object_poses,
                            "contact_signature": contact_signature,
                        }
                    )

                    with torch.no_grad():
                        video_target_tokens = raw_model.encode_video_target(video_target)
                        state_target_tokens = raw_model.encode_state_target(
                            human_shape=human_shape,
                            human_pose=body_pose,
                            human_translation=cam_t,
                            human_gaussians=human_gaussians,
                            object_gaussians=object_gaussians,
                            joints_3d=joints_3d,
                            object_transforms=object_poses,
                            contact_signature=contact_signature,
                        )

                    batch_size = video_target_tokens.shape[0]
                    timesteps = torch.rand(batch_size, device=accelerator.device, dtype=torch.float32)

                    video_noise = torch.randn_like(video_target_tokens)
                    if model_uses_wan_teacher(raw_model):
                        video_noise = raw_model.apply_video_valid_mask(video_noise)
                    state_noise = torch.randn_like(state_target_tokens)
                    video_xt, video_velocity_target = flow_match_sample(video_target_tokens, video_noise, timesteps)
                    if model_uses_wan_teacher(raw_model):
                        video_xt = raw_model.apply_video_valid_mask(video_xt)
                        video_velocity_target = raw_model.apply_video_valid_mask(video_velocity_target)
                    state_xt, state_velocity_target = flow_match_sample(state_target_tokens, state_noise, timesteps)

                    write_rank_debug_status(
                        rank_status_path,
                        phase="forward",
                        rank=accelerator.process_index,
                        world_size=accelerator.num_processes,
                        step=global_step,
                        sequence_names=sequence_names,
                        extra={"target_step": global_step + 1},
                    )
                    forward_kwargs = {}
                    if model_uses_wan_teacher(raw_model):
                        forward_kwargs["condition_latents"] = video_target_tokens
                    output = model(
                        video_xt=video_xt,
                        state_xt=state_xt,
                        timesteps=timesteps,
                        condition_video=condition_video,
                        camera_intrinsics=camera_intrinsics_render,
                        sequence_names=sequence_names,
                        object_categories=object_categories,
                        state_to_video_scale=state_to_video_scale,
                        video_to_state_scale=video_to_state_scale,
                        **forward_kwargs,
                    )

                    loss, metrics = compute_losses(
                        model=raw_model,
                        output=output,
                        video_xt=video_xt,
                        video_velocity_target=video_velocity_target,
                        state_xt=state_xt,
                        state_velocity_target=state_velocity_target,
                        teacher_state=teacher_state,
                        human_supervision_target=human_supervision_target,
                        human_supervision_weights=human_supervision_weights,
                        object_supervision_target=object_supervision_target,
                        object_supervision_weights=object_supervision_weights,
                        masks_human=masks_human,
                        masks_object=masks_object,
                        keypoint_heatmaps=keypoint_heatmaps,
                        depth=depth,
                        camera_intrinsics_render=camera_intrinsics_render,
                        renderer=renderer,
                        timesteps=timesteps,
                        video_target_tokens=video_target_tokens,
                        state_target_tokens=state_target_tokens,
                        weights=loss_weights,
                        video_teacher_is_frozen=bool(args.freeze_video_backbone),
                    )
                    metrics["state_to_video_scale"] = loss.new_tensor(state_to_video_scale)
                    metrics["video_to_state_scale"] = loss.new_tensor(video_to_state_scale)
                    effective_cross_scale = video_to_state_scale if args.freeze_video_backbone else state_to_video_scale
                    metrics["cross_branch_scale"] = loss.new_tensor(effective_cross_scale)

                    write_rank_debug_status(
                        rank_status_path,
                        phase="backward",
                        rank=accelerator.process_index,
                        world_size=accelerator.num_processes,
                        step=global_step,
                        sequence_names=sequence_names,
                        extra={
                            "target_step": global_step + 1,
                            "loss_total": metrics["loss_total"],
                        },
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(all_parameters, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                step_compute_sec = time.time() - step_compute_start
                step_wall_sec = batch_wait_sec + step_compute_sec
                batch_wait_start = time.time()
                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                write_rank_debug_status(
                    rank_status_path,
                    phase="metric_reduce",
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    step=global_step,
                    sequence_names=sequence_names,
                )
                reduced_metrics = reduce_scalar_metrics(accelerator, metrics)
                reduced_metrics.update(curriculum_metrics)
                reduced_metrics.update(video_schedule_metrics)
                reduced_metrics["lr"] = scheduler.get_last_lr()[0]
                reduced_metrics["batch_wait_sec"] = batch_wait_sec
                reduced_metrics["step_compute_sec"] = step_compute_sec
                reduced_metrics["step_wall_sec"] = step_wall_sec
                reduced_metrics["trainable_params_m"] = (
                    count_parameters(collect_trainable_parameters(accelerator.unwrap_model(model))) / 1e6
                )
                write_rank_debug_status(
                    rank_status_path,
                    phase="step_complete",
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    step=global_step,
                    sequence_names=sequence_names,
                    extra={
                        "loss_total": reduced_metrics["loss_total"],
                        "batch_wait_sec": batch_wait_sec,
                        "step_compute_sec": step_compute_sec,
                        "step_wall_sec": step_wall_sec,
                    },
                )
                if slow_step_threshold > 0.0 and step_wall_sec >= slow_step_threshold:
                    joined_sequences = ",".join(sequence_names[:4])
                    if len(sequence_names) > 4:
                        joined_sequences = f"{joined_sequences},..."
                    print(
                        f"[train_dual_branch_fm] slow step "
                        f"| rank={accelerator.process_index} "
                        f"| step={global_step:07d} "
                        f"| duration={step_wall_sec:.2f}s "
                        f"| wait={batch_wait_sec:.2f}s "
                        f"| compute={step_compute_sec:.2f}s "
                        f"| sequences={joined_sequences}",
                        flush=True,
                    )

                if accelerator.is_main_process and checkpoint_progress_bar is not None:
                    checkpoint_progress_bar.update(1)
                    checkpoint_progress_bar.set_postfix(
                        loss=f"{reduced_metrics['loss_total']:.4f}",
                        vfm=f"{reduced_metrics['loss_video_fm']:.4f}",
                        sfm=f"{reduced_metrics['loss_state_fm']:.4f}",
                        lr=f"{reduced_metrics['lr']:.2e}",
                        refresh=False,
                    )

                if global_step % args.log_every == 0 and accelerator.is_main_process and log_with is not None:
                    log_metrics = reduced_metrics
                    if wandb_enabled:
                        log_metrics = build_wandb_loss_metrics(reduced_metrics, loss_weights)
                    accelerator.log(log_metrics, step=global_step)

                if global_step % args.print_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(global_step, 1)) * max(args.max_steps - global_step, 0)
                    next_save = min(
                        ((global_step // max(int(args.save_every), 1)) + 1) * max(int(args.save_every), 1),
                        args.max_steps,
                    )
                    next_honest = checkpoint_segment_end if checkpoint_segment_end is not None else args.max_steps
                    print(
                        f"[train_dual_branch_fm] step={global_step:07d} "
                        f"loss={reduced_metrics['loss_total']:.4f} "
                        f"vfm={reduced_metrics['loss_video_fm']:.4f} "
                        f"sfm={reduced_metrics['loss_state_fm']:.4f} "
                        f"hg={reduced_metrics['loss_human_gaussian']:.4f} "
                        f"og={reduced_metrics['loss_object_gaussian']:.4f} "
                        f"j3d={reduced_metrics['loss_joints_3d']:.4f} "
                        f"jheat={reduced_metrics['loss_joint_heat']:.4f} "
                        f"objSil={reduced_metrics['loss_geo_object_silhouette']:.4f} "
                        f"stage={int(reduced_metrics['curriculum_stage'])} "
                        f"video_stage={int(reduced_metrics['video_optim_stage'])} "
                        f"stv={reduced_metrics['state_to_video_scale']:.2f} "
                        f"vts={reduced_metrics['video_to_state_scale']:.2f} "
                        f"unfrozen={int(reduced_metrics['video_unfrozen_blocks'])} "
                        f"wait={reduced_metrics['batch_wait_sec']:.2f}s "
                        f"compute={reduced_metrics['step_compute_sec']:.2f}s "
                        f"next_save={next_save:07d} "
                        f"next_honest={next_honest:07d} "
                        f"eta={eta / 3600.0:.2f}h"
                        ,
                        flush=True,
                    )

                if global_step % args.save_every == 0:
                    write_rank_debug_status(
                        rank_status_path,
                        phase="checkpoint_barrier",
                        rank=accelerator.process_index,
                        world_size=accelerator.num_processes,
                        step=global_step,
                    )
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=global_step,
                            args=args,
                        )
                    accelerator.wait_for_everyone()
                if args.honest_val_every > 0 and global_step % args.honest_val_every == 0:
                    write_rank_debug_status(
                        rank_status_path,
                        phase="honest_validation_barrier",
                        rank=accelerator.process_index,
                        world_size=accelerator.num_processes,
                        step=global_step,
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and honest_val_sequence_dir is not None:
                        try:
                            honest_metrics = run_honest_3d_validation(
                                model=accelerator.unwrap_model(model),
                                dataset=dataset,
                                sequence_dir=honest_val_sequence_dir,
                                device=accelerator.device,
                                output_dir=args.output_dir,
                                step=global_step,
                                args=args,
                                wandb_enabled=wandb_enabled,
                                wandb_run=wandb_run,
                            )
                            if log_with is not None and not wandb_enabled:
                                accelerator.log(honest_metrics, step=global_step)
                        except Exception as exc:
                            print(
                                f"[train_dual_branch_fm] warning: honest 3D validation failed "
                                f"| step={global_step:07d} "
                                f"| error={exc}",
                                flush=True,
                            )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
                if (
                    accelerator.is_main_process
                    and checkpoint_progress_bar is not None
                    and checkpoint_segment_end is not None
                    and global_step >= checkpoint_segment_end
                ):
                    checkpoint_progress_bar.close()
                    checkpoint_progress_bar = None
                    checkpoint_segment_end = None
                    if global_step < args.max_steps:
                        checkpoint_progress_bar, checkpoint_segment_end = create_checkpoint_progress_bar(global_step)

                if global_step >= args.max_steps:
                    break
    finally:
        write_rank_debug_status(
            rank_status_path,
            phase="training_exit",
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            step=global_step,
        )
        if checkpoint_progress_bar is not None:
            checkpoint_progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            args=args,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
