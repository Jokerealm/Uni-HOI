#!/usr/bin/env python3
"""
Train the dual-branch co-generative Flow Matching model for 4D HOI reconstruction.

The training graph is unified:

- Video branch:
  predicts latent velocities for human/object amodal video tokens.
- State branch:
  predicts latent velocities for HOI state tokens
  (human/object Gaussians, joints, object motion, contact).
- Cross-branch supervision:
  object render consistency, 3D->2D geometry consistency, contact consistency.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model.dual_branch_cogenerative_fm import DecodedHOIState, DualBranchCoGenerativeFlowMatching
from model.joint_renderer_loss import DiffRasterizationLayer

LOSS_NAMES = (
    "video_fm",
    "state_fm",
    "video_latent",
    "state_latent",
    "human_visible",
    "object_video",
    "object_render",
    "branch_coupling",
    "human_gaussian",
    "object_gaussian",
    "joints",
    "object_motion",
    "contact",
    "joint_heat",
    "object_silhouette",
    "object_depth",
    "geometry_distill",
)


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


def collect_trainable_parameters(*modules: nn.Module) -> Tuple[nn.Parameter, ...]:
    params = []
    for module in modules:
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters were found.")
    return tuple(params)


def count_parameters(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def infer_condition_channels(dataset: DualBranchHOIDataset) -> int:
    return int(dataset.condition_channels)


def configure_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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


def build_teacher_state(batch: Dict[str, Tensor]) -> DecodedHOIState:
    return DecodedHOIState(
        human_gaussians=batch["human_gaussians"],
        object_gaussians=batch["object_gaussians"],
        joints_3d=batch["joints_3d"],
        object_transforms=batch["object_poses"],
        contact_signature=batch["contact_signature"],
    )


def compute_masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.expand_as(target)
    denom = mask.sum().clamp(min=1.0)
    return ((prediction - target).abs() * mask).sum() / denom


def resolve_curriculum_boundaries(args: argparse.Namespace) -> Tuple[int, int]:
    fusion_start = max(0, int(round(args.max_steps * args.curriculum_fusion_start_ratio)))
    full_start = max(fusion_start + 1, int(round(args.max_steps * args.curriculum_full_start_ratio)))
    full_start = min(full_start, max(args.max_steps, 1))
    return fusion_start, full_start


def resolve_video_backbone_unfreeze_step(args: argparse.Namespace) -> int:
    ratio = args.video_unfreeze_start_ratio
    if ratio < 0.0:
        ratio = args.curriculum_fusion_start_ratio
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"`video_unfreeze_start_ratio` must be in [0, 1] or <0 to follow fusion stage, got {ratio}.")
    return int(round(args.max_steps * ratio))


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def apply_video_backbone_schedule(
    model: DualBranchCoGenerativeFlowMatching,
    args: argparse.Namespace,
    step: int,
) -> Dict[str, float]:
    total_blocks = len(model.blocks)
    if total_blocks == 0:
        return {
            "video_optim_stage": 1.0,
            "video_unfrozen_blocks": 0.0,
            "video_unfreeze_progress": 1.0,
        }

    if not args.freeze_video_backbone:
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

    frozen_prefix = total_blocks - num_unfrozen_blocks
    for block_idx, block in enumerate(model.blocks):
        set_module_requires_grad(block.video_block, block_idx >= frozen_prefix)

    return {
        "video_optim_stage": stage,
        "video_unfrozen_blocks": float(num_unfrozen_blocks),
        "video_unfreeze_progress": progress,
    }


def linear_ramp(step: int, start: int, end: int) -> float:
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return float(step - start) / float(max(end - start, 1))


def build_curriculum_loss_weights(args: argparse.Namespace, step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    base_weights = {name: float(getattr(args, f"lambda_{name}")) for name in LOSS_NAMES}
    fusion_start, full_start = resolve_curriculum_boundaries(args)
    fusion_progress = linear_ramp(step, fusion_start, full_start)
    full_progress = linear_ramp(step, full_start, args.max_steps)

    multipliers = {
        "video_fm": 1.0,
        "state_fm": 1.0,
        "video_latent": 1.0,
        "state_latent": 1.0,
        "human_visible": 1.0,
        "object_video": 1.0,
        "joints": 1.0,
        "object_motion": 1.0,
        "object_render": fusion_progress,
        "branch_coupling": fusion_progress,
        "human_gaussian": fusion_progress,
        "object_gaussian": fusion_progress,
        "joint_heat": fusion_progress,
        "object_silhouette": fusion_progress,
        "geometry_distill": fusion_progress,
        "contact": full_progress,
        "object_depth": full_progress,
    }
    weights = {name: base_weights[name] * multipliers.get(name, 1.0) for name in LOSS_NAMES}
    stage = 0.0
    if fusion_progress > 0.0:
        stage = 1.0
    if full_progress > 0.0:
        stage = 2.0
    metrics = {
        "curriculum_stage": stage,
        "curriculum_fusion_progress": fusion_progress,
        "curriculum_full_progress": full_progress,
    }
    return weights, metrics


def compute_losses(
    *,
    model: DualBranchCoGenerativeFlowMatching,
    output,
    video_xt: Tensor,
    video_velocity_target: Tensor,
    state_xt: Tensor,
    state_velocity_target: Tensor,
    teacher_state: DecodedHOIState,
    teacher_object_video: Tensor,
    human_visible_target: Tensor,
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
) -> Tuple[Tensor, Dict[str, Tensor]]:
    t_view = timesteps.view(timesteps.shape[0], 1, 1)
    video_x1_hat = video_xt + (1.0 - t_view) * output.video_velocity
    state_x1_hat = state_xt + (1.0 - t_view) * output.state_velocity
    decoded_video = model.decode_video_tokens(video_x1_hat)
    decoded_state = model.decode_state_tokens(state_x1_hat)

    pred_human_video = decoded_video[:, :, :3]
    pred_object_video = decoded_video[:, :, 3:6]
    zero = output.video_velocity.new_zeros(())

    render_active = weights["object_render"] > 0.0 or weights["branch_coupling"] > 0.0
    if render_active:
        pred_object_render = render_object_branch(
            renderer,
            decoded_state.object_gaussians,
            decoded_state.object_transforms,
            camera_intrinsics_render,
        )
    else:
        pred_object_render = None

    geometry_active = any(
        weights[name] > 0.0
        for name in ("joint_heat", "object_silhouette", "object_depth", "geometry_distill")
    )
    if geometry_active:
        teacher_geometry = model.project_geometry(teacher_state, camera_intrinsics_render)
        pred_geometry = model.project_geometry(decoded_state, camera_intrinsics_render)
        token_hw = teacher_geometry["geometry_maps"].shape[-2:]
        target_keypoint_maps = downsample_spatial_map(keypoint_heatmaps, size=token_hw)
        target_depth = downsample_spatial_map(depth, size=token_hw)
        target_object_mask = downsample_spatial_map(masks_object, size=token_hw)
    else:
        teacher_geometry = None
        pred_geometry = None
        target_keypoint_maps = None
        target_depth = None
        target_object_mask = None

    loss_video_fm = F.mse_loss(output.video_velocity, video_velocity_target)
    loss_state_fm = F.mse_loss(output.state_velocity, state_velocity_target)
    loss_video_latent_recon = F.smooth_l1_loss(video_x1_hat, video_target_tokens)
    loss_state_latent_recon = F.smooth_l1_loss(state_x1_hat, state_target_tokens)
    loss_human_visible = compute_masked_l1(pred_human_video, human_visible_target, masks_human)
    loss_object_video = F.l1_loss(pred_object_video, teacher_object_video)
    loss_object_render = F.l1_loss(pred_object_render, teacher_object_video) if pred_object_render is not None else zero
    loss_branch_coupling = (
        F.l1_loss(pred_object_video, pred_object_render.detach()) if pred_object_render is not None else zero
    )

    loss_human_gaussian = F.smooth_l1_loss(decoded_state.human_gaussians, teacher_state.human_gaussians)
    loss_object_gaussian = F.smooth_l1_loss(decoded_state.object_gaussians, teacher_state.object_gaussians)
    loss_joints = F.smooth_l1_loss(decoded_state.joints_3d, teacher_state.joints_3d)
    loss_object_motion = F.smooth_l1_loss(decoded_state.object_transforms, teacher_state.object_transforms)
    loss_contact = F.smooth_l1_loss(decoded_state.contact_signature, teacher_state.contact_signature)

    loss_joint_heat = (
        F.l1_loss(pred_geometry["geometry_maps"][:, :, 0:1], target_keypoint_maps)
        if pred_geometry is not None
        else zero
    )
    loss_object_silhouette = (
        F.l1_loss(pred_geometry["geometry_maps"][:, :, 2:3], target_object_mask)
        if pred_geometry is not None
        else zero
    )
    loss_object_depth = (
        compute_masked_l1(
            pred_geometry["geometry_maps"][:, :, 3:4],
            target_depth,
            target_object_mask.clamp(min=0.0, max=1.0),
        )
        if pred_geometry is not None
        else zero
    )
    loss_geometry_distill = (
        F.l1_loss(pred_geometry["geometry_maps"], teacher_geometry["geometry_maps"])
        if pred_geometry is not None and teacher_geometry is not None
        else zero
    )

    losses = {
        "video_fm": loss_video_fm,
        "state_fm": loss_state_fm,
        "video_latent": loss_video_latent_recon,
        "state_latent": loss_state_latent_recon,
        "human_visible": loss_human_visible,
        "object_video": loss_object_video,
        "object_render": loss_object_render,
        "branch_coupling": loss_branch_coupling,
        "human_gaussian": loss_human_gaussian,
        "object_gaussian": loss_object_gaussian,
        "joints": loss_joints,
        "object_motion": loss_object_motion,
        "contact": loss_contact,
        "joint_heat": loss_joint_heat,
        "object_silhouette": loss_object_silhouette,
        "object_depth": loss_object_depth,
        "geometry_distill": loss_geometry_distill,
    }
    total_loss = zero
    for name, loss_value in losses.items():
        total_loss = total_loss + float(weights.get(name, 0.0)) * loss_value

    metrics = {"loss_total": total_loss.detach()}
    metrics.update({f"loss_{name}": loss_value.detach() for name, loss_value in losses.items()})
    return total_loss, metrics


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: DualBranchCoGenerativeFlowMatching,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{step:07d}.pt"
    accelerator.save(
        {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
        },
        str(path),
    )


def resume_if_available(
    *,
    args: argparse.Namespace,
    model: DualBranchCoGenerativeFlowMatching,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> int:
    if not args.resume_checkpoint:
        return 0
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch detected while resuming. "
            f"Missing keys: {incompatible.missing_keys[:10]} "
            f"| Unexpected keys: {incompatible.unexpected_keys[:10]}"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
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
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--clip_stride", type=int, default=4)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--dataset_cache_sequences", type=int, default=2)
    parser.add_argument("--cache_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index_progress_every", type=int, default=10)
    parser.add_argument("--background_value", type=float, default=1.0)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--video_channels", type=int, default=6)
    parser.add_argument("--num_human_gaussians", type=int, default=1024)
    parser.add_argument("--num_object_gaussians", type=int, default=1024)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)

    parser.add_argument("--lambda_video_fm", type=float, default=1.0)
    parser.add_argument("--lambda_state_fm", type=float, default=1.0)
    parser.add_argument("--lambda_video_latent", type=float, default=0.1)
    parser.add_argument("--lambda_state_latent", type=float, default=0.1)
    parser.add_argument("--lambda_human_visible", type=float, default=1.0)
    parser.add_argument("--lambda_object_video", type=float, default=1.0)
    parser.add_argument("--lambda_object_render", type=float, default=1.0)
    parser.add_argument("--lambda_branch_coupling", type=float, default=0.25)
    parser.add_argument("--lambda_human_gaussian", type=float, default=0.1)
    parser.add_argument("--lambda_object_gaussian", type=float, default=0.1)
    parser.add_argument("--lambda_joints", type=float, default=1.0)
    parser.add_argument("--lambda_object_motion", type=float, default=0.5)
    parser.add_argument("--lambda_contact", type=float, default=0.25)
    parser.add_argument("--lambda_joint_heat", type=float, default=0.5)
    parser.add_argument("--lambda_object_silhouette", type=float, default=0.5)
    parser.add_argument("--lambda_object_depth", type=float, default=0.25)
    parser.add_argument("--lambda_geometry_distill", type=float, default=0.25)
    parser.add_argument("--curriculum_fusion_start_ratio", type=float, default=0.2)
    parser.add_argument("--curriculum_full_start_ratio", type=float, default=0.6)
    parser.add_argument("--freeze_video_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_unfreeze_start_ratio", type=float, default=-1.0)
    parser.add_argument("--video_stage2_num_top_blocks", type=int, default=2)
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
    if args.image_height % args.patch_size != 0 or args.image_width % args.patch_size != 0:
        raise ValueError(
            f"Image size {(args.image_height, args.image_width)} must be divisible by patch_size={args.patch_size}."
        )
    if args.video_stage2_num_top_blocks < 0:
        raise ValueError(f"`video_stage2_num_top_blocks` must be >= 0, got {args.video_stage2_num_top_blocks}.")
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}.")
    if args.clip_length <= 0:
        raise ValueError(f"`clip_length` must be > 0, got {args.clip_length}.")
    if args.max_steps <= 0:
        raise ValueError(f"`max_steps` must be > 0, got {args.max_steps}.")
    if args.num_workers < 0:
        raise ValueError(f"`num_workers` must be >= 0, got {args.num_workers}.")
    if args.save_every <= 0 or args.log_every <= 0 or args.print_every <= 0:
        raise ValueError("`save_every`, `log_every`, and `print_every` must all be > 0.")
    if args.prefetch_factor <= 0:
        raise ValueError(f"`prefetch_factor` must be > 0, got {args.prefetch_factor}.")
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
    )
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process and log_with is not None:
        accelerator.init_trackers(args.project_name, config=vars(args))
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
        index_progress_every=args.index_progress_every,
        index_progress_callback=report_dataset_index_progress if accelerator.is_main_process else None,
        split_file=args.split_file,
        split_key=args.split_key,
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
    if accelerator.is_main_process:
        print(f"[train_dual_branch_fm] condition channels ready | channels={condition_channels}", flush=True)

    model = DualBranchCoGenerativeFlowMatching(
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
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    global_step = resume_if_available(args=args, model=model, optimizer=optimizer, scheduler=scheduler)
    video_schedule_metrics = apply_video_backbone_schedule(model, args, global_step)
    trainable_parameters = collect_trainable_parameters(model)

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    if len(dataloader) == 0:
        raise ValueError(
            "Dataloader has zero batches. "
            f"clips={len(dataset)}, batch_size={args.batch_size}, drop_last=True. "
            "Lower the batch size or provide more training clips."
        )
    if accelerator.is_main_process:
        print(
            f"[train_dual_branch_fm] dataloader ready "
            f"| batch_size_per_device={args.batch_size} "
            f"| workers={args.num_workers} "
            f"| persistent_workers={args.num_workers > 0}"
            ,
            flush=True,
        )

    renderer = DiffRasterizationLayer(
        image_height=args.image_height,
        image_width=args.image_width,
    )
    renderer = renderer.to(accelerator.device)
    if accelerator.is_main_process:
        print("[train_dual_branch_fm] renderer ready, preparing accelerator-wrapped modules...", flush=True)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        scheduler,
    )
    if accelerator.is_main_process:
        print("[train_dual_branch_fm] accelerator.prepare complete, entering training setup...", flush=True)
    all_parameters = tuple(model.parameters())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    last_video_stage = None
    last_unfrozen_video_blocks = None
    progress_bar = None
    checkpoint_progress_bar = None
    checkpoint_segment_end = None

    def create_checkpoint_progress_bar(step: int):
        if not accelerator.is_main_process:
            return None, None
        segment_size = max(int(args.save_every), 1)
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
            desc=f"checkpoint {segment_start:07d}->{segment_end:07d}",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
            position=1,
            leave=False,
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
            f"| workers={args.num_workers} "
            f"| save_every={args.save_every} "
            f"| log_every={args.log_every}"
            ,
            flush=True,
        )
        progress_bar = tqdm(
            total=args.max_steps,
            initial=global_step,
            desc="train_dual_branch_fm",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
        )
        checkpoint_progress_bar, checkpoint_segment_end = create_checkpoint_progress_bar(global_step)
    last_video_stage = int(video_schedule_metrics["video_optim_stage"])
    last_unfrozen_video_blocks = int(video_schedule_metrics["video_unfrozen_blocks"])

    try:
        while global_step < args.max_steps:
            for batch in dataloader:
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

                    teacher_object_video = render_object_branch(
                        renderer,
                        object_gaussians,
                        object_poses,
                        camera_intrinsics_render,
                    ).detach()
                    video_target = torch.cat([human_visible, teacher_object_video], dim=2)
                    teacher_state = build_teacher_state(
                        {
                            "human_gaussians": human_gaussians,
                            "object_gaussians": object_gaussians,
                            "joints_3d": joints_3d,
                            "object_poses": object_poses,
                            "contact_signature": contact_signature,
                        }
                    )

                    video_target_tokens = model.encode_video_target(video_target)
                    state_target_tokens = model.encode_state_target(
                        human_gaussians=human_gaussians,
                        object_gaussians=object_gaussians,
                        joints_3d=joints_3d,
                        object_transforms=object_poses,
                        contact_signature=contact_signature,
                    )

                    batch_size = video_target_tokens.shape[0]
                    timesteps = torch.rand(batch_size, device=accelerator.device, dtype=video_target_tokens.dtype)
                    t_view = timesteps.view(batch_size, 1, 1)

                    video_noise = torch.randn_like(video_target_tokens)
                    state_noise = torch.randn_like(state_target_tokens)
                    video_xt = t_view * video_target_tokens + (1.0 - t_view) * video_noise
                    state_xt = t_view * state_target_tokens + (1.0 - t_view) * state_noise
                    video_velocity_target = video_target_tokens - video_noise
                    state_velocity_target = state_target_tokens - state_noise

                    output = model(
                        video_xt=video_xt,
                        state_xt=state_xt,
                        timesteps=timesteps,
                        condition_video=condition_video,
                        camera_intrinsics=camera_intrinsics_render,
                    )

                    loss_weights, curriculum_metrics = build_curriculum_loss_weights(args, global_step)
                    loss, metrics = compute_losses(
                        model=model,
                        output=output,
                        video_xt=video_xt,
                        video_velocity_target=video_velocity_target,
                        state_xt=state_xt,
                        state_velocity_target=state_velocity_target,
                        teacher_state=teacher_state,
                        teacher_object_video=teacher_object_video,
                        human_visible_target=human_visible,
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
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(all_parameters, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                reduced_metrics = {
                    key: accelerator.reduce(value.to(accelerator.device), reduction="mean").item()
                    for key, value in metrics.items()
                }
                reduced_metrics.update(curriculum_metrics)
                reduced_metrics.update(video_schedule_metrics)
                reduced_metrics["lr"] = scheduler.get_last_lr()[0]
                reduced_metrics["trainable_params_m"] = (
                    count_parameters(collect_trainable_parameters(accelerator.unwrap_model(model))) / 1e6
                )

                if accelerator.is_main_process and progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=f"{reduced_metrics['loss_total']:.4f}",
                        vfm=f"{reduced_metrics['loss_video_fm']:.4f}",
                        sfm=f"{reduced_metrics['loss_state_fm']:.4f}",
                        stage=int(reduced_metrics["curriculum_stage"]),
                        vstage=int(reduced_metrics["video_optim_stage"]),
                        lr=f"{reduced_metrics['lr']:.2e}",
                        refresh=False,
                    )
                if accelerator.is_main_process and checkpoint_progress_bar is not None:
                    checkpoint_progress_bar.update(1)
                    checkpoint_progress_bar.set_postfix(
                        loss=f"{reduced_metrics['loss_total']:.4f}",
                        remain=max((checkpoint_segment_end or global_step) - global_step, 0),
                        refresh=False,
                    )

                if global_step % args.log_every == 0 and accelerator.is_main_process and log_with is not None:
                    accelerator.log(reduced_metrics, step=global_step)

                if global_step % args.print_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(global_step, 1)) * max(args.max_steps - global_step, 0)
                    next_save = checkpoint_segment_end if checkpoint_segment_end is not None else args.max_steps
                    print(
                        f"[train_dual_branch_fm] step={global_step:07d} "
                        f"loss={reduced_metrics['loss_total']:.4f} "
                        f"vfm={reduced_metrics['loss_video_fm']:.4f} "
                        f"sfm={reduced_metrics['loss_state_fm']:.4f} "
                        f"objR={reduced_metrics['loss_object_render']:.4f} "
                        f"joints={reduced_metrics['loss_joints']:.4f} "
                        f"stage={int(reduced_metrics['curriculum_stage'])} "
                        f"video_stage={int(reduced_metrics['video_optim_stage'])} "
                        f"unfrozen={int(reduced_metrics['video_unfrozen_blocks'])} "
                        f"next_save={next_save:07d} "
                        f"eta={eta / 3600.0:.2f}h"
                        ,
                        flush=True,
                    )

                if global_step % args.save_every == 0:
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
        if checkpoint_progress_bar is not None:
            checkpoint_progress_bar.close()
        if progress_bar is not None:
            progress_bar.close()

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
