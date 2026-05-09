#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model.cointeract_hoi_wan import CoInteractHOI4DModel, DecodedHOIState


def resize_video_batch(video: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
    if video.ndim != 5:
        raise ValueError(f"`video` must have shape [B, T, C, H, W], got {tuple(video.shape)}.")
    batch_size, num_frames, channels = video.shape[:3]
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    video = F.interpolate(
        video.reshape(batch_size * num_frames, channels, video.shape[-2], video.shape[-1]),
        size=size,
        mode=mode,
        align_corners=align_corners,
    )
    return video.reshape(batch_size, num_frames, channels, size[0], size[1])


def scale_camera_intrinsics(camera_intrinsics: Tensor, source_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tensor:
    if source_size == target_size:
        return camera_intrinsics
    source_h, source_w = source_size
    target_h, target_w = target_size
    scaled = camera_intrinsics.clone()
    scaled[..., 0, 0] *= float(target_w) / float(source_w)
    scaled[..., 1, 1] *= float(target_h) / float(source_h)
    scaled[..., 0, 2] *= float(target_w) / float(source_w)
    scaled[..., 1, 2] *= float(target_h) / float(source_h)
    return scaled


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def flow_match_sample(target: Tensor, noise: Tensor, timesteps: Tensor) -> Tuple[Tensor, Tensor]:
    t_view = timesteps.to(device=target.device, dtype=target.dtype).view(target.shape[0], *([1] * (target.ndim - 1)))
    xt = t_view * target + (1.0 - t_view) * noise
    velocity = target - noise
    return xt, velocity


def reconstruct_x1(xt: Tensor, velocity: Tensor, timesteps: Tensor) -> Tensor:
    t_view = timesteps.to(device=xt.device, dtype=xt.dtype).view(xt.shape[0], *([1] * (xt.ndim - 1)))
    return xt + (1.0 - t_view) * velocity


def prepare_batch(batch: Dict[str, object], *, device: torch.device, image_height: int, image_width: int) -> Dict[str, object]:
    tensor_batch: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            tensor_batch[key] = value.to(device=device, non_blocking=True)
        else:
            tensor_batch[key] = value

    rgb = tensor_batch["rgb"].float().clamp(0.0, 1.0)
    source_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
    target_hw = (int(image_height), int(image_width))
    if source_hw != target_hw:
        for key, mode in (
            ("rgb", "bilinear"),
            ("masks_human", "nearest"),
            ("masks_object", "nearest"),
            ("m_primary", "nearest"),
            ("m_secondary", "nearest"),
            ("m_object_region", "nearest"),
            ("depth", "bilinear"),
        ):
            if key in tensor_batch:
                tensor_batch[key] = resize_video_batch(tensor_batch[key].float(), target_hw, mode=mode)
        tensor_batch["camera_intrinsics"] = scale_camera_intrinsics(
            tensor_batch["camera_intrinsics"].float(),
            source_size=source_hw,
            target_size=target_hw,
        )
    tensor_batch["rgb"] = tensor_batch["rgb"].float().clamp(0.0, 1.0)
    return tensor_batch


def build_model(args: argparse.Namespace) -> CoInteractHOI4DModel:
    return CoInteractHOI4DModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_frames=args.clip_length,
        image_height=args.image_height,
        image_width=args.image_width,
        image_patch_size=args.image_patch_size,
        num_human_gaussians=args.num_human_gaussians,
        num_object_gaussians=args.num_object_gaussians,
        num_joints=args.num_joints,
        contact_dim=args.contact_dim,
        human_shape_dim=args.human_shape_dim,
        human_pose_dim=args.human_pose_dim,
        wan_model_id=args.wan_model_id,
        wan_dtype=args.wan_dtype,
        wan_hidden_dim=args.wan_hidden_dim,
        wan_prompt_max_sequence_length=args.wan_prompt_max_sequence_length,
        wan_local_files_only=args.wan_local_files_only,
        freeze_wan=args.freeze_wan,
    )


def encode_state_target(model: CoInteractHOI4DModel, batch: Dict[str, object]) -> Tensor:
    return model.encode_state_target(
        human_shape=batch["human_shape"],
        human_pose=batch["body_pose"],
        human_translation=batch["cam_t"],
        object_transforms=batch["object_poses"],
        contact_signature=batch["contact_signature"],
        human_gaussians=batch["human_gaussians"],
        object_gaussians=batch["object_gaussians"],
        joints_3d=batch["joints_3d"],
    )


def compute_state_losses(
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    *,
    weights: Dict[str, float],
) -> Dict[str, Tensor]:
    losses: Dict[str, Tensor] = {}
    losses["shape"] = F.smooth_l1_loss(decoded.human_shape.float(), batch["human_shape"].float())
    losses["pose"] = F.smooth_l1_loss(decoded.human_pose.float(), batch["body_pose"].float())
    losses["translation"] = F.smooth_l1_loss(decoded.human_translation.float(), batch["cam_t"].float())
    losses["object_pose"] = F.smooth_l1_loss(decoded.object_transforms.float(), batch["object_poses"].float())
    losses["contact"] = F.smooth_l1_loss(decoded.contact_signature.float(), batch["contact_signature"].float())
    losses["joints"] = F.smooth_l1_loss(decoded.joints_3d.float(), batch["joints_3d"].float())
    losses["human_gaussian"] = F.smooth_l1_loss(decoded.human_gaussians.float(), batch["human_gaussians"].float())
    losses["object_gaussian"] = F.smooth_l1_loss(decoded.object_gaussians.float(), batch["object_gaussians"].float())
    total = decoded.human_shape.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["supervised"] = total
    return losses


def _to_wandb_image_grid(frames: Tensor, *, max_frames: int, normalize: bool = False):
    frames = frames.detach().float().cpu()
    if frames.ndim == 3:
        frames = frames[:, None]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}.")
    frames = frames[:max_frames]
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    elif frames.shape[1] > 3:
        frames = frames[:, :3]
    if normalize:
        flat = frames.flatten(1)
        vmin = flat.min(dim=1).values.view(-1, 1, 1, 1)
        vmax = flat.max(dim=1).values.view(-1, 1, 1, 1)
        frames = (frames - vmin) / (vmax - vmin).clamp_min(1e-6)
    frames = frames.clamp(0.0, 1.0)
    try:
        from torchvision.utils import make_grid

        grid = make_grid(frames, nrow=min(int(frames.shape[0]), 4), padding=2)
    except Exception:
        rows: List[Tensor] = []
        for start in range(0, int(frames.shape[0]), 4):
            row = torch.cat(list(frames[start : start + 4]), dim=-1)
            rows.append(row)
        width = max(row.shape[-1] for row in rows)
        padded_rows = [F.pad(row, (0, width - row.shape[-1], 0, 0)) for row in rows]
        grid = torch.cat(padded_rows, dim=-2)
    image = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    import wandb

    return wandb.Image(image)


def _gaussians_to_wandb_object3d(gaussians: Tensor, *, max_points: int):
    points = gaussians.detach().float().cpu()
    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])
    if points.numel() == 0:
        return None
    if points.shape[-1] < 3:
        return None
    if points.shape[0] > max_points:
        indices = torch.linspace(0, points.shape[0] - 1, max_points).long()
        points = points.index_select(0, indices)
    xyz = points[:, :3]
    if points.shape[-1] >= 14:
        rgb = points[:, 11:14]
    elif points.shape[-1] >= 6:
        rgb = points[:, 3:6]
    else:
        rgb = torch.full_like(xyz, 0.65)
    rgb = rgb.clamp(0.0, 1.0)
    cloud = torch.cat([xyz, (rgb * 255.0).round()], dim=-1).numpy()
    import wandb

    return wandb.Object3D(cloud)


def log_wandb_visuals(
    *,
    batch: Dict[str, object],
    decoded: DecodedHOIState,
    step: int,
    args: argparse.Namespace,
) -> None:
    if args.log_with != "wandb" or args.train_visual_every <= 0:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[train_cointeract_hoi] wandb visual skipped: {exc}", flush=True)
        return
    if wandb.run is None:
        return

    sample_index = 0
    max_frames = int(args.train_visual_max_frames)
    max_points = int(args.train_visual_max_points)
    payload = {
        "train_visual/rgb": _to_wandb_image_grid(batch["rgb"][sample_index], max_frames=max_frames),
        "train_visual/human_mask": _to_wandb_image_grid(batch["masks_human"][sample_index], max_frames=max_frames),
        "train_visual/object_mask": _to_wandb_image_grid(batch["masks_object"][sample_index], max_frames=max_frames),
        "train_visual/depth": _to_wandb_image_grid(
            batch["depth"][sample_index],
            max_frames=max_frames,
            normalize=True,
        ),
    }
    object3d_items = {
        "train_visual/pred_human_gaussians": _gaussians_to_wandb_object3d(
            decoded.human_gaussians[sample_index],
            max_points=max_points,
        ),
        "train_visual/gt_human_gaussians": _gaussians_to_wandb_object3d(
            batch["human_gaussians"][sample_index],
            max_points=max_points,
        ),
        "train_visual/pred_object_gaussians": _gaussians_to_wandb_object3d(
            decoded.object_gaussians[sample_index],
            max_points=max_points,
        ),
        "train_visual/gt_object_gaussians": _gaussians_to_wandb_object3d(
            batch["object_gaussians"][sample_index],
            max_points=max_points,
        ),
    }
    payload.update({key: value for key, value in object3d_items.items() if value is not None})
    wandb.log(payload, step=step)


def ensure_wan_loaded_rank_by_rank(raw_model: CoInteractHOI4DModel, accelerator: Accelerator) -> None:
    for process_index in range(accelerator.num_processes):
        if accelerator.process_index == process_index:
            print(
                f"[train_cointeract_hoi] loading Wan stream on rank {process_index}/{accelerator.num_processes - 1}",
                flush=True,
            )
            raw_model.ensure_wan_loaded(accelerator.device)
            print(f"[train_cointeract_hoi] Wan stream loaded on rank {process_index}", flush=True)
        accelerator.wait_for_everyone()


def checkpoint_state_dict(model: CoInteractHOI4DModel) -> Dict[str, Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("rgb_stream.")
    }


def save_checkpoint(
    *,
    path: Path,
    model: CoInteractHOI4DModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model": checkpoint_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_model_checkpoint(model: CoInteractHOI4DModel, checkpoint_path: str) -> int:
    if not checkpoint_path:
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("rgb_stream.")]
    if missing or incompatible.unexpected_keys:
        print(
            "[train_cointeract_hoi] checkpoint mismatch "
            f"| missing={len(missing)} | unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def load_training_checkpoint(
    *,
    model: CoInteractHOI4DModel,
    optimizer,
    scheduler,
    checkpoint_path: str,
) -> int:
    if not checkpoint_path:
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("rgb_stream.")]
    if missing or incompatible.unexpected_keys:
        print(
            "[train_cointeract_hoi] resume mismatch "
            f"| missing={len(missing)} | unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoInteract-style RGB->HOI dual-stream Wan model.")
    parser.add_argument("--data_root", type=str, default="sample_data/behave_1pct/sequences")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--human_gaussian_source", type=str, default="smpl_mesh", choices=("smpl_mesh", "teacher"))
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="outputs/cointeract_hoi_wan")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--log_with", type=str, default="none", choices=("none", "tensorboard", "wandb"))
    parser.add_argument("--project_name", type=str, default="uni-hoi-4d")
    parser.add_argument("--run_name", type=str, default="cointeract_rgb_to_hoi_wan_ti2v")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=7000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--train_visual_every", type=int, default=500)
    parser.add_argument("--train_visual_max_frames", type=int, default=8)
    parser.add_argument("--train_visual_max_points", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--clip_length", type=int, default=9)
    parser.add_argument("--clip_stride", type=int, default=8)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--dataset_cache_sequences", type=int, default=4)
    parser.add_argument("--cache_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb_cache_max_frames", type=int, default=256)
    parser.add_argument("--prefer_h5_cache", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--wan_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--wan_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--wan_hidden_dim", type=int, default=3072)
    parser.add_argument("--wan_prompt_max_sequence_length", type=int, default=512)
    parser.add_argument("--wan_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_wan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial_wan_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb_to_hoi_scale", type=float, default=1.0)

    parser.add_argument("--num_human_gaussians", type=int, default=750)
    parser.add_argument("--num_object_gaussians", type=int, default=750)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--human_shape_dim", type=int, default=10)
    parser.add_argument("--human_pose_dim", type=int, default=72)

    parser.add_argument("--lambda_state_fm", type=float, default=1.0)
    parser.add_argument("--lambda_rgb_fm", type=float, default=0.0)
    parser.add_argument("--lambda_shape", type=float, default=0.1)
    parser.add_argument("--lambda_pose", type=float, default=0.5)
    parser.add_argument("--lambda_translation", type=float, default=0.5)
    parser.add_argument("--lambda_object_pose", type=float, default=0.5)
    parser.add_argument("--lambda_contact", type=float, default=0.1)
    parser.add_argument("--lambda_joints", type=float, default=1.0)
    parser.add_argument("--lambda_human_gaussian", type=float, default=1.0)
    parser.add_argument("--lambda_object_gaussian", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.clip_length - 1) % 4 != 0:
        raise ValueError(f"Wan2.2-TI2V requires clip_length = 4k + 1, got {args.clip_length}.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.log_with == "none" else args.log_with,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)
    if args.log_with != "none":
        tracker_kwargs = {"wandb": {"name": args.run_name}} if args.log_with == "wandb" else {}
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs=tracker_kwargs,
        )

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "launch_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(
            "[train_cointeract_hoi] starting "
            f"| data_root={args.data_root} | max_steps={args.max_steps} "
            f"| wan={args.wan_model_id} | input=image_only",
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
        max_sequences=args.max_sequences,
        cache_sequences=args.dataset_cache_sequences,
        cache_rgb=args.cache_rgb,
        rgb_cache_max_frames=args.rgb_cache_max_frames,
        split_file=args.split_file,
        split_key=args.split_key,
        prefer_h5_cache=args.prefer_h5_cache,
        include_human_vertices=False,
        include_keypoint_heatmaps=False,
    )

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    model = build_model(args)
    if args.init_checkpoint:
        load_model_checkpoint(model, args.init_checkpoint)
    if not args.freeze_wan and not args.serial_wan_load:
        model.ensure_wan_loaded(accelerator.device)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    raw_model: CoInteractHOI4DModel = accelerator.unwrap_model(model)
    global_step = 0
    if args.resume_checkpoint:
        global_step = load_training_checkpoint(
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=args.resume_checkpoint,
        )
    if args.serial_wan_load:
        ensure_wan_loaded_rank_by_rank(raw_model, accelerator)

    supervised_weights = {
        "shape": args.lambda_shape,
        "pose": args.lambda_pose,
        "translation": args.lambda_translation,
        "object_pose": args.lambda_object_pose,
        "contact": args.lambda_contact,
        "joints": args.lambda_joints,
        "human_gaussian": args.lambda_human_gaussian,
        "object_gaussian": args.lambda_object_gaussian,
    }

    model.train()
    start_time = time.time()
    progress = tqdm(total=args.max_steps, initial=global_step, disable=not accelerator.is_main_process)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed + accelerator.process_index)

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break
            batch = prepare_batch(
                batch,
                device=accelerator.device,
                image_height=args.image_height,
                image_width=args.image_width,
            )
            with accelerator.accumulate(model):
                rgb = batch["rgb"]
                timesteps = torch.rand(rgb.shape[0], device=accelerator.device).clamp(1e-4, 1.0 - 1e-4)

                with torch.no_grad():
                    video_target = raw_model.encode_video_target(rgb)
                    first_frame_latents = raw_model.encode_first_frame(rgb[:, 0])
                    video_xt, video_velocity_target = raw_model.build_noisy_video_latents(
                        video_target,
                        first_frame_latents,
                        timesteps,
                        generator=generator,
                    )
                    state_target = encode_state_target(raw_model, batch)
                    state_noise = torch.randn(
                        state_target.shape,
                        generator=generator,
                        device=state_target.device,
                        dtype=state_target.dtype,
                    )
                    state_xt, state_velocity_target = flow_match_sample(state_target, state_noise, timesteps)

                output = model(
                    video_xt=video_xt,
                    state_xt=state_xt,
                    timesteps=timesteps,
                    first_frame=rgb[:, 0],
                    rgb_to_hoi_scale=args.rgb_to_hoi_scale,
                )
                state_fm = F.mse_loss(output.state_velocity.float(), state_velocity_target.float())
                state_losses = compute_state_losses(output.decoded_state, batch, weights=supervised_weights)
                loss = args.lambda_state_fm * state_fm + state_losses["supervised"]

                rgb_fm = video_xt.new_zeros(())
                if args.lambda_rgb_fm > 0.0:
                    rgb_fm = F.mse_loss(output.rgb_velocity[:, :, 1:].float(), video_velocity_target[:, :, 1:].float())
                    loss = loss + args.lambda_rgb_fm * rgb_fm

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if accelerator.is_main_process and (global_step == 1 or global_step % args.log_every == 0):
                    elapsed = time.time() - start_time
                    metrics = {
                        "loss": float(loss.detach().item()),
                        "loss_state_fm": float(state_fm.detach().item()),
                        "loss_rgb_fm": float(rgb_fm.detach().item()),
                        "loss_supervised": float(state_losses["supervised"].detach().item()),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    for key, value in state_losses.items():
                        if key != "supervised":
                            metrics[f"loss_{key}"] = float(value.detach().item())
                    print(
                        f"[train_cointeract_hoi] step={global_step:07d}/{args.max_steps:07d} "
                        f"loss={metrics['loss']:.5f} state_fm={metrics['loss_state_fm']:.5f} "
                        f"sup={metrics['loss_supervised']:.5f} lr={metrics['lr']:.2e} "
                        f"elapsed={elapsed/60.0:.1f}m",
                        flush=True,
                    )
                    if args.log_with != "none":
                        accelerator.log(metrics, step=global_step)

                if (
                    accelerator.is_main_process
                    and args.train_visual_every > 0
                    and global_step % args.train_visual_every == 0
                ):
                    log_wandb_visuals(
                        batch=batch,
                        decoded=output.decoded_state,
                        step=global_step,
                        args=args,
                    )

                if accelerator.is_main_process and (global_step % args.save_every == 0 or global_step == args.max_steps):
                    save_checkpoint(
                        path=Path(args.output_dir) / "checkpoints" / f"checkpoint_{global_step:07d}.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        args=args,
                    )

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            path=Path(args.output_dir) / "checkpoints" / "last.pt",
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            args=args,
        )
        print(f"[train_cointeract_hoi] done | step={global_step:07d} | output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
