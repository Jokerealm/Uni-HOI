#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image, ImageDraw
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model import DecodedHOIState, UniModel, UniModelOutput


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


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str,
    min_lr_ratio: float,
) -> LambdaLR:
    scheduler_type = scheduler_type.lower()
    if scheduler_type not in {"constant", "cosine", "linear"}:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_type}.")
    min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if scheduler_type == "constant":
            return 1.0
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay = 1.0 - progress
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def flow_match_sample(target: Tensor, noise: Tensor, timesteps: Tensor) -> Tuple[Tensor, Tensor]:
    t_view = timesteps.to(device=target.device, dtype=target.dtype).view(target.shape[0], *([1] * (target.ndim - 1)))
    xt = t_view * target + (1.0 - t_view) * noise
    velocity = target - noise
    return xt, velocity


def gaussian_chamfer_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_xyz = pred[..., :3].float()
    target_xyz = target[..., :3].float()
    dists = torch.cdist(pred_xyz, target_xyz)
    pred_to_target = dists.min(dim=-1).values.mean()
    target_to_pred = dists.min(dim=-2).values.mean()
    return pred_to_target + target_to_pred


def gaussian_attr_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape[-1] <= 3 or target.shape[-1] <= 3:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[..., 3:].float(), target[..., 3:].float())


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
    losses["human_gaussian_chamfer"] = gaussian_chamfer_loss(decoded.human_gaussians, batch["human_gaussians"])
    losses["object_gaussian_chamfer"] = gaussian_chamfer_loss(decoded.object_gaussians, batch["object_gaussians"])
    losses["human_gaussian_xyz"] = F.smooth_l1_loss(
        decoded.human_gaussians[..., :3].float(),
        batch["human_gaussians"][..., :3].float(),
    )
    losses["object_gaussian_xyz"] = F.smooth_l1_loss(
        decoded.object_gaussians[..., :3].float(),
        batch["object_gaussians"][..., :3].float(),
    )
    losses["human_gaussian_attr"] = gaussian_attr_l1_loss(decoded.human_gaussians, batch["human_gaussians"])
    losses["object_gaussian_attr"] = gaussian_attr_l1_loss(decoded.object_gaussians, batch["object_gaussians"])

    total = decoded.human_shape.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["supervised"] = total
    return losses


def supervised_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "shape": args.lambda_shape,
        "pose": args.lambda_pose,
        "translation": args.lambda_translation,
        "object_pose": args.lambda_object_pose,
        "contact": args.lambda_contact,
        "joints": args.lambda_joints,
        "human_gaussian_chamfer": args.lambda_human_gaussian * args.lambda_gaussian_chamfer,
        "object_gaussian_chamfer": args.lambda_object_gaussian * args.lambda_gaussian_chamfer,
        "human_gaussian_xyz": args.lambda_human_gaussian * args.lambda_gaussian_xyz_l1,
        "object_gaussian_xyz": args.lambda_object_gaussian * args.lambda_gaussian_xyz_l1,
        "human_gaussian_attr": args.lambda_human_gaussian * args.lambda_gaussian_attr_l1,
        "object_gaussian_attr": args.lambda_object_gaussian * args.lambda_gaussian_attr_l1,
    }


def build_dataset(
    args: argparse.Namespace,
    *,
    include_human_vertices: bool = False,
    data_root: Optional[str] = None,
    split_file: Optional[str] = None,
    split_key: Optional[str] = None,
) -> DualBranchHOIDataset:
    return DualBranchHOIDataset(
        data_root=data_root or args.data_root,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        processed_subdir=args.processed_subdir,
        gs_subdir=args.gs_subdir,
        human_gaussian_source=args.human_gaussian_source,
        num_human_gaussians=args.num_human_gaussians,
        num_object_gaussians=args.num_object_gaussians,
        num_joints=args.num_joints,
        contact_dim=args.contact_dim,
        coordinate_mode=args.coordinate_mode,
        max_sequences=args.max_sequences,
        cache_sequences=args.dataset_cache_sequences,
        cache_rgb=args.cache_rgb,
        rgb_cache_max_frames=args.rgb_cache_max_frames,
        split_file=args.split_file if split_file is None else split_file,
        split_key=args.split_key if split_key is None else split_key,
        prefer_h5_cache=args.prefer_h5_cache,
        include_human_vertices=include_human_vertices,
        include_keypoint_heatmaps=False,
    )


def resolve_test_data_root(args: argparse.Namespace) -> str:
    return args.test_data_root or args.data_root


def resolve_test_split_file(args: argparse.Namespace) -> str:
    return args.test_split_file if args.test_split_file else args.split_file


def resolve_test_split_key(args: argparse.Namespace) -> str:
    return args.test_split_key if args.test_split_key else args.split_key


def build_test_dataset(args: argparse.Namespace, *, include_human_vertices: bool = False) -> DualBranchHOIDataset:
    return build_dataset(
        args,
        include_human_vertices=include_human_vertices,
        data_root=resolve_test_data_root(args),
        split_file=resolve_test_split_file(args),
        split_key=resolve_test_split_key(args),
    )


def build_dataloader(args: argparse.Namespace, dataset: DualBranchHOIDataset, *, train: bool) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": bool(train),
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": bool(train),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_model(args: argparse.Namespace) -> UniModel:
    return UniModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_frames=args.clip_length,
        image_height=args.image_height,
        image_width=args.image_width,
        latent_patch_size=tuple(args.latent_patch_size),
        num_human_gaussians=args.num_human_gaussians,
        num_object_gaussians=args.num_object_gaussians,
        num_joints=args.num_joints,
        contact_dim=args.contact_dim,
        human_shape_dim=args.human_shape_dim,
        human_pose_dim=args.human_pose_dim,
        wan_vae_model_id=args.wan_vae_model_id,
        wan_vae_subfolder=args.wan_vae_subfolder,
        wan_vae_dtype=args.wan_vae_dtype,
        wan_vae_local_files_only=args.wan_vae_local_files_only,
        vae_latent_channels=args.vae_latent_channels,
        vae_scale_factor_temporal=args.vae_scale_factor_temporal,
        vae_scale_factor_spatial=args.vae_scale_factor_spatial,
    )


def format_param_count(count: int) -> str:
    return f"{int(count):,} ({float(count) / 1_000_000.0:.2f}M)"


def summarize_model_parameters(model: UniModel) -> Dict[str, int]:
    total = 0
    trainable = 0
    vae_total = 0
    vae_trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
        if name.startswith("vae_encoder.vae."):
            vae_total += count
            if parameter.requires_grad:
                vae_trainable += count
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "vae_total": vae_total,
        "vae_trainable": vae_trainable,
        "vae_frozen": vae_total - vae_trainable,
        "non_vae_total": total - vae_total,
        "non_vae_trainable": trainable - vae_trainable,
    }


def print_model_parameter_summary(model: UniModel, *, prefix: str = "model params") -> None:
    stats = summarize_model_parameters(model)
    print(
        f"{prefix} "
        f"| total={format_param_count(stats['total'])} "
        f"| trainable={format_param_count(stats['trainable'])} "
        f"| frozen={format_param_count(stats['frozen'])} "
        f"| frozen_vae={format_param_count(stats['vae_frozen'])} "
        f"| non_vae_trainable={format_param_count(stats['non_vae_trainable'])}",
        flush=True,
    )


def encode_state_target(model: UniModel, batch: Dict[str, object]) -> Tensor:
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


def make_fake_vae_latents(args: argparse.Namespace, batch_size: int, device: torch.device) -> Tensor:
    latent_frames = (args.clip_length - 1) // args.vae_scale_factor_temporal + 1
    latent_h = args.image_height // args.vae_scale_factor_spatial
    latent_w = args.image_width // args.vae_scale_factor_spatial
    return torch.randn(
        batch_size,
        args.vae_latent_channels,
        latent_frames,
        latent_h,
        latent_w,
        device=device,
        dtype=torch.float32,
    )


def forward_and_loss(
    *,
    model,
    codec_model: UniModel,
    batch: Dict[str, object],
    args: argparse.Namespace,
    weights: Dict[str, float],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, Tensor], UniModelOutput]:
    rgb = batch["rgb"]
    batch_size = int(rgb.shape[0])
    state_fm = rgb.new_zeros(())
    timesteps: Optional[Tensor] = None
    state_xt: Optional[Tensor] = None

    if args.use_state_fm:
        timesteps = torch.rand(batch_size, device=rgb.device).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            state_target = encode_state_target(codec_model, batch)
            state_noise = torch.randn(
                state_target.shape,
                generator=generator,
                device=state_target.device,
                dtype=state_target.dtype,
            )
            state_xt, state_velocity_target = flow_match_sample(state_target, state_noise, timesteps)
    else:
        state_velocity_target = None

    if args.debug_fake_vae_latents:
        vae_latents = make_fake_vae_latents(args, batch_size, rgb.device)
        output = codec_model.forward_from_latents(vae_latents=vae_latents, timesteps=timesteps, state_xt=state_xt)
    else:
        output = model(rgb=rgb, timesteps=timesteps, state_xt=state_xt)

    if args.use_state_fm:
        if output.state_velocity is None or state_velocity_target is None:
            raise RuntimeError("UniModel did not return state_velocity while --use_state_fm is enabled.")
        state_fm = F.mse_loss(output.state_velocity.float(), state_velocity_target.float())

    state_losses = compute_state_losses(output.decoded_state, batch, weights=weights)
    loss = state_losses["supervised"] + float(args.lambda_state_fm) * state_fm
    metrics = {"loss": loss.detach(), "loss_state_fm": state_fm.detach()}
    for key, value in state_losses.items():
        metrics[f"loss_{key}"] = value.detach()
    return loss, metrics, output


def checkpoint_state_dict(model: UniModel) -> Dict[str, Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("vae_encoder.vae.")
    }


def save_checkpoint(
    *,
    path: Path,
    model: UniModel,
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model": checkpoint_state_dict(model),
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    *,
    model: UniModel,
    checkpoint_path: str,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
) -> int:
    if not checkpoint_path:
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("vae_encoder.vae.")]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(f"checkpoint mismatch | missing={len(missing)} | unexpected={len(unexpected)}", flush=True)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def ensure_vae_loaded_rank_by_rank(model: UniModel, accelerator: Accelerator, args: argparse.Namespace) -> None:
    if args.debug_fake_vae_latents:
        return
    for process_index in range(accelerator.num_processes):
        if accelerator.process_index == process_index:
            print(
                f"loading frozen Wan VAE on rank {process_index}/{accelerator.num_processes - 1}",
                flush=True,
            )
            model.ensure_vae_loaded(accelerator.device)
            print(f"frozen Wan VAE loaded on rank {process_index}", flush=True)
        accelerator.wait_for_everyone()


def tensor_metrics_to_float(metrics: Dict[str, Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().float().item()) for key, value in metrics.items()}


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def transform_object_points(points: Tensor, transforms: Tensor) -> Tensor:
    points = points.float()
    transforms = transforms.float()
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat([points, ones], dim=-1)
    return torch.matmul(transforms.unsqueeze(-3), homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]


def select_render_points(points: Tensor, max_points: int) -> Tensor:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).round().long()
    return points.index_select(0, indices)


def project_points(
    points: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
    *,
    flip_y: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    points = points.detach().float()
    intrinsics = intrinsics.detach().float()
    z = points[:, 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-4)
    x = points[:, 0] / z.clamp(min=1e-4)
    y = points[:, 1] / z.clamp(min=1e-4)
    u = intrinsics[0, 0] * x + intrinsics[0, 2]
    v = intrinsics[1, 1] * y + intrinsics[1, 2]
    if flip_y:
        v = float(height - 1) - v
    valid = valid & torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def rgb_tensor_to_image(frame: Tensor) -> Image.Image:
    array = (
        frame.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def draw_points(
    image: Image.Image,
    points: Tensor,
    intrinsics: Tensor,
    *,
    color: Tuple[int, int, int],
    radius: int,
    max_points: int,
    flip_y: bool = False,
) -> Image.Image:
    points = select_render_points(points.detach().float(), max_points=max_points).cpu()
    intrinsics = intrinsics.detach().float().cpu()
    width, height = image.size
    u, v, valid = project_points(points, intrinsics, height=height, width=width, flip_y=flip_y)
    if not bool(valid.any()):
        return image

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    depth = points[valid_indices, 2]
    draw_order = valid_indices[torch.argsort(depth, descending=True)]
    draw = ImageDraw.Draw(image, mode="RGBA")
    rgba = (int(color[0]), int(color[1]), int(color[2]), 220)
    outline = (0, 0, 0, 100)
    for idx in draw_order.tolist():
        cx = float(u[idx])
        cy = float(v[idx])
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=rgba,
            outline=outline,
        )
    return image


def add_label(image: Image.Image, label: str) -> Image.Image:
    width, height = image.size
    labeled = Image.new("RGB", (width, height + 22), (255, 255, 255))
    labeled.paste(image, (0, 22))
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, width, 21), fill=(20, 20, 20))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return labeled


def build_screen_camera(intrinsics: Tensor, *, height: int, width: int, device: torch.device):
    from pytorch3d.renderer import PerspectiveCameras

    rotation = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
    rotation[:, 0, 0] = -1.0
    rotation[:, 1, 1] = -1.0
    translation = torch.zeros(1, 3, device=device, dtype=torch.float32)
    focal = torch.tensor(
        [[float(intrinsics[0, 0]), float(intrinsics[1, 1])]],
        device=device,
        dtype=torch.float32,
    )
    principal = torch.tensor(
        [[float(intrinsics[0, 2]), float(intrinsics[1, 2])]],
        device=device,
        dtype=torch.float32,
    )
    return PerspectiveCameras(
        focal_length=focal,
        principal_point=principal,
        image_size=((int(height), int(width)),),
        in_ndc=False,
        R=rotation,
        T=translation,
        device=device,
    )


def render_pointcloud_with_pytorch3d(
    *,
    point_groups: Tuple[Tuple[Tensor, Tuple[float, float, float]], ...],
    intrinsics: Tensor,
    height: int,
    width: int,
    max_points: int,
    point_radius_px: int,
    points_per_pixel: int = 8,
) -> Image.Image:
    from pytorch3d.structures import Pointclouds
    from render.pyt3d_wrapper import PcloudRenderer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_all = []
    colors_all = []
    for points, color in point_groups:
        points = select_render_points(points.detach().float(), max_points=max_points).to(device)
        valid = torch.isfinite(points).all(dim=-1) & (points[:, 2] > 1e-4)
        points = points[valid]
        if points.numel() == 0:
            continue
        colors = points.new_tensor(color).view(1, 3).expand(points.shape[0], -1)
        points_all.append(points)
        colors_all.append(colors)

    if not points_all:
        return Image.new("RGB", (width, height), (255, 255, 255))

    pointcloud = Pointclouds(points=[torch.cat(points_all, dim=0)], features=[torch.cat(colors_all, dim=0)])
    camera = build_screen_camera(intrinsics.detach().float().cpu(), height=height, width=width, device=device)
    radius_ndc = max(float(point_radius_px) * 2.0 / float(max(height, width)), 1e-4)
    renderer = PcloudRenderer(
        image_size=(height, width),
        radius=radius_ndc,
        points_per_pixel=int(points_per_pixel),
        device=str(device),
        bin_size=0,
        background_color=(1.0, 1.0, 1.0),
    )
    image_np = renderer.render(pointcloud, cameras=camera, mode="image")
    image_np = (image_np.clip(0.0, 1.0) * 255.0).round().astype("uint8")
    return Image.fromarray(image_np)


def write_ascii_pointcloud_ply(path: Path, points: Tensor, color: Tuple[int, int, int]) -> None:
    points_np = points.detach().float().cpu().numpy()
    valid = np.isfinite(points_np).all(axis=-1)
    points_np = points_np[valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    r, g, b = (int(color[0]), int(color[1]), int(color[2]))
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points_np.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z in points_np:
            handle.write(f"{x:.7f} {y:.7f} {z:.7f} {r} {g} {b}\n")


def export_pointcloud_debug(
    sample_dir: Path,
    *,
    sequence_name: object,
    pred_human_base: Tensor,
    pred_object_base: Tensor,
    target_human_base: Tensor,
    target_object_base: Tensor,
    pred_human_relative0: Tensor,
    pred_object_relative0: Tensor,
    target_human_relative0: Tensor,
    target_object_relative0: Tensor,
) -> None:
    pointcloud_dir = sample_dir / "pointcloud"
    write_ascii_pointcloud_ply(pointcloud_dir / "pred_human_canonical.ply", pred_human_base, (45, 130, 255))
    write_ascii_pointcloud_ply(pointcloud_dir / "pred_object_canonical.ply", pred_object_base, (255, 140, 30))
    write_ascii_pointcloud_ply(pointcloud_dir / "gt_human_canonical.ply", target_human_base, (39, 174, 96))
    write_ascii_pointcloud_ply(pointcloud_dir / "gt_object_canonical.ply", target_object_base, (155, 89, 182))
    write_ascii_pointcloud_ply(pointcloud_dir / "pred_human_relative_frame0000.ply", pred_human_relative0, (45, 130, 255))
    write_ascii_pointcloud_ply(pointcloud_dir / "pred_object_relative_frame0000.ply", pred_object_relative0, (255, 140, 30))
    write_ascii_pointcloud_ply(pointcloud_dir / "gt_human_relative_frame0000.ply", target_human_relative0, (39, 174, 96))
    write_ascii_pointcloud_ply(pointcloud_dir / "gt_object_relative_frame0000.ply", target_object_relative0, (155, 89, 182))
    meta = {
        "sequence_name": sequence_name,
        "num_pred_human_points": int(pred_human_base.shape[0]),
        "num_pred_object_points": int(pred_object_base.shape[0]),
        "num_gt_human_points": int(target_human_base.shape[0]),
        "num_gt_object_points": int(target_object_base.shape[0]),
        "frame_index": 0,
        "notes": [
            "canonical clouds use decoded/target Gaussian xyz before per-frame transforms",
            "relative_frame0000 applies human-relative translation or object transform for frame 0",
            "PNG/GIF renders are produced with render.pyt3d_wrapper.PcloudRenderer",
        ],
    }
    write_json(pointcloud_dir / "pointcloud_meta.json", meta)


def overlay_pointcloud_render(base: Image.Image, render: Image.Image) -> Image.Image:
    render_rgb = render.convert("RGB")
    base_rgb = base.convert("RGB")
    render_tensor = torch.from_numpy(np.asarray(render_rgb).copy()).float()
    mask = (render_tensor < 250.0).any(dim=-1).byte().cpu().numpy() * 190
    alpha = Image.fromarray(mask)
    return Image.composite(render_rgb, base_rgb, alpha)


def make_render_panel(
    *,
    rgb_frame: Tensor,
    intrinsics: Tensor,
    pred_human: Tensor,
    pred_object: Tensor,
    target_human: Tensor,
    target_object: Tensor,
    radius: int,
    max_points: int,
    flip_y: bool = True,
) -> Image.Image:
    input_image = rgb_tensor_to_image(rgb_frame)
    width, height = input_image.size
    try:
        pred_image = render_pointcloud_with_pytorch3d(
            point_groups=(
                (pred_human, (45 / 255.0, 130 / 255.0, 1.0)),
                (pred_object, (1.0, 140 / 255.0, 30 / 255.0)),
            ),
            intrinsics=intrinsics,
            height=height,
            width=width,
            max_points=max_points,
            point_radius_px=radius,
        )
        target_image = render_pointcloud_with_pytorch3d(
            point_groups=(
                (target_human, (39 / 255.0, 174 / 255.0, 96 / 255.0)),
                (target_object, (155 / 255.0, 89 / 255.0, 182 / 255.0)),
            ),
            intrinsics=intrinsics,
            height=height,
            width=width,
            max_points=max_points,
            point_radius_px=radius,
        )
        if flip_y:
            # Gaussian panels are visualized in image space, where y grows downward.
            pred_image = pred_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            target_image = target_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        overlay_image = overlay_pointcloud_render(input_image, pred_image)
    except Exception:
        pred_image = Image.new("RGB", (width, height), (255, 255, 255))
        target_image = Image.new("RGB", (width, height), (255, 255, 255))
        overlay_image = input_image.copy()
        pred_image = draw_points(
            pred_image,
            pred_human,
            intrinsics,
            color=(45, 130, 255),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )
        pred_image = draw_points(
            pred_image,
            pred_object,
            intrinsics,
            color=(255, 140, 30),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )
        target_image = draw_points(
            target_image,
            target_human,
            intrinsics,
            color=(39, 174, 96),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )
        target_image = draw_points(
            target_image,
            target_object,
            intrinsics,
            color=(155, 89, 182),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )
        overlay_image = draw_points(
            overlay_image,
            pred_human,
            intrinsics,
            color=(45, 130, 255),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )
        overlay_image = draw_points(
            overlay_image,
            pred_object,
            intrinsics,
            color=(255, 140, 30),
            radius=radius,
            max_points=max_points,
            flip_y=flip_y,
        )

    panels = [
        add_label(input_image, "input rgb"),
        add_label(pred_image, "pred human/object"),
        add_label(target_image, "target human/object"),
        add_label(overlay_image, "pred overlay"),
    ]
    panel = Image.new("RGB", (sum(item.size[0] for item in panels), panels[0].size[1]), (255, 255, 255))
    offset = 0
    for item in panels:
        panel.paste(item, (offset, 0))
        offset += item.size[0]
    return panel


def render_prediction_batch(
    render_dir: Path,
    *,
    output: UniModelOutput,
    batch: Dict[str, object],
    radius: int,
    max_points: int,
) -> None:
    decoded = output.decoded_state
    rgb = batch["rgb"].detach().float().cpu()
    intrinsics = batch["camera_intrinsics"].detach().float().cpu()
    pred_human_base = decoded.human_gaussians.detach().float().cpu()[..., :3]
    pred_object_base = decoded.object_gaussians.detach().float().cpu()[..., :3]
    pred_human_translation = decoded.human_translation.detach().float().cpu()
    pred_object_transforms = decoded.object_transforms.detach().float().cpu()

    target_human_base = batch["human_gaussians"].detach().float().cpu()[..., :3]
    target_object_base = batch["object_gaussians"].detach().float().cpu()[..., :3]
    target_human_translation = batch["cam_t"].detach().float().cpu()
    target_object_transforms = batch["object_poses"].detach().float().cpu()

    render_dir.mkdir(parents=True, exist_ok=True)
    batch_size, num_frames = rgb.shape[:2]
    for sample_idx in range(batch_size):
        sample_dir = render_dir if batch_size == 1 else render_dir / f"sample_{sample_idx:02d}"
        frames_dir = sample_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        pred_object_relative0 = transform_object_points(
            pred_object_base[sample_idx],
            pred_object_transforms[sample_idx, 0],
        )
        target_object_relative0 = transform_object_points(
            target_object_base[sample_idx],
            target_object_transforms[sample_idx, 0],
        )
        pred_human_relative0 = pred_human_base[sample_idx] + pred_human_translation[sample_idx, 0].view(1, 3)
        target_human_relative0 = target_human_base[sample_idx] + target_human_translation[sample_idx, 0].view(1, 3)
        sequence_name = batch.get("sequence_name")
        if isinstance(sequence_name, (list, tuple)):
            sequence_value = sequence_name[sample_idx]
        else:
            sequence_value = sequence_name
        export_pointcloud_debug(
            sample_dir,
            sequence_name=sequence_value,
            pred_human_base=pred_human_base[sample_idx],
            pred_object_base=pred_object_base[sample_idx],
            target_human_base=target_human_base[sample_idx],
            target_object_base=target_object_base[sample_idx],
            pred_human_relative0=pred_human_relative0,
            pred_object_relative0=pred_object_relative0,
            target_human_relative0=target_human_relative0,
            target_object_relative0=target_object_relative0,
        )
        for frame_idx in range(num_frames):
            pred_human = pred_human_base[sample_idx] + pred_human_translation[sample_idx, frame_idx].view(1, 3)
            target_human = target_human_base[sample_idx] + target_human_translation[sample_idx, frame_idx].view(1, 3)
            pred_object = transform_object_points(
                pred_object_base[sample_idx],
                pred_object_transforms[sample_idx, frame_idx],
            )
            target_object = transform_object_points(
                target_object_base[sample_idx],
                target_object_transforms[sample_idx, frame_idx],
            )
            panel = make_render_panel(
                rgb_frame=rgb[sample_idx, frame_idx],
                intrinsics=intrinsics[sample_idx, frame_idx],
                pred_human=pred_human,
                pred_object=pred_object,
                target_human=target_human,
                target_object=target_object,
                radius=int(radius),
                max_points=int(max_points),
            )
            frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
            panel.save(frame_path)
            panels.append(panel)
        if panels:
            panels[0].save(
                sample_dir / "animation.gif",
                save_all=True,
                append_images=panels[1:],
                duration=180,
                loop=0,
            )


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


def _to_wandb_video(frames: Tensor, *, fps: int):
    frames = frames.detach().float().cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}.")
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    elif frames.shape[1] > 3:
        frames = frames[:, :3]
    frames = frames.clamp(0.0, 1.0)
    video = (frames * 255.0).round().to(torch.uint8).numpy()
    import wandb

    return wandb.Video(video, fps=int(fps), format="mp4")


def _relative_reconstruction_point_groups(
    *,
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    sample_index: int,
    frame_index: int,
) -> Tuple[
    Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
]:
    pred_human_base = decoded.human_gaussians[sample_index, ..., :3]
    pred_object_base = decoded.object_gaussians[sample_index, ..., :3]
    target_human_base = batch["human_gaussians"][sample_index, ..., :3]
    target_object_base = batch["object_gaussians"][sample_index, ..., :3]

    pred_human = pred_human_base + decoded.human_translation[sample_index, frame_index].view(1, 3)
    target_human = target_human_base + batch["cam_t"][sample_index, frame_index].view(1, 3)
    pred_object = transform_object_points(
        pred_object_base,
        decoded.object_transforms[sample_index, frame_index],
    )
    target_object = transform_object_points(
        target_object_base,
        batch["object_poses"][sample_index, frame_index],
    )
    pred_groups = (
        (pred_human, (45, 130, 255)),
        (pred_object, (255, 140, 30)),
    )
    target_groups = (
        (target_human, (39, 174, 96)),
        (target_object, (155, 89, 182)),
    )
    return pred_groups, target_groups


def _collect_colored_points(
    point_groups: Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    *,
    max_points: int,
) -> Optional[Tuple[Tensor, Tensor]]:
    points_all: List[Tensor] = []
    colors_all: List[Tensor] = []
    for points, color in point_groups:
        points = points.detach().float().reshape(-1, points.shape[-1])[..., :3]
        points = select_render_points(points, max_points=max_points).cpu()
        valid = torch.isfinite(points).all(dim=-1)
        points = points[valid]
        if points.numel() == 0:
            continue
        colors = points.new_tensor(color, dtype=torch.float32).view(1, 3).expand(points.shape[0], -1)
        points_all.append(points)
        colors_all.append(colors)
    if not points_all:
        return None
    return torch.cat(points_all, dim=0), torch.cat(colors_all, dim=0)


def _point_groups_to_wandb_object3d(
    point_groups: Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    *,
    max_points: int,
):
    packed = _collect_colored_points(point_groups, max_points=max_points)
    if packed is None:
        return None
    points, colors = packed
    cloud = torch.cat([points, colors.clamp(0.0, 255.0).round()], dim=-1).numpy()
    import wandb

    return wandb.Object3D(cloud)


def _wandb_reconstruction_3d_table(
    *,
    pred_groups: Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    target_groups: Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    frame_index: int,
    max_points: int,
):
    pred_object = _point_groups_to_wandb_object3d(pred_groups, max_points=max_points)
    target_object = _point_groups_to_wandb_object3d(target_groups, max_points=max_points)
    if pred_object is None and target_object is None:
        return None
    import wandb

    table = wandb.Table(columns=["frame", "prediction", "ground_truth"])
    table.add_data(int(frame_index), pred_object, target_object)
    return table


def render_rotating_pointcloud_video(
    point_groups: Tuple[Tuple[Tensor, Tuple[int, int, int]], ...],
    *,
    max_points: int,
    num_frames: int,
    image_size: int,
    point_radius: int,
) -> Tensor:
    packed = _collect_colored_points(point_groups, max_points=max_points)
    num_frames = max(int(num_frames), 1)
    image_size = max(int(image_size), 32)
    point_radius = max(int(point_radius), 1)
    if packed is None:
        return torch.ones(num_frames, 3, image_size, image_size)

    points, colors = packed
    bbox_min = points.min(dim=0).values
    bbox_max = points.max(dim=0).values
    center = (bbox_min + bbox_max) * 0.5
    extent = (bbox_max - bbox_min).max().clamp_min(1e-6)
    scale = float(image_size) * 0.72 / float(extent)
    centered = points - center
    pitch = math.radians(18.0)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    rotation_x = centered.new_tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ]
    )

    frames = []
    for frame_idx in range(num_frames):
        yaw = 2.0 * math.pi * float(frame_idx) / float(num_frames)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation_y = centered.new_tensor(
            [
                [cos_yaw, 0.0, sin_yaw],
                [0.0, 1.0, 0.0],
                [-sin_yaw, 0.0, cos_yaw],
            ]
        )
        view_points = centered @ rotation_y.T @ rotation_x.T
        px = view_points[:, 0] * scale + float(image_size - 1) * 0.5
        py = -view_points[:, 1] * scale + float(image_size - 1) * 0.5
        valid = (
            torch.isfinite(view_points).all(dim=-1)
            & (px >= 0)
            & (px < image_size)
            & (py >= 0)
            & (py < image_size)
        )
        draw_order = torch.argsort(view_points[:, 2])
        image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image, mode="RGBA")
        for point_idx in draw_order.tolist():
            if not bool(valid[point_idx]):
                continue
            cx = float(px[point_idx])
            cy = float(py[point_idx])
            red, green, blue = colors[point_idx].clamp(0.0, 255.0).round().int().tolist()
            draw.ellipse(
                (
                    cx - point_radius,
                    cy - point_radius,
                    cx + point_radius,
                    cy + point_radius,
                ),
                fill=(int(red), int(green), int(blue), 220),
                outline=(0, 0, 0, 35),
            )
        frames.append(np.asarray(image, dtype=np.uint8))

    video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    return video


def log_wandb_visuals(
    *,
    batch: Dict[str, object],
    output: UniModelOutput,
    step: int,
    args: argparse.Namespace,
    prefix: str = "train_visual",
) -> None:
    if args.log_with != "wandb":
        return
    try:
        import wandb
    except Exception as exc:
        print(f"wandb visual skipped: {exc}", flush=True)
        return
    if wandb.run is None:
        return

    sample_index = 0
    max_frames = int(args.train_visual_max_frames)
    max_points = int(args.train_visual_max_points)
    payload = {
        f"{prefix}/rgb": _to_wandb_image_grid(batch["rgb"][sample_index], max_frames=max_frames),
    }
    for key, log_name, normalize in (
        ("masks_human", "human_mask", False),
        ("masks_object", "object_mask", False),
        ("depth", "depth", True),
    ):
        if key in batch:
            payload[f"{prefix}/{log_name}"] = _to_wandb_image_grid(
                batch[key][sample_index],
                max_frames=max_frames,
                normalize=normalize,
            )

    num_frames = int(batch["rgb"].shape[1])
    frame_index = min(max(int(args.train_visual_3d_frame_index), 0), max(num_frames - 1, 0))
    pred_groups, target_groups = _relative_reconstruction_point_groups(
        decoded=output.decoded_state,
        batch=batch,
        sample_index=sample_index,
        frame_index=frame_index,
    )

    try:
        render_panel = make_render_panel(
            rgb_frame=batch["rgb"][sample_index, frame_index],
            intrinsics=batch["camera_intrinsics"][sample_index, frame_index],
            pred_human=pred_groups[0][0],
            pred_object=pred_groups[1][0],
            target_human=target_groups[0][0],
            target_object=target_groups[1][0],
            radius=int(args.render_point_radius),
            max_points=max_points,
        )
        payload[f"{prefix}/render_panel"] = wandb.Image(np.asarray(render_panel))
    except Exception as exc:
        print(f"wandb render panel skipped: {exc}", flush=True)

    try:
        comparison_table = _wandb_reconstruction_3d_table(
            pred_groups=pred_groups,
            target_groups=target_groups,
            frame_index=frame_index,
            max_points=max_points,
        )
        if comparison_table is not None:
            payload[f"{prefix}/reconstruction_3d"] = comparison_table
    except Exception as exc:
        print(f"wandb Object3D skipped: {exc}", flush=True)

    try:
        orbit_video = render_rotating_pointcloud_video(
            pred_groups,
            max_points=max_points,
            num_frames=int(args.train_visual_orbit_frames),
            image_size=int(args.train_visual_orbit_size),
            point_radius=int(args.render_point_radius),
        )
        payload[f"{prefix}/reconstruction_orbit"] = _to_wandb_video(
            orbit_video,
            fps=int(args.train_visual_video_fps),
        )
    except Exception as exc:
        print(f"wandb rotating 3D render skipped: {exc}", flush=True)

    wandb.log(payload, step=step)


def save_prediction(
    path: Path,
    *,
    output: UniModelOutput,
    batch: Dict[str, object],
    render_dir: Optional[Path] = None,
    render_point_radius: int = 3,
    render_max_points: int = 256,
) -> None:
    decoded = output.decoded_state
    payload = {
        "sequence_name": batch.get("sequence_name"),
        "object_category": batch.get("object_category"),
        "input": {
            "rgb_uint8": batch["rgb"].detach().float().clamp(0.0, 1.0).mul(255.0).round().byte().cpu(),
            "camera_intrinsics": batch["camera_intrinsics"].detach().cpu(),
        },
        "decoded": {
            "human_shape": decoded.human_shape.detach().cpu(),
            "human_pose": decoded.human_pose.detach().cpu(),
            "human_translation": decoded.human_translation.detach().cpu(),
            "human_gaussians": decoded.human_gaussians.detach().cpu(),
            "object_gaussians": decoded.object_gaussians.detach().cpu(),
            "joints_3d": decoded.joints_3d.detach().cpu(),
            "object_transforms": decoded.object_transforms.detach().cpu(),
            "contact_signature": decoded.contact_signature.detach().cpu(),
        },
        "target": {
            "human_shape": batch["human_shape"].detach().cpu(),
            "body_pose": batch["body_pose"].detach().cpu(),
            "cam_t": batch["cam_t"].detach().cpu(),
            "human_gaussians": batch["human_gaussians"].detach().cpu(),
            "object_gaussians": batch["object_gaussians"].detach().cpu(),
            "joints_3d": batch["joints_3d"].detach().cpu(),
            "object_poses": batch["object_poses"].detach().cpu(),
            "contact_signature": batch["contact_signature"].detach().cpu(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    if render_dir is not None:
        render_prediction_batch(
            render_dir,
            output=output,
            batch=batch,
            radius=render_point_radius,
            max_points=render_max_points,
        )


def average_eval_metrics(
    *,
    totals: Dict[str, float],
    metric_values: Dict[str, Tensor],
    batch_size: int,
    accelerator: Accelerator,
) -> int:
    weight = torch.tensor([float(batch_size)], device=accelerator.device)
    gathered_weight = accelerator.gather_for_metrics(weight).sum().item()
    for key, value in metric_values.items():
        weighted_value = value.detach().float().reshape(1) * float(batch_size)
        gathered_value = accelerator.gather_for_metrics(weighted_value).sum().item()
        totals[key] = totals.get(key, 0.0) + gathered_value
    return int(gathered_weight)


@torch.no_grad()
def run_periodic_test(
    *,
    model,
    codec_model: UniModel,
    dataloader: DataLoader,
    weights: Dict[str, float],
    args: argparse.Namespace,
    accelerator: Accelerator,
    step: int,
) -> Dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    saved_batches = 0
    max_batches = int(args.periodic_test_max_batches)
    if max_batches < 0:
        max_batches = int(args.test_max_batches)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = prepare_batch(
            batch,
            device=accelerator.device,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        _, loss_metrics, output = forward_and_loss(
            model=model,
            codec_model=codec_model,
            batch=batch,
            args=args,
            weights=weights,
            generator=None,
        )
        metrics = dict(loss_metrics)
        metrics.update(compute_test_eval_metrics(output, batch, args))
        count += average_eval_metrics(
            totals=totals,
            metric_values=metrics,
            batch_size=int(batch["rgb"].shape[0]),
            accelerator=accelerator,
        )

        if accelerator.is_main_process and args.save_test_predictions and saved_batches < args.test_save_batches:
            save_prediction(
                Path(args.output_dir) / "test_predictions" / f"step_{step:07d}" / f"batch_{batch_idx:05d}.pt",
                output=output,
                batch=batch,
                render_dir=(
                    Path(args.output_dir) / "test_renders" / f"step_{step:07d}" / f"batch_{batch_idx:05d}"
                    if args.render_test_predictions
                    else None
                ),
                render_point_radius=args.render_point_radius,
                render_max_points=args.render_max_points,
            )
            saved_batches += 1

        if (
            accelerator.is_main_process
            and args.test_visual_upload
            and batch_idx < int(args.test_visual_upload_batches)
        ):
            log_wandb_visuals(
                batch=batch,
                output=output,
                step=step,
                args=args,
                prefix="test_visual",
            )

    averages = {key: value / max(count, 1) for key, value in totals.items()}
    if accelerator.is_main_process:
        write_json(Path(args.output_dir) / f"test_metrics_step_{step:07d}.json", averages)
        if args.log_with != "none":
            accelerator.log({f"test/{key}": value for key, value in averages.items()}, step=step)
        print(f"periodic test step={step:07d} " + json.dumps(averages, indent=2), flush=True)
    accelerator.wait_for_everyone()
    if was_training:
        model.train()
    return averages


def transform_object_points_sequence(points: Tensor, transforms: Tensor) -> Tensor:
    rotation = transforms[..., :3, :3].float()
    translation = transforms[..., :3, 3].float()
    return torch.einsum("btij,bnj->btni", rotation, points.float()) + translation.unsqueeze(-2)


def _select_eval_points(points: Tensor, max_points: int) -> Tensor:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, steps=max_points, device=points.device).round().long()
    return points.index_select(0, indices)


def _mean_min_distance_chunked(source: Tensor, target: Tensor, *, chunk_size: int) -> Tensor:
    chunk_size = max(int(chunk_size), 1)
    total = source.new_zeros(())
    count = 0
    for start in range(0, int(source.shape[0]), chunk_size):
        chunk = source[start : start + chunk_size]
        dists = torch.cdist(chunk.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
        total = total + dists.min(dim=1).values.sum()
        count += int(chunk.shape[0])
    return total / float(max(count, 1))


def chamfer_distance_points(pred: Tensor, target: Tensor, *, max_points: int, chunk_size: int) -> Tensor:
    pred = pred.detach().float().reshape(-1, pred.shape[-1])[..., :3]
    target = target.detach().float().reshape(-1, target.shape[-1])[..., :3]
    pred = pred[torch.isfinite(pred).all(dim=-1)]
    target = target[torch.isfinite(target).all(dim=-1)]
    pred = _select_eval_points(pred, max_points=max_points)
    target = _select_eval_points(target, max_points=max_points)
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_zeros(())
    pred_to_target = _mean_min_distance_chunked(pred, target, chunk_size=chunk_size)
    target_to_pred = _mean_min_distance_chunked(target, pred, chunk_size=chunk_size)
    return pred_to_target + target_to_pred


def chamfer_distance_sequence(pred: Tensor, target: Tensor, *, max_points: int, chunk_size: int) -> Tensor:
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"Expected point sequences with shape [B, T, N, 3], got {tuple(pred.shape)} and {tuple(target.shape)}."
        )
    batch_size = min(int(pred.shape[0]), int(target.shape[0]))
    num_frames = min(int(pred.shape[1]), int(target.shape[1]))
    values = []
    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            values.append(
                chamfer_distance_points(
                    pred[batch_idx, frame_idx],
                    target[batch_idx, frame_idx],
                    max_points=max_points,
                    chunk_size=chunk_size,
                )
            )
    if not values:
        return pred.new_zeros(())
    return torch.stack(values).mean()


def _smpl_vertices_to_eval_space(vertices: Tensor, joints_3d: Tensor, cam_t: Tensor) -> Tensor:
    if vertices.shape[-2] == 0:
        return vertices
    raw_center = vertices.float().mean(dim=-2)
    joints_center = joints_3d.float().mean(dim=-2)
    raw_distance = torch.linalg.norm(raw_center - joints_center, dim=-1).mean()
    translated_vertices = vertices.float() + cam_t.float().unsqueeze(-2)
    translated_center = translated_vertices.mean(dim=-2)
    translated_distance = torch.linalg.norm(translated_center - joints_center, dim=-1).mean()
    if bool(translated_distance < raw_distance):
        return translated_vertices
    return vertices.float()


def build_eval_point_sequences(decoded: DecodedHOIState, batch: Dict[str, object]) -> Dict[str, Tensor]:
    pred_human = decoded.human_gaussians[..., :3].float().unsqueeze(1) + decoded.human_translation.float().unsqueeze(-2)
    human_vertices = batch.get("human_vertices")
    if isinstance(human_vertices, Tensor) and human_vertices.shape[-2] > 0:
        target_human = _smpl_vertices_to_eval_space(human_vertices, batch["joints_3d"], batch["cam_t"])
    else:
        target_human = batch["human_gaussians"][..., :3].float().unsqueeze(1) + batch["cam_t"].float().unsqueeze(-2)

    pred_object = transform_object_points_sequence(
        decoded.object_gaussians[..., :3],
        decoded.object_transforms,
    )
    target_object = transform_object_points_sequence(
        batch["object_gaussians"][..., :3],
        batch["object_poses"],
    )
    return {
        "pred_human": pred_human,
        "target_human": target_human,
        "pred_object": pred_object,
        "target_object": target_object,
        "pred_combined": torch.cat([pred_human, pred_object], dim=-2),
        "target_combined": torch.cat([target_human, target_object], dim=-2),
    }


def acceleration_error(pred: Tensor, target: Tensor) -> Optional[Tensor]:
    if pred.shape[1] < 3 or target.shape[1] < 3:
        return None
    pred_acc = pred[:, 2:].float() - 2.0 * pred[:, 1:-1].float() + pred[:, :-2].float()
    target_acc = target[:, 2:].float() - 2.0 * target[:, 1:-1].float() + target[:, :-2].float()
    return torch.linalg.norm(pred_acc - target_acc, dim=-1).mean()


def compute_test_eval_metrics(
    output: UniModelOutput,
    batch: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, Tensor]:
    decoded = output.decoded_state
    max_points = int(args.test_eval_max_points)
    chunk_size = int(args.test_eval_chamfer_chunk_size)
    point_sequences = build_eval_point_sequences(decoded, batch)

    metrics: Dict[str, Tensor] = {
        "CD-h": chamfer_distance_sequence(
            point_sequences["pred_human"],
            point_sequences["target_human"],
            max_points=max_points,
            chunk_size=chunk_size,
        ),
        "CD-o": chamfer_distance_sequence(
            point_sequences["pred_object"],
            point_sequences["target_object"],
            max_points=max_points,
            chunk_size=chunk_size,
        ),
        "CD-c": chamfer_distance_sequence(
            point_sequences["pred_combined"],
            point_sequences["target_combined"],
            max_points=max_points,
            chunk_size=chunk_size,
        ),
    }
    if args.test_eval_acceleration:
        human_acc = acceleration_error(decoded.joints_3d, batch["joints_3d"])
        object_acc = acceleration_error(decoded.object_transforms[..., :3, 3], batch["object_poses"][..., :3, 3])
        if human_acc is not None:
            metrics["Acc-h"] = human_acc
        if object_acc is not None:
            metrics["Acc-o"] = object_acc
    return metrics


def train(args: argparse.Namespace, accelerator: Accelerator) -> None:
    dataset = build_dataset(args)
    dataloader = build_dataloader(args, dataset, train=True)
    eval_dataloader = None
    if args.test_every > 0:
        eval_dataset = build_test_dataset(args, include_human_vertices=args.test_eval_use_smpl_vertices)
        eval_dataloader = build_dataloader(args, eval_dataset, train=False)
    min_required_samples = int(args.batch_size) * int(accelerator.num_processes)
    if len(dataset) < min_required_samples:
        raise ValueError(
            "Training dataset is too small for the active distributed setup: "
            f"samples={len(dataset)}, batch_size={args.batch_size}, "
            f"num_processes={accelerator.num_processes}, required_at_least={min_required_samples}. "
            "Increase --max_sequences, reduce CUDA_VISIBLE_DEVICES/NUM_PROCESSES, or lower --batch_size."
        )
    model = build_model(args)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        scheduler_type=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio,
    )

    global_step = load_checkpoint(
        model=model,
        checkpoint_path=args.resume_checkpoint or args.init_checkpoint,
        optimizer=optimizer if args.resume_checkpoint else None,
        scheduler=scheduler if args.resume_checkpoint else None,
    )
    ensure_vae_loaded_rank_by_rank(model, accelerator, args)
    if accelerator.is_main_process:
        print_model_parameter_summary(model)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    raw_model: UniModel = accelerator.unwrap_model(model)
    weights = supervised_weights_from_args(args)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        global_batch = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        steps_per_epoch = math.ceil(len(dataset) / max(global_batch, 1))
        effective_epochs = float(args.max_steps * global_batch) / float(max(len(dataset), 1))
        print(
            "dataset "
            f"| sequences={len(dataset.sequence_dirs)} | samples={len(dataset)} "
            f"| global_batch={global_batch} | steps_per_epoch~={steps_per_epoch} "
            f"| effective_epochs~={effective_epochs:.2f}",
            flush=True,
        )
        if eval_dataloader is not None:
            print(
                "eval dataset "
                f"| data_root={resolve_test_data_root(args)} "
                f"| sequences={len(eval_dataset.sequence_dirs)} | samples={len(eval_dataset)} "
                f"| periodic_max_batches={args.periodic_test_max_batches}",
                flush=True,
            )

    model.train()
    progress = tqdm(
        total=args.max_steps,
        initial=global_step,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
    while global_step < args.max_steps:
        saw_batch = False
        for batch in dataloader:
            saw_batch = True
            if global_step >= args.max_steps:
                break
            batch = prepare_batch(
                batch,
                device=accelerator.device,
                image_height=args.image_height,
                image_width=args.image_width,
            )
            with accelerator.accumulate(model):
                loss, metrics, output = forward_and_loss(
                    model=model,
                    codec_model=raw_model,
                    batch=batch,
                    args=args,
                    weights=weights,
                    generator=generator,
                )
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
                    scalar_metrics = tensor_metrics_to_float(metrics)
                    scalar_metrics["lr"] = float(scheduler.get_last_lr()[0])
                    progress.set_postfix(
                        loss=f"{scalar_metrics['loss']:.5f}",
                        lr=f"{scalar_metrics['lr']:.2e}",
                        refresh=True,
                    )
                    if args.log_with != "none":
                        accelerator.log(scalar_metrics, step=global_step)

                if (
                    accelerator.is_main_process
                    and args.train_visual_every > 0
                    and global_step % args.train_visual_every == 0
                ):
                    log_wandb_visuals(
                        batch=batch,
                        output=output,
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

                if eval_dataloader is not None and global_step % args.test_every == 0:
                    run_periodic_test(
                        model=model,
                        codec_model=raw_model,
                        dataloader=eval_dataloader,
                        weights=weights,
                        args=args,
                        accelerator=accelerator,
                        step=global_step,
                    )
        if not saw_batch:
            raise RuntimeError(
                "Training dataloader produced no batches. "
                "This usually means the dataset is too small for the number of GPUs with drop_last=True."
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
        print(f"train done | steps={global_step:07d} | output_dir={args.output_dir}", flush=True)
    accelerator.wait_for_everyone()


@torch.no_grad()
def test(args: argparse.Namespace, accelerator: Accelerator) -> Dict[str, float]:
    dataset = build_test_dataset(args, include_human_vertices=args.test_eval_use_smpl_vertices)
    dataloader = build_dataloader(args, dataset, train=False)
    model = build_model(args)
    load_checkpoint(model=model, checkpoint_path=args.test_checkpoint or args.init_checkpoint)
    ensure_vae_loaded_rank_by_rank(model, accelerator, args)

    model, dataloader = accelerator.prepare(model, dataloader)
    raw_model: UniModel = accelerator.unwrap_model(model)
    weights = supervised_weights_from_args(args)
    model.eval()

    totals: Dict[str, float] = {}
    count = 0
    saved_batches = 0
    max_batches = int(args.test_max_batches)
    iterator: Iterable = dataloader
    progress = tqdm(iterator, disable=not accelerator.is_main_process, dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = prepare_batch(
            batch,
            device=accelerator.device,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        _, loss_metrics, output = forward_and_loss(
            model=model,
            codec_model=raw_model,
            batch=batch,
            args=args,
            weights=weights,
            generator=None,
        )
        metrics = dict(loss_metrics)
        metrics.update(compute_test_eval_metrics(output, batch, args))
        count += average_eval_metrics(
            totals=totals,
            metric_values=metrics,
            batch_size=int(batch["rgb"].shape[0]),
            accelerator=accelerator,
        )
        if accelerator.is_main_process and args.save_test_predictions and saved_batches < args.test_save_batches:
            save_prediction(
                Path(args.output_dir) / "test_predictions" / f"batch_{batch_idx:05d}.pt",
                output=output,
                batch=batch,
                render_dir=(
                    Path(args.output_dir) / "test_renders" / f"batch_{batch_idx:05d}"
                    if args.render_test_predictions
                    else None
                ),
                render_point_radius=args.render_point_radius,
                render_max_points=args.render_max_points,
            )
            saved_batches += 1

        if (
            accelerator.is_main_process
            and args.test_visual_upload
            and batch_idx < int(args.test_visual_upload_batches)
        ):
            log_wandb_visuals(
                batch=batch,
                output=output,
                step=batch_idx,
                args=args,
                prefix="test_visual",
            )

    accelerator.wait_for_everyone()
    averages = {key: value / max(count, 1) for key, value in totals.items()}
    if accelerator.is_main_process:
        write_json(Path(args.output_dir) / "test_metrics.json", averages)
        if args.log_with != "none":
            accelerator.log({f"test/{key}": value for key, value in averages.items()})
        print("test metrics " + json.dumps(averages, indent=2), flush=True)
    return averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test the experimental UniModel Wan-VAE + DiT baseline.")
    parser.add_argument("--mode", type=str, default="train", choices=("train", "test", "train_test"))

    parser.add_argument("--data_root", type=str, default="sample_data/WAI_prepared/sequences")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--human_gaussian_source", type=str, default="smpl_mesh", choices=("smpl_mesh", "teacher"))
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument(
        "--test_data_root",
        type=str,
        default="",
        help="Optional independent dataset root for periodic/final evaluation. Defaults to --data_root.",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default="",
        help="Optional split file for evaluation. Defaults to --split_file.",
    )
    parser.add_argument(
        "--test_split_key",
        type=str,
        default="",
        help="Optional split key for evaluation. Defaults to --split_key.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/unimodel")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--test_checkpoint", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--log_with", type=str, default="none", choices=("none", "tensorboard", "wandb"))
    parser.add_argument("--project_name", type=str, default="uni-hoi-4d")
    parser.add_argument("--run_name", type=str, default="unimodel_wan_vae_dit")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--train_visual_every", type=int, default=1000)
    parser.add_argument("--train_visual_max_frames", type=int, default=1)
    parser.add_argument("--train_visual_max_points", type=int, default=4096)
    parser.add_argument("--train_visual_3d_frame_index", type=int, default=0)
    parser.add_argument("--train_visual_orbit_frames", type=int, default=24)
    parser.add_argument("--train_visual_orbit_size", type=int, default=384)
    parser.add_argument("--train_visual_video_fps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=("constant", "cosine", "linear"))
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--clip_length", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--coordinate_mode", type=str, default="relative", choices=("relative", "absolute"))
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--dataset_cache_sequences", type=int, default=2)
    parser.add_argument("--cache_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb_cache_max_frames", type=int, default=256)
    parser.add_argument("--prefer_h5_cache", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--latent_patch_size", type=int, nargs=3, default=(1, 2, 2))

    parser.add_argument("--wan_vae_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--wan_vae_subfolder", type=str, default="vae")
    parser.add_argument("--wan_vae_dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--wan_vae_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae_latent_channels", type=int, default=48)
    parser.add_argument("--vae_scale_factor_temporal", type=int, default=4)
    parser.add_argument("--vae_scale_factor_spatial", type=int, default=16)
    parser.add_argument("--debug_fake_vae_latents", action="store_true")

    parser.add_argument("--num_human_gaussians", type=int, default=256)
    parser.add_argument("--num_object_gaussians", type=int, default=256)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--human_shape_dim", type=int, default=10)
    parser.add_argument("--human_pose_dim", type=int, default=72)

    parser.add_argument("--use_state_fm", action="store_true")
    parser.add_argument("--lambda_state_fm", type=float, default=1.0)
    parser.add_argument("--lambda_shape", type=float, default=0.1)
    parser.add_argument("--lambda_pose", type=float, default=0.5)
    parser.add_argument("--lambda_translation", type=float, default=0.5)
    parser.add_argument("--lambda_object_pose", type=float, default=0.5)
    parser.add_argument("--lambda_contact", type=float, default=0.1)
    parser.add_argument("--lambda_joints", type=float, default=1.0)
    parser.add_argument("--lambda_human_gaussian", type=float, default=1.0)
    parser.add_argument("--lambda_object_gaussian", type=float, default=1.0)
    parser.add_argument("--lambda_gaussian_chamfer", type=float, default=1.0)
    parser.add_argument("--lambda_gaussian_xyz_l1", type=float, default=0.05)
    parser.add_argument("--lambda_gaussian_attr_l1", type=float, default=0.1)

    parser.add_argument("--test_max_batches", type=int, default=0)
    parser.add_argument("--test_every", type=int, default=0)
    parser.add_argument(
        "--periodic_test_max_batches",
        type=int,
        default=-1,
        help="Batch cap for periodic evaluation during training. -1 reuses --test_max_batches; 0 means full eval.",
    )
    parser.add_argument("--test_eval_max_points", type=int, default=4096)
    parser.add_argument("--test_eval_chamfer_chunk_size", type=int, default=2048)
    parser.add_argument("--test_eval_use_smpl_vertices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_eval_acceleration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_visual_upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_visual_upload_batches", type=int, default=1)
    parser.add_argument("--save_test_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_save_batches", type=int, default=4)
    parser.add_argument("--render_test_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_point_radius", type=int, default=3)
    parser.add_argument("--render_max_points", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_length != 1:
        raise ValueError(f"This entrypoint is single-image only; set --clip_length 1, got {args.clip_length}.")
    if (args.clip_length - 1) % args.vae_scale_factor_temporal != 0:
        raise ValueError(
            "Wan VAE clip length should satisfy "
            f"(clip_length - 1) % vae_scale_factor_temporal == 0, got {args.clip_length}."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.log_with == "none" else args.log_with,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.device.type == "cuda":
        device_index = accelerator.device.index
        if device_index is None:
            device_index = int(accelerator.local_process_index)
        torch.cuda.set_device(device_index)
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
        write_json(Path(args.output_dir) / "launch_config.json", vars(args))
        print(
            "starting "
            f"| mode={args.mode} | data_root={args.data_root} | output_dir={args.output_dir} "
            f"| test_data_root={resolve_test_data_root(args)} "
            f"| input=single_image | coordinate_mode={args.coordinate_mode} "
            f"| fake_vae={args.debug_fake_vae_latents} "
            f"| distributed={accelerator.distributed_type} | processes={accelerator.num_processes} "
            f"| cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            flush=True,
        )
    print(
        "rank map "
        f"| rank={accelerator.process_index}/{accelerator.num_processes} "
        f"| local_rank={accelerator.local_process_index} | device={accelerator.device}",
        flush=True,
    )

    if args.mode in {"train", "train_test"}:
        train(args, accelerator)
    if args.mode in {"test", "train_test"}:
        if args.mode == "train_test" and not args.test_checkpoint:
            args.test_checkpoint = str(Path(args.output_dir) / "checkpoints" / "last.pt")
        test(args, accelerator)

    accelerator.end_training()


if __name__ == "__main__":
    main()
