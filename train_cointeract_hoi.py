#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from model.comovi_hoi_rgb_wan import CoMoViHOIRGBModel


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


def stage1_full_attention_steps(args: argparse.Namespace) -> int:
    return max(int(getattr(args, "stage1_full_attention_steps", 0)), 0)


def stage1_hoi_to_rgb_scale(args: argparse.Namespace) -> float:
    return float(getattr(args, "stage1_hoi_to_rgb_scale", 1.0))


def hoi_to_rgb_scale_for_step(args: argparse.Namespace, step: int) -> float:
    if int(step) < stage1_full_attention_steps(args):
        return stage1_hoi_to_rgb_scale(args)
    return float(getattr(args, "hoi_to_rgb_scale", 0.0))


def coattention_stage_for_step(args: argparse.Namespace, step: int) -> int:
    return 1 if int(step) < stage1_full_attention_steps(args) else 2


def should_enable_hoi_to_rgb(args: argparse.Namespace) -> bool:
    if abs(float(getattr(args, "hoi_to_rgb_scale", 0.0))) > 0.0:
        return True
    return stage1_full_attention_steps(args) > 0 and abs(stage1_hoi_to_rgb_scale(args)) > 0.0


def zero_loss_anchor_from_output(output) -> Tensor:
    anchor = output.state_velocity.new_zeros(())
    for attr_name in (
        "rgb_velocity",
        "rgb_hidden_tokens",
        "rgb_context_tokens",
        "rgb_prior_tokens",
        "hoi_tokens",
        "predicted_clean_state",
    ):
        tensor = getattr(output, attr_name, None)
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            anchor = anchor + tensor.float().sum() * 0.0
    return anchor


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
    common_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "num_frames": args.clip_length,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "image_patch_size": args.image_patch_size,
        "num_human_gaussians": args.num_human_gaussians,
        "num_object_gaussians": args.num_object_gaussians,
        "num_joints": args.num_joints,
        "contact_dim": args.contact_dim,
        "human_shape_dim": args.human_shape_dim,
        "human_pose_dim": args.human_pose_dim,
        "wan_model_id": args.wan_model_id,
        "wan_dtype": args.wan_dtype,
        "wan_hidden_dim": args.wan_hidden_dim,
        "wan_prompt_max_sequence_length": args.wan_prompt_max_sequence_length,
        "wan_local_files_only": args.wan_local_files_only,
        "freeze_wan": args.freeze_wan,
        "detach_rgb_context": args.detach_rgb_context,
        "enable_hoi_to_rgb": should_enable_hoi_to_rgb(args),
    }
    if args.model_variant == "comovi":
        return CoMoViHOIRGBModel(
            **common_kwargs,
            visual_prior_num_global_tokens=args.visual_prior_num_global_tokens,
            visual_resampler_depth=args.visual_resampler_depth,
            cross_3d2d_depth=args.cross_3d2d_depth,
        )
    return CoInteractHOI4DModel(**common_kwargs)


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


def _parse_contact_joint_indices(value: str, num_joints: int) -> List[int]:
    if value.strip():
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif num_joints >= 52:
        indices = list(range(20, 52))
    elif num_joints >= 22:
        indices = [20, 21]
    else:
        indices = list(range(max(0, num_joints - 2), num_joints))
    if not indices:
        indices = [max(0, num_joints - 1)]
    return [min(max(idx, 0), num_joints - 1) for idx in indices]


def _transform_object_points(object_xyz: Tensor, object_transforms: Tensor) -> Tensor:
    rotation = object_transforms[..., :3, :3].float()
    translation = object_transforms[..., :3, 3].float()
    return torch.einsum("btij,bnj->btni", rotation, object_xyz.float()) + translation.unsqueeze(2)


def _even_indices(length: int, max_count: int, device: torch.device) -> Tensor:
    if max_count <= 0 or length <= max_count:
        return torch.arange(length, device=device, dtype=torch.long)
    return torch.linspace(0, length - 1, steps=max_count, device=device).round().long()


def _contact_loss_with_surface_samples(
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    *,
    contact_joint_indices: Sequence[int],
) -> Tensor:
    if not contact_joint_indices:
        return decoded.joints_3d.new_zeros(())
    object_relative = _transform_object_points(decoded.object_gaussians[..., :3], decoded.object_transforms)
    joints = decoded.joints_3d[:, :, list(contact_joint_indices)].float()
    batch_size, num_frames, num_contact_joints = joints.shape[:3]
    dists = torch.cdist(
        joints.reshape(batch_size * num_frames, num_contact_joints, 3),
        object_relative.reshape(batch_size * num_frames, object_relative.shape[2], 3),
    ).pow(2)
    min_dists = dists.min(dim=-1).values.reshape(batch_size, num_frames, num_contact_joints)

    contact_signature = batch["contact_signature"].float()
    if contact_signature.shape[-1] >= 4 and num_contact_joints >= 2:
        left_count = max(1, num_contact_joints // 2)
        right_count = num_contact_joints - left_count
        left_mask = contact_signature[..., 2:3].expand(-1, -1, left_count)
        right_mask = contact_signature[..., 3:4].expand(-1, -1, max(right_count, 1))
        mask = torch.cat([left_mask, right_mask[..., :right_count]], dim=-1)
        return (min_dists * mask).sum() / mask.sum().clamp_min(1.0)
    return min_dists.mean()


def _query_smpl_signed_distance(points: Tensor, vertices: Tensor, faces: Tensor, *, chunk_size: int = 512) -> Tensor:
    faces = faces.long()
    tri_vertices = vertices[faces]
    face_centers = tri_vertices.mean(dim=1)
    face_normals = F.normalize(
        torch.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0], dim=-1),
        dim=-1,
        eps=1e-6,
    )
    signed_distances: List[Tensor] = []
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        point_chunk = points[start:end]
        nearest_vertex_dist = torch.cdist(point_chunk.unsqueeze(0), vertices.unsqueeze(0)).squeeze(0).min(dim=1).values
        nearest_face = torch.cdist(point_chunk.unsqueeze(0), face_centers.unsqueeze(0)).squeeze(0).argmin(dim=1)
        direction = point_chunk - face_centers[nearest_face]
        sign = torch.where(
            (direction * face_normals[nearest_face]).sum(dim=-1) >= 0.0,
            torch.ones_like(nearest_vertex_dist),
            -torch.ones_like(nearest_vertex_dist),
        )
        signed_distances.append(nearest_vertex_dist * sign)
    return torch.cat(signed_distances, dim=0)


def _smpl_volume_penetration_loss(
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    *,
    max_frames: int,
    max_object_points: int,
) -> Tensor:
    human_vertices = batch.get("human_vertices")
    human_faces = batch.get("human_faces")
    if not isinstance(human_vertices, Tensor) or not isinstance(human_faces, Tensor):
        return decoded.object_gaussians.new_zeros(())
    if human_vertices.shape[-2] == 0 or human_faces.numel() == 0:
        return decoded.object_gaussians.new_zeros(())

    object_relative = _transform_object_points(decoded.object_gaussians[..., :3], decoded.object_transforms)
    frame_indices = _even_indices(object_relative.shape[1], max_frames, object_relative.device)
    point_indices = _even_indices(object_relative.shape[2], max_object_points, object_relative.device)
    object_relative = object_relative.index_select(1, frame_indices).index_select(2, point_indices)
    human_vertices = human_vertices.float().index_select(1, frame_indices)

    losses: List[Tensor] = []
    for batch_idx in range(object_relative.shape[0]):
        faces = human_faces[batch_idx] if human_faces.ndim == 3 else human_faces
        if faces.numel() == 0:
            continue
        for frame_idx in range(object_relative.shape[1]):
            signed_distance = _query_smpl_signed_distance(
                object_relative[batch_idx, frame_idx],
                human_vertices[batch_idx, frame_idx],
                faces,
            )
            losses.append(F.relu(-signed_distance).mean())
    if not losses:
        return decoded.object_gaussians.new_zeros(())
    return torch.stack(losses).mean()


def compute_physics_losses(
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    *,
    args: argparse.Namespace,
) -> Dict[str, Tensor]:
    losses: Dict[str, Tensor] = {}
    if args.lambda_phys_contact > 0.0:
        contact_indices = _parse_contact_joint_indices(args.contact_joint_indices, args.num_joints)
        losses["phys_contact"] = _contact_loss_with_surface_samples(
            decoded,
            batch,
            contact_joint_indices=contact_indices,
        )
    else:
        losses["phys_contact"] = decoded.object_gaussians.new_zeros(())
    if args.lambda_phys_penetration > 0.0:
        losses["phys_penetration"] = _smpl_volume_penetration_loss(
            decoded,
            batch,
            max_frames=args.phys_loss_max_frames,
            max_object_points=args.phys_loss_max_object_points,
        )
    else:
        losses["phys_penetration"] = decoded.object_gaussians.new_zeros(())
    total = decoded.object_gaussians.new_zeros(())
    total = total + float(args.lambda_phys_contact) * losses["phys_contact"]
    total = total + float(args.lambda_phys_penetration) * losses["phys_penetration"]
    losses["physics"] = total
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


def _to_wandb_video(frames: Tensor, *, max_frames: int, fps: int):
    frames = frames.detach().float().cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}.")
    frames = frames[:max_frames]
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    elif frames.shape[1] > 3:
        frames = frames[:, :3]
    frames = frames.clamp(0.0, 1.0)
    video = (frames * 255.0).round().to(torch.uint8).numpy()
    import wandb

    return wandb.Video(video, fps=int(fps), format="mp4")


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


def _gaussian_comparison_table(
    *,
    decoded: DecodedHOIState,
    batch: Dict[str, object],
    sample_index: int,
    max_points: int,
):
    import wandb

    table = wandb.Table(columns=["part", "prediction", "ground_truth"])
    has_rows = False
    for part, pred, target in (
        ("human", decoded.human_gaussians[sample_index], batch["human_gaussians"][sample_index]),
        ("object", decoded.object_gaussians[sample_index], batch["object_gaussians"][sample_index]),
    ):
        pred_object = _gaussians_to_wandb_object3d(pred, max_points=max_points)
        target_object = _gaussians_to_wandb_object3d(target, max_points=max_points)
        if pred_object is not None and target_object is not None:
            table.add_data(part, pred_object, target_object)
            has_rows = True
    return table if has_rows else None


def _wan_video_comparison_table(
    *,
    model: CoInteractHOI4DModel,
    batch: Dict[str, object],
    video_xt: Tensor,
    rgb_velocity: Tensor,
    timesteps: Tensor,
    sample_index: int,
    max_frames: int,
    fps: int,
):
    import wandb

    sample_slice = slice(sample_index, sample_index + 1)
    with torch.no_grad():
        clean_latents = reconstruct_x1(
            video_xt[sample_slice].detach(),
            rgb_velocity[sample_slice].detach(),
            timesteps[sample_slice].detach(),
        )
        pred_video = model.decode_video_latents(clean_latents)[0]
    table = wandb.Table(columns=["input_rgb", "wan_prediction"])
    table.add_data(
        _to_wandb_video(batch["rgb"][sample_index], max_frames=max_frames, fps=fps),
        _to_wandb_video(pred_video, max_frames=max_frames, fps=fps),
    )
    return table


def log_wandb_visuals(
    *,
    model: CoInteractHOI4DModel,
    batch: Dict[str, object],
    decoded: DecodedHOIState,
    video_xt: Tensor,
    rgb_velocity: Tensor,
    timesteps: Tensor,
    step: int,
    args: argparse.Namespace,
) -> None:
    if args.log_with != "wandb" or args.train_visual_every <= 0:
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
        "train_visual/rgb": _to_wandb_image_grid(batch["rgb"][sample_index], max_frames=max_frames),
        "train_visual/human_mask": _to_wandb_image_grid(batch["masks_human"][sample_index], max_frames=max_frames),
        "train_visual/object_mask": _to_wandb_image_grid(batch["masks_object"][sample_index], max_frames=max_frames),
        "train_visual/depth": _to_wandb_image_grid(
            batch["depth"][sample_index],
            max_frames=max_frames,
            normalize=True,
        ),
    }
    comparison_table = _gaussian_comparison_table(
        decoded=decoded,
        batch=batch,
        sample_index=sample_index,
        max_points=max_points,
    )
    if comparison_table is not None:
        payload["train_visual/gaussian_recon_vs_gt"] = comparison_table

    if args.train_visual_log_wan_video:
        try:
            payload["train_visual/wan_video"] = _wan_video_comparison_table(
                model=model,
                batch=batch,
                video_xt=video_xt,
                rgb_velocity=rgb_velocity,
                timesteps=timesteps,
                sample_index=sample_index,
                max_frames=max_frames,
                fps=args.train_visual_video_fps,
            )
        except Exception as exc:
            print(f"wandb Wan video skipped: {exc}", flush=True)
    wandb.log(payload, step=step)


def print_wandb_run_url(args: argparse.Namespace, accelerator: Accelerator) -> None:
    if args.log_with != "wandb" or not accelerator.is_main_process:
        return
    try:
        import wandb

        if wandb.run is None:
            return
        url = wandb.run.get_url()
    except Exception as exc:
        print(f"wandb url unavailable: {exc}", flush=True)
        return
    if url:
        print(f"wandb url: {url}", flush=True)


def ensure_wan_loaded_rank_by_rank(raw_model: CoInteractHOI4DModel, accelerator: Accelerator) -> None:
    for process_index in range(accelerator.num_processes):
        if accelerator.process_index == process_index:
            print(
                f"loading Wan stream on rank {process_index}/{accelerator.num_processes - 1}",
                flush=True,
            )
            raw_model.ensure_wan_loaded(accelerator.device)
            print(f"Wan stream loaded on rank {process_index}", flush=True)
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
            "checkpoint mismatch "
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
            "resume mismatch "
            f"| missing={len(missing)} | unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOI-primary CoInteract-style RGB-guided dual-stream Wan model.")
    parser.add_argument("--model_variant", type=str, default="cointeract", choices=("cointeract", "comovi"))
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
    parser.add_argument("--train_visual_max_frames", type=int, default=1)
    parser.add_argument("--train_visual_max_points", type=int, default=4096)
    parser.add_argument("--train_visual_log_wan_video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_visual_video_fps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=("constant", "cosine", "linear"))
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--clip_length", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--coordinate_mode", type=str, default="relative", choices=("relative", "absolute"))
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
    parser.add_argument("--detach_rgb_context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial_wan_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb_to_hoi_scale", type=float, default=1.0)
    parser.add_argument("--hoi_to_rgb_scale", type=float, default=0.0)
    parser.add_argument("--stage1_full_attention_steps", type=int, default=0)
    parser.add_argument("--stage1_hoi_to_rgb_scale", type=float, default=1.0)
    parser.add_argument("--cross_3d2d_scale", type=float, default=1.0)
    parser.add_argument("--visual_prior_num_global_tokens", type=int, default=8)
    parser.add_argument("--visual_resampler_depth", type=int, default=2)
    parser.add_argument("--cross_3d2d_depth", type=int, default=6)

    parser.add_argument("--num_human_gaussians", type=int, default=850)
    parser.add_argument("--num_object_gaussians", type=int, default=850)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--contact_joint_indices", type=str, default="")
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
    parser.add_argument("--lambda_gaussian_chamfer", type=float, default=1.0)
    parser.add_argument("--lambda_gaussian_xyz_l1", type=float, default=0.05)
    parser.add_argument("--lambda_gaussian_attr_l1", type=float, default=0.1)
    parser.add_argument("--lambda_phys_contact", type=float, default=0.01)
    parser.add_argument("--lambda_phys_penetration", type=float, default=0.01)
    parser.add_argument("--phys_loss_max_frames", type=int, default=1)
    parser.add_argument("--phys_loss_max_object_points", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage1_full_attention_steps < 0:
        raise ValueError("--stage1_full_attention_steps must be non-negative.")
    if args.clip_length != 1:
        raise ValueError(f"This entrypoint is single-image only; set --clip_length 1, got {args.clip_length}.")
    if (args.clip_length - 1) % 4 != 0:
        raise ValueError(f"Wan2.2-TI2V requires clip_length = 4k + 1, got {args.clip_length}.")

    ddp_find_unused_parameters = False
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
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
    print_wandb_run_url(args, accelerator)

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "launch_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(
            "starting "
            f"| data_root={args.data_root} | max_steps={args.max_steps} "
            f"| per_gpu_batch={args.batch_size} "
            f"| global_batch={args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
            f"| model={args.model_variant} | wan={args.wan_model_id} "
            f"| input=single_image | coordinate_mode={args.coordinate_mode} "
            f"| coattention_stage1_steps={stage1_full_attention_steps(args)} "
            f"| stage1_hoi_to_rgb={stage1_hoi_to_rgb_scale(args):.3g} "
            f"| stage2_hoi_to_rgb={float(args.hoi_to_rgb_scale):.3g} "
            f"| ddp_find_unused_parameters={ddp_find_unused_parameters}",
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
        coordinate_mode=args.coordinate_mode,
        max_sequences=args.max_sequences,
        cache_sequences=args.dataset_cache_sequences,
        cache_rgb=args.cache_rgb,
        rgb_cache_max_frames=args.rgb_cache_max_frames,
        split_file=args.split_file,
        split_key=args.split_key,
        prefer_h5_cache=args.prefer_h5_cache,
        include_human_vertices=args.lambda_phys_penetration > 0.0,
        include_keypoint_heatmaps=False,
    )
    if accelerator.is_main_process:
        num_samples = len(dataset)
        global_batch = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        effective_epochs = float(args.max_steps * global_batch) / float(max(num_samples, 1))
        steps_per_epoch = math.ceil(float(num_samples) / float(max(global_batch, 1)))
        print(
            "dataset "
            f"| sequences={len(dataset.sequence_dirs)} | samples={num_samples} "
            f"| global_batch={global_batch} | steps_per_epoch~={steps_per_epoch} "
            f"| effective_epochs~={effective_epochs:.2f}",
            flush=True,
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
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        scheduler_type=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

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
        "human_gaussian_chamfer": args.lambda_human_gaussian * args.lambda_gaussian_chamfer,
        "object_gaussian_chamfer": args.lambda_object_gaussian * args.lambda_gaussian_chamfer,
        "human_gaussian_xyz": args.lambda_human_gaussian * args.lambda_gaussian_xyz_l1,
        "object_gaussian_xyz": args.lambda_object_gaussian * args.lambda_gaussian_xyz_l1,
        "human_gaussian_attr": args.lambda_human_gaussian * args.lambda_gaussian_attr_l1,
        "object_gaussian_attr": args.lambda_object_gaussian * args.lambda_gaussian_attr_l1,
    }

    model.train()
    progress = tqdm(
        total=args.max_steps,
        initial=global_step,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
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
                    first_frame_latents = raw_model.encode_first_frame(rgb[:, 0])
                    video_target = first_frame_latents if args.clip_length == 1 else raw_model.encode_video_target(rgb)
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

                current_hoi_to_rgb_scale = hoi_to_rgb_scale_for_step(args, global_step)
                current_coattention_stage = coattention_stage_for_step(args, global_step)
                forward_kwargs = {
                    "video_xt": video_xt,
                    "state_xt": state_xt,
                    "timesteps": timesteps,
                    "first_frame": rgb[:, 0],
                    "rgb_to_hoi_scale": args.rgb_to_hoi_scale,
                    "hoi_to_rgb_scale": current_hoi_to_rgb_scale,
                }
                if args.model_variant == "comovi":
                    forward_kwargs["cross_3d2d_scale"] = args.cross_3d2d_scale
                output = model(**forward_kwargs)
                state_fm = F.mse_loss(output.state_velocity.float(), state_velocity_target.float())
                state_losses = compute_state_losses(output.decoded_state, batch, weights=supervised_weights)
                physics_losses = compute_physics_losses(output.decoded_state, batch, args=args)
                loss = args.lambda_state_fm * state_fm + state_losses["supervised"] + physics_losses["physics"]

                rgb_fm = video_xt.new_zeros(())
                if args.lambda_rgb_fm > 0.0:
                    rgb_fm = F.mse_loss(output.rgb_velocity[:, :, 1:].float(), video_velocity_target[:, :, 1:].float())
                    loss = loss + args.lambda_rgb_fm * rgb_fm
                loss = loss + zero_loss_anchor_from_output(output)

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
                    metrics = {
                        "loss": float(loss.detach().item()),
                        "loss_state_fm": float(state_fm.detach().item()),
                        "loss_rgb_fm": float(rgb_fm.detach().item()),
                        "loss_supervised": float(state_losses["supervised"].detach().item()),
                        "loss_physics": float(physics_losses["physics"].detach().item()),
                        "cross_3d2d_scale": float(args.cross_3d2d_scale),
                        "rgb_to_hoi_scale": float(args.rgb_to_hoi_scale),
                        "hoi_to_rgb_scale": float(current_hoi_to_rgb_scale),
                        "coattention_stage": float(current_coattention_stage),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    for key, value in state_losses.items():
                        if key != "supervised":
                            metrics[f"loss_{key}"] = float(value.detach().item())
                    for key, value in physics_losses.items():
                        if key != "physics":
                            metrics[f"loss_{key}"] = float(value.detach().item())
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.5f}",
                        lr=f"{metrics['lr']:.2e}",
                        stage=f"s{current_coattention_stage}",
                        refresh=True,
                    )
                    if args.log_with != "none":
                        accelerator.log(metrics, step=global_step)

                if (
                    accelerator.is_main_process
                    and args.train_visual_every > 0
                    and global_step % args.train_visual_every == 0
                ):
                    log_wandb_visuals(
                        model=raw_model,
                        batch=batch,
                        decoded=output.decoded_state,
                        video_xt=video_xt,
                        rgb_velocity=output.rgb_velocity,
                        timesteps=timesteps,
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
        print(f"done | steps={global_step:07d} | output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
