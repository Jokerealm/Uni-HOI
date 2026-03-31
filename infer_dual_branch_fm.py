#!/usr/bin/env python3
"""
Run dual-branch co-generative Flow Matching inference for a single sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.dual_branch_fm_dataset import _build_keypoint_heatmaps, load_dual_branch_sequence_bundle, load_rgb_image
from model.dual_branch_cogenerative_fm import DecodedHOIState, DualBranchCoGenerativeFlowMatching
from train_dual_branch_fm import resize_video_batch, scale_camera_intrinsics
from train_dual_branch_fm import build_arg_parser as build_train_arg_parser


def resolve_video_dir(input_dir: str, video_name: str) -> Path:
    base = Path(input_dir).expanduser().resolve()
    candidates = [
        base / video_name,
        base / "sequences" / video_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve sequence directory for video={video_name} under {input_dir}. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def namespace_from_checkpoint_args(checkpoint_args: Dict[str, object]) -> argparse.Namespace:
    parser = build_train_arg_parser()
    defaults = parser.parse_args([])
    for key, value in checkpoint_args.items():
        setattr(defaults, key, value)
    return defaults


def pad_sequence(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    if tensor.shape[0] == target_frames:
        return tensor
    if tensor.shape[0] > target_frames:
        return tensor[:target_frames]
    pad = tensor[-1:].expand(target_frames - tensor.shape[0], *tensor.shape[1:])
    return torch.cat([tensor, pad], dim=0)


def load_inference_clip(
    *,
    video_dir: Path,
    processed_subdir: str,
    gs_subdir: str,
    human_gaussian_source: str,
    clip_length: int,
    num_human_gaussians: int,
    num_object_gaussians: int,
    num_joints: int,
    contact_dim: int,
    background_value: float,
) -> Dict[str, torch.Tensor]:
    bundle = load_dual_branch_sequence_bundle(
        str(video_dir),
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        human_gaussian_source=human_gaussian_source,
        num_human_gaussians=max(1, num_human_gaussians),
        num_object_gaussians=max(1, num_object_gaussians),
        num_joints=num_joints,
        contact_dim=contact_dim,
        require_gaussian_targets=False,
    )

    rgb = torch.stack([load_rgb_image(str(path)) for path in bundle["rgb_paths"]], dim=0)
    masks_human = bundle["masks_human"]
    masks_object = bundle["masks_object"]
    m_primary = bundle["m_primary"]
    m_secondary = bundle["m_secondary"]
    m_object_region = bundle["m_object_region"]
    depth = bundle["depth"]
    keypoint_heatmaps = _build_keypoint_heatmaps(
        bundle["keypoints_2d"],
        height=depth.shape[-2],
        width=depth.shape[-1],
    )
    camera_intrinsics = bundle["intrinsics"]

    rgb = pad_sequence(rgb, clip_length)
    masks_human = pad_sequence(masks_human, clip_length)
    masks_object = pad_sequence(masks_object, clip_length)
    m_primary = pad_sequence(m_primary, clip_length)
    m_secondary = pad_sequence(m_secondary, clip_length)
    m_object_region = pad_sequence(m_object_region, clip_length)
    depth = pad_sequence(depth, clip_length)
    keypoint_heatmaps = pad_sequence(keypoint_heatmaps, clip_length)
    camera_intrinsics = pad_sequence(camera_intrinsics, clip_length)

    background = torch.full_like(rgb, background_value)
    human_visible = rgb * masks_human + background * (1.0 - masks_human)
    object_visible = rgb * masks_object + background * (1.0 - masks_object)
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
        dim=1,
    )

    return {
        "rgb": rgb.unsqueeze(0),
        "human_visible": human_visible.unsqueeze(0),
        "object_visible": object_visible.unsqueeze(0),
        "condition_video": condition_video.unsqueeze(0),
        "masks_human": masks_human.unsqueeze(0),
        "masks_object": masks_object.unsqueeze(0),
        "camera_intrinsics": camera_intrinsics.unsqueeze(0),
        "sequence_name": bundle["sequence_name"],
        "teacher_human_gaussians": bundle.get("human_gaussians"),
        "teacher_object_gaussians": bundle.get("object_gaussians"),
        "teacher_object_pose_frame0": bundle.get("object_poses")[0] if bundle.get("object_poses") is not None else None,
    }


def save_video_branch(video: torch.Tensor, branch_dir: Path, *, save_frames: bool, fps: int) -> None:
    import imageio.v2 as imageio

    branch_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().clamp(0.0, 1.0).cpu()
    frames_uint8 = []
    frames_dir = branch_dir / "frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(video):
        frame_uint8 = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        frames_uint8.append(frame_uint8)
        if save_frames:
            Image.fromarray(frame_uint8).save(frames_dir / f"{frame_idx:06d}.png")
    imageio.mimwrite(branch_dir / "inpaint_out.mp4", frames_uint8, fps=fps, quality=7)


def save_gaussian_tokens(tokens: torch.Tensor, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "xyz": tokens[..., 0:3].squeeze(0).cpu(),
        "rotation": tokens[..., 3:7].squeeze(0).cpu(),
        "scaling": tokens[..., 7:10].squeeze(0).cpu(),
        "opacity": tokens[..., 10:11].squeeze(0).cpu(),
        "shs": tokens[..., 11:14].squeeze(0).cpu(),
        "raw": tokens.squeeze(0).cpu(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def save_combined_state(
    decoded_state: DecodedHOIState,
    output_dir: Path,
    *,
    sequence_name: str,
    num_ode_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "G_h": {
            "raw": decoded_state.human_gaussians.squeeze(0).cpu(),
        },
        "G_o": {
            "raw": decoded_state.object_gaussians.squeeze(0).cpu(),
        },
        "motion": {
            "joints_3d": decoded_state.joints_3d.squeeze(0).cpu(),
            "object_poses": decoded_state.object_transforms.squeeze(0).cpu(),
            "contact_signature": decoded_state.contact_signature.squeeze(0).cpu(),
        },
        "metadata": {
            "sequence_name": sequence_name,
            "num_frames": int(decoded_state.joints_3d.shape[1]),
            "num_ode_steps": int(num_ode_steps),
        },
    }
    torch.save(payload, output_dir / "gs_init_combined.pt")


def gaussian_tokens_to_xyz_rgb(tokens: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    raw = tokens.detach().cpu()
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError(f"Expected batched Gaussian tokens with batch=1, got {tuple(raw.shape)}.")
        raw = raw.squeeze(0)
    if raw.ndim != 2:
        raise ValueError(f"Expected Gaussian tokens with shape [N, 14], got {tuple(raw.shape)}.")
    xyz = raw[:, 0:3].numpy()
    rgb = raw[:, 11:14].clamp(0.0, 1.0).numpy()
    return xyz, rgb


def transform_points(points: np.ndarray, transform: torch.Tensor) -> np.ndarray:
    transform_np = transform.detach().cpu().numpy()
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (transform_np @ points_h.T).T[:, :3]


def cloud_array_from_tokens(tokens: torch.Tensor) -> np.ndarray:
    xyz, rgb = gaussian_tokens_to_xyz_rgb(tokens)
    return np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1)


def transform_cloud_array(cloud: np.ndarray, transform: torch.Tensor) -> np.ndarray:
    out = cloud.copy()
    out[:, :3] = transform_points(out[:, :3], transform)
    return out


def cloud_array_to_object3d(cloud: np.ndarray) -> np.ndarray:
    xyz = cloud[:, :3].astype(np.float32)
    rgb = (np.clip(cloud[:, 3:6], 0.0, 1.0) * 255.0).astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def bidirectional_nn_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> Dict[str, float]:
    pred = torch.from_numpy(pred_xyz.astype(np.float32))
    gt = torch.from_numpy(gt_xyz.astype(np.float32))
    distances = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    return {
        "pred_to_gt_mean_nn": float(distances.min(dim=1).values.mean().item()),
        "gt_to_pred_mean_nn": float(distances.min(dim=0).values.mean().item()),
    }


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for idx in range(xyz.shape[0]):
            handle.write(
                f"{xyz[idx, 0]:.6f} {xyz[idx, 1]:.6f} {xyz[idx, 2]:.6f} "
                f"{int(rgb_u8[idx, 0])} {int(rgb_u8[idx, 1])} {int(rgb_u8[idx, 2])}\n"
            )


def export_point_cloud_visualization(
    decoded_state: DecodedHOIState,
    output_dir: Path,
    *,
    sequence_name: str,
    teacher_human_gaussians: torch.Tensor | None = None,
    teacher_object_gaussians: torch.Tensor | None = None,
    teacher_object_pose_frame0: torch.Tensor | None = None,
) -> Tuple[Dict[str, Path], Dict[str, np.ndarray], Dict[str, float]]:
    pointcloud_dir = output_dir / "pointcloud"
    pred_human_canonical = cloud_array_from_tokens(decoded_state.human_gaussians)
    pred_object_canonical = cloud_array_from_tokens(decoded_state.object_gaussians)
    pred_object_world = transform_cloud_array(pred_object_canonical, decoded_state.object_transforms[0, 0])

    pred_human_path = pointcloud_dir / "pred_human_canonical.ply"
    pred_object_canonical_path = pointcloud_dir / "pred_object_canonical.ply"
    pred_object_world_path = pointcloud_dir / "pred_object_world_frame0000.ply"
    write_ascii_ply(pred_human_path, pred_human_canonical[:, :3], pred_human_canonical[:, 3:6])
    write_ascii_ply(pred_object_canonical_path, pred_object_canonical[:, :3], pred_object_canonical[:, 3:6])
    write_ascii_ply(pred_object_world_path, pred_object_world[:, :3], pred_object_world[:, 3:6])

    clouds = {
        "pred_human_canonical": pred_human_canonical,
        "pred_object_canonical": pred_object_canonical,
        "pred_object_world_frame0000": pred_object_world,
    }
    metrics: Dict[str, float] = {}

    if teacher_human_gaussians is not None:
        gt_human_canonical = cloud_array_from_tokens(teacher_human_gaussians)
        gt_human_path = pointcloud_dir / "gt_human_canonical.ply"
        write_ascii_ply(gt_human_path, gt_human_canonical[:, :3], gt_human_canonical[:, 3:6])
        clouds["gt_human_canonical"] = gt_human_canonical
        metrics.update(
            {
                f"human_canonical_{name}": value
                for name, value in bidirectional_nn_metrics(
                    pred_human_canonical[:, :3],
                    gt_human_canonical[:, :3],
                ).items()
            }
        )
    else:
        gt_human_path = None

    if teacher_object_gaussians is not None:
        gt_object_canonical = cloud_array_from_tokens(teacher_object_gaussians)
        gt_object_canonical_path = pointcloud_dir / "gt_object_canonical.ply"
        write_ascii_ply(gt_object_canonical_path, gt_object_canonical[:, :3], gt_object_canonical[:, 3:6])
        clouds["gt_object_canonical"] = gt_object_canonical
        if teacher_object_pose_frame0 is not None:
            gt_object_world = transform_cloud_array(gt_object_canonical, teacher_object_pose_frame0)
            gt_object_world_path = pointcloud_dir / "gt_object_world_frame0000.ply"
            write_ascii_ply(gt_object_world_path, gt_object_world[:, :3], gt_object_world[:, 3:6])
            clouds["gt_object_world_frame0000"] = gt_object_world
            metrics.update(
                {
                    f"object_world_frame0000_{name}": value
                    for name, value in bidirectional_nn_metrics(
                        pred_object_world[:, :3],
                        gt_object_world[:, :3],
                    ).items()
                }
            )
        else:
            gt_object_world_path = None
    else:
        gt_object_canonical_path = None
        gt_object_world_path = None

    metadata = {
        "sequence_name": sequence_name,
        "num_pred_human_points": int(pred_human_canonical.shape[0]),
        "num_pred_object_points": int(pred_object_world.shape[0]),
        "num_gt_human_points": int(clouds["gt_human_canonical"].shape[0]) if "gt_human_canonical" in clouds else 0,
        "num_gt_object_points": int(clouds["gt_object_canonical"].shape[0]) if "gt_object_canonical" in clouds else 0,
        "frame_index": 0,
        "notes": [
            "human is exported in canonical space",
            "object is exported in both canonical and frame-0 world space",
            "no mixed-space merged point cloud is exported by default",
        ],
    }
    (pointcloud_dir / "pointcloud_meta.json").write_text(json.dumps(metadata, indent=2))
    return (
        {
            "pred_human_canonical": pred_human_path,
            "pred_object_canonical": pred_object_canonical_path,
            "pred_object_world_frame0000": pred_object_world_path,
            **({"gt_human_canonical": gt_human_path} if gt_human_path is not None else {}),
            **({"gt_object_canonical": gt_object_canonical_path} if gt_object_canonical_path is not None else {}),
            **({"gt_object_world_frame0000": gt_object_world_path} if gt_object_world_path is not None else {}),
        },
        clouds,
        metrics,
    )


def maybe_log_wandb_visualization(
    *,
    args: argparse.Namespace,
    sequence_name: str,
    pointcloud_paths: Dict[str, Path],
    pointcloud_payload: Dict[str, np.ndarray],
    pointcloud_metrics: Dict[str, float],
) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        print(f"[infer_dual_branch_fm] wandb import failed: {exc}")
        return

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or f"dual-branch-infer-{sequence_name}",
            mode=args.wandb_mode,
            config={
                "sequence_name": sequence_name,
                "checkpoint": args.checkpoint,
                "num_ode_steps": args.num_ode_steps,
            },
        )
    except Exception as exc:  # pragma: no cover - best-effort logging path
        print(f"[infer_dual_branch_fm] wandb init failed: {exc}")
        return

    try:
        wandb_log = {}
        for name, cloud in pointcloud_payload.items():
            wandb_log[f"pointcloud/{name}"] = wandb.Object3D(cloud_array_to_object3d(cloud))
        for name, value in pointcloud_metrics.items():
            wandb_log[f"pointcloud_metrics/{name}"] = value
        wandb.log(wandb_log)
        artifact = wandb.Artifact(
            name=(args.wandb_artifact_name or f"{sequence_name}-pointcloud-honest").replace("/", "-"),
            type="pointcloud",
        )
        for path in pointcloud_paths.values():
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def build_runtime_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-branch co-generative Flow Matching inference.")
    parser.add_argument("--input_dir", type=str, required=False, default="")
    parser.add_argument("--video_name", type=str, required=False, default="")
    parser.add_argument("--checkpoint", type=str, required=False, default="")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--output_subdir", type=str, default="amodal")
    parser.add_argument("--gs_output_subdir", type=str, default="gs_init")
    parser.add_argument("--num_ode_steps", type=int, default=50)
    parser.add_argument("--prior_noise_std", type=float, default=1.0)
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clamp_visible_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="uni-hoi-4d")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_artifact_name", type=str, default="")
    return parser


def run_dual_branch_inference(args: argparse.Namespace) -> Dict[str, Path]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = namespace_from_checkpoint_args(checkpoint.get("args", {}))
    video_dir = resolve_video_dir(args.input_dir, args.video_name)
    inputs = load_inference_clip(
        video_dir=video_dir,
        processed_subdir=args.processed_subdir,
        gs_subdir=args.gs_subdir,
        human_gaussian_source=getattr(checkpoint_args, "human_gaussian_source", "smpl_mesh"),
        clip_length=checkpoint_args.clip_length,
        num_human_gaussians=checkpoint_args.num_human_gaussians,
        num_object_gaussians=checkpoint_args.num_object_gaussians,
        num_joints=checkpoint_args.num_joints,
        contact_dim=checkpoint_args.contact_dim,
        background_value=getattr(checkpoint_args, "background_value", 1.0),
    )
    condition_channels = int(inputs["condition_video"].shape[2])

    model = DualBranchCoGenerativeFlowMatching(
        hidden_dim=checkpoint_args.hidden_dim,
        num_heads=checkpoint_args.num_heads,
        depth=checkpoint_args.depth,
        mlp_ratio=checkpoint_args.mlp_ratio,
        dropout=checkpoint_args.dropout,
        condition_channels=condition_channels,
        video_channels=checkpoint_args.video_channels,
        patch_size=checkpoint_args.patch_size,
        num_frames=checkpoint_args.clip_length,
        image_height=checkpoint_args.image_height,
        image_width=checkpoint_args.image_width,
        num_human_gaussians=checkpoint_args.num_human_gaussians,
        num_object_gaussians=checkpoint_args.num_object_gaussians,
        num_joints=checkpoint_args.num_joints,
        contact_dim=checkpoint_args.contact_dim,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device=device).eval()
    condition_video = inputs["condition_video"].to(device)
    masks_human = inputs["masks_human"].to(device)
    masks_object = inputs["masks_object"].to(device)
    human_visible = inputs["human_visible"].to(device)
    object_visible = inputs["object_visible"].to(device)
    camera_intrinsics = inputs["camera_intrinsics"].to(device)

    if condition_video.shape[-2:] != (checkpoint_args.image_height, checkpoint_args.image_width):
        source_hw = condition_video.shape[-2:]
        condition_video = resize_video_batch(
            condition_video,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="bilinear",
        )
        human_visible = resize_video_batch(
            human_visible,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="bilinear",
        )
        object_visible = resize_video_batch(
            object_visible,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="bilinear",
        )
        masks_human = resize_video_batch(
            masks_human,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="nearest",
        )
        masks_object = resize_video_batch(
            masks_object,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="nearest",
        )
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics,
            source_size=(int(source_hw[-2]), int(source_hw[-1])),
            target_size=(checkpoint_args.image_height, checkpoint_args.image_width),
        )

    video_latents = torch.randn(
        1,
        model.video_codec.num_frames * model.video_codec.num_patches_per_frame,
        checkpoint_args.hidden_dim,
        device=device,
    ) * args.prior_noise_std
    state_latents = torch.randn(
        1,
        model.state_codec.total_tokens,
        checkpoint_args.hidden_dim,
        device=device,
    ) * args.prior_noise_std

    times = torch.linspace(0.0, 1.0, args.num_ode_steps + 1, device=device, dtype=video_latents.dtype)
    with torch.no_grad():
        for step_idx in range(args.num_ode_steps):
            t_cur = times[step_idx].expand(1)
            dt = times[step_idx + 1] - times[step_idx]
            output = model(
                video_xt=video_latents,
                state_xt=state_latents,
                timesteps=t_cur,
                condition_video=condition_video,
                camera_intrinsics=camera_intrinsics,
            )
            video_latents = video_latents + dt.view(1, 1, 1) * output.video_velocity
            state_latents = state_latents + dt.view(1, 1, 1) * output.state_velocity

        decoded_video = model.decode_video_tokens(video_latents)
        decoded_state = model.decode_state_tokens(state_latents)

    pred_human = decoded_video[:, :, :3]
    pred_object = decoded_video[:, :, 3:6]
    if args.clamp_visible_rgb:
        pred_human = pred_human * (1.0 - masks_human) + human_visible * masks_human
        pred_object = pred_object * (1.0 - masks_object) + object_visible * masks_object

    amodal_dir = video_dir / args.output_subdir
    gs_output_dir = video_dir / args.gs_output_subdir
    human_branch_dir = amodal_dir / "human_amodal"
    object_branch_dir = amodal_dir / "object_amodal"
    save_video_branch(pred_human.squeeze(0), human_branch_dir, save_frames=args.save_frames, fps=args.save_fps)
    save_video_branch(pred_object.squeeze(0), object_branch_dir, save_frames=args.save_frames, fps=args.save_fps)

    metadata = {
        "sequence_name": inputs["sequence_name"],
        "num_frames": int(decoded_video.shape[1]),
        "num_ode_steps": int(args.num_ode_steps),
    }
    save_gaussian_tokens(decoded_state.human_gaussians, gs_output_dir / "G_h.pt", metadata)
    save_gaussian_tokens(decoded_state.object_gaussians, gs_output_dir / "G_o.pt", metadata)
    save_combined_state(
        decoded_state,
        gs_output_dir,
        sequence_name=inputs["sequence_name"],
        num_ode_steps=args.num_ode_steps,
    )
    pointcloud_paths, pointcloud_payload, pointcloud_metrics = export_point_cloud_visualization(
        decoded_state,
        gs_output_dir,
        sequence_name=inputs["sequence_name"],
        teacher_human_gaussians=inputs.get("teacher_human_gaussians"),
        teacher_object_gaussians=inputs.get("teacher_object_gaussians"),
        teacher_object_pose_frame0=inputs.get("teacher_object_pose_frame0"),
    )
    maybe_log_wandb_visualization(
        args=args,
        sequence_name=inputs["sequence_name"],
        pointcloud_paths=pointcloud_paths,
        pointcloud_payload=pointcloud_payload,
        pointcloud_metrics=pointcloud_metrics,
    )
    (gs_output_dir / "dual_branch_inference.json").write_text(json.dumps(metadata, indent=2))

    return {
        "video_dir": video_dir,
        "human_amodal_dir": human_branch_dir,
        "object_amodal_dir": object_branch_dir,
        "gs_dir": gs_output_dir,
        "pointcloud_dir": next(iter(pointcloud_paths.values())).parent,
    }


def main() -> None:
    args = build_runtime_arg_parser().parse_args()
    run_dual_branch_inference(args)


if __name__ == "__main__":
    main()
