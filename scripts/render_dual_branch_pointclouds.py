#!/usr/bin/env python3
"""
Render honest dual-branch 3D comparisons and optionally upload them to WandB.

The script avoids the misleading mixed-space `merged_world` view:
- human is visualized in canonical space, pred vs GT
- object is visualized in frame-0 world space, pred vs GT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from pytorch3d.renderer import OrthographicCameras, look_at_view_transform
from pytorch3d.structures import Pointclouds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_utils import render_pointcloud_batch_pytorch3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render honest dual-branch pred-vs-GT point-cloud comparisons.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--sequences", type=str, nargs="+", required=True)
    parser.add_argument("--pred_gs_subdir", type=str, default="gs_pred_step10000_behave_test_vis")
    parser.add_argument("--gt_gs_subdir", type=str, default="gs_init")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--render_subdir", type=str, default="gs_pred_step10000_behave_test_vis/honest_renders")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=0.012)
    parser.add_argument("--points_per_pixel", type=int, default=12)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--dist", type=float, default=1.9)
    parser.add_argument("--azim_start", type=float, default=180.0)
    parser.add_argument("--focal_scale", type=float, default=3.0)
    parser.add_argument("--scene_scale", type=float, default=0.85)
    parser.add_argument("--quantile", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="dual-branch-fm")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_id", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_step", type=int, default=0)
    return parser.parse_args()


def load_gaussian_tokens(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}.")
    if "raw" in payload and torch.is_tensor(payload["raw"]):
        raw = payload["raw"].float()
    else:
        required = ("xyz", "rotation", "scaling", "opacity", "shs")
        if any(key not in payload for key in required):
            raise KeyError(f"Missing Gaussian keys in {path}: expected {required}.")
        raw = torch.cat(
            [
                payload["xyz"].float(),
                payload["rotation"].float(),
                payload["scaling"].float(),
                payload["opacity"].float(),
                payload["shs"].float(),
            ],
            dim=-1,
        )
    if raw.ndim != 2 or raw.shape[-1] != 14:
        raise ValueError(f"Expected raw Gaussian tokens [N, 14] in {path}, got {tuple(raw.shape)}.")
    return raw


def gaussian_tokens_to_cloud(tokens: torch.Tensor) -> np.ndarray:
    xyz = tokens[:, 0:3].detach().cpu().numpy().astype(np.float32)
    rgb = tokens[:, 11:14].detach().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def load_object_pose_frame0(path: Path) -> torch.Tensor:
    payload = np.load(path)
    if "object_poses" not in payload:
        raise KeyError(f"`object_poses` missing in {path}. Available keys: {payload.files}")
    poses = payload["object_poses"]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected object poses [T, 4, 4] in {path}, got {tuple(poses.shape)}.")
    return torch.from_numpy(poses[0].astype(np.float32))


def load_predicted_object_pose_frame0(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    motion = payload.get("motion")
    if not isinstance(motion, dict) or "object_poses" not in motion:
        raise KeyError(f"Missing `motion.object_poses` in {path}.")
    poses = motion["object_poses"].float()
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected predicted object poses [T, 4, 4] in {path}, got {tuple(poses.shape)}.")
    return poses[0]


def transform_cloud(cloud: np.ndarray, transform: torch.Tensor) -> np.ndarray:
    transform_np = transform.detach().cpu().numpy().astype(np.float32)
    xyz = cloud[:, :3]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_world = (transform_np @ xyz_h.T).T[:, :3]
    out = cloud.copy()
    out[:, :3] = xyz_world
    return out


def write_ascii_ply(path: Path, cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = (np.clip(cloud[:, 3:6], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {cloud.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for idx in range(cloud.shape[0]):
            handle.write(
                f"{cloud[idx, 0]:.6f} {cloud[idx, 1]:.6f} {cloud[idx, 2]:.6f} "
                f"{int(rgb_u8[idx, 0])} {int(rgb_u8[idx, 1])} {int(rgb_u8[idx, 2])}\n"
            )


def cloud_to_object3d(cloud: np.ndarray) -> np.ndarray:
    xyz = cloud[:, :3].astype(np.float32)
    rgb = (np.clip(cloud[:, 3:6], 0.0, 1.0) * 255.0).astype(np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def compute_normalization(arrays: Sequence[np.ndarray], *, quantile: float, scene_scale: float) -> Tuple[np.ndarray, float]:
    xyz = np.concatenate([array[:, :3] for array in arrays], axis=0)
    lo = np.quantile(xyz, quantile, axis=0)
    hi = np.quantile(xyz, 1.0 - quantile, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = float(np.max(hi - lo) * 0.5)
    scale = max(scale, 1e-6) / max(scene_scale, 1e-6)
    return center, scale


def normalize_cloud(cloud: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    out = cloud.copy()
    out[:, :3] = (out[:, :3] - center[None, :]) / scale
    out[:, 3:6] = np.clip(out[:, 3:6], 0.0, 1.0)
    return out


def build_batch_pointclouds(arrays: Sequence[np.ndarray], device: torch.device) -> Pointclouds:
    return Pointclouds(
        points=[torch.from_numpy(array[:, :3]).float().to(device) for array in arrays],
        features=[torch.from_numpy(array[:, 3:6]).float().to(device) for array in arrays],
    ).to(device)


def make_camera(
    *,
    device: torch.device,
    dist: float,
    elev: float,
    azim: float,
    focal_scale: float,
    batch_size: int,
) -> OrthographicCameras:
    R, T = look_at_view_transform(
        dist=[dist] * batch_size,
        elev=[elev] * batch_size,
        azim=[azim] * batch_size,
        degrees=True,
        device=device,
    )
    return OrthographicCameras(
        focal_length=(0.25 * focal_scale),
        device=device,
        R=R,
        T=T,
    )


def add_comparison_banner(frame: np.ndarray, *, title: str, labels: Sequence[str]) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return frame
    _, width, _ = frame.shape
    panel_width = width // max(len(labels), 1)
    banner = np.full((64, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    for idx, label in enumerate(labels):
        x = idx * panel_width + 14
        cv2.putText(
            banner,
            label,
            (x, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    return np.concatenate([banner, frame], axis=0)


def compute_pair_metrics(pred_cloud: np.ndarray, gt_cloud: np.ndarray) -> Dict[str, float]:
    pred = torch.from_numpy(pred_cloud[:, :3].astype(np.float32))
    gt = torch.from_numpy(gt_cloud[:, :3].astype(np.float32))
    distances = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    return {
        "pred_to_gt_mean_nn": float(distances.min(dim=1).values.mean().item()),
        "gt_to_pred_mean_nn": float(distances.min(dim=0).values.mean().item()),
    }


def render_comparison(
    *,
    pair_key: str,
    title: str,
    pred_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    render_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Path]:
    outputs = {
        "video": render_dir / f"{pair_key}_pred_vs_gt_rotate.mp4",
        "preview": render_dir / f"{pair_key}_pred_vs_gt_front.png",
        "pred_ply": render_dir / f"{pair_key}_pred.ply",
        "gt_ply": render_dir / f"{pair_key}_gt.ply",
        "meta": render_dir / f"{pair_key}_meta.json",
    }
    if (not args.overwrite) and all(path.exists() for path in outputs.values()):
        return outputs

    write_ascii_ply(outputs["pred_ply"], pred_cloud)
    write_ascii_ply(outputs["gt_ply"], gt_cloud)

    center, scale = compute_normalization([pred_cloud, gt_cloud], quantile=args.quantile, scene_scale=args.scene_scale)
    pred_norm = normalize_cloud(pred_cloud, center=center, scale=scale)
    gt_norm = normalize_cloud(gt_cloud, center=center, scale=scale)
    batch_cloud = build_batch_pointclouds([pred_norm, gt_norm], device=device)

    azims = np.linspace(args.azim_start, args.azim_start + 360.0, num=args.num_frames, endpoint=False)
    frames: List[np.ndarray] = []
    for frame_idx, azim in enumerate(azims):
        camera = make_camera(
            device=device,
            dist=args.dist,
            elev=args.elev,
            azim=float(azim),
            focal_scale=args.focal_scale,
            batch_size=2,
        )
        rendered = render_pointcloud_batch_pytorch3d(
            camera,
            batch_cloud,
            image_size=args.image_size,
            radius=args.radius,
            points_per_pixel=args.points_per_pixel,
            background_color=(0.94, 0.94, 0.94),
        )
        rendered_np = (rendered.detach().cpu().numpy() * 255.0).astype(np.uint8)
        frame = np.concatenate([rendered_np[0], rendered_np[1]], axis=1)
        frame = add_comparison_banner(frame, title=title, labels=("pred", "GT"))
        frames.append(frame)
        if frame_idx == 0:
            imageio.imwrite(outputs["preview"], frame)

    imageio.mimwrite(outputs["video"], frames, fps=args.fps)
    outputs["meta"].write_text(
        json.dumps(
            {
                "pair_key": pair_key,
                "title": title,
                "normalization_center": center.tolist(),
                "normalization_scale": scale,
                "num_frames": args.num_frames,
                "fps": args.fps,
                "image_size": args.image_size,
                "dist": args.dist,
                "elev": args.elev,
                "azim_start": args.azim_start,
                "focal_scale": args.focal_scale,
                "radius": args.radius,
                "points_per_pixel": args.points_per_pixel,
                "scene_scale": args.scene_scale,
                "quantile": args.quantile,
                "pred_points": int(pred_cloud.shape[0]),
                "gt_points": int(gt_cloud.shape[0]),
            },
            indent=2,
        )
    )
    return outputs


def resolve_gt_human_path(seq_root: Path, gt_gs_subdir: str) -> Path:
    preferred = seq_root / gt_gs_subdir / "G_h_smpl.pt"
    if preferred.is_file():
        return preferred
    fallback = seq_root / gt_gs_subdir / "G_h.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing GT human Gaussian file under {seq_root / gt_gs_subdir}")


def render_sequence(
    *,
    seq_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Path], Dict[str, np.ndarray], Dict[str, float]]:
    pred_root = seq_root / args.pred_gs_subdir
    gt_root = seq_root / args.gt_gs_subdir
    render_dir = seq_root / args.render_subdir
    render_dir.mkdir(parents=True, exist_ok=True)

    pred_human = gaussian_tokens_to_cloud(load_gaussian_tokens(pred_root / "G_h.pt"))
    pred_object = gaussian_tokens_to_cloud(load_gaussian_tokens(pred_root / "G_o.pt"))
    gt_human = gaussian_tokens_to_cloud(load_gaussian_tokens(resolve_gt_human_path(seq_root, args.gt_gs_subdir)))
    gt_object = gaussian_tokens_to_cloud(load_gaussian_tokens(gt_root / "G_o.pt"))

    pred_pose_frame0 = load_predicted_object_pose_frame0(pred_root / "gs_init_combined.pt")
    gt_pose_frame0 = load_object_pose_frame0(seq_root / args.processed_subdir / "object_poses.npz")
    pred_object_world = transform_cloud(pred_object, pred_pose_frame0)
    gt_object_world = transform_cloud(gt_object, gt_pose_frame0)

    human_outputs = render_comparison(
        pair_key="human_canonical",
        title="human canonical: pred vs GT",
        pred_cloud=pred_human,
        gt_cloud=gt_human,
        render_dir=render_dir,
        args=args,
        device=device,
    )
    object_outputs = render_comparison(
        pair_key="object_world_frame0000",
        title="object world frame0: pred vs GT",
        pred_cloud=pred_object_world,
        gt_cloud=gt_object_world,
        render_dir=render_dir,
        args=args,
        device=device,
    )

    outputs = {
        "human_canonical_video": human_outputs["video"],
        "human_canonical_preview": human_outputs["preview"],
        "human_canonical_pred_ply": human_outputs["pred_ply"],
        "human_canonical_gt_ply": human_outputs["gt_ply"],
        "object_world_video": object_outputs["video"],
        "object_world_preview": object_outputs["preview"],
        "object_world_pred_ply": object_outputs["pred_ply"],
        "object_world_gt_ply": object_outputs["gt_ply"],
        "render_meta": render_dir / "honest_render_meta.json",
    }

    clouds = {
        "human_canonical_pred": pred_human,
        "human_canonical_gt": gt_human,
        "object_world_frame0000_pred": pred_object_world,
        "object_world_frame0000_gt": gt_object_world,
    }
    metrics = {
        "human_canonical/pred_to_gt_mean_nn": compute_pair_metrics(pred_human, gt_human)["pred_to_gt_mean_nn"],
        "human_canonical/gt_to_pred_mean_nn": compute_pair_metrics(pred_human, gt_human)["gt_to_pred_mean_nn"],
        "object_world_frame0000/pred_to_gt_mean_nn": compute_pair_metrics(pred_object_world, gt_object_world)["pred_to_gt_mean_nn"],
        "object_world_frame0000/gt_to_pred_mean_nn": compute_pair_metrics(pred_object_world, gt_object_world)["gt_to_pred_mean_nn"],
    }
    outputs["render_meta"].write_text(
        json.dumps(
            {
                "sequence_name": seq_root.name,
                "pred_gs_subdir": args.pred_gs_subdir,
                "gt_gs_subdir": args.gt_gs_subdir,
                "processed_subdir": args.processed_subdir,
                "render_dir": str(render_dir),
                "metrics": metrics,
                "notes": [
                    "human is compared in canonical space",
                    "object is compared in frame-0 world space using predicted and GT object poses separately",
                    "no mixed-space merged visualization is produced",
                ],
            },
            indent=2,
        )
    )
    return outputs, clouds, metrics


def init_wandb(args: argparse.Namespace, sequences: List[str]):
    if not args.wandb:
        return None
    import wandb

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
        "mode": args.wandb_mode,
        "job_type": "render3d_honest",
        "config": {
            "render_sequences": sequences,
            "render_image_size": args.image_size,
            "render_num_frames": args.num_frames,
            "render_fps": args.fps,
            "render_dist": args.dist,
            "render_elev": args.elev,
            "pred_gs_subdir": args.pred_gs_subdir,
            "gt_gs_subdir": args.gt_gs_subdir,
            "processed_subdir": args.processed_subdir,
            "render_subdir": args.render_subdir,
        },
    }
    if args.wandb_name:
        init_kwargs["name"] = args.wandb_name
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"
    try:
        run = wandb.init(**init_kwargs)
    except Exception:
        if not args.wandb_run_id:
            raise
        init_kwargs.pop("id", None)
        init_kwargs.pop("resume", None)
        init_kwargs["name"] = args.wandb_name or "render3d-honest-fallback"
        run = wandb.init(**init_kwargs)
    print(f"[wandb] {run.url}")
    return run


def log_wandb_outputs(
    run,
    *,
    sequence: str,
    outputs: Dict[str, Path],
    clouds: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    step: int,
) -> None:
    import wandb

    prefix = f"honest3d/{sequence}"
    wandb_log = {
        f"{prefix}/human_canonical_pred_vs_gt_video": wandb.Video(str(outputs["human_canonical_video"]), format="mp4"),
        f"{prefix}/human_canonical_pred_vs_gt_preview": wandb.Image(str(outputs["human_canonical_preview"])),
        f"{prefix}/object_world_frame0000_pred_vs_gt_video": wandb.Video(str(outputs["object_world_video"]), format="mp4"),
        f"{prefix}/object_world_frame0000_pred_vs_gt_preview": wandb.Image(str(outputs["object_world_preview"])),
        f"{prefix}/pointcloud/human_canonical_pred": wandb.Object3D(cloud_to_object3d(clouds["human_canonical_pred"])),
        f"{prefix}/pointcloud/human_canonical_gt": wandb.Object3D(cloud_to_object3d(clouds["human_canonical_gt"])),
        f"{prefix}/pointcloud/object_world_frame0000_pred": wandb.Object3D(cloud_to_object3d(clouds["object_world_frame0000_pred"])),
        f"{prefix}/pointcloud/object_world_frame0000_gt": wandb.Object3D(cloud_to_object3d(clouds["object_world_frame0000_gt"])),
    }
    for key, value in metrics.items():
        wandb_log[f"{prefix}/metrics/{key}"] = value
    wandb.log(wandb_log, step=step)

    artifact = wandb.Artifact(name=f"{sequence}-honest-render3d".replace("/", "-"), type="render3d")
    for path in outputs.values():
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA rendering requested but CUDA is not available.")

    input_root = Path(args.input_root).expanduser().resolve()
    run = init_wandb(args, args.sequences)
    wandb_step_base = args.wandb_step
    if run is not None:
        current_step = int(getattr(run, "step", 0) or 0)
        wandb_step_base = max(args.wandb_step, current_step + 1)

    try:
        for sequence_idx, sequence in enumerate(args.sequences):
            seq_root = input_root / sequence
            if not seq_root.exists():
                raise FileNotFoundError(seq_root)
            outputs, clouds, metrics = render_sequence(
                seq_root=seq_root,
                args=args,
                device=device,
            )
            print(f"[render] {sequence} -> {outputs['human_canonical_video']}")
            print(f"[render] {sequence} -> {outputs['object_world_video']}")
            if run is not None:
                log_wandb_outputs(
                    run,
                    sequence=sequence,
                    outputs=outputs,
                    clouds=clouds,
                    metrics=metrics,
                    step=wandb_step_base + sequence_idx,
                )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
