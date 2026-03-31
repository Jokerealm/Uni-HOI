#!/usr/bin/env python3
"""
Prepare BEHAVE sequences for the dual-branch FM dataloader without touching
the original dataset tree.

This script creates a lightweight mirror under `output_root`:
  - symlinks raw `t*.000` timestep directories
  - symlinks existing Step-1 / Step-3 assets from `processed/` and `gs_init/`
  - materializes missing `processed/cropped/*`
  - materializes `processed/object_poses.npz`
  - materializes `processed/cropped/dual_branch_targets.npz`
  - materializes `gs_init/G_h_smpl.pt`

It is intended for BEHAVE sequences that already have:
  - processed/depth_aligned.npz
  - processed/masks_raw.npz
  - processed/region_masks.npz
  - processed/keypoints_2d.npz
  - processed/smpl_params.npz
  - gs_init/G_o.pt (or gs_init_combined.pt)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from dataset.dual_branch_fm_dataset import load_object_pose_sequence
from dataset.video_transforms import infer_camera_intrinsics
from preprocess_procigen_gt import prepare_dual_branch_target_caches, preprocess_frame_with_intrinsics, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BEHAVE assets for dual-branch FM evaluation.")
    parser.add_argument(
        "--source_root",
        type=str,
        default="/data4/guanz/data/Behave/sequences",
        help="BEHAVE source sequence root.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(REPO_ROOT / "preprocessed" / "behave_test_dual_branch"),
        help="Writable mirror root used by the dual-branch dataloader.",
    )
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="test")
    parser.add_argument("--sequence_name", action="append", dest="sequence_names")
    parser.add_argument("--camera_id", type=str, default="k1")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--scale_ratio", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--bbox_expand", type=float, default=1.1)
    parser.add_argument("--init_gaussian_scale", type=float, default=0.01)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_split_sequence_names(split_file: str, split_key: str) -> List[str]:
    with open(split_file, "rb") as handle:
        payload = pickle.load(handle)
    if split_key not in payload:
        raise KeyError(f"Missing split `{split_key}` in {split_file}. Available keys: {sorted(payload.keys())}")
    sequence_names = set()
    for entry in payload[split_key]:
        values = entry if isinstance(entry, (list, tuple)) else (entry,)
        for value in values:
            normalized = str(value).replace("\\", "/").strip("/")
            if normalized:
                sequence_names.add(normalized.split("/")[0])
    return sorted(sequence_names)


def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def ensure_symlink(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src.resolve():
            return
        if not overwrite:
            return
        safe_unlink(dst)
    os.symlink(str(src.resolve()), str(dst))


def discover_sequence_names(args: argparse.Namespace) -> List[str]:
    sequence_names = list(args.sequence_names or [])
    if args.split_file:
        sequence_names.extend(load_split_sequence_names(args.split_file, args.split_key))
    if not sequence_names:
        source_root = Path(args.source_root).expanduser().resolve()
        sequence_names = sorted(child.name for child in source_root.iterdir() if child.is_dir())
    sequence_names = sorted(set(sequence_names))
    if args.max_sequences > 0:
        sequence_names = sequence_names[: args.max_sequences]
    return sequence_names


def list_timestep_dirs(sequence_dir: Path) -> List[Path]:
    return sorted(path for path in sequence_dir.iterdir() if path.is_dir() and path.name.startswith("t"))


def resolve_frame_paths(source_seq: Path, camera_id: str, num_frames: int) -> List[Path]:
    timestep_dirs = list_timestep_dirs(source_seq)
    if len(timestep_dirs) < num_frames:
        raise ValueError(
            f"{source_seq} provides only {len(timestep_dirs)} timestep dirs for {num_frames} required frames."
        )
    frame_paths = []
    for timestep_dir in timestep_dirs[:num_frames]:
        frame_path = timestep_dir / f"{camera_id}.color.jpg"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing BEHAVE RGB frame: {frame_path}")
        frame_paths.append(frame_path)
    return frame_paths


def build_intrinsics_matrix(image_width: int, image_height: int) -> np.ndarray:
    fx, fy, cx, cy = infer_camera_intrinsics(image_width=image_width, image_height=image_height, scale_ratio=1)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = np.float32(fx)
    K[1, 1] = np.float32(fy)
    K[0, 2] = np.float32(cx)
    K[1, 2] = np.float32(cy)
    return K


def load_raw_keypoints_2d(timestep_dir: Path, camera_id: str) -> np.ndarray:
    keypoint_path = timestep_dir / f"{camera_id}.color.json"
    if not keypoint_path.is_file():
        raise FileNotFoundError(f"Missing BEHAVE 2D keypoint file: {keypoint_path}")
    with keypoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "body_joints" not in payload:
        raise KeyError(f"Missing `body_joints` in {keypoint_path}")
    keypoints = np.asarray(payload["body_joints"], dtype=np.float32).reshape(-1)
    usable = (keypoints.size // 3) * 3
    if usable == 0:
        raise ValueError(f"No usable 2D keypoints found in {keypoint_path}")
    return keypoints[:usable].reshape(-1, 3).astype(np.float32)


def ensure_target_layout(
    source_seq: Path,
    target_seq: Path,
    *,
    processed_subdir: str,
    gs_subdir: str,
    overwrite: bool,
) -> None:
    target_seq.mkdir(parents=True, exist_ok=True)

    for timestep_dir in list_timestep_dirs(source_seq):
        ensure_symlink(timestep_dir, target_seq / timestep_dir.name, overwrite=overwrite)

    source_frames_dir = source_seq / "frames"
    if source_frames_dir.is_dir():
        ensure_symlink(source_frames_dir, target_seq / "frames", overwrite=overwrite)

    processed_dir = target_seq / processed_subdir
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "depth_aligned.npz",
        "masks_raw.npz",
        "region_masks.npz",
        "keypoints_2d.npz",
        "smpl_params.npz",
        "joints_3d.npz",
    ):
        source_path = source_seq / processed_subdir / name
        if source_path.is_file():
            ensure_symlink(source_path, processed_dir / name, overwrite=overwrite)

    for name in ("masks_human", "masks_object"):
        source_path = source_seq / processed_subdir / name
        if source_path.is_dir():
            ensure_symlink(source_path, processed_dir / name, overwrite=overwrite)

    gs_dir = target_seq / gs_subdir
    gs_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "G_h.pt",
        "G_o.pt",
        "G_h_mesh.obj",
        "G_o_mesh.obj",
        "gs_init_combined.pt",
    ):
        source_path = source_seq / gs_subdir / name
        if source_path.exists():
            ensure_symlink(source_path, gs_dir / name, overwrite=overwrite)


def build_cropped_assets(
    *,
    source_seq: Path,
    target_seq: Path,
    camera_id: str,
    processed_subdir: str,
    scale_ratio: int,
    crop_size: Sequence[int],
    bbox_expand: float,
    overwrite: bool,
) -> int:
    processed_dir = target_seq / processed_subdir
    cropped_dir = processed_dir / "cropped"
    rgb_dir = cropped_dir / "rgb"
    object_pose_path = processed_dir / "object_poses.npz"
    meta_path = cropped_dir / "meta.npz"

    if meta_path.is_file() and object_pose_path.is_file() and not overwrite:
        with np.load(meta_path) as meta_npz:
            return int(meta_npz["fx"].shape[0])

    with np.load(processed_dir / "masks_raw.npz") as masks_npz:
        masks_human = np.asarray(masks_npz["human"], dtype=np.float32)
        masks_object = np.asarray(masks_npz["object"], dtype=np.float32)
    with np.load(processed_dir / "region_masks.npz") as region_npz:
        m_primary = np.asarray(region_npz["M_p"], dtype=np.float32)
        m_secondary = np.asarray(region_npz["M_s"], dtype=np.float32)
        m_object_region = np.asarray(region_npz["M_object"], dtype=np.float32)
    with np.load(processed_dir / "depth_aligned.npz") as depth_npz:
        depth = np.asarray(depth_npz["depth"], dtype=np.float32)
    with np.load(processed_dir / "smpl_params.npz") as smpl_npz:
        num_frames = int(smpl_npz["body_pose"].shape[0])

    num_frames = min(
        num_frames,
        int(masks_human.shape[0]),
        int(masks_object.shape[0]),
        int(m_primary.shape[0]),
        int(m_secondary.shape[0]),
        int(m_object_region.shape[0]),
        int(depth.shape[0]),
    )
    timestep_dirs = list_timestep_dirs(source_seq)
    if len(timestep_dirs) < num_frames:
        raise ValueError(f"Not enough BEHAVE timesteps for {source_seq.name}: {len(timestep_dirs)} < {num_frames}")
    frame_paths = resolve_frame_paths(source_seq, camera_id=camera_id, num_frames=num_frames)

    rgb_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_frame in rgb_dir.glob("*.png"):
            old_frame.unlink()

    cropped_depth = []
    cropped_h = []
    cropped_o = []
    cropped_mp = []
    cropped_ms = []
    cropped_mobj = []
    cropped_kp = []
    bbox_xywh = []
    fx_all = []
    fy_all = []
    cx_all = []
    cy_all = []
    orig_hw = []
    down_hw = []

    K = None
    for frame_idx, frame_path in enumerate(frame_paths[:num_frames]):
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read BEHAVE frame: {frame_path}")
        if K is None:
            image_h, image_w = frame_bgr.shape[:2]
            K = build_intrinsics_matrix(image_width=image_w, image_height=image_h)
        keypoints_frame = load_raw_keypoints_2d(timestep_dirs[frame_idx], camera_id=camera_id)

        cropped = preprocess_frame_with_intrinsics(
            frame=frame_bgr,
            mask_human=masks_human[frame_idx],
            mask_object=masks_object[frame_idx],
            depth=depth[frame_idx],
            keypoints_2d=keypoints_frame,
            extra_maps={
                "M_p": m_primary[frame_idx],
                "M_s": m_secondary[frame_idx],
                "M_object": m_object_region[frame_idx],
            },
            K=K,
            scale_ratio=scale_ratio,
            bbox_expand=bbox_expand,
            out_size=(int(crop_size[0]), int(crop_size[1])),
        )
        cv2.imwrite(str(rgb_dir / f"frame_{frame_idx:06d}.png"), cropped["rgb"])
        cropped_depth.append(cropped["depth"].astype(np.float32))
        cropped_h.append(cropped["mask_human"].astype(np.float32))
        cropped_o.append(cropped["mask_object"].astype(np.float32))
        cropped_mp.append(cropped["extra_maps"]["M_p"].astype(np.float32))
        cropped_ms.append(cropped["extra_maps"]["M_s"].astype(np.float32))
        cropped_mobj.append(cropped["extra_maps"]["M_object"].astype(np.float32))
        cropped_kp.append(cropped["keypoints_2d"].astype(np.float32))
        bbox_xywh.append(cropped["bbox_xywh"].astype(np.float32))
        fx_all.append(np.float32(cropped["fx"]))
        fy_all.append(np.float32(cropped["fy"]))
        cx_all.append(np.float32(cropped["cx"]))
        cy_all.append(np.float32(cropped["cy"]))
        orig_hw.append(cropped["orig_size_hw"].astype(np.int32))
        down_hw.append(cropped["downsampled_size_hw"].astype(np.int32))

    write_npz(cropped_dir / "depth_aligned.npz", depth=np.stack(cropped_depth, axis=0).astype(np.float32))
    write_npz(
        cropped_dir / "masks_raw.npz",
        human=np.stack(cropped_h, axis=0).astype(np.float32),
        object=np.stack(cropped_o, axis=0).astype(np.float32),
    )
    write_npz(
        cropped_dir / "region_masks.npz",
        M_p=np.stack(cropped_mp, axis=0).astype(np.float32),
        M_s=np.stack(cropped_ms, axis=0).astype(np.float32),
        M_object=np.stack(cropped_mobj, axis=0).astype(np.float32),
    )
    write_npz(cropped_dir / "keypoints_2d.npz", keypoints=np.stack(cropped_kp, axis=0).astype(np.float32))
    write_npz(
        cropped_dir / "meta.npz",
        bbox_xywh=np.stack(bbox_xywh, axis=0).astype(np.float32),
        fx=np.asarray(fx_all, dtype=np.float32),
        fy=np.asarray(fy_all, dtype=np.float32),
        cx=np.asarray(cx_all, dtype=np.float32),
        cy=np.asarray(cy_all, dtype=np.float32),
        orig_size_hw=np.stack(orig_hw, axis=0).astype(np.int32),
        downsampled_size_hw=np.stack(down_hw, axis=0).astype(np.int32),
        scale_ratio=np.asarray([int(scale_ratio)], dtype=np.int32),
    )

    object_poses = load_object_pose_sequence(target_seq, num_frames, processed_subdir=processed_subdir)
    write_npz(object_pose_path, object_poses=object_poses.cpu().numpy().astype(np.float32))
    return num_frames


def prepare_sequence(
    *,
    source_seq: Path,
    target_seq: Path,
    camera_id: str,
    processed_subdir: str,
    gs_subdir: str,
    scale_ratio: int,
    crop_size: Sequence[int],
    bbox_expand: float,
    init_gaussian_scale: float,
    overwrite: bool,
) -> int:
    ensure_target_layout(
        source_seq,
        target_seq,
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        overwrite=overwrite,
    )
    num_frames = build_cropped_assets(
        source_seq=source_seq,
        target_seq=target_seq,
        camera_id=camera_id,
        processed_subdir=processed_subdir,
        scale_ratio=scale_ratio,
        crop_size=crop_size,
        bbox_expand=bbox_expand,
        overwrite=overwrite,
    )
    prepare_dual_branch_target_caches(
        target_seq,
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        init_gaussian_scale=init_gaussian_scale,
    )
    return num_frames


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_names = discover_sequence_names(args)
    if not sequence_names:
        raise RuntimeError("No BEHAVE sequences were selected.")

    print(
        f"[behave-dual-branch-prep] start | total={len(sequence_names)} "
        f"| source_root={source_root} | output_root={output_root}"
    )
    prepared = 0
    for sequence_idx, sequence_name in enumerate(sequence_names, start=1):
        source_seq = source_root / sequence_name
        if not source_seq.is_dir():
            raise FileNotFoundError(f"Missing source sequence: {source_seq}")
        target_seq = output_root / sequence_name
        num_frames = prepare_sequence(
            source_seq=source_seq,
            target_seq=target_seq,
            camera_id=args.camera_id,
            processed_subdir=args.processed_subdir,
            gs_subdir=args.gs_subdir,
            scale_ratio=int(args.scale_ratio),
            crop_size=args.crop_size,
            bbox_expand=float(args.bbox_expand),
            init_gaussian_scale=float(args.init_gaussian_scale),
            overwrite=bool(args.overwrite),
        )
        prepared += 1
        print(
            f"[behave-dual-branch-prep] {sequence_idx}/{len(sequence_names)} "
            f"{sequence_name} | frames={num_frames}"
        )
    print(f"[behave-dual-branch-prep] done | prepared={prepared} | output_root={output_root}")


if __name__ == "__main__":
    main()
