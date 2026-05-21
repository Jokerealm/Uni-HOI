#!/usr/bin/env python3
"""
Prepare the WAI quick-validation subset for DualBranchHOIDataset.

The WAI raw subset intentionally contains only selected BEHAVE timestep
directories.  This script builds a dataloader-ready mirror by:
  - symlinking the selected WAI raw timesteps,
  - slicing BEHAVE precomputed processed arrays to the selected timesteps,
  - generating cropped 256x256 assets,
  - generating object pose and dual-branch target caches,
  - generating SMPL-derived human Gaussian tokens.

By default, large sliced full-resolution intermediates are removed after the
cropped assets are created.  The resulting output keeps the files needed by
DualBranchHOIDataset while staying small enough for fast iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_behave_dual_branch_assets import build_cropped_assets
from backfill_behave_smpl_geometry import build_geometry_arrays
from preprocess_procigen_gt import (
    find_object_name,
    load_mesh_vertices_faces,
    mesh_to_gaussians,
    prepare_dual_branch_target_caches,
)


TRANSIENT_PROCESSED_FILES = (
    "depth_aligned.npz",
    "masks_raw.npz",
    "region_masks.npz",
    "keypoints_2d.npz",
    "joints_3d.npz",
)
SLICED_PROCESSED_FILES = TRANSIENT_PROCESSED_FILES + ("smpl_params.npz",)
GS_FILES = (
    "G_h.pt",
    "G_h_mesh.obj",
)
OBJECT_GAUSSIAN_META_FILENAME = "G_o_wai_metric_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WAI raw subset for DualBranchHOIDataset.")
    parser.add_argument(
        "--wai_root",
        type=str,
        default=str(REPO_ROOT / "sample_data" / "WAI" / "sequences"),
        help="Raw WAI sequence root.",
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default="/data4/guanz/data/Behave/sequences",
        help="Original full BEHAVE sequence root containing processed assets.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(REPO_ROOT / "sample_data" / "WAI_prepared" / "sequences"),
        help="Prepared WAI sequence root.",
    )
    parser.add_argument("--camera_id", type=str, default="k1")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--scale_ratio", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--bbox_expand", type=float, default=1.1)
    parser.add_argument("--init_gaussian_scale", type=float, default=0.01)
    parser.add_argument("--num_object_gaussians", type=int, default=4096)
    parser.add_argument("--min_frames", type=int, default=9)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--sequence_name", action="append", dest="sequence_names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail_on_short_sequence",
        action="store_true",
        help="Raise instead of skipping sequences that cannot provide min_frames prepared frames.",
    )
    parser.add_argument(
        "--keep_fullres_assets",
        action="store_true",
        help="Keep sliced full-resolution processed npz files after cropped assets are generated.",
    )
    parser.add_argument(
        "--compress_base_assets",
        action="store_true",
        help="Use compressed npz for temporary sliced processed assets. Slower, but smaller if retained.",
    )
    parser.add_argument("--dataset_name", type=str, default="WAI_prepared")
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def ensure_symlink(src: Path, dst: Path, *, overwrite: bool) -> None:
    src = src.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)).resolve() == src:
            return
        if not overwrite:
            return
        safe_unlink(dst)
    os.symlink(str(src), str(dst))


def list_timestep_dirs(sequence_dir: Path) -> List[Path]:
    return sorted(path for path in sequence_dir.iterdir() if path.is_dir() and path.name.startswith("t"))


def discover_sequence_names(wai_root: Path, requested: Sequence[str] | None, max_sequences: int) -> List[str]:
    if requested:
        names = sorted(set(str(name) for name in requested))
    else:
        names = sorted(child.name for child in wai_root.iterdir() if child.is_dir())
    if max_sequences > 0:
        names = names[: int(max_sequences)]
    return names


def save_npz(path: Path, arrays: Dict[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def infer_processed_frame_count(source_seq: Path, *, processed_subdir: str) -> int:
    smpl_path = source_seq / processed_subdir / "smpl_params.npz"
    if not smpl_path.is_file():
        raise FileNotFoundError(f"Missing source SMPL params: {smpl_path}")
    with np.load(smpl_path, allow_pickle=False) as payload:
        if "body_pose" not in payload:
            raise KeyError(f"Missing `body_pose` in {smpl_path}")
        return int(payload["body_pose"].shape[0])


def count_prepared_frames(target_seq: Path, *, processed_subdir: str) -> int:
    meta_path = target_seq / processed_subdir / "cropped" / "meta.npz"
    if not meta_path.is_file():
        return 0
    with np.load(meta_path, allow_pickle=False) as payload:
        if "fx" not in payload:
            return 0
        return int(payload["fx"].shape[0])


def target_is_prepared(target_seq: Path, *, processed_subdir: str, gs_subdir: str, min_frames: int) -> bool:
    processed_dir = target_seq / processed_subdir
    cropped_dir = processed_dir / "cropped"
    gs_dir = target_seq / gs_subdir
    required = (
        cropped_dir / "rgb",
        cropped_dir / "masks_raw.npz",
        cropped_dir / "region_masks.npz",
        cropped_dir / "depth_aligned.npz",
        cropped_dir / "keypoints_2d.npz",
        cropped_dir / "meta.npz",
        cropped_dir / "dual_branch_targets.npz",
        processed_dir / "smpl_params.npz",
        processed_dir / "object_poses.npz",
        gs_dir / "G_o.pt",
        gs_dir / OBJECT_GAUSSIAN_META_FILENAME,
        gs_dir / "G_h_smpl.pt",
    )
    if not all(path.exists() for path in required):
        return False
    frame_count = count_prepared_frames(target_seq, processed_subdir=processed_subdir)
    if frame_count < int(min_frames):
        return False
    rgb_count = len(list((cropped_dir / "rgb").glob("*.png")))
    return rgb_count >= int(min_frames)


def slice_npz_file(
    source_path: Path,
    target_path: Path,
    *,
    selected_indices: Sequence[int],
    source_raw_frame_count: int,
    source_processed_frame_count: int,
    overwrite: bool,
    compressed: bool,
) -> Dict[str, object]:
    if target_path.exists() and not overwrite:
        with np.load(target_path) as existing:
            return {"path": str(target_path), "keys": list(existing.files), "status": "exists"}

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source npz: {source_path}")

    indices = np.asarray(selected_indices, dtype=np.int64)
    arrays: Dict[str, np.ndarray] = {}
    sliced_keys: List[str] = []
    copied_keys: List[str] = []
    temporal_lengths = {int(source_raw_frame_count), int(source_processed_frame_count)}
    temporal_lengths.discard(0)
    with np.load(source_path, allow_pickle=False) as payload:
        for key in payload.files:
            value = np.asarray(payload[key])
            if value.ndim >= 1 and int(value.shape[0]) in temporal_lengths:
                if int(indices.max(initial=-1)) >= int(value.shape[0]):
                    raise IndexError(
                        f"Selected index out of range for {source_path}:{key}: "
                        f"max={int(indices.max())}, frames={int(value.shape[0])}"
                    )
                arrays[key] = value[indices]
                sliced_keys.append(key)
            else:
                arrays[key] = value
                copied_keys.append(key)

    if target_path.exists() or target_path.is_symlink():
        safe_unlink(target_path)
    save_npz(target_path, arrays, compressed=compressed)
    return {
        "path": str(target_path),
        "keys": sorted(arrays.keys()),
        "sliced_keys": sorted(sliced_keys),
        "copied_keys": sorted(copied_keys),
        "status": "written",
    }


def link_root_files(source_seq: Path, target_seq: Path, *, overwrite: bool) -> List[str]:
    linked: List[str] = []
    for source_path in sorted(source_seq.iterdir()):
        if not source_path.is_file():
            continue
        target_path = target_seq / source_path.name
        ensure_symlink(source_path, target_path, overwrite=overwrite)
        linked.append(source_path.name)
    return linked


def link_selected_timesteps(wai_seq: Path, target_seq: Path, *, overwrite: bool) -> List[str]:
    names: List[str] = []
    for timestep_dir in list_timestep_dirs(wai_seq):
        ensure_symlink(timestep_dir, target_seq / timestep_dir.name, overwrite=overwrite)
        names.append(timestep_dir.name)
    return names


def link_gaussian_assets(source_seq: Path, target_seq: Path, *, gs_subdir: str, overwrite: bool) -> List[str]:
    linked: List[str] = []
    source_gs_dir = source_seq / gs_subdir
    target_gs_dir = target_seq / gs_subdir
    target_gs_dir.mkdir(parents=True, exist_ok=True)
    for name in GS_FILES:
        source_path = source_gs_dir / name
        if source_path.exists():
            ensure_symlink(source_path, target_gs_dir / name, overwrite=overwrite)
            linked.append(name)
    return linked


def axis_angle_to_matrix(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float32).reshape(3)
    theta = float(np.linalg.norm(angle))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = angle / theta
    x, y, z = [float(value) for value in axis]
    skew = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = np.eye(3, dtype=np.float32) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    return rotation.astype(np.float32)


def load_object_fit_pose(fit_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fit_path.open("rb") as handle:
        fit = pickle.load(handle, encoding="latin1")
    translation = np.asarray(
        fit.get("trans", fit.get("translation", np.zeros(3, dtype=np.float32))),
        dtype=np.float32,
    ).reshape(3)
    if "rot" in fit and fit["rot"] is not None:
        rotation = np.asarray(fit["rot"], dtype=np.float32).reshape(3, 3)
    else:
        rotation = axis_angle_to_matrix(np.asarray(fit.get("angle", np.zeros(3, dtype=np.float32)), dtype=np.float32))
    return rotation.astype(np.float32), translation.astype(np.float32)


def resolve_object_fit_files(timestep_dir: Path) -> tuple[str, Path, Path]:
    object_name = find_object_name(timestep_dir)
    fit_dir = timestep_dir / object_name / "fit01"
    mesh_path = fit_dir / f"{object_name}_fit.ply"
    fit_path = fit_dir / f"{object_name}_fit.pkl"
    if not mesh_path.is_file():
        mesh_candidates = sorted(fit_dir.glob("*_fit.ply"))
        if not mesh_candidates:
            raise FileNotFoundError(f"Missing object fit mesh under {fit_dir}")
        mesh_path = mesh_candidates[0]
    if not fit_path.is_file():
        fit_candidates = sorted(fit_dir.glob("*_fit.pkl"))
        if not fit_candidates:
            raise FileNotFoundError(f"Missing object fit pose under {fit_dir}")
        fit_path = fit_candidates[0]
    return object_name, mesh_path, fit_path


def export_metric_object_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def rebuild_metric_object_gaussians(
    target_seq: Path,
    *,
    gs_subdir: str,
    num_object_gaussians: int,
    init_gaussian_scale: float,
) -> Dict[str, object]:
    timestep_dirs = list_timestep_dirs(target_seq)
    if not timestep_dirs:
        raise RuntimeError(f"No raw timesteps linked under {target_seq}")

    reference_index = len(timestep_dirs) // 2
    reference_timestep = timestep_dirs[reference_index]
    object_name, mesh_path, fit_path = resolve_object_fit_files(reference_timestep)
    object_vertices_world, object_faces = load_mesh_vertices_faces(mesh_path)
    object_rotation, object_translation = load_object_fit_pose(fit_path)
    object_vertices_canonical = (
        object_vertices_world.astype(np.float32) - object_translation.reshape(1, 3)
    ) @ object_rotation

    seed = zlib.adler32(target_seq.name.encode("utf-8")) & 0xFFFFFFFF
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        object_gaussians_payload = mesh_to_gaussians(
            object_vertices_canonical,
            object_faces,
            num_sample_points=int(num_object_gaussians),
            init_gaussian_scale=float(init_gaussian_scale),
        )
    finally:
        np.random.set_state(rng_state)

    gs_dir = target_seq / gs_subdir
    gs_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("G_o.pt", "G_o_mesh.obj", "gs_init_combined.pt"):
        stale_path = gs_dir / stale_name
        if stale_path.exists() or stale_path.is_symlink():
            safe_unlink(stale_path)

    torch.save(object_gaussians_payload, gs_dir / "G_o.pt")
    export_metric_object_mesh(gs_dir / "G_o_mesh.obj", object_vertices_canonical, object_faces)

    xyz = object_gaussians_payload["xyz"].detach().cpu()
    extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).numpy().astype(float).tolist()
    meta = {
        "created_at": iso_now(),
        "source": "wai_raw_fit_mesh",
        "sequence": target_seq.name,
        "object_name": object_name,
        "reference_timestep": reference_timestep.name,
        "reference_index": int(reference_index),
        "mesh_path": str(mesh_path),
        "fit_path": str(fit_path),
        "num_object_gaussians": int(num_object_gaussians),
        "init_gaussian_scale": float(init_gaussian_scale),
        "canonical_xyz_min": xyz.min(dim=0).values.numpy().astype(float).tolist(),
        "canonical_xyz_max": xyz.max(dim=0).values.numpy().astype(float).tolist(),
        "canonical_extent": extent,
    }
    write_json_atomic(gs_dir / OBJECT_GAUSSIAN_META_FILENAME, meta)
    return meta


def cleanup_transient_processed_files(target_seq: Path, *, processed_subdir: str) -> List[str]:
    removed: List[str] = []
    processed_dir = target_seq / processed_subdir
    for name in TRANSIENT_PROCESSED_FILES:
        path = processed_dir / name
        if path.exists() or path.is_symlink():
            safe_unlink(path)
            removed.append(name)
    return removed


def ensure_target_smpl_geometry(
    target_seq: Path,
    *,
    processed_subdir: str,
    compressed: bool,
) -> bool:
    processed_dir = target_seq / processed_subdir
    smpl_path = processed_dir / "smpl_params.npz"
    if not smpl_path.is_file():
        raise FileNotFoundError(f"Missing target SMPL params: {smpl_path}")
    with np.load(smpl_path, allow_pickle=False) as payload:
        smpl_params = {key: np.asarray(payload[key]) for key in payload.files}
    if "vertices" in smpl_params and "faces" in smpl_params and "joints_3d" in smpl_params:
        return False

    timestep_dirs = list_timestep_dirs(target_seq)
    if not timestep_dirs:
        raise RuntimeError(f"No raw timesteps linked under {target_seq}")
    if "body_pose" in smpl_params:
        num_frames = min(int(smpl_params["body_pose"].shape[0]), len(timestep_dirs))
    else:
        num_frames = len(timestep_dirs)
    vertices, faces, joints_3d = build_geometry_arrays([str(path) for path in timestep_dirs], num_frames)
    smpl_params["vertices"] = vertices.astype(np.float32)
    smpl_params["faces"] = faces.astype(np.int32)
    smpl_params["joints_3d"] = joints_3d.astype(np.float32)
    save_npz(smpl_path, smpl_params, compressed=compressed)
    save_npz(processed_dir / "joints_3d.npz", {"joints_3d": joints_3d.astype(np.float32)}, compressed=compressed)
    return True


def prepare_sequence(
    *,
    sequence_name: str,
    wai_root: Path,
    source_root: Path,
    output_root: Path,
    camera_id: str,
    processed_subdir: str,
    gs_subdir: str,
    scale_ratio: int,
    crop_size: Sequence[int],
    bbox_expand: float,
    init_gaussian_scale: float,
    num_object_gaussians: int,
    overwrite: bool,
    min_frames: int,
    fail_on_short_sequence: bool,
    keep_fullres_assets: bool,
    compress_base_assets: bool,
) -> Dict[str, object]:
    wai_seq = wai_root / sequence_name
    source_seq = source_root / sequence_name
    target_seq = output_root / sequence_name
    if not wai_seq.is_dir():
        raise FileNotFoundError(f"Missing WAI sequence: {wai_seq}")
    if not source_seq.is_dir():
        raise FileNotFoundError(f"Missing source BEHAVE sequence: {source_seq}")
    if not overwrite and target_is_prepared(
        target_seq,
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        min_frames=min_frames,
    ):
        frame_count = count_prepared_frames(target_seq, processed_subdir=processed_subdir)
        return {
            "sequence": sequence_name,
            "status": "exists",
            "selected_timestep_count": 0,
            "selected_timesteps": [],
            "selected_source_indices": [],
            "source_timestep_count": 0,
            "source_processed_frame_count": 0,
            "cropped_frames": frame_count,
            "cached_frames": frame_count,
            "target_dir": str(target_seq),
            "removed_transient_processed_files": [],
        }

    selected_timestep_dirs = list_timestep_dirs(wai_seq)
    source_timestep_dirs = list_timestep_dirs(source_seq)
    if not selected_timestep_dirs:
        raise RuntimeError(f"No selected timesteps found under {wai_seq}")
    if not source_timestep_dirs:
        raise RuntimeError(f"No source timesteps found under {source_seq}")

    source_index_by_name = {path.name: idx for idx, path in enumerate(source_timestep_dirs)}
    missing_names = [path.name for path in selected_timestep_dirs if path.name not in source_index_by_name]
    if missing_names:
        raise FileNotFoundError(
            f"{sequence_name} has WAI timesteps not present in source sequence: {missing_names[:5]}"
        )
    selected_indices = [source_index_by_name[path.name] for path in selected_timestep_dirs]
    source_processed_frame_count = infer_processed_frame_count(source_seq, processed_subdir=processed_subdir)
    short_reason = ""
    if len(selected_timestep_dirs) < int(min_frames):
        short_reason = f"selected_timestep_count={len(selected_timestep_dirs)} < min_frames={int(min_frames)}"
    elif max(selected_indices) >= source_processed_frame_count:
        short_reason = (
            f"selected max source index {max(selected_indices)} is outside processed frame count "
            f"{source_processed_frame_count}"
        )
    if short_reason:
        if overwrite and target_seq.exists():
            safe_unlink(target_seq)
        if fail_on_short_sequence:
            raise ValueError(f"{sequence_name} cannot provide a full clip: {short_reason}")
        return {
            "sequence": sequence_name,
            "status": "skipped_short",
            "reason": short_reason,
            "selected_timestep_count": len(selected_timestep_dirs),
            "selected_timesteps": [path.name for path in selected_timestep_dirs],
            "selected_source_indices": selected_indices,
            "source_timestep_count": len(source_timestep_dirs),
            "source_processed_frame_count": source_processed_frame_count,
            "cropped_frames": 0,
            "cached_frames": 0,
            "target_dir": str(target_seq),
        }

    target_seq.mkdir(parents=True, exist_ok=True)
    linked_root_files = link_root_files(wai_seq, target_seq, overwrite=overwrite)
    linked_timesteps = link_selected_timesteps(wai_seq, target_seq, overwrite=overwrite)
    linked_gs_files = link_gaussian_assets(source_seq, target_seq, gs_subdir=gs_subdir, overwrite=overwrite)
    object_gaussian_report = rebuild_metric_object_gaussians(
        target_seq,
        gs_subdir=gs_subdir,
        num_object_gaussians=int(num_object_gaussians),
        init_gaussian_scale=float(init_gaussian_scale),
    )

    processed_dir = target_seq / processed_subdir
    processed_dir.mkdir(parents=True, exist_ok=True)
    npz_reports = []
    for name in SLICED_PROCESSED_FILES:
        source_path = source_seq / processed_subdir / name
        if not source_path.is_file():
            if name == "joints_3d.npz":
                continue
            raise FileNotFoundError(f"Missing required source processed file: {source_path}")
        npz_reports.append(
            slice_npz_file(
                source_path,
                processed_dir / name,
                selected_indices=selected_indices,
                source_raw_frame_count=len(source_timestep_dirs),
                source_processed_frame_count=source_processed_frame_count,
                overwrite=overwrite,
                compressed=compress_base_assets,
            )
        )

    backfilled_smpl_geometry = ensure_target_smpl_geometry(
        target_seq,
        processed_subdir=processed_subdir,
        compressed=compress_base_assets,
    )
    num_frames = build_cropped_assets(
        source_seq=target_seq,
        target_seq=target_seq,
        camera_id=camera_id,
        processed_subdir=processed_subdir,
        scale_ratio=int(scale_ratio),
        crop_size=crop_size,
        bbox_expand=float(bbox_expand),
        overwrite=overwrite,
    )
    cached_frames = prepare_dual_branch_target_caches(
        target_seq,
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        init_gaussian_scale=float(init_gaussian_scale),
    )
    if int(num_frames) < int(min_frames) or int(cached_frames) < int(min_frames):
        short_reason = f"prepared frames {int(num_frames)}/{int(cached_frames)} < min_frames={int(min_frames)}"
        if overwrite and target_seq.exists():
            safe_unlink(target_seq)
        if fail_on_short_sequence:
            raise ValueError(f"{sequence_name} cannot provide a full clip: {short_reason}")
        return {
            "sequence": sequence_name,
            "status": "skipped_short",
            "reason": short_reason,
            "selected_timestep_count": len(selected_timestep_dirs),
            "selected_timesteps": linked_timesteps,
            "selected_source_indices": selected_indices,
            "source_timestep_count": len(source_timestep_dirs),
            "source_processed_frame_count": source_processed_frame_count,
            "cropped_frames": int(num_frames),
            "cached_frames": int(cached_frames),
            "target_dir": str(target_seq),
        }

    removed_files: List[str] = []
    if not keep_fullres_assets:
        removed_files = cleanup_transient_processed_files(target_seq, processed_subdir=processed_subdir)

    return {
        "sequence": sequence_name,
        "status": "prepared",
        "selected_timestep_count": len(selected_timestep_dirs),
        "selected_timesteps": linked_timesteps,
        "selected_source_indices": selected_indices,
        "source_timestep_count": len(source_timestep_dirs),
        "source_processed_frame_count": source_processed_frame_count,
        "cropped_frames": int(num_frames),
        "cached_frames": int(cached_frames),
        "target_dir": str(target_seq),
        "linked_root_files": linked_root_files,
        "linked_gs_files": linked_gs_files,
        "object_gaussian": object_gaussian_report,
        "sliced_npz": npz_reports,
        "backfilled_smpl_geometry": bool(backfilled_smpl_geometry),
        "removed_transient_processed_files": removed_files,
    }


def write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    wai_root = Path(args.wai_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_names = discover_sequence_names(wai_root, args.sequence_names, int(args.max_sequences))
    if not sequence_names:
        raise RuntimeError(f"No WAI sequences found under {wai_root}")

    print(
        f"[wai-prep] start | sequences={len(sequence_names)} | wai_root={wai_root} "
        f"| source_root={source_root} | output_root={output_root}",
        flush=True,
    )
    reports: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for sequence_idx, sequence_name in enumerate(sequence_names, start=1):
        try:
            report = prepare_sequence(
                sequence_name=sequence_name,
                wai_root=wai_root,
                source_root=source_root,
                output_root=output_root,
                camera_id=args.camera_id,
                processed_subdir=args.processed_subdir,
                gs_subdir=args.gs_subdir,
                scale_ratio=int(args.scale_ratio),
                crop_size=args.crop_size,
                bbox_expand=float(args.bbox_expand),
                init_gaussian_scale=float(args.init_gaussian_scale),
                num_object_gaussians=int(args.num_object_gaussians),
                overwrite=bool(args.overwrite),
                min_frames=int(args.min_frames),
                fail_on_short_sequence=bool(args.fail_on_short_sequence),
                keep_fullres_assets=bool(args.keep_fullres_assets),
                compress_base_assets=bool(args.compress_base_assets),
            )
            reports.append(report)
            if report["status"] == "skipped_short":
                print(
                    f"[wai-prep] {sequence_idx}/{len(sequence_names)} {sequence_name} "
                    f"| skipped_short | {report['reason']}",
                    flush=True,
                )
            else:
                print(
                    f"[wai-prep] {sequence_idx}/{len(sequence_names)} {sequence_name} "
                    f"| frames={report['cropped_frames']} | transient_removed="
                    f"{len(report['removed_transient_processed_files'])}",
                    flush=True,
                )
        except Exception as exc:
            failure = {"sequence": sequence_name, "status": "failed", "error": repr(exc)}
            failures.append(failure)
            print(f"[wai-prep] FAILED {sequence_idx}/{len(sequence_names)} {sequence_name}: {exc}", flush=True)
            raise

    manifest = {
        "dataset_name": args.dataset_name,
        "created_at": iso_now(),
        "wai_root": str(wai_root),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "camera_id": args.camera_id,
        "processed_subdir": args.processed_subdir,
        "gs_subdir": args.gs_subdir,
        "scale_ratio": int(args.scale_ratio),
        "crop_size": [int(args.crop_size[0]), int(args.crop_size[1])],
        "bbox_expand": float(args.bbox_expand),
        "num_object_gaussians": int(args.num_object_gaussians),
        "min_frames": int(args.min_frames),
        "keep_fullres_assets": bool(args.keep_fullres_assets),
        "sequence_count": int(sum(1 for item in reports if item["status"] in {"prepared", "exists"})),
        "skipped_short_count": int(sum(1 for item in reports if item["status"] == "skipped_short")),
        "total_cropped_frames": int(
            sum(int(item["cropped_frames"]) for item in reports if item["status"] in {"prepared", "exists"})
        ),
        "failures": failures,
        "sequences": reports,
    }
    manifest_path = output_root.parent / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(
        f"[wai-prep] done | prepared={manifest['sequence_count']} "
        f"| skipped_short={manifest['skipped_short_count']} | frames={manifest['total_cropped_frames']} "
        f"| manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
