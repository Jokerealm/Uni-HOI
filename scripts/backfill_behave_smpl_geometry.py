#!/usr/bin/env python3
"""
Backfill BEHAVE Step-1 SMPL geometry into an existing `processed/smpl_params.npz`.

This is a lightweight repair utility for sequences that already finished Step 1
but were generated before `vertices / faces / joints_3d` were exported.
It reads the raw BEHAVE sequence only for:
  - `person/fit02/person_fit.ply` or `person/person.ply`
  - `person/person_J3d.json`
  - `person/fit02/person_fit.pkl` (only if `smpl_params.npz` is missing)

By default it patches one sequence in place and writes a `.bak.npz` backup of
the previous archive before saving the new one.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.behave_gt_loader import _load_person_joints3d, _load_person_mesh


def resolve_sequence_dir(input_dir: str, video_name: str) -> Path:
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


def discover_sequence_dirs(root: str, video_names: Optional[List[str]]) -> List[Path]:
    if video_names:
        return [resolve_sequence_dir(root, video_name) for video_name in video_names]

    root_path = Path(root).expanduser().resolve()
    if glob.glob(str(root_path / "t*.000")):
        return [root_path]

    candidates = [root_path]
    if (root_path / "sequences").is_dir():
        candidates.append(root_path / "sequences")

    discovered: List[Path] = []
    for base in candidates:
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if glob.glob(str(child / "t*.000")):
                discovered.append(child)
    if not discovered:
        raise FileNotFoundError(f"No BEHAVE sequences found under {root_path}")
    return discovered


def infer_temporal_length(existing: Dict[str, np.ndarray], timesteps: List[str]) -> int:
    for key in ("body_pose", "cam_t", "shape", "vertices", "joints_3d"):
        if key in existing and existing[key].ndim > 0:
            return int(existing[key].shape[0])
    return len(timesteps)


def load_or_initialize_existing_smpl(smpl_path: Path, timesteps: List[str], num_frames: int) -> Dict[str, np.ndarray]:
    if smpl_path.is_file():
        existing = {key: value for key, value in np.load(smpl_path).items()}
        for key, value in existing.items():
            if value.ndim > 0 and value.shape[0] == 0:
                raise ValueError(f"Found empty array for `{key}` in {smpl_path}")
        return existing

    body_pose = np.zeros((num_frames, 72), dtype=np.float32)
    shape = np.zeros((num_frames, 10), dtype=np.float32)
    cam_t = np.zeros((num_frames, 3), dtype=np.float32)
    for frame_idx, t_dir in enumerate(timesteps[:num_frames]):
        fit_path = os.path.join(t_dir, "person", "fit02", "person_fit.pkl")
        if not os.path.isfile(fit_path):
            continue
        with open(fit_path, "rb") as f:
            smpl = pickle.load(f, encoding="latin1")
        body_pose[frame_idx] = np.asarray(smpl.get("pose", np.zeros(72)), dtype=np.float32)
        shape[frame_idx] = np.asarray(smpl.get("betas", np.zeros(10)), dtype=np.float32)
        cam_t[frame_idx] = np.asarray(smpl.get("trans", np.zeros(3)), dtype=np.float32)
    return {
        "body_pose": body_pose,
        "shape": shape,
        "cam_t": cam_t,
    }


def build_geometry_arrays(timesteps: List[str], num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_list: List[np.ndarray] = []
    joints_list: List[np.ndarray] = []
    canonical_faces = None

    max_joints = 0
    for t_dir in timesteps[:num_frames]:
        vertices, faces = _load_person_mesh(t_dir)
        joints = _load_person_joints3d(t_dir).astype(np.float32)

        if canonical_faces is None:
            canonical_faces = faces.astype(np.int32)
        elif canonical_faces.shape != faces.shape or not np.array_equal(canonical_faces, faces):
            raise ValueError(
                f"Mesh topology changed across frames under {t_dir}. Expected constant faces."
            )

        vertices_list.append(vertices.astype(np.float32))
        joints_list.append(joints)
        max_joints = max(max_joints, joints.shape[0])

    if not vertices_list:
        raise RuntimeError("No BEHAVE geometry could be loaded.")

    vertices = np.stack(vertices_list, axis=0).astype(np.float32)
    joints_3d = np.zeros((len(joints_list), max_joints, 3), dtype=np.float32)
    for idx, joints in enumerate(joints_list):
        joints_3d[idx, : joints.shape[0]] = joints
    return vertices, canonical_faces, joints_3d


def patch_sequence(
    seq_dir: Path,
    *,
    processed_subdir: str,
    backup: bool,
    dry_run: bool,
) -> Path:
    processed_dir = seq_dir / processed_subdir
    smpl_path = processed_dir / "smpl_params.npz"
    joints_path = processed_dir / "joints_3d.npz"

    timesteps = sorted(glob.glob(str(seq_dir / "t*.000")))
    if not timesteps:
        raise FileNotFoundError(f"No BEHAVE timesteps found under {seq_dir}")

    existing = load_or_initialize_existing_smpl(smpl_path, timesteps, len(timesteps))
    num_frames = infer_temporal_length(existing, timesteps)
    if num_frames > len(timesteps):
        raise ValueError(
            f"{smpl_path} expects {num_frames} frames, but only found {len(timesteps)} BEHAVE timesteps."
        )

    vertices, faces, joints_3d = build_geometry_arrays(timesteps, num_frames)
    patched = dict(existing)
    patched["vertices"] = vertices
    patched["faces"] = faces.astype(np.int32)
    patched["joints_3d"] = joints_3d.astype(np.float32)

    if dry_run:
        print(
            f"[backfill] {seq_dir.name}: would write vertices={vertices.shape}, "
            f"faces={faces.shape}, joints_3d={joints_3d.shape} -> {smpl_path}"
        )
        return smpl_path

    processed_dir.mkdir(parents=True, exist_ok=True)
    if backup and smpl_path.is_file():
        backup_path = smpl_path.with_suffix(smpl_path.suffix + ".bak")
        shutil.copy2(smpl_path, backup_path)
        print(f"[backfill] Backup -> {backup_path}")

    np.savez_compressed(smpl_path, **patched)
    np.savez_compressed(joints_path, joints_3d=joints_3d.astype(np.float32))
    print(
        f"[backfill] Patched {seq_dir.name}: "
        f"vertices={vertices.shape}, faces={faces.shape}, joints_3d={joints_3d.shape}"
    )
    return smpl_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill BEHAVE SMPL geometry into existing Step-1 outputs.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--video-name", action="append", dest="video_names")
    parser.add_argument("--processed-subdir", type=str, default="processed")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    sequence_dirs = discover_sequence_dirs(args.input_dir, args.video_names)
    for seq_dir in sequence_dirs:
        patch_sequence(
            seq_dir,
            processed_subdir=args.processed_subdir,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
