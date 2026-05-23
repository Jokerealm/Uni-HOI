#!/usr/bin/env python3
"""Build BEHAVE-style dense contact caches for prepared sequences.

This follows the official BEHAVE `compute_contacts.py` recipe: sample points on
the posed object surface, query signed distance to the fitted SMPL mesh, and
mark object samples whose signed distance is below a threshold as contacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dual_branch_fm_dataset import _discover_timestep_dirs
from pipeline.behave_gt_loader import _read_ply_mesh

DENSE_CONTACT_CACHE_VERSION = 1
DENSE_CONTACTS_FILENAME = "dense_contacts.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BEHAVE dense contact caches.")
    parser.add_argument("--data_root", type=str, required=True, help="Prepared sequence root.")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--output_filename", type=str, default=DENSE_CONTACTS_FILENAME)
    parser.add_argument("--sequence_name", action="append", dest="sequence_names")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--contact_threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--write_frame_contacts", action="store_true")
    return parser.parse_args()


def discover_sequences(data_root: Path, sequence_names: Optional[Sequence[str]], max_sequences: int) -> List[Path]:
    if sequence_names:
        sequences = [data_root / name for name in sequence_names]
    else:
        sequences = sorted(path for path in data_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if max_sequences > 0:
        sequences = sequences[:max_sequences]
    missing = [str(path) for path in sequences if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing sequence directories: {missing[:5]}")
    return sequences


def load_mesh(path: Path) -> trimesh.Trimesh:
    vertices, faces = _read_ply_mesh(str(path))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def resolve_frame_meshes(timestep_dir: Path) -> Tuple[Path, Path]:
    smpl_mesh = timestep_dir / "person" / "fit02" / "person_fit.ply"
    if not smpl_mesh.is_file():
        raise FileNotFoundError(f"Missing SMPL fit mesh: {smpl_mesh}")

    object_meshes = sorted(timestep_dir.glob("*/fit01/*_fit.ply"))
    object_meshes = [path for path in object_meshes if "/person/" not in str(path)]
    if not object_meshes:
        raise FileNotFoundError(f"Missing object fit mesh under {timestep_dir}")
    return smpl_mesh, object_meshes[0]


def compute_contact_labels(
    smpl_mesh: trimesh.Trimesh,
    object_mesh: trimesh.Trimesh,
    *,
    num_samples: int,
    threshold: float,
) -> Dict[str, np.ndarray]:
    import igl

    object_points = object_mesh.sample(int(num_samples)).astype(np.float64)
    signed_output = igl.signed_distance(
        object_points,
        np.asarray(smpl_mesh.vertices, dtype=np.float64),
        np.asarray(smpl_mesh.faces, dtype=np.int64),
    )
    signed_distances, _, closest_points = signed_output[:3]
    signed_distances = np.asarray(signed_distances, dtype=np.float32)
    closest_points = np.asarray(closest_points, dtype=np.float32)
    contact_labels = signed_distances < float(threshold)
    return {
        "object_points": object_points.astype(np.float32),
        "contact_labels": contact_labels.astype(np.bool_),
        "contact_human_points": closest_points,
        "signed_distances": signed_distances,
    }


def maybe_write_official_frame_contact(frame_output_path: Path, frame_contact: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        frame_output_path,
        object_points=frame_contact["object_points"],
        contact_label=frame_contact["contact_labels"],
        contact_vertices=frame_contact["contact_human_points"],
        signed_distances=frame_contact["signed_distances"],
    )


def build_sequence_cache(
    sequence_dir: Path,
    *,
    processed_subdir: str,
    output_filename: str,
    num_samples: int,
    threshold: float,
    seed: int,
    redo: bool,
    write_frame_contacts: bool,
) -> Dict[str, object]:
    output_path = sequence_dir / processed_subdir / "cropped" / output_filename
    if output_path.is_file() and not redo:
        return {"sequence": sequence_dir.name, "status": "skipped", "path": str(output_path)}

    timestep_dirs = _discover_timestep_dirs(sequence_dir)
    if not timestep_dirs:
        raise FileNotFoundError(f"No BEHAVE timestep directories found under {sequence_dir}")

    object_points: List[np.ndarray] = []
    contact_labels: List[np.ndarray] = []
    contact_human_points: List[np.ndarray] = []
    signed_distances: List[np.ndarray] = []
    frame_names: List[str] = []

    np.random.seed(int(seed))
    for frame_idx, timestep_dir in enumerate(timestep_dirs):
        smpl_path, object_path = resolve_frame_meshes(timestep_dir)
        frame_contact = compute_contact_labels(
            load_mesh(smpl_path),
            load_mesh(object_path),
            num_samples=int(num_samples),
            threshold=float(threshold),
        )
        object_points.append(frame_contact["object_points"])
        contact_labels.append(frame_contact["contact_labels"])
        contact_human_points.append(frame_contact["contact_human_points"])
        signed_distances.append(frame_contact["signed_distances"])
        frame_names.append(timestep_dir.name)

        if write_frame_contacts:
            frame_output = object_path.with_name(object_path.stem + "_contact.npz")
            maybe_write_official_frame_contact(frame_output, frame_contact)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.asarray([DENSE_CONTACT_CACHE_VERSION], dtype=np.int32),
        num_samples=np.asarray([int(num_samples)], dtype=np.int32),
        contact_threshold=np.asarray([float(threshold)], dtype=np.float32),
        frame_names=np.asarray(frame_names),
        object_points=np.stack(object_points, axis=0).astype(np.float32),
        contact_labels=np.stack(contact_labels, axis=0).astype(np.bool_),
        contact_human_points=np.stack(contact_human_points, axis=0).astype(np.float32),
        signed_distances=np.stack(signed_distances, axis=0).astype(np.float32),
    )
    return {
        "sequence": sequence_dir.name,
        "status": "built",
        "frames": len(frame_names),
        "samples_per_frame": int(num_samples),
        "path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    sequences = discover_sequences(data_root, args.sequence_names, args.max_sequences)
    reports = []
    for index, sequence_dir in enumerate(sequences, start=1):
        report = build_sequence_cache(
            sequence_dir,
            processed_subdir=args.processed_subdir,
            output_filename=args.output_filename,
            num_samples=args.num_samples,
            threshold=args.contact_threshold,
            seed=args.seed + index,
            redo=args.redo,
            write_frame_contacts=args.write_frame_contacts,
        )
        reports.append(report)
        print(f"[dense-contact] {index}/{len(sequences)} {report}", flush=True)
    print(json.dumps({"total": len(reports), "built": sum(r["status"] == "built" for r in reports)}, indent=2))


if __name__ == "__main__":
    main()
