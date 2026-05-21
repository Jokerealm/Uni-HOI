#!/usr/bin/env python3
"""Create a WAI-style heldout BEHAVE subset for independent evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/data4/guanz/data/Behave/sequences")
DEFAULT_DEST_ROOT = REPO_ROOT / "sample_data" / "BEHAVE_heldout" / "sequences"
DEFAULT_WAI_RAW_ROOT = REPO_ROOT / "sample_data" / "WAI" / "sequences"
DEFAULT_WAI_PREPARED_MANIFEST = REPO_ROOT / "sample_data" / "WAI_prepared" / "manifest.json"
DEFAULT_TARGET_SEQUENCES = 80
DEFAULT_TIMESTEPS_PER_SEQUENCE = 9


GENERATED_DIRS = {
    ".dual_branch_index_cache",
    "amodal",
    "amodal_smoke",
    "frames",
    "gs_aligned",
    "gs_init",
    "gs_init_aligned",
    "gs_init_smoke",
    "joint_opt",
    "processed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an independent BEHAVE heldout raw subset using the same "
            "sequence-level and contiguous-clip style as scripts/create_wai_dataset.py."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--wai-raw-root", type=Path, default=DEFAULT_WAI_RAW_ROOT)
    parser.add_argument("--wai-prepared-manifest", type=Path, default=DEFAULT_WAI_PREPARED_MANIFEST)
    parser.add_argument(
        "--target-sequences",
        type=int,
        default=DEFAULT_TARGET_SEQUENCES,
        help="Heldout sequence count to sample after excluding WAI. Default is about 50%% of WAI.",
    )
    parser.add_argument(
        "--sequence-ratio",
        type=float,
        default=0.0,
        help="Optional fraction of eligible non-WAI sequences. Overrides --target-sequences when > 0.",
    )
    parser.add_argument(
        "--timesteps-per-sequence",
        type=int,
        default=DEFAULT_TIMESTEPS_PER_SEQUENCE,
        help="Contiguous raw timestep count copied per selected sequence.",
    )
    parser.add_argument("--processed-subdir", type=str, default="processed")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Remove destination sequences before recreating them.",
    )
    return parser.parse_args()


def list_sequence_dirs(source_root: Path) -> List[Path]:
    return sorted(path for path in source_root.iterdir() if path.is_dir())


def list_timestep_dirs(sequence_dir: Path) -> List[Path]:
    return sorted(path for path in sequence_dir.iterdir() if path.is_dir() and path.name.startswith("t"))


def select_uniform(items: Sequence[Path], count: int) -> List[Path]:
    items = list(items)
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    selected: List[Path] = []
    seen = set()
    for index in indices:
        index = min(max(int(index), 0), len(items) - 1)
        if index in seen:
            continue
        seen.add(index)
        selected.append(items[index])
    cursor = 0
    while len(selected) < count and cursor < len(items):
        if cursor not in seen:
            selected.append(items[cursor])
            seen.add(cursor)
        cursor += 1
    return sorted(selected)


def select_contiguous_clip(items: Sequence[Path], count: int) -> List[Path]:
    items = list(items)
    if count <= 0:
        return []
    if len(items) < count:
        raise ValueError(f"Need at least {count} timesteps, got {len(items)}.")
    start = (len(items) - count) // 2
    return items[start : start + count]


def safe_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def copy_root_files(source_seq: Path, dest_seq: Path) -> List[str]:
    copied = []
    for entry in sorted(source_seq.iterdir()):
        if entry.is_file() or entry.is_symlink():
            copy_item(entry, dest_seq / entry.name)
            copied.append(entry.name)
    return copied


def compute_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_file():
                total += path.stat().st_size
    return total


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_wai_exclusions(wai_raw_root: Path, wai_prepared_manifest: Path) -> Set[str]:
    excluded: Set[str] = set()
    if wai_raw_root.is_dir():
        excluded.update(child.name for child in wai_raw_root.iterdir() if child.is_dir())
    sequences_txt = wai_raw_root.parent / "sequences.txt"
    if sequences_txt.is_file():
        excluded.update(line.strip() for line in sequences_txt.read_text(encoding="utf-8").splitlines() if line.strip())
    if wai_prepared_manifest.is_file():
        with wai_prepared_manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload.get("sequences", []):
            name = item.get("sequence") or item.get("name")
            if name:
                excluded.add(str(name))
    return excluded


def infer_processed_frame_count(sequence_dir: Path, processed_subdir: str) -> int:
    smpl_path = sequence_dir / processed_subdir / "smpl_params.npz"
    if not smpl_path.is_file():
        return 0
    try:
        with np.load(smpl_path, allow_pickle=False) as payload:
            if "body_pose" in payload:
                return int(payload["body_pose"].shape[0])
    except Exception:
        return 0
    return 0


def sequence_is_eligible(sequence_dir: Path, processed_subdir: str, timesteps_per_sequence: int) -> bool:
    timestep_count = len(list_timestep_dirs(sequence_dir))
    processed_count = infer_processed_frame_count(sequence_dir, processed_subdir)
    return min(timestep_count, processed_count) >= int(timesteps_per_sequence)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    wai_raw_root = args.wai_raw_root.expanduser().resolve()
    wai_prepared_manifest = args.wai_prepared_manifest.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")
    if args.timesteps_per_sequence <= 0:
        raise ValueError("--timesteps-per-sequence must be positive.")
    if args.sequence_ratio < 0.0 or args.sequence_ratio > 1.0:
        raise ValueError("--sequence-ratio must be in [0, 1].")

    all_sequence_dirs = list_sequence_dirs(source_root)
    if not all_sequence_dirs:
        raise RuntimeError(f"No source sequences found under {source_root}")

    excluded_names = load_wai_exclusions(wai_raw_root, wai_prepared_manifest)
    eligible_dirs = [
        seq
        for seq in all_sequence_dirs
        if seq.name not in excluded_names
        and sequence_is_eligible(seq, args.processed_subdir, int(args.timesteps_per_sequence))
    ]
    if not eligible_dirs:
        raise RuntimeError("No eligible non-WAI BEHAVE sequences found.")
    if args.sequence_ratio > 0.0:
        selected_sequence_count = math.ceil(len(eligible_dirs) * float(args.sequence_ratio))
    else:
        selected_sequence_count = int(args.target_sequences)
    selected_sequence_count = min(max(selected_sequence_count, 1), len(eligible_dirs))
    sequence_dirs = select_uniform(eligible_dirs, selected_sequence_count)

    if args.refresh and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": "BEHAVE_heldout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "wai_raw_root": str(wai_raw_root),
        "wai_prepared_manifest": str(wai_prepared_manifest),
        "excluded_wai_sequence_count": len(excluded_names),
        "total_source_sequences": len(all_sequence_dirs),
        "eligible_non_wai_sequences": len(eligible_dirs),
        "target_sequences": int(args.target_sequences),
        "sequence_ratio_requested": float(args.sequence_ratio),
        "timesteps_per_sequence": int(args.timesteps_per_sequence),
        "processed_subdir": args.processed_subdir,
        "excluded_generated_dirs": sorted(GENERATED_DIRS),
        "sequences": [],
    }

    total_copied_timesteps = 0
    for seq_idx, source_seq in enumerate(sequence_dirs, start=1):
        dest_seq = dest_root / source_seq.name
        if args.refresh and dest_seq.exists():
            shutil.rmtree(dest_seq)
        dest_seq.mkdir(parents=True, exist_ok=True)

        timestep_dirs = list_timestep_dirs(source_seq)
        processed_count = infer_processed_frame_count(source_seq, args.processed_subdir)
        usable_timestep_dirs = timestep_dirs[:processed_count]
        sampled_timestep_dirs = select_contiguous_clip(usable_timestep_dirs, int(args.timesteps_per_sequence))
        root_files = copy_root_files(source_seq, dest_seq)
        for timestep_dir in sampled_timestep_dirs:
            target = dest_seq / timestep_dir.name
            if target.exists() or target.is_symlink():
                safe_remove(target)
            copy_item(timestep_dir, target)

        total_copied_timesteps += len(sampled_timestep_dirs)
        seq_size_mb = compute_size_bytes(dest_seq) / (1024 * 1024)
        manifest["sequences"].append(
            {
                "name": source_seq.name,
                "source_timesteps": len(timestep_dirs),
                "source_processed_frames": processed_count,
                "sampled_timesteps": [path.name for path in sampled_timestep_dirs],
                "sampled_count": len(sampled_timestep_dirs),
                "root_files": root_files,
                "size_mb": round(seq_size_mb, 2),
            }
        )
        print(
            f"[behave-heldout] {seq_idx:03d}/{len(sequence_dirs):03d} "
            f"{source_seq.name} | sampled={len(sampled_timestep_dirs)}/"
            f"{len(timestep_dirs)} | processed={processed_count} | size={seq_size_mb:.2f} MB",
            flush=True,
        )

    total_size_mb = compute_size_bytes(dest_root) / (1024 * 1024)
    manifest.update(
        {
            "total_sampled_sequences": len(sequence_dirs),
            "total_sampled_timesteps": total_copied_timesteps,
            "total_size_mb": round(total_size_mb, 2),
        }
    )

    manifest_path = dest_root.parent / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    write_lines(dest_root.parent / "sequences.txt", (item["name"] for item in manifest["sequences"]))

    print(f"[behave-heldout] wrote manifest: {manifest_path}")
    print(
        f"[behave-heldout] done | sequences={len(sequence_dirs)} "
        f"| timesteps={total_copied_timesteps} | size={total_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
