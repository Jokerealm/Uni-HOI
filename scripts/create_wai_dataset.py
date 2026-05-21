#!/usr/bin/env python3
"""Create the WAI raw BEHAVE subset with trainable clip-length snippets."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


DEFAULT_SOURCE_ROOT = Path("/data4/guanz/data/Behave/sequences")
DEFAULT_DEST_ROOT = Path("/data4/guanz/coding/HDM/sample_data/WAI/sequences")
DEFAULT_SEQUENCE_RATIO = 0.5
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
            "Build the WAI raw dataset by sampling a subset of BEHAVE sequences "
            "and copying one contiguous trainable clip from each selected sequence."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument(
        "--sequence-ratio",
        type=float,
        default=DEFAULT_SEQUENCE_RATIO,
        help="Fraction of source sequences to include.",
    )
    parser.add_argument(
        "--timesteps-per-sequence",
        type=int,
        default=DEFAULT_TIMESTEPS_PER_SEQUENCE,
        help="Contiguous raw timestep count copied per selected sequence.",
    )
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


def select_uniform(items: List[Path], count: int) -> List[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    selected = []
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


def select_contiguous_clip(items: List[Path], count: int) -> List[Path]:
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


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")
    if not (0.0 < args.sequence_ratio <= 1.0):
        raise ValueError("--sequence-ratio must be in (0, 1].")
    if args.timesteps_per_sequence <= 0:
        raise ValueError("--timesteps-per-sequence must be positive.")

    all_sequence_dirs = list_sequence_dirs(source_root)
    if not all_sequence_dirs:
        raise RuntimeError(f"No source sequences found under {source_root}")

    selected_sequence_count = math.ceil(len(all_sequence_dirs) * float(args.sequence_ratio))
    sequence_dirs = select_uniform(all_sequence_dirs, selected_sequence_count)
    timestep_dirs_by_sequence = {seq.name: list_timestep_dirs(seq) for seq in all_sequence_dirs}
    sequence_timesteps = {name: len(paths) for name, paths in timestep_dirs_by_sequence.items()}
    short_selected_sequences = [
        seq.name
        for seq in sequence_dirs
        if sequence_timesteps[seq.name] < int(args.timesteps_per_sequence)
    ]
    if short_selected_sequences:
        raise RuntimeError(
            f"Selected sequences shorter than --timesteps-per-sequence={args.timesteps_per_sequence}: "
            f"{short_selected_sequences[:5]}"
        )

    if args.refresh and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": "WAI",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "sequence_ratio_requested": float(args.sequence_ratio),
        "timesteps_per_sequence": int(args.timesteps_per_sequence),
        "excluded_generated_dirs": sorted(GENERATED_DIRS),
        "sequences": [],
    }

    total_copied_timesteps = 0
    for seq_idx, source_seq in enumerate(sequence_dirs, start=1):
        dest_seq = dest_root / source_seq.name
        if args.refresh and dest_seq.exists():
            shutil.rmtree(dest_seq)
        dest_seq.mkdir(parents=True, exist_ok=True)

        root_files = copy_root_files(source_seq, dest_seq)
        sampled_timestep_dirs = select_contiguous_clip(
            timestep_dirs_by_sequence[source_seq.name],
            int(args.timesteps_per_sequence),
        )
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
                "source_timesteps": sequence_timesteps[source_seq.name],
                "sampled_timesteps": [path.name for path in sampled_timestep_dirs],
                "sampled_count": len(sampled_timestep_dirs),
                "root_files": root_files,
                "size_mb": round(seq_size_mb, 2),
            }
        )
        print(
            f"[WAI] {seq_idx:03d}/{len(sequence_dirs):03d} "
            f"{source_seq.name} | sampled={len(sampled_timestep_dirs)}/"
            f"{sequence_timesteps[source_seq.name]} | size={seq_size_mb:.2f} MB",
            flush=True,
        )

    total_source_timesteps = sum(sequence_timesteps.values())
    total_size_mb = compute_size_bytes(dest_root) / (1024 * 1024)
    manifest.update(
        {
            "total_source_sequences": len(all_sequence_dirs),
            "total_sampled_sequences": len(sequence_dirs),
            "actual_sequence_ratio": len(sequence_dirs) / max(len(all_sequence_dirs), 1),
            "total_source_timesteps": total_source_timesteps,
            "total_sampled_timesteps": total_copied_timesteps,
            "actual_timestep_ratio": total_copied_timesteps / max(total_source_timesteps, 1),
            "total_size_mb": round(total_size_mb, 2),
        }
    )

    manifest_path = dest_root.parent / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    write_lines(dest_root.parent / "sequences.txt", (item["name"] for item in manifest["sequences"]))

    print(f"[WAI] wrote manifest: {manifest_path}")
    print(
        f"[WAI] done | sequences={len(sequence_dirs)}/{len(all_sequence_dirs)} "
        f"| timesteps={total_copied_timesteps}/{total_source_timesteps} "
        f"({manifest['actual_timestep_ratio'] * 100:.2f}%) "
        f"| size={total_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
