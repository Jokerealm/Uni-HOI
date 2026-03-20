#!/usr/bin/env python3
"""Create a raw-only ~1% BEHAVE sample dataset for fast end-to-end testing."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/data4/guanz/data/Behave/sequences")
DEFAULT_DEST_ROOT = Path("/data4/guanz/coding/HDM/sample_data/behave_1pct/sequences")
DEFAULT_SEQUENCES = [
    "Date03_Sub03_chairblack_sitstand",
    "Date07_Sub08_trashbin",
    "Date07_Sub08_toolbox",
    "Date04_Sub05_monitor_sit",
    "Date07_Sub04_basketball",
    "Date05_Sub06_plasticcontainer",
    "Date03_Sub04_yogaball_play",
    "Date03_Sub04_plasticcontainer_lift",
    "Date01_Sub01_chairwood_hand",
    "Date01_Sub01_trashbin",
    "Date07_Sub08_plasticcontainer",
    "Date05_Sub06_yogaball_sit",
]
GENERATED_DIRS = {
    "frames",
    "processed",
    "amodal",
    "gs_init",
    "gs_init_aligned",
    "gs_aligned",
    "joint_opt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    p.add_argument("--sequence", action="append", dest="sequences")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Remove the destination sequence directory before recopying raw data.",
    )
    return p.parse_args()


def remove_generated_outputs(seq_dir: Path) -> list[str]:
    removed = []
    for name in sorted(GENERATED_DIRS):
        target = seq_dir / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return removed


def copy_sequence(src_seq: Path, dst_seq: Path, refresh: bool) -> dict:
    if refresh and dst_seq.exists():
        shutil.rmtree(dst_seq)
    dst_seq.mkdir(parents=True, exist_ok=True)

    copied_items = []
    for entry in sorted(src_seq.iterdir()):
        if entry.name in GENERATED_DIRS:
            continue
        if entry.is_dir() and not entry.name.startswith("t"):
            continue
        dst = dst_seq / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst)
        copied_items.append(entry.name)

    removed = remove_generated_outputs(dst_seq)
    return {"copied": copied_items, "removed_generated": removed}


def compute_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_file():
                total += path.stat().st_size
    return total


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    dest_root = args.dest_root
    sequences = args.sequences or DEFAULT_SEQUENCES

    dest_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "sequences": [],
        "excluded_dirs": sorted(GENERATED_DIRS),
    }

    for seq_name in sequences:
        src_seq = source_root / seq_name
        if not src_seq.is_dir():
            raise FileNotFoundError(f"Missing source sequence: {src_seq}")
        dst_seq = dest_root / seq_name
        result = copy_sequence(src_seq, dst_seq, refresh=args.refresh)
        seq_size_mb = compute_size_bytes(dst_seq) / (1024 * 1024)
        manifest["sequences"].append({
            "name": seq_name,
            "size_mb": round(seq_size_mb, 2),
            **result,
        })
        print(f"[sample] {seq_name}: {seq_size_mb:.2f} MB")

    total_bytes = compute_size_bytes(dest_root)
    manifest["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    manifest_path = dest_root.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[sample] Wrote manifest: {manifest_path}")
    print(f"[sample] Total size: {manifest['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
