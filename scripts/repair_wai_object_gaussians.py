#!/usr/bin/env python3
"""
Repair WAI prepared object targets without rebuilding cropped image assets.

Older WAI prepared folders reused BEHAVE gs_init/G_o.pt, which can be a
normalized object rather than the metric fitted object mesh.  This script
rebuilds each prepared sequence's object Gaussian cloud from its selected raw
BEHAVE timestep and refreshes dual_branch_targets.npz, whose contact signature
depends on G_o.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_wai_dual_branch_assets import rebuild_metric_object_gaussians
from preprocess_procigen_gt import prepare_dual_branch_target_caches


H5_SEQUENCE_CACHE_FILENAME = "dual_branch_clip_cache_v3.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair metric object Gaussian targets in WAI_prepared.")
    parser.add_argument(
        "--prepared_root",
        type=str,
        default=str(REPO_ROOT / "sample_data" / "WAI_prepared" / "sequences"),
        help="Prepared WAI sequence root.",
    )
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--num_object_gaussians", type=int, default=4096)
    parser.add_argument("--init_gaussian_scale", type=float, default=0.01)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--sequence_name", action="append", dest="sequence_names")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_sequences(prepared_root: Path, requested: Sequence[str] | None, max_sequences: int) -> List[Path]:
    if requested:
        sequence_dirs = [prepared_root / str(name) for name in sorted(set(requested))]
    else:
        sequence_dirs = sorted(path for path in prepared_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    sequence_dirs = [path for path in sequence_dirs if path.is_dir()]
    if max_sequences > 0:
        sequence_dirs = sequence_dirs[: int(max_sequences)]
    return sequence_dirs


def remove_dataset_caches(prepared_root: Path, sequence_dirs: Sequence[Path], *, processed_subdir: str) -> List[str]:
    removed: List[str] = []
    index_cache_dir = prepared_root / ".dual_branch_index_cache"
    if index_cache_dir.exists():
        shutil.rmtree(index_cache_dir)
        removed.append(str(index_cache_dir))

    for sequence_dir in sequence_dirs:
        h5_cache = sequence_dir / processed_subdir / "cropped" / H5_SEQUENCE_CACHE_FILENAME
        if h5_cache.exists():
            h5_cache.unlink()
            removed.append(str(h5_cache))
    return removed


def repair_sequence(
    sequence_dir: Path,
    *,
    processed_subdir: str,
    gs_subdir: str,
    num_object_gaussians: int,
    init_gaussian_scale: float,
    dry_run: bool,
) -> Dict[str, object]:
    if dry_run:
        return {"sequence": sequence_dir.name, "status": "dry_run", "target_dir": str(sequence_dir)}

    object_meta = rebuild_metric_object_gaussians(
        sequence_dir,
        gs_subdir=gs_subdir,
        num_object_gaussians=int(num_object_gaussians),
        init_gaussian_scale=float(init_gaussian_scale),
    )
    cached_frames = prepare_dual_branch_target_caches(
        sequence_dir,
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        init_gaussian_scale=float(init_gaussian_scale),
    )
    return {
        "sequence": sequence_dir.name,
        "status": "repaired",
        "target_dir": str(sequence_dir),
        "cached_frames": int(cached_frames),
        "object_name": object_meta.get("object_name"),
        "reference_timestep": object_meta.get("reference_timestep"),
        "canonical_extent": object_meta.get("canonical_extent"),
    }


def main() -> None:
    args = parse_args()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    sequence_dirs = discover_sequences(prepared_root, args.sequence_names, int(args.max_sequences))
    if not sequence_dirs:
        raise RuntimeError(f"No prepared WAI sequences found under {prepared_root}")

    print(f"[wai-object-repair] start | sequences={len(sequence_dirs)} | root={prepared_root}", flush=True)
    reports: List[Dict[str, object]] = []
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        report = repair_sequence(
            sequence_dir,
            processed_subdir=args.processed_subdir,
            gs_subdir=args.gs_subdir,
            num_object_gaussians=int(args.num_object_gaussians),
            init_gaussian_scale=float(args.init_gaussian_scale),
            dry_run=bool(args.dry_run),
        )
        reports.append(report)
        extent = report.get("canonical_extent", "n/a")
        frames = report.get("cached_frames", "n/a")
        print(
            f"[wai-object-repair] {index}/{len(sequence_dirs)} {sequence_dir.name} "
            f"| status={report['status']} | frames={frames} | extent={extent}",
            flush=True,
        )

    removed_caches: List[str] = []
    if not args.dry_run:
        removed_caches = remove_dataset_caches(prepared_root, sequence_dirs, processed_subdir=args.processed_subdir)
    print(
        f"[wai-object-repair] done | repaired={sum(1 for item in reports if item['status'] == 'repaired')} "
        f"| removed_caches={len(removed_caches)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
