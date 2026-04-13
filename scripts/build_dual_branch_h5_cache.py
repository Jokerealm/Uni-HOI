#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dual_branch_fm_dataset import (
    DEFAULT_SEQUENCE_H5_CHUNK_FRAMES,
    build_dual_branch_sequence_h5_cache,
    _discover_sequence_dirs,
    _load_sequence_names_from_split_file,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize per-sequence H5 caches for clip-level lazy reads."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument(
        "--human_gaussian_source",
        type=str,
        default="smpl_mesh",
        choices=("smpl_mesh", "teacher"),
    )
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--num_human_gaussians", type=int, default=1024)
    parser.add_argument("--num_object_gaussians", type=int, default=1024)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--chunk_frames", type=int, default=DEFAULT_SEQUENCE_H5_CHUNK_FRAMES)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    allowed_sequence_names = None
    if args.split_file:
        allowed_sequence_names = _load_sequence_names_from_split_file(args.split_file, args.split_key)

    sequence_dirs = _discover_sequence_dirs(
        args.data_root,
        args.processed_subdir,
        args.gs_subdir,
        allowed_sequence_names=allowed_sequence_names,
    )
    if args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]
    if not sequence_dirs:
        raise RuntimeError("No matching sequences found to build H5 caches.")

    overall_start = time.perf_counter()
    print(
        "[build_dual_branch_h5_cache] starting "
        f"| sequences={len(sequence_dirs)} "
        f"| data_root={Path(args.data_root).expanduser().resolve()} "
        f"| overwrite={int(bool(args.overwrite))} "
        f"| chunk_frames={args.chunk_frames}",
        flush=True,
    )
    failures = 0
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_name = Path(sequence_dir).name
        start_time = time.perf_counter()
        try:
            cache_path = build_dual_branch_sequence_h5_cache(
                sequence_dir,
                processed_subdir=args.processed_subdir,
                gs_subdir=args.gs_subdir,
                human_gaussian_source=args.human_gaussian_source,
                num_human_gaussians=args.num_human_gaussians,
                num_object_gaussians=args.num_object_gaussians,
                num_joints=args.num_joints,
                contact_dim=args.contact_dim,
                overwrite=args.overwrite,
                chunk_frames=args.chunk_frames,
            )
            size_mb = cache_path.stat().st_size / (1024.0 * 1024.0)
            elapsed = time.perf_counter() - start_time
            print(
                "[build_dual_branch_h5_cache] built "
                f"| sequence={index}/{len(sequence_dirs)} "
                f"| name={sequence_name} "
                f"| size_mb={size_mb:.1f} "
                f"| elapsed={elapsed:.1f}s "
                f"| path={cache_path}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(
                "[build_dual_branch_h5_cache] failed "
                f"| sequence={index}/{len(sequence_dirs)} "
                f"| name={sequence_name} "
                f"| error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if not args.continue_on_error:
                raise

    total_elapsed = time.perf_counter() - overall_start
    print(
        "[build_dual_branch_h5_cache] finished "
        f"| sequences={len(sequence_dirs)} "
        f"| failures={failures} "
        f"| elapsed={total_elapsed:.1f}s",
        flush=True,
    )
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
