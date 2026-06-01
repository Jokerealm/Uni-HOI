#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_cointeract_hoi as train_hoi
from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model.hoi_state_codec import HOIStateCodec


def _parse_budgets(value: str) -> List[Tuple[int, int]]:
    budgets: List[Tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" in item:
            left, right = item.split("x", 1)
            budgets.append((int(left), int(right)))
        else:
            count = int(item)
            budgets.append((count, count))
    if not budgets:
        raise ValueError("At least one token budget is required.")
    return budgets


def _mean_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for item in items:
        count += 1
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return {key: value / float(max(count, 1)) for key, value in totals.items()}


def _encode_state_target(codec: HOIStateCodec, batch: Dict[str, object]) -> torch.Tensor:
    return codec.encode_targets(
        human_shape=batch["human_shape"],
        human_pose=batch["body_pose"],
        human_translation=batch["cam_t"],
        object_transforms=batch["object_poses"],
        contact_signature=batch["contact_signature"],
        human_gaussians=batch["human_gaussians"],
        object_gaussians=batch["object_gaussians"],
        joints_3d=batch["joints_3d"],
    )


def _tensor_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in batch.items():
        result[key] = value.to(device=device) if isinstance(value, torch.Tensor) else value
    return result


def _state_fm_summary(
    *,
    codec: HOIStateCodec,
    batch: Dict[str, object],
    generator: torch.Generator,
    group_weights: Dict[str, float],
) -> Dict[str, object]:
    target = _encode_state_target(codec, batch)
    noise = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    velocity_target = target - noise
    predicted_velocity = torch.zeros_like(velocity_target)
    token_slices = codec._slices()

    uniform_loss, uniform_group_losses, uniform_contributions = train_hoi.compute_state_fm_loss(
        predicted_velocity,
        velocity_target,
        token_slices=token_slices,
        mode="uniform",
        group_weights=group_weights,
    )
    balanced_loss, balanced_group_losses, balanced_contributions = train_hoi.compute_state_fm_loss(
        predicted_velocity,
        velocity_target,
        token_slices=token_slices,
        mode="group_balanced",
        group_weights=group_weights,
    )

    def _to_float_map(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {key: float(value.detach().float().cpu().item()) for key, value in values.items()}

    def _share_map(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
        detached = {key: value.detach().float().cpu() for key, value in values.items()}
        total = sum(detached.values(), torch.tensor(0.0)).clamp_min(1e-12)
        return {key: float((value / total).item()) for key, value in detached.items()}

    return {
        "uniform_loss": float(uniform_loss.detach().float().cpu().item()),
        "group_balanced_loss": float(balanced_loss.detach().float().cpu().item()),
        "uniform_group_loss": _to_float_map(uniform_group_losses),
        "group_balanced_group_loss": _to_float_map(balanced_group_losses),
        "uniform_contribution": _to_float_map(uniform_contributions),
        "group_balanced_contribution": _to_float_map(balanced_contributions),
        "uniform_contribution_share": _share_map(uniform_contributions),
        "group_balanced_contribution_share": _share_map(balanced_contributions),
    }


def _run_budget(args: argparse.Namespace, *, human_count: int, object_count: int) -> Dict[str, object]:
    dataset = DualBranchHOIDataset(
        data_root=str(args.data_root),
        clip_length=int(args.clip_length),
        clip_stride=int(args.clip_stride),
        processed_subdir=str(args.processed_subdir),
        gs_subdir=str(args.gs_subdir),
        human_gaussian_source=str(args.human_gaussian_source),
        num_human_gaussians=int(human_count),
        num_object_gaussians=int(object_count),
        num_joints=int(args.num_joints),
        contact_dim=int(args.contact_dim),
        coordinate_mode=str(args.coordinate_mode),
        max_sequences=int(args.max_sequences),
        cache_sequences=int(args.dataset_cache_sequences),
        cache_rgb=False,
        rgb_cache_max_frames=0,
        split_file=str(args.split_file),
        split_key=str(args.split_key),
        prefer_h5_cache=bool(args.prefer_h5_cache),
        include_human_vertices=False,
        include_keypoint_heatmaps=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        drop_last=False,
    )
    device = torch.device(args.device)
    codec = HOIStateCodec(
        hidden_dim=int(args.hidden_dim),
        num_human_gaussians=int(human_count),
        num_object_gaussians=int(object_count),
        num_frames=int(args.clip_length),
        num_joints=int(args.num_joints),
        contact_dim=int(args.contact_dim),
        human_shape_dim=int(args.human_shape_dim),
        human_pose_dim=int(args.human_pose_dim),
    ).to(device)
    codec.eval()
    group_weights = train_hoi.parse_state_fm_group_weights(args.state_fm_group_weights)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + human_count * 17 + object_count * 31)

    batch_summaries: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= int(args.max_batches):
                break
            batch = _tensor_batch_to_device(raw_batch, device)
            batch_summaries.append(
                _state_fm_summary(
                    codec=codec,
                    batch=batch,
                    generator=generator,
                    group_weights=group_weights,
                )
            )

    token_slices = codec._slices()
    token_counts = {
        name: int(token_slices[name].stop - token_slices[name].start)
        for name in ("context", *train_hoi.STATE_FM_GROUPS)
    }
    token_shares = {key: value / float(max(codec.total_tokens, 1)) for key, value in token_counts.items()}
    scalar_keys = ("uniform_loss", "group_balanced_loss")
    aggregate: Dict[str, object] = {
        key: sum(float(item[key]) for item in batch_summaries) / float(max(len(batch_summaries), 1))
        for key in scalar_keys
    }
    for nested_key in (
        "uniform_group_loss",
        "group_balanced_group_loss",
        "uniform_contribution",
        "group_balanced_contribution",
        "uniform_contribution_share",
        "group_balanced_contribution_share",
    ):
        aggregate[nested_key] = _mean_dict(item[nested_key] for item in batch_summaries)

    return {
        "num_human_gaussians": int(human_count),
        "num_object_gaussians": int(object_count),
        "total_tokens": int(codec.total_tokens),
        "token_counts": token_counts,
        "token_shares": token_shares,
        "num_batches": int(len(batch_summaries)),
        "aggregate": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HOI state FM token imbalance without loading Wan.")
    parser.add_argument("--data_root", type=Path, default=Path("sample_data/behave_1pct/sequences"))
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--human_gaussian_source", type=str, default="smpl_mesh", choices=("smpl_mesh", "teacher"))
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--coordinate_mode", type=str, default="relative", choices=("relative", "absolute"))
    parser.add_argument("--prefer_h5_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_cache_sequences", type=int, default=1)
    parser.add_argument("--max_sequences", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_length", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=1)
    parser.add_argument("--num_joints", type=int, default=22)
    parser.add_argument("--contact_dim", type=int, default=4)
    parser.add_argument("--human_shape_dim", type=int, default=10)
    parser.add_argument("--human_pose_dim", type=int, default=72)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--token_budgets", type=str, default="850x850,128x128")
    parser.add_argument("--state_fm_group_weights", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("outputs/hoi_token_fm_imbalance/summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = _parse_budgets(args.token_budgets)
    results = [
        _run_budget(args, human_count=human_count, object_count=object_count)
        for human_count, object_count in budgets
    ]
    payload = {
        "analysis": "hoi_state_fm_token_imbalance",
        "data_root": str(args.data_root),
        "max_batches": int(args.max_batches),
        "state_fm_groups": list(train_hoi.STATE_FM_GROUPS),
        "state_fm_group_weights": train_hoi.parse_state_fm_group_weights(args.state_fm_group_weights),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    for result in results:
        uniform_share = result["aggregate"]["uniform_contribution_share"]
        balanced_share = result["aggregate"]["group_balanced_contribution_share"]
        gaussian_uniform = uniform_share["human_gaussians"] + uniform_share["object_gaussians"]
        gaussian_balanced = balanced_share["human_gaussians"] + balanced_share["object_gaussians"]
        print(
            "budget "
            f"H={result['num_human_gaussians']} O={result['num_object_gaussians']} "
            f"tokens={result['total_tokens']} "
            f"gaussian_token_share={result['token_shares']['human_gaussians'] + result['token_shares']['object_gaussians']:.4f} "
            f"uniform_gaussian_contribution_share={gaussian_uniform:.4f} "
            f"balanced_gaussian_contribution_share={gaussian_balanced:.4f}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
