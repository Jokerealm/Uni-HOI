#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as unimain
import train_cointeract_hoi as train_hoi
from dataset.dual_branch_fm_dataset import DualBranchHOIDataset


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _parse_steps(value: str) -> Optional[set[int]]:
    text = value.strip()
    if not text:
        return None
    return {int(item) for item in re.split(r"[,\\s]+", text) if item}


def _select_checkpoints(output_dir: Path, requested_steps: Optional[set[int]]) -> List[Path]:
    checkpoints = sorted(
        output_dir.joinpath("checkpoints").glob("checkpoint_*.pt"),
        key=_checkpoint_step,
    )
    if requested_steps is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step(path) in requested_steps]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {output_dir / 'checkpoints'}")
    return checkpoints


def _namespace_from_train_config(config: Dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(**config)


def _build_dataset(train_args: argparse.Namespace, eval_args: argparse.Namespace) -> DualBranchHOIDataset:
    return DualBranchHOIDataset(
        data_root=str(eval_args.data_root),
        clip_length=int(train_args.clip_length),
        clip_stride=int(train_args.clip_stride),
        processed_subdir=str(train_args.processed_subdir),
        gs_subdir=str(train_args.gs_subdir),
        human_gaussian_source=str(train_args.human_gaussian_source),
        num_human_gaussians=int(train_args.num_human_gaussians),
        num_object_gaussians=int(train_args.num_object_gaussians),
        num_joints=int(train_args.num_joints),
        contact_dim=int(train_args.contact_dim),
        coordinate_mode=str(train_args.coordinate_mode),
        max_sequences=int(eval_args.max_sequences),
        cache_sequences=int(eval_args.dataset_cache_sequences),
        cache_rgb=bool(train_args.cache_rgb),
        rgb_cache_max_frames=int(train_args.rgb_cache_max_frames),
        split_file=str(eval_args.split_file),
        split_key=str(eval_args.split_key),
        prefer_h5_cache=bool(train_args.prefer_h5_cache),
        include_human_vertices=True,
        include_keypoint_heatmaps=False,
    )


def _build_eval_loader(dataset: DualBranchHOIDataset, eval_args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(eval_args.batch_size),
        shuffle=False,
        num_workers=int(eval_args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _state_noise(model, batch_size: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> Tensor:
    return torch.randn(
        batch_size,
        int(model.state_codec.total_tokens),
        int(model.hidden_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _sample_state(
    *,
    model,
    batch: Dict[str, object],
    num_ode_steps: int,
    generator: torch.Generator,
    rgb_to_hoi_scale: float,
    hoi_to_rgb_scale: float,
    cross_3d2d_scale: float,
) -> object:
    rgb = batch["rgb"]
    batch_size = int(rgb.shape[0])
    first_frame = rgb[:, 0]
    first_frame_latents = model.encode_first_frame(first_frame)
    video_xt = first_frame_latents
    state_xt = _state_noise(
        model,
        batch_size=batch_size,
        device=rgb.device,
        dtype=first_frame_latents.dtype,
        generator=generator,
    )
    steps = max(int(num_ode_steps), 1)
    dt = 1.0 / float(steps)
    output = None
    for step_idx in range(steps):
        t_value = min(step_idx * dt, 1.0 - 1e-4)
        timesteps = torch.full((batch_size,), t_value, device=rgb.device, dtype=torch.float32)
        forward_kwargs = {
            "video_xt": video_xt,
            "state_xt": state_xt,
            "timesteps": timesteps,
            "first_frame": first_frame,
            "rgb_to_hoi_scale": float(rgb_to_hoi_scale),
            "hoi_to_rgb_scale": float(hoi_to_rgb_scale),
        }
        if model.__class__.__name__ == "CoMoViHOIRGBModel":
            forward_kwargs["cross_3d2d_scale"] = float(cross_3d2d_scale)
        output = model(**forward_kwargs)
        state_xt = state_xt + dt * output.state_velocity.to(dtype=state_xt.dtype)

    decoded = model.decode_state_tokens(state_xt)
    return SimpleNamespace(
        decoded_state=decoded,
        rgb_velocity=output.rgb_velocity if output is not None else None,
        state_velocity=output.state_velocity if output is not None else None,
    )


def _weighted_state_losses(train_args: argparse.Namespace) -> Dict[str, float]:
    return {
        "shape": float(train_args.lambda_shape),
        "pose": float(train_args.lambda_pose),
        "translation": float(train_args.lambda_translation),
        "object_pose": float(train_args.lambda_object_pose),
        "contact": float(train_args.lambda_contact),
        "joints": float(train_args.lambda_joints),
        "human_gaussian_chamfer": float(train_args.lambda_human_gaussian) * float(train_args.lambda_gaussian_chamfer),
        "object_gaussian_chamfer": float(train_args.lambda_object_gaussian) * float(train_args.lambda_gaussian_chamfer),
        "human_gaussian_xyz": float(train_args.lambda_human_gaussian) * float(train_args.lambda_gaussian_xyz_l1),
        "object_gaussian_xyz": float(train_args.lambda_object_gaussian) * float(train_args.lambda_gaussian_xyz_l1),
        "human_gaussian_attr": float(train_args.lambda_human_gaussian) * float(train_args.lambda_gaussian_attr_l1),
        "object_gaussian_attr": float(train_args.lambda_object_gaussian) * float(train_args.lambda_gaussian_attr_l1),
    }


def _to_float_dict(metrics: Dict[str, Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().float().cpu().item()) for key, value in metrics.items()}


def _evaluate_checkpoint(
    *,
    model,
    checkpoint: Path,
    dataloader: DataLoader,
    train_args: argparse.Namespace,
    eval_args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    step = train_hoi.load_model_checkpoint(model, str(checkpoint))
    model.to(device)
    model.eval()
    weights = _weighted_state_losses(train_args)
    metric_args = SimpleNamespace(
        test_eval_max_points=int(eval_args.test_eval_max_points),
        test_eval_chamfer_chunk_size=int(eval_args.test_eval_chamfer_chunk_size),
        test_eval_acceleration=False,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_args.seed) + int(step))
    totals: Dict[str, float] = {}
    count = 0

    max_batches = int(eval_args.max_batches)
    autocast_dtype = torch.bfloat16 if str(train_args.mixed_precision) == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and str(train_args.mixed_precision) in {"bf16", "fp16"}

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = train_hoi.prepare_batch(
                raw_batch,
                device=device,
                image_height=int(train_args.image_height),
                image_width=int(train_args.image_width),
            )
            context = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else torch.no_grad()
            with context:
                output = _sample_state(
                    model=model,
                    batch=batch,
                    num_ode_steps=int(eval_args.num_ode_steps),
                    generator=generator,
                    rgb_to_hoi_scale=float(train_args.rgb_to_hoi_scale),
                    hoi_to_rgb_scale=float(train_args.hoi_to_rgb_scale),
                    cross_3d2d_scale=float(getattr(train_args, "cross_3d2d_scale", 1.0)),
                )
                state_losses = train_hoi.compute_state_losses(output.decoded_state, batch, weights=weights)
                eval_metrics = unimain.compute_test_eval_metrics(output, batch, metric_args)
            metrics = _to_float_dict(state_losses)
            metrics.update(_to_float_dict(eval_metrics))
            batch_size = int(batch["rgb"].shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            count += batch_size

    averages = {key: value / float(max(count, 1)) for key, value in totals.items()}
    cd_mean = (averages["CD-h"] + averages["CD-o"] + averages["CD-c"]) / 3.0
    averages["CD-mean"] = cd_mean
    return {
        "checkpoint": str(checkpoint),
        "step": int(step if step > 0 else _checkpoint_step(checkpoint)),
        "num_samples": int(count),
        "metrics": averages,
    }


def _summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
    best = min(results, key=lambda item: item["metrics"]["CD-mean"])
    return {
        "best_step": int(best["step"]),
        "best_cd_mean": float(best["metrics"]["CD-mean"]),
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HOI/RGB dual-stream checkpoints on BEHAVE heldout.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cointeract_hoi_wan_ti2v"))
    parser.add_argument("--data_root", type=Path, default=Path("sample_data/BEHAVE_heldout_prepared/sequences"))
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="")
    parser.add_argument("--steps", type=str, default="", help="Comma/space separated checkpoint steps. Empty means all.")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset_cache_sequences", type=int, default=2)
    parser.add_argument("--num_ode_steps", type=int, default=12)
    parser.add_argument("--test_eval_max_points", type=int, default=4096)
    parser.add_argument("--test_eval_chamfer_chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    eval_args = parse_args()
    output_dir = eval_args.output_dir
    train_config = _load_json(output_dir / "launch_config.json")
    train_args = _namespace_from_train_config(train_config)
    requested_steps = _parse_steps(eval_args.steps)
    checkpoints = _select_checkpoints(output_dir, requested_steps)
    device = torch.device(eval_args.device if torch.cuda.is_available() or eval_args.device == "cpu" else "cpu")

    dataset = _build_dataset(train_args, eval_args)
    dataloader = _build_eval_loader(dataset, eval_args)
    model = train_hoi.build_model(train_args).to(device)
    model.ensure_wan_loaded(device)

    results = []
    for checkpoint in checkpoints:
        result = _evaluate_checkpoint(
            model=model,
            checkpoint=checkpoint,
            dataloader=dataloader,
            train_args=train_args,
            eval_args=eval_args,
            device=device,
        )
        results.append(result)
        metrics = result["metrics"]
        print(
            "eval "
            f"step={int(result['step']):07d} "
            f"loss={metrics.get('supervised', math.nan):.6f} "
            f"CD-h={metrics['CD-h']:.6f} "
            f"CD-o={metrics['CD-o']:.6f} "
            f"CD-c={metrics['CD-c']:.6f} "
            f"CD-mean={metrics['CD-mean']:.6f}",
            flush=True,
        )

    payload = {
        "run": str(output_dir),
        "train_config": train_config,
        "eval": {
            "data_root": str(eval_args.data_root),
            "max_batches": int(eval_args.max_batches),
            "num_ode_steps": int(eval_args.num_ode_steps),
            "batch_size": int(eval_args.batch_size),
            "num_checkpoints": len(results),
        },
        "summary": _summarize(results),
        "results": results,
    }
    out_path = eval_args.out or output_dir / (
        f"dual_stream_eval_ode{int(eval_args.num_ode_steps)}_b{int(eval_args.max_batches)}.json"
    )
    _write_json(out_path, payload)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
