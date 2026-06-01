#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as unimain
import train_cointeract_hoi as train_hoi
from scripts import eval_dual_stream_hoi_rgb_checkpoints as eval_ckpts


COLORS: Dict[str, Tuple[int, int, int]] = {
    "pred_human": (45, 130, 255),
    "pred_object": (255, 140, 30),
    "gt_human": (39, 174, 96),
    "gt_object": (155, 89, 182),
}


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_points(points: Tensor, max_points: int) -> Tensor:
    points = points.detach().float().reshape(-1, points.shape[-1])[..., :3].cpu()
    valid = torch.isfinite(points).all(dim=-1)
    points = points[valid]
    if max_points > 0 and points.shape[0] > max_points:
        indices = torch.linspace(0, points.shape[0] - 1, steps=max_points).round().long()
        points = points.index_select(0, indices)
    return points


def _write_ascii_ply(path: Path, groups: Iterable[Tuple[str, Tensor]]) -> None:
    rows: List[Tuple[float, float, float, int, int, int]] = []
    for name, points in groups:
        red, green, blue = COLORS[name]
        for x, y, z in points.detach().float().cpu().numpy():
            rows.append((float(x), float(y), float(z), red, green, blue))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, red, green, blue in rows:
            handle.write(f"{x:.7f} {y:.7f} {z:.7f} {red} {green} {blue}\n")


def _save_input_rgb(path: Path, rgb_frame: Tensor) -> None:
    array = (
        rgb_frame.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _plotly_html(path: Path, groups: List[Tuple[str, Tensor]], *, title: str) -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    traces = []
    for name, points in groups:
        rgb = COLORS[name]
        xyz = points.numpy()
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=name.replace("_", " "),
                marker={
                    "size": 2.2,
                    "color": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                    "opacity": 0.82,
                },
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
        },
        legend={"orientation": "h", "y": 1.02},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(path), include_plotlyjs=True, full_html=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an interactive 3D reconstruction view for a dual-stream checkpoint.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("sample_data/BEHAVE_heldout_prepared/sequences"))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_ode_steps", type=int, default=12)
    parser.add_argument("--max_points_per_group", type=int, default=4096)
    parser.add_argument("--render_max_points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = _load_json(args.output_dir / "launch_config.json")
    train_args = argparse.Namespace(**train_config)
    eval_args = SimpleNamespace(
        data_root=args.data_root,
        split_file="",
        split_key="",
        max_sequences=0,
        dataset_cache_sequences=2,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = eval_ckpts._build_dataset(train_args, eval_args)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample_index {args.sample_index} is outside dataset length {len(dataset)}")
    dataloader = DataLoader(Subset(dataset, [args.sample_index]), batch_size=1, shuffle=False, num_workers=0)
    raw_batch = next(iter(dataloader))
    batch = train_hoi.prepare_batch(
        raw_batch,
        device=device,
        image_height=int(train_args.image_height),
        image_width=int(train_args.image_width),
    )

    model = train_hoi.build_model(train_args).to(device)
    model.ensure_wan_loaded(device)
    step = train_hoi.load_model_checkpoint(model, str(args.checkpoint))
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))
    with torch.no_grad():
        autocast_dtype = torch.bfloat16 if str(train_args.mixed_precision) == "bf16" else torch.float16
        use_autocast = device.type == "cuda" and str(train_args.mixed_precision) in {"bf16", "fp16"}
        context = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else torch.no_grad()
        with context:
            output = eval_ckpts._sample_state(
                model=model,
                batch=batch,
                num_ode_steps=int(args.num_ode_steps),
                generator=generator,
                rgb_to_hoi_scale=float(train_args.rgb_to_hoi_scale),
                hoi_to_rgb_scale=float(train_args.hoi_to_rgb_scale),
                cross_3d2d_scale=float(getattr(train_args, "cross_3d2d_scale", 1.0)),
                drop_rgb_branch=False,
            )

    point_sequences = unimain.build_eval_point_sequences(output.decoded_state, batch)
    pred_human = _select_points(point_sequences["pred_human"][0, 0], int(args.max_points_per_group))
    pred_object = _select_points(point_sequences["pred_object"][0, 0], int(args.max_points_per_group))
    gt_human = _select_points(point_sequences["target_human"][0, 0], int(args.max_points_per_group))
    gt_object = _select_points(point_sequences["target_object"][0, 0], int(args.max_points_per_group))

    sequence_name = batch.get("sequence_name")
    if isinstance(sequence_name, (list, tuple)):
        sequence_name = sequence_name[0]
    out_dir = args.out_dir or args.output_dir / "visualizations" / f"step_{int(step):07d}_sample_{args.sample_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("pred_human", pred_human),
        ("pred_object", pred_object),
        ("gt_human", gt_human),
        ("gt_object", gt_object),
    ]
    _plotly_html(
        out_dir / "reconstruction_overlay.html",
        groups,
        title=f"{sequence_name} | checkpoint step {int(step):07d} | sample {args.sample_index}",
    )
    _write_ascii_ply(out_dir / "pred_reconstruction.ply", groups[:2])
    _write_ascii_ply(out_dir / "gt_reconstruction.ply", groups[2:])
    _write_ascii_ply(out_dir / "overlay_reconstruction.ply", groups)
    _save_input_rgb(out_dir / "input_rgb.png", batch["rgb"][0, 0])

    unimain.render_prediction_batch(
        out_dir / "projection_debug",
        output=output,
        batch=batch,
        radius=3,
        max_points=int(args.render_max_points),
    )

    meta = {
        "checkpoint": str(args.checkpoint),
        "step": int(step),
        "sample_index": int(args.sample_index),
        "sequence_name": str(sequence_name),
        "num_ode_steps": int(args.num_ode_steps),
        "files": {
            "interactive_html": str(out_dir / "reconstruction_overlay.html"),
            "input_rgb": str(out_dir / "input_rgb.png"),
            "pred_ply": str(out_dir / "pred_reconstruction.ply"),
            "gt_ply": str(out_dir / "gt_reconstruction.ply"),
            "overlay_ply": str(out_dir / "overlay_reconstruction.ply"),
            "projection_panel": str(out_dir / "projection_debug" / "frames" / "frame_000.png"),
        },
    }
    (out_dir / "visualization_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
