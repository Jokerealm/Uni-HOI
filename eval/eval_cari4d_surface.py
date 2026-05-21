#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.chamfer_distance import chamfer_distance


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _fps_indices(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= count:
        return np.arange(points.shape[0], dtype=np.int64)
    centroid = points.mean(axis=0, keepdims=True)
    farthest = int(np.sum((points - centroid) ** 2, axis=1).argmax())
    selected = np.empty(count, dtype=np.int64)
    min_dist = np.full(points.shape[0], np.inf, dtype=np.float32)
    for idx in range(count):
        selected[idx] = farthest
        dist = np.sum((points - points[farthest : farthest + 1]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        farthest = int(min_dist.argmax())
    return selected


def _sample_points(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == count:
        return points
    if points.shape[0] > count:
        return points[_fps_indices(points, count)]
    pad = np.repeat(points[-1:], count - points.shape[0], axis=0)
    return np.concatenate([points, pad], axis=0)


def _sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return np.asarray(mesh.sample(count), dtype=np.float32)
        finally:
            np.random.set_state(state)
    except Exception:
        return _sample_points(vertices, count)


def _resolve_surface_points(payload: Dict[str, np.ndarray], prefix: str, count: int, seed: int) -> np.ndarray:
    points_key = f"{prefix}_points"
    vertices_key = f"{prefix}_vertices"
    faces_key = f"{prefix}_faces"
    if points_key in payload:
        return _sample_points(payload[points_key], count)
    if vertices_key in payload and faces_key in payload:
        return _sample_mesh_surface(payload[vertices_key], payload[faces_key], count, seed)
    if vertices_key in payload:
        return _sample_points(payload[vertices_key], count)
    raise KeyError(
        f"Missing `{points_key}` or `{vertices_key}`/`{faces_key}` in evaluation payload."
    )


def _normalize_like_cari4d(gt_human: np.ndarray, gt_object: np.ndarray, pred_human: np.ndarray, pred_object: np.ndarray):
    gt_combined = np.concatenate([gt_human, gt_object], axis=0)
    center = gt_combined.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(gt_combined - center, axis=1).max()
    radius = max(float(radius), 1e-8)
    return (
        (gt_human - center) / radius,
        (gt_object - center) / radius,
        (pred_human - center) / radius,
        (pred_object - center) / radius,
    )


def _compute_scores(gt: np.ndarray, pred: np.ndarray, thresholds: Iterable[float]) -> Tuple[float, Dict[str, float]]:
    chamfer, pred_to_gt, gt_to_pred = chamfer_distance(gt, pred, ret_intermediate=True)
    result: Dict[str, float] = {"chamfer": float(chamfer)}
    for threshold in thresholds:
        recall = float(np.mean(gt_to_pred < threshold))
        precision = float(np.mean(pred_to_gt < threshold))
        fscore = 0.0 if recall + precision <= 0 else 2.0 * recall * precision / (recall + precision)
        suffix = f"@{threshold:g}"
        result[f"fscore{suffix}"] = float(fscore)
        result[f"precision{suffix}"] = float(precision)
        result[f"recall{suffix}"] = float(recall)
    return float(chamfer), result


def evaluate_pair(gt_path: Path, pred_path: Path, *, num_samples: int, thresholds: Iterable[float], seed: int):
    gt = _load_npz(gt_path)
    pred = _load_npz(pred_path)
    gt_human = _resolve_surface_points(gt, "human", num_samples, seed)
    gt_object = _resolve_surface_points(gt, "object", num_samples, seed + 1)
    pred_human = _resolve_surface_points(pred, "human", num_samples, seed + 2)
    pred_object = _resolve_surface_points(pred, "object", num_samples, seed + 3)
    gt_human, gt_object, pred_human, pred_object = _normalize_like_cari4d(
        gt_human,
        gt_object,
        pred_human,
        pred_object,
    )

    _, human = _compute_scores(gt_human, pred_human, thresholds)
    _, obj = _compute_scores(gt_object, pred_object, thresholds)
    _, combined = _compute_scores(
        np.concatenate([gt_human, gt_object], axis=0),
        np.concatenate([pred_human, pred_object], axis=0),
        thresholds,
    )
    return {"file": gt_path.name, "human": human, "object": obj, "combined": combined}


def _mean_nested(rows):
    keys = ("human", "object", "combined")
    summary = {}
    for part in keys:
        metric_names = rows[0][part].keys()
        summary[part] = {
            metric: float(np.mean([row[part][metric] for row in rows]))
            for metric in metric_names
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CARI4D-style 8196/8196 surface Chamfer/F-score evaluation.")
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--num_samples", type=int, default=8196)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    rows = []
    for gt_path in sorted(gt_dir.glob(args.pattern)):
        pred_path = pred_dir / gt_path.name
        if not pred_path.is_file():
            print(f"skip missing prediction: {pred_path}", flush=True)
            continue
        rows.append(
            evaluate_pair(
                gt_path,
                pred_path,
                num_samples=args.num_samples,
                thresholds=args.thresholds,
                seed=args.seed,
            )
        )
    if not rows:
        raise RuntimeError("No matching evaluation pairs found.")

    summary = _mean_nested(rows)
    result = {"num_pairs": len(rows), "num_samples_per_part": args.num_samples, "summary": summary, "items": rows}
    print(json.dumps(result["summary"], indent=2), flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
