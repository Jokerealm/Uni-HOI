#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dual_branch_fm_dataset import DualBranchHOIDataset
from model.joint_renderer_loss import DiffRasterizationLayer
from train_dual_branch_fm import (
    SequencePrefetchBatchIterator,
    build_arg_parser as build_train_arg_parser,
    build_model_from_args,
    build_curriculum_loss_weights,
    build_human_supervision_target,
    build_object_supervision_target,
    build_teacher_state,
    compute_losses,
    configure_torch_runtime,
    filter_video_teacher_state_dict,
    flow_match_sample,
    infer_condition_channels,
    model_uses_wan_teacher,
    render_human_proxy_branch,
    render_object_branch,
    resolve_state_to_video_scale,
    resolve_video_to_state_scale,
    resize_video_batch,
    scale_camera_intrinsics,
)


def namespace_from_checkpoint_args(checkpoint_args: dict[str, Any]) -> argparse.Namespace:
    parser = build_train_arg_parser()
    defaults = parser.parse_args([])
    if "video_backend" not in checkpoint_args:
        defaults.video_backend = "legacy_codec"
        defaults.video_channels = 6
    for key, value in checkpoint_args.items():
        setattr(defaults, key, value)
    return defaults


def build_eval_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a dual-branch FM checkpoint on a dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--split_key", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    configure_torch_runtime()
    torch.manual_seed(args.seed)

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested, but `torch.cuda.is_available()` is false.")
    device = torch.device(args.device if args.device == "cpu" else "cuda")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = namespace_from_checkpoint_args(checkpoint.get("args", {}))
    step = int(checkpoint.get("step", 0))

    dataset = DualBranchHOIDataset(
        data_root=args.data_root,
        clip_length=checkpoint_args.clip_length,
        clip_stride=checkpoint_args.clip_stride,
        processed_subdir=checkpoint_args.processed_subdir,
        gs_subdir=checkpoint_args.gs_subdir,
        human_gaussian_source=checkpoint_args.human_gaussian_source,
        num_human_gaussians=checkpoint_args.num_human_gaussians,
        num_object_gaussians=checkpoint_args.num_object_gaussians,
        num_joints=checkpoint_args.num_joints,
        contact_dim=checkpoint_args.contact_dim,
        background_value=checkpoint_args.background_value,
        max_sequences=0,
        cache_sequences=checkpoint_args.dataset_cache_sequences,
        cache_rgb=bool(getattr(checkpoint_args, "cache_rgb", True)),
        rgb_cache_max_frames=checkpoint_args.rgb_cache_max_frames,
        index_progress_every=0,
        index_progress_callback=None,
        split_file=args.split_file,
        split_key=args.split_key,
    )
    iterator = SequencePrefetchBatchIterator(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        seed=args.seed,
        process_index=0,
        num_processes=1,
        warm_start_short_sequences=False,
    )

    condition_channels = infer_condition_channels(dataset)
    model = build_model_from_args(checkpoint_args, condition_channels=condition_channels)
    incompatible = model.load_state_dict(filter_video_teacher_state_dict(checkpoint["model"]), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[eval_dual_branch_fm_checkpoint] warning: checkpoint/model mismatch detected during evaluation "
            f"| missing={incompatible.missing_keys[:10]} "
            f"| unexpected={incompatible.unexpected_keys[:10]} "
            "| continuing because Uni-HOI stage handoff uses strict=False.",
            flush=True,
        )
    model.to(device=device).eval()

    renderer = DiffRasterizationLayer(
        image_height=checkpoint_args.image_height,
        image_width=checkpoint_args.image_width,
    ).to(device=device)

    loss_weights, curriculum_metrics = build_curriculum_loss_weights(checkpoint_args, step)
    active_losses = [name for name, value in loss_weights.items() if value > 0.0]

    metric_sums: dict[str, float] = {}
    evaluated_clips = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in iterator:
            batch = _move_batch_to_device(batch, device)
            batch_size = int(batch["rgb"].shape[0])
            sequence_names = [str(name) for name in batch["sequence_name"]]
            if args.max_clips > 0 and evaluated_clips >= args.max_clips:
                break
            if args.max_clips > 0 and evaluated_clips + batch_size > args.max_clips:
                keep = args.max_clips - evaluated_clips
                batch = {
                    key: value[:keep] if isinstance(value, torch.Tensor) and value.shape[0] == batch_size else value
                    for key, value in batch.items()
                }
                batch_size = keep
            if batch_size <= 0:
                break

            rgb = batch["rgb"]
            human_visible = batch["human_visible"]
            masks_human = batch["masks_human"]
            masks_object = batch["masks_object"]
            m_primary = batch["m_primary"]
            m_secondary = batch["m_secondary"]
            m_object_region = batch["m_object_region"]
            keypoint_heatmaps = batch["keypoint_heatmaps"]
            depth = batch["depth"]
            camera_intrinsics = batch["camera_intrinsics"]
            object_poses = batch["object_poses"]
            human_shape = batch["human_shape"]
            body_pose = batch["body_pose"]
            cam_t = batch["cam_t"]
            human_gaussians = batch["human_gaussians"]
            object_gaussians = batch["object_gaussians"]
            joints_3d = batch["joints_3d"]
            contact_signature = batch["contact_signature"]
            object_categories = [str(name) for name in batch["object_category"]]

            if rgb.shape[-2:] != (checkpoint_args.image_height, checkpoint_args.image_width):
                rgb = resize_video_batch(rgb, size=(checkpoint_args.image_height, checkpoint_args.image_width), mode="bilinear")
                human_visible = resize_video_batch(
                    human_visible,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="bilinear",
                )
                masks_human = resize_video_batch(
                    masks_human,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="nearest",
                )
                masks_object = resize_video_batch(
                    masks_object,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="nearest",
                )
                m_primary = resize_video_batch(
                    m_primary,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="nearest",
                )
                m_secondary = resize_video_batch(
                    m_secondary,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="nearest",
                )
                m_object_region = resize_video_batch(
                    m_object_region,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="nearest",
                )
                keypoint_heatmaps = resize_video_batch(
                    keypoint_heatmaps,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="bilinear",
                )
                depth = resize_video_batch(
                    depth,
                    size=(checkpoint_args.image_height, checkpoint_args.image_width),
                    mode="bilinear",
                )
                source_hw = batch["rgb"].shape[-2:]
                camera_intrinsics_render = scale_camera_intrinsics(
                    camera_intrinsics,
                    source_size=(int(source_hw[0]), int(source_hw[1])),
                    target_size=(checkpoint_args.image_height, checkpoint_args.image_width),
                )
            else:
                camera_intrinsics_render = camera_intrinsics

            condition_video = torch.cat(
                [
                    rgb,
                    masks_human,
                    masks_object,
                    depth,
                    m_primary,
                    m_secondary,
                    m_object_region,
                    keypoint_heatmaps,
                ],
                dim=2,
            )
            teacher_state = build_teacher_state(
                {
                    "human_shape": human_shape,
                    "body_pose": body_pose,
                    "cam_t": cam_t,
                    "human_gaussians": human_gaussians,
                    "object_gaussians": object_gaussians,
                    "joints_3d": joints_3d,
                    "object_poses": object_poses,
                    "contact_signature": contact_signature,
                }
            )

            human_supervision_target = None
            human_supervision_weights = None
            object_supervision_target = None
            object_supervision_weights = None
            video_target = rgb
            video_target_tokens = model.encode_video_target(video_target)
            state_target_tokens = model.encode_state_target(
                human_shape=human_shape,
                human_pose=body_pose,
                human_translation=cam_t,
                human_gaussians=human_gaussians,
                object_gaussians=object_gaussians,
                joints_3d=joints_3d,
                object_transforms=object_poses,
                contact_signature=contact_signature,
            )

            timesteps = torch.rand(batch_size, device=device, dtype=torch.float32)
            video_noise = torch.randn_like(video_target_tokens)
            state_noise = torch.randn_like(state_target_tokens)
            video_xt, video_velocity_target = flow_match_sample(video_target_tokens, video_noise, timesteps)
            state_xt, state_velocity_target = flow_match_sample(state_target_tokens, state_noise, timesteps)

            state_to_video_scale = resolve_state_to_video_scale(checkpoint_args, step)
            video_to_state_scale = resolve_video_to_state_scale(checkpoint_args, step)
            forward_kwargs = {}
            if model_uses_wan_teacher(model):
                forward_kwargs["condition_latents"] = video_target_tokens
            output = model(
                video_xt=video_xt,
                state_xt=state_xt,
                timesteps=timesteps,
                condition_video=condition_video,
                camera_intrinsics=camera_intrinsics_render,
                sequence_names=sequence_names,
                object_categories=object_categories,
                state_to_video_scale=state_to_video_scale,
                video_to_state_scale=video_to_state_scale,
                **forward_kwargs,
            )
            _, metrics = compute_losses(
                model=model,
                output=output,
                video_xt=video_xt,
                video_velocity_target=video_velocity_target,
                state_xt=state_xt,
                state_velocity_target=state_velocity_target,
                teacher_state=teacher_state,
                human_supervision_target=human_supervision_target,
                human_supervision_weights=human_supervision_weights,
                object_supervision_target=object_supervision_target,
                object_supervision_weights=object_supervision_weights,
                masks_human=masks_human,
                masks_object=masks_object,
                keypoint_heatmaps=keypoint_heatmaps,
                depth=depth,
                camera_intrinsics_render=camera_intrinsics_render,
                renderer=renderer,
                timesteps=timesteps,
                video_target_tokens=video_target_tokens,
                state_target_tokens=state_target_tokens,
                weights=loss_weights,
                video_teacher_is_frozen=bool(getattr(checkpoint_args, "freeze_video_backbone", False)),
            )

            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.item()) * batch_size
            evaluated_clips += batch_size

    if evaluated_clips == 0:
        raise RuntimeError("No clips were evaluated.")

    elapsed_seconds = time.time() - start_time
    averaged_metrics = {
        name: value / float(evaluated_clips)
        for name, value in sorted(metric_sums.items())
    }

    result = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split_file": str(Path(args.split_file).expanduser().resolve()),
        "split_key": args.split_key,
        "num_sequences": len(dataset.sequence_dirs),
        "num_clips": len(dataset),
        "evaluated_clips": evaluated_clips,
        "batch_size": int(args.batch_size),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "curriculum": {key: float(value) for key, value in curriculum_metrics.items()},
        "active_losses": active_losses,
        "metrics": averaged_metrics,
    }
    return result


def main() -> None:
    args = build_eval_arg_parser().parse_args()
    result = evaluate_checkpoint(args)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
