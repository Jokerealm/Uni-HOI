#!/usr/bin/env python3
"""
Run latent-space joint video-3D Flow Matching inference.

This script loads a `train_fm.py` checkpoint, restores the frozen Hunyuan3D-2
ControlNet wrapper plus the 14D->64D latent bridge, samples a 3DGS result with
Euler integration, renders the predicted object amodal video, and writes the
outputs in the same directory structure expected by the existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from train_fm import (
    ObjectFMSequenceDataset,
    build_arg_parser as build_train_arg_parser,
    build_backbone,
    build_hy3d_conditioner,
    build_hy3d_contexts,
    decode_gaussian_tokens,
    load_rgb_image,
    load_training_components,
    parse_block_list,
    resize_video_batch,
    resolve_repo_root,
    scale_camera_intrinsics,
)


def resolve_video_dir(input_dir: str, video_name: str) -> Path:
    base = Path(input_dir).expanduser().resolve()
    candidates = [
        base / video_name,
        base / "sequences" / video_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve sequence directory for video={video_name} under {input_dir}. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def _sorted_frame_paths(frame_dir: Path) -> Iterable[Path]:
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No RGB frames found under {frame_dir}")
    return paths


def _load_object_pose_sequence(seq_path: Path, num_frames: int) -> torch.Tensor:
    try:
        return ObjectFMSequenceDataset._load_object_pose_sequence(seq_path, num_frames)
    except RuntimeError:
        return torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)


def _normalize_pose_dim(body_pose: np.ndarray, target_dim: int = 144) -> np.ndarray:
    return ObjectFMSequenceDataset._normalize_pose_dim(body_pose, target_dim=target_dim)


def load_inference_inputs(
    video_dir: Path,
    *,
    processed_subdir: str,
    max_frames: int,
    background_value: float,
) -> Dict[str, torch.Tensor]:
    processed_dir = video_dir / processed_subdir
    cropped_dir = processed_dir / "cropped"
    rgb_dir = cropped_dir / "rgb"
    rgb_paths = list(_sorted_frame_paths(rgb_dir))
    if max_frames > 0:
        rgb_paths = rgb_paths[:max_frames]

    masks_npz = np.load(cropped_dir / "masks_raw.npz")
    masks_human = torch.from_numpy(masks_npz["human"]).float().unsqueeze(1)
    masks_object = torch.from_numpy(masks_npz["object"]).float().unsqueeze(1)

    smpl_npz = np.load(processed_dir / "smpl_params.npz")
    h_pose = torch.from_numpy(_normalize_pose_dim(smpl_npz["body_pose"].astype(np.float32), target_dim=144))

    meta_npz = np.load(cropped_dir / "meta.npz")
    fx = torch.from_numpy(meta_npz["fx"]).float()
    fy = torch.from_numpy(meta_npz["fy"]).float()
    cx = torch.from_numpy(meta_npz["cx"]).float()
    cy = torch.from_numpy(meta_npz["cy"]).float()
    intrinsics = torch.zeros(len(fx), 3, 3, dtype=torch.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    num_frames = min(
        len(rgb_paths),
        masks_human.shape[0],
        masks_object.shape[0],
        h_pose.shape[0],
        intrinsics.shape[0],
    )
    if num_frames <= 0:
        raise RuntimeError(f"No valid inference frames found under {video_dir}")

    rgb_paths = rgb_paths[:num_frames]
    rgb_clip = torch.stack([load_rgb_image(str(path)) for path in rgb_paths], dim=0)
    masks_human = masks_human[:num_frames]
    masks_object = masks_object[:num_frames]
    h_pose = h_pose[:num_frames]
    intrinsics = intrinsics[:num_frames]
    object_poses = _load_object_pose_sequence(video_dir, num_frames)[:num_frames]

    background = torch.full_like(rgb_clip, float(background_value))
    v_masked = rgb_clip * (1.0 - masks_human) + background * masks_human
    v_object_visible = rgb_clip * masks_object + background * (1.0 - masks_object)
    v_human_visible = rgb_clip * masks_human + background * (1.0 - masks_human)

    return {
        "rgb": rgb_clip.unsqueeze(0),
        "v_masked": v_masked.unsqueeze(0),
        "v_object_visible": v_object_visible.unsqueeze(0),
        "v_human_visible": v_human_visible.unsqueeze(0),
        "m_object": masks_object.unsqueeze(0),
        "m_human": masks_human.unsqueeze(0),
        "h_pose": h_pose.unsqueeze(0),
        "camera_intrinsics": intrinsics.unsqueeze(0),
        "object_poses": object_poses.unsqueeze(0),
        "sequence_name": video_dir.name,
    }


def build_runtime_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hunyuan latent-bridge FM inference.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--output_subdir", type=str, default="amodal")
    parser.add_argument("--gs_output_subdir", type=str, default="gs_init")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--num_ode_steps", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--video_h", type=int, default=256)
    parser.add_argument("--video_w", type=int, default=256)
    parser.add_argument("--prior_noise_std", type=float, default=1.0)
    parser.add_argument("--clamp_visible_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_fps", type=int, default=24)
    parser.add_argument("--background_value", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="float32", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--human_branch_mode", type=str, default="segmented_visible", choices=("segmented_visible", "skip"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def namespace_from_checkpoint_args(checkpoint_args: Dict[str, object]) -> argparse.Namespace:
    defaults = build_train_arg_parser().parse_args([])
    for key, value in checkpoint_args.items():
        setattr(defaults, key, value)
    return defaults


def strict_load_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], *, name: str) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load `{name}` strictly. Missing keys: {missing}. Unexpected keys: {unexpected}."
        )


def resolve_autocast_dtype(device: torch.device, precision: str) -> Optional[torch.dtype]:
    if device.type != "cuda" or precision == "float32":
        return None
    if precision == "bfloat16":
        return torch.bfloat16
    if precision in {"auto", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def get_inference_context(device: torch.device, precision: str):
    autocast_dtype = resolve_autocast_dtype(device, precision)
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


def save_video_branch(video: torch.Tensor, branch_dir: Path, *, save_frames: bool, fps: int) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().clamp(0.0, 1.0).cpu()
    frames_uint8 = []

    frames_dir = branch_dir / "frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame in enumerate(video):
        frame_uint8 = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        frames_uint8.append(frame_uint8)
        if save_frames:
            Image.fromarray(frame_uint8).save(frames_dir / f"{frame_idx:06d}.png")

    imageio.mimwrite(branch_dir / "inpaint_out.mp4", frames_uint8, fps=fps, quality=7)


def save_object_gaussians(
    raw_gs_tokens: torch.Tensor,
    gs_params: Dict[str, torch.Tensor],
    output_path: Path,
    *,
    sequence_name: str,
    num_ode_steps: int,
    num_frames: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "xyz": gs_params["means"].squeeze(0).cpu(),
        "rotation": gs_params["rotations"].squeeze(0).cpu(),
        "scaling": gs_params["scales"].squeeze(0).cpu(),
        "opacity": gs_params["opacities"].squeeze(0).cpu(),
        "shs": gs_params["shs"].squeeze(0).cpu(),
        "raw": raw_gs_tokens.squeeze(0).cpu(),
        "metadata": {
            "sequence_name": sequence_name,
            "num_frames": int(num_frames),
            "num_ode_steps": int(num_ode_steps),
        },
    }
    torch.save(payload, output_path)


def save_inference_metadata(
    metadata_path: Path,
    *,
    sequence_name: str,
    checkpoint: str,
    num_frames: int,
    num_points: int,
    num_ode_steps: int,
    render_size: Tuple[int, int],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sequence_name": sequence_name,
        "checkpoint": checkpoint,
        "num_frames": int(num_frames),
        "num_points": int(num_points),
        "num_ode_steps": int(num_ode_steps),
        "render_height": int(render_size[0]),
        "render_width": int(render_size[1]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def sample_object_latents(
    *,
    model: torch.nn.Module,
    cond_tokens: Dict[str, torch.Tensor],
    v_masked: torch.Tensor,
    m_human: torch.Tensor,
    h_pose: torch.Tensor,
    num_points: int,
    latent_dim: int,
    num_ode_steps: int,
    prior_noise_std: float,
    precision: str,
    device: torch.device,
) -> torch.Tensor:
    if num_ode_steps <= 0:
        raise ValueError(f"`num_ode_steps` must be positive, got {num_ode_steps}.")
    if num_points <= 0:
        raise ValueError(f"`num_points` must be positive, got {num_points}.")

    latents = torch.randn(1, num_points, latent_dim, device=device) * float(prior_noise_std)
    times = torch.linspace(0.0, 1.0, num_ode_steps + 1, device=device, dtype=latents.dtype)
    context_manager = get_inference_context(device, precision)

    with torch.no_grad():
        for step_idx in range(num_ode_steps):
            t_cur = times[step_idx].expand(1)
            dt = times[step_idx + 1] - times[step_idx]
            with context_manager:
                velocity = model(
                    x=latents,
                    t=t_cur,
                    contexts=cond_tokens,
                    v_masked=v_masked,
                    m_human=m_human,
                    h_pose=h_pose,
                ).sample
            latents = latents + dt.view(1, 1, 1) * velocity
    return latents


def run_joint_fm_inference(args: argparse.Namespace) -> Dict[str, Path]:
    repo_root = resolve_repo_root()
    torch_home = repo_root / "pretrained" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("`infer_fm.py` requires CUDA because 3DGS rendering uses diff-gaussian-rasterization.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "latent_bridge" not in checkpoint:
        raise KeyError(f"{args.checkpoint} is not a valid `train_fm.py` checkpoint.")

    checkpoint_args = namespace_from_checkpoint_args(checkpoint.get("args", {}))
    GaussianLatentBridge, Hunyuan3D2ControlNet, DiffRasterizationLayer, _ = load_training_components(repo_root)

    bridge_state = checkpoint["latent_bridge"]
    bridge_token_dim = int(bridge_state["encoder_proj.weight"].shape[1])
    bridge_latent_dim = int(bridge_state["encoder_proj.weight"].shape[0])

    backbone = build_backbone(checkpoint_args)
    model = Hunyuan3D2ControlNet(
        frozen_hunyuan_dit=backbone,
        condition_dim=checkpoint_args.condition_dim,
        inject_single_blocks=parse_block_list(str(checkpoint_args.inject_single_blocks)),
        attention_dropout=checkpoint_args.attention_dropout,
        proj_dropout=checkpoint_args.proj_dropout,
    )
    if bridge_latent_dim != model.input_dim:
        raise ValueError(
            f"Checkpoint latent bridge expects latent_dim={bridge_latent_dim}, "
            f"but the restored Hunyuan backbone uses input_dim={model.input_dim}."
        )
    latent_bridge = GaussianLatentBridge(
        token_dim=bridge_token_dim,
        latent_dim=bridge_latent_dim,
        encoder_depth=checkpoint_args.bridge_encoder_depth,
        decoder_depth=checkpoint_args.bridge_decoder_depth,
        dropout=checkpoint_args.bridge_dropout,
    )
    strict_load_state_dict(model, checkpoint["model"], name="model")
    strict_load_state_dict(latent_bridge, checkpoint["latent_bridge"], name="latent_bridge")

    model.to(device=device).eval()
    latent_bridge.to(device=device).eval()
    hy3d_conditioner, hy3d_image_processor = build_hy3d_conditioner(checkpoint_args, device)

    video_dir = resolve_video_dir(args.input_dir, args.video_name)
    inputs = load_inference_inputs(
        video_dir,
        processed_subdir=args.processed_subdir,
        max_frames=args.max_frames,
        background_value=args.background_value,
    )
    num_frames = int(inputs["v_masked"].shape[1])

    render_size = (int(args.video_h), int(args.video_w))
    source_size = (int(inputs["v_masked"].shape[-2]), int(inputs["v_masked"].shape[-1]))
    v_masked = inputs["v_masked"].to(device)
    m_human = inputs["m_human"].to(device)
    h_pose = inputs["h_pose"].to(device)
    m_object = inputs["m_object"].to(device)
    object_poses = inputs["object_poses"].to(device)
    v_object_visible = inputs["v_object_visible"].to(device)
    v_human_visible = inputs["v_human_visible"].to(device)
    camera_intrinsics = inputs["camera_intrinsics"].to(device)

    if source_size != render_size:
        v_object_visible_render = resize_video_batch(v_object_visible, size=render_size, mode="bilinear")
        v_human_visible_render = resize_video_batch(v_human_visible, size=render_size, mode="bilinear")
        m_human_render = resize_video_batch(m_human, size=render_size, mode="nearest")
        camera_intrinsics_render = scale_camera_intrinsics(
            camera_intrinsics,
            source_size=source_size,
            target_size=render_size,
        )
    else:
        v_object_visible_render = v_object_visible
        v_human_visible_render = v_human_visible
        m_human_render = m_human
        camera_intrinsics_render = camera_intrinsics

    contexts = build_hy3d_contexts(
        v_gt=v_object_visible,
        m_object=m_object,
        conditioner=hy3d_conditioner,
        image_processor=hy3d_image_processor,
        frame_policy=checkpoint_args.hy3d_condition_frame,
    )
    contexts = {key: value.to(device) for key, value in contexts.items()}

    latent_tokens = sample_object_latents(
        model=model,
        cond_tokens=contexts,
        v_masked=v_masked,
        m_human=m_human,
        h_pose=h_pose,
        num_points=args.num_points,
        latent_dim=model.input_dim,
        num_ode_steps=args.num_ode_steps,
        prior_noise_std=args.prior_noise_std,
        precision=args.precision,
        device=device,
    )

    with torch.no_grad():
        raw_gs_tokens = latent_bridge.decode(latent_tokens)
        gs_params = decode_gaussian_tokens(raw_gs_tokens)
        renderer = DiffRasterizationLayer(
            image_height=render_size[0],
            image_width=render_size[1],
        ).to(device)
        v_render = renderer(
            gs_params,
            object_poses=object_poses,
            camera_intrinsics=camera_intrinsics_render,
        )

    if args.clamp_visible_rgb:
        object_video = v_render * m_human_render + v_object_visible_render * (1.0 - m_human_render)
    else:
        object_video = v_render

    amodal_dir = video_dir / args.output_subdir
    gs_output_dir = video_dir / args.gs_output_subdir
    object_branch_dir = amodal_dir / "object_amodal"
    save_video_branch(
        object_video.squeeze(0),
        object_branch_dir,
        save_frames=args.save_frames,
        fps=args.save_fps,
    )

    if args.human_branch_mode == "segmented_visible":
        save_video_branch(
            v_human_visible_render.squeeze(0),
            amodal_dir / "human_amodal",
            save_frames=args.save_frames,
            fps=args.save_fps,
        )

    save_object_gaussians(
        raw_gs_tokens,
        gs_params,
        gs_output_dir / "G_o.pt",
        sequence_name=inputs["sequence_name"],
        num_ode_steps=args.num_ode_steps,
        num_frames=num_frames,
    )
    save_inference_metadata(
        gs_output_dir / "joint_fm_inference.json",
        sequence_name=inputs["sequence_name"],
        checkpoint=args.checkpoint,
        num_frames=num_frames,
        num_points=args.num_points,
        num_ode_steps=args.num_ode_steps,
        render_size=render_size,
    )

    print(
        f"[infer_fm] Saved object_amodal -> {object_branch_dir} | "
        f"G_o -> {gs_output_dir / 'G_o.pt'} | frames={num_frames} points={args.num_points}"
    )
    return {
        "video_dir": video_dir,
        "object_amodal_dir": object_branch_dir,
        "gs_path": gs_output_dir / "G_o.pt",
    }


def main() -> None:
    parser = build_runtime_arg_parser()
    args = parser.parse_args()
    run_joint_fm_inference(args)


if __name__ == "__main__":
    main()
