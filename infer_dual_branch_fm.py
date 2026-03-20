#!/usr/bin/env python3
"""
Run dual-branch co-generative Flow Matching inference for a single sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image

from dataset.dual_branch_fm_dataset import load_dual_branch_sequence_bundle, load_rgb_image
from model.dual_branch_cogenerative_fm import DecodedHOIState, DualBranchCoGenerativeFlowMatching
from train_dual_branch_fm import resize_video_batch, scale_camera_intrinsics
from train_dual_branch_fm import build_arg_parser as build_train_arg_parser


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


def namespace_from_checkpoint_args(checkpoint_args: Dict[str, object]) -> argparse.Namespace:
    parser = build_train_arg_parser()
    defaults = parser.parse_args([])
    for key, value in checkpoint_args.items():
        setattr(defaults, key, value)
    return defaults


def pad_sequence(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    if tensor.shape[0] == target_frames:
        return tensor
    if tensor.shape[0] > target_frames:
        return tensor[:target_frames]
    pad = tensor[-1:].expand(target_frames - tensor.shape[0], *tensor.shape[1:])
    return torch.cat([tensor, pad], dim=0)


def load_inference_clip(
    *,
    video_dir: Path,
    processed_subdir: str,
    gs_subdir: str,
    human_gaussian_source: str,
    clip_length: int,
    num_human_gaussians: int,
    num_object_gaussians: int,
    num_joints: int,
    contact_dim: int,
    background_value: float,
) -> Dict[str, torch.Tensor]:
    bundle = load_dual_branch_sequence_bundle(
        str(video_dir),
        processed_subdir=processed_subdir,
        gs_subdir=gs_subdir,
        human_gaussian_source=human_gaussian_source,
        num_human_gaussians=max(1, num_human_gaussians),
        num_object_gaussians=max(1, num_object_gaussians),
        num_joints=num_joints,
        contact_dim=contact_dim,
        require_gaussian_targets=False,
    )

    rgb = torch.stack([load_rgb_image(str(path)) for path in bundle["rgb_paths"]], dim=0)
    masks_human = bundle["masks_human"]
    masks_object = bundle["masks_object"]
    m_primary = bundle["m_primary"]
    m_secondary = bundle["m_secondary"]
    m_object_region = bundle["m_object_region"]
    depth = bundle["depth"]
    keypoint_heatmaps = bundle["keypoint_heatmaps"]
    camera_intrinsics = bundle["intrinsics"]

    rgb = pad_sequence(rgb, clip_length)
    masks_human = pad_sequence(masks_human, clip_length)
    masks_object = pad_sequence(masks_object, clip_length)
    m_primary = pad_sequence(m_primary, clip_length)
    m_secondary = pad_sequence(m_secondary, clip_length)
    m_object_region = pad_sequence(m_object_region, clip_length)
    depth = pad_sequence(depth, clip_length)
    keypoint_heatmaps = pad_sequence(keypoint_heatmaps, clip_length)
    camera_intrinsics = pad_sequence(camera_intrinsics, clip_length)

    background = torch.full_like(rgb, background_value)
    human_visible = rgb * masks_human + background * (1.0 - masks_human)
    object_visible = rgb * masks_object + background * (1.0 - masks_object)
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
        dim=1,
    )

    return {
        "rgb": rgb.unsqueeze(0),
        "human_visible": human_visible.unsqueeze(0),
        "object_visible": object_visible.unsqueeze(0),
        "condition_video": condition_video.unsqueeze(0),
        "masks_human": masks_human.unsqueeze(0),
        "masks_object": masks_object.unsqueeze(0),
        "camera_intrinsics": camera_intrinsics.unsqueeze(0),
        "sequence_name": bundle["sequence_name"],
    }


def save_video_branch(video: torch.Tensor, branch_dir: Path, *, save_frames: bool, fps: int) -> None:
    import imageio.v2 as imageio

    branch_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().clamp(0.0, 1.0).cpu()
    frames_uint8 = []
    frames_dir = branch_dir / "frames"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(video):
        frame_uint8 = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        frames_uint8.append(frame_uint8)
        if save_frames:
            Image.fromarray(frame_uint8).save(frames_dir / f"{frame_idx:06d}.png")
    imageio.mimwrite(branch_dir / "inpaint_out.mp4", frames_uint8, fps=fps, quality=7)


def save_gaussian_tokens(tokens: torch.Tensor, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "xyz": tokens[..., 0:3].squeeze(0).cpu(),
        "rotation": tokens[..., 3:7].squeeze(0).cpu(),
        "scaling": tokens[..., 7:10].squeeze(0).cpu(),
        "opacity": tokens[..., 10:11].squeeze(0).cpu(),
        "shs": tokens[..., 11:14].squeeze(0).cpu(),
        "raw": tokens.squeeze(0).cpu(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def save_combined_state(
    decoded_state: DecodedHOIState,
    output_dir: Path,
    *,
    sequence_name: str,
    num_ode_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "G_h": {
            "raw": decoded_state.human_gaussians.squeeze(0).cpu(),
        },
        "G_o": {
            "raw": decoded_state.object_gaussians.squeeze(0).cpu(),
        },
        "motion": {
            "joints_3d": decoded_state.joints_3d.squeeze(0).cpu(),
            "object_poses": decoded_state.object_transforms.squeeze(0).cpu(),
            "contact_signature": decoded_state.contact_signature.squeeze(0).cpu(),
        },
        "metadata": {
            "sequence_name": sequence_name,
            "num_frames": int(decoded_state.joints_3d.shape[1]),
            "num_ode_steps": int(num_ode_steps),
        },
    }
    torch.save(payload, output_dir / "gs_init_combined.pt")


def build_runtime_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual-branch co-generative Flow Matching inference.")
    parser.add_argument("--input_dir", type=str, required=False, default="")
    parser.add_argument("--video_name", type=str, required=False, default="")
    parser.add_argument("--checkpoint", type=str, required=False, default="")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--output_subdir", type=str, default="amodal")
    parser.add_argument("--gs_output_subdir", type=str, default="gs_init")
    parser.add_argument("--num_ode_steps", type=int, default=50)
    parser.add_argument("--prior_noise_std", type=float, default=1.0)
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clamp_visible_rgb", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_dual_branch_inference(args: argparse.Namespace) -> Dict[str, Path]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = namespace_from_checkpoint_args(checkpoint.get("args", {}))
    video_dir = resolve_video_dir(args.input_dir, args.video_name)
    inputs = load_inference_clip(
        video_dir=video_dir,
        processed_subdir=args.processed_subdir,
        gs_subdir=args.gs_subdir,
        human_gaussian_source=getattr(checkpoint_args, "human_gaussian_source", "smpl_mesh"),
        clip_length=checkpoint_args.clip_length,
        num_human_gaussians=checkpoint_args.num_human_gaussians,
        num_object_gaussians=checkpoint_args.num_object_gaussians,
        num_joints=checkpoint_args.num_joints,
        contact_dim=checkpoint_args.contact_dim,
        background_value=getattr(checkpoint_args, "background_value", 1.0),
    )
    condition_channels = int(inputs["condition_video"].shape[2])

    model = DualBranchCoGenerativeFlowMatching(
        hidden_dim=checkpoint_args.hidden_dim,
        num_heads=checkpoint_args.num_heads,
        depth=checkpoint_args.depth,
        mlp_ratio=checkpoint_args.mlp_ratio,
        dropout=checkpoint_args.dropout,
        condition_channels=condition_channels,
        video_channels=checkpoint_args.video_channels,
        patch_size=checkpoint_args.patch_size,
        num_frames=checkpoint_args.clip_length,
        image_height=checkpoint_args.image_height,
        image_width=checkpoint_args.image_width,
        num_human_gaussians=checkpoint_args.num_human_gaussians,
        num_object_gaussians=checkpoint_args.num_object_gaussians,
        num_joints=checkpoint_args.num_joints,
        contact_dim=checkpoint_args.contact_dim,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device=device).eval()
    condition_video = inputs["condition_video"].to(device)
    masks_human = inputs["masks_human"].to(device)
    masks_object = inputs["masks_object"].to(device)
    human_visible = inputs["human_visible"].to(device)
    object_visible = inputs["object_visible"].to(device)
    camera_intrinsics = inputs["camera_intrinsics"].to(device)

    if condition_video.shape[-2:] != (checkpoint_args.image_height, checkpoint_args.image_width):
        source_hw = condition_video.shape[-2:]
        condition_video = resize_video_batch(
            condition_video,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="bilinear",
        )
        human_visible = resize_video_batch(
            human_visible,
            size=(checkpoint_args.image_height, checkpoint_args.image_width),
            mode="bilinear",
        )
        object_visible = resize_video_batch(
            object_visible,
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
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics,
            source_size=(int(source_hw[-2]), int(source_hw[-1])),
            target_size=(checkpoint_args.image_height, checkpoint_args.image_width),
        )

    video_latents = torch.randn(
        1,
        model.video_codec.num_frames * model.video_codec.num_patches_per_frame,
        checkpoint_args.hidden_dim,
        device=device,
    ) * args.prior_noise_std
    state_latents = torch.randn(
        1,
        model.state_codec.total_tokens,
        checkpoint_args.hidden_dim,
        device=device,
    ) * args.prior_noise_std

    times = torch.linspace(0.0, 1.0, args.num_ode_steps + 1, device=device, dtype=video_latents.dtype)
    with torch.no_grad():
        for step_idx in range(args.num_ode_steps):
            t_cur = times[step_idx].expand(1)
            dt = times[step_idx + 1] - times[step_idx]
            output = model(
                video_xt=video_latents,
                state_xt=state_latents,
                timesteps=t_cur,
                condition_video=condition_video,
                camera_intrinsics=camera_intrinsics,
            )
            video_latents = video_latents + dt.view(1, 1, 1) * output.video_velocity
            state_latents = state_latents + dt.view(1, 1, 1) * output.state_velocity

        decoded_video = model.decode_video_tokens(video_latents)
        decoded_state = model.decode_state_tokens(state_latents)

    pred_human = decoded_video[:, :, :3]
    pred_object = decoded_video[:, :, 3:6]
    if args.clamp_visible_rgb:
        pred_human = pred_human * masks_object + human_visible * (1.0 - masks_object)
        pred_object = pred_object * masks_human + object_visible * (1.0 - masks_human)

    amodal_dir = video_dir / args.output_subdir
    gs_output_dir = video_dir / args.gs_output_subdir
    human_branch_dir = amodal_dir / "human_amodal"
    object_branch_dir = amodal_dir / "object_amodal"
    save_video_branch(pred_human.squeeze(0), human_branch_dir, save_frames=args.save_frames, fps=args.save_fps)
    save_video_branch(pred_object.squeeze(0), object_branch_dir, save_frames=args.save_frames, fps=args.save_fps)

    metadata = {
        "sequence_name": inputs["sequence_name"],
        "num_frames": int(decoded_video.shape[1]),
        "num_ode_steps": int(args.num_ode_steps),
    }
    save_gaussian_tokens(decoded_state.human_gaussians, gs_output_dir / "G_h.pt", metadata)
    save_gaussian_tokens(decoded_state.object_gaussians, gs_output_dir / "G_o.pt", metadata)
    save_combined_state(
        decoded_state,
        gs_output_dir,
        sequence_name=inputs["sequence_name"],
        num_ode_steps=args.num_ode_steps,
    )
    (gs_output_dir / "dual_branch_inference.json").write_text(json.dumps(metadata, indent=2))

    return {
        "video_dir": video_dir,
        "human_amodal_dir": human_branch_dir,
        "object_amodal_dir": object_branch_dir,
        "gs_dir": gs_output_dir,
    }


def main() -> None:
    args = build_runtime_arg_parser().parse_args()
    run_dual_branch_inference(args)


if __name__ == "__main__":
    main()
