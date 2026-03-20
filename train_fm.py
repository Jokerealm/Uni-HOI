#!/usr/bin/env python3
"""
Train Joint Video-3D Flow Matching with a Hunyuan3D-2 ControlNet adapter.

This script is intentionally centered on the new Hunyuan-based training path:
  1. Rectified Flow interpolation in Hunyuan's native latent space
  2. Trainable latent bridge between 14D 3DGS tokens and native 64D latents
  3. Hunyuan3D2ControlNet velocity prediction conditioned on interaction cues
  4. One-step Euler reconstruction in latent space
  5. Bridge decoding back to 3DGS + differentiable video rendering supervision
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


REQUIRED_HUNYUAN_ATTRS = (
    "latent_in",
    "time_in",
    "cond_in",
    "double_blocks",
    "single_blocks",
    "final_layer",
    "hidden_size",
    "num_heads",
)


def load_local_module(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_training_components(repo_root: Path):
    conditioned_mod = load_local_module(
        "conditioned_hunyuan_fm_local",
        str(repo_root / "model" / "conditioned_hunyuan_fm.py"),
    )
    renderer_mod = load_local_module(
        "joint_renderer_loss_local",
        str(repo_root / "model" / "joint_renderer_loss.py"),
    )
    return (
        conditioned_mod.GaussianLatentBridge,
        conditioned_mod.Hunyuan3D2ControlNet,
        renderer_mod.DiffRasterizationLayer,
        renderer_mod.JointVideo3DLoss,
    )


def is_hunyuan_compatible(module: object) -> bool:
    return all(hasattr(module, attr) for attr in REQUIRED_HUNYUAN_ATTRS)


def strip_state_dict_prefixes(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    prefixes = ("module.", "model.", "state_dict.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_state_dict_file(path: str) -> Dict[str, Tensor]:
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                f"`safetensors` is required to load {path}. Install it or use a `.ckpt`/`.pt` checkpoint."
            ) from exc
        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def resolve_hy3d_artifacts(
    model_path: str,
    subfolder: str,
    *,
    variant: str = "fp16",
    use_safetensors: bool = True,
) -> Tuple[Path, Path]:
    base = Path(model_path).expanduser().resolve()
    if subfolder and (base / subfolder).is_dir():
        base = base / subfolder

    config_path = base / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hunyuan config not found at {config_path}")

    extension = "safetensors" if use_safetensors else "ckpt"
    candidates = []
    if variant:
        candidates.extend(
            [
                base / f"model.{variant}.{extension}",
                base / f"model_{variant}.{extension}",
            ]
        )
    candidates.extend(
        [
            base / f"model.{extension}",
            base / f"model_{extension}.{extension}",
        ]
    )
    checkpoint_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not find a Hunyuan checkpoint under {base}. Tried: {[str(path) for path in candidates]}"
        )
    return config_path, checkpoint_path


def extract_prefixed_state_dict(
    state_dict: Dict[str, Tensor],
    prefix: str,
    *,
    fallback_to_full: bool = False,
) -> Dict[str, Tensor]:
    dotted_prefix = f"{prefix}."
    extracted = {
        key[len(dotted_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(dotted_prefix)
    }
    return extracted if extracted else (state_dict if fallback_to_full else {})


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    angle = torch.linalg.norm(axis_angle)
    if float(angle) < 1e-8:
        return torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)

    axis = axis_angle / angle
    x, y, z = axis.unbind(dim=0)
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=axis_angle.dtype,
        device=axis_angle.device,
    )
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    return identity + sin * K + (1.0 - cos) * (K @ K)


def make_extrinsic_from_axis_angle_and_translation(angle: np.ndarray, translation: np.ndarray) -> Tensor:
    angle_t = torch.as_tensor(angle, dtype=torch.float32)
    translation_t = torch.as_tensor(translation, dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = axis_angle_to_matrix(angle_t)
    extrinsic[:3, 3] = translation_t
    return extrinsic


def load_rgb_image(path: str) -> Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_frame_to_rgba_pil(rgb: Tensor, alpha: Tensor) -> Image.Image:
    rgb_np = (rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    alpha_np = (alpha.detach().clamp(0.0, 1.0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    rgba_np = np.concatenate([rgb_np, alpha_np[..., None]], axis=-1)
    return Image.fromarray(rgba_np)


def choose_reference_frame_indices(num_frames: int, batch_size: int, policy: str) -> List[int]:
    if policy == "first":
        return [0] * batch_size
    if policy == "last":
        return [num_frames - 1] * batch_size
    if policy == "random":
        return [random.randrange(num_frames) for _ in range(batch_size)]
    return [num_frames // 2] * batch_size


def discover_sequence_dirs(root: str) -> List[str]:
    root_path = Path(root).expanduser().resolve()
    candidates: List[Path] = []

    def is_sequence_dir(path: Path) -> bool:
        return (
            (path / "processed" / "cropped" / "meta.npz").is_file()
            and (path / "processed" / "cropped" / "rgb").is_dir()
            and (path / "gs_init" / "G_o.pt").is_file()
        )

    if is_sequence_dir(root_path):
        return [str(root_path)]

    for base in (root_path, root_path / "sequences"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and is_sequence_dir(child):
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No valid sequence directories found under {root_path}. "
            "Expected `processed/cropped/meta.npz`, `processed/cropped/rgb/`, and `gs_init/G_o.pt`."
        )
    return [str(path) for path in candidates]


class ObjectFMSequenceDataset(Dataset):
    """Minimal dataset for object-centric FM training on cropped BEHAVE-style sequences."""

    def __init__(
        self,
        data_root: str,
        clip_length: int,
        clip_stride: int,
        gaussian_relpath: str = "gs_init/G_o.pt",
        max_sequences: int = 0,
        background_value: float = 1.0,
        prefer_cropped_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.gaussian_relpath = gaussian_relpath
        self.background_value = float(background_value)
        self.prefer_cropped_supervision = prefer_cropped_supervision
        self.sequence_dirs = discover_sequence_dirs(data_root)
        if max_sequences > 0:
            self.sequence_dirs = self.sequence_dirs[:max_sequences]

        self._cache: Dict[str, Dict[str, object]] = {}
        self.samples: List[Tuple[str, int]] = []

        for seq_dir in self.sequence_dirs:
            bundle = self._load_sequence_bundle(seq_dir)
            num_frames = int(bundle["num_frames"])
            if num_frames < self.clip_length:
                continue
            for start in range(0, num_frames - self.clip_length + 1, self.clip_stride):
                self.samples.append((seq_dir, start))

        if not self.samples:
            raise RuntimeError(
                f"No training clips found under {data_root} for clip_length={self.clip_length}."
            )

        self.gaussian_dim = int(self._load_sequence_bundle(self.sequence_dirs[0])["x1"].shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sequence_bundle(self, seq_dir: str) -> Dict[str, object]:
        if seq_dir in self._cache:
            return self._cache[seq_dir]

        seq_path = Path(seq_dir)
        rgb_dir = seq_path / "processed" / "cropped" / "rgb"
        rgb_paths = sorted(glob.glob(str(rgb_dir / "*.png")))
        if not rgb_paths:
            rgb_paths = sorted(glob.glob(str(rgb_dir / "*.jpg")))
        if not rgb_paths:
            raise FileNotFoundError(f"No cropped RGB frames found under {rgb_dir}")

        mask_npz = np.load(seq_path / "processed" / "cropped" / "masks_raw.npz")
        masks_human = torch.from_numpy(mask_npz["human"]).float().unsqueeze(1)
        masks_object = torch.from_numpy(mask_npz["object"]).float().unsqueeze(1)

        smpl_npz = np.load(seq_path / "processed" / "smpl_params.npz")
        body_pose = smpl_npz["body_pose"].astype(np.float32)
        h_pose = torch.from_numpy(self._normalize_pose_dim(body_pose, target_dim=144))

        meta_npz = np.load(seq_path / "processed" / "cropped" / "meta.npz")
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

        object_poses = self._load_object_pose_sequence(seq_path, len(rgb_paths))
        gaussian_target = torch.load(seq_path / self.gaussian_relpath, map_location="cpu", weights_only=False)
        if "raw" in gaussian_target:
            x1 = gaussian_target["raw"].float()
        else:
            x1 = torch.cat(
                [
                    gaussian_target["xyz"].float(),
                    F.normalize(gaussian_target["rotation"].float(), dim=-1),
                    gaussian_target["scaling"].float().clamp(min=1e-6),
                    gaussian_target["opacity"].float().clamp(0.0, 1.0),
                    gaussian_target["shs"].float().clamp(0.0, 1.0),
                ],
                dim=-1,
            )

        num_frames = min(len(rgb_paths), masks_human.shape[0], masks_object.shape[0], h_pose.shape[0], intrinsics.shape[0], object_poses.shape[0])
        bundle = {
            "rgb_paths": rgb_paths[:num_frames],
            "masks_human": masks_human[:num_frames],
            "masks_object": masks_object[:num_frames],
            "h_pose": h_pose[:num_frames],
            "intrinsics": intrinsics[:num_frames],
            "object_poses": object_poses[:num_frames],
            "x1": x1,
            "sequence_name": seq_path.name,
            "num_frames": num_frames,
        }
        self._cache[seq_dir] = bundle
        return bundle

    @staticmethod
    def _normalize_pose_dim(body_pose: np.ndarray, target_dim: int) -> np.ndarray:
        if body_pose.shape[1] == target_dim:
            return body_pose
        if body_pose.shape[1] > target_dim:
            return body_pose[:, :target_dim]

        padded = np.zeros((body_pose.shape[0], target_dim), dtype=np.float32)
        padded[:, : body_pose.shape[1]] = body_pose
        return padded

    @staticmethod
    def _load_object_pose_sequence(seq_path: Path, num_frames: int) -> Tensor:
        timestep_dirs = sorted(glob.glob(str(seq_path / "t*.000")))
        poses: List[Tensor] = []
        for t_dir in timestep_dirs[:num_frames]:
            fit_paths = sorted(
                glob.glob(os.path.join(t_dir, "*", "fit01", "*_fit.pkl"))
            )
            fit_paths = [path for path in fit_paths if "/person/" not in path]
            if not fit_paths:
                poses.append(torch.eye(4, dtype=torch.float32))
                continue

            with open(fit_paths[0], "rb") as f:
                fit = pickle.load(f, encoding="latin1")
            angle = np.asarray(fit.get("angle", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            trans = np.asarray(fit.get("trans", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            poses.append(make_extrinsic_from_axis_angle_and_translation(angle, trans))

        if not poses:
            raise RuntimeError(f"No object pose files found under {seq_path}")
        return torch.stack(poses, dim=0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        seq_dir, start = self.samples[index]
        bundle = self._load_sequence_bundle(seq_dir)
        end = start + self.clip_length

        rgb_clip = torch.stack([load_rgb_image(path) for path in bundle["rgb_paths"][start:end]], dim=0)
        m_human = bundle["masks_human"][start:end].clone()
        m_object = bundle["masks_object"][start:end].clone()
        h_pose = bundle["h_pose"][start:end].clone()
        camera_intrinsics = bundle["intrinsics"][start:end].clone()
        object_poses = bundle["object_poses"][start:end].clone()
        x1 = bundle["x1"].clone()

        bg = torch.full_like(rgb_clip, self.background_value)
        v_masked = rgb_clip * (1.0 - m_human) + bg * m_human
        v_gt = rgb_clip * m_object + bg * (1.0 - m_object)

        return {
            "v_masked": v_masked,
            "m_object": m_object,
            "m_human": m_human,
            "h_pose": h_pose,
            "v_gt": v_gt,
            "x1": x1,
            "object_poses": object_poses,
            "camera_intrinsics": camera_intrinsics,
            "sequence_name": bundle["sequence_name"],
        }


def load_hy3d_component_bundle(
    args: argparse.Namespace,
    *,
    load_conditioner: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, object]]]:
    if not args.hy3d_model_path:
        return None, None, None

    config_path, checkpoint_path = resolve_hy3d_artifacts(
        args.hy3d_model_path,
        args.hy3d_subfolder,
        variant=args.hy3d_variant,
        use_safetensors=args.hy3d_use_safetensors,
    )
    if not load_conditioner:
        return config_path, checkpoint_path, None

    try:
        import yaml
        from hy3dgen.shapegen.pipelines import instantiate_from_config
    except ImportError as exc:
        raise ImportError(
            "Loading the Hunyuan conditioner requires `pyyaml` and the local `hy3dgen` package."
        ) from exc

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bundle = {
        "config": config,
        "instantiate_from_config": instantiate_from_config,
    }
    return config_path, checkpoint_path, bundle


def build_hy3d_conditioner(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, object]:
    _, checkpoint_path, bundle = load_hy3d_component_bundle(args, load_conditioner=True)
    if bundle is None:
        raise RuntimeError("Failed to resolve the Hunyuan conditioner bundle.")

    instantiate_from_config = bundle["instantiate_from_config"]
    config = bundle["config"]

    conditioner = instantiate_from_config(config["conditioner"])
    image_processor = instantiate_from_config(config["image_processor"])

    state_dict = load_state_dict_file(str(checkpoint_path))
    conditioner_state = extract_prefixed_state_dict(state_dict, "conditioner")
    if not conditioner_state:
        raise KeyError(f"No `conditioner.*` weights found in {checkpoint_path}")
    missing, unexpected = conditioner.load_state_dict(conditioner_state, strict=False)

    if args.hy3d_condition_dtype == "auto":
        if device.type == "cuda" and args.mixed_precision in {"fp16", "bf16"}:
            conditioner_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        else:
            conditioner_dtype = torch.float32
    elif args.hy3d_condition_dtype == "float16":
        conditioner_dtype = torch.float16
    else:
        conditioner_dtype = torch.float32

    conditioner.requires_grad_(False)
    conditioner.eval()
    conditioner.to(device=device, dtype=conditioner_dtype)

    print(
        f"[train_fm] Loaded Hunyuan conditioner from {checkpoint_path}. "
        f"missing={len(missing)} unexpected={len(unexpected)} dtype={conditioner_dtype}"
    )
    return conditioner, image_processor


@torch.no_grad()
def build_hy3d_contexts(
    *,
    v_gt: Tensor,
    m_object: Tensor,
    conditioner: nn.Module,
    image_processor: object,
    frame_policy: str,
) -> Dict[str, Tensor]:
    batch_size, num_frames = v_gt.shape[:2]
    frame_indices = choose_reference_frame_indices(num_frames, batch_size, frame_policy)

    processor_outputs = []
    for batch_idx, frame_idx in enumerate(frame_indices):
        rgba_image = tensor_frame_to_rgba_pil(v_gt[batch_idx, frame_idx], m_object[batch_idx, frame_idx])
        processor_outputs.append(image_processor(rgba_image))

    cond_inputs = {}
    for key in processor_outputs[0].keys():
        if torch.is_tensor(processor_outputs[0][key]):
            cond_inputs[key] = torch.cat([output[key] for output in processor_outputs], dim=0)
        else:
            cond_inputs[key] = [output[key] for output in processor_outputs]

    contexts = conditioner(**cond_inputs)
    return {
        key: value
        for key, value in contexts.items()
        if torch.is_tensor(value)
    }


def build_backbone_from_factory(args: argparse.Namespace) -> nn.Module:
    if ":" not in args.backbone_factory:
        raise ValueError("`--backbone_factory` must have the form `module.submodule:function_name`.")
    module_name, function_name = args.backbone_factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    backbone = factory(args)
    if not is_hunyuan_compatible(backbone):
        raise ValueError(
            f"Factory `{args.backbone_factory}` returned an incompatible backbone. "
            f"Expected attrs: {REQUIRED_HUNYUAN_ATTRS}"
        )
    return backbone


def build_backbone_from_hy3dgen(args: argparse.Namespace) -> nn.Module:
    try:
        from hy3dgen.shapegen.models import Hunyuan3DDiT
    except ImportError as exc:
        raise ImportError(
            "`hy3dgen` is required for `--backbone_type hy3dgen`. "
            "Install Tencent Hunyuan3D-2's Python package or use `--backbone_type factory`."
        ) from exc

    backbone = Hunyuan3DDiT(
        in_channels=args.hy3d_in_channels,
        context_in_dim=args.hy3d_context_in_dim,
        hidden_size=args.hy3d_hidden_size,
        mlp_ratio=args.hy3d_mlp_ratio,
        num_heads=args.hy3d_num_heads,
        depth=args.hy3d_depth,
        depth_single_blocks=args.hy3d_single_depth,
        axes_dim=[args.hy3d_axes_dim],
        theta=args.hy3d_theta,
        qkv_bias=args.hy3d_qkv_bias,
    )

    checkpoint_path = args.backbone_checkpoint
    if not checkpoint_path and args.hy3d_model_path:
        _, resolved_checkpoint_path, _ = load_hy3d_component_bundle(args, load_conditioner=False)
        checkpoint_path = str(resolved_checkpoint_path)

    if checkpoint_path:
        state_dict = load_state_dict_file(checkpoint_path)
        state_dict = strip_state_dict_prefixes(extract_prefixed_state_dict(state_dict, "model", fallback_to_full=True))
        checkpoint_input_dim = int(state_dict["latent_in.weight"].shape[1])
        checkpoint_output_dim = int(state_dict["final_layer.linear.weight"].shape[0])
        if checkpoint_input_dim != args.hy3d_in_channels or checkpoint_output_dim != args.hy3d_in_channels:
            raise ValueError(
                f"Hunyuan checkpoint expects native latent dim {checkpoint_input_dim}, "
                f"but `hy3d_in_channels={args.hy3d_in_channels}`. "
                "Use the native Hunyuan latent width and align 3DGS tokens via the latent bridge instead."
            )
        model_state = backbone.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = backbone.load_state_dict(compatible, strict=False)
        print(
            f"[train_fm] Loaded {len(compatible)} Hunyuan params from {checkpoint_path}. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    return backbone


def build_backbone(args: argparse.Namespace) -> nn.Module:
    if args.backbone_type == "factory":
        return build_backbone_from_factory(args)
    if args.backbone_type == "hy3dgen":
        return build_backbone_from_hy3dgen(args)
    raise ValueError(f"Unsupported backbone type: {args.backbone_type}")


def decode_gaussian_tokens(x: Tensor) -> Dict[str, Tensor]:
    if x.shape[-1] < 14:
        raise ValueError(f"Expected Gaussian token dim >= 14, got {x.shape[-1]}.")
    return {
        "means": x[..., 0:3],
        "rotations": F.normalize(x[..., 3:7], dim=-1),
        "scales": x[..., 7:10].clamp(min=1e-6),
        "opacities": x[..., 10:11].clamp(0.0, 1.0),
        "shs": x[..., 11:14].clamp(0.0, 1.0),
    }


def resize_video_batch(video: Tensor, size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
    if video.ndim != 5:
        raise ValueError(f"`video` must have shape [B, T, C, H, W], got {tuple(video.shape)}.")
    batch_size, num_frames, channels, _, _ = video.shape
    video = video.reshape(batch_size * num_frames, channels, video.shape[-2], video.shape[-1])
    video = F.interpolate(video, size=size, mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)
    return video.reshape(batch_size, num_frames, channels, size[0], size[1])


def scale_camera_intrinsics(
    camera_intrinsics: Tensor,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tensor:
    source_h, source_w = source_size
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return camera_intrinsics

    scale_x = float(target_w) / float(source_w)
    scale_y = float(target_h) / float(source_h)
    scaled = camera_intrinsics.clone()
    scaled[..., 0, 0] = scaled[..., 0, 0] * scale_x
    scaled[..., 1, 1] = scaled[..., 1, 1] * scale_y
    scaled[..., 0, 2] = scaled[..., 0, 2] * scale_x
    scaled[..., 1, 2] = scaled[..., 1, 2] * scale_y
    return scaled


def compute_bridge_reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(prediction, target)


def collect_trainable_parameters(*modules: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for module in modules:
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return params


def assert_backbone_is_frozen(model: nn.Module) -> None:
    backbone_params = list(model.frozen_hunyuan_dit.parameters())
    if any(parameter.requires_grad for parameter in backbone_params):
        raise AssertionError("Frozen Hunyuan backbone contains trainable parameters.")


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    latent_bridge: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step:07d}.pt"
    state = {
        "model": accelerator.unwrap_model(model).state_dict(),
        "latent_bridge": accelerator.unwrap_model(latent_bridge).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "args": vars(args),
    }
    accelerator.save(state, str(checkpoint_path))


def resume_if_available(
    args: argparse.Namespace,
    model: nn.Module,
    latent_bridge: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
) -> int:
    if not args.resume_checkpoint:
        return 0
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    if "latent_bridge" in checkpoint:
        latent_bridge.load_state_dict(checkpoint["latent_bridge"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("step", 0))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Hunyuan ControlNet FM model.")

    parser.add_argument("--data_root", type=str, default="sample_data/behave_1pct/sequences")
    parser.add_argument("--output_dir", type=str, default="outputs/train_fm")
    parser.add_argument("--project_name", type=str, default="hunyuan-fm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_with", type=str, default="tensorboard", choices=("tensorboard", "wandb", "none"))
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--resume_checkpoint", type=str, default="")

    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--clip_stride", type=int, default=4)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--gaussian_relpath", type=str, default="gs_init/G_o.pt")
    parser.add_argument("--background_value", type=float, default=1.0)

    parser.add_argument("--backbone_type", type=str, default="hy3dgen", choices=("hy3dgen", "factory"))
    parser.add_argument("--backbone_factory", type=str, default="")
    parser.add_argument("--backbone_checkpoint", type=str, default="")
    parser.add_argument("--hy3d_model_path", type=str, default="/data4/guanz/models/Hunyuan3D-2")
    parser.add_argument("--hy3d_subfolder", type=str, default="hunyuan3d-dit-v2-0")
    parser.add_argument("--hy3d_variant", type=str, default="fp16")
    parser.add_argument("--hy3d_use_safetensors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--hy3d_in_channels", type=int, default=64)
    parser.add_argument("--hy3d_context_in_dim", type=int, default=1536)
    parser.add_argument("--hy3d_hidden_size", type=int, default=1024)
    parser.add_argument("--hy3d_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--hy3d_num_heads", type=int, default=16)
    parser.add_argument("--hy3d_depth", type=int, default=16)
    parser.add_argument("--hy3d_single_depth", type=int, default=32)
    parser.add_argument("--hy3d_axes_dim", type=int, default=64)
    parser.add_argument("--hy3d_theta", type=int, default=10000)
    parser.add_argument("--hy3d_qkv_bias", action="store_true", default=True)

    parser.add_argument("--condition_dim", type=int, default=1024)
    parser.add_argument("--inject_single_blocks", type=str, default="")
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--bridge_encoder_depth", type=int, default=2)
    parser.add_argument("--bridge_decoder_depth", type=int, default=2)
    parser.add_argument("--bridge_dropout", type=float, default=0.0)
    parser.add_argument("--bridge_recon_weight", type=float, default=0.1)
    parser.add_argument("--bridge_prior_weight", type=float, default=0.01)
    parser.add_argument("--hy3d_condition_frame", type=str, default="middle", choices=("first", "middle", "last", "random"))
    parser.add_argument("--hy3d_condition_dtype", type=str, default="auto", choices=("auto", "float16", "float32"))

    parser.add_argument("--render_height", type=int, default=256)
    parser.add_argument("--render_width", type=int, default=256)

    parser.add_argument("--flow_matching_weight", type=float, default=1.0)
    parser.add_argument("--video_l1_weight", type=float, default=1.0)
    parser.add_argument("--video_perceptual_weight", type=float, default=0.1)
    parser.add_argument("--lpips_backbone", type=str, default="vgg")

    return parser


def parse_block_list(spec: str) -> Tuple[int, ...]:
    if not spec.strip():
        return ()
    return tuple(sorted({int(item) for item in spec.split(",") if item.strip()}))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    torch_home = repo_root / "pretrained" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_with = None if args.log_with == "none" else args.log_with
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        cpu=args.cpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_config=project_config,
    )
    set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process and log_with is not None:
        accelerator.init_trackers(args.project_name, config=vars(args))

    GaussianLatentBridge, Hunyuan3D2ControlNet, DiffRasterizationLayer, JointVideo3DLoss = load_training_components(repo_root)

    dataset = ObjectFMSequenceDataset(
        data_root=args.data_root,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        gaussian_relpath=args.gaussian_relpath,
        max_sequences=args.max_sequences,
        background_value=args.background_value,
    )

    backbone = build_backbone(args)
    if not is_hunyuan_compatible(backbone):
        raise ValueError(f"The constructed backbone is not Hunyuan-compatible: {type(backbone)}")

    model = Hunyuan3D2ControlNet(
        frozen_hunyuan_dit=backbone,
        condition_dim=args.condition_dim,
        inject_single_blocks=parse_block_list(args.inject_single_blocks),
        attention_dropout=args.attention_dropout,
        proj_dropout=args.proj_dropout,
    )
    assert_backbone_is_frozen(model)
    latent_bridge = GaussianLatentBridge(
        token_dim=dataset.gaussian_dim,
        latent_dim=model.input_dim,
        encoder_depth=args.bridge_encoder_depth,
        decoder_depth=args.bridge_decoder_depth,
        dropout=args.bridge_dropout,
    )

    hy3d_conditioner, hy3d_image_processor = build_hy3d_conditioner(args, accelerator.device)

    renderer = DiffRasterizationLayer(
        image_height=args.render_height,
        image_width=args.render_width,
    )
    criterion = JointVideo3DLoss(
        flow_matching_weight=args.flow_matching_weight,
        video_l1_weight=args.video_l1_weight,
        video_perceptual_weight=args.video_perceptual_weight,
        lpips_backbone=args.lpips_backbone,
    )

    trainable_params = collect_trainable_parameters(model, latent_bridge)
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)

    global_step = resume_if_available(args, model, latent_bridge, optimizer, scheduler)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model, latent_bridge, optimizer, dataloader, scheduler, renderer, criterion = accelerator.prepare(
        model, latent_bridge, optimizer, dataloader, scheduler, renderer, criterion
    )

    if accelerator.is_main_process:
        num_trainable = sum(parameter.numel() for parameter in trainable_params)
        print(
            f"[train_fm] dataset clips={len(dataset)} | trainable params={num_trainable / 1e6:.2f}M "
            f"| renderer={renderer.backend} | hy3d_condition=object_rgba | latent_bridge={dataset.gaussian_dim}->{model.input_dim}"
        )

    model.train()
    latent_bridge.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    while global_step < args.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                x1_gs = batch["x1"].to(accelerator.device)
                v_masked = batch["v_masked"].to(accelerator.device)
                m_object = batch["m_object"].to(accelerator.device)
                m_human = batch["m_human"].to(accelerator.device)
                h_pose = batch["h_pose"].to(accelerator.device)
                v_gt = batch["v_gt"].to(accelerator.device)
                object_poses = batch["object_poses"].to(accelerator.device)
                camera_intrinsics = batch["camera_intrinsics"].to(accelerator.device)
                source_hw = (int(v_gt.shape[-2]), int(v_gt.shape[-1]))
                target_hw = (args.render_height, args.render_width)
                if source_hw != target_hw:
                    v_gt_for_loss = resize_video_batch(v_gt, size=target_hw, mode="bilinear")
                    camera_intrinsics_render = scale_camera_intrinsics(
                        camera_intrinsics,
                        source_size=source_hw,
                        target_size=target_hw,
                    )
                else:
                    v_gt_for_loss = v_gt
                    camera_intrinsics_render = camera_intrinsics

                x1_latent = latent_bridge.encode(x1_gs)
                x1_gs_recon = latent_bridge.decode(x1_latent)

                batch_size = x1_latent.shape[0]
                t = torch.rand(batch_size, device=accelerator.device, dtype=x1_latent.dtype)
                t_3d = t.view(batch_size, 1, 1)

                x0 = torch.randn_like(x1_latent)
                x_t = t_3d * x1_latent + (1.0 - t_3d) * x0
                v_target = x1_latent - x0

                contexts = build_hy3d_contexts(
                    v_gt=v_gt,
                    m_object=m_object,
                    conditioner=hy3d_conditioner,
                    image_processor=hy3d_image_processor,
                    frame_policy=args.hy3d_condition_frame,
                )

                output = model(
                    x=x_t,
                    t=t,
                    contexts=contexts,
                    v_masked=v_masked,
                    m_human=m_human,
                    h_pose=h_pose,
                )
                v_pred = output.sample
                x_hat_1_latent = x_t + (1.0 - t_3d) * v_pred
                x_hat_1_gs = latent_bridge.decode(x_hat_1_latent)

                gs_params = decode_gaussian_tokens(x_hat_1_gs)
                v_render = renderer(
                    gs_params,
                    object_poses=object_poses,
                    camera_intrinsics=camera_intrinsics_render,
                )
                loss_main = criterion(v_pred=v_pred, v_target=v_target, v_render=v_render, v_gt=v_gt_for_loss)
                loss_bridge_recon = compute_bridge_reconstruction_loss(x1_gs_recon, x1_gs)
                loss_bridge_prior = latent_bridge.latent_prior_loss(x1_latent)
                loss = (
                    loss_main
                    + args.bridge_recon_weight * loss_bridge_recon
                    + args.bridge_prior_weight * loss_bridge_prior
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            reduced_metrics = {
                "loss_total": reduced_loss.item(),
                "lr": scheduler.get_last_lr()[0],
            }
            for key, value in criterion.last_loss_dict.items():
                metric_name = "loss_joint_core" if key == "loss_total" else key
                reduced_metrics[metric_name] = accelerator.reduce(
                    value.to(accelerator.device), reduction="mean"
                ).item()
            reduced_metrics["loss_bridge_recon"] = accelerator.reduce(
                loss_bridge_recon.detach(), reduction="mean"
            ).item()
            reduced_metrics["loss_bridge_prior"] = accelerator.reduce(
                loss_bridge_prior.detach(), reduction="mean"
            ).item()

            if global_step % args.log_every == 0 and accelerator.is_main_process and log_with is not None:
                accelerator.log(reduced_metrics, step=global_step)

            if global_step % args.print_every == 0 and accelerator.is_main_process:
                elapsed = time.time() - start_time
                steps_done = max(global_step, 1)
                eta = (elapsed / steps_done) * max(args.max_steps - global_step, 0)
                print(
                    f"[train_fm] step={global_step:07d} "
                    f"loss={reduced_metrics['loss_total']:.4f} "
                    f"core={reduced_metrics['loss_joint_core']:.4f} "
                    f"fm={reduced_metrics['loss_flow_matching']:.4f} "
                    f"l1={reduced_metrics['loss_video_l1']:.4f} "
                    f"lpips={reduced_metrics['loss_video_lpips']:.4f} "
                    f"bridge={reduced_metrics['loss_bridge_recon']:.4f} "
                    f"prior={reduced_metrics['loss_bridge_prior']:.4f} "
                    f"lr={reduced_metrics['lr']:.2e} "
                    f"eta={eta / 3600.0:.2f}h"
                )

            if global_step % args.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, model, latent_bridge, optimizer, scheduler, global_step, args)

            if global_step >= args.max_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, latent_bridge, optimizer, scheduler, global_step, args)
    accelerator.end_training()


if __name__ == "__main__":
    main()
