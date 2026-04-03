"""
Dataset utilities for dual-branch co-generative Flow Matching.

The dataset consumes Step-1 assets plus an asymmetric teacher contract:

- processed/cropped/rgb/*
- processed/cropped/masks_raw.npz
- processed/cropped/region_masks.npz
- processed/cropped/depth_aligned.npz
- processed/cropped/meta.npz
- processed/smpl_params.npz
  (default human pseudo-GT comes from `vertices/faces`; `G_h.pt` is only needed in `teacher` mode)
- gs_init/G_o.pt

It returns clips that jointly supervise:
- video branch conditions
- pseudo object amodal targets
- 4D state targets (human/object Gaussians, joints, object motion, contact)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

INDEX_CACHE_VERSION = 5
VALID_HUMAN_GAUSSIAN_SOURCES = ("smpl_mesh", "teacher")
DUAL_BRANCH_TARGET_CACHE_VERSION = 1
DUAL_BRANCH_TARGETS_FILENAME = "dual_branch_targets.npz"
SMPL_HUMAN_GAUSSIANS_FILENAME = "G_h_smpl.pt"
DEFAULT_KEYPOINT_HEATMAP_SIGMA = 6.0
DEFAULT_CONTACT_SIGNATURE_DIM = 4


def load_rgb_image(path: str) -> Tensor:
    image = load_rgb_image_uint8(path)
    return image.float().div(255.0)


def load_rgb_image_uint8(path: str) -> Tensor:
    image = Image.open(path).convert("RGB")
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _sorted_image_paths(frame_dir: Path) -> List[Path]:
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No RGB frames found under {frame_dir}")
    return paths


def _normalize_pose_dim(body_pose: np.ndarray, target_dim: int = 144) -> np.ndarray:
    if body_pose.shape[1] == target_dim:
        return body_pose
    if body_pose.shape[1] > target_dim:
        return body_pose[:, :target_dim]
    padded = np.zeros((body_pose.shape[0], target_dim), dtype=np.float32)
    padded[:, : body_pose.shape[1]] = body_pose
    return padded


def _normalize_keypoints_2d_targets(keypoints_2d: np.ndarray, num_joints: int) -> Tensor:
    keypoints = torch.from_numpy(np.asarray(keypoints_2d, dtype=np.float32))
    if keypoints.ndim != 3 or keypoints.shape[-1] not in (2, 3):
        raise ValueError(
            f"Expected keypoints_2d with shape [T, J, 2/3], got {tuple(keypoints.shape)}."
        )
    if keypoints.shape[-1] == 2:
        confidence = torch.ones(
            keypoints.shape[0],
            keypoints.shape[1],
            1,
            dtype=keypoints.dtype,
        )
        keypoints = torch.cat([keypoints, confidence], dim=-1)
    keypoints = keypoints[:, :num_joints]
    if keypoints.shape[1] < num_joints:
        pad = num_joints - keypoints.shape[1]
        padding = keypoints.new_zeros(keypoints.shape[0], pad, keypoints.shape[2])
        keypoints = torch.cat([keypoints, padding], dim=1)
    return keypoints


def _resolve_joint_targets_from_smpl_params(smpl_params: Dict[str, np.ndarray], num_joints: int) -> Tensor:
    if "joints_3d" in smpl_params:
        joints = torch.from_numpy(smpl_params["joints_3d"].astype(np.float32))
    elif "keypoints_3d" in smpl_params:
        joints = torch.from_numpy(smpl_params["keypoints_3d"].astype(np.float32))
    else:
        body_pose = torch.from_numpy(_normalize_pose_dim(smpl_params["body_pose"].astype(np.float32), target_dim=144))
        joints = body_pose.new_zeros(body_pose.shape[0], num_joints, 3)
    if joints.ndim != 3:
        raise ValueError(f"Expected joints tensor with shape [T, J, 3], got {tuple(joints.shape)}.")
    joints = joints[:, :num_joints]
    if joints.shape[1] < num_joints:
        pad = num_joints - joints.shape[1]
        joints = torch.cat([joints, joints[:, -1:].expand(-1, pad, -1)], dim=1)
    return joints


def _expand_fallback_vector(vector: Tensor, fallback: Sequence[float]) -> Tensor:
    fallback_t = torch.as_tensor(fallback, dtype=vector.dtype, device=vector.device)
    while fallback_t.ndim < vector.ndim:
        fallback_t = fallback_t.unsqueeze(0)
    return fallback_t.expand_as(vector)


def _safe_normalize(vector: Tensor, fallback: Sequence[float]) -> Tensor:
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    normalized = vector / norm.clamp(min=1e-6)
    return torch.where(norm > 1e-6, normalized, _expand_fallback_vector(vector, fallback))


def _canonicalize_vertices_with_body_frame(vertices: Tensor, joints_3d: Tensor) -> Tensor:
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"`vertices` must have shape [T, V, 3], got {tuple(vertices.shape)}.")
    if joints_3d.ndim != 3 or joints_3d.shape[-1] != 3:
        raise ValueError(f"`joints_3d` must have shape [T, J, 3], got {tuple(joints_3d.shape)}.")
    if vertices.shape[0] != joints_3d.shape[0]:
        raise ValueError(
            f"`vertices` and `joints_3d` must share the same frame count, got {vertices.shape[0]} and {joints_3d.shape[0]}."
        )

    root = joints_3d[:, 0]
    if joints_3d.shape[1] >= 3:
        hip_center = 0.5 * (joints_3d[:, 1] + joints_3d[:, 2])
        lateral = joints_3d[:, 2] - joints_3d[:, 1]
    else:
        hip_center = root
        lateral = root.new_zeros(root.shape)

    if joints_3d.shape[1] >= 18:
        shoulder_center = 0.5 * (joints_3d[:, 16] + joints_3d[:, 17])
        torso_up = shoulder_center - hip_center
    elif joints_3d.shape[1] >= 4:
        torso_up = joints_3d[:, 3] - root
    else:
        torso_up = root.new_zeros(root.shape)

    axis_x = _safe_normalize(lateral, fallback=(1.0, 0.0, 0.0))
    axis_y_hint = _safe_normalize(torso_up, fallback=(0.0, 1.0, 0.0))
    axis_z = _safe_normalize(torch.cross(axis_x, axis_y_hint, dim=-1), fallback=(0.0, 0.0, 1.0))
    axis_y = _safe_normalize(torch.cross(axis_z, axis_x, dim=-1), fallback=(0.0, 1.0, 0.0))
    axis_z = _safe_normalize(torch.cross(axis_x, axis_y, dim=-1), fallback=(0.0, 0.0, 1.0))
    basis = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    centered = vertices - root.unsqueeze(1)
    return torch.matmul(centered, basis)


def _select_reference_vertices(vertices: Tensor) -> Tensor:
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"`vertices` must have shape [T, V, 3], got {tuple(vertices.shape)}.")
    if vertices.shape[0] == 1:
        return vertices[0]
    mean_vertices = vertices.mean(dim=0, keepdim=True)
    distances = (vertices - mean_vertices).square().mean(dim=(1, 2))
    return vertices[int(distances.argmin().item())]


def _axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
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
    return identity + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def make_extrinsic_from_axis_angle_and_translation(angle: np.ndarray, translation: np.ndarray) -> Tensor:
    angle_t = torch.as_tensor(angle, dtype=torch.float32)
    translation_t = torch.as_tensor(translation, dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = _axis_angle_to_matrix(angle_t)
    extrinsic[:3, 3] = translation_t
    return extrinsic


def make_extrinsic_from_rotation_and_translation(rotation: np.ndarray, translation: np.ndarray) -> Tensor:
    rotation_t = torch.as_tensor(rotation, dtype=torch.float32)
    translation_t = torch.as_tensor(translation, dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = rotation_t
    extrinsic[:3, 3] = translation_t
    return extrinsic


def _project_rotations_to_so3(rotations: Tensor) -> Tensor:
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"`rotations` must have shape [..., 3, 3], got {tuple(rotations.shape)}.")
    flat = rotations.reshape(-1, 3, 3)
    u, _, vh = torch.linalg.svd(flat)
    projected = torch.matmul(u, vh)
    det = torch.det(projected)
    if bool((det < 0.0).any()):
        correction = torch.eye(3, dtype=projected.dtype, device=projected.device).unsqueeze(0).repeat(flat.shape[0], 1, 1)
        correction[det < 0.0, 2, 2] = -1.0
        projected = torch.matmul(torch.matmul(u, correction), vh)
    return projected.reshape_as(rotations)


def _project_object_poses_to_se3(object_poses: Tensor) -> Tensor:
    if object_poses.ndim != 3 or object_poses.shape[-2:] != (4, 4):
        raise ValueError(f"`object_poses` must have shape [T, 4, 4], got {tuple(object_poses.shape)}.")
    projected = object_poses.clone()
    projected[:, :3, :3] = _project_rotations_to_so3(projected[:, :3, :3])
    projected[:, 3, :] = 0.0
    projected[:, 3, 3] = 1.0
    return projected


def _load_sequence_names_from_split_file(split_file: str, split_key: str) -> List[str]:
    with open(split_file, "rb") as handle:
        split_payload = pickle.load(handle)
    if split_key not in split_payload:
        raise KeyError(f"Missing split `{split_key}` in {split_file}. Available keys: {sorted(split_payload.keys())}")
    entries = split_payload[split_key]
    sequence_names = set()
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            values = entry
        else:
            values = (entry,)
        for value in values:
            normalized = str(value).replace("\\", "/").strip("/")
            if not normalized:
                continue
            sequence_names.add(normalized.split("/")[0])
    if not sequence_names:
        raise ValueError(f"No sequence names could be inferred from {split_file}:{split_key}")
    return sorted(sequence_names)


def _discover_timestep_dirs(sequence_dir: Path) -> List[Path]:
    behave_dirs = sorted(Path(path) for path in glob.glob(str(sequence_dir / "t*.000")))
    if behave_dirs:
        return behave_dirs
    return sorted(
        child
        for child in sequence_dir.iterdir()
        if child.is_dir() and any(grandchild.is_dir() for grandchild in child.iterdir())
    )


def load_object_pose_sequence(sequence_dir: Path, num_frames: int, processed_subdir: str = "processed") -> Tensor:
    object_pose_path = sequence_dir / processed_subdir / "object_poses.npz"
    if object_pose_path.is_file():
        with np.load(object_pose_path) as object_pose_npz:
            if "object_poses" not in object_pose_npz:
                raise KeyError(f"Missing `object_poses` in {object_pose_path}")
            object_poses = torch.from_numpy(object_pose_npz["object_poses"].astype(np.float32))
        if object_poses.ndim != 3 or object_poses.shape[-2:] != (4, 4):
            raise ValueError(f"Expected object poses with shape [T, 4, 4], got {tuple(object_poses.shape)}.")
        if object_poses.shape[0] < num_frames:
            raise ValueError(
                f"Object pose sequence in {object_pose_path} is too short: "
                f"{object_poses.shape[0]} < required {num_frames} frames."
            )
        return _project_object_poses_to_se3(object_poses[:num_frames])

    timestep_dirs = _discover_timestep_dirs(sequence_dir)
    poses: List[Tensor] = []
    for timestep_dir in timestep_dirs[:num_frames]:
        fit_paths = sorted(glob.glob(os.path.join(timestep_dir, "*", "fit01", "*_fit.pkl")))
        fit_paths = [path for path in fit_paths if "/person/" not in path]
        if not fit_paths:
            raise FileNotFoundError(
                f"Missing object fit for {timestep_dir}. Expected `<object>/fit01/*_fit.pkl` "
                "or a precomputed processed/object_poses.npz."
            )
        with open(fit_paths[0], "rb") as handle:
            fit = pickle.load(handle, encoding="latin1")
        trans = np.asarray(fit.get("trans", fit.get("translation", np.zeros(3, dtype=np.float32))), dtype=np.float32)
        if "rot" in fit:
            rotation = np.asarray(fit["rot"], dtype=np.float32).reshape(3, 3)
            poses.append(make_extrinsic_from_rotation_and_translation(rotation, trans))
        else:
            angle = np.asarray(fit.get("angle", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            poses.append(make_extrinsic_from_axis_angle_and_translation(angle, trans))
    if not poses:
        raise FileNotFoundError(
            f"No object poses could be resolved for {sequence_dir}. "
            "Expected processed/object_poses.npz or BEHAVE fit01 pose files."
        )
    if len(poses) < num_frames:
        raise ValueError(
            f"Object pose sequence for {sequence_dir} is too short: "
            f"{len(poses)} < required {num_frames} frames."
        )
    return _project_object_poses_to_se3(torch.stack(poses[:num_frames], dim=0))


def _assemble_raw_gaussian_tokens(payload: Dict[str, Tensor]) -> Tensor:
    if "raw" in payload:
        raw = payload["raw"].float()
        if raw.shape[-1] != 14:
            raise ValueError(f"Expected `raw` Gaussian tensor with 14 channels, got {tuple(raw.shape)}.")
        return raw
    required = ("xyz", "rotation", "scaling", "opacity", "shs")
    for key in required:
        if key not in payload:
            raise KeyError(f"Gaussian payload is missing `{key}`.")
    xyz = payload["xyz"].float()
    rotation = torch.nn.functional.normalize(payload["rotation"].float(), dim=-1)
    scaling = payload["scaling"].float().clamp(min=1e-6)
    opacity = payload["opacity"].float().clamp(0.0, 1.0)
    shs = payload["shs"].float().clamp(0.0, 1.0)
    return torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)


def _subsample_tokens(tokens: Tensor, num_tokens: int) -> Tensor:
    if tokens.shape[0] == num_tokens:
        return tokens
    if tokens.shape[0] < num_tokens:
        pad = num_tokens - tokens.shape[0]
        tail = tokens[-1:].expand(pad, -1)
        return torch.cat([tokens, tail], dim=0)
    indices = torch.linspace(0, tokens.shape[0] - 1, steps=num_tokens, device=tokens.device)
    indices = indices.round().long().clamp(min=0, max=tokens.shape[0] - 1)
    return tokens.index_select(0, indices)


def _sort_gaussian_tokens_by_xyz(tokens: Tensor) -> Tensor:
    if tokens.ndim != 2 or tokens.shape[-1] < 3:
        raise ValueError(f"`tokens` must have shape [N, C>=3], got {tuple(tokens.shape)}.")
    if tokens.shape[0] <= 1:
        return tokens
    # Stabilize token ordering across sequences so state-token learning is not
    # dominated by arbitrary surface-sampling permutations.
    xyz = tokens[:, :3].detach().cpu().numpy()
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    order_t = torch.from_numpy(order.astype(np.int64)).to(device=tokens.device)
    return tokens.index_select(0, order_t)


def _maybe_load_gaussians(sequence_dir: Path, gs_subdir: str, filename: str) -> Optional[Tensor]:
    path = sequence_dir / gs_subdir / filename
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return _assemble_raw_gaussian_tokens(payload)
    combined = sequence_dir / gs_subdir / "gs_init_combined.pt"
    if combined.is_file():
        payload = torch.load(combined, map_location="cpu", weights_only=False)
        if filename == "G_h.pt" and "G_h" in payload:
            return _assemble_raw_gaussian_tokens(payload["G_h"])
        if filename == "G_o.pt" and "G_o" in payload:
            return _assemble_raw_gaussian_tokens(payload["G_o"])
    return None


def _normalize_human_gaussian_source(human_gaussian_source: str) -> str:
    source = str(human_gaussian_source).strip().lower()
    if source not in VALID_HUMAN_GAUSSIAN_SOURCES:
        raise ValueError(
            f"`human_gaussian_source` must be one of {VALID_HUMAN_GAUSSIAN_SOURCES}, got {human_gaussian_source!r}."
        )
    return source


def _placeholder_gaussian_tokens(num_tokens: int) -> Tensor:
    tokens = torch.zeros(int(num_tokens), 14, dtype=torch.float32)
    tokens[:, 3] = 1.0
    tokens[:, 7:10] = 0.01
    tokens[:, 11:14] = 0.5
    return tokens


def _normals_to_quaternions(normals: Tensor) -> Tensor:
    normals = normals.float()
    quats = normals.new_zeros(normals.shape[0], 4)
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    safe_normals = torch.where(norm > 1e-6, normals / norm.clamp(min=1e-6), z_axis)
    dot = (z_axis * safe_normals).sum(dim=-1).clamp(-1.0, 1.0)
    same = dot > 0.9999
    opposite = dot < -0.9999
    general = ~(same | opposite)
    quats[same, 0] = 1.0
    quats[opposite, 1] = 1.0
    if bool(general.any()):
        cross = torch.cross(z_axis[general], safe_normals[general], dim=-1)
        quat = torch.cat([1.0 + dot[general].unsqueeze(-1), cross], dim=-1)
        quats[general] = torch.nn.functional.normalize(quat, dim=-1)
    return quats


def _compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"`vertices` must have shape [V, 3], got {tuple(vertices.shape)}.")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"`faces` must have shape [F, 3], got {tuple(faces.shape)}.")
    tri_vertices = vertices[faces.long()]
    face_normals = torch.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
        dim=-1,
    )
    normals = vertices.new_zeros(vertices.shape)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner].long(), face_normals)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    fallback = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    return torch.where(norm > 1e-6, normals / norm.clamp(min=1e-6), fallback)


def build_human_gaussian_raw_tokens_from_smpl_params(
    smpl_params: Dict[str, np.ndarray],
    *,
    init_gaussian_scale: float = 0.01,
) -> Tensor:
    if "vertices" not in smpl_params or "faces" not in smpl_params:
        raise FileNotFoundError(
            "SMPL-anchored human pseudo-GT requires `vertices` and `faces` in processed/smpl_params.npz."
        )

    vertices = np.asarray(smpl_params["vertices"], dtype=np.float32)
    faces = np.asarray(smpl_params["faces"], dtype=np.int64)
    if vertices.ndim == 3:
        vertices_t = torch.from_numpy(vertices.astype(np.float32))
        joints_np = smpl_params.get("joints_3d", smpl_params.get("keypoints_3d"))
        if joints_np is not None:
            joints_t = torch.from_numpy(np.asarray(joints_np, dtype=np.float32))
            num_frames = min(vertices_t.shape[0], joints_t.shape[0])
            vertices_t = _canonicalize_vertices_with_body_frame(vertices_t[:num_frames], joints_t[:num_frames])
        else:
            num_frames = vertices_t.shape[0]
            if "cam_t" in smpl_params:
                cam_t = np.asarray(smpl_params["cam_t"], dtype=np.float32)
                if cam_t.ndim == 2 and cam_t.shape[0] >= num_frames:
                    vertices_t = vertices_t[:num_frames] - torch.from_numpy(cam_t[:num_frames]).float().unsqueeze(1)
                else:
                    vertices_t = vertices_t[:num_frames]
            else:
                vertices_t = vertices_t[:num_frames]
        vertices_t = _select_reference_vertices(vertices_t)
    elif vertices.ndim != 2:
        raise ValueError(f"Expected SMPL vertices with shape [T, V, 3] or [V, 3], got {tuple(vertices.shape)}.")
    else:
        vertices_t = torch.from_numpy(vertices.astype(np.float32))

    faces_t = torch.from_numpy(faces.astype(np.int64))
    normals_t = _compute_vertex_normals(vertices_t, faces_t)
    rotations_t = _normals_to_quaternions(normals_t)
    scaling_t = torch.full((vertices_t.shape[0], 3), float(init_gaussian_scale), dtype=torch.float32)
    opacity_t = torch.full((vertices_t.shape[0], 1), 0.9, dtype=torch.float32)
    shs_t = torch.full((vertices_t.shape[0], 3), 0.5, dtype=torch.float32)
    return torch.cat([vertices_t, rotations_t, scaling_t, opacity_t, shs_t], dim=-1)


def _build_human_gaussians_from_smpl_params(
    smpl_params: Dict[str, np.ndarray],
    *,
    num_human_gaussians: int,
    init_gaussian_scale: float = 0.01,
) -> Tensor:
    raw = build_human_gaussian_raw_tokens_from_smpl_params(
        smpl_params,
        init_gaussian_scale=init_gaussian_scale,
    )
    return _subsample_tokens(raw, num_human_gaussians)


def compute_contact_signature(
    joints_3d: Tensor,
    object_gaussians: Tensor,
    object_poses: Tensor,
    *,
    contact_dim: int,
    hand_joint_indices: Optional[Sequence[int]] = None,
) -> Tensor:
    if joints_3d.ndim != 3:
        raise ValueError(f"`joints_3d` must have shape [T, J, 3], got {tuple(joints_3d.shape)}.")
    object_xyz = object_gaussians[:, :3]
    num_frames = joints_3d.shape[0]
    num_joints = joints_3d.shape[1]
    if hand_joint_indices:
        indices = [min(max(int(idx), 0), num_joints - 1) for idx in hand_joint_indices]
    else:
        num_tail = min(4, num_joints)
        indices = list(range(num_joints - num_tail, num_joints))
    if len(indices) < 2:
        indices = list(range(max(0, num_joints - 2), num_joints))
    left_indices = indices[: max(1, len(indices) // 2)]
    right_indices = indices[max(1, len(indices) // 2) :]
    if not right_indices:
        right_indices = left_indices

    signature = joints_3d.new_zeros(num_frames, contact_dim)
    for frame_idx in range(num_frames):
        transform = object_poses[frame_idx]
        object_h = torch.cat([object_xyz, torch.ones_like(object_xyz[:, :1])], dim=-1)
        object_world = torch.matmul(transform.unsqueeze(0), object_h.unsqueeze(-1)).squeeze(-1)[..., :3]

        left_hand = joints_3d[frame_idx, left_indices]
        right_hand = joints_3d[frame_idx, right_indices]
        left_dist = torch.cdist(left_hand.unsqueeze(0), object_world.unsqueeze(0)).min().detach()
        right_dist = torch.cdist(right_hand.unsqueeze(0), object_world.unsqueeze(0)).min().detach()
        left_contact = torch.exp(-left_dist / 0.08)
        right_contact = torch.exp(-right_dist / 0.08)
        values = [left_dist, right_dist, left_contact, right_contact]
        signature[frame_idx, : min(contact_dim, len(values))] = torch.stack(values[:contact_dim]).to(signature)
    return signature


def _compute_contact_signature(
    joints_3d: Tensor,
    object_gaussians: Tensor,
    object_poses: Tensor,
    *,
    contact_dim: int,
    hand_joint_indices: Optional[Sequence[int]] = None,
) -> Tensor:
    return compute_contact_signature(
        joints_3d,
        object_gaussians,
        object_poses,
        contact_dim=contact_dim,
        hand_joint_indices=hand_joint_indices,
    )


def build_keypoint_heatmaps(
    keypoints_2d: Tensor,
    height: int,
    width: int,
    sigma: float = DEFAULT_KEYPOINT_HEATMAP_SIGMA,
) -> Tensor:
    num_frames = keypoints_2d.shape[0]
    grid_y = torch.arange(height, dtype=keypoints_2d.dtype).view(1, 1, height, 1)
    grid_x = torch.arange(width, dtype=keypoints_2d.dtype).view(1, 1, 1, width)
    coords = keypoints_2d[..., :2].unsqueeze(-2).unsqueeze(-2)
    conf = keypoints_2d[..., 2].reshape(num_frames, keypoints_2d.shape[1], 1, 1)
    sq_dist = (grid_x - coords[..., 0]) ** 2 + (grid_y - coords[..., 1]) ** 2
    weights = torch.exp(-sq_dist / (2.0 * sigma * sigma)) * conf
    return weights.sum(dim=1, keepdim=True).reshape(num_frames, 1, height, width)


def _build_keypoint_heatmaps(
    keypoints_2d: Tensor,
    height: int,
    width: int,
    sigma: float = DEFAULT_KEYPOINT_HEATMAP_SIGMA,
) -> Tensor:
    return build_keypoint_heatmaps(
        keypoints_2d,
        height=height,
        width=width,
        sigma=sigma,
    )


def _match_contact_signature_dim(signature: Tensor, contact_dim: int) -> Tensor:
    contact_dim = int(contact_dim)
    if signature.shape[-1] == contact_dim:
        return signature
    if signature.shape[-1] > contact_dim:
        return signature[:, :contact_dim]
    pad = signature.new_zeros(signature.shape[0], contact_dim - signature.shape[-1])
    return torch.cat([signature, pad], dim=-1)


def _maybe_load_precomputed_dual_branch_targets(
    sequence_dir: Path,
    *,
    processed_subdir: str,
    contact_dim: int,
    hand_joint_indices: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Tensor]]:
    path = sequence_dir / processed_subdir / "cropped" / DUAL_BRANCH_TARGETS_FILENAME
    if not path.is_file():
        return None
    try:
        with np.load(path) as target_npz:
            version = int(target_npz["version"][0]) if "version" in target_npz else DUAL_BRANCH_TARGET_CACHE_VERSION
            if version != DUAL_BRANCH_TARGET_CACHE_VERSION:
                return None
            keypoint_heatmaps = torch.from_numpy(np.asarray(target_npz["keypoint_heatmaps"], dtype=np.float32))
            contact_signature = torch.from_numpy(np.asarray(target_npz["contact_signature"], dtype=np.float32))
            cached_indices = tuple(
                int(index)
                for index in np.asarray(target_npz["hand_joint_indices"], dtype=np.int64).reshape(-1).tolist()
            ) if "hand_joint_indices" in target_npz else ()
    except Exception:
        return None

    requested_indices = tuple(int(index) for index in (hand_joint_indices or ()))
    if cached_indices != requested_indices:
        return None
    if keypoint_heatmaps.ndim != 4 or contact_signature.ndim != 2:
        return None
    return {
        "keypoint_heatmaps": keypoint_heatmaps,
        "contact_signature": _match_contact_signature_dim(contact_signature, contact_dim),
    }


def _validate_mask_tensor(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    min_value = float(tensor.min())
    max_value = float(tensor.max())
    if min_value < -1e-5 or max_value > 1.0 + 1e-5:
        raise ValueError(
            f"{name} must be normalized to [0, 1], got min={min_value:.4f}, max={max_value:.4f}."
        )


def _validate_positive_intrinsics(intrinsics: Tensor) -> None:
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"`intrinsics` must have shape [T, 3, 3], got {tuple(intrinsics.shape)}.")
    if not torch.isfinite(intrinsics).all():
        raise ValueError("Camera intrinsics contain non-finite values.")
    if not torch.all(intrinsics[:, 0, 0] > 0.0):
        raise ValueError("Camera intrinsics contain non-positive fx values.")
    if not torch.all(intrinsics[:, 1, 1] > 0.0):
        raise ValueError("Camera intrinsics contain non-positive fy values.")


def _validate_sequence_bundle(bundle: Dict[str, object], *, num_frames: int, num_joints: int) -> None:
    if num_frames <= 0:
        raise ValueError("Sequence bundle resolved zero valid frames.")
    if len(bundle["rgb_paths"]) != num_frames:
        raise ValueError(
            f"RGB path count mismatch: expected {num_frames}, got {len(bundle['rgb_paths'])}."
        )
    for name in ("masks_human", "masks_object", "m_primary", "m_secondary", "m_object_region"):
        tensor = bundle[name]
        if tensor.shape[0] != num_frames:
            raise ValueError(f"{name} frame count mismatch: expected {num_frames}, got {tensor.shape[0]}.")
        _validate_mask_tensor(name, tensor)
    depth = bundle["depth"]
    if depth.shape[0] != num_frames:
        raise ValueError(f"Depth frame count mismatch: expected {num_frames}, got {depth.shape[0]}.")
    if not torch.isfinite(depth).all():
        raise ValueError("Depth contains non-finite values.")
    if float(depth.min()) < 0.0:
        raise ValueError(f"Depth must be non-negative, got min={float(depth.min()):.4f}.")
    _validate_positive_intrinsics(bundle["intrinsics"])
    keypoints_2d = bundle["keypoints_2d"]
    if keypoints_2d.shape != (num_frames, num_joints, 3):
        raise ValueError(
            f"keypoints_2d must have shape [{num_frames}, {num_joints}, 3], got {tuple(keypoints_2d.shape)}."
        )
    if not torch.isfinite(keypoints_2d).all():
        raise ValueError("keypoints_2d contains non-finite values.")
    joints_3d = bundle["joints_3d"]
    if joints_3d.shape != (num_frames, num_joints, 3):
        raise ValueError(
            f"joints_3d must have shape [{num_frames}, {num_joints}, 3], got {tuple(joints_3d.shape)}."
        )
    if not torch.isfinite(joints_3d).all():
        raise ValueError("joints_3d contains non-finite values.")
    object_poses = bundle["object_poses"]
    if object_poses.shape != (num_frames, 4, 4):
        raise ValueError(
            f"object_poses must have shape [{num_frames}, 4, 4], got {tuple(object_poses.shape)}."
        )
    if not torch.isfinite(object_poses).all():
        raise ValueError("object_poses contains non-finite values.")
    for name in ("human_gaussians", "object_gaussians"):
        tensor = bundle[name]
        if tensor.ndim != 2 or tensor.shape[-1] != 14:
            raise ValueError(f"{name} must have shape [N, 14], got {tuple(tensor.shape)}.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")
    keypoint_heatmaps = bundle.get("keypoint_heatmaps")
    if keypoint_heatmaps is not None:
        if keypoint_heatmaps.shape[0] != num_frames or keypoint_heatmaps.ndim != 4:
            raise ValueError(
                f"keypoint_heatmaps must have shape [{num_frames}, 1, H, W], got {tuple(keypoint_heatmaps.shape)}."
            )
        if not torch.isfinite(keypoint_heatmaps).all():
            raise ValueError("keypoint_heatmaps contains non-finite values.")
    contact_signature = bundle.get("contact_signature")
    if contact_signature is not None:
        if contact_signature.shape[0] != num_frames or contact_signature.ndim != 2:
            raise ValueError(
                f"contact_signature must have shape [{num_frames}, C], got {tuple(contact_signature.shape)}."
            )
        if not torch.isfinite(contact_signature).all():
            raise ValueError("contact_signature contains non-finite values.")


def _discover_sequence_dirs(
    root: str,
    processed_subdir: str,
    gs_subdir: str,
    allowed_sequence_names: Optional[Iterable[str]] = None,
) -> List[str]:
    root_path = Path(root).expanduser().resolve()
    candidates: List[Path] = []
    allowed = {str(name) for name in allowed_sequence_names or []}

    def is_sequence_dir(path: Path) -> bool:
        cropped = path / processed_subdir / "cropped"
        return (
            cropped.is_dir()
            and (cropped / "rgb").is_dir()
            and (cropped / "masks_raw.npz").is_file()
            and (path / processed_subdir / "smpl_params.npz").is_file()
            and (path / gs_subdir).is_dir()
        )

    def keep_sequence(path: Path) -> bool:
        return not allowed or path.name in allowed

    if is_sequence_dir(root_path):
        if not keep_sequence(root_path):
            raise FileNotFoundError(
                f"Sequence dir {root_path} exists but is not included in the active split filter."
            )
        return [str(root_path)]

    for base in (root_path, root_path / "sequences"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and keep_sequence(child) and is_sequence_dir(child):
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No valid dual-branch training sequences found under {root_path}. "
            "Expected `processed/cropped/rgb`, `processed/smpl_params.npz`, and `gs_init/`."
        )
    return [str(path) for path in candidates]


def _path_signature(path: str) -> Dict[str, object]:
    resolved = str(Path(path).expanduser().resolve())
    try:
        stat = Path(resolved).stat()
    except OSError:
        return {"path": resolved, "exists": False}
    return {
        "path": resolved,
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _build_index_cache_path(
    *,
    data_root: str,
    processed_subdir: str,
    gs_subdir: str,
    human_gaussian_source: str,
    max_sequences: int,
    split_file: str,
    split_key: str,
) -> Path:
    root_path = Path(data_root).expanduser().resolve()
    payload = {
        "version": INDEX_CACHE_VERSION,
        "data_root": str(root_path),
        "processed_subdir": processed_subdir,
        "gs_subdir": gs_subdir,
        "human_gaussian_source": _normalize_human_gaussian_source(human_gaussian_source),
        "max_sequences": int(max_sequences),
        "split_file": _path_signature(split_file) if split_file else {"path": "", "exists": False},
        "split_key": split_key,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return root_path / ".dual_branch_index_cache" / f"{digest}.pkl"


def inspect_sequence_num_frames(
    sequence_dir: str,
    *,
    processed_subdir: str = "processed",
    gs_subdir: str = "gs_init",
    human_gaussian_source: str = "smpl_mesh",
) -> int:
    sequence_path = Path(sequence_dir)
    human_gaussian_source = _normalize_human_gaussian_source(human_gaussian_source)
    processed_dir = sequence_path / processed_subdir
    cropped_dir = processed_dir / "cropped"

    rgb_paths = _sorted_image_paths(cropped_dir / "rgb")
    num_frames = len(rgb_paths)

    with np.load(cropped_dir / "masks_raw.npz") as masks_npz:
        num_frames = min(
            num_frames,
            int(masks_npz["human"].shape[0]),
            int(masks_npz["object"].shape[0]),
        )
    with np.load(cropped_dir / "region_masks.npz") as region_npz:
        num_frames = min(
            num_frames,
            int(region_npz["M_p"].shape[0]),
            int(region_npz["M_s"].shape[0]),
            int(region_npz["M_object"].shape[0]),
        )
    with np.load(cropped_dir / "depth_aligned.npz") as depth_npz:
        num_frames = min(num_frames, int(depth_npz["depth"].shape[0]))
    with np.load(cropped_dir / "meta.npz") as meta_npz:
        num_frames = min(
            num_frames,
            int(meta_npz["fx"].shape[0]),
            int(meta_npz["fy"].shape[0]),
            int(meta_npz["cx"].shape[0]),
            int(meta_npz["cy"].shape[0]),
        )

    keypoints_path = cropped_dir / "keypoints_2d.npz"
    if not keypoints_path.is_file():
        raise FileNotFoundError(f"Missing required keypoints file: {keypoints_path}")
    with np.load(keypoints_path) as keypoints_npz:
        num_frames = min(num_frames, int(keypoints_npz["keypoints"].shape[0]))

    object_pose_path = sequence_path / processed_subdir / "object_poses.npz"
    if object_pose_path.is_file():
        with np.load(object_pose_path) as object_pose_npz:
            if "object_poses" not in object_pose_npz:
                raise KeyError(f"Missing `object_poses` in {object_pose_path}")
            num_frames = min(num_frames, int(object_pose_npz["object_poses"].shape[0]))
    else:
        timestep_dirs = _discover_timestep_dirs(sequence_path)
        if len(timestep_dirs) < num_frames:
            raise ValueError(
                f"{sequence_path} provides only {len(timestep_dirs)} timestep dirs for {num_frames} required frames."
            )
        for timestep_dir in timestep_dirs[:num_frames]:
            fit_paths = sorted(glob.glob(os.path.join(timestep_dir, "*", "fit01", "*_fit.pkl")))
            fit_paths = [path for path in fit_paths if "/person/" not in path]
            if not fit_paths:
                raise FileNotFoundError(
                    f"Missing object pose fit for {timestep_dir}. "
                    "Expected processed/object_poses.npz or `<object>/fit01/*_fit.pkl`."
                )

    if _maybe_load_gaussians(sequence_path, gs_subdir, "G_o.pt") is None:
        raise FileNotFoundError(
            f"Missing object Gaussian teacher under {sequence_path / gs_subdir}. "
            "Expected `G_o.pt` or `gs_init_combined.pt` with key `G_o`."
        )

    with np.load(processed_dir / "smpl_params.npz") as smpl_params:
        if human_gaussian_source == "teacher":
            if _maybe_load_gaussians(sequence_path, gs_subdir, "G_h.pt") is None:
                raise FileNotFoundError(
                    f"Missing human Gaussian teacher under {sequence_path / gs_subdir}. "
                    "Expected `G_h.pt` or `gs_init_combined.pt` with key `G_h`."
                )
        else:
            if "vertices" not in smpl_params or "faces" not in smpl_params:
                raise FileNotFoundError(
                    f"Missing SMPL mesh geometry under {processed_dir / 'smpl_params.npz'}. "
                    "Expected `vertices` and `faces` for `human_gaussian_source=smpl_mesh`."
                )
        num_frames = min(num_frames, int(smpl_params["body_pose"].shape[0]))
        if "cam_t" in smpl_params:
            num_frames = min(num_frames, int(smpl_params["cam_t"].shape[0]))
        if "joints_3d" in smpl_params:
            num_frames = min(num_frames, int(smpl_params["joints_3d"].shape[0]))
        elif "keypoints_3d" in smpl_params:
            num_frames = min(num_frames, int(smpl_params["keypoints_3d"].shape[0]))

    return int(num_frames)


def load_dual_branch_sequence_bundle(
    sequence_dir: str,
    *,
    processed_subdir: str = "processed",
    gs_subdir: str = "gs_init",
    human_gaussian_source: str = "smpl_mesh",
    num_human_gaussians: int = 1024,
    num_object_gaussians: int = 1024,
    num_joints: int = 22,
    contact_dim: int = 4,
    hand_joint_indices: Optional[Sequence[int]] = None,
    require_gaussian_targets: bool = True,
    preload_rgb: bool = False,
    validate_bundle: bool = False,
) -> Dict[str, object]:
    sequence_path = Path(sequence_dir)
    human_gaussian_source = _normalize_human_gaussian_source(human_gaussian_source)
    processed_dir = sequence_path / processed_subdir
    cropped_dir = processed_dir / "cropped"
    rgb_paths = _sorted_image_paths(cropped_dir / "rgb")

    with np.load(cropped_dir / "masks_raw.npz") as masks_npz:
        masks_human = torch.from_numpy(masks_npz["human"]).float().unsqueeze(1)
        masks_object = torch.from_numpy(masks_npz["object"]).float().unsqueeze(1)

    with np.load(cropped_dir / "region_masks.npz") as region_npz:
        m_primary = torch.from_numpy(region_npz["M_p"]).float().unsqueeze(1)
        m_secondary = torch.from_numpy(region_npz["M_s"]).float().unsqueeze(1)
        m_object_region = torch.from_numpy(region_npz["M_object"]).float().unsqueeze(1)

    with np.load(cropped_dir / "depth_aligned.npz") as depth_npz:
        depth = torch.from_numpy(depth_npz["depth"]).float().unsqueeze(1)

    with np.load(cropped_dir / "meta.npz") as meta_npz:
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

    keypoints_path = cropped_dir / "keypoints_2d.npz"
    if not keypoints_path.is_file():
        raise FileNotFoundError(f"Missing required keypoints file: {keypoints_path}")
    with np.load(keypoints_path) as keypoints_npz:
        keypoints_2d = _normalize_keypoints_2d_targets(keypoints_npz["keypoints"], num_joints)

    human_gaussians: Optional[Tensor] = None
    human_vertices: Optional[Tensor] = None
    with np.load(processed_dir / "smpl_params.npz") as smpl_params:
        joints_3d = _resolve_joint_targets_from_smpl_params(smpl_params, num_joints)
        body_pose = torch.from_numpy(_normalize_pose_dim(smpl_params["body_pose"].astype(np.float32), target_dim=144))
        if "cam_t" in smpl_params:
            cam_t = torch.from_numpy(smpl_params["cam_t"].astype(np.float32))
        else:
            cam_t = torch.zeros(body_pose.shape[0], 3, dtype=torch.float32)
        if "vertices" in smpl_params:
            human_vertices = torch.from_numpy(smpl_params["vertices"].astype(np.float32))
        if human_gaussian_source == "smpl_mesh":
            human_gaussians = _maybe_load_gaussians(sequence_path, gs_subdir, SMPL_HUMAN_GAUSSIANS_FILENAME)
            if human_gaussians is None:
                if "vertices" in smpl_params and "faces" in smpl_params:
                    human_gaussians = _build_human_gaussians_from_smpl_params(
                        smpl_params,
                        num_human_gaussians=num_human_gaussians,
                    )
                elif require_gaussian_targets:
                    raise FileNotFoundError(
                        f"Missing SMPL mesh geometry under {processed_dir / 'smpl_params.npz'}. "
                        "Expected `vertices` and `faces` for `human_gaussian_source=smpl_mesh`."
                    )

    num_frames = min(
        len(rgb_paths),
        masks_human.shape[0],
        masks_object.shape[0],
        m_primary.shape[0],
        depth.shape[0],
        intrinsics.shape[0],
        keypoints_2d.shape[0],
        joints_3d.shape[0],
        body_pose.shape[0],
        cam_t.shape[0],
        human_vertices.shape[0] if human_vertices is not None else 10**9,
    )
    rgb_paths = rgb_paths[:num_frames]
    masks_human = masks_human[:num_frames]
    masks_object = masks_object[:num_frames]
    m_primary = m_primary[:num_frames]
    m_secondary = m_secondary[:num_frames]
    m_object_region = m_object_region[:num_frames]
    depth = depth[:num_frames]
    intrinsics = intrinsics[:num_frames]
    keypoints_2d = keypoints_2d[:num_frames]
    joints_3d = joints_3d[:num_frames]
    body_pose = body_pose[:num_frames]
    cam_t = cam_t[:num_frames]
    if human_vertices is not None:
        human_vertices = human_vertices[:num_frames]
    object_poses = load_object_pose_sequence(sequence_path, num_frames, processed_subdir=processed_subdir)

    if human_gaussian_source == "teacher":
        human_gaussians = _maybe_load_gaussians(sequence_path, gs_subdir, "G_h.pt")
    object_gaussians = _maybe_load_gaussians(sequence_path, gs_subdir, "G_o.pt")
    if human_gaussians is None and human_gaussian_source == "teacher" and require_gaussian_targets:
        raise FileNotFoundError(
            f"Missing human Gaussian teacher under {sequence_path / gs_subdir}. "
            "Expected `G_h.pt` or `gs_init_combined.pt` with key `G_h`."
        )
    if object_gaussians is None and require_gaussian_targets:
        raise FileNotFoundError(
            f"Missing object Gaussian teacher under {sequence_path / gs_subdir}. "
            "Expected `G_o.pt` or `gs_init_combined.pt` with key `G_o`."
        )
    if human_gaussians is None:
        human_gaussians = _placeholder_gaussian_tokens(num_human_gaussians)
    else:
        human_gaussians = _subsample_tokens(human_gaussians, num_human_gaussians)
        human_gaussians = _sort_gaussian_tokens_by_xyz(human_gaussians)
    if object_gaussians is None:
        object_gaussians = _placeholder_gaussian_tokens(num_object_gaussians)
    else:
        object_gaussians = _subsample_tokens(object_gaussians, num_object_gaussians)
        object_gaussians = _sort_gaussian_tokens_by_xyz(object_gaussians)

    precomputed_targets = _maybe_load_precomputed_dual_branch_targets(
        sequence_path,
        processed_subdir=processed_subdir,
        contact_dim=contact_dim,
        hand_joint_indices=hand_joint_indices,
    )
    if precomputed_targets is not None:
        keypoint_heatmaps = precomputed_targets["keypoint_heatmaps"][:num_frames]
        contact_signature = precomputed_targets["contact_signature"][:num_frames]
        if keypoint_heatmaps.shape[0] < num_frames or contact_signature.shape[0] < num_frames:
            precomputed_targets = None
    if precomputed_targets is None:
        keypoint_heatmaps = _build_keypoint_heatmaps(
            keypoints_2d,
            height=depth.shape[-2],
            width=depth.shape[-1],
        )
        contact_signature = _compute_contact_signature(
            joints_3d,
            object_gaussians,
            object_poses,
            contact_dim=contact_dim,
            hand_joint_indices=hand_joint_indices,
        )

    bundle = {
        "rgb_paths": rgb_paths,
        "masks_human": masks_human,
        "masks_object": masks_object,
        "m_primary": m_primary,
        "m_secondary": m_secondary,
        "m_object_region": m_object_region,
        "depth": depth,
        "intrinsics": intrinsics,
        "keypoints_2d": keypoints_2d,
        "keypoint_heatmaps": keypoint_heatmaps,
        "joints_3d": joints_3d,
        "body_pose": body_pose,
        "cam_t": cam_t,
        "human_vertices": human_vertices,
        "object_poses": object_poses[:num_frames],
        "contact_signature": contact_signature,
        "human_gaussians": human_gaussians,
        "object_gaussians": object_gaussians,
        "num_frames": num_frames,
        "sequence_name": sequence_path.name,
    }
    if preload_rgb:
        bundle["rgb_uint8"] = torch.stack(
            [load_rgb_image_uint8(str(path)) for path in rgb_paths],
            dim=0,
        )
    if validate_bundle:
        _validate_sequence_bundle(bundle, num_frames=num_frames, num_joints=num_joints)
    return bundle


class DualBranchHOIDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str,
        clip_length: int,
        clip_stride: int,
        processed_subdir: str = "processed",
        gs_subdir: str = "gs_init",
        human_gaussian_source: str = "smpl_mesh",
        num_human_gaussians: int = 1024,
        num_object_gaussians: int = 1024,
        num_joints: int = 22,
        contact_dim: int = 4,
        background_value: float = 1.0,
        max_sequences: int = 0,
        hand_joint_indices: Optional[Sequence[int]] = None,
        cache_sequences: int = 2,
        cache_rgb: bool = True,
        rgb_cache_max_frames: int = 256,
        index_progress_every: int = 0,
        index_progress_callback: Optional[Callable[[int, int, str, int], None]] = None,
        split_file: str = "",
        split_key: str = "train",
        validate_sequence_bundles: bool = False,
    ) -> None:
        super().__init__()
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride)
        self.processed_subdir = processed_subdir
        self.gs_subdir = gs_subdir
        self.human_gaussian_source = _normalize_human_gaussian_source(human_gaussian_source)
        self.num_human_gaussians = int(num_human_gaussians)
        self.num_object_gaussians = int(num_object_gaussians)
        self.num_joints = int(num_joints)
        self.contact_dim = int(contact_dim)
        self.background_value = float(background_value)
        self.hand_joint_indices = tuple(hand_joint_indices or [])
        self.cache_sequences = max(int(cache_sequences), 0)
        self.cache_rgb = bool(cache_rgb)
        self.rgb_cache_max_frames = max(int(rgb_cache_max_frames), 0)
        self.index_progress_every = max(int(index_progress_every), 0)
        self.index_progress_callback = index_progress_callback
        self.validate_sequence_bundles = bool(validate_sequence_bundles)
        self.loaded_from_disk_cache = False
        self.index_cache_path = _build_index_cache_path(
            data_root=data_root,
            processed_subdir=processed_subdir,
            gs_subdir=gs_subdir,
            human_gaussian_source=self.human_gaussian_source,
            max_sequences=max_sequences,
            split_file=split_file,
            split_key=split_key,
        )
        split_sequence_names = None
        if split_file:
            split_sequence_names = _load_sequence_names_from_split_file(split_file, split_key)
        self._cache: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        self._cache_lock = threading.RLock()
        self.sequence_frame_counts: Dict[str, int] = {}
        self.sequence_sample_indices: Dict[str, List[int]] = OrderedDict()
        self.samples: List[Tuple[str, int]] = []
        cached_payload = self._load_disk_index_cache()
        if cached_payload is not None:
            self.loaded_from_disk_cache = True
            self.sequence_dirs = [str(path) for path in cached_payload["sequence_dirs"]]
            if max_sequences > 0:
                self.sequence_dirs = self.sequence_dirs[:max_sequences]
            cached_counts = cached_payload["sequence_frame_counts"]
            for sequence_dir in self.sequence_dirs:
                num_frames = int(cached_counts[sequence_dir])
                self.sequence_frame_counts[sequence_dir] = num_frames
                self._append_sequence_samples(sequence_dir, num_frames)
        else:
            self.sequence_dirs = _discover_sequence_dirs(
                data_root,
                processed_subdir,
                gs_subdir,
                allowed_sequence_names=split_sequence_names,
            )
            if max_sequences > 0:
                self.sequence_dirs = self.sequence_dirs[:max_sequences]

            total_sequences = len(self.sequence_dirs)
            for sequence_idx, sequence_dir in enumerate(self.sequence_dirs, start=1):
                num_frames = inspect_sequence_num_frames(
                    sequence_dir,
                    processed_subdir=self.processed_subdir,
                    gs_subdir=self.gs_subdir,
                    human_gaussian_source=self.human_gaussian_source,
                )
                self.sequence_frame_counts[sequence_dir] = num_frames
                self._append_sequence_samples(sequence_dir, num_frames)
                should_report = (
                    self.index_progress_callback is not None
                    and (
                        sequence_idx == 1
                        or sequence_idx == total_sequences
                        or (
                            self.index_progress_every > 0
                            and sequence_idx % self.index_progress_every == 0
                        )
                    )
                )
                if should_report:
                    self.index_progress_callback(
                        sequence_idx,
                        total_sequences,
                        Path(sequence_dir).name,
                        num_frames,
                    )
            self._save_disk_index_cache()

        if not self.samples:
            raise RuntimeError(
                f"No training clips found under {data_root} for clip_length={self.clip_length} and clip_stride={self.clip_stride}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def condition_channels(self) -> int:
        return 10

    def get_sequence_bundle(self, sequence_dir: str) -> Dict[str, object]:
        return self._load_sequence_bundle(sequence_dir)

    def _resolve_joint_targets(self, smpl_params: Dict[str, np.ndarray]) -> Tensor:
        return _resolve_joint_targets_from_smpl_params(smpl_params, self.num_joints)

    def _compute_contact_signature(self, joints_3d: Tensor, object_gaussians: Tensor, object_poses: Tensor) -> Tensor:
        return _compute_contact_signature(
            joints_3d,
            object_gaussians,
            object_poses,
            contact_dim=self.contact_dim,
            hand_joint_indices=self.hand_joint_indices,
        )

    def _append_sequence_samples(self, sequence_dir: str, num_frames: int) -> None:
        if num_frames < self.clip_length:
            return
        sample_indices = self.sequence_sample_indices.setdefault(sequence_dir, [])
        for start in range(0, num_frames - self.clip_length + 1, self.clip_stride):
            sample_indices.append(len(self.samples))
            self.samples.append((sequence_dir, start))

    def _load_disk_index_cache(self) -> Optional[Dict[str, object]]:
        cache_path = self.index_cache_path
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", -1)) != INDEX_CACHE_VERSION:
            return None
        sequence_dirs = payload.get("sequence_dirs")
        sequence_frame_counts = payload.get("sequence_frame_counts")
        if not isinstance(sequence_dirs, list) or not isinstance(sequence_frame_counts, dict):
            return None
        if any(str(path) not in sequence_frame_counts for path in sequence_dirs):
            return None
        return payload

    def _save_disk_index_cache(self) -> None:
        cache_path = self.index_cache_path
        payload = {
            "version": INDEX_CACHE_VERSION,
            "sequence_dirs": list(self.sequence_dirs),
            "sequence_frame_counts": dict(self.sequence_frame_counts),
        }
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except OSError:
            return

    def _load_sequence_bundle(self, sequence_dir: str) -> Dict[str, object]:
        with self._cache_lock:
            if sequence_dir in self._cache:
                self._cache.move_to_end(sequence_dir)
                return self._cache[sequence_dir]
            num_frames = int(self.sequence_frame_counts.get(sequence_dir, 0))
            preload_rgb = (
                self.cache_rgb
                and self.cache_sequences > 0
                and (
                    self.rgb_cache_max_frames == 0
                    or (num_frames > 0 and num_frames <= self.rgb_cache_max_frames)
                )
            )
            bundle = load_dual_branch_sequence_bundle(
                sequence_dir,
                processed_subdir=self.processed_subdir,
                gs_subdir=self.gs_subdir,
                human_gaussian_source=self.human_gaussian_source,
                num_human_gaussians=self.num_human_gaussians,
                num_object_gaussians=self.num_object_gaussians,
                num_joints=self.num_joints,
                contact_dim=self.contact_dim,
                hand_joint_indices=self.hand_joint_indices,
                require_gaussian_targets=True,
                preload_rgb=preload_rgb,
                validate_bundle=self.validate_sequence_bundles,
            )
            if self.cache_sequences > 0:
                self._cache[sequence_dir] = bundle
                self._cache.move_to_end(sequence_dir)
                while len(self._cache) > self.cache_sequences:
                    self._cache.popitem(last=False)
            return bundle

    def get_sample_by_index(
        self,
        index: int,
        *,
        bundle: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Tensor]:
        sequence_dir, start = self.samples[index]
        return self.get_sample(sequence_dir, start, bundle=bundle)

    def get_sample(
        self,
        sequence_dir: str,
        start: int,
        *,
        bundle: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Tensor]:
        if bundle is None:
            bundle = self._load_sequence_bundle(sequence_dir)
        end = start + self.clip_length

        if "rgb_uint8" in bundle:
            rgb = bundle["rgb_uint8"][start:end].float().div(255.0)
        else:
            rgb = torch.stack([load_rgb_image(str(path)) for path in bundle["rgb_paths"][start:end]], dim=0)
        masks_human = bundle["masks_human"][start:end].clone()
        masks_object = bundle["masks_object"][start:end].clone()
        m_primary = bundle["m_primary"][start:end].clone()
        m_secondary = bundle["m_secondary"][start:end].clone()
        m_object_region = bundle["m_object_region"][start:end].clone()
        depth = bundle["depth"][start:end].clone()
        keypoints_2d = bundle["keypoints_2d"][start:end].clone()
        joints_3d = bundle["joints_3d"][start:end].clone()
        human_vertices = bundle["human_vertices"][start:end].clone() if bundle.get("human_vertices") is not None else None
        object_poses = bundle["object_poses"][start:end].clone()
        camera_intrinsics = bundle["intrinsics"][start:end].clone()
        human_gaussians = bundle["human_gaussians"].clone()
        object_gaussians = bundle["object_gaussians"].clone()
        keypoint_heatmaps = bundle["keypoint_heatmaps"][start:end].clone()
        contact_signature = bundle["contact_signature"][start:end].clone()

        background = torch.full_like(rgb, self.background_value)
        human_visible = rgb * masks_human + background * (1.0 - masks_human)

        return {
            "rgb": rgb,
            "human_visible": human_visible,
            "masks_human": masks_human,
            "masks_object": masks_object,
            "m_primary": m_primary,
            "m_secondary": m_secondary,
            "m_object_region": m_object_region,
            "depth": depth,
            "keypoints_2d": keypoints_2d,
            "keypoint_heatmaps": keypoint_heatmaps,
            "joints_3d": joints_3d,
            "human_vertices": human_vertices,
            "camera_intrinsics": camera_intrinsics,
            "object_poses": object_poses,
            "contact_signature": contact_signature,
            "human_gaussians": human_gaussians,
            "object_gaussians": object_gaussians,
            "sequence_name": bundle["sequence_name"],
        }

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.get_sample_by_index(index)


__all__ = [
    "DEFAULT_CONTACT_SIGNATURE_DIM",
    "DEFAULT_KEYPOINT_HEATMAP_SIGMA",
    "DUAL_BRANCH_TARGET_CACHE_VERSION",
    "DUAL_BRANCH_TARGETS_FILENAME",
    "DualBranchHOIDataset",
    "SMPL_HUMAN_GAUSSIANS_FILENAME",
    "build_human_gaussian_raw_tokens_from_smpl_params",
    "build_keypoint_heatmaps",
    "compute_contact_signature",
    "inspect_sequence_num_frames",
    "load_dual_branch_sequence_bundle",
    "load_rgb_image",
    "load_object_pose_sequence",
]
