#!/usr/bin/env python3
"""
Build new-pipeline training assets for ProciGen directly from GT annotations.

Outputs per sequence:
  - processed/depth_aligned.npz
  - processed/region_masks.npz
  - processed/masks_raw.npz
  - processed/keypoints_2d.npz
  - processed/smpl_params.npz
  - processed/object_poses.npz
  - processed/cropped/*
  - gs_init/G_o.pt

Runtime features:
  - multi-process sequence parallelism
  - resumable execution by skipping already completed sequences
  - JSON status/progress logs under `status_dir`
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import socket
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.video_transforms import (
    compute_bbox_from_masks,
    compute_roi_intrinsics,
    crop_and_resize,
    spatial_downsample,
    transform_keypoints_to_crop,
)
from pipeline.multi_region_mask import compute_multi_region_masks

TIMESTEP_DIR_PATTERN = re.compile(r"^\d+-\d+$")


@dataclass(frozen=True)
class PreprocessConfig:
    camera_id: str
    max_frames: int
    processed_subdir: str
    gs_subdir: str
    scale_ratio: int
    crop_size: Tuple[int, int]
    bbox_expand: float
    num_object_gaussians: int
    init_gaussian_scale: float
    smpl_model_root: str
    overwrite: bool


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600:
        return f"{seconds / 3600.0:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60.0:.2f}m"
    return f"{seconds:.1f}s"


def normalize_camera_id(cam_id: str) -> str:
    cam = str(cam_id).strip()
    if cam == "auto":
        return cam
    if cam.startswith("k"):
        return cam
    if cam.isdigit():
        return f"k{cam}"
    raise ValueError(f"Unsupported camera id: {cam_id}")


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line))
            handle.write("\n")


def open_temp_memmap(
    tmp_dir: Path,
    name: str,
    *,
    dtype: np.dtype,
    shape: Tuple[int, ...],
) -> np.memmap:
    return np.lib.format.open_memmap(
        str(tmp_dir / f"{name}.npy"),
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def detect_camera(seq_dir: Path) -> str:
    cameras = []
    for timestep_dir in iter_timestep_dirs(seq_dir):
        for image_path in sorted(timestep_dir.glob("k*.color.jpg")):
            cameras.append(image_path.stem.split(".")[0])
    if not cameras:
        raise FileNotFoundError(f"No color frames found under {seq_dir}")
    if "k1" in cameras:
        return "k1"
    return sorted(set(cameras))[0]


def load_split_sequence_names(split_file: str, split_key: str) -> List[str]:
    with open(split_file, "rb") as handle:
        payload = pickle.load(handle)
    if split_key not in payload:
        raise KeyError(f"Missing split `{split_key}` in {split_file}. Available keys: {sorted(payload.keys())}")
    sequence_names = set()
    for entry in payload[split_key]:
        values = entry if isinstance(entry, (list, tuple)) else (entry,)
        for value in values:
            normalized = str(value).replace("\\", "/").strip("/")
            if normalized:
                sequence_names.add(normalized.split("/")[0])
    return sorted(sequence_names)


def iter_timestep_dirs(seq_dir: Path) -> List[Path]:
    def is_timestep_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        if not TIMESTEP_DIR_PATTERN.match(path.name):
            return False
        if not any(path.glob("k*.color.jpg")):
            return False
        if not (path / "person" / "fit01").is_dir():
            return False
        return True

    return sorted(
        child
        for child in seq_dir.iterdir()
        if is_timestep_dir(child)
    )


def discover_sequence_dirs(
    raw_root: str,
    sequence_names: Optional[Iterable[str]] = None,
) -> List[Path]:
    root_path = Path(raw_root).expanduser().resolve()
    requested = {str(name) for name in sequence_names or []}
    if any((root_path / candidate).is_dir() for candidate in requested):
        paths = [root_path / name for name in sorted(requested) if (root_path / name).is_dir()]
        if paths:
            return paths

    candidates = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if requested and child.name not in requested:
            continue
        if iter_timestep_dirs(child):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No ProciGen sequences found under {root_path}")
    return candidates


def estimate_sequence_length(sequence_dir: Path) -> int:
    try:
        return len(iter_timestep_dirs(sequence_dir))
    except OSError:
        return 10**9


def sort_sequence_dirs_by_estimated_size(sequence_dirs: Sequence[Path]) -> List[Path]:
    return sorted(
        sequence_dirs,
        key=lambda sequence_dir: (estimate_sequence_length(sequence_dir), sequence_dir.name),
    )


def load_intrinsic_matrix(info_path: Path) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    intrinsic = info.get("intrinsic_params", {})
    if isinstance(intrinsic, dict) and "K" in intrinsic:
        K = np.asarray(intrinsic["K"], dtype=np.float32)
    else:
        raise KeyError(f"Could not read `intrinsic_params.K` from {info_path}")
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {info_path}, got {K.shape}")
    extrinsics = info.get("extrinsic_params", [])
    if not isinstance(extrinsics, list) or not extrinsics:
        raise KeyError(f"Could not read `extrinsic_params` from {info_path}")
    return K, extrinsics


def inverse_local_to_world(rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    trans = np.asarray(translation, dtype=np.float32).reshape(3)
    return rot, trans


def transform_world_to_camera(
    points_world: np.ndarray,
    cam_rotation: np.ndarray,
    cam_translation: np.ndarray,
) -> np.ndarray:
    return (points_world - cam_translation[None, :]) @ cam_rotation


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    u, _, vh = np.linalg.svd(rotation, full_matrices=False)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected.astype(np.float32)


def convert_world_pose_to_camera(
    object_rotation_world: np.ndarray,
    object_translation_world: np.ndarray,
    cam_rotation: np.ndarray,
    cam_translation: np.ndarray,
) -> np.ndarray:
    object_rotation_world = project_rotation_to_so3(object_rotation_world)
    object_translation_world = np.asarray(object_translation_world, dtype=np.float32).reshape(3)
    cam_rotation = np.asarray(cam_rotation, dtype=np.float32).reshape(3, 3)
    cam_translation = np.asarray(cam_translation, dtype=np.float32).reshape(3)

    rotation_camera = cam_rotation.T @ object_rotation_world
    translation_camera = (object_translation_world - cam_translation) @ cam_rotation
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation_camera
    extrinsic[:3, 3] = translation_camera
    return extrinsic


def project_points(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.clip(points_camera[:, 2:3], 1e-6, None)
    x = points_camera[:, 0:1] * float(K[0, 0]) / z + float(K[0, 2])
    y = points_camera[:, 1:2] * float(K[1, 1]) / z + float(K[1, 2])
    conf = np.ones((points_camera.shape[0], 1), dtype=np.float32)
    return np.concatenate([x, y, conf], axis=1).astype(np.float32)


def load_joint_regressor(model_root: str, gender: str) -> np.ndarray:
    gender_norm = "female" if str(gender).lower().startswith("f") else "male"
    model_path = Path(model_root) / "smplh" / f"SMPLH_{gender_norm}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing SMPL-H model file: {model_path}")
    with model_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    regressor = np.asarray(payload["J_regressor"], dtype=np.float32)
    if regressor.ndim != 2 or regressor.shape[1] != 6890:
        raise ValueError(f"Unexpected J_regressor shape in {model_path}: {regressor.shape}")
    return regressor


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    return ((mask_uint8 > 127).astype(np.uint8) * 255).astype(np.uint8)


def find_object_name(timestep_dir: Path) -> str:
    candidates = [child.name for child in timestep_dir.iterdir() if child.is_dir() and child.name != "person"]
    if not candidates:
        raise FileNotFoundError(f"No object directory found under {timestep_dir}")
    return sorted(candidates)[0]


def load_mesh_vertices_faces(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangular mesh from {mesh_path}, got {type(mesh).__name__}")
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    quats = np.zeros((normals.shape[0], 4), dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for idx, normal in enumerate(normals):
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            quats[idx] = [1.0, 0.0, 0.0, 0.0]
            continue
        normal = normal / norm
        dot = float(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        if dot > 0.9999:
            quats[idx] = [1.0, 0.0, 0.0, 0.0]
        elif dot < -0.9999:
            quats[idx] = [0.0, 1.0, 0.0, 0.0]
        else:
            cross = np.cross(z_axis, normal)
            quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float32)
            quats[idx] = quat / np.linalg.norm(quat)
    return quats


def mesh_to_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_sample_points: int,
    init_gaussian_scale: float,
) -> Dict[str, torch.Tensor]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_sample_points)
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
        colors = mesh.visual.face_colors[face_indices, :3].astype(np.float32) / 255.0
    elif hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) > 0:
        colors = mesh.visual.vertex_colors[mesh.faces[face_indices, 0], :3].astype(np.float32) / 255.0
    else:
        colors = np.full((num_sample_points, 3), 0.5, dtype=np.float32)
    normals = mesh.face_normals[face_indices].astype(np.float32)
    rotations = normals_to_quaternions(normals)

    xyz = torch.from_numpy(points.astype(np.float32))
    rotation = torch.from_numpy(rotations)
    scaling = torch.full((num_sample_points, 3), float(init_gaussian_scale), dtype=torch.float32)
    opacity = torch.full((num_sample_points, 1), 0.9, dtype=torch.float32)
    shs = torch.from_numpy(colors.astype(np.float32))
    raw = torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)
    return {
        "xyz": xyz,
        "rotation": rotation,
        "scaling": scaling,
        "opacity": opacity,
        "shs": shs,
        "raw": raw,
    }


def preprocess_frame_with_intrinsics(
    *,
    frame: np.ndarray,
    mask_human: np.ndarray,
    mask_object: np.ndarray,
    depth: np.ndarray,
    keypoints_2d: np.ndarray,
    extra_maps: Dict[str, np.ndarray],
    K: np.ndarray,
    scale_ratio: int,
    bbox_expand: float,
    out_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    frame_ds = spatial_downsample(frame, scale_ratio=scale_ratio, is_mask=False)
    mask_h_ds = spatial_downsample(mask_human, scale_ratio=scale_ratio, is_mask=True)
    mask_o_ds = spatial_downsample(mask_object, scale_ratio=scale_ratio, is_mask=True)
    bbox_xywh = compute_bbox_from_masks(mask_h_ds, mask_o_ds, bbox_expand=bbox_expand)

    K_ds = np.array(K, dtype=np.float32, copy=True)
    K_ds[0, :] /= float(scale_ratio)
    K_ds[1, :] /= float(scale_ratio)
    fx_roi, fy_roi, cx_roi, cy_roi = compute_roi_intrinsics(
        float(K_ds[0, 0]),
        float(K_ds[1, 1]),
        float(K_ds[0, 2]),
        float(K_ds[1, 2]),
        bbox_xywh,
        out_size,
    )

    processed = {
        "rgb": crop_and_resize(frame_ds, bbox_xywh, out_size, is_mask=False),
        "mask_human": crop_and_resize(mask_h_ds, bbox_xywh, out_size, is_mask=True).astype(np.float32) / 255.0,
        "mask_object": crop_and_resize(mask_o_ds, bbox_xywh, out_size, is_mask=True).astype(np.float32) / 255.0,
        "depth": crop_and_resize(
            spatial_downsample(depth.astype(np.float32), scale_ratio=scale_ratio, is_mask=False),
            bbox_xywh,
            out_size,
            is_mask=False,
        ).astype(np.float32),
        "keypoints_2d": transform_keypoints_to_crop(keypoints_2d, bbox_xywh, out_size, scale_ratio=scale_ratio),
        "bbox_xywh": bbox_xywh.astype(np.float32),
        "fx": np.float32(fx_roi),
        "fy": np.float32(fy_roi),
        "cx": np.float32(cx_roi),
        "cy": np.float32(cy_roi),
        "orig_size_hw": np.asarray(frame.shape[:2], dtype=np.int32),
        "downsampled_size_hw": np.asarray(frame_ds.shape[:2], dtype=np.int32),
    }
    processed["extra_maps"] = {
        key: crop_and_resize(
            spatial_downsample(value.astype(np.float32), scale_ratio=scale_ratio, is_mask=False),
            bbox_xywh,
            out_size,
            is_mask=False,
        ).astype(np.float32)
        for key, value in extra_maps.items()
    }
    return processed


def stack_smpl_params(smpl_params_list: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = smpl_params_list[0].keys()
    stacked = {}
    for key in keys:
        values = [params[key] for params in smpl_params_list]
        if key == "faces":
            first = np.asarray(values[0], dtype=np.int32)
            for value in values[1:]:
                if not np.array_equal(first, np.asarray(value, dtype=np.int32)):
                    raise ValueError("SMPL faces changed across frames; expected a constant topology.")
            stacked[key] = first
        else:
            stacked[key] = np.stack(values, axis=0)
    return stacked


def sequence_is_prepared(seq_out_dir: Path, processed_subdir: str, gs_subdir: str) -> bool:
    cropped_dir = seq_out_dir / processed_subdir / "cropped"
    required = [
        cropped_dir / "rgb",
        cropped_dir / "masks_raw.npz",
        cropped_dir / "region_masks.npz",
        cropped_dir / "depth_aligned.npz",
        cropped_dir / "meta.npz",
        seq_out_dir / processed_subdir / "smpl_params.npz",
        seq_out_dir / processed_subdir / "object_poses.npz",
        seq_out_dir / gs_subdir / "G_o.pt",
    ]
    return all(path.exists() for path in required)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prepare_sequence(
    raw_seq_dir: Path,
    out_seq_dir: Path,
    *,
    camera_id: str,
    max_frames: int,
    processed_subdir: str,
    gs_subdir: str,
    scale_ratio: int,
    crop_size: Tuple[int, int],
    bbox_expand: float,
    num_object_gaussians: int,
    init_gaussian_scale: float,
    smpl_model_root: str,
    overwrite: bool,
) -> Dict[str, object]:
    if sequence_is_prepared(out_seq_dir, processed_subdir, gs_subdir) and not overwrite:
        return {"sequence": raw_seq_dir.name, "status": "skipped"}

    info_path = raw_seq_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json under {raw_seq_dir}")
    K, extrinsics = load_intrinsic_matrix(info_path)
    camera_id = detect_camera(raw_seq_dir) if camera_id == "auto" else camera_id
    camera_index = int(camera_id[1:])
    cam_rotation, cam_translation = inverse_local_to_world(
        extrinsics[camera_index]["rotation"],
        extrinsics[camera_index]["translation"],
    )

    timestep_dirs = iter_timestep_dirs(raw_seq_dir)
    if max_frames > 0:
        timestep_dirs = timestep_dirs[:max_frames]
    if not timestep_dirs:
        raise RuntimeError(f"No timesteps discovered under {raw_seq_dir}")

    processed_dir = out_seq_dir / processed_subdir
    cropped_dir = processed_dir / "cropped"
    gs_dir = out_seq_dir / gs_subdir
    rgb_dir = cropped_dir / "rgb"
    out_seq_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gs_dir.mkdir(parents=True, exist_ok=True)

    object_name = find_object_name(timestep_dirs[0])
    joint_regressor_cache: Dict[str, np.ndarray] = {}
    canonical_faces: Optional[np.ndarray] = None
    object_gaussians_payload: Optional[Dict[str, torch.Tensor]] = None
    reference_frame_idx = len(timestep_dirs) // 2
    num_frames = len(timestep_dirs)

    with tempfile.TemporaryDirectory(prefix=".procigen_tmp_", dir=str(out_seq_dir)) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        depth_array: Optional[np.memmap] = None
        human_mask_array: Optional[np.memmap] = None
        object_mask_array: Optional[np.memmap] = None
        keypoints_array: Optional[np.memmap] = None
        object_pose_array: Optional[np.memmap] = None
        body_pose_array: Optional[np.memmap] = None
        shape_array: Optional[np.memmap] = None
        cam_t_array: Optional[np.memmap] = None
        vertices_array: Optional[np.memmap] = None
        joints_3d_array: Optional[np.memmap] = None
        focal_length_array: Optional[np.memmap] = None
        region_mp_array: Optional[np.memmap] = None
        region_ms_array: Optional[np.memmap] = None
        region_object_array: Optional[np.memmap] = None
        cropped_depth_array: Optional[np.memmap] = None
        cropped_h_array: Optional[np.memmap] = None
        cropped_o_array: Optional[np.memmap] = None
        cropped_mp_array: Optional[np.memmap] = None
        cropped_ms_array: Optional[np.memmap] = None
        cropped_mobj_array: Optional[np.memmap] = None
        cropped_kp_array: Optional[np.memmap] = None
        bbox_xywh_array: Optional[np.memmap] = None
        fx_array: Optional[np.memmap] = None
        fy_array: Optional[np.memmap] = None
        cx_array: Optional[np.memmap] = None
        cy_array: Optional[np.memmap] = None
        orig_hw_array: Optional[np.memmap] = None
        down_hw_array: Optional[np.memmap] = None
        image_hw: Optional[Tuple[int, int]] = None

        for frame_idx, timestep_dir in enumerate(timestep_dirs):
            color_path = timestep_dir / f"{camera_id}.color.jpg"
            depth_path = timestep_dir / f"{camera_id}.depth.png"
            human_mask_path = timestep_dir / f"{camera_id}.person_mask.png"
            object_mask_path = timestep_dir / f"{camera_id}.obj_rend_mask.png"
            if not color_path.is_file():
                raise FileNotFoundError(f"Missing frame {color_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth {depth_path}")
            if not human_mask_path.is_file():
                raise FileNotFoundError(f"Missing human mask {human_mask_path}")
            if not object_mask_path.is_file():
                raise FileNotFoundError(f"Missing object mask {object_mask_path}")

            frame_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            human_mask = ensure_binary_mask(cv2.imread(str(human_mask_path), cv2.IMREAD_GRAYSCALE))
            object_mask = ensure_binary_mask(cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE))
            depth_m = depth_mm.astype(np.float32) / 1000.0

            with (timestep_dir / "person" / "fit01" / "person_fit.pkl").open("rb") as handle:
                person_fit = pickle.load(handle, encoding="latin1")
            with (timestep_dir / object_name / "fit01" / f"{object_name}_fit.pkl").open("rb") as handle:
                object_fit = pickle.load(handle, encoding="latin1")

            gender = str(person_fit.get("gender", "male")).lower()
            if gender not in joint_regressor_cache:
                joint_regressor_cache[gender] = load_joint_regressor(smpl_model_root, gender)
            joint_regressor = joint_regressor_cache[gender]

            person_vertices, person_faces = load_mesh_vertices_faces(timestep_dir / "person" / "fit01" / "person_fit.ply")
            if canonical_faces is None:
                canonical_faces = person_faces.astype(np.int32)
            elif not np.array_equal(canonical_faces, person_faces.astype(np.int32)):
                raise ValueError(f"Person mesh topology changed in {raw_seq_dir.name}")

            joints_world = (joint_regressor @ person_vertices).astype(np.float32)
            joints_camera = transform_world_to_camera(joints_world, cam_rotation, cam_translation)
            joints_body = joints_camera[:22].astype(np.float32)
            keypoints_frame = project_points(joints_body, K).astype(np.float32)

            object_rotation_world = project_rotation_to_so3(object_fit["rot"])
            object_translation_world = np.asarray(object_fit["trans"], dtype=np.float32).reshape(3)
            object_pose_camera = convert_world_pose_to_camera(
                object_rotation_world,
                object_translation_world,
                cam_rotation,
                cam_translation,
            ).astype(np.float32)

            if frame_idx == reference_frame_idx:
                object_vertices_world, object_faces = load_mesh_vertices_faces(
                    timestep_dir / object_name / "fit01" / f"{object_name}_fit.ply"
                )
                object_vertices_canonical = (object_vertices_world - object_translation_world[None, :]) @ object_rotation_world
                object_gaussians_payload = mesh_to_gaussians(
                    object_vertices_canonical,
                    object_faces,
                    num_sample_points=num_object_gaussians,
                    init_gaussian_scale=init_gaussian_scale,
                )

            if image_hw is None:
                image_hw = tuple(int(dim) for dim in depth_m.shape[:2])
                img_h, img_w = image_hw
                depth_array = open_temp_memmap(tmp_dir, "depth", dtype=np.float32, shape=(num_frames, img_h, img_w))
                human_mask_array = open_temp_memmap(tmp_dir, "human_mask", dtype=np.uint8, shape=(num_frames, img_h, img_w))
                object_mask_array = open_temp_memmap(tmp_dir, "object_mask", dtype=np.uint8, shape=(num_frames, img_h, img_w))
                keypoints_array = open_temp_memmap(
                    tmp_dir,
                    "keypoints_2d",
                    dtype=np.float32,
                    shape=(num_frames,) + tuple(keypoints_frame.shape),
                )
                object_pose_array = open_temp_memmap(
                    tmp_dir,
                    "object_poses",
                    dtype=np.float32,
                    shape=(num_frames,) + tuple(object_pose_camera.shape),
                )
                body_pose_array = open_temp_memmap(tmp_dir, "body_pose", dtype=np.float32, shape=(num_frames, 72))
                shape_array = open_temp_memmap(tmp_dir, "shape", dtype=np.float32, shape=(num_frames, 10))
                cam_t_array = open_temp_memmap(tmp_dir, "cam_t", dtype=np.float32, shape=(num_frames, 3))
                vertices_array = open_temp_memmap(
                    tmp_dir,
                    "vertices",
                    dtype=np.float32,
                    shape=(num_frames,) + tuple(person_vertices.shape),
                )
                joints_3d_array = open_temp_memmap(
                    tmp_dir,
                    "joints_3d",
                    dtype=np.float32,
                    shape=(num_frames,) + tuple(joints_body.shape),
                )
                focal_length_array = open_temp_memmap(tmp_dir, "focal_length", dtype=np.float32, shape=(num_frames, 1))

            assert depth_array is not None
            assert human_mask_array is not None
            assert object_mask_array is not None
            assert keypoints_array is not None
            assert object_pose_array is not None
            assert body_pose_array is not None
            assert shape_array is not None
            assert cam_t_array is not None
            assert vertices_array is not None
            assert joints_3d_array is not None
            assert focal_length_array is not None

            depth_array[frame_idx] = depth_m
            human_mask_array[frame_idx] = human_mask
            object_mask_array[frame_idx] = object_mask
            keypoints_array[frame_idx] = keypoints_frame
            object_pose_array[frame_idx] = object_pose_camera
            body_pose = np.zeros((72,), dtype=np.float32)
            body_pose_src = np.asarray(
                person_fit.get("pose", np.zeros(72, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            body_pose[: min(body_pose.size, body_pose_src.size)] = body_pose_src[: body_pose.size]
            body_pose_array[frame_idx] = body_pose

            body_shape = np.zeros((10,), dtype=np.float32)
            body_shape_src = np.asarray(
                person_fit.get("betas", np.zeros(10, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            body_shape[: min(body_shape.size, body_shape_src.size)] = body_shape_src[: body_shape.size]
            shape_array[frame_idx] = body_shape

            cam_t = np.zeros((3,), dtype=np.float32)
            cam_t_src = np.asarray(
                person_fit.get("trans", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            cam_t[: min(cam_t.size, cam_t_src.size)] = cam_t_src[: cam_t.size]
            cam_t_array[frame_idx] = cam_t
            vertices_array[frame_idx] = person_vertices.astype(np.float32)
            joints_3d_array[frame_idx] = joints_body
            focal_length_array[frame_idx] = np.asarray([K[0, 0]], dtype=np.float32)

        if canonical_faces is None or image_hw is None:
            raise RuntimeError(f"Failed to prepare buffers for {raw_seq_dir.name}")
        if object_gaussians_payload is None:
            raise RuntimeError(f"Failed to build object gaussians for {raw_seq_dir.name}")

        img_h, img_w = image_hw
        assert human_mask_array is not None
        assert object_mask_array is not None
        assert keypoints_array is not None
        region_mp_array = open_temp_memmap(tmp_dir, "region_mp", dtype=np.float32, shape=(num_frames, img_h, img_w))
        region_ms_array = open_temp_memmap(tmp_dir, "region_ms", dtype=np.float32, shape=(num_frames, img_h, img_w))
        region_object_array = open_temp_memmap(
            tmp_dir,
            "region_object",
            dtype=np.float32,
            shape=(num_frames, img_h, img_w),
        )
        for frame_idx in range(num_frames):
            result = compute_multi_region_masks(
                human_mask_array[frame_idx],
                object_mask_array[frame_idx],
                {"keypoints_2d": keypoints_array[frame_idx]},
                img_h,
                img_w,
            )
            region_mp_array[frame_idx] = result["M_p"]
            region_ms_array[frame_idx] = result["M_s"]
            region_object_array[frame_idx] = result["M_object"]

        assert depth_array is not None
        assert object_pose_array is not None
        assert body_pose_array is not None
        assert shape_array is not None
        assert cam_t_array is not None
        assert vertices_array is not None
        assert joints_3d_array is not None
        assert focal_length_array is not None
        write_npz(processed_dir / "depth_aligned.npz", depth=depth_array)
        write_npz(
            processed_dir / "region_masks.npz",
            M_p=region_mp_array,
            M_s=region_ms_array,
            M_object=region_object_array,
        )
        write_npz(processed_dir / "masks_raw.npz", human=human_mask_array, object=object_mask_array)
        write_npz(processed_dir / "keypoints_2d.npz", keypoints=keypoints_array)
        write_npz(
            processed_dir / "smpl_params.npz",
            body_pose=body_pose_array,
            shape=shape_array,
            cam_t=cam_t_array,
            vertices=vertices_array,
            faces=canonical_faces.astype(np.int32),
            joints_3d=joints_3d_array,
            keypoints_2d=keypoints_array,
            focal_length=focal_length_array,
        )
        write_npz(processed_dir / "object_poses.npz", object_poses=object_pose_array)

        crop_h, crop_w = crop_size
        cropped_depth_array = open_temp_memmap(
            tmp_dir,
            "cropped_depth",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_h_array = open_temp_memmap(
            tmp_dir,
            "cropped_human_mask",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_o_array = open_temp_memmap(
            tmp_dir,
            "cropped_object_mask",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_mp_array = open_temp_memmap(
            tmp_dir,
            "cropped_mp",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_ms_array = open_temp_memmap(
            tmp_dir,
            "cropped_ms",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_mobj_array = open_temp_memmap(
            tmp_dir,
            "cropped_mobj",
            dtype=np.float32,
            shape=(num_frames, crop_h, crop_w),
        )
        cropped_kp_array = open_temp_memmap(
            tmp_dir,
            "cropped_keypoints",
            dtype=np.float32,
            shape=(num_frames,) + tuple(keypoints_array.shape[1:]),
        )
        bbox_xywh_array = open_temp_memmap(tmp_dir, "bbox_xywh", dtype=np.float32, shape=(num_frames, 4))
        fx_array = open_temp_memmap(tmp_dir, "fx", dtype=np.float32, shape=(num_frames,))
        fy_array = open_temp_memmap(tmp_dir, "fy", dtype=np.float32, shape=(num_frames,))
        cx_array = open_temp_memmap(tmp_dir, "cx", dtype=np.float32, shape=(num_frames,))
        cy_array = open_temp_memmap(tmp_dir, "cy", dtype=np.float32, shape=(num_frames,))
        orig_hw_array = open_temp_memmap(tmp_dir, "orig_hw", dtype=np.int32, shape=(num_frames, 2))
        down_hw_array = open_temp_memmap(tmp_dir, "down_hw", dtype=np.int32, shape=(num_frames, 2))

        for frame_idx, timestep_dir in enumerate(timestep_dirs):
            color_path = timestep_dir / f"{camera_id}.color.jpg"
            if not color_path.is_file():
                raise FileNotFoundError(f"Missing frame {color_path}")
            frame_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            cropped = preprocess_frame_with_intrinsics(
                frame=frame_bgr,
                mask_human=human_mask_array[frame_idx],
                mask_object=object_mask_array[frame_idx],
                depth=depth_array[frame_idx],
                keypoints_2d=keypoints_array[frame_idx],
                extra_maps={
                    "M_p": region_mp_array[frame_idx],
                    "M_s": region_ms_array[frame_idx],
                    "M_object": region_object_array[frame_idx],
                },
                K=K,
                scale_ratio=scale_ratio,
                bbox_expand=bbox_expand,
                out_size=(crop_h, crop_w),
            )

            cv2.imwrite(str(rgb_dir / f"frame_{frame_idx:06d}.png"), cropped["rgb"])
            cropped_depth_array[frame_idx] = cropped["depth"]
            cropped_h_array[frame_idx] = cropped["mask_human"]
            cropped_o_array[frame_idx] = cropped["mask_object"]
            cropped_mp_array[frame_idx] = cropped["extra_maps"]["M_p"]
            cropped_ms_array[frame_idx] = cropped["extra_maps"]["M_s"]
            cropped_mobj_array[frame_idx] = cropped["extra_maps"]["M_object"]
            cropped_kp_array[frame_idx] = cropped["keypoints_2d"]
            bbox_xywh_array[frame_idx] = cropped["bbox_xywh"]
            fx_array[frame_idx] = cropped["fx"]
            fy_array[frame_idx] = cropped["fy"]
            cx_array[frame_idx] = cropped["cx"]
            cy_array[frame_idx] = cropped["cy"]
            orig_hw_array[frame_idx] = cropped["orig_size_hw"]
            down_hw_array[frame_idx] = cropped["downsampled_size_hw"]

        write_npz(cropped_dir / "depth_aligned.npz", depth=cropped_depth_array)
        write_npz(
            cropped_dir / "region_masks.npz",
            M_p=cropped_mp_array,
            M_s=cropped_ms_array,
            M_object=cropped_mobj_array,
        )
        write_npz(
            cropped_dir / "masks_raw.npz",
            human=cropped_h_array,
            object=cropped_o_array,
        )
        write_npz(cropped_dir / "keypoints_2d.npz", keypoints=cropped_kp_array)
        write_npz(
            cropped_dir / "meta.npz",
            bbox_xywh=bbox_xywh_array,
            fx=fx_array,
            fy=fy_array,
            cx=cx_array,
            cy=cy_array,
            orig_size_hw=orig_hw_array,
            downsampled_size_hw=down_hw_array,
            scale_ratio=np.asarray([scale_ratio], dtype=np.int32),
        )

        torch.save(object_gaussians_payload, gs_dir / "G_o.pt")

    return {
        "sequence": raw_seq_dir.name,
        "status": "prepared",
        "frames": num_frames,
        "camera": camera_id,
        "output_dir": str(out_seq_dir),
    }


def worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def run_prepare_task(raw_seq_dir: str, output_root: str, config: PreprocessConfig) -> Dict[str, object]:
    raw_path = Path(raw_seq_dir)
    out_seq_dir = Path(output_root) / raw_path.name
    start_time = time.time()
    try:
        result = prepare_sequence(
            raw_path,
            out_seq_dir,
            camera_id=config.camera_id,
            max_frames=config.max_frames,
            processed_subdir=config.processed_subdir,
            gs_subdir=config.gs_subdir,
            scale_ratio=config.scale_ratio,
            crop_size=config.crop_size,
            bbox_expand=config.bbox_expand,
            num_object_gaussians=config.num_object_gaussians,
            init_gaussian_scale=config.init_gaussian_scale,
            smpl_model_root=config.smpl_model_root,
            overwrite=config.overwrite,
        )
        result["elapsed_seconds"] = round(time.time() - start_time, 3)
        return result
    except Exception as exc:
        return {
            "sequence": raw_path.name,
            "status": "failed",
            "elapsed_seconds": round(time.time() - start_time, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "output_dir": str(out_seq_dir),
        }


def build_progress_payload(
    *,
    run_id: str,
    raw_root: Path,
    output_root: Path,
    status_dir: Path,
    split_file: str,
    split_key: str,
    total_sequences: int,
    already_completed: int,
    pending_total: int,
    prepared_new: int,
    failed_new: int,
    last_event: str,
    num_workers: int,
    start_time: float,
    state: str,
    recent_sequence: str = "",
    active_sequences: Optional[Sequence[str]] = None,
    avg_sequence_seconds: Optional[float] = None,
    error_type: str = "",
    error_message: str = "",
) -> Dict[str, object]:
    processed_new = prepared_new + failed_new
    processed_total = already_completed + processed_new
    pending_remaining = max(pending_total - processed_new, 0)
    elapsed_seconds = time.time() - start_time
    rate = processed_new / elapsed_seconds if processed_new > 0 and elapsed_seconds > 0 else None
    eta_seconds = pending_remaining / rate if rate and pending_remaining > 0 else 0.0 if pending_remaining == 0 else None
    payload = {
        "run_id": run_id,
        "status": state,
        "timestamp": iso_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "status_dir": str(status_dir),
        "split_file": split_file,
        "split_key": split_key,
        "num_workers": int(num_workers),
        "total_sequences": int(total_sequences),
        "already_completed": int(already_completed),
        "scheduled_sequences": int(pending_total),
        "prepared_new": int(prepared_new),
        "failed_new": int(failed_new),
        "processed_total": int(processed_total),
        "pending_remaining": int(pending_remaining),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "avg_sequence_seconds": round(avg_sequence_seconds, 3) if avg_sequence_seconds is not None else None,
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "last_event": last_event,
        "recent_sequence": recent_sequence,
    }
    if error_type:
        payload["error_type"] = error_type
    if error_message:
        payload["error"] = error_message
    if active_sequences is not None:
        payload["active_sequences"] = [str(sequence) for sequence in active_sequences]
    return payload


def log_sequence_result(
    *,
    status_dir: Path,
    run_id: str,
    result: Dict[str, object],
) -> None:
    event = {"timestamp": iso_now(), "run_id": run_id, "type": "sequence_result"}
    event.update(result)
    append_jsonl(status_dir / "events.jsonl", event)
    if result["status"] == "failed":
        append_jsonl(status_dir / "failures.jsonl", event)


def log_run_terminal_event(
    *,
    status_dir: Path,
    run_id: str,
    state: str,
    prepared_new: int,
    failed_new: int,
    start_time: float,
    error_type: str = "",
    error_message: str = "",
    recent_sequence: str = "",
) -> None:
    event = {
        "timestamp": iso_now(),
        "run_id": run_id,
        "type": "run_finished" if state.startswith("completed") else "run_failed",
        "state": state,
        "prepared_new": prepared_new,
        "failed_new": failed_new,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    if error_type:
        event["error_type"] = error_type
    if error_message:
        event["error"] = error_message
    if recent_sequence:
        event["recent_sequence"] = recent_sequence
    append_jsonl(status_dir / "events.jsonl", event)


def process_pending_sequences(
    *,
    raw_root: Path,
    output_root: Path,
    status_dir: Path,
    split_file: str,
    split_key: str,
    config: PreprocessConfig,
    sequence_dirs: Sequence[Path],
    already_completed: int,
    num_workers: int,
    heartbeat_interval: int,
) -> Tuple[int, int]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pending_total = len(sequence_dirs)
    start_time = time.time()
    prepared_new = 0
    failed_new = 0
    duration_sum = 0.0
    recent_sequence = ""

    progress = build_progress_payload(
        run_id=run_id,
        raw_root=raw_root,
        output_root=output_root,
        status_dir=status_dir,
        split_file=split_file,
        split_key=split_key,
        total_sequences=already_completed + pending_total,
        already_completed=already_completed,
        pending_total=pending_total,
        prepared_new=prepared_new,
        failed_new=failed_new,
        last_event="run_started",
        num_workers=num_workers,
        start_time=start_time,
        state="running",
    )
    atomic_write_json(status_dir / "progress.json", progress)
    append_jsonl(
        status_dir / "events.jsonl",
        {
            "timestamp": iso_now(),
            "run_id": run_id,
            "type": "run_started",
            "already_completed": already_completed,
            "scheduled_sequences": pending_total,
            "num_workers": num_workers,
        },
    )

    try:
        if pending_total == 0:
            progress = build_progress_payload(
                run_id=run_id,
                raw_root=raw_root,
                output_root=output_root,
                status_dir=status_dir,
                split_file=split_file,
                split_key=split_key,
                total_sequences=already_completed,
                already_completed=already_completed,
                pending_total=0,
                prepared_new=0,
                failed_new=0,
                last_event="nothing_to_do",
                num_workers=num_workers,
                start_time=start_time,
                state="completed",
            )
            atomic_write_json(status_dir / "progress.json", progress)
            log_run_terminal_event(
                status_dir=status_dir,
                run_id=run_id,
                state="completed",
                prepared_new=0,
                failed_new=0,
                start_time=start_time,
            )
            return 0, 0

        if num_workers <= 1:
            for sequence_dir in sequence_dirs:
                recent_sequence = sequence_dir.name
                avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
                progress = build_progress_payload(
                    run_id=run_id,
                    raw_root=raw_root,
                    output_root=output_root,
                    status_dir=status_dir,
                    split_file=split_file,
                    split_key=split_key,
                    total_sequences=already_completed + pending_total,
                    already_completed=already_completed,
                    pending_total=pending_total,
                    prepared_new=prepared_new,
                    failed_new=failed_new,
                    last_event="sequence_started",
                    recent_sequence=recent_sequence,
                    active_sequences=[recent_sequence],
                    num_workers=num_workers,
                    start_time=start_time,
                    avg_sequence_seconds=avg_seconds,
                    state="running",
                )
                atomic_write_json(status_dir / "progress.json", progress)
                result = run_prepare_task(str(sequence_dir), str(output_root), config)
                if result["status"] == "prepared":
                    prepared_new += 1
                    duration_sum += float(result["elapsed_seconds"])
                    print(
                        f"[procigen-prep] OK {sequence_dir.name} | frames={result['frames']} "
                        f"| camera={result['camera']} | took={format_seconds(result['elapsed_seconds'])}"
                    )
                elif result["status"] == "skipped":
                    print(f"[procigen-prep] SKIP {sequence_dir.name}")
                else:
                    failed_new += 1
                    print(
                        f"[procigen-prep] FAIL {sequence_dir.name}: "
                        f"{result.get('error_type', 'Error')}: {result.get('error', '')}"
                    )
                log_sequence_result(status_dir=status_dir, run_id=run_id, result=result)
                avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
                progress = build_progress_payload(
                    run_id=run_id,
                    raw_root=raw_root,
                    output_root=output_root,
                    status_dir=status_dir,
                    split_file=split_file,
                    split_key=split_key,
                    total_sequences=already_completed + pending_total,
                    already_completed=already_completed,
                    pending_total=pending_total,
                    prepared_new=prepared_new,
                    failed_new=failed_new,
                    last_event=result["status"],
                    recent_sequence=str(result["sequence"]),
                    active_sequences=[],
                    num_workers=num_workers,
                    start_time=start_time,
                    avg_sequence_seconds=avg_seconds,
                    state="running",
                )
                atomic_write_json(status_dir / "progress.json", progress)
        else:
            mp_context = get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp_context,
                initializer=worker_init,
            ) as executor:
                future_to_sequence: Dict[object, str] = {}
                remaining_sequences = iter(sequence_dirs)

                while len(future_to_sequence) < num_workers:
                    try:
                        sequence_dir = next(remaining_sequences)
                    except StopIteration:
                        break
                    future = executor.submit(run_prepare_task, str(sequence_dir), str(output_root), config)
                    future_to_sequence[future] = sequence_dir.name

                if future_to_sequence:
                    avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
                    progress = build_progress_payload(
                        run_id=run_id,
                        raw_root=raw_root,
                        output_root=output_root,
                        status_dir=status_dir,
                        split_file=split_file,
                        split_key=split_key,
                        total_sequences=already_completed + pending_total,
                        already_completed=already_completed,
                        pending_total=pending_total,
                        prepared_new=prepared_new,
                        failed_new=failed_new,
                        last_event="workers_started",
                        recent_sequence=recent_sequence,
                        active_sequences=sorted(future_to_sequence.values()),
                        num_workers=num_workers,
                        start_time=start_time,
                        avg_sequence_seconds=avg_seconds,
                        state="running",
                    )
                    atomic_write_json(status_dir / "progress.json", progress)

                while future_to_sequence:
                    done_futures, _ = wait(
                        set(future_to_sequence.keys()),
                        timeout=max(int(heartbeat_interval), 1),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done_futures:
                        avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
                        progress = build_progress_payload(
                            run_id=run_id,
                            raw_root=raw_root,
                            output_root=output_root,
                            status_dir=status_dir,
                            split_file=split_file,
                            split_key=split_key,
                            total_sequences=already_completed + pending_total,
                            already_completed=already_completed,
                            pending_total=pending_total,
                            prepared_new=prepared_new,
                            failed_new=failed_new,
                            last_event="heartbeat",
                            recent_sequence=recent_sequence,
                            active_sequences=sorted(future_to_sequence.values()),
                            num_workers=num_workers,
                            start_time=start_time,
                            avg_sequence_seconds=avg_seconds,
                            state="running",
                        )
                        atomic_write_json(status_dir / "progress.json", progress)
                        continue

                    for future in done_futures:
                        sequence_name = future_to_sequence.pop(future)
                        recent_sequence = sequence_name
                        result = future.result()
                        if result["status"] == "prepared":
                            prepared_new += 1
                            duration_sum += float(result["elapsed_seconds"])
                            print(
                                f"[procigen-prep] OK {sequence_name} | frames={result['frames']} "
                                f"| camera={result['camera']} | took={format_seconds(result['elapsed_seconds'])}"
                            )
                        elif result["status"] == "skipped":
                            print(f"[procigen-prep] SKIP {sequence_name}")
                        else:
                            failed_new += 1
                            print(
                                f"[procigen-prep] FAIL {sequence_name}: "
                                f"{result.get('error_type', 'Error')}: {result.get('error', '')}"
                            )
                        log_sequence_result(status_dir=status_dir, run_id=run_id, result=result)

                        while len(future_to_sequence) < num_workers:
                            try:
                                sequence_dir = next(remaining_sequences)
                            except StopIteration:
                                break
                            next_future = executor.submit(run_prepare_task, str(sequence_dir), str(output_root), config)
                            future_to_sequence[next_future] = sequence_dir.name

                        avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
                        progress = build_progress_payload(
                            run_id=run_id,
                            raw_root=raw_root,
                            output_root=output_root,
                            status_dir=status_dir,
                            split_file=split_file,
                            split_key=split_key,
                            total_sequences=already_completed + pending_total,
                            already_completed=already_completed,
                            pending_total=pending_total,
                            prepared_new=prepared_new,
                            failed_new=failed_new,
                            last_event=result["status"],
                            recent_sequence=str(result["sequence"]),
                            active_sequences=sorted(future_to_sequence.values()),
                            num_workers=num_workers,
                            start_time=start_time,
                            avg_sequence_seconds=avg_seconds,
                            state="running",
                        )
                        atomic_write_json(status_dir / "progress.json", progress)
    except BrokenProcessPool as exc:
        avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
        error_message = str(exc)
        progress = build_progress_payload(
            run_id=run_id,
            raw_root=raw_root,
            output_root=output_root,
            status_dir=status_dir,
            split_file=split_file,
            split_key=split_key,
            total_sequences=already_completed + pending_total,
            already_completed=already_completed,
            pending_total=pending_total,
            prepared_new=prepared_new,
            failed_new=failed_new,
            last_event="broken_process_pool",
            recent_sequence=recent_sequence,
            num_workers=num_workers,
            start_time=start_time,
            avg_sequence_seconds=avg_seconds,
            state="crashed",
            error_type=type(exc).__name__,
            error_message=error_message,
        )
        atomic_write_json(status_dir / "progress.json", progress)
        log_run_terminal_event(
            status_dir=status_dir,
            run_id=run_id,
            state="crashed",
            prepared_new=prepared_new,
            failed_new=failed_new,
            start_time=start_time,
            error_type=type(exc).__name__,
            error_message=error_message,
            recent_sequence=recent_sequence,
        )
        print(f"[procigen-prep] CRASH {type(exc).__name__}: {error_message}", file=sys.stderr)
        raise
    except KeyboardInterrupt:
        avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
        progress = build_progress_payload(
            run_id=run_id,
            raw_root=raw_root,
            output_root=output_root,
            status_dir=status_dir,
            split_file=split_file,
            split_key=split_key,
            total_sequences=already_completed + pending_total,
            already_completed=already_completed,
            pending_total=pending_total,
            prepared_new=prepared_new,
            failed_new=failed_new,
            last_event="keyboard_interrupt",
            recent_sequence=recent_sequence,
            num_workers=num_workers,
            start_time=start_time,
            avg_sequence_seconds=avg_seconds,
            state="aborted",
            error_type="KeyboardInterrupt",
            error_message="Interrupted by user.",
        )
        atomic_write_json(status_dir / "progress.json", progress)
        log_run_terminal_event(
            status_dir=status_dir,
            run_id=run_id,
            state="aborted",
            prepared_new=prepared_new,
            failed_new=failed_new,
            start_time=start_time,
            error_type="KeyboardInterrupt",
            error_message="Interrupted by user.",
            recent_sequence=recent_sequence,
        )
        raise
    except Exception as exc:
        avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
        error_message = str(exc)
        progress = build_progress_payload(
            run_id=run_id,
            raw_root=raw_root,
            output_root=output_root,
            status_dir=status_dir,
            split_file=split_file,
            split_key=split_key,
            total_sequences=already_completed + pending_total,
            already_completed=already_completed,
            pending_total=pending_total,
            prepared_new=prepared_new,
            failed_new=failed_new,
            last_event="fatal_error",
            recent_sequence=recent_sequence,
            num_workers=num_workers,
            start_time=start_time,
            avg_sequence_seconds=avg_seconds,
            state="crashed",
            error_type=type(exc).__name__,
            error_message=error_message,
        )
        atomic_write_json(status_dir / "progress.json", progress)
        log_run_terminal_event(
            status_dir=status_dir,
            run_id=run_id,
            state="crashed",
            prepared_new=prepared_new,
            failed_new=failed_new,
            start_time=start_time,
            error_type=type(exc).__name__,
            error_message=error_message,
            recent_sequence=recent_sequence,
        )
        raise

    final_state = "completed_with_failures" if failed_new > 0 else "completed"
    avg_seconds = duration_sum / prepared_new if prepared_new > 0 else None
    progress = build_progress_payload(
        run_id=run_id,
        raw_root=raw_root,
        output_root=output_root,
        status_dir=status_dir,
        split_file=split_file,
        split_key=split_key,
        total_sequences=already_completed + pending_total,
        already_completed=already_completed,
        pending_total=pending_total,
        prepared_new=prepared_new,
        failed_new=failed_new,
        last_event=final_state,
        num_workers=num_workers,
        start_time=start_time,
        avg_sequence_seconds=avg_seconds,
        state=final_state,
    )
    atomic_write_json(status_dir / "progress.json", progress)
    log_run_terminal_event(
        status_dir=status_dir,
        run_id=run_id,
        state=final_state,
        prepared_new=prepared_new,
        failed_new=failed_new,
        start_time=start_time,
    )
    return prepared_new, failed_new


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return min(8, max(2, cpu_count // 2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ProciGen GT assets for the new dual-branch FM pipeline.")
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--status_dir", type=str, default="")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--split_key", type=str, default="train")
    parser.add_argument("--sequence_name", action="append", dest="sequence_names")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--camera_id", type=str, default="k1")
    parser.add_argument("--processed_subdir", type=str, default="processed")
    parser.add_argument("--gs_subdir", type=str, default="gs_init")
    parser.add_argument("--scale_ratio", type=int, default=2)
    parser.add_argument("--crop_size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--bbox_expand", type=float, default=1.1)
    parser.add_argument("--num_object_gaussians", type=int, default=4096)
    parser.add_argument("--init_gaussian_scale", type=float, default=0.01)
    parser.add_argument("--smpl_model_root", type=str, default=str(REPO_ROOT / "model" / "smpl_models"))
    parser.add_argument("--num_workers", type=int, default=default_num_workers())
    parser.add_argument("--heartbeat_interval", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    camera_id = normalize_camera_id(args.camera_id)
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else raw_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_dir = (
        Path(args.status_dir).expanduser().resolve()
        if args.status_dir
        else output_root / "_preprocess_logs"
    )
    status_dir.mkdir(parents=True, exist_ok=True)

    sequence_names = list(args.sequence_names or [])
    if args.split_file:
        sequence_names.extend(load_split_sequence_names(args.split_file, args.split_key))
    sequence_names = sorted(set(sequence_names))
    sequence_dirs = discover_sequence_dirs(str(raw_root), sequence_names=sequence_names)
    if args.max_sequences > 0:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    config = PreprocessConfig(
        camera_id=camera_id,
        max_frames=int(args.max_frames),
        processed_subdir=args.processed_subdir,
        gs_subdir=args.gs_subdir,
        scale_ratio=int(args.scale_ratio),
        crop_size=(int(args.crop_size[0]), int(args.crop_size[1])),
        bbox_expand=float(args.bbox_expand),
        num_object_gaussians=int(args.num_object_gaussians),
        init_gaussian_scale=float(args.init_gaussian_scale),
        smpl_model_root=args.smpl_model_root,
        overwrite=bool(args.overwrite),
    )

    if config.overwrite:
        ready_sequence_dirs: List[Path] = []
        pending_sequence_dirs = list(sequence_dirs)
    else:
        ready_sequence_dirs = []
        pending_sequence_dirs = []
        for sequence_dir in sequence_dirs:
            if sequence_is_prepared(output_root / sequence_dir.name, config.processed_subdir, config.gs_subdir):
                ready_sequence_dirs.append(sequence_dir)
            else:
                pending_sequence_dirs.append(sequence_dir)
    pending_sequence_dirs = sort_sequence_dirs_by_estimated_size(pending_sequence_dirs)

    write_lines(status_dir / "all_sequences.txt", [sequence_dir.name for sequence_dir in sequence_dirs])
    write_lines(status_dir / "ready_sequences.txt", [sequence_dir.name for sequence_dir in ready_sequence_dirs])
    write_lines(status_dir / "pending_sequences.txt", [sequence_dir.name for sequence_dir in pending_sequence_dirs])
    atomic_write_json(
        status_dir / "run_config.json",
        {
            "timestamp": iso_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "status_dir": str(status_dir),
            "args": vars(args),
            "resolved_config": asdict(config),
            "total_sequences": len(sequence_dirs),
            "already_completed": len(ready_sequence_dirs),
            "pending_sequences": len(pending_sequence_dirs),
        },
    )

    print(
        f"[procigen-prep] start | total={len(sequence_dirs)} "
        f"| already_completed={len(ready_sequence_dirs)} "
        f"| pending={len(pending_sequence_dirs)} "
        f"| workers={max(int(args.num_workers), 1)}"
    )

    prepared_new, failures = process_pending_sequences(
        raw_root=raw_root,
        output_root=output_root,
        status_dir=status_dir,
        split_file=args.split_file,
        split_key=args.split_key,
        config=config,
        sequence_dirs=pending_sequence_dirs,
        already_completed=len(ready_sequence_dirs),
        num_workers=max(int(args.num_workers), 1),
        heartbeat_interval=max(int(args.heartbeat_interval), 1),
    )

    total = len(sequence_dirs)
    print(
        f"[procigen-prep] done | total={total} "
        f"| already_completed={len(ready_sequence_dirs)} "
        f"| prepared_new={prepared_new} | failures={failures}"
    )
    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
