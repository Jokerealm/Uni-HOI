"""
BEHAVE dataset GT data loader.

Directly reads GT masks, SMPL parameters, and 3D joints from the BEHAVE
dataset structure, bypassing perception models (SAM3, OpenPose, etc.).

BEHAVE frame layout:
  <seq_dir>/t*.000/k{cam_id}.color.jpg
  <seq_dir>/t*.000/k{cam_id}.person_mask.jpg
  <seq_dir>/t*.000/k{cam_id}.obj_rend_mask.jpg
  <seq_dir>/t*.000/person/fit02/person_fit.pkl
  <seq_dir>/t*.000/person/person_J3d.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from dataset.video_transforms import infer_camera_intrinsics


def normalize_camera_id(cam_id: Union[str, int]) -> str:
    """Normalize camera identifiers like 1 / "1" / "k1" to "k1"."""
    if isinstance(cam_id, int):
        return f"k{cam_id}"
    cam = str(cam_id).strip()
    if cam.startswith("k"):
        return cam
    if cam.isdigit():
        return f"k{cam}"
    raise ValueError(f"Unsupported camera id: {cam_id}")


def detect_camera(seq_dir: str) -> str:
    """Auto-detect the best camera ID from available frames."""
    from collections import Counter
    files = sorted(glob.glob(os.path.join(seq_dir, "t*.000", "k*.color.jpg")))
    if not files:
        return "k1"  # default
    cam_counts = Counter(os.path.basename(p).split(".")[0] for p in files)
    max_count = max(cam_counts.values())
    candidates = [cam for cam, count in cam_counts.items() if count == max_count]
    if "k1" in candidates:
        return "k1"
    return min(candidates, key=lambda cam: int(cam[1:]))


_PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def _read_ply_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a triangular PLY mesh into `(vertices, faces)`.

    BEHAVE exports binary little-endian PLY files, but this reader also accepts
    ASCII files with the same vertex/face schema.
    """

    with open(mesh_path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {mesh_path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError(f"Unsupported PLY file (missing magic): {mesh_path}")

        fmt = None
        vertex_count = 0
        face_count = 0
        vertex_props: List[Tuple[str, str]] = []
        face_count_type = None
        face_index_type = None
        section = None

        for line in header_lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                section = parts[1]
                if section == "vertex":
                    vertex_count = int(parts[2])
                elif section == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and section == "vertex":
                vertex_props.append((parts[2], parts[1]))
            elif parts[0] == "property" and section == "face":
                if parts[1] != "list":
                    raise ValueError(f"Unsupported face property in {mesh_path}: {line}")
                face_count_type = parts[2]
                face_index_type = parts[3]

        if fmt not in {"binary_little_endian", "ascii"}:
            raise ValueError(f"Unsupported PLY format in {mesh_path}: {fmt}")
        if vertex_count <= 0:
            raise ValueError(f"PLY mesh has no vertices: {mesh_path}")
        if not vertex_props:
            raise ValueError(f"PLY mesh is missing vertex properties: {mesh_path}")

        if fmt == "binary_little_endian":
            vertex_dtype = np.dtype([(name, _PLY_DTYPE_MAP[dtype]) for name, dtype in vertex_props])
            vertices_raw = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
            vertices = np.stack(
                [vertices_raw["x"], vertices_raw["y"], vertices_raw["z"]],
                axis=-1,
            ).astype(np.float32)

            faces = np.zeros((face_count, 3), dtype=np.int32)
            if face_count > 0:
                if face_count_type not in _PLY_DTYPE_MAP or face_index_type not in _PLY_DTYPE_MAP:
                    raise ValueError(f"Unsupported face list types in {mesh_path}: {face_count_type}, {face_index_type}")
                count_dtype = np.dtype(_PLY_DTYPE_MAP[face_count_type])
                index_dtype = np.dtype(_PLY_DTYPE_MAP[face_index_type])
                for face_idx in range(face_count):
                    num_indices = int(np.fromfile(f, dtype=count_dtype, count=1)[0])
                    indices = np.fromfile(f, dtype=index_dtype, count=num_indices).astype(np.int32)
                    if num_indices != 3:
                        raise ValueError(f"Non-triangular face with {num_indices} vertices in {mesh_path}")
                    faces[face_idx] = indices
            return vertices, faces

        vertices_list: List[List[float]] = []
        for _ in range(vertex_count):
            parts = f.readline().decode("ascii").strip().split()
            prop_values = {
                name: float(parts[prop_idx])
                for prop_idx, (name, _) in enumerate(vertex_props)
            }
            vertices_list.append([prop_values["x"], prop_values["y"], prop_values["z"]])
        vertices = np.asarray(vertices_list, dtype=np.float32)

        faces_list: List[List[int]] = []
        for _ in range(face_count):
            parts = f.readline().decode("ascii").strip().split()
            num_indices = int(parts[0])
            if num_indices != 3:
                raise ValueError(f"Non-triangular ASCII face with {num_indices} vertices in {mesh_path}")
            faces_list.append([int(parts[1]), int(parts[2]), int(parts[3])])
        faces = np.asarray(faces_list, dtype=np.int32)
        return vertices, faces


def _load_person_mesh(t_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    candidate_paths = [
        os.path.join(t_dir, "person", "fit02", "person_fit.ply"),
        os.path.join(t_dir, "person", "person.ply"),
    ]
    mesh_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
    if mesh_path is None:
        raise FileNotFoundError(
            f"Missing BEHAVE person mesh in {t_dir}. Tried: {candidate_paths}"
        )
    return _read_ply_mesh(mesh_path)


def _load_person_joints3d(t_dir: str) -> np.ndarray:
    j3d_path = os.path.join(t_dir, "person", "person_J3d.json")
    if os.path.isfile(j3d_path):
        with open(j3d_path) as f:
            j3d_data = json.load(f)
        if isinstance(j3d_data, dict) and "body_joints3d" in j3d_data:
            arr = np.array(j3d_data["body_joints3d"], dtype=np.float32)
            return arr.reshape(-1, 4)[:, :3]
        if isinstance(j3d_data, dict):
            return np.array(list(j3d_data.values()), dtype=np.float32)
        if isinstance(j3d_data, list):
            return np.array(j3d_data, dtype=np.float32)
    raise FileNotFoundError(f"Missing BEHAVE 3D joints file: {j3d_path}")


def _reshape_keypoints_2d(raw_keypoints: object) -> Optional[np.ndarray]:
    """Convert a flattened BEHAVE/OpenPose keypoint list into `(J, 3)`."""
    if raw_keypoints is None:
        return None

    arr = np.asarray(raw_keypoints, dtype=np.float32).reshape(-1)
    usable = (arr.size // 3) * 3
    if usable == 0:
        return None
    return arr[:usable].reshape(-1, 3)


def _project_joints3d_to_pixels(
    joints_3d: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Fallback projection for sequences without stored 2D detections."""
    image_h, image_w = image_shape
    fx, fy, cx, cy = infer_camera_intrinsics(
        image_width=image_w,
        image_height=image_h,
        scale_ratio=1,
    )

    joints = np.asarray(joints_3d, dtype=np.float32)
    z = np.clip(joints[:, 2], 1e-6, None)
    x = joints[:, 0] * fx / z + cx
    y = joints[:, 1] * fy / z + cy
    conf = np.ones((joints.shape[0], 1), dtype=np.float32)
    return np.concatenate([x[:, None], y[:, None], conf], axis=1).astype(np.float32)


def _load_person_keypoints2d(
    t_dir: str,
    cam_id: str,
    image_shape: Tuple[int, int],
    joints_3d: np.ndarray,
) -> np.ndarray:
    """
    Load BEHAVE body keypoints from `<cam>.color.json`.
    """
    color_json_path = os.path.join(t_dir, f"{cam_id}.color.json")
    if not os.path.isfile(color_json_path):
        raise FileNotFoundError(f"Missing BEHAVE 2D keypoint file: {color_json_path}")
    try:
        with open(color_json_path) as f:
            detections = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse BEHAVE keypoints file: {color_json_path}") from exc
    keypoints_2d = _reshape_keypoints_2d(detections.get("body_joints"))
    if keypoints_2d is None:
        raise ValueError(
            f"BEHAVE keypoints file does not contain valid `body_joints`: {color_json_path}"
        )
    return keypoints_2d.astype(np.float32)


def load_behave_sequence(
    seq_dir: str,
    cam_id: Optional[Union[str, int]] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """
    Load all GT data from a BEHAVE sequence directory.

    Returns dict with:
      - frames: list of (H, W, 3) BGR uint8
      - frame_paths: list of str
      - masks_human: list of (H, W) uint8 {0, 255}
      - masks_object: list of (H, W) uint8 {0, 255}
      - smpl_params: list of dicts with SMPL-H parameters + mesh geometry
      - keypoints_3d: (T, J, 3) float32
      - cam_id: str
    """
    if cam_id is None:
        cam_id = detect_camera(seq_dir)
    else:
        cam_id = normalize_camera_id(cam_id)
    print(f"[BEHAVE-GT] Using camera: {cam_id}")

    # Find all timestep directories
    timesteps = sorted(glob.glob(os.path.join(seq_dir, "t*.000")))
    if max_frames is not None:
        timesteps = timesteps[:max_frames]
    assert len(timesteps) > 0, f"No timesteps found in {seq_dir}"

    frames, frame_paths = [], []
    masks_human, masks_object = [], []
    smpl_params_list = []
    joints_3d_list = []
    keypoints_2d_list = []
    canonical_faces = None

    for t_dir in timesteps:
        t_name = os.path.basename(t_dir)

        # --- RGB frame ---
        color_path = os.path.join(t_dir, f"{cam_id}.color.jpg")
        if not os.path.isfile(color_path):
            raise FileNotFoundError(f"Missing BEHAVE RGB frame: {color_path}")
        img = cv2.imread(color_path)
        if img is None:
            raise RuntimeError(f"Failed to read BEHAVE RGB frame: {color_path}")
        frames.append(img)
        frame_paths.append(color_path)

        H, W = img.shape[:2]

        # --- Person mask ---
        pmask_path = os.path.join(t_dir, f"{cam_id}.person_mask.jpg")
        if not os.path.isfile(pmask_path):
            raise FileNotFoundError(f"Missing BEHAVE person mask: {pmask_path}")
        pm = cv2.imread(pmask_path, cv2.IMREAD_GRAYSCALE)
        if pm is None:
            raise RuntimeError(f"Failed to read BEHAVE person mask: {pmask_path}")
        masks_human.append((pm > 127).astype(np.uint8) * 255)

        # --- Object mask ---
        omask_path = os.path.join(t_dir, f"{cam_id}.obj_rend_mask.jpg")
        if not os.path.isfile(omask_path):
            raise FileNotFoundError(f"Missing BEHAVE object mask: {omask_path}")
        om = cv2.imread(omask_path, cv2.IMREAD_GRAYSCALE)
        if om is None:
            raise RuntimeError(f"Failed to read BEHAVE object mask: {omask_path}")
        masks_object.append((om > 127).astype(np.uint8) * 255)

        # --- 3D joints ---
        joints = _load_person_joints3d(t_dir)
        joints_3d_list.append(joints)
        keypoints_2d_list.append(
            _load_person_keypoints2d(
                t_dir=t_dir,
                cam_id=cam_id,
                image_shape=(H, W),
                joints_3d=joints,
            )
        )

        # --- Human mesh ---
        vertices, faces = _load_person_mesh(t_dir)
        if canonical_faces is None:
            canonical_faces = faces
        elif canonical_faces.shape != faces.shape or not np.array_equal(canonical_faces, faces):
            raise ValueError(
                f"BEHAVE person mesh topology changed across frames in {seq_dir}. "
                f"Expected constant faces, got mismatch at {t_name}."
            )

        # --- SMPL parameters ---
        smpl_fit_path = os.path.join(t_dir, "person", "fit02", "person_fit.pkl")
        if not os.path.isfile(smpl_fit_path):
            raise FileNotFoundError(f"Missing BEHAVE SMPL fit: {smpl_fit_path}")
        with open(smpl_fit_path, "rb") as f:
            smpl = pickle.load(f, encoding="latin1")
        missing_smpl_keys = [key for key in ("pose", "betas", "trans") if key not in smpl]
        if missing_smpl_keys:
            raise KeyError(
                f"BEHAVE SMPL fit {smpl_fit_path} is missing keys: {missing_smpl_keys}"
            )
        smpl_params_list.append({
            "body_pose": np.array(smpl["pose"], dtype=np.float32),
            "shape": np.array(smpl["betas"], dtype=np.float32),
            "cam_t": np.array(smpl["trans"], dtype=np.float32),
            "vertices": vertices.astype(np.float32),
            "faces": faces.astype(np.int32),
            "joints_3d": joints.astype(np.float32),
        })

    T = len(frames)
    print(f"[BEHAVE-GT] Loaded {T} frames from {seq_dir}")

    # Stack joints — pad to uniform shape if needed
    max_j = max(j.shape[0] for j in joints_3d_list) if joints_3d_list else 25
    joints_3d = np.zeros((T, max_j, 3), dtype=np.float32)
    for i, j in enumerate(joints_3d_list):
        joints_3d[i, :j.shape[0]] = j

    max_j2d = max(j.shape[0] for j in keypoints_2d_list) if keypoints_2d_list else max_j
    keypoints_2d = np.zeros((T, max_j2d, 3), dtype=np.float32)
    for i, j in enumerate(keypoints_2d_list):
        keypoints_2d[i, :j.shape[0]] = j

    return {
        "frames": frames,
        "frame_paths": frame_paths,
        "masks_human": masks_human,
        "masks_object": masks_object,
        "smpl_params": smpl_params_list,
        "keypoints_3d": joints_3d,
        "keypoints_2d": keypoints_2d,
        "cam_id": cam_id,
    }
