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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def detect_camera(seq_dir: str) -> str:
    """Auto-detect the best camera ID from available frames."""
    from collections import Counter
    files = glob.glob(os.path.join(seq_dir, "t*.000", "k*.color.jpg"))
    if not files:
        return "k1"  # default
    cam_counts = Counter(os.path.basename(p).split(".")[0] for p in files)
    return cam_counts.most_common(1)[0][0]


def load_behave_sequence(
    seq_dir: str,
    cam_id: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """
    Load all GT data from a BEHAVE sequence directory.

    Returns dict with:
      - frames: list of (H, W, 3) BGR uint8
      - frame_paths: list of str
      - masks_human: list of (H, W) uint8 {0, 255}
      - masks_object: list of (H, W) uint8 {0, 255}
      - smpl_params: list of dicts with SMPL-H parameters
      - keypoints_3d: (T, J, 3) float32
      - cam_id: str
    """
    if cam_id is None:
        cam_id = detect_camera(seq_dir)
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

    for t_dir in timesteps:
        t_name = os.path.basename(t_dir)

        # --- RGB frame ---
        color_path = os.path.join(t_dir, f"{cam_id}.color.jpg")
        if not os.path.isfile(color_path):
            print(f"[BEHAVE-GT] Warning: missing {color_path}, skipping")
            continue
        img = cv2.imread(color_path)
        frames.append(img)
        frame_paths.append(color_path)

        H, W = img.shape[:2]

        # --- Person mask ---
        pmask_path = os.path.join(t_dir, f"{cam_id}.person_mask.jpg")
        if os.path.isfile(pmask_path):
            pm = cv2.imread(pmask_path, cv2.IMREAD_GRAYSCALE)
            masks_human.append((pm > 127).astype(np.uint8) * 255)
        else:
            masks_human.append(np.zeros((H, W), dtype=np.uint8))

        # --- Object mask ---
        omask_path = os.path.join(t_dir, f"{cam_id}.obj_rend_mask.jpg")
        if os.path.isfile(omask_path):
            om = cv2.imread(omask_path, cv2.IMREAD_GRAYSCALE)
            masks_object.append((om > 127).astype(np.uint8) * 255)
        else:
            masks_object.append(np.zeros((H, W), dtype=np.uint8))

        # --- SMPL parameters ---
        smpl_fit_path = os.path.join(t_dir, "person", "fit02", "person_fit.pkl")
        if os.path.isfile(smpl_fit_path):
            with open(smpl_fit_path, "rb") as f:
                smpl = pickle.load(f, encoding="latin1")
            smpl_params_list.append({
                "body_pose": np.array(smpl.get("pose", np.zeros(72)), dtype=np.float32),
                "shape": np.array(smpl.get("betas", np.zeros(10)), dtype=np.float32),
                "cam_t": np.array(smpl.get("trans", np.zeros(3)), dtype=np.float32),
            })
        else:
            smpl_params_list.append({
                "body_pose": np.zeros(72, dtype=np.float32),
                "shape": np.zeros(10, dtype=np.float32),
                "cam_t": np.zeros(3, dtype=np.float32),
            })

        # --- 3D joints ---
        j3d_path = os.path.join(t_dir, "person", "person_J3d.json")
        if os.path.isfile(j3d_path):
            with open(j3d_path) as f:
                j3d_data = json.load(f)
            if isinstance(j3d_data, dict) and "body_joints3d" in j3d_data:
                # BEHAVE format: flat list of J*4 floats (x, y, z, conf)
                arr = np.array(j3d_data["body_joints3d"], dtype=np.float32)
                joints = arr.reshape(-1, 4)[:, :3]  # drop confidence, keep xyz
            elif isinstance(j3d_data, dict):
                joints = np.array(list(j3d_data.values()), dtype=np.float32)
            elif isinstance(j3d_data, list):
                joints = np.array(j3d_data, dtype=np.float32)
            else:
                joints = np.zeros((25, 3), dtype=np.float32)
            joints_3d_list.append(joints)
        else:
            joints_3d_list.append(np.zeros((25, 3), dtype=np.float32))

    T = len(frames)
    print(f"[BEHAVE-GT] Loaded {T} frames from {seq_dir}")

    # Stack joints — pad to uniform shape if needed
    max_j = max(j.shape[0] for j in joints_3d_list) if joints_3d_list else 25
    joints_3d = np.zeros((T, max_j, 3), dtype=np.float32)
    for i, j in enumerate(joints_3d_list):
        joints_3d[i, :j.shape[0]] = j

    return {
        "frames": frames,
        "frame_paths": frame_paths,
        "masks_human": masks_human,
        "masks_object": masks_object,
        "smpl_params": smpl_params_list,
        "keypoints_3d": joints_3d,
        "cam_id": cam_id,
    }
