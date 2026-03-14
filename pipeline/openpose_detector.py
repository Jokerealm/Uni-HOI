"""
OpenPose 2D keypoint detector wrapper.

Falls back to stub keypoints if OpenPose is unavailable.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np


# COCO 18 keypoints order
COCO_KEYPOINT_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
    "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
    "l_eye", "r_ear", "l_ear",
]


class OpenPoseDetector:
    """Extracts 2D body keypoints per frame."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self._net = None
        self._load(model_dir)

    def _load(self, model_dir: str):
        proto = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
        weights = os.path.join(model_dir, "pose_iter_440000.caffemodel")
        if os.path.isfile(proto) and os.path.isfile(weights):
            try:
                self._net = cv2.dnn.readNetFromCaffe(proto, weights)
                if "cuda" in self.device:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[OpenPose] Loaded Caffe model.")
            except Exception as e:
                print(f"[OpenPose] Failed to load: {e}. Using stub.")
        else:
            print(f"[OpenPose] Weights not found in {model_dir}. Using stub.")

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect 2D keypoints for a single frame.

        Returns
        -------
        keypoints : np.ndarray, shape (18, 3) — (x, y, confidence)
        """
        if self._net is None:
            return self._detect_stub(frame_bgr)

        H, W = frame_bgr.shape[:2]
        inp_h = 368
        inp_w = int(W * inp_h / H)
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1.0 / 255, (inp_w, inp_h), (0, 0, 0),
            swapRB=False, crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()  # shape: (1, 57, out_h, out_w)

        n_parts = 18
        keypoints = np.zeros((n_parts, 3), dtype=np.float32)
        for i in range(n_parts):
            heatmap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            x = int(point[0] * W / output.shape[3])
            y = int(point[1] * H / output.shape[2])
            keypoints[i] = [x, y, conf]
        return keypoints

    def detect_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Detect keypoints for all frames.

        Returns
        -------
        all_kps : np.ndarray, shape (T, 18, 3)
        """
        from tqdm import tqdm
        kps = [self.detect(f) for f in tqdm(frames, desc="OpenPose")]
        return np.stack(kps, axis=0)

    @staticmethod
    def _detect_stub(frame_bgr: np.ndarray) -> np.ndarray:
        """Return dummy keypoints at image center."""
        H, W = frame_bgr.shape[:2]
        cx, cy = W // 2, H // 2
        kps = np.zeros((18, 3), dtype=np.float32)
        # Place keypoints in a rough body layout around center
        offsets = [
            (0, -80), (0, -60),                          # nose, neck
            (-30, -50), (-50, -20), (-60, 10),            # r_shoulder..r_wrist
            (30, -50), (50, -20), (60, 10),               # l_shoulder..l_wrist
            (-20, 20), (-20, 60), (-20, 100),             # r_hip..r_ankle
            (20, 20), (20, 60), (20, 100),                # l_hip..l_ankle
            (-10, -90), (10, -90), (-20, -85), (20, -85), # eyes, ears
        ]
        for i, (dx, dy) in enumerate(offsets):
            kps[i] = [cx + dx, cy + dy, 0.5]
        return kps
