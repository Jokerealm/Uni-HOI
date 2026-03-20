"""OpenPose 2D keypoint detector wrapper."""
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
                raise RuntimeError(
                    f"[OpenPose] Failed to load model from {model_dir}."
                ) from e
        else:
            raise FileNotFoundError(
                f"[OpenPose] Missing Caffe weights under {model_dir}."
            )

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Detect 2D keypoints for a single frame.

        Returns
        -------
        keypoints : np.ndarray, shape (18, 3) — (x, y, confidence)
        """
        if self._net is None:
            raise RuntimeError("OpenPose network is unavailable.")

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
