"""
SAM 3D Body wrapper: robust SMPL-H mesh recovery from single images.

Falls back to stub T-pose parameters if the model is unavailable.
"""
from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np
import torch


class SAM3DBodyEstimator:
    """Estimates SMPL-H body parameters per frame."""

    def __init__(self, checkpoint: str, mhr_model: str,
                 config_yaml: str, device: str = "cuda"):
        self.device = device
        self._estimator = None
        self._load(checkpoint, mhr_model, config_yaml)

    def _load(self, checkpoint: str, mhr_model: str, config_yaml: str):
        try:
            from sam3d_body.notebook.utils import setup_sam_3d_body
            self._estimator = setup_sam_3d_body(
                checkpoint_path=checkpoint,
                mhr_path=mhr_model,
                config_path=config_yaml,
                device=self.device,
            )
            print("[SAM3D-Body] Loaded estimator.")
        except Exception as e:
            print(f"[SAM3D-Body] Could not load: {e}. Using stub.")

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict SMPL-H parameters for a single frame.

        Returns dict with keys:
          - body_pose: (1, 63) body pose params
          - hand_pose: (1, 90) hand pose params
          - shape: (1, 10) shape betas
          - cam_t: (1, 3) camera translation
          - keypoints_3d: (J, 3) 3D joint positions
          - keypoints_2d: (J, 2) projected 2D joints
          - vertices: (V, 3) mesh vertices
          - focal_length: float
        """
        if self._estimator is None:
            return self._predict_stub(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self._estimator.process_one_image(rgb)

        # Take the first detected person
        if isinstance(outputs, list):
            out = outputs[0] if len(outputs) > 0 else self._predict_stub(frame_bgr)
        else:
            out = outputs

        result = {}
        for key in ["body_pose_params", "hand_pose_params", "shape_params",
                     "pred_cam_t", "pred_keypoints_3d", "pred_keypoints_2d",
                     "pred_vertices", "focal_length"]:
            val = out.get(key)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                result[key] = np.asarray(val)

        # Normalize key names
        return {
            "body_pose": result.get("body_pose_params", np.zeros((1, 63))),
            "hand_pose": result.get("hand_pose_params", np.zeros((1, 90))),
            "shape": result.get("shape_params", np.zeros((1, 10))),
            "cam_t": result.get("pred_cam_t", np.zeros((1, 3))),
            "keypoints_3d": result.get("pred_keypoints_3d", np.zeros((22, 3))),
            "keypoints_2d": result.get("pred_keypoints_2d", np.zeros((22, 2))),
            "vertices": result.get("pred_vertices", np.zeros((6890, 3))),
            "focal_length": result.get("focal_length", np.array([1000.0])),
        }

    def predict_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """Predict SMPL-H for all frames."""
        from tqdm import tqdm
        return [self.predict(f) for f in tqdm(frames, desc="SAM3D-Body")]

    @staticmethod
    def _predict_stub(frame_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Return T-pose stub parameters."""
        H, W = frame_bgr.shape[:2]
        return {
            "body_pose": np.zeros((1, 63), dtype=np.float32),
            "hand_pose": np.zeros((1, 90), dtype=np.float32),
            "shape": np.zeros((1, 10), dtype=np.float32),
            "cam_t": np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
            "keypoints_3d": np.zeros((22, 3), dtype=np.float32),
            "keypoints_2d": np.zeros((22, 2), dtype=np.float32),
            "vertices": np.zeros((6890, 3), dtype=np.float32),
            "focal_length": np.array([1000.0], dtype=np.float32),
        }
