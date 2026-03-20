"""SAM 3D Body wrapper: robust SMPL-H mesh recovery from single images."""
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
            raise RuntimeError(
                "[SAM3D-Body] Failed to load the estimator. "
                "Step 1 cannot continue without SMPL-H recovery."
            ) from e

    def predict(self, frame_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict SMPL-H parameters for a single frame.

        Returns dict with keys:
          - body_pose: (1, 63) body pose params
          - hand_pose: (1, 90) hand pose params
          - shape: (1, 10) shape betas
          - cam_t: (1, 3) camera translation
          - keypoints_3d: (J, 3) 3D joint positions
          - keypoints_2d: (J, 2) projected 2D joints in input-image pixels
          - vertices: (V, 3) mesh vertices
          - focal_length: float
        """
        if self._estimator is None:
            raise RuntimeError("SAM3D-Body estimator is unavailable.")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self._estimator.process_one_image(rgb)

        # Take the first detected person
        if isinstance(outputs, list):
            if len(outputs) == 0:
                raise RuntimeError("SAM3D-Body returned no detections for the input frame.")
            out = outputs[0]
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

        required = (
            "body_pose_params",
            "hand_pose_params",
            "shape_params",
            "pred_cam_t",
            "pred_keypoints_3d",
            "pred_keypoints_2d",
            "pred_vertices",
        )
        missing = [key for key in required if key not in result]
        if missing:
            raise KeyError(
                f"SAM3D-Body output is missing required fields: {missing}."
            )

        return {
            "body_pose": result["body_pose_params"],
            "hand_pose": result["hand_pose_params"],
            "shape": result["shape_params"],
            "cam_t": result["pred_cam_t"],
            "keypoints_3d": result["pred_keypoints_3d"],
            "keypoints_2d": result["pred_keypoints_2d"],
            "vertices": result["pred_vertices"],
            "focal_length": result.get("focal_length", np.array([1000.0])),
        }

    def predict_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """Predict SMPL-H for all frames."""
        from tqdm import tqdm
        return [self.predict(f) for f in tqdm(frames, desc="SAM3D-Body")]
