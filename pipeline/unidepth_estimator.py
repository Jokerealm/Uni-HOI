"""
UniDepth V2 metric depth estimator wrapper.

Falls back to inverse-distance stub if UniDepth is unavailable.
"""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch


class UniDepthEstimator:
    """Estimates per-pixel metric (absolute) depth."""

    def __init__(self, model_dir: str, backbone: str = "vitl14",
                 device: str = "cuda"):
        self.device = device
        self._model = None
        self._load(model_dir, backbone)

    def _load(self, model_dir: str, backbone: str):
        try:
            from unidepth.models import UniDepthV2
            self._model = UniDepthV2.from_pretrained(
                model_dir
            ).to(self.device).eval()
            print("[UniDepth] Loaded model.")
        except Exception as e:
            print(f"[UniDepth] Could not load: {e}. Using stub.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Predict metric depth for a single frame.

        Returns depth map (H, W) float32 in meters.
        """
        if self._model is None:
            return self._predict_stub(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self._model.infer(img_t)
        depth = preds["depth"].squeeze().cpu().numpy()  # (H, W)
        return depth.astype(np.float32)

    def predict_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Predict depth for all frames.

        Returns (T, H, W) float32.
        """
        from tqdm import tqdm
        depths = [self.predict(f) for f in tqdm(frames, desc="UniDepth")]
        return np.stack(depths, axis=0)

    @staticmethod
    def _predict_stub(frame_bgr: np.ndarray) -> np.ndarray:
        """Return a synthetic depth map (linear gradient, 1-5m range)."""
        H, W = frame_bgr.shape[:2]
        depth = np.linspace(1.0, 5.0, H, dtype=np.float32)
        depth = np.tile(depth[:, None], (1, W))
        return depth
