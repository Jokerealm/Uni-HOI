"""UniDepth V2 metric depth estimator wrapper."""
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
            import os as _os
            # Force offline mode — weights already cached locally, skip HF hub checks
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            from unidepth.models import UniDepthV2
            hf_name = f"lpiccinelli/unidepth-v2-{backbone}"
            self._model = UniDepthV2.from_pretrained(hf_name).to(self.device).eval()
            print(f"[UniDepth] Loaded model from {hf_name} (offline)")
        except Exception as e:
            raise RuntimeError(
                f"[UniDepth] Failed to load model `{backbone}`. "
                "Step 1 cannot continue without metric depth."
            ) from e

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Predict metric depth for a single frame.

        Returns depth map (H, W) float32 in meters.
        """
        if self._model is None:
            raise RuntimeError("UniDepth model is unavailable.")

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
