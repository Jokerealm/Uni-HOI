"""
SAM 3 wrapper: text-prompted 2D segmentation + video tracking.

Uses the HuggingFace transformers API:
  - Sam3Model / Sam3Processor for per-image text-prompted segmentation
  - Sam3VideoModel / Sam3VideoProcessor for video-level tracking
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


class SAM3Segmenter:
    """
    Wraps SAM 3 for text-prompted instance segmentation and video tracking.
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self._image_model = None
        self._image_processor = None
        self._video_model = None
        self._video_processor = None

        self._load_models(model_dir)

    def _load_models(self, model_dir: str):
        try:
            from transformers import (
                Sam3Model, Sam3Processor,
                Sam3VideoModel, Sam3VideoProcessor,
            )

            self._image_model = Sam3Model.from_pretrained(
                model_dir, trust_remote_code=True
            ).to(self.device).eval()
            self._image_processor = Sam3Processor.from_pretrained(
                model_dir, trust_remote_code=True
            )
            self._video_model = Sam3VideoModel.from_pretrained(
                model_dir, trust_remote_code=True
            ).to(self.device, dtype=torch.bfloat16).eval()
            self._video_processor = Sam3VideoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )
            print("[SAM3] Loaded image + video models.")
        except Exception as e:
            raise RuntimeError(
                f"[SAM3] Failed to load SAM3 models from {model_dir}. "
                "Step 1 cannot continue without a real segmenter."
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_first_frame(
        self,
        frame_bgr: np.ndarray,
        text_prompts: List[str],
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> dict:
        """
        Segment the first frame using text prompts.

        Returns dict mapping prompt text -> binary mask (H, W) uint8 {0, 255}.
        """
        if self._image_model is None:
            raise RuntimeError("SAM3 image model is unavailable.")

        from PIL import Image as PILImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        H, W = frame_bgr.shape[:2]

        result = {}
        for prompt in text_prompts:
            inputs = self._image_processor(
                images=pil_img, text=prompt, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self._image_model(**inputs)
            post = self._image_processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            if len(post["masks"]) > 0:
                # Merge all instance masks for this prompt into one
                merged = post["masks"].any(dim=0).cpu().numpy().astype(np.uint8) * 255
            else:
                merged = np.zeros((H, W), dtype=np.uint8)
            result[prompt] = merged
        return result

    def track_video(
        self,
        frames: List[np.ndarray],
        text_prompts: List[str],
    ) -> dict:
        """
        Track objects across all frames using text prompts on the video model.

        Returns dict mapping prompt text -> list of (H, W) uint8 masks.
        """
        if self._video_model is None:
            raise RuntimeError("SAM3 video model is unavailable.")

        from PIL import Image as PILImage
        H, W = frames[0].shape[:2]

        # Convert BGR -> RGB PIL
        pil_frames = [
            PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames
        ]

        result = {p: [None] * len(frames) for p in text_prompts}

        for prompt in text_prompts:
            inference_session = self._video_processor.init_video_session(
                video=pil_frames,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            inference_session = self._video_processor.add_text_prompt(
                inference_session=inference_session,
                text=prompt,
            )
            for model_outputs in self._video_model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(frames),
            ):
                processed = self._video_processor.postprocess_outputs(
                    inference_session, model_outputs
                )
                fidx = model_outputs.frame_idx
                if len(processed["masks"]) > 0:
                    merged = processed["masks"].any(dim=0).squeeze().cpu().numpy()
                    mask = (merged > 0).astype(np.uint8) * 255
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
                result[prompt][fidx] = mask

            # Fill any None frames
            blank = np.zeros((H, W), dtype=np.uint8)
            result[prompt] = [m if m is not None else blank for m in result[prompt]]

        return result
