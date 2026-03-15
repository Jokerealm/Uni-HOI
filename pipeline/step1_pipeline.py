"""
Step 1 Pipeline: Spatial Alignment & Multi-Region Mask Extraction.

Orchestrates:
  1. Frame loading
  2. SAM3 text-prompted segmentation + video tracking
  3. UniDepth V2 metric depth estimation
  4. SAM3D-Body SMPL-H mesh recovery + 2D keypoint extraction
  5. Depth scale alignment
  6. Multi-region contact-aware masking
  7. Output serialization (.npz)
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np

from configs.step1_config import Step1PipelineConfig
from pipeline.io_utils import load_frames, save_masks_png, save_npz
from pipeline.sam3_segmenter import SAM3Segmenter
from pipeline.unidepth_estimator import UniDepthEstimator
from pipeline.sam3d_body import SAM3DBodyEstimator
from pipeline.depth_align import align_sequence
from pipeline.multi_region_mask import compute_multi_region_sequence


class Step1Pipeline:
    """
    End-to-end data preparation pipeline for Step 1.

    Usage:
        cfg = Step1PipelineConfig()
        pipeline = Step1Pipeline(cfg)
        pipeline.run()
    """

    def __init__(self, cfg: Step1PipelineConfig):
        self.cfg = cfg
        self.device = cfg.data_prep.device

        # Resolve paths
        candidate = os.path.join(
            cfg.data_prep.input_dir, cfg.data_prep.video_name
        )
        candidate_seq = os.path.join(
            cfg.data_prep.input_dir, "sequences", cfg.data_prep.video_name
        )
        if os.path.isdir(candidate):
            self.input_dir = candidate
        elif os.path.isdir(candidate_seq):
            self.input_dir = candidate_seq
        else:
            self.input_dir = candidate  # fallback, will error at frame loading
        self.output_dir = os.path.join(
            self.input_dir, cfg.data_prep.output_subdir
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> Dict[str, np.ndarray]:
        """Execute the full Step 1 pipeline.

        Auto-detects BEHAVE sequences (by checking for t*.000/ subdirs)
        and uses GT masks + SMPL params directly, skipping perception models.
        """
        # Check if this is a BEHAVE sequence
        import glob
        behave_timesteps = glob.glob(os.path.join(self.input_dir, "t*.000"))
        if behave_timesteps:
            print("[Step1] Detected BEHAVE sequence — using GT data directly.")
            return self._run_behave()
        else:
            return self._run_wild()
    def _run_behave(self) -> Dict[str, np.ndarray]:
        """BEHAVE path: read GT masks, SMPL, joints; only run UniDepth + masking."""
        from pipeline.behave_gt_loader import load_behave_sequence

        t0 = time.time()

        gt = load_behave_sequence(
            self.input_dir,
            max_frames=self.cfg.data_prep.max_frames,
        )
        frames = gt["frames"]
        masks_human = gt["masks_human"]
        masks_object = gt["masks_object"]
        smpl_params_list = gt["smpl_params"]
        keypoints_3d = gt["keypoints_3d"]
        T = len(frames)
        H, W = frames[0].shape[:2]
        print(f"[Step1-BEHAVE] {T} frames, resolution {W}x{H}")

        # Save intermediate masks as PNGs
        save_masks_png(masks_human, os.path.join(self.output_dir, "masks_human"), "human")
        save_masks_png(masks_object, os.path.join(self.output_dir, "masks_object"), "object")

        # Export frames as flat PNGs for Step 2 (ProPainter expects frames/ dir)
        import cv2
        frames_dir = os.path.join(self.input_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), f)
        print(f"[Step1-BEHAVE] Exported {T} frames to {frames_dir}")

        # --- UniDepth V2 metric depth ---
        print("=" * 60)
        print("[Step1] Running UniDepth V2 depth estimation...")
        unidepth = UniDepthEstimator(
            model_dir=self.cfg.unidepth.model_dir,
            backbone=self.cfg.unidepth.backbone,
            device=self.device,
        )
        depths_pred = unidepth.predict_sequence(frames)
        del unidepth
        self._cleanup_gpu()

        # Build keypoints_2d with confidence=1.0 from 3D joints (placeholder projection)
        # Step 4 uses these for L_j2d; actual projection happens there via camera intrinsics
        kp2d_list = []
        for i in range(T):
            j3d = keypoints_3d[i]  # (J, 3)
            J = j3d.shape[0]
            conf = np.ones((J, 1), dtype=np.float32)
            # Store 3D joints; 2D projection will be done in Step 4 with camera params
            kp2d_list.append(np.concatenate([j3d[:, :2], conf], axis=1))
        keypoints_2d = np.stack(kp2d_list, axis=0)  # (T, J, 3)

        # --- Depth scale alignment ---
        print("=" * 60)
        print("[Step1] Aligning depth scale...")
        depths_aligned, smpl_aligned = align_sequence(
            depths_pred, smpl_params_list, masks_human, masks_object
        )

        # --- Multi-region masking ---
        print("=" * 60)
        print("[Step1] Computing multi-region masks...")
        mcfg = self.cfg.masking
        region_masks = compute_multi_region_sequence(
            masks_human, masks_object, smpl_aligned,
            dilate_ksize=mcfg.dilate_kernel_size,
            dilate_iters=mcfg.dilate_iterations,
            contact_radius=mcfg.contact_radius,
            blur_ksize=mcfg.gaussian_blur_ksize,
            blur_sigma=mcfg.gaussian_blur_sigma,
        )

        # --- Save outputs ---
        print("=" * 60)
        print("[Step1] Saving outputs...")
        self._save_outputs(
            depths_aligned, smpl_aligned, region_masks,
            keypoints_2d, masks_human, masks_object,
        )
        # Also save 3D joints for Step 4
        save_npz(os.path.join(self.output_dir, "joints_3d.npz"),
                 joints_3d=keypoints_3d)

        elapsed = time.time() - t0
        print(f"[Step1] Done in {elapsed:.1f}s. Outputs: {self.output_dir}")

        return {
            "depths_aligned": depths_aligned,
            "smpl_params": smpl_aligned,
            "M_p": region_masks["M_p"],
            "M_s": region_masks["M_s"],
            "M_object": region_masks["M_object"],
            "keypoints_2d": keypoints_2d,
        }

    def _run_wild(self) -> Dict[str, np.ndarray]:
        """Wild video path: run full perception pipeline."""

        # --- 1. Load frames ---
        print("=" * 60)
        print("[Step1] Loading frames...")
        frames, paths = load_frames(
            self.input_dir, max_frames=self.cfg.data_prep.max_frames
        )
        T = len(frames)
        H, W = frames[0].shape[:2]
        print(f"[Step1] {T} frames, resolution {W}x{H}")

        # --- 2. SAM3 segmentation + tracking ---
        print("=" * 60)
        print("[Step1] Running SAM3 segmentation & tracking...")
        sam3 = SAM3Segmenter(
            model_dir=self.cfg.sam3.model_dir,
            device=self.device,
        )
        tracked = sam3.track_video(frames, self.cfg.sam3.text_prompts)
        masks_human = tracked.get("human", tracked.get(
            self.cfg.sam3.text_prompts[0], [np.zeros((H, W), np.uint8)] * T
        ))
        masks_object = tracked.get("object", tracked.get(
            self.cfg.sam3.text_prompts[1], [np.zeros((H, W), np.uint8)] * T
        ))

        # Save intermediate masks as PNGs
        save_masks_png(masks_human, os.path.join(self.output_dir, "masks_human"), "human")
        save_masks_png(masks_object, os.path.join(self.output_dir, "masks_object"), "object")

        # Free SAM3 GPU memory
        del sam3
        self._cleanup_gpu()

        # --- 3. UniDepth V2 metric depth ---
        print("=" * 60)
        print("[Step1] Running UniDepth V2 depth estimation...")
        unidepth = UniDepthEstimator(
            model_dir=self.cfg.unidepth.model_dir,
            backbone=self.cfg.unidepth.backbone,
            device=self.device,
        )
        depths_pred = unidepth.predict_sequence(frames)  # (T, H, W)
        del unidepth
        self._cleanup_gpu()

        # --- 4. SAM3D-Body SMPL-H estimation ---
        print("=" * 60)
        print("[Step1] Running SAM3D-Body mesh recovery...")
        sam3d = SAM3DBodyEstimator(
            checkpoint=self.cfg.sam3d.checkpoint,
            mhr_model=self.cfg.sam3d.mhr_model,
            config_yaml=self.cfg.sam3d.config_yaml,
            device=self.device,
        )
        smpl_params_list = sam3d.predict_sequence(frames)  # list of dicts
        del sam3d
        self._cleanup_gpu()

        # Extract 2D keypoints from SAM3D-Body output (replaces OpenPose)
        keypoints_2d_list = []
        for sp in smpl_params_list:
            kp2d = sp.get("keypoints_2d", np.zeros((22, 2)))  # (J, 2)
            J = kp2d.shape[0]
            # Add confidence=1.0 column to match (J, 3) format
            conf = np.ones((J, 1), dtype=np.float32)
            keypoints_2d_list.append(np.concatenate([kp2d, conf], axis=1))
        keypoints_2d = np.stack(keypoints_2d_list, axis=0)  # (T, J, 3)
        print(f"[Step1] Extracted 2D keypoints from SAM3D-Body: {keypoints_2d.shape}")

        # --- 5. Depth scale alignment ---
        print("=" * 60)
        print("[Step1] Aligning depth scale...")
        depths_aligned, smpl_aligned = align_sequence(
            depths_pred, smpl_params_list, masks_human, masks_object
        )

        # --- 6. Multi-region masking ---
        print("=" * 60)
        print("[Step1] Computing multi-region masks...")
        mcfg = self.cfg.masking
        region_masks = compute_multi_region_sequence(
            masks_human, masks_object, smpl_aligned,
            dilate_ksize=mcfg.dilate_kernel_size,
            dilate_iters=mcfg.dilate_iterations,
            contact_radius=mcfg.contact_radius,
            blur_ksize=mcfg.gaussian_blur_ksize,
            blur_sigma=mcfg.gaussian_blur_sigma,
        )

        # --- 7. Save outputs ---
        print("=" * 60)
        print("[Step1] Saving outputs...")
        self._save_outputs(
            depths_aligned, smpl_aligned, region_masks,
            keypoints_2d, masks_human, masks_object,
        )

        elapsed = time.time() - t0
        print(f"[Step1] Done in {elapsed:.1f}s. Outputs: {self.output_dir}")

        return {
            "depths_aligned": depths_aligned,
            "smpl_params": smpl_aligned,
            "M_p": region_masks["M_p"],
            "M_s": region_masks["M_s"],
            "M_object": region_masks["M_object"],
            "keypoints_2d": keypoints_2d,
        }

    def _save_outputs(
        self,
        depths_aligned: np.ndarray,
        smpl_aligned: list,
        region_masks: dict,
        keypoints_2d: np.ndarray,
        masks_human: list,
        masks_object: list,
    ):
        """Serialize all outputs to .npz files."""
        out = self.output_dir

        # Depth
        save_npz(os.path.join(out, "depth_aligned.npz"),
                 depth=depths_aligned)

        # SMPL-H parameters (per-frame dicts -> stacked arrays)
        smpl_stacked = {}
        keys = smpl_aligned[0].keys()
        for k in keys:
            smpl_stacked[k] = np.stack([p[k] for p in smpl_aligned], axis=0)
        save_npz(os.path.join(out, "smpl_params.npz"), **smpl_stacked)

        # Multi-region masks
        save_npz(os.path.join(out, "region_masks.npz"),
                 M_p=region_masks["M_p"],
                 M_s=region_masks["M_s"],
                 M_object=region_masks["M_object"])

        # OpenPose keypoints
        save_npz(os.path.join(out, "keypoints_2d.npz"),
                 keypoints=keypoints_2d)

        # Raw binary masks (as uint8 arrays)
        save_npz(os.path.join(out, "masks_raw.npz"),
                 human=np.stack(masks_human, axis=0),
                 object=np.stack(masks_object, axis=0))

    @staticmethod
    def _cleanup_gpu():
        """Free GPU memory between model stages."""
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
