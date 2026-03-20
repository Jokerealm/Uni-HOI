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

import cv2
import numpy as np

from configs.step1_config import Step1PipelineConfig
from pipeline.io_utils import load_frames, save_masks_png, save_npz
from dataset.video_transforms import preprocess_frame_offline, validate_pixel_keypoints
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
            raise FileNotFoundError(
                f"Input video directory does not exist: {candidate} "
                f"(or {candidate_seq})."
            )
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
            cam_id=self.cfg.data_prep.behave_cam_id,
            max_frames=self.cfg.data_prep.max_frames,
        )
        frames = gt["frames"]
        masks_human = gt["masks_human"]
        masks_object = gt["masks_object"]
        smpl_params_list = gt["smpl_params"]
        keypoints_3d = gt["keypoints_3d"]
        keypoints_2d = gt.get("keypoints_2d")
        T = len(frames)
        H, W = frames[0].shape[:2]
        print(f"[Step1-BEHAVE] {T} frames, resolution {W}x{H}")

        # Save intermediate masks as PNGs
        save_masks_png(masks_human, os.path.join(self.output_dir, "masks_human"), "human")
        save_masks_png(masks_object, os.path.join(self.output_dir, "masks_object"), "object")

        # Export frames as flat PNGs for Step 2 (ProPainter expects frames/ dir)
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

        if keypoints_2d is None:
            raise RuntimeError(
                "[Step1-BEHAVE] Expected GT/OpenPose 2D keypoints from the BEHAVE loader."
            )
        keypoints_2d = validate_pixel_keypoints(
            keypoints_2d,
            image_size_hw=frames[0].shape[:2],
            context="Step1 BEHAVE keypoints_2d",
        )
        print(f"[Step1-BEHAVE] Loaded GT/OpenPose 2D keypoints: {keypoints_2d.shape}")

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
        self._save_cropped_training_data(
            frames=frames,
            depths_aligned=depths_aligned,
            keypoints_2d=keypoints_2d,
            masks_human=masks_human,
            masks_object=masks_object,
            region_masks=region_masks,
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
        t0 = time.time()

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
        expected_prompts = list(self.cfg.sam3.text_prompts)
        missing_prompts = [prompt for prompt in expected_prompts if prompt not in tracked]
        if missing_prompts:
            raise KeyError(
                f"SAM3 tracking result is missing prompts: {missing_prompts}. "
                f"Available keys: {sorted(tracked.keys())}"
            )
        masks_human = tracked[expected_prompts[0]]
        masks_object = tracked[expected_prompts[1]]

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
        for frame_idx, sp in enumerate(smpl_params_list):
            kp2d = np.asarray(
                sp.get("keypoints_2d", np.zeros((22, 2), dtype=np.float32)),
                dtype=np.float32,
            )
            if kp2d.ndim != 2 or kp2d.shape[1] not in (2, 3):
                raise ValueError(
                    f"SAM3D-Body returned malformed keypoints_2d for frame {frame_idx}: {kp2d.shape}"
                )

            if kp2d.shape[1] == 2:
                valid = np.isfinite(kp2d).all(axis=1) & (np.linalg.norm(kp2d, axis=1) > 1e-6)
                conf = valid.astype(np.float32)[:, None]
                kp2d = np.concatenate([kp2d, conf], axis=1)
            else:
                kp2d[:, 2] = np.clip(np.nan_to_num(kp2d[:, 2], nan=0.0), 0.0, 1.0)

            kp2d = validate_pixel_keypoints(
                kp2d,
                image_size_hw=frames[frame_idx].shape[:2],
                context=f"Step1 SAM3D-Body keypoints frame {frame_idx}",
            )
            keypoints_2d_list.append(kp2d)
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
        self._save_cropped_training_data(
            frames=frames,
            depths_aligned=depths_aligned,
            keypoints_2d=keypoints_2d,
            masks_human=masks_human,
            masks_object=masks_object,
            region_masks=region_masks,
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

        # SMPL-H parameters and geometry in a unified archive.
        # Topology is constant across frames, so `faces` is stored once.
        smpl_stacked = {}
        keys = smpl_aligned[0].keys()
        for k in keys:
            values = [p[k] for p in smpl_aligned]
            if k == "faces":
                first = values[0].astype(np.int32)
                for value in values[1:]:
                    if value.shape != first.shape or not np.array_equal(value, first):
                        raise ValueError("SMPL `faces` changed across frames; expected a constant topology.")
                smpl_stacked[k] = first
            else:
                smpl_stacked[k] = np.stack(values, axis=0)
        save_npz(os.path.join(out, "smpl_params.npz"), **smpl_stacked)

        # Multi-region masks
        save_npz(os.path.join(out, "region_masks.npz"),
                 M_p=region_masks["M_p"],
                 M_s=region_masks["M_s"],
                 M_object=region_masks["M_object"])

        # 2D keypoints in pixel coordinates
        save_npz(os.path.join(out, "keypoints_2d.npz"),
                 keypoints=keypoints_2d)

        # Raw binary masks (as uint8 arrays)
        save_npz(os.path.join(out, "masks_raw.npz"),
                 human=np.stack(masks_human, axis=0),
                 object=np.stack(masks_object, axis=0))

    def _save_cropped_training_data(
        self,
        frames: list,
        depths_aligned: np.ndarray,
        keypoints_2d: np.ndarray,
        masks_human: list,
        masks_object: list,
        region_masks: dict,
    ):
        """
        Save CARI4D-style cropped training assets for the frame-based Step 4 model.

        Unlike CARI4D's clip-based video inputs, this project optimizes per-frame
        RGB targets. We therefore serialize spatially preprocessed frame patches and
        their ROI intrinsics so Step 4 can train on small, geometry-consistent crops.
        """
        crop_root = os.path.join(self.output_dir, "cropped")
        rgb_dir = os.path.join(crop_root, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        crop_h, crop_w = [int(v) for v in self.cfg.data_prep.crop_size]
        scale_ratio = int(self.cfg.data_prep.scale_ratio)
        bbox_expand = float(self.cfg.data_prep.bbox_expand)

        cropped_depth = []
        cropped_mh = []
        cropped_mo = []
        cropped_mp = []
        cropped_ms = []
        cropped_mobj = []
        cropped_kp = []
        bbox_xywh = []
        fx_all = []
        fy_all = []
        cx_all = []
        cy_all = []
        orig_hw = []
        down_hw = []

        for i, frame in enumerate(frames):
            kp_frame = keypoints_2d[i] if keypoints_2d is not None else None
            cropped = preprocess_frame_offline(
                frame=frame,
                mask_human=masks_human[i],
                mask_object=masks_object[i],
                depth=depths_aligned[i],
                keypoints_2d=kp_frame,
                extra_maps={
                    "M_p": region_masks["M_p"][i],
                    "M_s": region_masks["M_s"][i],
                    "M_object": region_masks["M_object"][i],
                },
                scale_ratio=scale_ratio,
                bbox_expand=bbox_expand,
                out_size=(crop_h, crop_w),
            )

            frame_name = f"frame_{i:06d}.png"
            cv2.imwrite(os.path.join(rgb_dir, frame_name), cropped["rgb"])

            cropped_depth.append(cropped["depth"])
            cropped_mh.append(cropped["mask_human"])
            cropped_mo.append(cropped["mask_object"])
            cropped_mp.append(cropped["extra_maps"]["M_p"])
            cropped_ms.append(cropped["extra_maps"]["M_s"])
            cropped_mobj.append(cropped["extra_maps"]["M_object"])
            if "keypoints_2d" in cropped:
                cropped_kp.append(cropped["keypoints_2d"])
            bbox_xywh.append(cropped["bbox_xywh"])
            fx_all.append(cropped["fx"])
            fy_all.append(cropped["fy"])
            cx_all.append(cropped["cx"])
            cy_all.append(cropped["cy"])
            orig_hw.append(cropped["orig_size_hw"])
            down_hw.append(cropped["downsampled_size_hw"])

        save_npz(
            os.path.join(crop_root, "depth_aligned.npz"),
            depth=np.stack(cropped_depth, axis=0).astype(np.float32),
        )
        save_npz(
            os.path.join(crop_root, "region_masks.npz"),
            M_p=np.stack(cropped_mp, axis=0).astype(np.float32),
            M_s=np.stack(cropped_ms, axis=0).astype(np.float32),
            M_object=np.stack(cropped_mobj, axis=0).astype(np.float32),
        )
        save_npz(
            os.path.join(crop_root, "masks_raw.npz"),
            human=np.stack(cropped_mh, axis=0).astype(np.float32),
            object=np.stack(cropped_mo, axis=0).astype(np.float32),
        )
        if cropped_kp:
            save_npz(
                os.path.join(crop_root, "keypoints_2d.npz"),
                keypoints=np.stack(cropped_kp, axis=0).astype(np.float32),
            )
        save_npz(
            os.path.join(crop_root, "meta.npz"),
            bbox_xywh=np.stack(bbox_xywh, axis=0).astype(np.float32),
            fx=np.asarray(fx_all, dtype=np.float32),
            fy=np.asarray(fy_all, dtype=np.float32),
            cx=np.asarray(cx_all, dtype=np.float32),
            cy=np.asarray(cy_all, dtype=np.float32),
            orig_size_hw=np.stack(orig_hw, axis=0).astype(np.int32),
            downsampled_size_hw=np.stack(down_hw, axis=0).astype(np.int32),
            scale_ratio=np.asarray([scale_ratio], dtype=np.int32),
        )

    @staticmethod
    def _cleanup_gpu():
        """Free GPU memory between model stages."""
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
