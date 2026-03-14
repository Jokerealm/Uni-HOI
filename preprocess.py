#!/usr/bin/env python
"""
Offline Data Preprocessing & High-Precision Prior Extraction
=============================================================
Standalone script that runs all heavy 2D/3D foundation models (SAM3, SAM3D-Body,
UniDepth V2, OpenPose) on raw video frames, performs metric depth alignment and
multi-region mask computation, then serialises every result to disk.

After this script finishes, the training DataLoader only needs to read files.

Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess.py \
        input_dir=./sample_data video_name=test_video

    # Full dataset
    CUDA_VISIBLE_DEVICES=0 python preprocess.py \
        input_dir=/data4/guanz/data/Behave video_name=Date01_Sub01_backpack_back
"""
import os
import sys
import glob
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _setup_output_dirs(out_root: str) -> dict:
    """Create the canonical processed/ sub-directory tree and return paths."""
    dirs = {
        "masks_human":       os.path.join(out_root, "masks", "human"),
        "masks_object":      os.path.join(out_root, "masks", "object"),
        "masks_multi_region": os.path.join(out_root, "masks", "multi_region"),
        "depth":             os.path.join(out_root, "depth"),
        "poses":             os.path.join(out_root, "poses"),
        "keypoints":         os.path.join(out_root, "keypoints"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _load_frames(frames_dir: str, max_frames=None):
    """Load sorted frame paths from a directory."""
    exts = ("*.jpg", "*.png", "*.jpeg")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths = sorted(paths)
    if max_frames is not None:
        paths = paths[:max_frames]
    log.info(f"Found {len(paths)} frames in {frames_dir}")
    return paths


# ===========================================================================
# 1. SAM3 — Text-prompted segmentation & video tracking
# ===========================================================================

def load_sam3(cfg):
    """Load SAM3 model for text-prompted segmentation."""
    from sam3 import build_sam3_model  # project-local wrapper
    model = build_sam3_model(cfg.sam3.model_dir, device=cfg.device)
    log.info("SAM3 model loaded.")
    return model


def run_sam3(model, frames, cfg):
    """
    Run SAM3 on every frame to extract human & object masks.

    Returns
    -------
    masks_human : list[np.ndarray]   – binary uint8 masks (H, W)
    masks_object : list[np.ndarray]
    """
    masks_human, masks_object = [], []
    prompts = cfg.sam3.text_prompts  # ["human", "object"]

    for i, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- per-prompt inference ---
        results = model.predict(
            img_rgb,
            text_prompts=prompts,
            score_threshold=cfg.sam3.score_threshold,
            mask_threshold=cfg.sam3.mask_threshold,
        )
        # results is expected to be a dict: prompt_text -> binary mask (H, W)
        m_hum = (results.get("human", np.zeros(img.shape[:2], dtype=np.uint8)) * 255).astype(np.uint8)
        m_obj = (results.get("object", np.zeros(img.shape[:2], dtype=np.uint8)) * 255).astype(np.uint8)

        masks_human.append(m_hum)
        masks_object.append(m_obj)

        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  SAM3 [{i+1}/{len(frames)}]")

    return masks_human, masks_object


# ===========================================================================
# 2. OpenPose — 2D keypoint detection
# ===========================================================================

def load_openpose(cfg):
    """Load OpenPose model."""
    from openpose import build_openpose_model  # project-local wrapper
    model = build_openpose_model(cfg.openpose.model_dir, device=cfg.device)
    log.info("OpenPose model loaded.")
    return model


def run_openpose(model, frames, cfg):
    """
    Returns
    -------
    keypoints_all : np.ndarray  (N_frames, J, 3)  – x, y, confidence
    """
    kps_list = []
    for i, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        kps = model.detect(img)  # (J, 3)
        kps_list.append(kps)
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  OpenPose [{i+1}/{len(frames)}]")
    return np.stack(kps_list, axis=0)


# ===========================================================================
# 3. UniDepth V2 — Metric monocular depth
# ===========================================================================

def load_unidepth(cfg):
    """Load UniDepth V2 model."""
    from unidepth import build_unidepth_model  # project-local wrapper
    model = build_unidepth_model(cfg.unidepth.model_dir, backbone=cfg.unidepth.backbone, device=cfg.device)
    log.info("UniDepth V2 model loaded.")
    return model


def run_unidepth(model, frames, cfg):
    """
    Returns
    -------
    depths : list[np.ndarray]  – float32 depth maps (H, W) in metric scale
    """
    depths = []
    for i, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = model.predict(img_rgb)  # (H, W) float32 metres
        depths.append(depth.astype(np.float32))
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  UniDepth [{i+1}/{len(frames)}]")
    return depths


# ===========================================================================
# 4. SAM3D Body — SMPL-H parameter estimation
# ===========================================================================

def load_sam3d(cfg):
    """Load SAM3D-Body model for SMPL-H recovery."""
    from sam3d_body import build_sam3d_model  # project-local wrapper
    model = build_sam3d_model(
        checkpoint=cfg.sam3d.checkpoint,
        mhr_model=cfg.sam3d.mhr_model,
        config_yaml=cfg.sam3d.config_yaml,
        device=cfg.device,
    )
    log.info("SAM3D-Body model loaded.")
    return model


def run_sam3d(model, frames, cfg):
    """
    Returns
    -------
    smplh_params : dict  – keys like 'betas', 'body_pose', 'global_orient', 'transl', …
                           each value is np.ndarray with leading dim = N_frames
    """
    all_params = []
    target_size = tuple(cfg.sam3d.image_size)  # e.g. (512, 512)

    for i, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size)
        params = model.predict(img_resized)  # dict of np arrays per frame
        all_params.append(params)
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  SAM3D [{i+1}/{len(frames)}]")

    # Stack per-frame dicts into a single dict of arrays
    smplh_params = {}
    for key in all_params[0]:
        smplh_params[key] = np.stack([p[key] for p in all_params], axis=0)
    return smplh_params


# ===========================================================================
# 5. CPU-side geometry: depth alignment & multi-region masks
# ===========================================================================

def align_depth(depth_pred, mask_union, method="median"):
    """
    Align predicted depth to metric scale inside the mask region.

    D_align = s * D_pred + t

    where s, t are solved via median matching inside mask_union.

    Parameters
    ----------
    depth_pred : np.ndarray (H, W) float32
    mask_union : np.ndarray (H, W) bool
    method     : str  ("median")

    Returns
    -------
    depth_aligned : np.ndarray (H, W) float32
    s, t          : float
    """
    valid = mask_union & (depth_pred > 0)
    if valid.sum() < 10:
        log.warning("Too few valid depth pixels for alignment, returning raw depth.")
        return depth_pred.copy(), 1.0, 0.0

    d_vals = depth_pred[valid]
    if method == "median":
        median_d = np.median(d_vals)
        # Simple scale-shift: normalise so that median maps to 1.0
        s = 1.0 / (median_d + 1e-8)
        t = 0.0
    else:
        raise ValueError(f"Unknown depth alignment method: {method}")

    depth_aligned = (s * depth_pred + t).astype(np.float32)
    return depth_aligned, s, t


def compute_multi_region_masks(
    mask_human,
    mask_object,
    smplh_joints_2d,
    masking_cfg,
):
    """
    Compute multi-region soft masks on CPU.

    Parameters
    ----------
    mask_human   : np.ndarray (H, W) uint8 binary
    mask_object  : np.ndarray (H, W) uint8 binary
    smplh_joints_2d : np.ndarray (J, 2) – projected 2D joint locations (x, y)
    masking_cfg  : OmegaConf node with dilate_*, contact_radius, gaussian_blur_*

    Returns
    -------
    dict with keys: M_contact, M_boundary, M_hull, M_p, M_s,
                    soft_M_p, soft_M_s, soft_M_obj  (all float32 H×W)
    """
    H, W = mask_human.shape[:2]
    hum_bool = mask_human > 127
    obj_bool = mask_object > 127

    # --- Contact mask from projected SMPL-H joints ---
    M_contact = np.zeros((H, W), dtype=np.uint8)
    r = masking_cfg.contact_radius
    for jx, jy in smplh_joints_2d.astype(int):
        cv2.circle(M_contact, (jx, jy), r, 255, -1)

    # --- Interaction boundary (dilated intersection) ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (masking_cfg.dilate_kernel_size, masking_cfg.dilate_kernel_size),
    )
    hum_dilated = cv2.dilate(mask_human, kernel, iterations=masking_cfg.dilate_iterations)
    obj_dilated = cv2.dilate(mask_object, kernel, iterations=masking_cfg.dilate_iterations)
    M_boundary = ((hum_dilated > 127) & (obj_dilated > 127)).astype(np.uint8) * 255

    # --- Convex hull of boundary ∪ contact ---
    union = (M_boundary > 127) | (M_contact > 127)
    pts = np.column_stack(np.where(union))  # (N, 2) row, col
    M_hull = np.zeros((H, W), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts[:, ::-1].astype(np.int32))  # needs (x, y)
        cv2.fillConvexPoly(M_hull, hull, 255)

    hull_bool = M_hull > 127

    # --- Primary / secondary occlusion regions ---
    M_p = (hum_bool & hull_bool).astype(np.uint8) * 255
    M_s = (hum_bool & ~hull_bool).astype(np.uint8) * 255

    # --- Gaussian-blurred soft edges ---
    ksize = masking_cfg.gaussian_blur_ksize
    sigma = masking_cfg.gaussian_blur_sigma
    soft_M_p = cv2.GaussianBlur(M_p.astype(np.float32) / 255.0, (ksize, ksize), sigma)
    soft_M_s = cv2.GaussianBlur(M_s.astype(np.float32) / 255.0, (ksize, ksize), sigma)
    soft_M_obj = cv2.GaussianBlur((obj_bool.astype(np.float32)), (ksize, ksize), sigma)

    return {
        "M_contact": M_contact,
        "M_boundary": M_boundary,
        "M_hull": M_hull,
        "M_p": M_p,
        "M_s": M_s,
        "soft_M_p": soft_M_p.astype(np.float32),
        "soft_M_s": soft_M_s.astype(np.float32),
        "soft_M_obj": soft_M_obj.astype(np.float32),
    }


def project_smplh_joints(smplh_params_frame, K=None):
    """
    Simple pinhole projection of SMPL-H 3D joints → 2D.

    Parameters
    ----------
    smplh_params_frame : dict with at least 'joints_3d' (J, 3)
    K : (3, 3) intrinsic matrix or None (use identity → pixel = metre)

    Returns
    -------
    joints_2d : np.ndarray (J, 2)
    """
    joints_3d = smplh_params_frame["joints_3d"]  # (J, 3)
    if K is None:
        # Fallback: orthographic-ish
        return joints_3d[:, :2]
    # Perspective projection
    j_hom = joints_3d @ K.T  # (J, 3)
    j_2d = j_hom[:, :2] / (j_hom[:, 2:3] + 1e-8)
    return j_2d


# ===========================================================================
# 6. Serialization — persist all results to disk
# ===========================================================================

def save_masks(masks_human, masks_object, frames, dirs):
    """Save binary masks as lossless PNG (one per frame)."""
    for i, fpath in enumerate(frames):
        fname = Path(fpath).stem + ".png"
        cv2.imwrite(os.path.join(dirs["masks_human"], fname), masks_human[i])
        cv2.imwrite(os.path.join(dirs["masks_object"], fname), masks_object[i])
    log.info(f"Saved {len(frames)} human + object masks.")


def save_multi_region_masks(multi_region_list, frames, dirs):
    """
    Save soft multi-region masks as compressed .npz (float16 to save space).
    One .npz per frame containing soft_M_p, soft_M_s, soft_M_obj.
    """
    for i, fpath in enumerate(frames):
        fname = Path(fpath).stem + ".npz"
        mr = multi_region_list[i]
        np.savez_compressed(
            os.path.join(dirs["masks_multi_region"], fname),
            soft_M_p=mr["soft_M_p"].astype(np.float16),
            soft_M_s=mr["soft_M_s"].astype(np.float16),
            soft_M_obj=mr["soft_M_obj"].astype(np.float16),
            # Also store the hard masks for downstream convenience
            M_p=mr["M_p"],
            M_s=mr["M_s"],
        )
    log.info(f"Saved {len(frames)} multi-region mask files.")


def save_depths(depths_aligned, frames, dirs):
    """Save aligned depth maps as .npz (float32)."""
    for i, fpath in enumerate(frames):
        fname = Path(fpath).stem + ".npz"
        np.savez_compressed(
            os.path.join(dirs["depth"], fname),
            depth=depths_aligned[i],
        )
    log.info(f"Saved {len(frames)} depth maps.")


def save_smplh(smplh_params, dirs):
    """Save all-frame SMPL-H parameters as a single .npz."""
    out_path = os.path.join(dirs["poses"], "smplh_aligned.npz")
    np.savez_compressed(out_path, **smplh_params)
    log.info(f"Saved SMPL-H params → {out_path}")


def save_keypoints(keypoints_all, dirs):
    """Save all-frame OpenPose 2D keypoints as a single .npz."""
    out_path = os.path.join(dirs["keypoints"], "openpose_2d.npz")
    np.savez_compressed(out_path, keypoints=keypoints_all)
    log.info(f"Saved keypoints → {out_path}")


# ===========================================================================
# 7. Hydra entry point
# ===========================================================================

@hydra.main(config_path="conf", config_name="preprocess", version_base="1.3")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("Offline Preprocessing Pipeline — START")
    log.info("=" * 60)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # --- Resolve paths ---
    frames_dir = os.path.join(cfg.input_dir, cfg.video_name, "frames")
    out_root = os.path.join(cfg.input_dir, cfg.video_name, cfg.output_subdir)
    dirs = _setup_output_dirs(out_root)

    frames = _load_frames(frames_dir, cfg.max_frames)
    if len(frames) == 0:
        log.error(f"No frames found in {frames_dir}. Aborting.")
        return

    n_frames = len(frames)
    device = cfg.device

    # ------------------------------------------------------------------
    # Stage A: Heavy GPU inference (load one model at a time to save VRAM)
    # ------------------------------------------------------------------

    # A1 — SAM3 segmentation
    log.info("[1/4] Running SAM3 segmentation …")
    sam3_model = load_sam3(cfg)
    masks_human, masks_object = run_sam3(sam3_model, frames, cfg)
    del sam3_model
    torch.cuda.empty_cache()
    save_masks(masks_human, masks_object, frames, dirs)

    # A2 — OpenPose 2D keypoints
    log.info("[2/4] Running OpenPose …")
    openpose_model = load_openpose(cfg)
    keypoints_all = run_openpose(openpose_model, frames, cfg)
    del openpose_model
    torch.cuda.empty_cache()
    save_keypoints(keypoints_all, dirs)

    # A3 — UniDepth V2 metric depth
    log.info("[3/4] Running UniDepth V2 …")
    unidepth_model = load_unidepth(cfg)
    depths_raw = run_unidepth(unidepth_model, frames, cfg)
    del unidepth_model
    torch.cuda.empty_cache()

    # A4 — SAM3D Body SMPL-H
    log.info("[4/4] Running SAM3D-Body …")
    sam3d_model = load_sam3d(cfg)
    smplh_params = run_sam3d(sam3d_model, frames, cfg)
    del sam3d_model
    torch.cuda.empty_cache()
    save_smplh(smplh_params, dirs)

    # ------------------------------------------------------------------
    # Stage B: CPU geometry — depth alignment & multi-region masks
    # ------------------------------------------------------------------
    log.info("Running depth alignment & multi-region mask computation …")

    depths_aligned = []
    multi_region_list = []

    for i in range(n_frames):
        m_hum = masks_human[i]
        m_obj = masks_object[i]
        mask_union = (m_hum > 127) | (m_obj > 127)

        # Depth alignment
        d_aligned, s, t = align_depth(
            depths_raw[i], mask_union, method=cfg.depth_align.method,
        )
        depths_aligned.append(d_aligned)

        # Project SMPL-H joints for this frame
        frame_params = {k: v[i] for k, v in smplh_params.items()}
        joints_2d = project_smplh_joints(frame_params, K=None)

        # Multi-region masks
        mr = compute_multi_region_masks(m_hum, m_obj, joints_2d, cfg.masking)
        multi_region_list.append(mr)

        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  Geometry [{i+1}/{n_frames}] s={s:.4f} t={t:.4f}")

    save_depths(depths_aligned, frames, dirs)
    save_multi_region_masks(multi_region_list, frames, dirs)

    log.info("=" * 60)
    log.info("Offline Preprocessing Pipeline — DONE")
    log.info(f"All outputs saved to: {out_root}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
