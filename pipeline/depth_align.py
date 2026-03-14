"""
Depth scale alignment: align predicted depth to metric scale using
median-based scale+shift fitting within the combined human+object mask.

D_align = s * D_pred + t

Also aligns SAM3D camera translation to be consistent with D_align.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_scale_shift(
    depth_pred: np.ndarray,
    depth_ref: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Solve for scale s and shift t such that:
        D_align = s * D_pred + t
    minimises the error within the mask region.

    Uses least-squares fitting on the masked pixels:
        min_{s,t} || s * D_pred[mask] + t - D_ref[mask] ||^2

    This properly solves for both scale AND shift, unlike a simple
    median ratio which always yields t=0.

    Parameters
    ----------
    depth_pred : (H, W) predicted depth
    depth_ref  : (H, W) reference depth (e.g. from SAM3D body)
    mask       : (H, W) binary mask (>0 = valid region)

    Returns s, t
    """
    valid = mask > 0
    if valid.sum() < 10:
        return 1.0, 0.0

    pred_vals = depth_pred[valid].astype(np.float64)
    ref_vals = depth_ref[valid].astype(np.float64)

    # Filter out invalid depth values
    valid_depth = (pred_vals > 1e-6) & (ref_vals > 1e-6)
    if valid_depth.sum() < 10:
        return 1.0, 0.0

    pred_vals = pred_vals[valid_depth]
    ref_vals = ref_vals[valid_depth]

    # Least-squares: [s, t] = argmin || A @ [s, t]^T - b ||^2
    # where A = [pred_vals, 1], b = ref_vals
    A = np.stack([pred_vals, np.ones_like(pred_vals)], axis=1)
    result = np.linalg.lstsq(A, ref_vals, rcond=None)
    s, t = result[0]

    # Sanity check: scale should be positive
    if s <= 0:
        # Fall back to median ratio
        med_pred = np.median(pred_vals)
        med_ref = np.median(ref_vals)
        if abs(med_pred) < 1e-8:
            return 1.0, 0.0
        s = med_ref / med_pred
        t = 0.0

    return float(s), float(t)


def align_depth(depth_pred: np.ndarray, s: float, t: float) -> np.ndarray:
    """Apply scale+shift: D_align = s * D_pred + t."""
    return (s * depth_pred + t).astype(np.float32)


def align_cam_translation(
    cam_t: np.ndarray,
    s: float,
    t: float,
) -> np.ndarray:
    """
    Align SAM3D camera translation to be consistent with D_align.

    The z-component of cam_t is adjusted:
        cam_t_aligned[..., 2] = s * cam_t[..., 2] + t
    """
    cam_t_aligned = cam_t.copy()
    cam_t_aligned[..., 2] = s * cam_t[..., 2] + t
    return cam_t_aligned


def align_sequence(
    depths_pred: np.ndarray,
    smpl_params_list: List[Dict[str, np.ndarray]],
    masks_human: List[np.ndarray],
    masks_object: List[np.ndarray],
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    Align depth and SMPL-H parameters for the full sequence.

    For each frame:
      1. Compute combined mask = human | object
      2. Use SAM3D body depth (rendered from SMPL-H mesh) as reference.
         If per-pixel body depth is not available, fall back to using the
         camera translation z as a global scale reference.
      3. Solve scale+shift
      4. Apply to depth and cam_t

    Returns
    -------
    depths_aligned : (T, H, W) float32
    smpl_params_aligned : list of dicts with updated cam_t
    """
    T = len(depths_pred)
    depths_aligned = np.zeros_like(depths_pred)
    smpl_aligned = []

    for i in range(T):
        # Combined mask
        combined_mask = np.maximum(masks_human[i], masks_object[i])

        # Reference depth: prefer per-pixel rendered depth from SMPL-H mesh
        # if available (key "depth_body"), otherwise fall back to constant z.
        cam_t = smpl_params_list[i]["cam_t"]
        z_body = float(cam_t[0, 2]) if cam_t.ndim == 2 else float(cam_t[2])

        H, W = depths_pred[i].shape
        depth_ref_body = smpl_params_list[i].get("depth_body", None)
        if depth_ref_body is not None and depth_ref_body.shape == (H, W):
            # Use per-pixel body depth as reference (more accurate)
            depth_ref = depth_ref_body.astype(np.float32)
            # Only use pixels where both mask and body depth are valid
            valid_body = (depth_ref > 0) & (combined_mask > 0)
            if valid_body.sum() >= 10:
                s, t = compute_scale_shift(depths_pred[i], depth_ref, valid_body.astype(np.float32))
            else:
                # Fall back to global z
                depth_ref_const = np.full((H, W), z_body, dtype=np.float32)
                s, t = compute_scale_shift(depths_pred[i], depth_ref_const, combined_mask)
        else:
            # Fallback: use camera z as a global scale reference.
            # This is a coarse approximation — the entire masked region is
            # assumed to be at depth z_body, which is only roughly correct.
            depth_ref = np.full((H, W), z_body, dtype=np.float32)
            s, t = compute_scale_shift(depths_pred[i], depth_ref, combined_mask)

        depths_aligned[i] = align_depth(depths_pred[i], s, t)

        # Align camera translation
        params = {k: v.copy() if isinstance(v, np.ndarray) else v
                  for k, v in smpl_params_list[i].items()}
        params["cam_t"] = align_cam_translation(params["cam_t"], s, t)
        smpl_aligned.append(params)

    return depths_aligned, smpl_aligned
