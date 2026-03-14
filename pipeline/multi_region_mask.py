"""
Multi-region contact-aware masking.

Computes:
  M_boundary = dilate(M_human) ∩ dilate(M_object)
  M_contact  = projected SMPL-H contact joints
  M_hull     = ConvexHull(M_boundary ∪ M_contact)
  M_p        = M_human ∩ M_hull          (primary occlusion)
  M_s        = M_human \ M_p             (secondary occlusion)

Then applies Gaussian blur for soft edges.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


# SMPL-H joint indices commonly involved in hand/object contact
# (wrists, fingers, elbows — indices in the 22-joint body skeleton)
CONTACT_JOINT_INDICES = [20, 21]  # left_wrist=20, right_wrist=21


def project_joints_to_2d(
    keypoints_3d: np.ndarray,
    focal_length: float,
    img_w: int,
    img_h: int,
    cam_t: np.ndarray,
) -> np.ndarray:
    """
    Simple perspective projection of 3D joints to 2D pixel coords.

    Parameters
    ----------
    keypoints_3d : (J, 3)
    focal_length : scalar
    cam_t : (3,) or (1, 3)

    Returns (J, 2) pixel coordinates.
    """
    cam_t = cam_t.flatten()
    pts = keypoints_3d + cam_t[None, :]  # translate
    z = pts[:, 2:3].clip(min=1e-4)
    x_proj = pts[:, 0:1] * focal_length / z + img_w / 2
    y_proj = pts[:, 1:2] * focal_length / z + img_h / 2
    return np.concatenate([x_proj, y_proj], axis=1)  # (J, 2)


def make_contact_mask(
    keypoints_2d: np.ndarray,
    img_h: int,
    img_w: int,
    joint_indices: List[int] = None,
    radius: int = 10,
) -> np.ndarray:
    """
    Create a binary mask around projected contact joint locations.

    Returns (H, W) uint8 {0, 255}.
    """
    if joint_indices is None:
        joint_indices = CONTACT_JOINT_INDICES
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for idx in joint_indices:
        if idx < len(keypoints_2d):
            x, y = int(keypoints_2d[idx, 0]), int(keypoints_2d[idx, 1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(mask, (x, y), radius, 255, -1)
    return mask


def compute_multi_region_masks(
    mask_human: np.ndarray,
    mask_object: np.ndarray,
    smpl_params: Dict[str, np.ndarray],
    img_h: int,
    img_w: int,
    dilate_ksize: int = 15,
    dilate_iters: int = 2,
    contact_radius: int = 10,
    blur_ksize: int = 11,
    blur_sigma: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Compute multi-region masks for a single frame.

    Returns dict with keys: M_p, M_s, M_object, M_boundary, M_hull, M_contact
    All are (H, W) float32 in [0, 1] (soft-edged).
    """
    # Ensure binary
    mh = (mask_human > 127).astype(np.uint8)
    mo = (mask_object > 127).astype(np.uint8)

    # 1. Interaction boundary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)
    )
    mh_dilated = cv2.dilate(mh, kernel, iterations=dilate_iters)
    mo_dilated = cv2.dilate(mo, kernel, iterations=dilate_iters)
    m_boundary = (mh_dilated & mo_dilated).astype(np.uint8)

    # 2. Contact mask from projected SMPL-H joints
    kp2d = smpl_params.get("keypoints_2d", np.zeros((22, 2)))
    # If we have 3D keypoints + camera, project them
    kp3d = smpl_params.get("keypoints_3d")
    cam_t = smpl_params.get("cam_t")
    fl = smpl_params.get("focal_length")
    if kp3d is not None and cam_t is not None and fl is not None:
        fl_val = float(fl.flatten()[0]) if isinstance(fl, np.ndarray) else float(fl)
        kp2d = project_joints_to_2d(kp3d, fl_val, img_w, img_h, cam_t)

    m_contact = make_contact_mask(kp2d, img_h, img_w, radius=contact_radius)

    # 3. Convex hull of boundary ∪ contact
    combined = np.maximum(m_boundary * 255, m_contact)
    points = np.column_stack(np.where(combined > 0))  # (N, 2) as (row, col)

    m_hull = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(points) >= 3:
        # cv2.convexHull expects (N, 1, 2) with (x, y) = (col, row)
        hull_pts = points[:, ::-1].reshape(-1, 1, 2).astype(np.int32)
        hull = cv2.convexHull(hull_pts)
        cv2.fillConvexPoly(m_hull, hull, 1)
    elif len(points) > 0:
        # Too few points — just use the boundary+contact directly
        m_hull = (combined > 0).astype(np.uint8)

    # 4. Primary occlusion: M_p = M_human ∩ M_hull
    m_p = (mh & m_hull).astype(np.uint8) * 255

    # 5. Secondary occlusion: M_s = M_human \ M_p
    m_s = (mh & ~(m_p > 0).astype(np.uint8)).astype(np.uint8) * 255

    # 6. Object mask (already binary, scale to 255)
    m_obj = mo * 255

    # 7. Soft-edge Gaussian blur
    ksize = (blur_ksize, blur_ksize)
    m_p_soft = cv2.GaussianBlur(m_p.astype(np.float32), ksize, blur_sigma) / 255.0
    m_s_soft = cv2.GaussianBlur(m_s.astype(np.float32), ksize, blur_sigma) / 255.0
    m_obj_soft = cv2.GaussianBlur(m_obj.astype(np.float32), ksize, blur_sigma) / 255.0

    return {
        "M_p": m_p_soft.astype(np.float32),
        "M_s": m_s_soft.astype(np.float32),
        "M_object": m_obj_soft.astype(np.float32),
        "M_boundary": (m_boundary * 255).astype(np.uint8),
        "M_hull": (m_hull * 255).astype(np.uint8),
        "M_contact": m_contact,
    }


def compute_multi_region_sequence(
    masks_human: List[np.ndarray],
    masks_object: List[np.ndarray],
    smpl_params_list: List[Dict[str, np.ndarray]],
    dilate_ksize: int = 15,
    dilate_iters: int = 2,
    contact_radius: int = 10,
    blur_ksize: int = 11,
    blur_sigma: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Compute multi-region masks for the full sequence.

    Returns dict with keys M_p, M_s, M_object — each (T, H, W) float32.
    """
    T = len(masks_human)
    H, W = masks_human[0].shape[:2]

    all_mp, all_ms, all_mobj = [], [], []

    for i in range(T):
        result = compute_multi_region_masks(
            masks_human[i], masks_object[i], smpl_params_list[i],
            H, W,
            dilate_ksize=dilate_ksize,
            dilate_iters=dilate_iters,
            contact_radius=contact_radius,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
        )
        all_mp.append(result["M_p"])
        all_ms.append(result["M_s"])
        all_mobj.append(result["M_object"])

    return {
        "M_p": np.stack(all_mp, axis=0),
        "M_s": np.stack(all_ms, axis=0),
        "M_object": np.stack(all_mobj, axis=0),
    }
