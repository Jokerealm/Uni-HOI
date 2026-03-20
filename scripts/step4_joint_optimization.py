#!/usr/bin/env python3
"""
Step 4: Multi-Region Contact-Aware Joint 3DGS Optimization
===========================================================
Jointly optimises human + object 3D Gaussians against the original
unseparated video with physics-aware constraints:

  1. SE(3) coordinate registration (canonical → world)
  2. Multi-region weighted rendering loss (visible / primary-occ / secondary-occ)
  3. Contact loss (hand joints ↔ object Gaussians)
  4. 2D projection loss (SMPL-H joints → 2D vs OpenPose)
  5. Penetration loss (volumetric SMPL SDF)
  6. Temporal smoothness loss (3-frame acceleration)

All tensor shapes are annotated with inline comments.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Reuse existing components from the basic 3DGS module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.joint_3dgs_optimization import (
    GaussianModel,
    SimpleProjectionRenderer,
    ssim,
    photometric_loss,
)
from dataset.video_transforms import validate_pixel_keypoints


# ============================================================
# 1. SE(3) Coordinate Registration Module
# ============================================================

class SE3Transform(nn.Module):
    """
    Learnable SE(3) transform, optionally parameterised per frame.

    When `num_frames == 1`, this behaves like the original global rigid transform.
    When `num_frames > 1`, each frame owns its own axis-angle + translation pair.
    """

    def __init__(
        self,
        init_translation: Tuple[float, float, float] = (0., 0., 2.),
        num_frames: int = 1,
    ):
        super().__init__()
        self.num_frames = max(int(num_frames), 1)

        init_t = torch.tensor(init_translation, dtype=torch.float32)
        if self.num_frames == 1:
            translation = init_t.clone()
            axis_angle = torch.zeros(3, dtype=torch.float32)
        else:
            translation = init_t.unsqueeze(0).repeat(self.num_frames, 1)
            axis_angle = torch.zeros(self.num_frames, 3, dtype=torch.float32)

        self.translation = nn.Parameter(translation)
        self.axis_angle = nn.Parameter(axis_angle)

    @property
    def is_sequence(self) -> bool:
        return self.translation.ndim == 2

    def _resolve_frame_index(self, frame_idx: Optional[int]) -> int:
        if frame_idx is None:
            raise ValueError("Per-frame SE(3) requires an explicit `frame_idx`.")
        if torch.is_tensor(frame_idx):
            if frame_idx.numel() != 1:
                raise ValueError(f"`frame_idx` must be scalar, got shape {tuple(frame_idx.shape)}.")
            frame_idx = int(frame_idx.item())
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"`frame_idx={frame_idx}` is out of range for {self.num_frames} frames.")
        return frame_idx

    def _select_param(self, param: Tensor, frame_idx: Optional[int]) -> Tensor:
        if param.ndim == 1:
            return param
        return param[self._resolve_frame_index(frame_idx)]

    def rotation_matrix(self, frame_idx: Optional[int] = None) -> Tensor:
        """Convert axis-angle to 3x3 rotation matrix via Rodrigues formula."""
        axis_angle = self._select_param(self.axis_angle, frame_idx)
        return _axis_angle_to_matrix(axis_angle)

    def pose_vector(self, frame_idx: Optional[int] = None) -> Tensor:
        axis_angle = self._select_param(self.axis_angle, frame_idx)
        translation = self._select_param(self.translation, frame_idx)
        return torch.cat([axis_angle, translation], dim=-1)

    def pose_sequence(self) -> Tensor:
        if self.axis_angle.ndim == 1:
            return torch.cat(
                [self.axis_angle.unsqueeze(0), self.translation.unsqueeze(0)],
                dim=-1,
            )
        return torch.cat([self.axis_angle, self.translation], dim=-1)

    def forward(self, xyz: Tensor, frame_idx: Optional[int] = None) -> Tensor:
        """
        xyz : (N, 3) points in canonical space
        returns : (N, 3) points in world space for the selected frame
        """
        R = self.rotation_matrix(frame_idx)
        translation = self._select_param(self.translation, frame_idx)
        return xyz @ R.T + translation.unsqueeze(0)


def _axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """
    Rodrigues formula: axis-angle (3,) → rotation matrix (3, 3).
    Differentiable w.r.t. axis_angle.
    """
    theta = axis_angle.norm(p=2).clamp(min=1e-8)      # scalar
    k = axis_angle / theta                              # (3,) unit axis
    K = torch.zeros(3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R  # (3, 3)


# ============================================================
# 2. Multi-Region Rendering Loss
# ============================================================

def multi_region_rendering_loss(
    rendered: Tensor,
    gt_image: Tensor,
    mask_visible: Tensor,
    mask_primary_occ: Tensor,
    mask_secondary_occ: Tensor,
    w_visible: float = 1.0,
    w_primary: float = 0.3,
    w_secondary: float = 0.05,
    lambda_ssim: float = 0.2,
) -> Tensor:
    """
    Weighted photometric loss across three mask regions.

    rendered          : (3, H, W) rendered RGB in [0, 1]
    gt_image          : (3, H, W) ground-truth / inpainted RGB
    mask_visible      : (H, W) soft mask for visible region (object area)
    mask_primary_occ  : (H, W) soft mask for primary occlusion (M_p)
    mask_secondary_occ: (H, W) soft mask for secondary occlusion (M_s)
    returns           : scalar loss
    """
    # Per-pixel L1 difference: (3, H, W)
    diff = (rendered - gt_image).abs()

    # Compose weight map: (H, W)
    weight_map = (
        w_visible * mask_visible
        + w_primary * mask_primary_occ
        + w_secondary * mask_secondary_occ
    )
    # Avoid zero-weight everywhere (add small baseline)
    weight_map = weight_map + 0.01

    # Weighted L1: mean over channels, then weighted spatial mean
    l1_per_pixel = diff.mean(dim=0)                    # (H, W)
    weighted_l1 = (l1_per_pixel * weight_map).sum() / (weight_map.sum() + 1e-8)

    # SSIM component (global, not region-weighted — too expensive per-region)
    ssim_val = ssim(
        rendered.unsqueeze(0), gt_image.unsqueeze(0)
    )  # scalar
    ssim_loss = 1.0 - ssim_val

    return (1.0 - lambda_ssim) * weighted_l1 + lambda_ssim * ssim_loss


# ============================================================
# 3. Contact Loss
# ============================================================

def contact_loss(
    hand_joints_3d: Tensor,
    object_xyz: Tensor,
) -> Tensor:
    """
    Nearest-neighbor distance from hand joints to object Gaussian centres.

    hand_joints_3d : (J, 3) 3D positions of selected hand joints
    object_xyz     : (N_o, 3) object Gaussian centres
    returns        : scalar mean nearest-neighbor distance
    """
    if hand_joints_3d.shape[0] == 0 or object_xyz.shape[0] == 0:
        return torch.tensor(0.0, device=object_xyz.device)

    # (J, 1, 3) - (1, N_o, 3) → (J, N_o)
    dists = torch.cdist(hand_joints_3d.unsqueeze(0),
                        object_xyz.unsqueeze(0)).squeeze(0)  # (J, N_o)
    min_dists = dists.min(dim=1).values                       # (J,)
    return min_dists.mean()


# ============================================================
# 4. 2D Joint Projection Loss
# ============================================================

def projection_2d_loss(
    joints_3d: Tensor,
    keypoints_2d: Tensor,
    confidence: Tensor,
    focal: Optional[float] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: float = 0.0,
    cy: float = 0.0,
    conf_threshold: float = 0.3,
) -> Tensor:
    """
    Project 3D joints to 2D via pinhole model, compare with 2D detections.

    joints_3d    : (J_3d, 3)
    keypoints_2d : (J_2d, 2) in pixel coordinates
    confidence   : (J_2d,)
    """
    OPENPOSE_TO_SMPLH = {
        0: 15, 1: 12, 2: 17, 3: 19, 4: 21,
        5: 16, 6: 18, 7: 20, 8: 0,
        9: 2, 10: 5, 11: 8, 12: 1, 13: 4, 14: 7,
    }

    J_3d = joints_3d.shape[0]
    J_2d = keypoints_2d.shape[0]

    op_indices = []
    smplh_indices = []
    for op_idx, smplh_idx in OPENPOSE_TO_SMPLH.items():
        if op_idx < J_2d and smplh_idx < J_3d:
            op_indices.append(op_idx)
            smplh_indices.append(smplh_idx)

    if len(op_indices) == 0:
        return torch.tensor(0.0, device=joints_3d.device)

    op_indices_t = torch.tensor(op_indices, device=joints_3d.device)
    smplh_indices_t = torch.tensor(smplh_indices, device=joints_3d.device)

    matched_3d = joints_3d[smplh_indices_t]
    matched_2d = keypoints_2d[op_indices_t]
    matched_conf = confidence[op_indices_t]

    valid = matched_conf > conf_threshold
    if valid.sum() == 0:
        return torch.tensor(0.0, device=joints_3d.device)

    valid_2d = matched_2d[valid]
    if valid_2d.shape[0] >= 3 and valid_2d.abs().max() <= 4.0:
        raise ValueError(
            "projection_2d_loss received keypoints_2d that look normalized. "
            "The maintained pipeline requires pixel coordinates consistent with fx/fy/cx/cy."
        )

    if fx is None:
        fx = 500.0 if focal is None else focal
    if fy is None:
        fy = fx
    z = matched_3d[:, 2].clamp(min=0.1)
    proj_x = (matched_3d[:, 0] / z) * fx + cx
    proj_y = (matched_3d[:, 1] / z) * fy + cy
    proj_2d = torch.stack([proj_x, proj_y], dim=-1)

    diff = (proj_2d - matched_2d) ** 2
    diff_sum = diff.sum(dim=-1)
    weighted = (diff_sum * matched_conf * valid.float()).sum()
    return weighted / (valid.float().sum() + 1e-8)


# ============================================================
# 5. Volumetric SMPL Penetration Loss
# ============================================================

class VolumetricSMPLSDF(nn.Module):
    """
    Computes a voxelised SDF from SMPL-H mesh vertices.
    Object Gaussians inside the body (SDF < 0) are penalised.

    This is a simplified grid-based SDF: we voxelise the bounding box of
    the SMPL mesh and compute signed distance via closest-surface lookup.
    For efficiency, we use a coarse grid and trilinear interpolation.
    """

    def __init__(self, resolution: int = 64, padding: float = 0.1):
        super().__init__()
        self.resolution = resolution
        self.padding = padding
        self.distance_chunk_size = 4096
        self.sign_chunk_size = 2048
        self._grid_cache = {}

    def clear_cache(self) -> None:
        self._grid_cache.clear()

    def get_sdf_grid(
        self,
        vertices: Tensor,
        faces: Tensor,
        cache_key: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if cache_key is not None and cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        with torch.no_grad():
            sdf_grid, vmin, vmax = self.compute_sdf_grid(vertices, faces)
            cached = (sdf_grid.detach(), vmin.detach(), vmax.detach())

        if cache_key is not None:
            self._grid_cache[cache_key] = cached
        return cached

    @staticmethod
    def _chunked_nearest_distance(
        query: Tensor,
        ref: Tensor,
        chunk_size: int,
    ) -> Tensor:
        mins = []
        for start in range(0, query.shape[0], chunk_size):
            end = min(start + chunk_size, query.shape[0])
            dists = torch.cdist(query[start:end], ref)
            mins.append(dists.min(dim=1).values)
            del dists
        return torch.cat(mins, dim=0)

    def compute_sdf_grid(
        self, vertices: Tensor, faces: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build a coarse SDF grid from mesh vertices.

        vertices : (V, 3)
        faces    : (F, 3) long

        returns:
            sdf_grid : (1, 1, R, R, R) signed distance values
            grid_min : (3,) world-space minimum corner
            grid_max : (3,) world-space maximum corner
        """
        R = self.resolution
        device = vertices.device

        # Bounding box with padding
        vmin = vertices.min(dim=0).values - self.padding       # (3,)
        vmax = vertices.max(dim=0).values + self.padding       # (3,)

        # Build 3D grid coordinates
        lin = [torch.linspace(vmin[i].item(), vmax[i].item(), R,
                              device=device) for i in range(3)]
        gx, gy, gz = torch.meshgrid(lin[0], lin[1], lin[2], indexing='ij')
        grid_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)

        # Unsigned distance to nearest vertex (approximation — no face query).
        # Chunked to avoid materialising the full (R^3, V) distance matrix.
        unsigned_dist = self._chunked_nearest_distance(
            grid_pts,
            vertices,
            chunk_size=self.distance_chunk_size,
        )

        # Sign estimation: use winding number approximation
        # Simple heuristic — points whose nearest vertex normal points away
        # are considered inside. For robustness we use a voting scheme.
        sign = self._estimate_sign(grid_pts, vertices, faces)  # (R^3,)

        sdf = unsigned_dist * sign                             # (R^3,)
        sdf_grid = sdf.reshape(1, 1, R, R, R)                 # (1, 1, R, R, R)
        return sdf_grid, vmin, vmax

    def _estimate_sign(
        self, query: Tensor, vertices: Tensor, faces: Tensor
    ) -> Tensor:
        """
        Estimate inside/outside sign via nearest-face normal dot product.
        Returns +1 (outside) or -1 (inside) for each query point.

        query    : (Q, 3)
        vertices : (V, 3)
        faces    : (F, 3) long
        """
        # Compute face normals
        v0 = vertices[faces[:, 0]]                             # (F, 3)
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)
        face_normals = F.normalize(face_normals, dim=-1)
        face_centres = (v0 + v1 + v2) / 3.0                   # (F, 3)

        sign_chunks = []
        for start in range(0, query.shape[0], self.sign_chunk_size):
            end = min(start + self.sign_chunk_size, query.shape[0])
            query_chunk = query[start:end]
            dists = torch.cdist(query_chunk, face_centres)     # (chunk, F)
            nearest_face = dists.argmin(dim=1)                 # (chunk,)
            del dists

            # Direction from face centre to query
            direction = query_chunk - face_centres[nearest_face]
            normal = face_normals[nearest_face]

            # Dot product: positive = outside, negative = inside
            dot = (direction * normal).sum(dim=-1)
            sign_chunks.append(
                torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
            )

        return torch.cat(sign_chunks, dim=0)

    def query_sdf(
        self,
        points: Tensor,
        sdf_grid: Tensor,
        grid_min: Tensor,
        grid_max: Tensor,
    ) -> Tensor:
        """
        Trilinear interpolation of SDF values at arbitrary query points.

        points   : (N, 3) query positions
        sdf_grid : (1, 1, R, R, R)
        grid_min : (3,)
        grid_max : (3,)
        returns  : (N,) SDF values (negative = inside body)
        """
        # Normalise to [-1, 1] for grid_sample
        normalised = 2.0 * (points - grid_min) / (grid_max - grid_min + 1e-8) - 1.0
        # grid_sample expects (B, C, D, H, W) input and (B, D_out, H_out, W_out, 3) grid
        grid = normalised.reshape(1, 1, 1, -1, 3)             # (1, 1, 1, N, 3)
        sampled = F.grid_sample(
            sdf_grid, grid, align_corners=True, mode='bilinear', padding_mode='border'
        )  # (1, 1, 1, 1, N)
        return sampled.reshape(-1)                             # (N,)


def penetration_loss(
    object_xyz: Tensor,
    smpl_vertices: Tensor,
    smpl_faces: Tensor,
    sdf_module: VolumetricSMPLSDF,
    cache_key: Optional[int] = None,
) -> Tensor:
    """
    Penalise object Gaussians that penetrate the SMPL body mesh.

    object_xyz     : (N_o, 3) object Gaussian centres
    smpl_vertices  : (V, 3) SMPL-H mesh vertices for current frame
    smpl_faces     : (F, 3) long — SMPL face indices (constant across frames)
    sdf_module     : VolumetricSMPLSDF instance
    returns        : scalar penetration penalty
    """
    sdf_grid, vmin, vmax = sdf_module.get_sdf_grid(
        smpl_vertices,
        smpl_faces,
        cache_key=cache_key,
    )
    sdf_vals = sdf_module.query_sdf(object_xyz, sdf_grid, vmin, vmax)  # (N_o,)

    # Only penalise negative SDF (inside body)
    penetration = F.relu(-sdf_vals)                            # (N_o,)
    return penetration.mean()


# ============================================================
# 6. Temporal Smoothness Loss (Acceleration)
# ============================================================

def temporal_smoothness_loss(
    poses_prev: Tensor,
    poses_curr: Optional[Tensor] = None,
    poses_next: Optional[Tensor] = None,
) -> Tensor:
    """
    Acceleration-based smoothness on an actual frame-ordered pose sequence.

    Accepts either:
      - `poses_prev` as `(T, 6)` pose sequence
      - `poses_prev`, `poses_curr`, `poses_next` as three `(6,)` poses
    """
    if poses_curr is None and poses_next is None:
        poses = poses_prev
        if poses.ndim != 2 or poses.shape[-1] != 6:
            raise ValueError(f"`poses` must have shape [T, 6], got {tuple(poses.shape)}.")
        if poses.shape[0] < 3:
            return poses.new_zeros(())
        acc = poses[:-2] - 2.0 * poses[1:-1] + poses[2:]
        return (acc ** 2).sum(dim=-1).mean()

    if poses_curr is None or poses_next is None:
        raise ValueError("Provide either a single pose sequence or all three adjacent poses.")

    acc = poses_prev - 2.0 * poses_curr + poses_next
    return (acc ** 2).sum()


def temporal_smoothness_loss_live(
    se3_transform: 'SE3Transform',
    pose_prev_detached: Tensor,
    pose_next_detached: Tensor,
) -> Tensor:
    """
    Compute temporal smoothness with gradient flow through the *current*
    SE(3) parameters.  prev and next are detached snapshots.

    se3_transform       : current-frame SE3Transform (carries gradients)
    pose_prev_detached  : (6,) detached pose from frame t-1
    pose_next_detached  : (6,) detached pose from frame t+1
    returns             : scalar acceleration penalty
    """
    pose_curr = se3_transform.pose_vector()
    return temporal_smoothness_loss(pose_prev_detached, pose_curr, pose_next_detached)


# ============================================================
# 7. Joint Renderer (wraps SE(3) + rendering)
# ============================================================

class JointRenderer(nn.Module):
    """
    Applies per-entity SE(3) transforms then renders the union of
    human + object Gaussians.
    """

    def __init__(
        self,
        renderer: SimpleProjectionRenderer,
        se3_human: SE3Transform,
        se3_object: SE3Transform,
    ):
        super().__init__()
        self.renderer = renderer
        self.se3_human = se3_human
        self.se3_object = se3_object

    def forward(
        self,
        human_gs: GaussianModel,
        object_gs: GaussianModel,
        frame_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns:
            rendered : (3, H, W) composited image
            xyz_h_world : (N_h, 3) human centres in world space
            xyz_o_world : (N_o, 3) object centres in world space
        """
        # Transform to world coordinates
        xyz_h_world = self.se3_human(human_gs.get_xyz, frame_idx=frame_idx)
        xyz_o_world = self.se3_object(object_gs.get_xyz, frame_idx=frame_idx)

        # Gather appearance attributes
        col_all = torch.cat([human_gs.get_colors, object_gs.get_colors], 0)
        opa_all = torch.cat([human_gs.get_opacity, object_gs.get_opacity], 0)
        scl_all = torch.cat([human_gs.get_scaling, object_gs.get_scaling], 0)
        xyz_all = torch.cat([xyz_h_world, xyz_o_world], 0)

        rendered = self.renderer(xyz_all, col_all, opa_all, scl_all)
        return rendered, xyz_h_world, xyz_o_world


# ============================================================
# 8. Full Training Step
# ============================================================

def step4_training_step(
    human_gs: GaussianModel,
    object_gs: GaussianModel,
    joint_renderer: JointRenderer,
    gt_image: Tensor,
    mask_visible: Tensor,
    mask_primary_occ: Tensor,
    mask_secondary_occ: Tensor,
    optimizer: torch.optim.Optimizer,
    # Optional per-frame data
    smpl_joints_3d: Optional[Tensor] = None,
    hand_joint_indices: Optional[List[int]] = None,
    keypoints_2d: Optional[Tensor] = None,
    kp_confidence: Optional[Tensor] = None,
    smpl_vertices: Optional[Tensor] = None,
    smpl_faces: Optional[Tensor] = None,
    sdf_module: Optional[VolumetricSMPLSDF] = None,
    frame_indices: Optional[List[int]] = None,
    # Temporal: detached pose snapshots from adjacent frames
    se3_pose_prev_detached: Optional[Tensor] = None,
    se3_pose_next_detached: Optional[Tensor] = None,
    se3_human_module: Optional[SE3Transform] = None,
    se3_object_module: Optional[SE3Transform] = None,
    # Loss weights
    w_visible: float = 1.0,
    w_primary: float = 0.3,
    w_secondary: float = 0.05,
    lambda_ssim: float = 0.2,
    lambda_contact: float = 0.5,
    lambda_j2d: float = 0.1,
    lambda_pen: float = 1.0,
    lambda_acc: float = 0.5,
    focal: float = 500.0,
    fx: Optional[Tensor] = None,
    fy: Optional[Tensor] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    conf_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    One gradient step of the full Step 4 joint optimization.

    Supports batched frames via gradient accumulation: if gt_image is
    (B, 3, H, W), each frame is forward+backward'd separately so peak
    VRAM stays constant regardless of batch size. Gradients are accumulated
    and averaged before optimizer.step().

    Returns dict of scalar loss values for logging.
    """
    optimizer.zero_grad()

    # --- Detect batch vs single ---
    if gt_image.ndim == 4:
        B = gt_image.shape[0]
    else:
        B = 1
        gt_image = gt_image.unsqueeze(0)
        mask_visible = mask_visible.unsqueeze(0)
        mask_primary_occ = mask_primary_occ.unsqueeze(0)
        mask_secondary_occ = mask_secondary_occ.unsqueeze(0)

    H = mask_visible.shape[-2]
    W = mask_visible.shape[-1]
    requires_frame_indices = (
        getattr(joint_renderer.se3_human, "num_frames", 1) > 1
        or getattr(joint_renderer.se3_object, "num_frames", 1) > 1
    )
    if frame_indices is None:
        if requires_frame_indices:
            raise ValueError("`frame_indices` is required when using per-frame SE(3) trajectories.")
        frame_indices = [0] * B
    elif len(frame_indices) != B:
        raise ValueError(f"`frame_indices` must have length {B}, got {len(frame_indices)}.")

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    def _select_scalar(value, index: int, default: float) -> float:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.ndim == 0:
                return float(value.item())
            return float(value[index].item())
        if isinstance(value, (list, tuple)):
            return float(value[index])
        return float(value)

    sum_render = 0.0
    sum_contact = 0.0
    sum_j2d = 0.0
    sum_pen = 0.0

    for b in range(B):
        frame_idx_b = int(frame_indices[b])
        fx_b = _select_scalar(fx, b, focal)
        fy_b = _select_scalar(fy, b, fx_b)
        cx_b = _select_scalar(cx, b, W / 2.0)
        cy_b = _select_scalar(cy, b, H / 2.0)
        if hasattr(joint_renderer.renderer, "set_camera"):
            joint_renderer.renderer.set_camera(fx=fx_b, fy=fy_b, cx=cx_b, cy=cy_b)

        # ---- Forward render ----
        rendered, xyz_h_world, xyz_o_world = joint_renderer(
            human_gs,
            object_gs,
            frame_idx=frame_idx_b,
        )

        # ---- Loss 1: Multi-region rendering loss ----
        loss_render = multi_region_rendering_loss(
            rendered, gt_image[b],
            mask_visible[b], mask_primary_occ[b], mask_secondary_occ[b],
            w_visible=w_visible, w_primary=w_primary, w_secondary=w_secondary,
            lambda_ssim=lambda_ssim,
        )

        loss_contact_b = torch.tensor(0.0, device=gt_image.device)
        loss_j2d_b = torch.tensor(0.0, device=gt_image.device)
        loss_pen_b = torch.tensor(0.0, device=gt_image.device)

        # ---- Loss 2: Contact loss ----
        if smpl_joints_3d is not None and hand_joint_indices is not None:
            joints = smpl_joints_3d[b] if smpl_joints_3d.ndim == 3 else smpl_joints_3d
            hand_joints = joints[hand_joint_indices]
            loss_contact_b = contact_loss(hand_joints, xyz_o_world)

        # ---- Loss 3: 2D projection loss ----
        if (smpl_joints_3d is not None and keypoints_2d is not None
                and kp_confidence is not None):
            joints = smpl_joints_3d[b] if smpl_joints_3d.ndim == 3 else smpl_joints_3d
            kp = keypoints_2d[b] if keypoints_2d.ndim == 3 else keypoints_2d
            kpc = kp_confidence[b] if kp_confidence.ndim == 2 else kp_confidence
            loss_j2d_b = projection_2d_loss(
                joints, kp, kpc,
                focal=focal, fx=fx_b, fy=fy_b, cx=cx_b, cy=cy_b,
                conf_threshold=conf_threshold,
            )

        # ---- Loss 4: Penetration loss ----
        if (smpl_vertices is not None and smpl_faces is not None
                and sdf_module is not None):
            verts = smpl_vertices[b] if smpl_vertices.ndim == 3 else smpl_vertices
            loss_pen_b = penetration_loss(
                xyz_o_world, verts, smpl_faces, sdf_module,
                cache_key=frame_idx_b,
            )

        # Per-frame loss (scaled by 1/B for gradient accumulation)
        frame_loss = (
            loss_render
            + lambda_contact * loss_contact_b
            + lambda_j2d * loss_j2d_b
            + lambda_pen * loss_pen_b
        ) / B

        # Backward per frame — accumulates gradients, frees render graph
        frame_loss.backward()

        # Track scalars (unscaled)
        sum_render += loss_render.item()
        sum_contact += loss_contact_b.item()
        sum_j2d += loss_j2d_b.item()
        sum_pen += loss_pen_b.item()

    # ---- Loss 5: Temporal smoothness (shared, not per-frame) ----
    loss_acc = torch.tensor(0.0, device=gt_image.device)
    temporal_terms = []
    if se3_human_module is not None and getattr(se3_human_module, "num_frames", 1) >= 3:
        temporal_terms.append(temporal_smoothness_loss(se3_human_module.pose_sequence()))
    if se3_object_module is not None and getattr(se3_object_module, "num_frames", 1) >= 3:
        temporal_terms.append(temporal_smoothness_loss(se3_object_module.pose_sequence()))
    if temporal_terms:
        loss_acc = torch.stack(temporal_terms).mean()
        (lambda_acc * loss_acc).backward()
    elif (
        se3_pose_prev_detached is not None
        and se3_pose_next_detached is not None
        and se3_object_module is not None
    ):
        loss_acc = temporal_smoothness_loss_live(
            se3_object_module,
            se3_pose_prev_detached,
            se3_pose_next_detached,
        )
        (lambda_acc * loss_acc).backward()

    optimizer.step()

    avg_render = sum_render / B
    avg_contact = sum_contact / B
    avg_j2d = sum_j2d / B
    avg_pen = sum_pen / B
    total = avg_render + lambda_contact * avg_contact + lambda_j2d * avg_j2d + lambda_pen * avg_pen + lambda_acc * loss_acc.item()

    return {
        "loss_total": total,
        "loss_render": avg_render,
        "loss_contact": avg_contact,
        "loss_j2d": avg_j2d,
        "loss_penetration": avg_pen,
        "loss_temporal": loss_acc.item(),
    }


# ============================================================
# 9. Data Loading Utilities for Step 4
# ============================================================

def load_step1_outputs(processed_dir: str, device: torch.device) -> dict:
    """
    Load Step 1 outputs: soft masks, SMPL-H params, depth, keypoints.

    Supports two formats:
      1. Unified Step-1 archives:
         - region_masks.npz  (M_p, M_s, M_object)
         - masks_raw.npz     (human, object)
         - smpl_params.npz   (body_pose, shape, cam_t, optional vertices/faces/joints_3d)
         - keypoints_2d.npz  (keypoints)
      2. Legacy per-file directories:
         - masks_primary_occ/, masks_secondary_occ/, masks_object/
         - smplh_params/
         - keypoints_2d/
         - camera_intrinsics.npy
    """
    import glob
    import numpy as np

    data = {
        'masks_visible': [],
        'masks_primary_occ': [],
        'masks_secondary_occ': [],
        'smplh_params': [],
        'keypoints_2d': [],
        'kp_confidence': [],
        'camera_K': None,
    }

    # Unified Step-1 outputs
    region_masks_path = os.path.join(processed_dir, 'region_masks.npz')
    raw_masks_path = os.path.join(processed_dir, 'masks_raw.npz')
    smpl_params_path = os.path.join(processed_dir, 'smpl_params.npz')
    keypoints_path = os.path.join(processed_dir, 'keypoints_2d.npz')

    if os.path.isfile(region_masks_path):
        region_masks = np.load(region_masks_path)
        if 'M_object' in region_masks:
            data['masks_visible'] = [
                torch.from_numpy(mask).float().to(device)
                for mask in region_masks['M_object']
            ]
        if 'M_p' in region_masks:
            data['masks_primary_occ'] = [
                torch.from_numpy(mask).float().to(device)
                for mask in region_masks['M_p']
            ]
        if 'M_s' in region_masks:
            data['masks_secondary_occ'] = [
                torch.from_numpy(mask).float().to(device)
                for mask in region_masks['M_s']
            ]

    if not data['masks_visible'] and os.path.isfile(raw_masks_path):
        raw_masks = np.load(raw_masks_path)
        if 'object' in raw_masks:
            data['masks_visible'] = [
                torch.from_numpy(mask).float().to(device)
                for mask in raw_masks['object']
            ]

    if os.path.isfile(smpl_params_path):
        smpl_npz = np.load(smpl_params_path)
        time_dim = None
        for key in ('cam_t', 'body_pose', 'vertices', 'joints_3d'):
            if key in smpl_npz:
                time_dim = int(smpl_npz[key].shape[0])
                break
        if time_dim is None:
            raise ValueError(f'Could not infer temporal dimension from {smpl_params_path}')

        for frame_idx in range(time_dim):
            param_dict = {}
            for key in smpl_npz.files:
                value = smpl_npz[key]
                if key == 'faces':
                    tensor = torch.from_numpy(value).to(device)
                    param_dict[key] = tensor.long()
                elif value.ndim > 0 and value.shape[0] == time_dim:
                    param_dict[key] = torch.from_numpy(value[frame_idx]).float().to(device)
                else:
                    param_dict[key] = torch.from_numpy(value).float().to(device)
            data['smplh_params'].append(param_dict)

    if os.path.isfile(keypoints_path):
        keypoints_npz = np.load(keypoints_path)
        if 'keypoints' in keypoints_npz:
            for kp in keypoints_npz['keypoints']:
                kp_tensor = torch.from_numpy(
                    validate_pixel_keypoints(
                        kp,
                        context=f"{keypoints_path} keypoints",
                    )
                ).float().to(device)
                data['keypoints_2d'].append(kp_tensor[:, :2])
                data['kp_confidence'].append(kp_tensor[:, 2] if kp_tensor.shape[-1] > 2 else torch.ones(kp_tensor.shape[0], device=device))

    # Camera intrinsics
    K_path = os.path.join(processed_dir, 'camera_intrinsics.npy')
    if os.path.isfile(K_path):
        data['camera_K'] = torch.from_numpy(np.load(K_path)).float().to(device)

    # Soft masks
    for mask_name, key in [
        ('masks_object', 'masks_visible'),
        ('masks_primary_occ', 'masks_primary_occ'),
        ('masks_secondary_occ', 'masks_secondary_occ'),
    ]:
        mask_dir = os.path.join(processed_dir, mask_name)
        if data[key]:
            continue
        if os.path.isdir(mask_dir):
            paths = sorted(glob.glob(os.path.join(mask_dir, '*.npy')))
            for p in paths:
                m = torch.from_numpy(np.load(p)).float().to(device)
                data[key].append(m)

    # Legacy SMPL-H parameters
    smplh_dir = os.path.join(processed_dir, 'smplh_params')
    if not data['smplh_params'] and os.path.isdir(smplh_dir):
        paths = sorted(glob.glob(os.path.join(smplh_dir, '*.npz')))
        for p in paths:
            params = dict(np.load(p, allow_pickle=True))
            # Convert to tensors
            param_dict = {}
            for k, v in params.items():
                param_dict[k] = torch.from_numpy(v).float().to(device)
            data['smplh_params'].append(param_dict)

    # Legacy 2D keypoints (OpenPose format: (J, 3) where last col is confidence)
    kp_dir = os.path.join(processed_dir, 'keypoints_2d')
    if not data['keypoints_2d'] and os.path.isdir(kp_dir):
        paths = sorted(glob.glob(os.path.join(kp_dir, '*.npy')))
        for p in paths:
            kp = validate_pixel_keypoints(
                np.load(p),
                context=f"{p} keypoints",
            )  # (J, 3) — x, y, confidence
            kp_tensor = torch.from_numpy(kp).float().to(device)
            data['keypoints_2d'].append(kp_tensor[:, :2])      # (J, 2)
            data['kp_confidence'].append(kp_tensor[:, 2])       # (J,)

    return data


def load_gs_init(gs_init_dir: str, device: torch.device) -> Tuple[GaussianModel, GaussianModel]:
    """
    Load Step 3 initial 3DGS parameters.

    Supports formats (checked in priority order):
      0. Metric-aligned .npz from Step 3.5 (gs_aligned/)
      1. GaussianModel state_dict: human_gs.pt / object_gs.pt
      2. Step 3 pipeline output: G_h.pt / G_o.pt (with 'raw' key containing
         the 14-channel tensor, or activated individual fields)
      3. Combined file: gs_init_combined.pt
    """
    # Format 0: Metric-aligned .npz from Step 3.5 Alignment Bridge
    # These live in a sibling directory gs_aligned/ next to gs_init/
    aligned_base = gs_init_dir.rstrip('/').rstrip('\\')
    candidate_aligned_dirs = []
    if aligned_base.endswith('gs_init'):
        parent = os.path.dirname(aligned_base)
        candidate_aligned_dirs.append(os.path.join(parent, 'gs_aligned'))
        candidate_aligned_dirs.append(aligned_base + '_aligned')
    else:
        candidate_aligned_dirs.append(aligned_base + '_aligned')
        candidate_aligned_dirs.append(os.path.join(os.path.dirname(aligned_base), 'gs_aligned'))

    for aligned_dir in candidate_aligned_dirs:
        hum_aligned = os.path.join(aligned_dir, 'human_gaussians_metric.npz')
        obj_aligned = os.path.join(aligned_dir, 'object_gaussians_metric.npz')
        if os.path.isfile(hum_aligned) and os.path.isfile(obj_aligned):
            import numpy as _np
            h_raw = torch.from_numpy(_np.load(hum_aligned)['raw']).float()
            o_raw = torch.from_numpy(_np.load(obj_aligned)['raw']).float()
            human_gs = GaussianModel.from_phase2(h_raw)
            object_gs = GaussianModel.from_phase2(o_raw)
            print(f'[Step4] Loaded metric-aligned 3DGS from {aligned_dir}')
            return human_gs.to(device), object_gs.to(device)

    # Format 1: GaussianModel state_dicts
    human_path = os.path.join(gs_init_dir, 'human_gs.pt')
    object_path = os.path.join(gs_init_dir, 'object_gs.pt')

    if os.path.isfile(human_path) and os.path.isfile(object_path):
        human_sd = torch.load(human_path, map_location='cpu', weights_only=False)
        object_sd = torch.load(object_path, map_location='cpu', weights_only=False)
        # Detect format: if 'raw' key exists, it's Step 3 output (activated values)
        if 'raw' in human_sd:
            human_gs = GaussianModel.from_phase2(human_sd['raw'])
            object_gs = GaussianModel.from_phase2(object_sd['raw'])
        else:
            n_h = human_sd['xyz'].shape[0]
            n_o = object_sd['xyz'].shape[0]
            human_gs = GaussianModel(num_points=n_h)
            object_gs = GaussianModel(num_points=n_o)
            human_gs.load_state_dict(human_sd)
            object_gs.load_state_dict(object_sd)
        return human_gs.to(device), object_gs.to(device)

    # Format 2: Step 3 pipeline output (G_h.pt / G_o.pt)
    gh_path = os.path.join(gs_init_dir, 'G_h.pt')
    go_path = os.path.join(gs_init_dir, 'G_o.pt')
    if os.path.isfile(gh_path) and os.path.isfile(go_path):
        gh_data = torch.load(gh_path, map_location='cpu', weights_only=False)
        go_data = torch.load(go_path, map_location='cpu', weights_only=False)
        # Use 'raw' tensor (14-channel) for proper initialization
        human_gs = GaussianModel.from_phase2(gh_data['raw'])
        object_gs = GaussianModel.from_phase2(go_data['raw'])
        return human_gs.to(device), object_gs.to(device)

    # Format 3: Combined file from Step 3
    combined_path = os.path.join(gs_init_dir, 'gs_init_combined.pt')
    if os.path.isfile(combined_path):
        ckpt = torch.load(combined_path, map_location='cpu', weights_only=False)
        if 'G_h' in ckpt and 'G_o' in ckpt:
            human_gs = GaussianModel.from_phase2(ckpt['G_h']['raw'])
            object_gs = GaussianModel.from_phase2(ckpt['G_o']['raw'])
            return human_gs.to(device), object_gs.to(device)

    # Format 4: Legacy combined file
    combined_path2 = os.path.join(gs_init_dir, 'final_3dgs.pt')
    if os.path.isfile(combined_path2):
        ckpt = torch.load(combined_path2, map_location='cpu', weights_only=False)
        n_h = ckpt['human_gs']['xyz'].shape[0]
        n_o = ckpt['object_gs']['xyz'].shape[0]
        human_gs = GaussianModel(num_points=n_h)
        object_gs = GaussianModel(num_points=n_o)
        human_gs.load_state_dict(ckpt['human_gs'])
        object_gs.load_state_dict(ckpt['object_gs'])
        return human_gs.to(device), object_gs.to(device)

    print(f'[Step4] No GS init found in {gs_init_dir}, using random init')
    human_gs = GaussianModel(num_points=4096, init_extent=0.5)
    object_gs = GaussianModel(num_points=2048, init_extent=0.3)
    return human_gs.to(device), object_gs.to(device)


# ============================================================
# 10. Final Output Export
# ============================================================

def _export_final_results(
    human_gs: GaussianModel,
    object_gs: GaussianModel,
    se3_human: SE3Transform,
    se3_object: SE3Transform,
    joint_renderer: JointRenderer,
    frames: List[Tensor],
    output_dir: str,
    H: int,
    W: int,
    device: torch.device,
) -> None:
    """
    Export final optimized 3DGS as:
      1. Merged point cloud PLY (human + object in world space)
      2. Per-entity PLY files
      3. Rendered images for each input frame
      4. Final checkpoint with all parameters
    """
    import numpy as np

    with torch.no_grad():
        frame_idx = 0 if getattr(se3_human, "num_frames", 1) > 1 else None
        xyz_h = se3_human(human_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
        xyz_o = se3_object(object_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
        col_h = human_gs.get_colors.cpu().numpy()               # (N_h, 3)
        col_o = object_gs.get_colors.cpu().numpy()              # (N_o, 3)

    # --- PLY export (simple ASCII format) ---
    def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
        N = xyz.shape[0]
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        with open(path, 'w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {N}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            for i in range(N):
                f.write(f'{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} '
                        f'{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n')

    _write_ply(os.path.join(output_dir, 'human_world.ply'), xyz_h, col_h)
    _write_ply(os.path.join(output_dir, 'object_world.ply'), xyz_o, col_o)
    _write_ply(
        os.path.join(output_dir, 'merged_world.ply'),
        np.concatenate([xyz_h, xyz_o], axis=0),
        np.concatenate([col_h, col_o], axis=0),
    )
    print(f'  [Export] PLY files saved to {output_dir}')

    # --- Render each frame ---
    render_dir = os.path.join(output_dir, 'renders')
    os.makedirs(render_dir, exist_ok=True)
    import cv2
    with torch.no_grad():
        for i, gt in enumerate(frames):
            rendered, _, _ = joint_renderer(human_gs, object_gs, frame_idx=i)
            rendered_np = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            gt_np = gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            # Side-by-side: GT | Rendered
            combined = np.concatenate([gt_np, rendered_np], axis=1)
            combined_bgr = cv2.cvtColor(
                (combined * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(
                os.path.join(render_dir, f'frame_{i:04d}.png'), combined_bgr
            )
    print(f'  [Export] Rendered {len(frames)} frames to {render_dir}')


# ============================================================
# 11. Main Pipeline Runner
# ============================================================

def run_step4_pipeline(cfg) -> None:
    """
    Full Step 4 pipeline: load all prior outputs, build models, run optimization.

    cfg : Step4PipelineConfig (or OmegaConf DictConfig with same structure)
    """
    import glob
    import time as _time
    import numpy as np
    import cv2
    from tqdm import tqdm
    from einops import rearrange

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'[Step4] Device: {device}')

    base_dir = os.path.join(cfg.input_dir, cfg.video_name)
    processed_dir = os.path.join(base_dir, cfg.processed_subdir)
    gs_init_dir = os.path.join(base_dir, cfg.gs_init_subdir)
    output_dir = os.path.join(base_dir, cfg.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    H, W = cfg.image_height, cfg.image_width

    # ---- Load original video frames ----
    frames_dir = os.path.join(base_dir, 'frames')
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    assert len(frame_paths) > 0, f'No frames found in {frames_dir}'

    frames = []
    for p in frame_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        t = rearrange(torch.from_numpy(img).float(), 'H W C -> C H W') / 255.0
        frames.append(t.to(device))
    print(f'[Step4] Loaded {len(frames)} frames at {H}x{W}')

    # ---- Load Step 1 outputs ----
    step1_data = load_step1_outputs(processed_dir, device)

    # Generate fallback masks if Step 1 outputs not available
    num_frames = len(frames)
    if not step1_data['masks_visible']:
        print('[Step4] Warning: No soft masks found, using uniform weights')
        step1_data['masks_visible'] = [torch.ones(H, W, device=device)] * num_frames
        step1_data['masks_primary_occ'] = [torch.zeros(H, W, device=device)] * num_frames
        step1_data['masks_secondary_occ'] = [torch.zeros(H, W, device=device)] * num_frames

    # ---- Load initial 3DGS from Step 3 (or Step 3.5 metric-aligned) ----
    human_gs, object_gs = load_gs_init(gs_init_dir, device)
    print(f'[Step4] Human GS: {human_gs.num_points} pts, Object GS: {object_gs.num_points} pts')

    # Detect metric-aligned data (Step 3.5 Alignment Bridge).
    # If gs_aligned/ exists, the 3DGS is already in metric space →
    # SE(3) starts from identity (residual refinement only).
    aligned_dir = os.path.join(base_dir, 'gs_aligned')
    _metric_aligned = (
        os.path.isfile(os.path.join(aligned_dir, 'human_gaussians_metric.npz'))
        and os.path.isfile(os.path.join(aligned_dir, 'object_gaussians_metric.npz'))
    )
    if _metric_aligned:
        print('[Step4] Metric-aligned 3DGS detected → SE(3) init = identity (residual mode)')

    # ---- Build SE(3) transforms ----
    if _metric_aligned:
        # Identity init: data is already in metric space, SE(3) learns residual
        se3_human = SE3Transform(init_translation=(0., 0., 0.), num_frames=num_frames).to(device)
        se3_object = SE3Transform(init_translation=(0., 0., 0.), num_frames=num_frames).to(device)
    else:
        se3_human = SE3Transform(
            init_translation=cfg.se3.init_translation_human,
            num_frames=num_frames,
        ).to(device)
        se3_object = SE3Transform(
            init_translation=cfg.se3.init_translation_object,
            num_frames=num_frames,
        ).to(device)

    # ---- Build renderer ----
    base_renderer = SimpleProjectionRenderer(H, W, focal=cfg.focal).to(device)
    joint_renderer = JointRenderer(base_renderer, se3_human, se3_object).to(device)

    # ---- Build SDF module ----
    sdf_module = VolumetricSMPLSDF(
        resolution=cfg.penetration.sdf_grid_resolution,
        padding=cfg.penetration.sdf_padding,
    ).to(device) if cfg.penetration.enabled else None

    # ---- Optimizer ----
    param_groups = [
        {'params': [human_gs.xyz, object_gs.xyz], 'lr': cfg.lr_xyz},
        {'params': [human_gs.opacity, object_gs.opacity], 'lr': cfg.lr_opacity},
        {'params': [human_gs.scaling, object_gs.scaling], 'lr': cfg.lr_scaling},
        {'params': [human_gs.rotation, object_gs.rotation], 'lr': cfg.lr_rotation},
        {'params': [human_gs.shs, object_gs.shs], 'lr': cfg.lr_color},
        # SE(3) registration parameters
        {'params': [se3_human.translation, se3_object.translation],
         'lr': cfg.se3.lr_translation},
        {'params': [se3_human.axis_angle, se3_object.axis_angle],
         'lr': cfg.se3.lr_rotation},
    ]
    optimizer = torch.optim.Adam(param_groups)

    # ---- LR Scheduler: cosine annealing to 1% of initial LR ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_iters, eta_min=cfg.lr_xyz * 0.01,
    )

    # ---- Resolve hand joint indices ----
    hand_indices = list(cfg.contact.hand_joint_indices) if cfg.contact.enabled else None

    # ---- Extract camera intrinsics ----
    focal = cfg.focal
    cx, cy = W / 2.0, H / 2.0
    if step1_data['camera_K'] is not None:
        K = step1_data['camera_K']
        focal = K[0, 0].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()

    # ---- Training loop ----
    print(f'[Step4] Starting optimization ({cfg.num_iters} iters)...')
    print(f'[Step4] Losses: render(vis={cfg.region_loss.weight_visible}/'
          f'pri={cfg.region_loss.weight_primary_occ}/'
          f'sec={cfg.region_loss.weight_secondary_occ}) '
          f'contact={cfg.contact.lambda_contact if cfg.contact.enabled else 0} '
          f'j2d={cfg.proj2d.lambda_j2d if cfg.proj2d.enabled else 0} '
          f'pen={cfg.penetration.lambda_pen if cfg.penetration.enabled else 0} '
          f'acc={cfg.temporal.lambda_acc if cfg.temporal.enabled else 0}')

    pbar = tqdm(range(1, cfg.num_iters + 1), desc='Step4 Joint Opt')
    train_start = _time.time()

    for step in pbar:
        step_start = _time.time()
        idx = (step - 1) % num_frames

        # Current frame data
        gt_image = frames[idx]
        m_vis = step1_data['masks_visible'][idx % len(step1_data['masks_visible'])]
        m_pri = step1_data['masks_primary_occ'][idx % len(step1_data['masks_primary_occ'])]
        m_sec = step1_data['masks_secondary_occ'][idx % len(step1_data['masks_secondary_occ'])]

        # Optional SMPL-H data for this frame
        smpl_joints = None
        smpl_verts = None
        smpl_faces_t = None
        kp2d = None
        kp_conf = None

        if step1_data['smplh_params'] and idx < len(step1_data['smplh_params']):
            params = step1_data['smplh_params'][idx]
            if 'joints_3d' in params:
                smpl_joints = params['joints_3d']              # (J, 3)
            if 'vertices' in params:
                smpl_verts = params['vertices']                # (V, 3)
            if 'faces' in params:
                smpl_faces_t = params['faces'].long()          # (F, 3)

        if step1_data['keypoints_2d'] and idx < len(step1_data['keypoints_2d']):
            kp2d = step1_data['keypoints_2d'][idx]
            kp_conf = step1_data['kp_confidence'][idx]

        log = step4_training_step(
            human_gs=human_gs,
            object_gs=object_gs,
            joint_renderer=joint_renderer,
            gt_image=gt_image,
            mask_visible=m_vis,
            mask_primary_occ=m_pri,
            mask_secondary_occ=m_sec,
            optimizer=optimizer,
            smpl_joints_3d=smpl_joints,
            hand_joint_indices=hand_indices,
            keypoints_2d=kp2d,
            kp_confidence=kp_conf,
            smpl_vertices=smpl_verts,
            smpl_faces=smpl_faces_t,
            sdf_module=sdf_module,
            frame_indices=[idx],
            se3_human_module=se3_human if cfg.temporal.enabled else None,
            se3_object_module=se3_object if cfg.temporal.enabled else None,
            w_visible=cfg.region_loss.weight_visible,
            w_primary=cfg.region_loss.weight_primary_occ,
            w_secondary=cfg.region_loss.weight_secondary_occ,
            lambda_ssim=cfg.region_loss.lambda_ssim,
            lambda_contact=cfg.contact.lambda_contact if cfg.contact.enabled else 0.0,
            lambda_j2d=cfg.proj2d.lambda_j2d if cfg.proj2d.enabled else 0.0,
            lambda_pen=cfg.penetration.lambda_pen if cfg.penetration.enabled else 0.0,
            lambda_acc=cfg.temporal.lambda_acc if cfg.temporal.enabled else 0.0,
            focal=focal,
            cx=cx,
            cy=cy,
            conf_threshold=cfg.proj2d.confidence_threshold if cfg.proj2d.enabled else 0.3,
        )

        # Step LR scheduler
        scheduler.step()

        # Logging with ETA (Master Guidelines requirement)
        if step % cfg.log_every == 0:
            elapsed = _time.time() - train_start
            steps_done = step
            step_time = _time.time() - step_start
            eta_s = elapsed / steps_done * (cfg.num_iters - step)
            if eta_s >= 3600:
                eta_str = f'{eta_s / 3600:.1f}h'
            elif eta_s >= 60:
                eta_str = f'{eta_s / 60:.1f}m'
            else:
                eta_str = f'{eta_s:.0f}s'
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                render=f"{log['loss_render']:.4f}",
                contact=f"{log['loss_contact']:.4f}",
                pen=f"{log['loss_penetration']:.5f}",
                acc=f"{log['loss_temporal']:.5f}",
                lr=f"{lr_now:.1e}",
                eta=eta_str,
            )

        # Checkpoint
        if step % cfg.save_every == 0 or step == cfg.num_iters:
            ckpt = {
                'step': step,
                'human_gs': human_gs.state_dict(),
                'object_gs': object_gs.state_dict(),
                'se3_human': se3_human.state_dict(),
                'se3_object': se3_object.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_path = os.path.join(output_dir, f'step4_ckpt_{step:06d}.pt')
            torch.save(ckpt, ckpt_path)
            print(f'\n  [Save] {ckpt_path}')

    # ---- Final output export ----
    _export_final_results(
        human_gs, object_gs, se3_human, se3_object,
        joint_renderer, frames, output_dir, H, W, device,
    )

    total_time = _time.time() - train_start
    print(f'[Step4] Optimization complete in {total_time / 60:.1f}min. Results in {output_dir}')


# ============================================================
# 12. CLI Entry Point
# ============================================================

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Step 4: Joint 3DGS Optimization')
    p.add_argument('--input_dir', type=str, default='./sample_data')
    p.add_argument('--video_name', type=str, default='test_video')
    p.add_argument('--processed_subdir', type=str, default='processed')
    p.add_argument('--gs_init_subdir', type=str, default='gs_init')
    p.add_argument('--output_subdir', type=str, default='joint_opt')
    p.add_argument('--image_height', type=int, default=256)
    p.add_argument('--image_width', type=int, default=256)
    p.add_argument('--num_iters', type=int, default=5000)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    # Build config from CLI args
    from configs.step4_config import Step4PipelineConfig
    cfg = Step4PipelineConfig(
        input_dir=args.input_dir,
        video_name=args.video_name,
        processed_subdir=args.processed_subdir,
        gs_init_subdir=args.gs_init_subdir,
        output_subdir=args.output_subdir,
        image_height=args.image_height,
        image_width=args.image_width,
        num_iters=args.num_iters,
        device=args.device,
    )
    run_step4_pipeline(cfg)
