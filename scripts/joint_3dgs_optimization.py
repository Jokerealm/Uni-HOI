#!/usr/bin/env python3
"""
Phase 3: Joint 3D Gaussian Splatting Optimisation on Unseparated Video
======================================================================
Jointly optimises *human* and *object* 3D Gaussians against the original
monocular video frames (no separation needed at training time).

Losses:
    1. Photometric  : L1 + λ_ssim * (1 - SSIM)  on the composited render
    2. Anti-penetration : repulsive penalty when human/object centres are
       closer than ε
    3. Occlusion regularisation : mild penalty on object Gaussians that
       project into the known human-mask region

Usage
-----
    python scripts/joint_3dgs_optimization.py \
        --frames_dir  data/frames \
        --masks_dir   data/masks_human \
        --num_iters   3000

All tensor shapes are annotated with inline comments.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1.  SSIM (structural similarity) – differentiable, window-based
# ---------------------------------------------------------------------------

def _fspecial_gauss(size: int, sigma: float, device: torch.device) -> Tensor:
    """2-D Gaussian kernel (normalised).  Returns (size, size)."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)                           # (size, size)
    return g / g.sum()


def ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> Tensor:
    """
    Compute mean SSIM between two images.

    img1, img2 : (B, C, H, W) float in [0, 1]
    returns    : scalar SSIM value (higher = more similar)
    """
    device = img1.device
    C = img1.shape[1]
    window = _fspecial_gauss(window_size, 1.5, device)  # (ws, ws)
    # (ws, ws) → (C, 1, ws, ws)  — depthwise conv kernel
    window = repeat(window, 'h w -> C 1 h w', C=C)

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)   # (B, C, H, W)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)   # (B, C, H, W)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=C) - mu1_sq  # (B, C, H, W)
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=C) - mu2_sq  # (B, C, H, W)
    sigma12   = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2  # (B, C, H, W)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )  # (B, C, H, W)
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# 2.  GaussianModel – learnable 3DGS parameter set
# ---------------------------------------------------------------------------

class GaussianModel(nn.Module):
    """
    A single set of 3D Gaussians with learnable parameters.

    Attributes (all nn.Parameter):
        xyz       : (N, 3)   – centres
        opacity   : (N, 1)   – log-opacity (sigmoid-activated at render)
        scaling   : (N, 3)   – log-scale
        rotation  : (N, 4)   – quaternion
        shs       : (N, 3)   – base spherical-harmonic colour (DC band)
    """

    def __init__(self, num_points: int = 4096, init_extent: float = 1.0):
        super().__init__()
        self.num_points = num_points

        self.xyz = nn.Parameter(
            (torch.rand(num_points, 3) - 0.5) * 2 * init_extent
        )  # (N, 3)
        self.opacity = nn.Parameter(
            torch.logit(torch.full((num_points, 1), 0.1))
        )  # (N, 1)  – inverse-sigmoid space
        self.scaling = nn.Parameter(
            torch.log(torch.full((num_points, 3), 0.01))
        )  # (N, 3)  – log space
        self.rotation = nn.Parameter(
            F.normalize(torch.randn(num_points, 4), dim=-1)
        )  # (N, 4)  – unit quaternion
        self.shs = nn.Parameter(
            torch.rand(num_points, 3) * 0.5
        )  # (N, 3)  – DC colour

    @classmethod
    def from_phase2(cls, tensor: Tensor) -> 'GaussianModel':
        """
        Issue 3: Initialize from Phase 2 output tensor.
        
        tensor : (N, 14) — xyz(0:3), rotation(3:7), scaling(7:10),
                            opacity(10:11), shs(11:14)
        """
        N = tensor.shape[0]
        model = cls(num_points=N)
        with torch.no_grad():
            model.xyz.data = tensor[:, 0:3].clone()
            model.rotation.data = F.normalize(tensor[:, 3:7].clone(), dim=-1)
            # Store in log space (scaling) and logit space (opacity)
            model.scaling.data = torch.log(tensor[:, 7:10].clamp(min=1e-6).clone())
            model.opacity.data = torch.logit(tensor[:, 10:11].clamp(1e-4, 1 - 1e-4).clone())
            # Store color in logit space (sigmoid-activated at render)
            model.shs.data = torch.logit(tensor[:, 11:14].clamp(1e-4, 1 - 1e-4).clone())
        return model

    @property
    def get_xyz(self) -> Tensor:
        return self.xyz

    @property
    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self.opacity)

    @property
    def get_scaling(self) -> Tensor:
        return torch.exp(self.scaling)

    @property
    def get_rotation(self) -> Tensor:
        return F.normalize(self.rotation, dim=-1)

    @property
    def get_colors(self) -> Tensor:
        return torch.sigmoid(self.shs)


# ---------------------------------------------------------------------------
# 3.  Differentiable Rasteriser (with fallback)
# ---------------------------------------------------------------------------

def _try_import_gsplat():
    """Try to import a 3DGS diff rasteriser (gsplat or diff-gaussian-rasterization)."""
    # Try gsplat first (Issue 8)
    try:
        import gsplat
        print('[3DGS] Using gsplat rasteriser')
        return gsplat, 'gsplat'
    except ImportError:
        pass
    # Fallback to diff-gaussian-rasterization
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        print('[3DGS] Using diff-gaussian-rasterization')
        return GaussianRasterizationSettings, GaussianRasterizer
    except ImportError:
        print('[3DGS] No CUDA rasteriser found, using SimpleProjectionRenderer fallback')
        return None, None


class SimpleProjectionRenderer(nn.Module):
    """
    Fallback differentiable renderer when the CUDA 3DGS rasteriser is not
    installed.  Projects Gaussians onto the image plane via a pinhole model
    and alpha-composites front-to-back using depth ordering.

    This is *not* a full splatting renderer — it is a lightweight stand-in
    that preserves gradient flow so the training loop can be developed and
    tested without the custom CUDA op.
    """

    def __init__(self, H: int, W: int, focal: float = 500.0):
        super().__init__()
        self.H, self.W = H, W
        self.focal = focal
        self.cx, self.cy = W / 2.0, H / 2.0

    def forward(
        self,
        xyz: Tensor,
        colors: Tensor,
        opacity: Tensor,
        scaling: Tensor,
    ) -> Tensor:
        """
        xyz     : (N, 3)
        colors  : (N, 3)  in [0, 1]
        opacity : (N, 1)  in [0, 1]
        scaling : (N, 3)  positive

        returns : (3, H, W) rendered RGB in [0, 1]
        """
        N = xyz.shape[0]
        device = xyz.device

        # ---- Pinhole projection (camera at origin, looking down +Z) ----
        z = xyz[:, 2].clamp(min=0.1)                          # (N,)
        px = (xyz[:, 0] / z) * self.focal + self.cx            # (N,)
        py = (xyz[:, 1] / z) * self.focal + self.cy            # (N,)

        # Approximate splat radius from mean scale
        radius_px = (scaling.mean(dim=-1) / z) * self.focal    # (N,)
        radius_px = radius_px.clamp(min=0.5, max=50.0)

        # ---- Depth-sort (front to back) ----
        order = z.argsort()                                    # (N,)
        px, py = px[order], py[order]                          # (N,)
        colors_s  = colors[order]                              # (N, 3)
        opacity_s = opacity[order, 0]                          # (N,)
        radius_s  = radius_px[order]                           # (N,)

        # ---- Rasterise via soft splatting on a grid ----
        yy, xx = torch.meshgrid(
            torch.arange(self.H, device=device, dtype=torch.float32),
            torch.arange(self.W, device=device, dtype=torch.float32),
            indexing="ij",
        )  # each (H, W)

        image = torch.zeros(3, self.H, self.W, device=device)  # (3, H, W)
        accum = torch.zeros(1, self.H, self.W, device=device)  # (1, H, W)

        # Process in chunks to limit memory
        CHUNK = min(N, 2048)
        for i in range(0, N, CHUNK):
            j = min(i + CHUNK, N)

            # (chunk, H, W) — pixel distance from each Gaussian centre
            dx = xx.unsqueeze(0) - px[i:j, None, None]        # (chunk, H, W)
            dy = yy.unsqueeze(0) - py[i:j, None, None]        # (chunk, H, W)
            r  = radius_s[i:j, None, None]                     # (chunk, 1, 1)

            gauss = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / (r ** 2 + 1e-6))
            # gauss : (chunk, H, W)

            alpha = gauss * opacity_s[i:j, None, None]         # (chunk, H, W)

            # Front-to-back compositing
            transmittance = (1.0 - accum).clamp(min=0.0)       # (1, H, W)
            weight = alpha * transmittance                      # (chunk, H, W)

            # (chunk, 3) → (chunk, 3, 1, 1) * (chunk, 1, H, W) → sum → (3, H, W)
            c = colors_s[i:j]                                   # (chunk, 3)
            image += (weight.unsqueeze(1) * c[:, :, None, None]).sum(0)
            accum += weight.sum(0, keepdim=True)                # (1, H, W)

        # Clamp accumulated opacity to [0, 1] before adding background
        accum = accum.clamp(0.0, 1.0)
        image = image + (1.0 - accum)                           # white background
        return image.clamp(0.0, 1.0)                            # (3, H, W)


# ---------------------------------------------------------------------------
# 4.  Loss functions
# ---------------------------------------------------------------------------

def photometric_loss(
    rendered: Tensor,
    gt: Tensor,
    lambda_ssim: float = 0.2,
) -> Tensor:
    """
    L1 + λ * (1 - SSIM).

    rendered : (B, 3, H, W) or (3, H, W)
    gt       : same shape
    returns  : scalar loss
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)  # (3, H, W) → (1, 3, H, W)
        gt = gt.unsqueeze(0)
    l1 = F.l1_loss(rendered, gt)
    ssim_val = ssim(rendered, gt)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)


def anti_penetration_loss(
    xyz_human: Tensor,
    xyz_object: Tensor,
    epsilon: float = 0.005,
    sigma: float = 0.002,
) -> Tensor:
    """
    Repulsive loss that penalises human–object Gaussian centres closer than ε.

    Uses a soft hinge:  loss = Σ max(0, ε - d_ij)  for nearest neighbours,
    computed efficiently via pytorch3d knn.

    xyz_human  : (N_h, 3)
    xyz_object : (N_o, 3)
    returns    : scalar repulsive loss
    """
    from pytorch3d.ops import knn_points

    # knn_points expects (B, N, 3)
    h = xyz_human.unsqueeze(0)                                 # (1, N_h, 3)
    o = xyz_object.unsqueeze(0)                                # (1, N_o, 3)

    # For each human point, find nearest object point
    knn_ho = knn_points(h, o, K=1)                             # dists: (1, N_h, 1)
    # (1, N_h, 1) → (N_h,)
    dist_ho = torch.sqrt(knn_ho.dists.squeeze(0).squeeze(-1) + 1e-8)

    # For each object point, find nearest human point
    knn_oh = knn_points(o, h, K=1)                             # dists: (1, N_o, 1)
    # (1, N_o, 1) → (N_o,)
    dist_oh = torch.sqrt(knn_oh.dists.squeeze(0).squeeze(-1) + 1e-8)

    # Hinge penalty: penalise distances < epsilon
    pen_ho = F.relu(epsilon - dist_ho)                         # (N_h,)
    pen_oh = F.relu(epsilon - dist_oh)                         # (N_o,)

    return pen_ho.sum() + pen_oh.sum()


def occlusion_regularisation(
    xyz_object: Tensor,
    opacity_object: Tensor,
    human_mask: Tensor,
    focal: float = 500.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    lambda_occ: float = 0.01,
) -> Tensor:
    """
    Mild regularisation on object Gaussians that project into the human mask.

    xyz_object      : (N_o, 3)
    opacity_object  : (N_o, 1) in [0, 1]
    human_mask      : (H, W) float {0, 1} — 1 = human region
    returns         : scalar regularisation loss
    """
    H, W = human_mask.shape
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # ---- Project object centres to pixel coords ----
    z  = xyz_object[:, 2].clamp(min=0.1)                      # (N_o,)
    px = (xyz_object[:, 0] / z) * focal + cx                  # (N_o,)
    py = (xyz_object[:, 1] / z) * focal + cy                  # (N_o,)

    # Convert to integer pixel indices (round for accuracy)
    px_idx = px.round().long().clamp(0, W - 1)                # (N_o,)
    py_idx = py.round().long().clamp(0, H - 1)                # (N_o,)

    # Only penalise points that actually project within the image
    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)   # (N_o,) bool

    # Look up mask value at each projected location
    in_human = human_mask[py_idx, px_idx] * in_bounds.float()  # (N_o,)

    # Penalise high opacity where the object is occluded by the human
    # (N_o, 1) → (N_o,)
    occ_penalty = (opacity_object.squeeze(-1) * in_human) ** 2 # (N_o,)

    return lambda_occ * occ_penalty.mean()


# ---------------------------------------------------------------------------
# 5.  Core training step
# ---------------------------------------------------------------------------

def training_step(
    human_gs: GaussianModel,
    object_gs: GaussianModel,
    renderer: SimpleProjectionRenderer,
    gt_image: Tensor,
    human_mask: Tensor,
    object_mask: Tensor,
    optimizer: torch.optim.Optimizer,
    # Loss weights
    lambda_ssim: float = 0.2,
    lambda_pen: float = 0.1,
    lambda_occ: float = 0.01,
    epsilon: float = 0.005,
) -> Dict[str, float]:
    """
    One gradient step of joint human + object 3DGS optimisation.

    gt_image    : (3, H, W)  ground-truth RGB in [0, 1]
    human_mask  : (H, W)     binary float mask (1 = human region)
    object_mask : (H, W)     binary float mask (1 = object region)
    returns     : dict of scalar loss values for logging
    """
    optimizer.zero_grad()

    # ---- Concatenate both Gaussian sets ----
    xyz_h, xyz_o = human_gs.get_xyz, object_gs.get_xyz         # (N_h, 3), (N_o, 3)
    col_h, col_o = human_gs.get_colors, object_gs.get_colors   # (N_h, 3), (N_o, 3)
    opa_h, opa_o = human_gs.get_opacity, object_gs.get_opacity  # (N_h, 1), (N_o, 1)
    scl_h, scl_o = human_gs.get_scaling, object_gs.get_scaling  # (N_h, 3), (N_o, 3)

    xyz_all = torch.cat([xyz_h, xyz_o], dim=0)                 # (N_h+N_o, 3)
    col_all = torch.cat([col_h, col_o], dim=0)                 # (N_h+N_o, 3)
    opa_all = torch.cat([opa_h, opa_o], dim=0)                 # (N_h+N_o, 1)
    scl_all = torch.cat([scl_h, scl_o], dim=0)                 # (N_h+N_o, 3)

    # ---- Render joint image ----
    rendered = renderer(xyz_all, col_all, opa_all, scl_all)     # (3, H, W)

    # ---- Loss 1: Photometric (L1 + SSIM) ----
    loss_photo = photometric_loss(rendered, gt_image, lambda_ssim=lambda_ssim)

    # ---- Loss 2: Anti-penetration (BIGS-style contact loss) ----
    loss_pen = anti_penetration_loss(xyz_h, xyz_o, epsilon=epsilon)

    # ---- Loss 3: Bidirectional occlusion regularisation ----
    loss_occ_obj = occlusion_regularisation(
        xyz_o, opa_o, human_mask, focal=renderer.focal, lambda_occ=lambda_occ,
    )
    loss_occ_hum = occlusion_regularisation(
        xyz_h, opa_h, object_mask, focal=renderer.focal, lambda_occ=lambda_occ,
    )
    loss_occ = loss_occ_obj + loss_occ_hum

    # ---- Total loss ----
    loss = loss_photo + lambda_pen * loss_pen + loss_occ

    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_photo": loss_photo.item(),
        "loss_penetration": loss_pen.item(),
        "loss_occlusion": loss_occ.item(),
    }


# ---------------------------------------------------------------------------
# 6.  Full training loop + CLI
# ---------------------------------------------------------------------------

def load_image(path: str, H: int, W: int) -> Tensor:
    """Load an image as (3, H, W) float tensor in [0, 1]."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
    # (H, W, 3) → (3, H, W)
    return rearrange(torch.from_numpy(img).float(), 'H W C -> C H W') / 255.0


def load_mask(path: str, H: int, W: int) -> Tensor:
    """Load a binary mask as (H, W) float tensor {0, 1}."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (W, H))
    return (torch.from_numpy(m).float() / 255.0).round()       # (H, W)


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    H, W = args.image_height, args.image_width

    # ---- Load data ----
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    human_mask_paths = sorted(glob.glob(os.path.join(args.masks_human_dir, "*.png"))) if args.masks_human_dir else []
    object_mask_paths = sorted(glob.glob(os.path.join(args.masks_object_dir, "*.png"))) if args.masks_object_dir else []

    assert len(frame_paths) > 0, f"No frames found in {args.frames_dir}"
    print(f"[Data] {len(frame_paths)} frames, {len(human_mask_paths)} human masks, {len(object_mask_paths)} object masks")

    # Pre-load all frames and masks to GPU
    frames = [load_image(p, H, W).to(device) for p in frame_paths]
    human_masks = [load_mask(p, H, W).to(device) for p in human_mask_paths] if human_mask_paths else [
        torch.zeros(H, W, device=device) for _ in frames
    ]
    object_masks = [load_mask(p, H, W).to(device) for p in object_mask_paths] if object_mask_paths else [
        torch.zeros(H, W, device=device) for _ in frames
    ]

    # ---- Initialise models ----
    human_gs = GaussianModel(num_points=args.num_points_human, init_extent=0.5).to(device)
    object_gs = GaussianModel(num_points=args.num_points_object, init_extent=0.3).to(device)

    renderer = SimpleProjectionRenderer(H, W, focal=args.focal).to(device)

    # ---- Optimiser (separate LR for xyz vs appearance) ----
    param_groups = [
        {"params": [human_gs.xyz, object_gs.xyz], "lr": args.lr_xyz},
        {"params": [human_gs.opacity, object_gs.opacity], "lr": args.lr_opacity},
        {"params": [human_gs.scaling, object_gs.scaling], "lr": args.lr_scaling},
        {"params": [human_gs.rotation, object_gs.rotation], "lr": args.lr_rotation},
        {"params": [human_gs.shs, object_gs.shs], "lr": args.lr_color},
    ]
    optimizer = torch.optim.Adam(param_groups)

    # ---- Training loop ----
    num_frames = len(frames)
    pbar = tqdm(range(1, args.num_iters + 1), desc="Joint 3DGS")

    for step in pbar:
        idx = (step - 1) % num_frames
        gt_image = frames[idx]          # (3, H, W)
        h_mask = human_masks[idx]       # (H, W)
        o_mask = object_masks[idx]      # (H, W)

        log = training_step(
            human_gs=human_gs, object_gs=object_gs, renderer=renderer,
            gt_image=gt_image, human_mask=h_mask, object_mask=o_mask,
            optimizer=optimizer,
            lambda_ssim=args.lambda_ssim, lambda_pen=args.lambda_pen,
            lambda_occ=args.lambda_occ, epsilon=args.epsilon,
        )

        if step % 100 == 0:
            pbar.set_postfix(
                photo=f"{log['loss_photo']:.4f}",
                pen=f"{log['loss_penetration']:.5f}",
                occ=f"{log['loss_occlusion']:.5f}",
            )

        # Save checkpoint
        if step % args.save_every == 0 or step == args.num_iters:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt = {
                "step": step,
                "human_gs": human_gs.state_dict(),
                "object_gs": object_gs.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            path = os.path.join(args.output_dir, f"ckpt_{step:06d}.pt")
            torch.save(ckpt, path)
            print(f"\n[Save] {path}")

    print("[Train] Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 – Joint 3DGS optimisation on unseparated video"
    )
    # Data
    p.add_argument("--frames_dir", type=str, required=True)
    p.add_argument("--masks_human_dir", type=str, default="",
                   help="Directory of human binary mask PNGs (from Phase 1).")
    p.add_argument("--masks_object_dir", type=str, default="",
                   help="Directory of object binary mask PNGs (from Phase 1).")
    p.add_argument("--image_height", type=int, default=256)
    p.add_argument("--image_width", type=int, default=256)
    # Model
    p.add_argument("--num_points_human", type=int, default=4096)
    p.add_argument("--num_points_object", type=int, default=2048)
    p.add_argument("--focal", type=float, default=500.0)
    # Training
    p.add_argument("--num_iters", type=int, default=3000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="outputs/phase3")
    # Learning rates
    p.add_argument("--lr_xyz", type=float, default=1.6e-4)
    p.add_argument("--lr_opacity", type=float, default=5e-2)
    p.add_argument("--lr_scaling", type=float, default=5e-3)
    p.add_argument("--lr_rotation", type=float, default=1e-3)
    p.add_argument("--lr_color", type=float, default=2.5e-3)
    # Loss weights
    p.add_argument("--lambda_ssim", type=float, default=0.2)
    p.add_argument("--lambda_pen", type=float, default=0.1)
    p.add_argument("--lambda_occ", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.005)
    return p.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
