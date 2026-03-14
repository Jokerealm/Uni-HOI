"""
Step 3.5: Metric Alignment Bridge.

Deterministic affine transform that maps Flow-Matching-generated 3DGS
from normalized canonical space to the metric physical space established
by the Preprocess stage.  Runs on CPU; negligible wall-clock cost.

Pipeline:
  1. Unproject observed depth inside mask → 3-D surface point cloud
  2. Estimate robust scale  s  and depth-compensated translation  t
  3. Apply affine transform to all 14-channel 3DGS attributes
  4. Validate & save
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from configs.alignment_config import AlignmentPipelineConfig, MetricAlignmentConfig

logger = logging.getLogger(__name__)


# ============================================================
# Phase 1: Observed Surface Unprojection
# ============================================================

def unproject_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_max: float = 10.0,
) -> np.ndarray:
    """
    Back-project masked depth pixels into 3-D camera-frame coordinates.

    Parameters
    ----------
    depth : (H, W) float32, metric depth in metres.
    mask  : (H, W) binary (>0 = valid).
    fx, fy, cx, cy : camera intrinsics.
    depth_max : discard pixels with depth > this value.

    Returns
    -------
    points : (N, 3) float64 — X, Y, Z in camera frame.
    """
    H, W = depth.shape
    vs, us = np.where(mask > 0)
    zs = depth[vs, us].astype(np.float64)

    # Filter invalid depth
    valid = (zs > 0) & (zs <= depth_max) & np.isfinite(zs)
    us = us[valid].astype(np.float64)
    vs = vs[valid].astype(np.float64)
    zs = zs[valid]

    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy

    return np.stack([xs, ys, zs], axis=-1)  # (N, 3)


# ============================================================
# Phase 2: Robust Scale & Translation Estimation
# ============================================================

def _robust_xy_radius(points: np.ndarray, percentile: float) -> Tuple[float, np.ndarray]:
    """
    Compute the robust XY-plane radius of a point cloud using a percentile
    of distances from the median centre.

    Returns (radius, median_xy).
    """
    xy = points[:, :2]
    med = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - med, axis=1)
    radius = float(np.percentile(dists, percentile))
    return radius, med


def _robust_z_half_thickness(points_z: np.ndarray, percentile: float) -> float:
    """
    Estimate the half-thickness of a point cloud along Z using the percentile
    of absolute deviations from the median.
    """
    med = float(np.median(points_z))
    abs_dev = np.abs(points_z - med)
    return float(np.percentile(abs_dev, percentile))


def estimate_scale_and_translation(
    P_obs: np.ndarray,
    P_norm: np.ndarray,
    cfg: MetricAlignmentConfig,
) -> Tuple[float, np.ndarray, dict]:
    """
    Estimate isotropic scale *s* and 3-D translation *t* that maps
    the normalised point cloud ``P_norm`` into the metric frame defined
    by the observed surface ``P_obs``.

    Returns (s, t, meta) where meta contains intermediate diagnostics.
    """
    meta: dict = {}

    # --- Scale ---
    R_obs, med_obs_xy = _robust_xy_radius(P_obs, cfg.percentile)
    R_norm, med_norm_xy = _robust_xy_radius(P_norm, cfg.percentile)

    meta["R_obs"] = R_obs
    meta["R_norm"] = R_norm

    if R_norm < cfg.scale_eps:
        logger.warning(
            "Normalised point cloud collapsed (R_norm=%.6f < eps=%.6f). "
            "Using default scale %.2f.",
            R_norm, cfg.scale_eps, cfg.scale_default,
        )
        s = cfg.scale_default
        meta["scale_degraded"] = True
    else:
        s = R_obs / R_norm
        meta["scale_degraded"] = False

    # Clamp
    s_raw = s
    s = float(np.clip(s, cfg.scale_min, cfg.scale_max))
    if s != s_raw:
        logger.warning("Scale clamped: %.4f → %.4f", s_raw, s)
    meta["scale"] = s

    # --- Translation (with depth compensation) ---
    R_norm_z = _robust_z_half_thickness(P_norm[:, 2], cfg.percentile)
    meta["R_norm_z"] = R_norm_z

    t_x = float(np.median(P_obs[:, 0]))
    t_y = float(np.median(P_obs[:, 1]))
    t_z = float(np.median(P_obs[:, 2])) + s * R_norm_z

    t = np.array([t_x, t_y, t_z], dtype=np.float64)
    meta["translation"] = t.copy()

    return s, t, meta


def estimate_scale_and_translation_smplh(
    P_norm: np.ndarray,
    P_obs: np.ndarray,
    smplh_transl: np.ndarray,
    cfg: MetricAlignmentConfig,
) -> Tuple[float, np.ndarray, dict]:
    """
    Human-specific alignment using SMPL-H translation as the metric centre.

    Scale is still estimated from depth-unprojected surface vs normalised
    point cloud.  Translation is taken directly from SMPL-H ``transl``.
    """
    meta: dict = {}

    # Scale from XY radii (same logic as object)
    R_obs, _ = _robust_xy_radius(P_obs, cfg.percentile)
    R_norm, _ = _robust_xy_radius(P_norm, cfg.percentile)
    meta["R_obs"] = R_obs
    meta["R_norm"] = R_norm

    if R_norm < cfg.scale_eps:
        logger.warning(
            "Human normalised cloud collapsed (R_norm=%.6f). "
            "Using default scale.",
            R_norm,
        )
        s = cfg.scale_default
        meta["scale_degraded"] = True
    else:
        s = R_obs / R_norm
        meta["scale_degraded"] = False

    s_raw = s
    s = float(np.clip(s, cfg.scale_min, cfg.scale_max))
    if s != s_raw:
        logger.warning("Human scale clamped: %.4f → %.4f", s_raw, s)
    meta["scale"] = s

    # Translation from SMPL-H
    t = smplh_transl.astype(np.float64).ravel()[:3].copy()
    meta["translation"] = t.copy()
    meta["source"] = "smplh"

    return s, t, meta


# ============================================================
# Phase 3: Affine Transform of 3DGS Attributes
# ============================================================

def transform_gaussians(
    raw: np.ndarray,
    s: float,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply isotropic scale + translation to a (N, 14) 3DGS parameter tensor.

    Channel layout (Step 3 ``raw`` output — model's native output space,
    NOT activated):
        0-2   : xyz  (means) — direct coordinates
        3-6   : rotation quaternion
        7-9   : scaling — raw model output; ``GaussianModel.from_phase2``
                 will apply ``log(clamp(val, 1e-6))`` to convert to log-space
        10    : opacity — raw model output; ``from_phase2`` applies
                 ``logit(clamp(val, 1e-4, 1-1e-4))``
        11-13 : SH DC colour — raw model output; ``from_phase2`` applies
                 ``logit(clamp(val, 1e-4, 1-1e-4))``

    Transform rules:
        - xyz: μ_metric = s * μ_norm + t
        - scaling: multiply by s (so that after from_phase2's log(),
          the effective log-scale shifts by ln(s))
        - rotation, opacity, SH: unchanged
    """
    out = raw.copy()
    t = t.astype(raw.dtype)

    # 3.1  Means: μ_metric = s * μ_norm + t
    out[:, 0:3] = s * raw[:, 0:3] + t[None, :]

    # 3.2  Rotation quaternion: unchanged

    # 3.3  Scaling: multiply by s in raw space.
    #   For positive raw values: from_phase2 does log(raw*s) = log(raw) + ln(s) ✓
    #   For negative raw values: from_phase2 clamps to 1e-6 anyway (pre-existing loss)
    out[:, 7:10] = raw[:, 7:10] * s

    # 3.4  Opacity: unchanged
    # 3.5  SH: unchanged

    return out


# ============================================================
# Phase 4: Post-Transform Validation
# ============================================================

def validate_transformed_gaussians(
    raw_metric: np.ndarray,
    cfg: MetricAlignmentConfig,
    label: str = "",
) -> np.ndarray:
    """
    Run sanity checks on the transformed 3DGS raw tensor and apply soft fixes.

    NOTE: The raw tensor is in the model's native output space (NOT activated).
    ``GaussianModel.from_phase2`` will later apply activations (log for scaling,
    logit for opacity/SH).  Validation here operates on the raw values.

    Returns the (possibly corrected) array.
    """
    prefix = f"[Validate {label}]" if label else "[Validate]"
    out = raw_metric.copy()

    # Check Z > 0 (xyz channels 0-2 are direct coordinates)
    if cfg.z_positive_check:
        z_vals = out[:, 2]
        n_neg = int((z_vals < 0).sum())
        if n_neg > 0:
            logger.warning(
                "%s %d / %d Gaussians have Z < 0. "
                "Setting their opacity to near-zero in raw space.",
                prefix, n_neg, len(z_vals),
            )
            # In raw space, opacity goes through sigmoid in from_phase2.
            # A large negative value → sigmoid → ~0.
            neg_mask = z_vals < 0
            out[neg_mask, 10] = -10.0  # sigmoid(-10) ≈ 4.5e-5

    # NaN / Inf check
    if not np.all(np.isfinite(out)):
        n_bad = int((~np.isfinite(out)).any(axis=1).sum())
        logger.error(
            "%s %d Gaussians contain NaN/Inf! Replacing with zeros.",
            prefix, n_bad,
        )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out


# ============================================================
# Full Pipeline
# ============================================================

class MetricAlignmentBridge:
    """
    Deterministic alignment bridge: canonical 3DGS → metric 3DGS.

    Usage::

        bridge = MetricAlignmentBridge(cfg)
        bridge.run()
    """

    def __init__(self, cfg: AlignmentPipelineConfig):
        self.cfg = cfg
        self.acfg = cfg.alignment

        self.base_dir = os.path.join(cfg.input_dir, cfg.video_name)
        self.processed_dir = os.path.join(self.base_dir, cfg.processed_subdir)
        self.gs_init_dir = os.path.join(self.base_dir, cfg.gs_init_subdir)
        self.output_dir = os.path.join(self.base_dir, cfg.output_subdir)

    # ---- I/O helpers ----

    def _load_gs(self, name: str) -> np.ndarray:
        """Load Step 3 output (.pt dict with 'raw' key) → (N, 14) numpy."""
        path = os.path.join(self.gs_init_dir, f"{name}.pt")
        if not os.path.isfile(path):
            # Try combined file
            combined = os.path.join(self.gs_init_dir, "gs_init_combined.pt")
            if os.path.isfile(combined):
                data = torch.load(combined, map_location="cpu", weights_only=False)
                return data[name]["raw"].numpy().astype(np.float64)
            raise FileNotFoundError(
                f"Cannot find {path} or gs_init_combined.pt"
            )
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["raw"].numpy().astype(np.float64)

    def _load_depth_frame(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """
        Load a single aligned depth frame.

        Supports multiple layouts:
          1. Per-frame files in processed/depth/*.npz
          2. Consolidated processed/depth_aligned.npz with key 'depth' shape (T,H,W)
          3. Per-frame .npy files in processed/depth/
        """
        import glob

        # Layout 1: per-frame directory
        depth_dir = os.path.join(self.processed_dir, "depth")
        if os.path.isdir(depth_dir):
            paths = sorted(
                glob.glob(os.path.join(depth_dir, "*.npz"))
                + glob.glob(os.path.join(depth_dir, "*.npy"))
            )
            if paths:
                idx = min(frame_idx, len(paths) - 1)
                p = paths[idx]
                if p.endswith(".npz"):
                    d = np.load(p)
                    return d["depth"] if "depth" in d else d[list(d.keys())[0]]
                return np.load(p)

        # Layout 2: consolidated depth_aligned.npz  (T, H, W)
        consolidated = os.path.join(self.processed_dir, "depth_aligned.npz")
        if os.path.isfile(consolidated):
            d = np.load(consolidated)
            key = "depth" if "depth" in d else list(d.keys())[0]
            arr = d[key]
            if arr.ndim == 3:  # (T, H, W)
                idx = min(frame_idx, arr.shape[0] - 1)
                return arr[idx]
            return arr  # (H, W)

        # Layout 3: single .npy
        single_npy = os.path.join(self.processed_dir, "depth_aligned.npy")
        if os.path.isfile(single_npy):
            arr = np.load(single_npy)
            if arr.ndim == 3:
                idx = min(frame_idx, arr.shape[0] - 1)
                return arr[idx]
            return arr

        return None

    def _load_mask(self, subdir: str, frame_idx: int = 0) -> Optional[np.ndarray]:
        """
        Load a binary mask (.png or .npy).

        Supports multiple layouts:
          1. processed/masks/<subdir>/  (design-doc layout)
          2. processed/masks_<subdir>/  (sample_data actual layout)
          3. processed/<subdir>/        (flat layout)
        """
        import glob

        candidates = [
            os.path.join(self.processed_dir, "masks", subdir),
            os.path.join(self.processed_dir, f"masks_{subdir}"),
            os.path.join(self.processed_dir, subdir),
        ]
        mask_dir = None
        for c in candidates:
            if os.path.isdir(c):
                mask_dir = c
                break
        if mask_dir is None:
            return None

        paths = sorted(
            glob.glob(os.path.join(mask_dir, "*.png"))
            + glob.glob(os.path.join(mask_dir, "*.npy"))
        )
        if not paths:
            return None
        idx = min(frame_idx, len(paths) - 1)
        p = paths[idx]
        if p.endswith(".npy"):
            return np.load(p)
        import cv2
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8) if m is not None else None

    def _load_camera_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        Load camera K matrix or construct from available data.

        Search order:
          1. processed/camera_intrinsics.npy  (3x3 matrix)
          2. focal_length from smpl_params.npz + image dimensions
          3. Fallback to config defaults
        """
        # Option 1: explicit K matrix
        K_path = os.path.join(self.processed_dir, "camera_intrinsics.npy")
        if os.path.isfile(K_path):
            K = np.load(K_path)
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            return fx, fy, cx, cy

        # Option 2: focal_length from smpl_params.npz
        smpl_path = os.path.join(self.processed_dir, "smpl_params.npz")
        if os.path.isfile(smpl_path):
            data = np.load(smpl_path, allow_pickle=True)
            if "focal_length" in data:
                fl = data["focal_length"]
                focal = float(fl.ravel()[0])
                if focal > 0:
                    # Determine image size from depth or mask
                    depth = self._load_depth_frame(0)
                    if depth is not None:
                        H, W = depth.shape[:2]
                    else:
                        H, W = self.cfg.image_height, self.cfg.image_width
                    cx, cy = W / 2.0, H / 2.0
                    logger.info(
                        "Using focal_length=%.1f from smpl_params.npz, "
                        "image=%dx%d",
                        focal, W, H,
                    )
                    return focal, focal, cx, cy

        # Option 3: fallback
        logger.warning(
            "No camera intrinsics found. Using fallback focal=%.1f, "
            "cx=%.1f, cy=%.1f.",
            self.cfg.focal,
            self.cfg.image_width / 2.0,
            self.cfg.image_height / 2.0,
        )
        return (
            self.cfg.focal,
            self.cfg.focal,
            self.cfg.image_width / 2.0,
            self.cfg.image_height / 2.0,
        )

    def _load_smplh_translation(self) -> Optional[np.ndarray]:
        """
        Load SMPL-H translation from processed poses.

        Search order:
          1. processed/poses/smplh_aligned.npz  (key: transl or cam_t)
          2. processed/smpl_params.npz           (key: cam_t or transl)
          3. processed/smplh_params/*.npz        (per-frame)
        """
        def _extract_transl(data) -> Optional[np.ndarray]:
            """Try to extract a (3,) translation vector from npz data."""
            for key in ("transl", "cam_t", "translation"):
                if key in data:
                    arr = np.array(data[key])
                    # Could be (3,), (1,3), (T,3), (T,1,3)
                    arr = arr.squeeze()
                    if arr.ndim == 1 and arr.shape[0] >= 3:
                        return arr[:3].astype(np.float64)
                    if arr.ndim == 2:
                        return arr[0, :3].astype(np.float64)
            return None

        # Layout 1: consolidated poses file
        for fname in ("smplh_aligned.npz", "smpl_params.npz"):
            for subdir in ("poses", ""):
                npz_path = os.path.join(self.processed_dir, subdir, fname)
                if os.path.isfile(npz_path):
                    data = np.load(npz_path, allow_pickle=True)
                    t = _extract_transl(data)
                    if t is not None:
                        return t

        # Layout 2: per-frame directory
        import glob
        for dirname in ("smplh_params", "smpl_params"):
            smplh_dir = os.path.join(self.processed_dir, dirname)
            if os.path.isdir(smplh_dir):
                paths = sorted(glob.glob(os.path.join(smplh_dir, "*.npz")))
                if paths:
                    data = np.load(paths[0], allow_pickle=True)
                    t = _extract_transl(data)
                    if t is not None:
                        return t

        return None

    # ---- Core alignment for one entity ----

    def _align_entity(
        self,
        raw_norm: np.ndarray,
        P_obs: Optional[np.ndarray],
        label: str,
        smplh_transl: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Align a single entity (human or object).

        Returns (raw_metric, meta_dict).
        """
        acfg = self.acfg
        P_norm = raw_norm[:, 0:3]  # normalised means

        # Check if we have enough observed points
        if P_obs is None or len(P_obs) < acfg.depth_min_pixels:
            n_obs = 0 if P_obs is None else len(P_obs)
            logger.warning(
                "[%s] Insufficient observed points (%d < %d). "
                "Skipping alignment — Step 4 will use SE(3) from scratch.",
                label, n_obs, acfg.depth_min_pixels,
            )
            return raw_norm.copy(), {"degraded": True, "reason": "insufficient_obs"}

        # Choose alignment strategy
        use_smplh = (
            label == "human"
            and acfg.human_align_strategy == "smplh"
            and smplh_transl is not None
        )

        if use_smplh:
            s, t, meta = estimate_scale_and_translation_smplh(
                P_norm, P_obs, smplh_transl, acfg,
            )
        else:
            s, t, meta = estimate_scale_and_translation(
                P_obs, P_norm, acfg,
            )

        meta["degraded"] = False

        # Apply transform
        raw_metric = transform_gaussians(raw_norm, s, t)

        # Validate
        if acfg.validate_transform:
            raw_metric = validate_transformed_gaussians(
                raw_metric, acfg, label=label,
            )

        return raw_metric, meta

    # ---- Main entry point ----

    def run(self) -> Dict[str, np.ndarray]:
        """
        Execute the full metric alignment bridge.

        Returns dict with keys 'G_o_metric', 'G_h_metric'.
        """
        print("=" * 60)
        print("[Alignment] Metric Alignment Bridge (Step 3 → Step 4)")
        print(f"  GS init:   {self.gs_init_dir}")
        print(f"  Processed: {self.processed_dir}")
        print(f"  Output:    {self.output_dir}")
        print("=" * 60)

        if not self.acfg.enabled:
            print("[Alignment] Disabled by config. Skipping.")
            return {}

        os.makedirs(self.output_dir, exist_ok=True)

        # Load inputs
        raw_obj = self._load_gs("G_o")
        raw_hum = self._load_gs("G_h")
        print(f"[Alignment] Loaded G_o: {raw_obj.shape}, G_h: {raw_hum.shape}")

        depth = self._load_depth_frame(frame_idx=0)
        mask_obj = self._load_mask("object", frame_idx=0)
        mask_hum = self._load_mask("human", frame_idx=0)
        fx, fy, cx, cy = self._load_camera_intrinsics()
        smplh_transl = self._load_smplh_translation()

        print(f"[Alignment] Intrinsics: fx={fx:.1f} fy={fy:.1f} "
              f"cx={cx:.1f} cy={cy:.1f}")

        # Phase 1: Unproject observed surfaces
        P_obs_obj = None
        P_obs_hum = None

        if depth is not None and mask_obj is not None:
            P_obs_obj = unproject_depth(
                depth, mask_obj, fx, fy, cx, cy,
                depth_max=self.acfg.depth_max,
            )
            print(f"[Alignment] Object observed points: {len(P_obs_obj)}")
        else:
            logger.warning("[Alignment] Missing depth or object mask.")

        if depth is not None and mask_hum is not None:
            P_obs_hum = unproject_depth(
                depth, mask_hum, fx, fy, cx, cy,
                depth_max=self.acfg.depth_max,
            )
            print(f"[Alignment] Human observed points: {len(P_obs_hum)}")
        else:
            logger.warning("[Alignment] Missing depth or human mask.")

        # Phase 2 + 3: Align each entity
        obj_metric, obj_meta = self._align_entity(
            raw_obj, P_obs_obj, label="object",
        )
        hum_metric, hum_meta = self._align_entity(
            raw_norm=raw_hum,
            P_obs=P_obs_hum,
            label="human",
            smplh_transl=smplh_transl,
        )

        # Print summary
        for label, meta in [("Object", obj_meta), ("Human", hum_meta)]:
            if meta.get("degraded"):
                print(f"[Alignment] {label}: DEGRADED ({meta.get('reason', '?')})")
            else:
                print(
                    f"[Alignment] {label}: s={meta['scale']:.4f}, "
                    f"t=[{meta['translation'][0]:.3f}, "
                    f"{meta['translation'][1]:.3f}, "
                    f"{meta['translation'][2]:.3f}], "
                    f"R_obs={meta.get('R_obs', 0):.4f}, "
                    f"R_norm={meta.get('R_norm', 0):.4f}"
                )

        # Sanity: print Z-range of transformed means
        for label, arr in [("Object", obj_metric), ("Human", hum_metric)]:
            z = arr[:, 2]
            print(
                f"[Alignment] {label} Z-range: "
                f"[{z.min():.3f}, {z.max():.3f}], mean={z.mean():.3f}"
            )

        # Save outputs
        np.savez_compressed(
            os.path.join(self.output_dir, "object_gaussians_metric.npz"),
            raw=obj_metric.astype(np.float32),
        )
        np.savez_compressed(
            os.path.join(self.output_dir, "human_gaussians_metric.npz"),
            raw=hum_metric.astype(np.float32),
        )
        np.savez_compressed(
            os.path.join(self.output_dir, "alignment_meta.npz"),
            object_meta=obj_meta,
            human_meta=hum_meta,
        )

        print(f"[Alignment] Saved to {self.output_dir}")
        print("[Alignment] Done.")

        return {
            "G_o_metric": obj_metric,
            "G_h_metric": hum_metric,
        }
