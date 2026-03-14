"""
Step 3.5: Metric Alignment Bridge — strongly-typed dataclass config.

Bridges the gap between Step 3 (normalized canonical 3DGS) and Step 4
(metric-space joint optimization) by computing a deterministic affine
transform (scale + translation) from observed depth and masks.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MetricAlignmentConfig:
    """度量对齐桥接模块配置 (Metric Alignment Bridge Config)."""

    enabled: bool = True

    # --- Depth validity filtering ---
    depth_max: float = 10.0           # max valid depth in metres
    depth_min_pixels: int = 50        # min valid pixels inside mask

    # --- Robust statistics ---
    percentile: float = 90.0          # percentile for radius estimation

    # --- Numerical safety ---
    scale_eps: float = 1e-4           # R_norm collapse threshold
    scale_default: float = 1.0        # fallback scale when collapsed
    scale_min: float = 0.01           # scale clamp lower bound
    scale_max: float = 100.0          # scale clamp upper bound

    # --- Human alignment strategy ---
    # "smplh"     : use SMPL-H translation for centre, depth-unproject for scale
    # "unproject" : same depth-unproject pipeline as object
    human_align_strategy: str = "smplh"

    # --- Post-transform validation ---
    validate_transform: bool = True
    z_positive_check: bool = True
    scale_range_min: float = 1e-5     # exp(S) lower bound
    scale_range_max: float = 10.0     # exp(S) upper bound


@dataclass
class AlignmentPipelineConfig:
    """Aggregated config for the alignment bridge pipeline."""

    project_root: str = "/data4/guanz/coding/HDM"
    input_dir: str = "./sample_data"
    video_name: str = "test_video"

    # Sub-directory names (relative to <input_dir>/<video_name>/)
    processed_subdir: str = "processed"   # Step 1 outputs
    gs_init_subdir: str = "gs_init"       # Step 3 outputs
    output_subdir: str = "gs_aligned"     # this module's outputs

    device: str = "cpu"  # deterministic, CPU is fine

    # Image dimensions (needed for fallback intrinsics)
    image_height: int = 256
    image_width: int = 256
    focal: float = 500.0  # fallback focal length

    # Core alignment config
    alignment: MetricAlignmentConfig = field(
        default_factory=MetricAlignmentConfig
    )
