"""
Step 4: Strongly-typed dataclass configs for Multi-Region Contact-Aware
Joint 3DGS Optimization.

Extends the basic Joint3DGSModelConfig with:
  - SE(3) coordinate registration parameters
  - Multi-region rendering loss weights (visible / primary-occlusion / secondary-occlusion)
  - Contact loss, 2D projection loss, penetration loss, temporal smoothness loss
  - SMPL-H body model paths for volumetric penetration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SE3RegistrationConfig:
    """Learnable SE(3) alignment from canonical space to world coordinates."""
    lr_translation: float = 1e-3
    lr_rotation: float = 1e-4
    init_translation_human: tuple = (0.0, 0.0, 2.0)
    init_translation_object: tuple = (0.0, 0.0, 2.0)


@dataclass
class MultiRegionLossConfig:
    """Weights for the three-zone rendering loss."""
    weight_visible: float = 1.0       # strict penalty on visible region
    weight_primary_occ: float = 0.3   # moderate penalty on primary occlusion
    weight_secondary_occ: float = 0.05  # light penalty on secondary occlusion
    lambda_ssim: float = 0.2


@dataclass
class ContactLossConfig:
    """Contact loss: hand joints ↔ object Gaussian nearest-neighbor."""
    enabled: bool = True
    lambda_contact: float = 0.5
    # Indices of SMPL-H hand joints (wrist + fingers) used for contact
    # SMPL-H: 20=L_wrist, 21=R_wrist, 22-36=L_hand, 37-51=R_hand
    hand_joint_indices: tuple = (20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51)


@dataclass
class Projection2DLossConfig:
    """2D joint projection loss: SMPL-H joints → 2D vs OpenPose detections."""
    enabled: bool = True
    lambda_j2d: float = 0.1
    confidence_threshold: float = 0.3  # ignore low-confidence OpenPose detections


@dataclass
class PenetrationLossConfig:
    """Volumetric SMPL penetration loss via SDF."""
    enabled: bool = True
    lambda_pen: float = 1.0
    sdf_grid_resolution: int = 64
    sdf_padding: float = 0.1  # padding around SMPL bounding box


@dataclass
class TemporalSmoothnessConfig:
    """Acceleration-based temporal smoothness on framewise SE(3) trajectories."""
    enabled: bool = True
    lambda_acc: float = 0.5


@dataclass
class Step4PipelineConfig:
    """Aggregated config for the Step 4 joint optimization pipeline."""
    project_root: str = "/data4/guanz/coding/HDM"
    # Input directories (from Step 1 / Step 2 / Step 3 outputs)
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    processed_subdir: str = "processed"   # Step 1 outputs
    amodal_subdir: str = "amodal"         # Step 2 outputs
    gs_init_subdir: str = "gs_init"       # Step 3 outputs
    output_subdir: str = "joint_opt"      # Step 4 outputs
    device: str = "cuda"

    # Image dimensions
    image_height: int = 256
    image_width: int = 256

    # Gaussian model
    num_points_human: int = 4096
    num_points_object: int = 2048
    focal: float = 500.0

    # Training
    num_iters: int = 5000
    save_every: int = 1000
    log_every: int = 50

    # Per-parameter learning rates
    lr_xyz: float = 1.6e-4
    lr_opacity: float = 5e-2
    lr_scaling: float = 5e-3
    lr_rotation: float = 1e-3
    lr_color: float = 2.5e-3
    lr_min_ratio: float = 0.01  # cosine annealing decays to lr * this ratio

    # Sub-configs
    se3: SE3RegistrationConfig = field(default_factory=SE3RegistrationConfig)
    region_loss: MultiRegionLossConfig = field(default_factory=MultiRegionLossConfig)
    contact: ContactLossConfig = field(default_factory=ContactLossConfig)
    proj2d: Projection2DLossConfig = field(default_factory=Projection2DLossConfig)
    penetration: PenetrationLossConfig = field(default_factory=PenetrationLossConfig)
    temporal: TemporalSmoothnessConfig = field(default_factory=TemporalSmoothnessConfig)

    # SMPL-H body model
    smplh_model_dir: str = "/data4/guanz/coding/HDM/model/smpl_models/smplh"
