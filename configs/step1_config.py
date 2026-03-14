"""
Step 1: Strongly-typed dataclass configs for the Data Prep & Perception pipeline.
Follows the project convention in configs/structured.py.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataPrepConfig:
    """Top-level config for the Step 1 pipeline."""
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    output_subdir: str = "processed"
    max_frames: Optional[int] = None
    device: str = "cuda"


@dataclass
class SAM3Config:
    """Config for SAM 3 text-prompted segmentation & video tracking."""
    model_dir: str = "/data4/guanz/coding/HDM/model/sam3"
    text_prompts: List[str] = field(default_factory=lambda: ["human", "object"])
    score_threshold: float = 0.5
    mask_threshold: float = 0.5


@dataclass
class SAM3DConfig:
    """Config for SAM 3D Body (SMPL-H mesh recovery)."""
    checkpoint: str = "/data4/guanz/coding/HDM/model/sam-3d-body/model.ckpt"
    mhr_model: str = "/data4/guanz/coding/HDM/model/sam-3d-body/assets/mhr_model.pt"
    config_yaml: str = "/data4/guanz/coding/HDM/model/sam-3d-body/model_config.yaml"
    image_size: Tuple[int, int] = (512, 512)


@dataclass
class UniDepthConfig:
    """Config for UniDepth V2 metric depth estimation."""
    model_dir: str = "/data4/guanz/coding/HDM/model/unidepth"
    backbone: str = "vitl14"


@dataclass
class OpenPoseConfig:
    """Config for OpenPose 2D keypoint detection."""
    model_dir: str = "/data4/guanz/coding/HDM/model/openpose"


@dataclass
class SMPLHConfig:
    """Config for SMPL-H body model files."""
    model_dir: str = "/data4/guanz/coding/HDM/model/smpl_models/smplh"


@dataclass
class DepthAlignConfig:
    """Config for depth scale alignment."""
    method: str = "median"


@dataclass
class MaskingConfig:
    """Config for multi-region contact-aware masking."""
    dilate_kernel_size: int = 15
    dilate_iterations: int = 2
    contact_radius: int = 10
    gaussian_blur_ksize: int = 11
    gaussian_blur_sigma: float = 5.0


@dataclass
class Step1PipelineConfig:
    """Aggregated config for the entire Step 1 pipeline."""
    base_weights_dir: str = "/data4/guanz/coding/HDM/model"
    project_root: str = "/data4/guanz/coding/HDM"
    data_prep: DataPrepConfig = field(default_factory=DataPrepConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    sam3d: SAM3DConfig = field(default_factory=SAM3DConfig)
    unidepth: UniDepthConfig = field(default_factory=UniDepthConfig)
    openpose: OpenPoseConfig = field(default_factory=OpenPoseConfig)
    smplh: SMPLHConfig = field(default_factory=SMPLHConfig)
    depth_align: DepthAlignConfig = field(default_factory=DepthAlignConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
