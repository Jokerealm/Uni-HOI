"""
Structured dataclass config for the offline preprocessing pipeline.
Registered with Hydra ConfigStore so that `preprocess.yaml` is validated.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SAM3Config:
    model_dir: str = ""
    text_prompts: List[str] = field(default_factory=lambda: ["human", "object"])
    score_threshold: float = 0.5
    mask_threshold: float = 0.5


@dataclass
class SAM3DConfig:
    checkpoint: str = ""
    mhr_model: str = ""
    config_yaml: str = ""
    image_size: List[int] = field(default_factory=lambda: [512, 512])


@dataclass
class UniDepthConfig:
    model_dir: str = ""
    backbone: str = "vitl14"


@dataclass
class OpenPoseConfig:
    model_dir: str = ""


@dataclass
class SMPLHConfig:
    model_dir: str = ""


@dataclass
class DepthAlignConfig:
    method: str = "median"


@dataclass
class MaskingConfig:
    dilate_kernel_size: int = 15
    dilate_iterations: int = 2
    contact_radius: int = 10
    gaussian_blur_ksize: int = 11
    gaussian_blur_sigma: float = 5.0


@dataclass
class PreprocessConfig:
    """Top-level config for preprocess.py"""
    base_weights_dir: str = "/data4/guanz/coding/HDM/model"
    project_root: str = "/data4/guanz/coding/HDM"
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    output_subdir: str = "processed"
    max_frames: Optional[int] = None
    device: str = "cuda"

    sam3: SAM3Config = field(default_factory=SAM3Config)
    sam3d: SAM3DConfig = field(default_factory=SAM3DConfig)
    unidepth: UniDepthConfig = field(default_factory=UniDepthConfig)
    openpose: OpenPoseConfig = field(default_factory=OpenPoseConfig)
    smplh: SMPLHConfig = field(default_factory=SMPLHConfig)
    depth_align: DepthAlignConfig = field(default_factory=DepthAlignConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
