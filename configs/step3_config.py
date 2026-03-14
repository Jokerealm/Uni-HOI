"""
Step 3: Strongly-typed dataclass configs for Flow Matching based 3D Lifting.
Converts amodal completion videos (from Step 2) into initial 3DGS representations
via ODE sampling with the trained dual-branch flow matching model.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlowMatchingInferenceConfig:
    """Config for the pre-trained Flow Matching model used for 3D lifting."""
    # Path to the trained checkpoint (from Phase 2 training)
    checkpoint: str = ""
    # Model architecture params (must match training config)
    video_channels: int = 3
    video_input_channels: int = 4
    point_channels: int = 14       # xyz(3)+rot(4)+scale(3)+opacity(1)+SH(3)
    mask_channels: int = 2
    dim: int = 384
    depth: int = 8
    num_heads: int = 6
    cond_dim: int = 192
    # ODE sampling
    num_ode_steps: int = 50
    # Data shape
    num_frames: int = 4
    video_h: int = 32
    video_w: int = 32
    num_points: int = 256


@dataclass
class Step3PipelineConfig:
    """Aggregated config for the Step 3 flow-matching 3D lifting pipeline."""
    project_root: str = "/data4/guanz/coding/HDM"
    # Input: amodal video frames from Step 2
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    amodal_subdir: str = "amodal"
    # Output
    output_subdir: str = "gs_init"
    # Device
    device: str = "cuda"
    # Flow matching model config
    fm: FlowMatchingInferenceConfig = field(default_factory=FlowMatchingInferenceConfig)
