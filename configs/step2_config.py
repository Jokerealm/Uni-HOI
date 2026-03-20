"""
Step 2: Strongly-typed dataclass configs for amodal video generation.
"""
from dataclasses import dataclass, field
from typing import Optional

from configs.step3_config import FlowMatchingInferenceConfig


@dataclass
class ProPainterConfig:
    """Config for ProPainter video inpainting model."""
    weights_dir: str = "/data4/guanz/coding/HDM/model/ProPainter/weights"
    # ProPainter inference hyper-parameters
    mask_dilation: int = 4
    ref_stride: int = 10
    neighbor_length: int = 10
    subvideo_length: int = 80
    raft_iter: int = 20
    fp16: bool = True
    save_frames: bool = True
    save_fps: int = 24


@dataclass
class Step2PipelineConfig:
    """Aggregated config for the Step 2 amodal video completion pipeline."""
    base_weights_dir: str = "/data4/guanz/coding/HDM/model"
    project_root: str = "/data4/guanz/coding/HDM"
    backend: str = "propainter"  # propainter | joint_flow_matching | dual_branch_flow_matching
    # Reuse step1 data_prep paths to locate input frames & masks
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    processed_subdir: str = "processed"
    output_subdir: str = "amodal"
    gs_output_subdir: str = "gs_init"
    max_frames: Optional[int] = None
    behave_cam_id: int = 1
    device: str = "cuda"
    propainter: ProPainterConfig = field(default_factory=ProPainterConfig)
    fm: FlowMatchingInferenceConfig = field(default_factory=FlowMatchingInferenceConfig)
