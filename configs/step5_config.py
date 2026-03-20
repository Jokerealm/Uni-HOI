"""
Step 5: Strongly-typed dataclass configs for Full Pipeline Integration
& End-to-End Validation.

Aggregates all step configs and adds evaluation / output settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EvalMetricsConfig:
    """Config for evaluation metrics computation."""
    # Chamfer Distance thresholds
    cd_thresholds: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05])
    # Number of points to sample for CD computation
    num_eval_points: int = 4096
    # Acceleration error
    compute_acceleration: bool = True


@dataclass
class VisualizationConfig:
    """Config for result visualization outputs."""
    save_rendered_images: bool = True
    save_overlay_keypoints: bool = True
    save_novel_views: bool = True
    novel_view_angles: List[float] = field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0])
    save_video: bool = True
    video_fps: int = 24


@dataclass
class Step5PipelineConfig:
    """Aggregated config for the Step 5 end-to-end pipeline."""
    project_root: str = "/data4/guanz/coding/HDM"
    # Data paths (shared across all steps)
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    device: str = "cuda"

    # Step subdirectories
    processed_subdir: str = "processed"
    amodal_subdir: str = "amodal"
    gs_init_subdir: str = "gs_init"
    joint_opt_subdir: str = "joint_opt"
    output_subdir: str = "final_output"

    # Training
    image_height: int = 256
    image_width: int = 256
    focal: float = 500.0

    # Checkpoint
    checkpoint_dir: str = ""  # auto-resolved to outputs/runs/<timestamp>/
    resume_checkpoint: str = ""

    # Sub-configs
    eval: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)
    vis: VisualizationConfig = field(default_factory=VisualizationConfig)
