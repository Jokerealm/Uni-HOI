"""
Step 3: Strongly-typed dataclass configs for 3D Lifting & Metric Alignment.

Uses Hunyuan3D-2 (zero-shot image-to-3D) to generate initial meshes from
amodal completion frames, then samples 3DGS parameters from the mesh surface.
No training involved — pure inference.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Hunyuan3DConfig:
    """Config for the Hunyuan3D-2 shape generation model."""
    # HuggingFace model ID or local path
    model_path: str = "tencent/Hunyuan3D-2"
    # Subfolder within the model repo (for different model variants)
    subfolder: str = "hunyuan3d-dit-v2-0"
    # Inference parameters
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    octree_resolution: int = 384
    # Number of points to sample from the generated mesh for 3DGS init
    num_sample_points: int = 4096
    # Initial Gaussian scale (in normalized space)
    init_gaussian_scale: float = 0.01
    # Whether to remove background from input images
    remove_background: bool = True
    # Device and precision
    dtype: str = "float16"  # float16 or float32


@dataclass
class Step3PipelineConfig:
    """Aggregated config for the Step 3 zero-shot 3D lifting pipeline."""
    project_root: str = "/data4/guanz/coding/HDM"
    # Input: amodal video frames from Step 2
    input_dir: str = "./sample_data"
    video_name: str = "test_video"
    amodal_subdir: str = "amodal"
    processed_subdir: str = "processed"
    # Output
    output_subdir: str = "gs_init"
    # Device
    device: str = "cuda"
    # Frame selection: which amodal frame to use for 3D generation
    # "middle" = middle frame, "first" = first frame, int = specific index
    frame_selection: str = "middle"
    # Hunyuan3D-2 model config
    hy3d: Hunyuan3DConfig = field(default_factory=Hunyuan3DConfig)
    # Whether to run metric alignment after 3D generation
    run_alignment: bool = True
