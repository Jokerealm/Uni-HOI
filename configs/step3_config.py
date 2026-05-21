"""
Step 3: Strongly-typed dataclass configs for 3D lifting.

Uses Hunyuan3D-2 (zero-shot image-to-3D) to generate initial meshes from
amodal completion frames, then samples 3DGS parameters from the mesh surface.
Also hosts the shared Flow Matching inference config used by the alternate
joint amodal-video + 3D generation path.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
class FlowMatchingInferenceConfig:
    """Shared inference config for video+3D Flow Matching generation."""
    checkpoint: str = ""
    model_type: str = "dual_branch_cogenerative"
    video_channels: int = 3
    video_input_channels: int = 4
    point_channels: int = 14
    mask_channels: int = 2
    dim: int = 384
    depth: int = 8
    num_heads: int = 6
    cond_dim: int = 192
    num_ode_steps: int = 50
    num_frames: int = 12
    video_h: int = 256
    video_w: int = 256
    num_points: int = 4096
    prior_noise_std: float = 1.0
    clamp_visible_rgb: bool = True
    save_frames: bool = True
    save_fps: int = 24
    precision: str = "float32"
    background_value: float = 1.0
    human_branch_mode: str = "segmented_visible"
    seed: int = 42
    patch_size: int = 16
    hidden_dim: int = 512
    fusion_depth: int = 8
    num_human_gaussians: int = 850
    num_object_gaussians: int = 850
    num_joints: int = 22
    contact_dim: int = 4


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
    # Fallback camera used by the alignment bridge when ROI intrinsics are missing
    image_height: int = 256
    image_width: int = 256
    focal: float = 500.0
    # Frame selection: which amodal frame to use for 3D generation
    # "middle" = middle frame, "first" = first frame, int = specific index
    frame_selection: str = "middle"
    # Hunyuan3D-2 model config
    hy3d: Hunyuan3DConfig = field(default_factory=Hunyuan3DConfig)
    # Shared FM config for alternate Flow Matching pipeline
    fm: FlowMatchingInferenceConfig = field(default_factory=FlowMatchingInferenceConfig)
    # Whether to run metric alignment after 3D generation
    run_alignment: bool = True
    # Optional overrides copied from the global Hydra `alignment` node
    alignment: Optional[Dict[str, Any]] = None
