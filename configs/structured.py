import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable
import os.path as osp

from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir


@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './outputs/${run.name}/single'


@dataclass
class RunConfig:
    name: str = 'debug'
    job: str = 'train'
    mixed_precision: str = 'fp16'  # 'no'
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = True
    vis_before_training: bool = True
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 100_000
    checkpoint_freq: int = 1_000
    val_freq: int = 5_000
    vis_freq: int = 5_000
    # vis_freq: int = 10_000
    log_step_freq: int = 20
    print_step_freq: int = 100

    # config to run demo
    stage1_name: str = 'stage1'     # experiment name to the stage 1 model
    stage2_name: str = 'stage2'     # experiment name to the stage 2 model
    image_path: str = ''            # the path to the images for running demo, can be a single file or a glob pattern
    share: bool = False             # whether to run gradio with a temporal public url or not
    input_cls: str = 'general'

    # abs path to working dir
    code_dir_abs: str = osp.dirname(osp.dirname(osp.abspath(__file__)))

    # Inference configs
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    # num_sample_batches: Optional[int] = None
    num_sample_batches: Optional[int] = 2000  # XH: change to 2
    sample_from_ema: bool = False
    sample_save_evolutions: bool = False  # temporarily set by default
    save_name: str = 'sample'  # XH: additional save name
    redo: bool = False

    # for parallel sampling in slurm
    batch_start: int = 0
    batch_end: Optional[int] = None

    # Training configs
    freeze_feature_model: bool = True

    # Coloring training configs
    coloring_training_noise_std: float = 0.0
    coloring_sample_dir: Optional[str] = None

    sample_mode: str = 'sample'  # whether from noise or from some intermediate steps
    sample_noise_step: int = 500  # add noise to GT up to some steps, and then denoise
    sample_save_gt: bool = True


@dataclass
class LoggingConfig:
    wandb: bool = True  # 启用wandb
    wandb_project: str = 'hdm-training'  # 可以自定义项目名称



@dataclass
class PointCloudProjectionModelConfig:
    # Feature extraction arguments
    image_size: int = '${dataset.image_size}'
    image_feature_model: str = 'vit_base_patch16_224_mae' # or 'vit_small_patch16_224_msn' or 'identity'
    use_local_colors: bool = True
    use_local_features: bool = True
    use_global_features: bool = False
    use_mask: bool = True
    use_distance_transform: bool = True

    # Point cloud data arguments. Note these are here because the processing happens
    # inside the model, rather than inside the dataset.
    scale_factor: float = "${dataset.scale_factor}"
    colors_mean: float = 0.5
    colors_std: float = 0.5
    color_channels: int = 3
    predict_shape: bool = True
    predict_color: bool = False

    # added by XH
    load_sample_init: bool = False  # load init samples from file
    sample_init_scale: float = 1.0  # scale the initial pc samples
    test_init_with_gtpc: bool = False  # test time init samples with GT samples
    consistent_center: bool = True  # use consistent center prediction by CCD-3DR
    voxel_resolution_multiplier: float = 1  # increase network voxel resolution

    # predict binary segmentation
    predict_binary: bool = False # True for stage 1 model, False for others
    lw_binary: float = 3.0  # to have roughly the same magnitude of the binary segmentation loss
    # for separate model
    binary_training_noise_std: float = 0.1  # from github doc for predicting color
    self_conditioning: bool = False

@dataclass
class PVCNNAEModelConfig(PointCloudProjectionModelConfig):
    "my own model config, must inherit parent class"
    model_name: str = 'pvcnn-ae'
    latent_dim: int = 1024
    num_dec_blocks: int = 6
    block_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_points: int = 1500
    bottleneck_dim: int = -1 # the input dim to the last MLP layer

@dataclass
class PointCloudDiffusionModelConfig(PointCloudProjectionModelConfig):
    model_name: str = 'pc2-diff-ho'  # default as behave

    # Diffusion arguments
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom'
    dm_pred_type: str = 'epsilon'  # diffusion model prediction type, sample (x0) or noise
    ddim_eta: float = 1.0  # DDIM eta parameter: 0 is the default one which does deterministic generation

    # Point cloud model arguments
    point_cloud_model: str = 'pvcnn'
    point_cloud_model_embed_dim: int = 64

    dataset_type: str = '${dataset.type}'

@dataclass
class CrossAttnHOModelConfig(PointCloudDiffusionModelConfig):
    model_name: str = 'diff-ho-attn'

    attn_type: str = 'coord3d+posenc-learnable'
    attn_weight: float = 1.0
    point_visible_test: str = 'combine'  # To compute point visibility: use all points or only human/object points



@dataclass
class PointCloudColoringModelConfig(PointCloudProjectionModelConfig):
    # Projection arguments
    predict_shape: bool = False
    predict_color: bool = True

    # Point cloud model arguments
    point_cloud_model: str = 'pvcnn'
    point_cloud_model_layers: int = 1
    point_cloud_model_embed_dim: int = 64


@dataclass
class DatasetConfig:
    type: str


@dataclass
class PointCloudDatasetConfig(DatasetConfig):
    eval_split: str = 'val'
    max_points: int = 16_384
    image_size: int = 224
    scale_factor: float = 1.0
    restrict_model_ids: Optional[List] = None  # for only running on a subset of data points


@dataclass
class CO3DConfig(PointCloudDatasetConfig):
    type: str = 'co3dv2'
    # root: str = os.getenv('CO3DV2_DATASET_ROOT')
    root: str = "/BS/xxie-2/work/co3d/hydrant"
    category: str = 'hydrant'
    subset_name: str = 'fewview_dev'
    mask_images: bool = '${model.use_mask}'


@dataclass
class ShapeNetR2N2Config(PointCloudDatasetConfig):
    # added by XH
    fix_sample: bool = True
    category: str = 'chair'

    type: str = 'shapenet_r2n2'
    root: str = "/BS/chiban2/work/data_shapenet/ShapeNetCore.v1"
    r2n2_dir: str = "/BS/databases20/3d-r2n2"
    shapenet_dir: str = "/BS/chiban2/work/data_shapenet/ShapeNetCore.v1"
    preprocessed_r2n2_dir: str = "${dataset.root}/r2n2_preprocessed_renders"
    splits_file: str = "${dataset.root}/r2n2_standard_splits_from_ShapeNet_taxonomy.json"
    # splits_file: str = "${dataset.root}/pix2mesh_splits_val05.json"  # <-- incorrect
    scale_factor: float = 7.0
    point_cloud_filename: str = 'pointcloud_r2n2.npz'  # should use 'pointcloud_mesh.npz'



@dataclass
class BehaveDatasetConfig(PointCloudDatasetConfig):
    # added by XH
    type: str = 'behave'

    fix_sample: bool = True
    behave_dir: str = "/data4/guanz/data/Behave/sequences" # BEHAVE sequences path
    procigen_dir: str = '/data4/guanz/data/ProciGen' # ProciGen path (no trailing slash)
    split_file: str = "/data4/guanz/data/train-procigen-test-behave.pkl" # dataset split file
    scale_factor: float = 7.0  # use the same as shapenet
    sample_ratio_hum: float = 0.5
    image_size: int = 224

    normalize_type: str = 'comb'
    smpl_type: str = 'gt'  # use which SMPL mesh to obtain normalization parameters
    test_transl_type: str = 'norm'

    load_corr_points: bool = False  # load autoencoder points for object and SMPL
    uniform_obj_sample: bool = False

    # configs for direct translation prediction
    bkg_type: str = 'none'
    bbox_params: str = 'none'
    ho_segm_pred_path: Optional[str] = None
    use_gt_transl: bool = False

    cam_noise_std: float = 0. # add noise to the camera pose
    sep_same_crop: bool = False # use same input image crop to separate models
    aug_blur: float = 0. # blur augmentation

    std_coverage: float=3.5 # a heuristic value to estimate translation

    v2v_path: str = '' # object v2v corr path

    # CARI4D-style spatial downsampling: downsample raw 2K images by this factor
    # before crop. Reduces memory/compute while preserving detail for 224×224 crops.
    scale_ratio: int = 2
    # CARI4D-style bbox expansion factor (1.1 = 10% padding around H+O bbox)
    bbox_expand: float = 1.1

@dataclass
class ShapeDatasetConfig(BehaveDatasetConfig):
    "the dataset to train AE for aligned shapes"
    type: str = 'shape'
    fix_sample: bool = False
    split_file: str = "/BS/xxie-2/work/pc2-diff/experiments/splits/shapes-chair.pkl"


# TODO
@dataclass
class ShapeNetNMRConfig(PointCloudDatasetConfig):
    type: str = 'shapenet_nmr'
    shapenet_nmr_dir: str = "/work/lukemk/machine-learning-datasets/3d-reconstruction/ShapeNet_NMR/NMR_Dataset"
    synset_names: str = 'chair'  # comma-separated or 'all'
    augmentation: str = 'all'
    scale_factor: float = 7.0


@dataclass
class AugmentationConfig:
    # need to specify the variable type in order to define it properly
    max_radius: int = 0  # generate a random square to mask object, this is the radius for the square in pixel size, zero means no occlusion


@dataclass
class DataloaderConfig:
    # batch_size: int = 8  # 2 for debug
    batch_size: int = 16
    num_workers: int = 14  # 0 for debug # suggested by accelerator for gpu20


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    rgb_weight: float = 1.0
    consistency_weight: float = 1.0


@dataclass
class CheckpointConfig:
    resume: Optional[str] = "test"
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True


@dataclass
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    # # From Diffusers EMA (should probably switch)
    # ema_inv_gamma: float = 1.0
    # ema_power: float = 0.75
    # ema_max_decay: float = 0.9999
    decay: float = 0.999
    update_every: int = 20


@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 3e-4
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=1e-6,
    ))


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps="${run.max_steps}",
    ))


###############################################################################
# Phase 2: Dual-Branch Flow Matching model config
###############################################################################
@dataclass
class FlowMatchingModelConfig:
    model_name: str = 'flow-matching'
    # Issue 6: video_channels is RGB-only output (3), input is 4 (RGB+mask per stream)
    video_channels: int = 3            # output channels (RGB only)
    video_input_channels: int = 4      # input channels per stream (RGB + individual mask)
    video_patch_size: int = 2
    point_channels: int = 14           # xyz(3)+rot(4)+scale(3)+opacity(1)+SH(3)
    mask_channels: int = 2             # human mask + object mask (Issue 5: cross-attn condition)
    # Issue 7: smaller model — dim=384, depth=8 → ~60-65M with temporal attn
    dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    cond_dim: int = 192
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    lambda_video: float = 1.0
    lambda_3d: float = 1.0
    # Data shape
    num_frames: int = 4
    video_h: int = 32
    video_w: int = 32
    num_points: int = 256
    # Real dataset (preprocessed Phase 1 outputs)
    preprocessed_dir: str = 'preprocessed/phase1'
    consolidated_dir: str = 'preprocessed/phase1_consolidated'
    cache_file: str = 'preprocessed/phase1_cache.pt'
    use_real_data: bool = True


###############################################################################
# Phase 3: Joint 3DGS Optimization config
###############################################################################
@dataclass
class Joint3DGSModelConfig:
    model_name: str = 'joint-3dgs'
    # Data (either preprocessed .pt or raw image dirs)
    preprocessed_pt: str = ''          # path to .pt from preprocess_phase1.py
    phase2_output: str = ''            # path to Phase 2 output .pt for initialization
    frames_dir: str = ''
    masks_human_dir: str = ''
    masks_object_dir: str = ''
    image_height: int = 256
    image_width: int = 256
    # Model
    num_points_human: int = 4096
    num_points_object: int = 2048
    focal: float = 500.0
    # Training
    num_iters: int = 5000
    save_every: int = 1000
    # Learning rates
    lr_xyz: float = 1.6e-4
    lr_opacity: float = 5e-2
    lr_scaling: float = 5e-3
    lr_rotation: float = 1e-3
    lr_color: float = 2.5e-3
    # SE(3) registration learning rates
    lr_se3_translation: float = 1e-3
    lr_se3_rotation: float = 1e-4
    # Multi-region rendering loss weights
    w_visible: float = 1.0
    w_primary_occ: float = 0.3
    w_secondary_occ: float = 0.05
    lambda_ssim: float = 0.2
    # Contact loss
    lambda_contact: float = 0.5
    enable_contact: bool = True
    # 2D projection loss
    lambda_j2d: float = 0.1
    enable_j2d: bool = True
    j2d_conf_threshold: float = 0.3
    # Penetration loss (volumetric SMPL SDF)
    lambda_pen: float = 1.0
    enable_penetration: bool = True
    sdf_resolution: int = 64
    sdf_padding: float = 0.1
    # Temporal smoothness loss on framewise SE(3) trajectories
    lambda_acc: float = 0.5
    enable_temporal: bool = True
    # Legacy basic loss weights (kept for backward compat)
    lambda_occ: float = 0.01
    epsilon: float = 0.005
    # Step 1 processed data directory (soft masks, SMPL-H, keypoints)
    processed_dir: str = ''


###############################################################################
# Preprocessed dataset config (pure file-read after `main.py run.job=step1`)
###############################################################################
@dataclass
class PreprocessedDatasetConfig(PointCloudDatasetConfig):
    type: str = 'preprocessed'
    root_dir: str = './sample_data/test_video'
    processed_subdir: str = 'processed'
    image_size: int = 256


@dataclass
class ProjectConfig:
    run: RunConfig
    logging: LoggingConfig
    dataset: PointCloudDatasetConfig
    augmentations: AugmentationConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    model: PointCloudProjectionModelConfig
    ema: ExponentialMovingAverageConfig
    checkpoint: CheckpointConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {'logging': 'default'},
        {'model': 'ho-attn'},
        # {'dataset': 'co3d'},
        {'dataset': 'behave'},
        {'augmentations': 'default'},
        {'dataloader': 'default'},
        {'ema': 'default'},
        {'loss': 'default'},
        {'checkpoint': 'default'},
        {'optimizer': 'adam'}, # default adamw
        {'scheduler': 'cosine'},
    ])


cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='diffrec', node=PointCloudDiffusionModelConfig)
cs.store(group='model', name='coloring_model', node=PointCloudColoringModelConfig)
cs.store(group='model', name='ho-attn', node=CrossAttnHOModelConfig)
cs.store(group='model', name='pvcnn-ae', node=PVCNNAEModelConfig)
cs.store(group='model', name='flow-matching', node=FlowMatchingModelConfig)
cs.store(group='model', name='joint-3dgs', node=Joint3DGSModelConfig)
cs.store(group='dataset', name='co3d', node=CO3DConfig)
# TODO
cs.store(group='dataset', name='shapenet_r2n2', node=ShapeNetR2N2Config)
cs.store(group='dataset', name='behave', node=BehaveDatasetConfig)
cs.store(group='dataset', name='preprocessed', node=PreprocessedDatasetConfig)
cs.store(group='dataset', name='shape', node=ShapeDatasetConfig)
# cs.store(group='dataset', name='shapenet_nmr', node=ShapeNetNMRConfig)
cs.store(group='augmentations', name='default', node=AugmentationConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='loss', name='default', node=LossConfig)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='configs', node=ProjectConfig)

# Step 3.5: Metric Alignment Bridge
from configs.alignment_config import MetricAlignmentConfig, AlignmentPipelineConfig
cs.store(group='alignment', name='default', node=MetricAlignmentConfig)
