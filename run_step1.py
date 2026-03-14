#!/usr/bin/env python3
"""
Step 1 Entry Point: Spatial Alignment & Multi-Region Mask Extraction.

Usage:
    # Default (sample_data):
    CUDA_VISIBLE_DEVICES=0 python run_step1.py

    # Override input path for full dataset:
    CUDA_VISIBLE_DEVICES=0 python run_step1.py \
        data_prep.input_dir=/data4/guanz/data/Behave \
        data_prep.video_name=Date03_Sub03_chairwood

    # Limit frames for quick test:
    CUDA_VISIBLE_DEVICES=0 python run_step1.py data_prep.max_frames=5
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf

from configs.step1_config import Step1PipelineConfig
from pipeline.step1_pipeline import Step1Pipeline


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Convert OmegaConf -> dataclass
    pipeline_cfg = Step1PipelineConfig(
        base_weights_dir=cfg.base_weights_dir,
        project_root=cfg.project_root,
    )
    # Override from Hydra config
    dp = cfg.data_prep
    pipeline_cfg.data_prep.input_dir = dp.input_dir
    pipeline_cfg.data_prep.video_name = dp.video_name
    pipeline_cfg.data_prep.output_subdir = dp.output_subdir
    pipeline_cfg.data_prep.max_frames = dp.get("max_frames", None)
    pipeline_cfg.data_prep.device = dp.device

    pipeline_cfg.sam3.model_dir = cfg.sam3.model_dir
    pipeline_cfg.sam3.text_prompts = list(cfg.sam3.text_prompts)
    pipeline_cfg.sam3.score_threshold = cfg.sam3.score_threshold
    pipeline_cfg.sam3.mask_threshold = cfg.sam3.mask_threshold

    pipeline_cfg.sam3d.checkpoint = cfg.sam3d.checkpoint
    pipeline_cfg.sam3d.mhr_model = cfg.sam3d.mhr_model
    pipeline_cfg.sam3d.config_yaml = cfg.sam3d.config_yaml

    pipeline_cfg.unidepth.model_dir = cfg.unidepth.model_dir
    pipeline_cfg.unidepth.backbone = cfg.unidepth.backbone

    pipeline_cfg.openpose.model_dir = cfg.openpose.model_dir
    pipeline_cfg.smplh.model_dir = cfg.smplh.model_dir

    mcfg = cfg.masking
    pipeline_cfg.masking.dilate_kernel_size = mcfg.dilate_kernel_size
    pipeline_cfg.masking.dilate_iterations = mcfg.dilate_iterations
    pipeline_cfg.masking.contact_radius = mcfg.contact_radius
    pipeline_cfg.masking.gaussian_blur_ksize = mcfg.gaussian_blur_ksize
    pipeline_cfg.masking.gaussian_blur_sigma = mcfg.gaussian_blur_sigma

    # Run
    pipeline = Step1Pipeline(pipeline_cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
