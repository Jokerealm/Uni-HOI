#!/usr/bin/env python3
"""
Step 3 Entry Point: Flow Matching based 3D Lifting.

Loads amodal completion videos from Step 2 and generates initial 3DGS
parameters (G_o, G_h) via ODE sampling through the trained flow matching model.

Usage:
    # Default (sample_data, no checkpoint — smoke test):
    CUDA_VISIBLE_DEVICES=0 python run_step3.py

    # With trained checkpoint:
    CUDA_VISIBLE_DEVICES=0 python run_step3.py \
        fm.checkpoint=outputs/flow-matching-phase2-real/single/checkpoint-latest.pth

    # Override input path for full dataset:
    CUDA_VISIBLE_DEVICES=0 python run_step3.py \
        data_prep.input_dir=/data4/guanz/data/Behave \
        data_prep.video_name=Date03_Sub03_chairwood

    # Adjust ODE steps for speed vs quality:
    CUDA_VISIBLE_DEVICES=0 python run_step3.py fm.num_ode_steps=100
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf

from configs.step3_config import Step3PipelineConfig, FlowMatchingInferenceConfig
from pipeline.step3_flow_matching_lifting import Step3Pipeline


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Build strongly-typed config from Hydra
    fm_cfg = FlowMatchingInferenceConfig(
        checkpoint=cfg.fm.checkpoint,
        video_channels=cfg.fm.video_channels,
        video_input_channels=cfg.fm.video_input_channels,
        point_channels=cfg.fm.point_channels,
        mask_channels=cfg.fm.mask_channels,
        dim=cfg.fm.dim,
        depth=cfg.fm.depth,
        num_heads=cfg.fm.num_heads,
        cond_dim=cfg.fm.cond_dim,
        num_ode_steps=cfg.fm.num_ode_steps,
        num_frames=cfg.fm.num_frames,
        video_h=cfg.fm.video_h,
        video_w=cfg.fm.video_w,
        num_points=cfg.fm.num_points,
    )

    dp = cfg.data_prep
    pipeline_cfg = Step3PipelineConfig(
        project_root=cfg.project_root,
        input_dir=dp.input_dir,
        video_name=dp.video_name,
        amodal_subdir=cfg.get("amodal", {}).get("output_subdir", "amodal"),
        output_subdir=cfg.get("step3", {}).get("output_subdir", "gs_init"),
        device=dp.device,
        fm=fm_cfg,
    )

    pipeline = Step3Pipeline(pipeline_cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
