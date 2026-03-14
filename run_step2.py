#!/usr/bin/env python3
"""
Step 2 Entry Point: Amodal Video Completion via ProPainter.

Uses masks from Step 1 to inpaint occluded regions, producing:
  - V_o_amodal: clean object video (human inpainted out)
  - V_h_amodal: clean human video (object inpainted out)

Usage:
    # Default (sample_data):
    CUDA_VISIBLE_DEVICES=0 python run_step2.py

    # Override input path for full dataset:
    CUDA_VISIBLE_DEVICES=0 python run_step2.py \
        data_prep.input_dir=/data4/guanz/data/Behave \
        data_prep.video_name=Date03_Sub03_chairwood

    # Disable fp16:
    CUDA_VISIBLE_DEVICES=0 python run_step2.py propainter.fp16=false
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf

from configs.step2_config import Step2PipelineConfig, ProPainterConfig
from pipeline.step2_amodal_completion import Step2Pipeline


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Build strongly-typed config from Hydra
    pp_cfg = ProPainterConfig(
        weights_dir=cfg.propainter.weights_dir,
        mask_dilation=cfg.propainter.mask_dilation,
        ref_stride=cfg.propainter.ref_stride,
        neighbor_length=cfg.propainter.neighbor_length,
        subvideo_length=cfg.propainter.subvideo_length,
        raft_iter=cfg.propainter.raft_iter,
        fp16=cfg.propainter.fp16,
        save_frames=cfg.propainter.save_frames,
        save_fps=cfg.propainter.save_fps,
    )

    dp = cfg.data_prep
    pipeline_cfg = Step2PipelineConfig(
        base_weights_dir=cfg.base_weights_dir,
        project_root=cfg.project_root,
        input_dir=dp.input_dir,
        video_name=dp.video_name,
        processed_subdir=dp.output_subdir,
        output_subdir=cfg.amodal.output_subdir,
        device=dp.device,
        propainter=pp_cfg,
    )

    pipeline = Step2Pipeline(pipeline_cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
