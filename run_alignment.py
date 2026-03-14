#!/usr/bin/env python3
"""
Step 3.5 Entry Point: Metric Alignment Bridge.

Transforms Step 3 canonical-space 3DGS into metric physical space using
observed depth, masks, and (optionally) SMPL-H translation from Preprocess.

Usage:
    # Default (sample_data):
    python run_alignment.py

    # Override paths:
    python run_alignment.py \
        data_prep.input_dir=/data4/guanz/data/Behave \
        data_prep.video_name=Date03_Sub03_chairwood

    # Disable alignment (pass-through):
    python run_alignment.py alignment.enabled=false

    # Use depth-unproject for human instead of SMPL-H:
    python run_alignment.py alignment.human_align_strategy=unproject
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf

from configs.alignment_config import AlignmentPipelineConfig, MetricAlignmentConfig
from pipeline.metric_alignment_bridge import MetricAlignmentBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    dp = cfg.data_prep
    acfg_raw = cfg.get("alignment", {})

    alignment_cfg = MetricAlignmentConfig(
        enabled=acfg_raw.get("enabled", True),
        depth_max=acfg_raw.get("depth_max", 10.0),
        depth_min_pixels=acfg_raw.get("depth_min_pixels", 50),
        percentile=acfg_raw.get("percentile", 90.0),
        scale_eps=acfg_raw.get("scale_eps", 1e-4),
        scale_default=acfg_raw.get("scale_default", 1.0),
        scale_min=acfg_raw.get("scale_min", 0.01),
        scale_max=acfg_raw.get("scale_max", 100.0),
        human_align_strategy=acfg_raw.get("human_align_strategy", "smplh"),
        validate_transform=acfg_raw.get("validate_transform", True),
    )

    step4_cfg = cfg.get("step4", {})
    pipeline_cfg = AlignmentPipelineConfig(
        project_root=cfg.project_root,
        input_dir=dp.input_dir,
        video_name=dp.video_name,
        processed_subdir=dp.get("output_subdir", "processed"),
        gs_init_subdir=cfg.get("step3", {}).get("output_subdir", "gs_init"),
        output_subdir="gs_aligned",
        device="cpu",
        image_height=step4_cfg.get("image_height", 256),
        image_width=step4_cfg.get("image_width", 256),
        focal=step4_cfg.get("focal", 500.0),
        alignment=alignment_cfg,
    )

    bridge = MetricAlignmentBridge(pipeline_cfg)
    bridge.run()


if __name__ == "__main__":
    main()
