"""
Step 4 Pipeline Module: Multi-Region Contact-Aware Joint 3DGS Optimization.

Thin wrapper around scripts/step4_joint_optimization.py that exposes
a clean Pipeline class interface consistent with Steps 1-3.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.step4_config import Step4PipelineConfig
from scripts.step4_joint_optimization import run_step4_pipeline


class Step4Pipeline:
    """Pipeline wrapper for Step 4 joint 3DGS optimization."""

    def __init__(self, cfg: Step4PipelineConfig):
        self.cfg = cfg

    def run(self) -> str:
        """
        Execute the full Step 4 optimization.
        Returns the output directory path.
        """
        print("=" * 60)
        print("[Step4] Multi-Region Contact-Aware Joint 3DGS Optimization")
        print("=" * 60)
        run_step4_pipeline(self.cfg)
        output_dir = os.path.join(
            self.cfg.input_dir, self.cfg.video_name, self.cfg.output_subdir
        )
        return output_dir
