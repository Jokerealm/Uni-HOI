"""
Step 2 pipeline wrapper for the dual-branch co-generative Flow Matching backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

from configs.step2_config import Step2PipelineConfig
from infer_dual_branch_fm import run_dual_branch_inference


class DualBranchFlowMatchingPipeline:
    def __init__(self, cfg: Step2PipelineConfig) -> None:
        self.cfg = cfg

    def run(self) -> Dict[str, object]:
        if not self.cfg.fm.checkpoint:
            raise ValueError("`fm.checkpoint` must be set for the dual-branch Flow Matching Step-2 pipeline.")

        print("=" * 60)
        print("[Step2] Dual-Branch Co-Generative Flow Matching Inference")
        print(f"  Input Dir : {self.cfg.input_dir}")
        print(f"  Video     : {self.cfg.video_name}")
        print(f"  Checkpoint: {self.cfg.fm.checkpoint}")
        print("=" * 60)

        args = SimpleNamespace(
            input_dir=self.cfg.input_dir,
            video_name=self.cfg.video_name,
            checkpoint=self.cfg.fm.checkpoint,
            processed_subdir=self.cfg.processed_subdir,
            gs_subdir=self.cfg.gs_output_subdir,
            output_subdir=self.cfg.output_subdir,
            gs_output_subdir=self.cfg.gs_output_subdir,
            num_ode_steps=self.cfg.fm.num_ode_steps,
            prior_noise_std=self.cfg.fm.prior_noise_std,
            save_frames=self.cfg.fm.save_frames,
            save_fps=self.cfg.fm.save_fps,
            seed=self.cfg.fm.seed,
            device=self.cfg.device,
            clamp_visible_rgb=self.cfg.fm.clamp_visible_rgb,
        )
        return run_dual_branch_inference(args)


__all__ = ["DualBranchFlowMatchingPipeline"]
