"""
Step 2 joint Flow Matching pipeline.

This wrapper runs the new latent-space Hunyuan3D-2 ControlNet inference path
and writes:
  - `amodal/object_amodal/*`
  - `amodal/human_amodal/*` (segmented human proxy for downstream lifting)
  - `gs_init/G_o.pt`
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

from configs.step2_config import Step2PipelineConfig
from infer_fm import run_joint_fm_inference


class JointFlowMatchingPipeline:
    def __init__(self, cfg: Step2PipelineConfig) -> None:
        self.cfg = cfg

    def run(self) -> Dict[str, object]:
        if not self.cfg.fm.checkpoint:
            raise ValueError("`fm.checkpoint` must be set for the joint Flow Matching Step-2 pipeline.")

        print("=" * 60)
        print("[Step2] Joint Video-3D Flow Matching Inference")
        print(f"  Input Dir : {self.cfg.input_dir}")
        print(f"  Video     : {self.cfg.video_name}")
        print(f"  Checkpoint: {self.cfg.fm.checkpoint}")
        print("=" * 60)

        args = SimpleNamespace(
            input_dir=self.cfg.input_dir,
            video_name=self.cfg.video_name,
            checkpoint=self.cfg.fm.checkpoint,
            processed_subdir=self.cfg.processed_subdir,
            output_subdir=self.cfg.output_subdir,
            gs_output_subdir=self.cfg.gs_output_subdir,
            max_frames=0 if self.cfg.max_frames is None else int(self.cfg.max_frames),
            num_ode_steps=self.cfg.fm.num_ode_steps,
            num_points=self.cfg.fm.num_points,
            video_h=self.cfg.fm.video_h,
            video_w=self.cfg.fm.video_w,
            prior_noise_std=self.cfg.fm.prior_noise_std,
            clamp_visible_rgb=self.cfg.fm.clamp_visible_rgb,
            save_frames=self.cfg.fm.save_frames,
            save_fps=self.cfg.fm.save_fps,
            background_value=self.cfg.fm.background_value,
            precision=self.cfg.fm.precision,
            human_branch_mode=self.cfg.fm.human_branch_mode,
            seed=self.cfg.fm.seed,
            device=self.cfg.device,
        )
        return run_joint_fm_inference(args)


__all__ = ["JointFlowMatchingPipeline"]
