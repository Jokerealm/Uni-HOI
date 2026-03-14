"""
Step 3 Pipeline: Flow Matching based 3D Lifting.

Loads amodal completion videos from Step 2, runs ODE sampling through the
trained dual-branch flow matching model, and outputs initial 3DGS parameter
files (G_o and G_h) in their respective normalised coordinate systems.
"""
import os
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.step3_config import Step3PipelineConfig, FlowMatchingInferenceConfig


class Step3Pipeline:
    """Converts 2D amodal videos → initial 3D Gaussian Splatting parameters."""

    def __init__(self, cfg: Step3PipelineConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Resolve paths
        self.video_dir = os.path.join(cfg.input_dir, cfg.video_name)
        self.amodal_dir = os.path.join(self.video_dir, cfg.amodal_subdir)
        self.output_dir = os.path.join(self.video_dir, cfg.output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load model
        self.model = None
        self._euler_ode_sample = None
        self._load_model(cfg.fm)

    def _load_model(self, fm_cfg: FlowMatchingInferenceConfig):
        """Load the trained flow matching model from checkpoint."""
        from importlib.util import spec_from_file_location, module_from_spec

        code_dir = self.cfg.project_root

        # Load dual-branch flow matching module
        mod_path = os.path.join(code_dir, "model", "dual_branch_flow_matching.py")
        spec = spec_from_file_location("dual_branch_flow_matching", mod_path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Load PVCNN-based model
        pvcnn_path = os.path.join(code_dir, "model", "pvcnn_flow_matching.py")
        pvcnn_spec = spec_from_file_location("pvcnn_flow_matching", pvcnn_path)
        pvcnn_mod = module_from_spec(pvcnn_spec)
        pvcnn_spec.loader.exec_module(pvcnn_mod)

        # Instantiate model
        model = pvcnn_mod.PVCNNFlowMatchingModel(
            video_channels=fm_cfg.video_channels,
            video_input_channels=fm_cfg.video_input_channels,
            point_channels=fm_cfg.point_channels,
            mask_channels=fm_cfg.mask_channels,
        )

        # Load checkpoint if provided
        ckpt_path = fm_cfg.checkpoint
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"[Step3] Loading FM checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            # Strip DDP 'module.' prefix if present
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Step3]  Missing keys: {len(missing)}")
            if unexpected:
                print(f"[Step3]  Unexpected keys: {len(unexpected)}")
        else:
            print("[Step3] WARNING: No FM checkpoint provided or found. Using random weights.")
            print("         Results will be meaningless — provide a trained checkpoint for real use.")

        model = model.to(self.device)
        model.eval()
        self.model = model
        self._euler_ode_sample = mod.euler_ode_sample

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[Step3] Model loaded: {num_params / 1e6:.2f}M parameters")

    def _load_amodal_frames(self, branch: str) -> torch.Tensor:
        """
        Load amodal video frames for a branch ('human_amodal' or 'object_amodal').

        Returns: (T, 3, H, W) float tensor in [0, 1].
        """
        frames_dir = os.path.join(self.amodal_dir, branch, "frames")
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(
                f"[Step3] Amodal frames not found: {frames_dir}\n"
                f"        Run Step 2 first to generate amodal completion videos."
            )

        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not paths:
            raise FileNotFoundError(f"[Step3] No image files in {frames_dir}")

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1)     # (3, H, W)
            frames.append(t)

        video = torch.stack(frames, dim=0)  # (T_full, 3, H, W)
        print(f"[Step3] Loaded {branch}: {video.shape[0]} frames, "
              f"resolution {video.shape[2]}x{video.shape[3]}")
        return video

    def _prepare_video_input(self, video: torch.Tensor) -> torch.Tensor:
        """
        Subsample and resize video to match model expectations.

        Input:  (T_full, 3, H_orig, W_orig)
        Output: (1, T, C_in, H_model, W_model) — batch dim added
        """
        fm = self.cfg.fm
        T_full = video.shape[0]
        T_model = fm.num_frames

        # Subsample frames uniformly
        if T_full >= T_model:
            indices = torch.linspace(0, T_full - 1, T_model).long()
        else:
            # Repeat last frame if video is shorter than model expects
            indices = list(range(T_full)) + [T_full - 1] * (T_model - T_full)
            indices = torch.tensor(indices)

        video_sub = video[indices]  # (T_model, 3, H, W)

        # Resize to model resolution
        H_m, W_m = fm.video_h, fm.video_w
        if video_sub.shape[2] != H_m or video_sub.shape[3] != W_m:
            video_sub = F.interpolate(
                video_sub, size=(H_m, W_m), mode="bilinear", align_corners=False
            )

        # Model expects C_in = video_input_channels (4: RGB + mask channel)
        # For inference from amodal video, the mask channel is all-ones (fully visible)
        if fm.video_input_channels > 3:
            extra_ch = fm.video_input_channels - 3
            ones = torch.ones(
                video_sub.shape[0], extra_ch, H_m, W_m,
                dtype=video_sub.dtype,
            )
            video_sub = torch.cat([video_sub, ones], dim=1)

        return video_sub.unsqueeze(0)  # (1, T, C_in, H, W)

    def _prepare_mask_features(self, branch: str) -> torch.Tensor:
        """
        Build mask condition features for the model.

        For object branch: human_mask=0, object_mask=1
        For human branch: human_mask=1, object_mask=0
        """
        fm = self.cfg.fm
        T, H, W = fm.num_frames, fm.video_h, fm.video_w
        C_m = fm.mask_channels  # 2: [human_mask, object_mask]

        mask = torch.zeros(1, T, C_m, H, W)
        if branch == "object_amodal":
            mask[:, :, 1, :, :] = 1.0  # object channel active
        elif branch == "human_amodal":
            mask[:, :, 0, :, :] = 1.0  # human channel active

        return mask

    @torch.no_grad()
    def _run_ode_sampling(
        self, video_input: torch.Tensor, mask_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Run Euler ODE sampling to generate 3D Gaussian parameters.

        Args:
            video_input:   (1, T, C_in, H, W)
            mask_features: (1, T, C_m, H, W)

        Returns:
            gen_3d: (1, T, N, C_3d) — generated 3D Gaussian parameters
        """
        fm = self.cfg.fm
        device = self.device

        video_input = video_input.to(device)
        mask_features = mask_features.to(device)

        # Initial noise for the RGB channels of video, but keep mask channel
        # as the actual mask (ones = fully visible for amodal input).
        # x_0_video must have the same shape as video_input.
        x_0_video = torch.randn_like(video_input)
        # Preserve the mask channel (channel index 3+) from the prepared input
        # so the model sees correct mask conditioning during ODE integration.
        C_rgb = self.cfg.fm.video_channels  # 3
        if video_input.shape[2] > C_rgb:
            x_0_video[:, :, C_rgb:] = video_input[:, :, C_rgb:]

        x_0_3d = torch.randn(
            1, fm.num_frames, fm.num_points, fm.point_channels,
            device=device,
        )

        t0 = time.time()
        gen_video, gen_3d = self._euler_ode_sample(
            self.model,
            x_0_video,
            x_0_3d,
            num_steps=fm.num_ode_steps,
            mask_features=mask_features,
        )
        elapsed = time.time() - t0
        print(f"[Step3] ODE sampling: {fm.num_ode_steps} steps in {elapsed:.1f}s")

        return gen_3d  # (1, T, N, 14)

    def _postprocess_gaussians(self, gen_3d: torch.Tensor) -> dict:
        """
        Convert raw model output to structured 3DGS parameters.

        Input: (1, T, N, 14) — xyz(3), rotation(4), scaling(3), opacity(1), SH(3)
        Output: dict with separated, activated parameters (averaged over frames).
        """
        # Average over temporal frames for a single canonical 3D representation
        gs_params = gen_3d[0].mean(dim=0)  # (N, 14)

        xyz = gs_params[:, 0:3]
        rotation = F.normalize(gs_params[:, 3:7], dim=-1)
        scaling = gs_params[:, 7:10].clamp(min=1e-6)
        opacity = gs_params[:, 10:11].sigmoid()
        shs = gs_params[:, 11:14].sigmoid()

        return {
            "xyz": xyz.cpu(),             # (N, 3)
            "rotation": rotation.cpu(),   # (N, 4)
            "scaling": scaling.cpu(),     # (N, 3)
            "opacity": opacity.cpu(),     # (N, 1)
            "shs": shs.cpu(),            # (N, 3)
            "raw": gs_params.cpu(),       # (N, 14) — full parameter vector
        }

    def _save_gaussians(self, gs: dict, name: str):
        """Save 3DGS parameters to disk."""
        out_path = os.path.join(self.output_dir, f"{name}.pt")
        torch.save(gs, out_path)
        print(f"[Step3] Saved {name}: {gs['xyz'].shape[0]} Gaussians → {out_path}")

    def run(self):
        """Execute the full Step 3 pipeline for both human and object branches."""
        print("=" * 60)
        print("[Step3] Flow Matching 3D Lifting Pipeline")
        print(f"  Input:  {self.amodal_dir}")
        print(f"  Output: {self.output_dir}")
        print("=" * 60)

        results = {}
        for branch, gs_name in [
            ("object_amodal", "G_o"),
            ("human_amodal", "G_h"),
        ]:
            print(f"\n--- Processing {branch} → {gs_name} ---")

            # 1. Load amodal frames
            video = self._load_amodal_frames(branch)

            # 2. Prepare model inputs
            video_input = self._prepare_video_input(video)
            mask_features = self._prepare_mask_features(branch)

            # 3. ODE sampling
            gen_3d = self._run_ode_sampling(video_input, mask_features)

            # 4. Post-process to structured 3DGS params
            gs = self._postprocess_gaussians(gen_3d)

            # 5. Save
            self._save_gaussians(gs, gs_name)
            results[gs_name] = gs

        # Also save a combined file for easy loading by Step 4
        combined_path = os.path.join(self.output_dir, "gs_init_combined.pt")
        torch.save({
            "G_o": results["G_o"],
            "G_h": results["G_h"],
            "config": {
                "num_points": self.cfg.fm.num_points,
                "point_channels": self.cfg.fm.point_channels,
                "num_ode_steps": self.cfg.fm.num_ode_steps,
            },
        }, combined_path)
        print(f"\n[Step3] Combined output → {combined_path}")
        print("[Step3] Done.")

        return results
