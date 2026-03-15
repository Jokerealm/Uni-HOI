"""
Step 3 Pipeline: Zero-shot 3D Lifting via Hunyuan3D-2.

Takes amodal completion frames from Step 2, generates 3D meshes using
Hunyuan3D-2 (zero-shot image-to-3D), samples 3DGS parameters from the
mesh surface, and optionally runs metric alignment.

No training involved — pure inference with a pretrained foundation model.
"""
import os
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from configs.step3_config import Step3PipelineConfig, Hunyuan3DConfig


class Step3Pipeline:
    """Zero-shot 3D lifting: amodal frames → initial 3DGS via Hunyuan3D-2."""

    def __init__(self, cfg: Step3PipelineConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Resolve paths — handle both flat and BEHAVE layouts
        base = os.path.join(cfg.input_dir, cfg.video_name)
        if not os.path.isdir(base):
            base_seq = os.path.join(cfg.input_dir, "sequences", cfg.video_name)
            if os.path.isdir(base_seq):
                base = base_seq
        self.video_dir = base
        self.amodal_dir = os.path.join(self.video_dir, cfg.amodal_subdir)
        self.processed_dir = os.path.join(self.video_dir, cfg.processed_subdir)
        self.output_dir = os.path.join(self.video_dir, cfg.output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy-load the model (heavy, ~6GB VRAM)
        self._pipeline = None
        self._rembg = None

    def _load_pipeline(self):
        """Load Hunyuan3D-2 shape generation pipeline (lazy)."""
        if self._pipeline is not None:
            return

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except (ImportError, AttributeError, Exception) as e:
            print(f"[Step3] WARNING: hy3dgen unavailable ({type(e).__name__}: {e})")
            print("[Step3] Will use synthetic 3DGS init as fallback.")
            self._pipeline = "FALLBACK"
            return

        hy3d = self.cfg.hy3d
        dtype = torch.float16 if hy3d.dtype == "float16" else torch.float32

        print(f"[Step3] Loading Hunyuan3D-2 from {hy3d.model_path} "
              f"(subfolder={hy3d.subfolder})...")
        t0 = time.time()
        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            hy3d.model_path,
            subfolder=hy3d.subfolder,
            device=str(self.device),
            dtype=dtype,
        )
        print(f"[Step3] Hunyuan3D-2 loaded in {time.time() - t0:.1f}s")

        if hy3d.remove_background:
            try:
                from hy3dgen.rembg import BackgroundRemover
                self._rembg = BackgroundRemover()
                print("[Step3] Background remover loaded")
            except Exception as e:
                print(f"[Step3] Warning: Could not load background remover: {e}")

    def _select_frame(self, frames_dir: str) -> str:
        """Select the best frame from amodal completion output."""
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not paths:
            raise FileNotFoundError(
                f"[Step3] No frames found in {frames_dir}. Run Step 2 first."
            )

        sel = self.cfg.frame_selection
        if sel == "middle":
            idx = len(paths) // 2
        elif sel == "first":
            idx = 0
        elif sel == "last":
            idx = -1
        else:
            try:
                idx = int(sel)
            except ValueError:
                idx = len(paths) // 2

        idx = max(0, min(idx, len(paths) - 1))
        print(f"[Step3] Selected frame {idx}/{len(paths)}: {paths[idx]}")
        return paths[idx]

    def _prepare_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image for Hunyuan3D-2."""
        image = Image.open(image_path).convert("RGBA")

        # Remove background if configured
        if self._rembg is not None and self.cfg.hy3d.remove_background:
            image = self._rembg(image)

        return image

    @torch.no_grad()
    def _generate_mesh(self, image: Image.Image) -> trimesh.Trimesh:
        """Run Hunyuan3D-2 to generate a 3D mesh from a single image."""
        hy3d = self.cfg.hy3d
        t0 = time.time()

        results = self._pipeline(
            image=image,
            num_inference_steps=hy3d.num_inference_steps,
            guidance_scale=hy3d.guidance_scale,
            octree_resolution=hy3d.octree_resolution,
            output_type="trimesh",
        )

        elapsed = time.time() - t0
        # results is a list of trimesh objects
        mesh = results[0] if isinstance(results, list) else results
        print(f"[Step3] Mesh generated in {elapsed:.1f}s: "
              f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh

    def _mesh_to_gaussians(self, mesh: trimesh.Trimesh, label: str) -> dict:
        """
        Sample points from mesh surface and create initial 3DGS parameters.

        Returns dict with keys: xyz, rotation, scaling, opacity, shs, raw
        matching the format expected by MetricAlignmentBridge and Step 4.
        """
        N = self.cfg.hy3d.num_sample_points
        init_scale = self.cfg.hy3d.init_gaussian_scale

        # Sample points uniformly on mesh surface
        points, face_indices = trimesh.sample.sample_surface(mesh, N)
        points = points.astype(np.float32)  # (N, 3)

        # Get vertex colors at sampled points (if available)
        if mesh.visual is not None and hasattr(mesh.visual, 'face_colors'):
            colors = mesh.visual.face_colors[face_indices, :3] / 255.0
        elif mesh.visual is not None and hasattr(mesh.visual, 'vertex_colors'):
            # Interpolate vertex colors
            face_verts = mesh.faces[face_indices]  # (N, 3)
            vc = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            colors = vc[face_verts[:, 0]]  # Approximate: use first vertex color
        else:
            colors = np.ones((N, 3), dtype=np.float32) * 0.5  # neutral gray

        colors = colors.astype(np.float32)

        # Get surface normals for rotation initialization
        normals = mesh.face_normals[face_indices].astype(np.float32)  # (N, 3)

        # Convert normals to quaternions (align z-axis with normal)
        rotations = self._normals_to_quaternions(normals)  # (N, 4)

        # Build 3DGS parameter tensor
        xyz = torch.from_numpy(points)                          # (N, 3)
        rotation = torch.from_numpy(rotations)                  # (N, 4)
        scaling = torch.full((N, 3), init_scale)                # (N, 3)
        opacity = torch.full((N, 1), 0.9)                      # (N, 1)
        shs = torch.from_numpy(colors)                          # (N, 3)

        # Raw 14-channel tensor (model output space, before activation)
        raw = torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)  # (N, 14)

        print(f"[Step3] {label}: sampled {N} Gaussians from mesh, "
              f"xyz range: [{xyz.min():.3f}, {xyz.max():.3f}]")

        return {
            "xyz": xyz,
            "rotation": rotation,
            "scaling": scaling,
            "opacity": opacity,
            "shs": shs,
            "raw": raw,
        }

    @staticmethod
    def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
        """
        Convert surface normals to quaternions that align the local z-axis
        with the normal direction.

        normals: (N, 3) unit vectors
        returns: (N, 4) quaternions [w, x, y, z]
        """
        N = normals.shape[0]
        quats = np.zeros((N, 4), dtype=np.float32)

        z_axis = np.array([0, 0, 1], dtype=np.float32)
        for i in range(N):
            n = normals[i]
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                quats[i] = [1, 0, 0, 0]
                continue
            n = n / n_norm

            dot = np.dot(z_axis, n)
            if dot > 0.9999:
                quats[i] = [1, 0, 0, 0]
            elif dot < -0.9999:
                quats[i] = [0, 1, 0, 0]  # 180° rotation around x
            else:
                cross = np.cross(z_axis, n)
                w = 1.0 + dot
                quats[i] = [w, cross[0], cross[1], cross[2]]
                quats[i] /= np.linalg.norm(quats[i])

        return quats

    def _save_mesh(self, mesh: trimesh.Trimesh, name: str):
        """Save generated mesh for debugging/visualization."""
        mesh_path = os.path.join(self.output_dir, f"{name}_mesh.obj")
        mesh.export(mesh_path)
        print(f"[Step3] Saved mesh: {mesh_path}")

    def _save_gaussians(self, gs: dict, name: str):
        """Save 3DGS parameters to disk."""
        out_path = os.path.join(self.output_dir, f"{name}.pt")
        torch.save(gs, out_path)
        print(f"[Step3] Saved {name}: {gs['xyz'].shape[0]} Gaussians → {out_path}")

    def _generate_synthetic_gs(self, label: str, num_points: int) -> dict:
        """Generate synthetic 3DGS init (random sphere) when hy3dgen is unavailable."""
        N = num_points
        init_scale = self.cfg.hy3d.init_gaussian_scale

        # Random points on a unit sphere
        theta = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
        phi = np.random.uniform(0, np.pi, N).astype(np.float32)
        r = np.random.uniform(0.3, 0.5, N).astype(np.float32)
        xyz_np = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ], axis=-1)

        xyz = torch.from_numpy(xyz_np)
        rotation = torch.zeros(N, 4)
        rotation[:, 0] = 1.0  # identity quaternion
        scaling = torch.full((N, 3), init_scale)
        opacity = torch.full((N, 1), 0.9)
        shs = torch.ones(N, 3) * 0.5  # neutral gray

        raw = torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)
        print(f"[Step3] {label}: generated {N} synthetic Gaussians (fallback)")
        return {"xyz": xyz, "rotation": rotation, "scaling": scaling,
                "opacity": opacity, "shs": shs, "raw": raw}

    def _run_fallback(self) -> dict:
        """Fallback path: generate synthetic 3DGS when hy3dgen is unavailable."""
        N = self.cfg.hy3d.num_sample_points
        results = {}
        for gs_name, n_pts in [("G_o", N), ("G_h", N)]:
            gs = self._generate_synthetic_gs(gs_name, n_pts)
            self._save_gaussians(gs, gs_name)
            results[gs_name] = gs

        combined_path = os.path.join(self.output_dir, "gs_init_combined.pt")
        torch.save({
            "G_o": results["G_o"], "G_h": results["G_h"],
            "config": {"model": "synthetic_fallback", "num_sample_points": N},
        }, combined_path)
        print(f"[Step3] Combined output → {combined_path}")

        if self.cfg.run_alignment:
            self._run_alignment()

        print("[Step3] Done (fallback mode).")
        return results

    def run(self):
        """Execute the full Step 3 pipeline for both human and object branches."""
        print("=" * 60)
        print("[Step3] Zero-shot 3D Lifting via Hunyuan3D-2")
        print(f"  Input:  {self.amodal_dir}")
        print(f"  Output: {self.output_dir}")
        print("=" * 60)

        # Load model
        self._load_pipeline()

        # Fallback: generate synthetic 3DGS when hy3dgen is unavailable
        if self._pipeline == "FALLBACK":
            return self._run_fallback()

        results = {}
        for branch, gs_name in [
            ("object_amodal", "G_o"),
            ("human_amodal", "G_h"),
        ]:
            print(f"\n--- Processing {branch} → {gs_name} ---")

            # 1. Select best frame from amodal completion
            frames_dir = os.path.join(self.amodal_dir, branch, "frames")
            frame_path = self._select_frame(frames_dir)

            # 2. Prepare image
            image = self._prepare_image(frame_path)

            # 3. Generate 3D mesh via Hunyuan3D-2
            mesh = self._generate_mesh(image)

            # 4. Save mesh for visualization
            self._save_mesh(mesh, gs_name)

            # 5. Sample 3DGS parameters from mesh
            gs = self._mesh_to_gaussians(mesh, gs_name)

            # 6. Save
            self._save_gaussians(gs, gs_name)
            results[gs_name] = gs

        # Save combined file for Step 4 compatibility
        combined_path = os.path.join(self.output_dir, "gs_init_combined.pt")
        torch.save({
            "G_o": results["G_o"],
            "G_h": results["G_h"],
            "config": {
                "model": self.cfg.hy3d.model_path,
                "num_sample_points": self.cfg.hy3d.num_sample_points,
                "num_inference_steps": self.cfg.hy3d.num_inference_steps,
            },
        }, combined_path)
        print(f"\n[Step3] Combined output → {combined_path}")

        # Run metric alignment if configured
        if self.cfg.run_alignment:
            self._run_alignment()

        print("[Step3] Done.")
        return results

    def _run_alignment(self):
        """Run metric alignment bridge (Step 3.5)."""
        from configs.alignment_config import AlignmentPipelineConfig
        from pipeline.metric_alignment_bridge import MetricAlignmentBridge

        print("\n--- Running Metric Alignment Bridge ---")
        align_cfg = AlignmentPipelineConfig(
            input_dir=self.cfg.input_dir,
            video_name=self.cfg.video_name,
            processed_subdir=self.cfg.processed_subdir,
            gs_init_subdir=self.cfg.output_subdir,
            output_subdir=self.cfg.output_subdir + "_aligned",
        )
        # Handle BEHAVE path
        if not os.path.isdir(os.path.join(self.cfg.input_dir, self.cfg.video_name)):
            seq_path = os.path.join(self.cfg.input_dir, "sequences")
            if os.path.isdir(os.path.join(seq_path, self.cfg.video_name)):
                align_cfg.input_dir = seq_path

        bridge = MetricAlignmentBridge(align_cfg)
        bridge.run()
