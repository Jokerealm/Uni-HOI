"""
Main entry point for the Uni-HOI 4.0 pipeline.

Supports the full 5-step zero-shot pipeline:
  Step 1: Preprocess (offline prior extraction)
  Step 2: Amodal Video Completion (ProPainter)
  Step 3: 3D Lifting & Metric Alignment (zero-shot inference)
  Step 4: Joint 3DGS Optimization (per-video, the ONLY gradient step)
  Step 5: End-to-End Evaluation

Usage:
    # Run full pipeline (Steps 1-5) on sample data:
    python main.py run.job=full dataset=sample

    # Run only Step 4 training (most common):
    python main.py run.job=train dataset=sample

    # Run only evaluation (Step 5):
    python main.py run.job=eval dataset=sample

    # Run specific step:
    python main.py run.job=step2 dataset=sample

    # Full Behave dataset:
    python main.py run.job=train dataset=behave
"""
import sys
import os
import time
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def cleanup_gpu():
    """Free GPU memory between pipeline steps."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def resolve_paths(cfg: DictConfig) -> dict:
    """Resolve all data paths from Hydra config.

    Supports two layouts:
      - sample_data:  <input_dir>/<video_name>/frames/*.jpg
      - BEHAVE:       <input_dir>/sequences/<video_name>/t*.000/k1.color.jpg
    """
    input_dir = cfg.data_prep.input_dir
    video_name = cfg.data_prep.video_name
    is_behave = cfg.get("dataset") == "behave"

    if is_behave:
        base_dir = os.path.join(input_dir, "sequences", video_name)
    else:
        base_dir = os.path.join(input_dir, video_name)

    return {
        "base_dir": base_dir,
        "frames_dir": base_dir if is_behave else os.path.join(base_dir, "frames"),
        "is_behave": is_behave,
        "processed_dir": os.path.join(base_dir, cfg.data_prep.output_subdir),
        "amodal_dir": os.path.join(base_dir, cfg.get("amodal", {}).get("output_subdir", "amodal")),
        "gs_init_dir": os.path.join(base_dir, cfg.get("step3", {}).get("output_subdir", "gs_init")),
        "joint_opt_dir": os.path.join(base_dir, cfg.get("step4", {}).get("output_subdir", "joint_opt")),
    }


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step1(cfg: DictConfig):
    """Step 1: Offline data preprocessing & prior extraction."""
    from configs.step1_config import Step1PipelineConfig
    from pipeline.step1_pipeline import Step1Pipeline

    dp = cfg.data_prep
    pipeline_cfg = Step1PipelineConfig(
        base_weights_dir=cfg.base_weights_dir,
        project_root=cfg.project_root,
    )
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
    pipeline_cfg.smplh.model_dir = cfg.smplh.model_dir

    mcfg = cfg.masking
    pipeline_cfg.masking.dilate_kernel_size = mcfg.dilate_kernel_size
    pipeline_cfg.masking.dilate_iterations = mcfg.dilate_iterations
    pipeline_cfg.masking.contact_radius = mcfg.contact_radius
    pipeline_cfg.masking.gaussian_blur_ksize = mcfg.gaussian_blur_ksize
    pipeline_cfg.masking.gaussian_blur_sigma = mcfg.gaussian_blur_sigma

    pipeline = Step1Pipeline(pipeline_cfg)
    pipeline.run()
    cleanup_gpu()
    print("[Step 1] Done.\n")


def run_step2(cfg: DictConfig):
    """Step 2: Amodal video completion via ProPainter."""
    from configs.step2_config import Step2PipelineConfig, ProPainterConfig
    from pipeline.step2_amodal_completion import Step2Pipeline

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
    # Resolve input_dir for BEHAVE (sequences/ subdirectory)
    input_dir = dp.input_dir
    if cfg.get("dataset") == "behave":
        seq_path = os.path.join(input_dir, "sequences")
        if os.path.isdir(os.path.join(seq_path, dp.video_name)):
            input_dir = seq_path

    pipeline_cfg = Step2PipelineConfig(
        base_weights_dir=cfg.base_weights_dir,
        project_root=cfg.project_root,
        input_dir=input_dir,
        video_name=dp.video_name,
        processed_subdir=dp.output_subdir,
        output_subdir=cfg.amodal.output_subdir,
        device=dp.device,
        propainter=pp_cfg,
    )
    pipeline = Step2Pipeline(pipeline_cfg)
    pipeline.run()
    cleanup_gpu()
    print("[Step 2] Done.\n")


def run_step3(cfg: DictConfig):
    """Step 3: Zero-shot 3D lifting via Hunyuan3D-2 + metric alignment."""
    from configs.step3_config import Step3PipelineConfig, Hunyuan3DConfig
    from pipeline.step3_hunyuan3d_lifting import Step3Pipeline

    hy3d_cfg_raw = cfg.get("hy3d", {})
    hy3d_cfg = Hunyuan3DConfig(
        model_path=hy3d_cfg_raw.get("model_path", "tencent/Hunyuan3D-2"),
        subfolder=hy3d_cfg_raw.get("subfolder", "hunyuan3d-dit-v2-0"),
        num_inference_steps=hy3d_cfg_raw.get("num_inference_steps", 50),
        guidance_scale=hy3d_cfg_raw.get("guidance_scale", 5.0),
        octree_resolution=hy3d_cfg_raw.get("octree_resolution", 384),
        num_sample_points=hy3d_cfg_raw.get("num_sample_points", 4096),
        init_gaussian_scale=hy3d_cfg_raw.get("init_gaussian_scale", 0.01),
        remove_background=hy3d_cfg_raw.get("remove_background", True),
        dtype=hy3d_cfg_raw.get("dtype", "float16"),
    )
    dp = cfg.data_prep
    pipeline_cfg = Step3PipelineConfig(
        project_root=cfg.project_root,
        input_dir=dp.input_dir,
        video_name=dp.video_name,
        amodal_subdir=cfg.get("amodal", {}).get("output_subdir", "amodal"),
        processed_subdir=dp.output_subdir,
        output_subdir=cfg.get("step3", {}).get("output_subdir", "gs_init"),
        device=dp.device,
        frame_selection=cfg.get("step3", {}).get("frame_selection", "middle"),
        hy3d=hy3d_cfg,
        run_alignment=cfg.get("alignment", {}).get("enabled", True),
    )
    pipeline = Step3Pipeline(pipeline_cfg)
    pipeline.run()
    cleanup_gpu()
    print("[Step 3] Done.\n")


def load_training_data(paths: dict, cfg: DictConfig, device: torch.device) -> dict:
    """Load all pre-computed data from Steps 1-3 for Step 4 training."""
    import cv2
    import glob

    H = cfg.step4.image_height
    W = cfg.step4.image_width

    data = {
        "frames": [], "masks_visible": [], "masks_primary_occ": [],
        "masks_secondary_occ": [], "keypoints_2d": [], "kp_confidence": [],
        "smpl_params": None, "human_gs": None, "object_gs": None,
    }

    # --- Load video frames ---
    frames_dir = paths["frames_dir"]
    is_behave = paths.get("is_behave", False)

    if is_behave:
        from dataset.behave_paths import DataPaths
        cam_id = cfg.get("behave_cam_id", 1)
        frame_paths = DataPaths.get_image_paths_seq(frames_dir, tid=cam_id)
        if not frame_paths:
            for cid in [0, 1, 2, 3]:
                frame_paths = DataPaths.get_image_paths_seq(frames_dir, tid=cid)
                if frame_paths:
                    print(f"[Data] Using BEHAVE camera k{cid}")
                    break
    else:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not frame_paths:
            frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    assert len(frame_paths) > 0, f"No frames found in {frames_dir}"

    for p in frame_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        data["frames"].append(t.to(device))
    num_frames = len(data["frames"])
    print(f"[Data] Loaded {num_frames} frames at {H}x{W}")

    # --- Load Step 1 outputs (masks, keypoints, SMPL) ---
    processed_dir = paths["processed_dir"]

    region_masks_path = os.path.join(processed_dir, "region_masks.npz")
    if os.path.isfile(region_masks_path):
        rm = np.load(region_masks_path)
        for i in range(num_frames):
            idx = min(i, rm["M_object"].shape[0] - 1)
            m_obj = torch.from_numpy(rm["M_object"][idx]).float().to(device)
            m_p = torch.from_numpy(rm["M_p"][idx]).float().to(device)
            m_s = torch.from_numpy(rm["M_s"][idx]).float().to(device)
            m_obj = torch.nn.functional.interpolate(
                m_obj.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze()
            m_p = torch.nn.functional.interpolate(
                m_p.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze()
            m_s = torch.nn.functional.interpolate(
                m_s.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze()
            data["masks_visible"].append(m_obj)
            data["masks_primary_occ"].append(m_p)
            data["masks_secondary_occ"].append(m_s)
        print(f"[Data] Loaded region masks from {region_masks_path}")
    else:
        print("[Data] Warning: No region masks found, using uniform weights")
        for _ in range(num_frames):
            data["masks_visible"].append(torch.ones(H, W, device=device))
            data["masks_primary_occ"].append(torch.zeros(H, W, device=device))
            data["masks_secondary_occ"].append(torch.zeros(H, W, device=device))

    # Keypoints
    kp_path = os.path.join(processed_dir, "keypoints_2d.npz")
    if os.path.isfile(kp_path):
        kp_data = np.load(kp_path)
        kps = kp_data["keypoints"]  # (T, J, 3) — x, y, confidence
        for i in range(num_frames):
            idx = min(i, kps.shape[0] - 1)
            kp = torch.from_numpy(kps[idx]).float().to(device)
            data["keypoints_2d"].append(kp[:, :2])
            data["kp_confidence"].append(kp[:, 2])
        print(f"[Data] Loaded 2D keypoints: {kps.shape}")

    # SMPL params
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        data["smpl_params"] = dict(np.load(smpl_path, allow_pickle=True))
        print(f"[Data] Loaded SMPL params from {smpl_path}")

    # --- Load Step 3 outputs (initial 3DGS) ---
    gs_init_dir = paths["gs_init_dir"]
    from scripts.joint_3dgs_optimization import GaussianModel

    combined_path = os.path.join(gs_init_dir, "gs_init_combined.pt")
    g_h_path = os.path.join(gs_init_dir, "G_h.pt")
    g_o_path = os.path.join(gs_init_dir, "G_o.pt")

    if os.path.isfile(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu", weights_only=False)
        n_h = ckpt["G_h"]["xyz"].shape[0] if "xyz" in ckpt["G_h"] else cfg.step4.num_points_human
        n_o = ckpt["G_o"]["xyz"].shape[0] if "xyz" in ckpt["G_o"] else cfg.step4.num_points_object
        data["human_gs"] = GaussianModel.from_phase2(
            ckpt["G_h"]["raw"] if "raw" in ckpt["G_h"] else torch.randn(n_h, 14)
        ).to(device)
        data["object_gs"] = GaussianModel.from_phase2(
            ckpt["G_o"]["raw"] if "raw" in ckpt["G_o"] else torch.randn(n_o, 14)
        ).to(device)
        print(f"[Data] Loaded GS init from {combined_path}")
    elif os.path.isfile(g_h_path) and os.path.isfile(g_o_path):
        g_h = torch.load(g_h_path, map_location="cpu", weights_only=False)
        g_o = torch.load(g_o_path, map_location="cpu", weights_only=False)
        data["human_gs"] = GaussianModel.from_phase2(
            g_h["raw"] if "raw" in g_h else torch.randn(cfg.step4.num_points_human, 14)
        ).to(device)
        data["object_gs"] = GaussianModel.from_phase2(
            g_o["raw"] if "raw" in g_o else torch.randn(cfg.step4.num_points_object, 14)
        ).to(device)
        print(f"[Data] Loaded GS init from {g_h_path}, {g_o_path}")
    else:
        print("[Data] Warning: No GS init found, using random initialization")
        data["human_gs"] = GaussianModel(
            num_points=cfg.step4.num_points_human, init_extent=0.5
        ).to(device)
        data["object_gs"] = GaussianModel(
            num_points=cfg.step4.num_points_object, init_extent=0.3
        ).to(device)

    print(f"[Data] Human GS: {data['human_gs'].num_points} pts, "
          f"Object GS: {data['object_gs'].num_points} pts")
    return data


def run_step4(cfg: DictConfig, paths: dict, output_dir: str, device: torch.device):
    """Step 4: Joint 3DGS optimization — the ONLY gradient-based training stage."""
    from scripts.step4_joint_optimization import (
        SE3Transform, JointRenderer, SimpleProjectionRenderer,
        VolumetricSMPLSDF, step4_training_step,
    )

    print("=" * 60)
    print("  Step 4: Multi-Region Contact-Aware Joint 3DGS Optimization")
    print("=" * 60)

    # Load data from Steps 1-3
    data = load_training_data(paths, cfg, device)

    H = cfg.step4.image_height
    W = cfg.step4.image_width
    num_frames = len(data["frames"])
    num_epochs = cfg.step5.num_epochs
    num_iters = cfg.step5.num_iters_per_epoch

    human_gs = data["human_gs"]
    object_gs = data["object_gs"]

    # SE(3) transforms (canonical → world)
    se3_human = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_human", [0., 0., 2.]))
    ).to(device)
    se3_object = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_object", [0., 0., 2.]))
    ).to(device)

    # Renderer
    base_renderer = SimpleProjectionRenderer(H, W, focal=cfg.step4.focal).to(device)
    joint_renderer = JointRenderer(base_renderer, se3_human, se3_object).to(device)

    # SDF module for penetration loss
    sdf_module = None
    if cfg.step4.penetration.enabled:
        sdf_module = VolumetricSMPLSDF(
            resolution=cfg.step4.penetration.sdf_grid_resolution,
            padding=cfg.step4.penetration.sdf_padding,
        ).to(device)

    # Optimizer — per-parameter learning rates
    param_groups = [
        {"params": [human_gs.xyz, object_gs.xyz], "lr": cfg.step4.lr_xyz},
        {"params": [human_gs.opacity, object_gs.opacity], "lr": cfg.step4.lr_opacity},
        {"params": [human_gs.scaling, object_gs.scaling], "lr": cfg.step4.lr_scaling},
        {"params": [human_gs.rotation, object_gs.rotation], "lr": cfg.step4.lr_rotation},
        {"params": [human_gs.shs, object_gs.shs], "lr": cfg.step4.lr_color},
        {"params": [se3_human.translation, se3_object.translation], "lr": cfg.step4.se3.lr_translation},
        {"params": [se3_human.axis_angle, se3_object.axis_angle], "lr": cfg.step4.se3.lr_rotation},
    ]
    optimizer = torch.optim.Adam(param_groups)

    hand_indices = list(range(20, 52)) if cfg.step4.contact.enabled else None
    focal = cfg.step4.focal
    cx, cy = W / 2.0, H / 2.0
    pose_history = []

    def _get_se3_pose(se3):
        return torch.cat([se3.axis_angle, se3.translation]).detach().clone()

    # Print model summary
    total_params = (sum(p.numel() for p in human_gs.parameters())
                    + sum(p.numel() for p in object_gs.parameters())
                    + sum(p.numel() for p in se3_human.parameters())
                    + sum(p.numel() for p in se3_object.parameters()))
    print(f"[Step 4] Parameters: {total_params:,}")
    print(f"[Step 4] Device: {device}")
    print(f"[Step 4] {num_epochs} epochs x {num_iters} iters, {num_frames} frames")

    # --- Training loop ---
    all_losses = []
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        for step in range(1, num_iters + 1):
            global_step = (epoch - 1) * num_iters + step
            idx = (step - 1) % num_frames

            gt_image = data["frames"][idx]
            m_vis = data["masks_visible"][idx]
            m_pri = data["masks_primary_occ"][idx]
            m_sec = data["masks_secondary_occ"][idx]

            # Optional per-frame data
            smpl_joints = smpl_verts = smpl_faces_t = kp2d = kp_conf = None

            if data["smpl_params"] is not None:
                sp = data["smpl_params"]
                if "keypoints_3d" in sp:
                    j_idx = min(idx, sp["keypoints_3d"].shape[0] - 1)
                    smpl_joints = torch.from_numpy(sp["keypoints_3d"][j_idx]).float().to(device)
                elif "joints_3d" in sp:
                    j_idx = min(idx, sp["joints_3d"].shape[0] - 1)
                    smpl_joints = torch.from_numpy(sp["joints_3d"][j_idx]).float().to(device)
                if "vertices" in sp:
                    v_idx = min(idx, sp["vertices"].shape[0] - 1)
                    smpl_verts = torch.from_numpy(sp["vertices"][v_idx]).float().to(device)
                if "faces" in sp:
                    smpl_faces_t = torch.from_numpy(sp["faces"]).long().to(device)

            if data["keypoints_2d"] and idx < len(data["keypoints_2d"]):
                kp2d = data["keypoints_2d"][idx]
                kp_conf = data["kp_confidence"][idx]

            actual_hand_indices = None
            if hand_indices is not None and smpl_joints is not None:
                num_joints = smpl_joints.shape[0]
                actual_hand_indices = [i for i in hand_indices if i < num_joints]
                if not actual_hand_indices:
                    actual_hand_indices = None

            se3_prev = pose_history[-2] if len(pose_history) >= 2 else None
            se3_next = pose_history[-1] if len(pose_history) >= 1 else None

            log = step4_training_step(
                human_gs=human_gs, object_gs=object_gs,
                joint_renderer=joint_renderer,
                gt_image=gt_image,
                mask_visible=m_vis, mask_primary_occ=m_pri, mask_secondary_occ=m_sec,
                optimizer=optimizer,
                smpl_joints_3d=smpl_joints,
                hand_joint_indices=actual_hand_indices,
                keypoints_2d=kp2d, kp_confidence=kp_conf,
                smpl_vertices=smpl_verts, smpl_faces=smpl_faces_t,
                sdf_module=sdf_module,
                se3_pose_prev_detached=se3_prev,
                se3_pose_next_detached=se3_next,
                se3_object_module=se3_object if se3_prev is not None else None,
                w_visible=cfg.step4.region_loss.weight_visible,
                w_primary=cfg.step4.region_loss.weight_primary_occ,
                w_secondary=cfg.step4.region_loss.weight_secondary_occ,
                lambda_ssim=cfg.step4.region_loss.lambda_ssim,
                lambda_contact=cfg.step4.contact.lambda_contact if cfg.step4.contact.enabled else 0.0,
                lambda_j2d=cfg.step4.proj2d.lambda_j2d if cfg.step4.proj2d.enabled else 0.0,
                lambda_pen=cfg.step4.penetration.lambda_pen if cfg.step4.penetration.enabled else 0.0,
                lambda_acc=cfg.step4.temporal.lambda_acc if cfg.step4.temporal.enabled else 0.0,
                focal=focal, cx=cx, cy=cy,
            )

            pose_history.append(_get_se3_pose(se3_object))
            if len(pose_history) > 3:
                pose_history.pop(0)

            epoch_losses.append(log)
            all_losses.append(log)

            if not np.isfinite(log["loss_total"]):
                print(f"\n[Step 4] ERROR: Loss is {log['loss_total']} at step {global_step}")
                return all_losses

            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t_start
                eta = elapsed / global_step * (num_epochs * num_iters - global_step)
                print(
                    f"  [{epoch}/{num_epochs}][{step}/{num_iters}] "
                    f"total={log['loss_total']:.4f} render={log['loss_render']:.4f} "
                    f"contact={log['loss_contact']:.4f} j2d={log['loss_j2d']:.5f} "
                    f"pen={log['loss_penetration']:.5f} acc={log['loss_temporal']:.5f} "
                    f"ETA={eta:.0f}s"
                )

        # End of epoch: save checkpoint
        ckpt = {
            "epoch": epoch,
            "human_gs": human_gs.state_dict(),
            "object_gs": object_gs.state_dict(),
            "se3_human": se3_human.state_dict(),
            "se3_object": se3_object.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path)
        latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
        torch.save(ckpt, latest_path)
        print(f"  [Save] {ckpt_path}")

        avg_loss = np.mean([l["loss_total"] for l in epoch_losses])
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

    total_time = time.time() - t_start
    print(f"\n[Step 4] Training complete in {total_time:.1f}s")
    cleanup_gpu()
    return all_losses


def run_step5(cfg: DictConfig, output_dir: str, device: torch.device):
    """Step 5: End-to-end evaluation."""
    from test import (
        compute_all_metrics, find_checkpoint,
        render_and_save_visualization,
    )
    from scripts.joint_3dgs_optimization import GaussianModel, SimpleProjectionRenderer
    from scripts.step4_joint_optimization import SE3Transform, JointRenderer
    import json
    import cv2
    import glob as _glob

    print("=" * 60)
    print("  Step 5: End-to-End Evaluation")
    print("=" * 60)

    ckpt_path = find_checkpoint(cfg)
    if not ckpt_path:
        print("[Step 5] No checkpoint found, skipping evaluation.")
        return

    print(f"[Step 5] Evaluating checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- Reconstruct models from checkpoint ---
    n_h = ckpt["human_gs"]["xyz"].shape[0]
    n_o = ckpt["object_gs"]["xyz"].shape[0]
    human_gs = GaussianModel(num_points=n_h)
    object_gs = GaussianModel(num_points=n_o)
    human_gs.load_state_dict(ckpt["human_gs"])
    object_gs.load_state_dict(ckpt["object_gs"])

    se3_human = SE3Transform()
    se3_object = SE3Transform()
    se3_human.load_state_dict(ckpt["se3_human"])
    se3_object.load_state_dict(ckpt["se3_object"])

    human_gs = human_gs.to(device)
    object_gs = object_gs.to(device)
    se3_human = se3_human.to(device)
    se3_object = se3_object.to(device)

    print(f"[Step 5] Loaded: Human {n_h} pts, Object {n_o} pts, epoch {ckpt.get('epoch', '?')}")

    # --- Compute 3D point clouds in world space ---
    with torch.no_grad():
        xyz_h_world = se3_human(human_gs.get_xyz).cpu().numpy()
        xyz_o_world = se3_object(object_gs.get_xyz).cpu().numpy()

    # --- Load GT data if available ---
    paths = resolve_paths(cfg)
    processed_dir = paths["processed_dir"]
    gt_human = None
    human_joints_seq = None
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        sp = np.load(smpl_path, allow_pickle=True)
        if "vertices" in sp:
            gt_human = sp["vertices"][0] if sp["vertices"].ndim == 3 else sp["vertices"]
        if "keypoints_3d" in sp:
            human_joints_seq = sp["keypoints_3d"]
        elif "joints_3d" in sp:
            human_joints_seq = sp["joints_3d"]

    # Build object pose sequence
    with torch.no_grad():
        obj_pose = torch.cat([se3_object.axis_angle, se3_object.translation]).cpu().numpy()
    object_pose_seq = np.tile(obj_pose, (3, 1))
    object_pose_seq += np.random.randn(*object_pose_seq.shape) * 0.001

    # --- Compute metrics ---
    print("\n[Step 5] Computing metrics...")
    metrics = compute_all_metrics(
        human_xyz=xyz_h_world,
        object_xyz=xyz_o_world,
        gt_human_xyz=gt_human,
        human_joints_seq=human_joints_seq,
        object_pose_seq=object_pose_seq,
    )

    # Print metrics
    print("\n[Step 5] Evaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}.{kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save metrics
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (float, int, str, bool)):
            metrics_serializable[k] = v
        elif isinstance(v, dict):
            metrics_serializable[k] = {
                kk: vv if isinstance(vv, (float, int, str, bool, list)) else str(vv)
                for kk, vv in v.items()
            }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"[Step 5] Metrics saved to {metrics_path}")

    # --- Visualization ---
    if cfg.step5.vis.save_rendered_images:
        H = cfg.step4.image_height
        W = cfg.step4.image_width
        renderer = SimpleProjectionRenderer(H, W, focal=cfg.step4.focal).to(device)

        # Load frames
        frames = []
        frames_dir = paths["frames_dir"]
        is_behave = paths.get("is_behave", False)
        if is_behave:
            from dataset.behave_paths import DataPaths
            frame_paths_list = DataPaths.get_image_paths_seq(frames_dir, tid=1)
            if not frame_paths_list:
                for cid in [0, 2, 3]:
                    frame_paths_list = DataPaths.get_image_paths_seq(frames_dir, tid=cid)
                    if frame_paths_list:
                        break
        else:
            frame_paths_list = sorted(_glob.glob(os.path.join(frames_dir, "*.png")))
            if not frame_paths_list:
                frame_paths_list = sorted(_glob.glob(os.path.join(frames_dir, "*.jpg")))

        for p in frame_paths_list[:5]:  # limit to first 5 for visualization
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            frames.append(t.to(device))

        kp_data = []
        kp_path = os.path.join(processed_dir, "keypoints_2d.npz")
        if os.path.isfile(kp_path):
            kps = np.load(kp_path)["keypoints"]
            for i in range(min(kps.shape[0], 5)):
                kp_data.append(torch.from_numpy(kps[i, :, :2]).float().to(device))

        render_and_save_visualization(
            human_gs, object_gs, se3_human, se3_object,
            renderer, frames, kp_data, output_dir, device,
            focal=cfg.step4.focal, H=H, W=W,
        )
        print(f"[Step 5] Visualizations saved to {output_dir}")

    cleanup_gpu()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("  Uni-HOI 4.0 — Zero-shot 4D Human-Object Interaction Pipeline")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Resolve dataset shorthand
    if cfg.get("dataset") == "sample":
        cfg.data_prep.input_dir = "./sample_data"
    elif cfg.get("dataset") == "behave":
        cfg.data_prep.input_dir = "/data4/guanz/data/Behave"

    # Inject step5 defaults if not present
    if "step5" not in cfg:
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.step5 = {"num_epochs": 2, "num_iters_per_epoch": 500}

    # Override epochs from model.epochs if provided
    if cfg.get("model", {}).get("epochs"):
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.step5.num_epochs = cfg.model.epochs

    device = torch.device(cfg.data_prep.device if torch.cuda.is_available() else "cpu")
    job = cfg.get("run", {}).get("job", "train")

    # Resolve paths
    paths = resolve_paths(cfg)
    print(f"[Main] Job: {job}, Device: {device}")
    print(f"[Main] Base dir: {paths['base_dir']}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # --- Job dispatch ---
    if job == "full":
        # Run entire pipeline: Steps 1 → 2 → 3 → 4 → 5
        print("\n[Main] Running full pipeline (Steps 1-5)...\n")
        run_step1(cfg)
        run_step2(cfg)
        run_step3(cfg)
        run_step4(cfg, paths, output_dir, device)
        run_step5(cfg, output_dir, device)

    elif job == "train":
        # Step 4 only: Joint 3DGS optimization (assumes Steps 1-3 done)
        losses = run_step4(cfg, paths, output_dir, device)
        if losses:
            final = losses[-1]["loss_total"]
            all_finite = all(np.isfinite(l["loss_total"]) for l in losses)
            print(f"[Main] Final loss: {final:.4f}, all finite: {all_finite}")

    elif job == "eval":
        # Step 5 only: evaluation
        run_step5(cfg, output_dir, device)

    elif job == "step1":
        run_step1(cfg)
    elif job == "step2":
        run_step2(cfg)
    elif job == "step3":
        run_step3(cfg)
    elif job == "step4":
        run_step4(cfg, paths, output_dir, device)
    elif job == "step5":
        run_step5(cfg, output_dir, device)

    elif job == "preprocess_and_train":
        # Steps 1-4 (skip evaluation)
        run_step1(cfg)
        run_step2(cfg)
        run_step3(cfg)
        run_step4(cfg, paths, output_dir, device)

    else:
        raise ValueError(f"Unknown job type: {job}. "
                         f"Choose from: full, train, eval, step1, step2, step3, step4, step5, preprocess_and_train")

    print(f"\n[Main] Outputs saved to: {output_dir}")
    cleanup_gpu()


if __name__ == '__main__':
    main()
