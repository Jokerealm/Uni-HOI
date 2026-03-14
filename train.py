#!/usr/bin/env python3
"""
train.py — Unified End-to-End Training Entry Point (Step 5)

Runs the full pipeline on sample_data or real dataset:
  Data Loading → Joint 3DGS Optimization → Checkpoint Saving

Usage:
    # Sample data (smoke test, 2 epochs):
    CUDA_VISIBLE_DEVICES=1 python train.py dataset=sample model.epochs=2

    # Full Behave dataset:
    CUDA_VISIBLE_DEVICES=1 python train.py dataset=behave

    # Override any parameter via Hydra:
    CUDA_VISIBLE_DEVICES=1 python train.py dataset=sample step5.num_epochs=5 step4.num_iters=1000
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


def cleanup_gpu():
    """Free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def print_gpu_status():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: Allocated {alloc:.2f}GB, Reserved {reserved:.2f}GB")


def resolve_paths(cfg: DictConfig) -> dict:
    """Resolve all data paths from Hydra config."""
    input_dir = cfg.data_prep.input_dir
    video_name = cfg.data_prep.video_name

    base_dir = os.path.join(input_dir, video_name)
    return {
        "base_dir": base_dir,
        "frames_dir": os.path.join(base_dir, "frames"),
        "processed_dir": os.path.join(base_dir, cfg.data_prep.output_subdir),
        "amodal_dir": os.path.join(base_dir, cfg.get("amodal", {}).get("output_subdir", "amodal")),
        "gs_init_dir": os.path.join(base_dir, cfg.get("step3", {}).get("output_subdir", "gs_init")),
        "joint_opt_dir": os.path.join(base_dir, cfg.get("step4", {}).get("output_subdir", "joint_opt")),
    }


def load_training_data(paths: dict, cfg: DictConfig, device: torch.device) -> dict:
    """
    Load all pre-computed data from Steps 1-3 for training.
    Returns a dict with all tensors needed for the optimization loop.
    """
    import cv2
    import glob

    H = cfg.step4.image_height
    W = cfg.step4.image_width

    data = {
        "frames": [],
        "masks_visible": [],
        "masks_primary_occ": [],
        "masks_secondary_occ": [],
        "keypoints_2d": [],
        "kp_confidence": [],
        "smpl_params": None,
        "human_gs": None,
        "object_gs": None,
    }

    # --- Load video frames ---
    frames_dir = paths["frames_dir"]
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    assert len(frame_paths) > 0, f"No frames found in {frames_dir}"

    for p in frame_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3, H, W)
        data["frames"].append(t.to(device))
    num_frames = len(data["frames"])
    print(f"[Train] Loaded {num_frames} frames at {H}x{W}")

    # --- Load Step 1 outputs (masks, keypoints, SMPL) ---
    processed_dir = paths["processed_dir"]

    # Region masks from .npz
    region_masks_path = os.path.join(processed_dir, "region_masks.npz")
    if os.path.isfile(region_masks_path):
        rm = np.load(region_masks_path)
        for i in range(num_frames):
            idx = min(i, rm["M_object"].shape[0] - 1)
            m_obj = torch.from_numpy(rm["M_object"][idx]).float().to(device)
            m_p = torch.from_numpy(rm["M_p"][idx]).float().to(device)
            m_s = torch.from_numpy(rm["M_s"][idx]).float().to(device)
            # Resize masks to match image
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
        print(f"[Train] Loaded region masks from {region_masks_path}")
    else:
        print("[Train] Warning: No region masks found, using uniform weights")
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
        print(f"[Train] Loaded 2D keypoints: {kps.shape}")
    else:
        print("[Train] Warning: No keypoints found")

    # SMPL params
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        data["smpl_params"] = dict(np.load(smpl_path, allow_pickle=True))
        print(f"[Train] Loaded SMPL params from {smpl_path}")

    # --- Load Step 3 outputs (initial 3DGS) ---
    gs_init_dir = paths["gs_init_dir"]
    from scripts.joint_3dgs_optimization import GaussianModel

    combined_path = os.path.join(gs_init_dir, "gs_init_combined.pt")
    g_h_path = os.path.join(gs_init_dir, "G_h.pt")
    g_o_path = os.path.join(gs_init_dir, "G_o.pt")

    if os.path.isfile(combined_path):
        ckpt = torch.load(combined_path, map_location="cpu")
        n_h = ckpt["G_h"]["xyz"].shape[0] if "xyz" in ckpt["G_h"] else cfg.step4.num_points_human
        n_o = ckpt["G_o"]["xyz"].shape[0] if "xyz" in ckpt["G_o"] else cfg.step4.num_points_object
        data["human_gs"] = GaussianModel.from_phase2(
            ckpt["G_h"]["raw"] if "raw" in ckpt["G_h"] else torch.randn(n_h, 14)
        ).to(device)
        data["object_gs"] = GaussianModel.from_phase2(
            ckpt["G_o"]["raw"] if "raw" in ckpt["G_o"] else torch.randn(n_o, 14)
        ).to(device)
        print(f"[Train] Loaded GS init from {combined_path}")
    elif os.path.isfile(g_h_path) and os.path.isfile(g_o_path):
        g_h = torch.load(g_h_path, map_location="cpu")
        g_o = torch.load(g_o_path, map_location="cpu")
        data["human_gs"] = GaussianModel.from_phase2(
            g_h["raw"] if "raw" in g_h else torch.randn(cfg.step4.num_points_human, 14)
        ).to(device)
        data["object_gs"] = GaussianModel.from_phase2(
            g_o["raw"] if "raw" in g_o else torch.randn(cfg.step4.num_points_object, 14)
        ).to(device)
        print(f"[Train] Loaded GS init from {g_h_path}, {g_o_path}")
    else:
        print("[Train] Warning: No GS init found, using random initialization")
        data["human_gs"] = GaussianModel(
            num_points=cfg.step4.num_points_human, init_extent=0.5
        ).to(device)
        data["object_gs"] = GaussianModel(
            num_points=cfg.step4.num_points_object, init_extent=0.3
        ).to(device)

    print(f"[Train] Human GS: {data['human_gs'].num_points} pts")
    print(f"[Train] Object GS: {data['object_gs'].num_points} pts")

    return data


def run_training(cfg: DictConfig, data: dict, output_dir: str, device: torch.device):
    """
    Core training loop: Joint 3DGS optimization with all loss terms.
    """
    from scripts.step4_joint_optimization import (
        SE3Transform, JointRenderer, SimpleProjectionRenderer,
        VolumetricSMPLSDF, step4_training_step,
    )

    H = cfg.step4.image_height
    W = cfg.step4.image_width
    num_frames = len(data["frames"])
    num_epochs = cfg.step5.num_epochs
    num_iters = cfg.step5.num_iters_per_epoch

    human_gs = data["human_gs"]
    object_gs = data["object_gs"]

    # SE(3) transforms
    se3_human = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_human", [0.0, 0.0, 2.0]))
    ).to(device)
    se3_object = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_object", [0.0, 0.0, 2.0]))
    ).to(device)

    # Renderer
    base_renderer = SimpleProjectionRenderer(H, W, focal=cfg.step4.focal).to(device)
    joint_renderer = JointRenderer(base_renderer, se3_human, se3_object).to(device)

    # SDF module
    sdf_module = None
    if cfg.step4.penetration.enabled:
        sdf_module = VolumetricSMPLSDF(
            resolution=cfg.step4.penetration.sdf_grid_resolution,
            padding=cfg.step4.penetration.sdf_padding,
        ).to(device)

    # Optimizer
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

    # Hand joint indices
    hand_indices = list(range(20, 52)) if cfg.step4.contact.enabled else None

    focal = cfg.step4.focal
    cx, cy = W / 2.0, H / 2.0

    # Pose history for temporal loss
    pose_history = []

    def _get_se3_pose(se3):
        return torch.cat([se3.axis_angle, se3.translation]).detach().clone()

    # --- Print model summary ---
    total_params = sum(p.numel() for p in human_gs.parameters()) + \
                   sum(p.numel() for p in object_gs.parameters()) + \
                   sum(p.numel() for p in se3_human.parameters()) + \
                   sum(p.numel() for p in se3_object.parameters())
    print(f"\n[Train] Model parameters: {total_params:,}")
    print(f"[Train] Device: {device}")
    print_gpu_status()

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
            smpl_joints = None
            smpl_verts = None
            smpl_faces_t = None
            kp2d = None
            kp_conf = None

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

            # Clamp hand indices to available joint count
            actual_hand_indices = None
            if hand_indices is not None and smpl_joints is not None:
                num_joints = smpl_joints.shape[0]
                actual_hand_indices = [i for i in hand_indices if i < num_joints]
                if not actual_hand_indices:
                    actual_hand_indices = None

            # Temporal
            se3_prev = pose_history[-2] if len(pose_history) >= 2 else None
            se3_curr = pose_history[-1] if len(pose_history) >= 1 else None

            log = step4_training_step(
                human_gs=human_gs,
                object_gs=object_gs,
                joint_renderer=joint_renderer,
                gt_image=gt_image,
                mask_visible=m_vis,
                mask_primary_occ=m_pri,
                mask_secondary_occ=m_sec,
                optimizer=optimizer,
                smpl_joints_3d=smpl_joints,
                hand_joint_indices=actual_hand_indices,
                keypoints_2d=kp2d,
                kp_confidence=kp_conf,
                smpl_vertices=smpl_verts,
                smpl_faces=smpl_faces_t,
                sdf_module=sdf_module,
                se3_poses_prev=se3_prev,
                se3_poses_curr=se3_curr,
                se3_poses_next=_get_se3_pose(se3_object) if se3_curr is not None else None,
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

            # Check for NaN/Inf
            if not np.isfinite(log["loss_total"]):
                print(f"\n[Train] ERROR: Loss is {log['loss_total']} at step {global_step}")
                print("[Train] Aborting training.")
                return all_losses

            # Logging
            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t_start
                steps_done = global_step
                steps_total = num_epochs * num_iters
                eta = elapsed / steps_done * (steps_total - steps_done) if steps_done > 0 else 0
                lr_current = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch}/{num_epochs}][{step}/{num_iters}] "
                    f"L_total={log['loss_total']:.4f} "
                    f"L_render={log['loss_render']:.4f} "
                    f"L_contact={log['loss_contact']:.4f} "
                    f"L_j2d={log['loss_j2d']:.5f} "
                    f"L_pen={log['loss_penetration']:.5f} "
                    f"L_acc={log['loss_temporal']:.5f} "
                    f"lr={lr_current:.2e} "
                    f"ETA={eta:.0f}s"
                )
                print_gpu_status()

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
        print(f"  [Save] {ckpt_path}")

        # Also save as latest
        latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
        torch.save(ckpt, latest_path)

        # Epoch summary
        avg_loss = np.mean([l["loss_total"] for l in epoch_losses])
        avg_render = np.mean([l["loss_render"] for l in epoch_losses])
        print(f"  Epoch {epoch} avg: total={avg_loss:.4f}, render={avg_render:.4f}")

    total_time = time.time() - t_start
    print(f"\n[Train] Training complete in {total_time:.1f}s")
    return all_losses


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Unified training entry point."""
    print("=" * 60)
    print("  HDM / Uni-HOI — End-to-End Training (Step 5)")
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
            cfg.step5 = {
                "num_epochs": 2,
                "num_iters_per_epoch": 500,
            }

    # Override epochs from model.epochs if provided
    if cfg.get("model", {}).get("epochs"):
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.step5.num_epochs = cfg.model.epochs

    device = torch.device(cfg.data_prep.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Resolve paths
    paths = resolve_paths(cfg)
    print(f"[Train] Base dir: {paths['base_dir']}")

    # Create output directory (Hydra manages cwd, but we also create explicit dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save Hydra config
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"[Train] Config saved to {config_save_path}")

    # Load data
    print("\n[Train] Loading training data...")
    data = load_training_data(paths, cfg, device)

    # Run training
    print("\n[Train] Starting training...")
    losses = run_training(cfg, data, output_dir, device)

    # Summary
    if losses:
        final_loss = losses[-1]["loss_total"]
        print(f"\n[Train] Final loss: {final_loss:.4f}")
        all_finite = all(np.isfinite(l["loss_total"]) for l in losses)
        print(f"[Train] All losses finite: {all_finite}")
    print(f"[Train] Outputs saved to: {output_dir}")

    cleanup_gpu()


if __name__ == "__main__":
    main()
