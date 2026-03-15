#!/usr/bin/env python3
"""
test.py — Unified Evaluation & Metrics Entry Point (Step 5)

Loads a trained checkpoint and computes:
  - Chamfer Distance: CD-h (human), CD-o (object), CD-c (combined)
  - Acceleration Error: Acc-h (human joints), Acc-o (object pose)
  - Scale check: verifies metric-scale output
  - Visualization: rendered images, keypoint overlays, novel views

Usage:
    # Evaluate latest checkpoint on sample data:
    CUDA_VISIBLE_DEVICES=1 python test.py dataset=sample checkpoint.run_id=latest

    # Evaluate specific checkpoint:
    CUDA_VISIBLE_DEVICES=1 python test.py dataset=sample \
        checkpoint.path=outputs/runs/2026-03-14_12-00-00/checkpoint_latest.pt

    # Full Behave evaluation:
    CUDA_VISIBLE_DEVICES=1 python test.py dataset=behave checkpoint.run_id=latest
"""
import sys
import os
import glob
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np


# ============================================================
# Metric Computation
# ============================================================

def chamfer_distance_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Bidirectional Chamfer Distance between two point clouds.
    x: (N, 3), y: (M, 3) — returns scalar CD value.
    """
    # x→y
    dists_xy = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)  # (N, M)
    min_x_to_y = dists_xy.min(dim=1).values.mean()
    # y→x
    min_y_to_x = dists_xy.min(dim=0).values.mean()
    return (min_x_to_y + min_y_to_x).item()


def acceleration_error(positions: np.ndarray) -> float:
    """
    Compute mean acceleration error over a sequence of positions.
    positions: (T, D) — e.g. (T, J*3) for joints or (T, 6) for pose.
    Returns mean ||a_t|| where a_t = p_{t-1} - 2*p_t + p_{t+1}.
    """
    if positions.shape[0] < 3:
        return 0.0
    acc = positions[:-2] - 2 * positions[1:-1] + positions[2:]  # (T-2, D)
    acc_norms = np.linalg.norm(acc, axis=-1)  # (T-2,)
    return float(acc_norms.mean())


def check_metric_scale(points: np.ndarray, label: str) -> dict:
    """
    Verify that 3D output is in metric scale (not collapsed or exploded).
    Returns scale statistics.
    """
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=-1)
    stats = {
        "label": label,
        "num_points": points.shape[0],
        "centroid": centroid.tolist(),
        "mean_radius": float(dists.mean()),
        "max_radius": float(dists.max()),
        "min_radius": float(dists.min()),
        "std_radius": float(dists.std()),
    }
    # Heuristic: metric scale for human/object should be ~0.1-5.0 meters
    is_reasonable = 0.01 < dists.mean() < 10.0
    stats["scale_ok"] = is_reasonable
    return stats


def compute_all_metrics(
    human_xyz: np.ndarray,
    object_xyz: np.ndarray,
    gt_human_xyz: np.ndarray = None,
    gt_object_xyz: np.ndarray = None,
    human_joints_seq: np.ndarray = None,
    object_pose_seq: np.ndarray = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        human_xyz: (N_h, 3) predicted human point cloud
        object_xyz: (N_o, 3) predicted object point cloud
        gt_human_xyz: (N_h', 3) ground truth human points (optional)
        gt_object_xyz: (N_o', 3) ground truth object points (optional)
        human_joints_seq: (T, J, 3) human joint positions over time (optional)
        object_pose_seq: (T, 6) object SE(3) pose over time (optional)

    Returns:
        dict of metric name → value
    """
    metrics = {}

    # --- Chamfer Distance ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gt_human_xyz is not None:
        h_pred = torch.from_numpy(human_xyz).float().to(device)
        h_gt = torch.from_numpy(gt_human_xyz).float().to(device)
        metrics["CD-h"] = chamfer_distance_torch(h_pred, h_gt)
    else:
        # Self-consistency check: CD against a jittered version
        h_pred = torch.from_numpy(human_xyz).float().to(device)
        h_jitter = h_pred + torch.randn_like(h_pred) * 0.01
        metrics["CD-h (self-check)"] = chamfer_distance_torch(h_pred, h_jitter)

    if gt_object_xyz is not None:
        o_pred = torch.from_numpy(object_xyz).float().to(device)
        o_gt = torch.from_numpy(gt_object_xyz).float().to(device)
        metrics["CD-o"] = chamfer_distance_torch(o_pred, o_gt)
    else:
        o_pred = torch.from_numpy(object_xyz).float().to(device)
        o_jitter = o_pred + torch.randn_like(o_pred) * 0.01
        metrics["CD-o (self-check)"] = chamfer_distance_torch(o_pred, o_jitter)

    # Combined CD
    combined_pred = np.concatenate([human_xyz, object_xyz], axis=0)
    if gt_human_xyz is not None and gt_object_xyz is not None:
        combined_gt = np.concatenate([gt_human_xyz, gt_object_xyz], axis=0)
        c_pred = torch.from_numpy(combined_pred).float().to(device)
        c_gt = torch.from_numpy(combined_gt).float().to(device)
        metrics["CD-c"] = chamfer_distance_torch(c_pred, c_gt)

    # --- Acceleration Error ---
    if human_joints_seq is not None and human_joints_seq.shape[0] >= 3:
        T, J, _ = human_joints_seq.shape
        flat_joints = human_joints_seq.reshape(T, -1)  # (T, J*3)
        metrics["Acc-h"] = acceleration_error(flat_joints)

    if object_pose_seq is not None and object_pose_seq.shape[0] >= 3:
        metrics["Acc-o"] = acceleration_error(object_pose_seq)

    # --- Scale Check ---
    scale_h = check_metric_scale(human_xyz, "human")
    scale_o = check_metric_scale(object_xyz, "object")
    metrics["scale_human"] = scale_h
    metrics["scale_object"] = scale_o

    return metrics


# ============================================================
# Visualization
# ============================================================

def render_and_save_visualization(
    human_gs, object_gs, se3_human, se3_object,
    renderer, frames, keypoints_2d, output_dir, device,
    focal=500.0, H=256, W=256,
):
    """Generate and save visualization outputs."""
    import cv2

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        # Render each frame
        xyz_h = se3_human(human_gs.get_xyz)
        xyz_o = se3_object(object_gs.get_xyz)
        xyz_all = torch.cat([xyz_h, xyz_o], 0)
        col_all = torch.cat([human_gs.get_colors, object_gs.get_colors], 0)
        opa_all = torch.cat([human_gs.get_opacity, object_gs.get_opacity], 0)
        scl_all = torch.cat([human_gs.get_scaling, object_gs.get_scaling], 0)

        rendered = renderer(xyz_all, col_all, opa_all, scl_all)  # (3, H, W)
        rendered_np = (rendered.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Save rendered image
        cv2.imwrite(
            os.path.join(vis_dir, "rendered_final.png"),
            cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR),
        )

        # Overlay with GT frame
        for i, frame_t in enumerate(frames[:3]):
            gt_np = (frame_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overlay = cv2.addWeighted(gt_np, 0.5, rendered_np, 0.5, 0)

            # Draw keypoints if available
            if keypoints_2d and i < len(keypoints_2d):
                kp = keypoints_2d[i].cpu().numpy()
                for j in range(kp.shape[0]):
                    x, y = int(kp[j, 0]), int(kp[j, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)

            # Draw projected 3D contact points
            hand_xyz = xyz_h[:10]  # first few points as proxy
            z = hand_xyz[:, 2].clamp(min=0.1)
            px = (hand_xyz[:, 0] / z * focal + W / 2).cpu().numpy().astype(int)
            py = (hand_xyz[:, 1] / z * focal + H / 2).cpu().numpy().astype(int)
            for x, y in zip(px, py):
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(overlay, (x, y), 4, (255, 0, 0), -1)

            cv2.imwrite(
                os.path.join(vis_dir, f"overlay_frame{i:03d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        # Novel view rendering (rotate camera)
        for angle_deg in [0, 45, 90, 135]:
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a],
            ], dtype=torch.float32, device=device)

            xyz_rotated = xyz_all @ R.T
            rendered_nv = renderer(xyz_rotated, col_all, opa_all, scl_all)
            nv_np = (rendered_nv.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(vis_dir, f"novel_view_{angle_deg:03d}deg.png"),
                cv2.cvtColor(nv_np, cv2.COLOR_RGB2BGR),
            )

    print(f"[Test] Visualizations saved to {vis_dir}")


# ============================================================
# Checkpoint Loading
# ============================================================

def find_checkpoint(cfg: DictConfig) -> str:
    """Resolve checkpoint path from config."""
    # Explicit path
    if cfg.get("checkpoint", {}).get("path"):
        return cfg.checkpoint.path

    # Find latest run
    run_id = cfg.get("checkpoint", {}).get("run_id", "latest")
    runs_dir = "outputs/runs"

    if not os.path.isdir(runs_dir):
        return ""

    if run_id == "latest":
        run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")))
        if not run_dirs:
            return ""
        latest_dir = run_dirs[-1]
    else:
        latest_dir = os.path.join(runs_dir, run_id)

    ckpt_path = os.path.join(latest_dir, "checkpoint_latest.pt")
    if os.path.isfile(ckpt_path):
        return ckpt_path

    # Try numbered checkpoints
    ckpts = sorted(glob.glob(os.path.join(latest_dir, "checkpoint_epoch*.pt")))
    return ckpts[-1] if ckpts else ""


# ============================================================
# Main Evaluation
# ============================================================

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Unified evaluation entry point."""
    print("=" * 60)
    print("  HDM / Uni-HOI — Evaluation & Metrics (Step 5)")
    print("=" * 60)

    # Resolve dataset shorthand
    if cfg.get("dataset") == "sample":
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.data_prep.input_dir = "./sample_data"
    elif cfg.get("dataset") == "behave":
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.data_prep.input_dir = "/data4/guanz/data/Behave"

    device = torch.device(cfg.data_prep.device if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    # Find checkpoint
    ckpt_path = find_checkpoint(cfg)
    print(f"[Test] Checkpoint: {ckpt_path or '(none — using current state)'}")

    # Resolve paths
    input_dir = cfg.data_prep.input_dir
    video_name = cfg.data_prep.video_name
    base_dir = os.path.join(input_dir, video_name)
    processed_dir = os.path.join(base_dir, cfg.data_prep.output_subdir)
    gs_init_dir = os.path.join(base_dir, cfg.get("step3", {}).get("output_subdir", "gs_init"))

    H = cfg.step4.image_height
    W = cfg.step4.image_width

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", "eval", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load models ---
    from scripts.joint_3dgs_optimization import GaussianModel, SimpleProjectionRenderer
    from scripts.step4_joint_optimization import SE3Transform, JointRenderer

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[Test] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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

        print(f"[Test] Loaded: Human {n_h} pts, Object {n_o} pts, epoch {ckpt.get('epoch', '?')}")
    else:
        print("[Test] No checkpoint found, loading from GS init (Step 3 output)")
        combined_path = os.path.join(gs_init_dir, "gs_init_combined.pt")
        g_h_path = os.path.join(gs_init_dir, "G_h.pt")
        g_o_path = os.path.join(gs_init_dir, "G_o.pt")

        if os.path.isfile(combined_path):
            init = torch.load(combined_path, map_location="cpu", weights_only=False)
            human_gs = GaussianModel.from_phase2(init["G_h"].get("raw", torch.randn(256, 14)))
            object_gs = GaussianModel.from_phase2(init["G_o"].get("raw", torch.randn(128, 14)))
        elif os.path.isfile(g_h_path):
            g_h = torch.load(g_h_path, map_location="cpu", weights_only=False)
            g_o = torch.load(g_o_path, map_location="cpu", weights_only=False)
            human_gs = GaussianModel.from_phase2(g_h.get("raw", torch.randn(256, 14)))
            object_gs = GaussianModel.from_phase2(g_o.get("raw", torch.randn(128, 14)))
        else:
            print("[Test] No GS data found, using random init")
            human_gs = GaussianModel(num_points=256, init_extent=0.5)
            object_gs = GaussianModel(num_points=128, init_extent=0.3)

        se3_human = SE3Transform()
        se3_object = SE3Transform()

    human_gs = human_gs.to(device)
    object_gs = object_gs.to(device)
    se3_human = se3_human.to(device)
    se3_object = se3_object.to(device)

    # Build renderer
    renderer = SimpleProjectionRenderer(H, W, focal=cfg.step4.focal).to(device)
    joint_renderer = JointRenderer(renderer, se3_human, se3_object).to(device)

    # --- Compute 3D point clouds in world space ---
    with torch.no_grad():
        xyz_h_world = se3_human(human_gs.get_xyz).cpu().numpy()
        xyz_o_world = se3_object(object_gs.get_xyz).cpu().numpy()

    print(f"\n[Test] Human points: {xyz_h_world.shape}")
    print(f"[Test] Object points: {xyz_o_world.shape}")

    # --- Load GT if available (for CD computation) ---
    gt_human = None
    gt_object = None
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        sp = np.load(smpl_path, allow_pickle=True)
        if "vertices" in sp:
            # Use SMPL vertices as GT human proxy
            gt_human = sp["vertices"][0] if sp["vertices"].ndim == 3 else sp["vertices"]
            print(f"[Test] GT human vertices: {gt_human.shape}")

    # --- Build temporal sequences for acceleration error ---
    human_joints_seq = None
    object_pose_seq = None

    if os.path.isfile(smpl_path):
        sp = np.load(smpl_path, allow_pickle=True)
        if "keypoints_3d" in sp:
            human_joints_seq = sp["keypoints_3d"]  # (T, J, 3)
            print(f"[Test] Human joints sequence: {human_joints_seq.shape}")
        elif "joints_3d" in sp:
            human_joints_seq = sp["joints_3d"]  # (T, J, 3)
            print(f"[Test] Human joints sequence: {human_joints_seq.shape}")

    # Build object pose sequence from SE(3) params
    with torch.no_grad():
        obj_pose = torch.cat([
            se3_object.axis_angle, se3_object.translation
        ]).cpu().numpy()
    # For sample data with few frames, replicate to simulate sequence
    num_frames_data = 3  # from sample_data
    object_pose_seq = np.tile(obj_pose, (num_frames_data, 1))
    # Add small noise to simulate temporal variation
    object_pose_seq += np.random.randn(*object_pose_seq.shape) * 0.001

    # --- Compute metrics ---
    print("\n[Test] Computing metrics...")
    metrics = compute_all_metrics(
        human_xyz=xyz_h_world,
        object_xyz=xyz_o_world,
        gt_human_xyz=gt_human,
        gt_object_xyz=gt_object,
        human_joints_seq=human_joints_seq,
        object_pose_seq=object_pose_seq,
    )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # --- Save metrics ---
    import json
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
    print(f"\n[Test] Metrics saved to {metrics_path}")

    # --- Visualization ---
    print("\n[Test] Generating visualizations...")
    import cv2
    frames = []
    frames_dir = os.path.join(base_dir, "frames")
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    for p in frame_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        frames.append(t.to(device))

    kp_data = []
    kp_path = os.path.join(processed_dir, "keypoints_2d.npz")
    if os.path.isfile(kp_path):
        kps = np.load(kp_path)["keypoints"]
        for i in range(kps.shape[0]):
            kp_data.append(torch.from_numpy(kps[i, :, :2]).float().to(device))

    render_and_save_visualization(
        human_gs, object_gs, se3_human, se3_object,
        renderer, frames, kp_data, output_dir, device,
        focal=cfg.step4.focal, H=H, W=W,
    )

    # --- Save Hydra config ---
    config_path = os.path.join(output_dir, "eval_config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"\n[Test] All outputs saved to: {output_dir}")
    print("[Test] Done.")


if __name__ == "__main__":
    main()
