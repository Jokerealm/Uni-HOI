#!/usr/bin/env python3
"""
Dataset-Level Training for BEHAVE
==================================
Proper epoch-based training over the full BEHAVE dataset.

1 epoch = 1 pass over ALL sequences (shuffled).
For each sequence in the epoch, we do K optimization steps
on randomly sampled frames from that sequence.

Per-sequence models (GaussianModel + SE3) are maintained in a
SequenceModelRegistry — kept in CPU, swapped to GPU as needed.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cari4d \\
        python scripts/legacy_train_dataset.py \\
        --epochs 3 --steps_per_seq 50 --batch_size 8

    # Resume from checkpoint:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cari4d \\
        python scripts/legacy_train_dataset.py --resume checkpoints/registry_latest.pt
"""
import sys
import os
import time
import gc
import argparse
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np



def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def train_one_sequence(
    seq_name: str,
    seq_dir: str,
    registry,
    cfg: dict,
    device: torch.device,
    global_step: int,
    wandb_run=None,
):
    """
    Run K optimization steps on one sequence.
    Returns (updated global_step, loss_dict, success).
    """
    from dataset.behave_train_dataset import load_sequence_data
    from scripts.step4_joint_optimization import (
        SE3Transform, JointRenderer, SimpleProjectionRenderer,
        VolumetricSMPLSDF, step4_training_step,
    )

    H, W = cfg["image_height"], cfg["image_width"]
    steps_per_seq = cfg["steps_per_seq"]
    batch_size = cfg["batch_size"]

    # Load sequence data to GPU
    data = load_sequence_data(
        seq_dir, image_size=(H, W), cam_id=1, device=device,
    )
    if data is None or data["num_frames"] == 0:
        return global_step, None, False

    num_frames = data["num_frames"]

    # Get or init per-sequence models
    gs_init_dir = os.path.join(seq_dir, "gs_init")
    human_gs, object_gs, se3_human, se3_object, opt_state = registry.get_or_init(
        seq_name, gs_init_dir, device, num_frames=num_frames,
    )

    # Build renderer
    base_renderer = SimpleProjectionRenderer(H, W, focal=cfg["focal"]).to(device)
    joint_renderer = JointRenderer(base_renderer, se3_human, se3_object).to(device)

    # SDF module
    sdf_module = None
    if cfg["pen_enabled"]:
        sdf_module = VolumetricSMPLSDF(
            resolution=cfg["sdf_resolution"], padding=cfg["sdf_padding"],
        ).to(device)

    # Optimizer
    param_groups = [
        {"params": [human_gs.xyz, object_gs.xyz], "lr": cfg["lr_xyz"]},
        {"params": [human_gs.opacity, object_gs.opacity], "lr": cfg["lr_opacity"]},
        {"params": [human_gs.scaling, object_gs.scaling], "lr": cfg["lr_scaling"]},
        {"params": [human_gs.rotation, object_gs.rotation], "lr": cfg["lr_rotation"]},
        {"params": [human_gs.shs, object_gs.shs], "lr": cfg["lr_color"]},
        {"params": [se3_human.translation, se3_object.translation], "lr": cfg["lr_se3_t"]},
        {"params": [se3_human.axis_angle, se3_object.axis_angle], "lr": cfg["lr_se3_r"]},
    ]
    optimizer = torch.optim.Adam(param_groups)

    # Restore optimizer state if available
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception:
            pass  # Shape mismatch etc — just use fresh optimizer

    hand_indices = list(range(20, 52)) if cfg["contact_enabled"] else None
    focal = cfg["focal"]

    seq_losses = []

    for step in range(1, steps_per_seq + 1):
        global_step += 1

        # Sample batch of frame indices
        if num_frames >= batch_size:
            batch_indices = np.random.choice(num_frames, batch_size, replace=False).tolist()
        else:
            batch_indices = list(range(num_frames))

        gt_images = torch.stack([data["frames"][i] for i in batch_indices])
        m_vis = torch.stack([data["masks_visible"][i] for i in batch_indices])
        m_pri = torch.stack([data["masks_primary_occ"][i] for i in batch_indices])
        m_sec = torch.stack([data["masks_secondary_occ"][i] for i in batch_indices])

        # Optional per-frame data
        smpl_joints = smpl_verts = smpl_faces_t = kp2d = kp_conf = None

        if data["smpl_params"] is not None:
            sp = data["smpl_params"]
            if "keypoints_3d" in sp:
                js = [torch.from_numpy(sp["keypoints_3d"][min(i, sp["keypoints_3d"].shape[0]-1)]).float().to(device) for i in batch_indices]
                smpl_joints = torch.stack(js)
            elif "joints_3d" in sp:
                js = [torch.from_numpy(sp["joints_3d"][min(i, sp["joints_3d"].shape[0]-1)]).float().to(device) for i in batch_indices]
                smpl_joints = torch.stack(js)
            if "vertices" in sp:
                vs = [torch.from_numpy(sp["vertices"][min(i, sp["vertices"].shape[0]-1)]).float().to(device) for i in batch_indices]
                smpl_verts = torch.stack(vs)
            if "faces" in sp:
                smpl_faces_t = torch.from_numpy(sp["faces"]).long().to(device)

        if data["keypoints_2d"]:
            kps, confs = [], []
            for i in batch_indices:
                if i < len(data["keypoints_2d"]):
                    kps.append(data["keypoints_2d"][i])
                    confs.append(data["kp_confidence"][i])
            if kps:
                kp2d = torch.stack(kps)
                kp_conf = torch.stack(confs)

        actual_hand_indices = None
        if hand_indices is not None and smpl_joints is not None:
            num_joints = smpl_joints.shape[-2]
            actual_hand_indices = [i for i in hand_indices if i < num_joints]
            if not actual_hand_indices:
                actual_hand_indices = None

        fx_batch = torch.tensor(
            [data["camera_fx"][i] for i in batch_indices],
            dtype=torch.float32,
            device=device,
        )
        fy_batch = torch.tensor(
            [data["camera_fy"][i] for i in batch_indices],
            dtype=torch.float32,
            device=device,
        )
        cx_batch = torch.tensor(
            [data["camera_cx"][i] for i in batch_indices],
            dtype=torch.float32,
            device=device,
        )
        cy_batch = torch.tensor(
            [data["camera_cy"][i] for i in batch_indices],
            dtype=torch.float32,
            device=device,
        )

        log = step4_training_step(
            human_gs=human_gs, object_gs=object_gs,
            joint_renderer=joint_renderer,
            gt_image=gt_images,
            mask_visible=m_vis, mask_primary_occ=m_pri, mask_secondary_occ=m_sec,
            optimizer=optimizer,
            smpl_joints_3d=smpl_joints,
            hand_joint_indices=actual_hand_indices,
            keypoints_2d=kp2d, kp_confidence=kp_conf,
            smpl_vertices=smpl_verts, smpl_faces=smpl_faces_t,
            sdf_module=sdf_module,
            frame_indices=batch_indices,
            se3_human_module=se3_human if cfg["lambda_acc"] > 0 else None,
            se3_object_module=se3_object if cfg["lambda_acc"] > 0 else None,
            w_visible=cfg["w_visible"], w_primary=cfg["w_primary"],
            w_secondary=cfg["w_secondary"], lambda_ssim=cfg["lambda_ssim"],
            lambda_contact=cfg["lambda_contact"],
            lambda_j2d=cfg["lambda_j2d"],
            lambda_pen=cfg["lambda_pen"],
            lambda_acc=cfg["lambda_acc"],
            focal=focal,
            fx=fx_batch,
            fy=fy_batch,
            cx=cx_batch,
            cy=cy_batch,
        )

        seq_losses.append(log)

        if not np.isfinite(log["loss_total"]):
            print(f"    [WARN] NaN/Inf loss at step {step} for {seq_name}")
            break

        # Log scalars to wandb
        if wandb_run is not None:
            import wandb
            wandb.log({
                "loss/total": log["loss_total"],
                "loss/render": log["loss_render"],
                "loss/contact": log["loss_contact"],
                "loss/j2d": log["loss_j2d"],
                "loss/pen": log["loss_penetration"],
                "loss/acc": log["loss_temporal"],
                "seq": seq_name,
            }, step=global_step)

    # Save models back to CPU registry
    registry.save(seq_name, human_gs, object_gs, se3_human, se3_object, optimizer)

    # Free GPU memory
    del human_gs, object_gs, se3_human, se3_object, optimizer
    del base_renderer, joint_renderer, sdf_module
    del data
    cleanup_gpu()

    avg_loss = np.mean([l["loss_total"] for l in seq_losses]) if seq_losses else 0.0
    return global_step, {"avg_loss": avg_loss, "steps": len(seq_losses)}, True


def export_sequence_3d(seq_name, seq_dir, registry, device, output_dir):
    """Export final 3D point clouds for one sequence."""
    import trimesh
    from scripts.step4_joint_optimization import SE3Transform

    human_gs, object_gs, se3_human, se3_object, _ = registry.get_or_init(
        seq_name, os.path.join(seq_dir, "gs_init"), device,
    )

    recon_dir = os.path.join(output_dir, "reconstructions", seq_name)
    os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        frame_idx = 0 if getattr(se3_human, "num_frames", 1) > 1 else None
        xyz_h = se3_human(human_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
        xyz_o = se3_object(object_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
        col_h = (human_gs.get_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        col_o = (object_gs.get_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    for xyz, col, name in [
        (xyz_h, col_h, "human.ply"),
        (xyz_o, col_o, "object.ply"),
        (np.concatenate([xyz_h, xyz_o]), np.concatenate([col_h, col_o]), "scene.ply"),
    ]:
        alpha = np.full((len(col), 1), 255, dtype=np.uint8)
        pc = trimesh.PointCloud(xyz, colors=np.hstack([col, alpha]))
        pc.export(os.path.join(recon_dir, name))

    # Cleanup
    del human_gs, object_gs, se3_human, se3_object
    cleanup_gpu()
    return recon_dir


def main():
    # Force line-buffered stdout for real-time log output
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Dataset-level BEHAVE training")
    parser.add_argument("--behave_dir", type=str,
                        default="/data4/guanz/data/Behave/sequences")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (1 epoch = 1 pass over all sequences)")
    parser.add_argument("--steps_per_seq", type=int, default=50,
                        help="Optimization steps per sequence per epoch")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Frames per optimization step")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--focal", type=float, default=500.0)
    parser.add_argument("--lr_xyz", type=float, default=1.6e-4)
    parser.add_argument("--lr_opacity", type=float, default=5e-2)
    parser.add_argument("--lr_scaling", type=float, default=5e-3)
    parser.add_argument("--lr_rotation", type=float, default=1e-3)
    parser.add_argument("--lr_color", type=float, default=2.5e-3)
    parser.add_argument("--lr_se3_t", type=float, default=1e-3)
    parser.add_argument("--lr_se3_r", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--vis_interval", type=int, default=5000,
                        help="Export 3D vis every N global steps")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Save registry checkpoint every N global steps")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to registry checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs/dataset_train")
    args = parser.parse_args()

    use_wandb = args.wandb and not args.no_wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config dict for training
    cfg = {
        "image_height": args.image_size,
        "image_width": args.image_size,
        "steps_per_seq": args.steps_per_seq,
        "batch_size": args.batch_size,
        "focal": args.focal,
        "lr_xyz": args.lr_xyz,
        "lr_opacity": args.lr_opacity,
        "lr_scaling": args.lr_scaling,
        "lr_rotation": args.lr_rotation,
        "lr_color": args.lr_color,
        "lr_se3_t": args.lr_se3_t,
        "lr_se3_r": args.lr_se3_r,
        # Loss weights
        "w_visible": 1.0, "w_primary": 0.3, "w_secondary": 0.05,
        "lambda_ssim": 0.2,
        "lambda_contact": 0.5,
        "lambda_j2d": 0.1,
        "lambda_pen": 1.0,
        "lambda_acc": 0.5,
        # Modules
        "contact_enabled": True,
        "pen_enabled": True,
        "sdf_resolution": 64,
        "sdf_padding": 0.1,
    }

    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Dataset ---
    from dataset.behave_train_dataset import BehaveTrainDataset, SequenceModelRegistry

    dataset = BehaveTrainDataset(args.behave_dir, image_size=(args.image_size, args.image_size))
    num_sequences = len(dataset)
    print(f"\n{'='*60}")
    print(f"  Dataset-Level Training")
    print(f"  Sequences: {num_sequences}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps/seq/epoch: {args.steps_per_seq}")
    print(f"  Total steps/epoch: {num_sequences * args.steps_per_seq}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # --- Model Registry ---
    registry = SequenceModelRegistry()
    start_epoch = 1
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading from {args.resume}")
        extra = registry.load_checkpoint(args.resume)
        start_epoch = extra.get("epoch", 0) + 1
        global_step = extra.get("global_step", 0)
        print(f"[Resume] Epoch {start_epoch}, global_step {global_step}, "
              f"{registry.num_initialized} sequences loaded")

    # --- WandB ---
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="uni-hoi-4d",
                name=f"dataset_train_{timestamp}",
                config={**cfg, "epochs": args.epochs, "num_sequences": num_sequences},
                tags=["behave", "dataset-level"],
                reinit=True,
            )
            print(f"[WandB] {wandb_run.url}")
        except Exception as e:
            print(f"[WandB] Init failed: {e}")
            use_wandb = False

    # --- Training Loop ---
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Shuffle sequence order each epoch
        seq_order = list(range(num_sequences))
        random.shuffle(seq_order)

        epoch_losses = []
        epoch_done = 0
        epoch_fail = 0

        for i, seq_idx in enumerate(seq_order):
            item = dataset[seq_idx]
            seq_name = item["seq_name"]
            seq_dir = item["seq_dir"]

            t_seq = time.time()
            global_step, result, success = train_one_sequence(
                seq_name=seq_name,
                seq_dir=seq_dir,
                registry=registry,
                cfg=cfg,
                device=device,
                global_step=global_step,
                wandb_run=wandb_run,
            )

            dt = time.time() - t_seq
            if success and result:
                epoch_losses.append(result["avg_loss"])
                epoch_done += 1
                if (i + 1) % 10 == 0 or (i + 1) == num_sequences:
                    elapsed = time.time() - t_start
                    avg = np.mean(epoch_losses[-10:])
                    eta = elapsed / max(global_step, 1) * (
                        args.epochs * num_sequences * args.steps_per_seq - global_step
                    )
                    eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
                    print(
                        f"  [{epoch}][{i+1}/{num_sequences}] {seq_name} "
                        f"loss={result['avg_loss']:.4f} ({dt:.1f}s) "
                        f"avg10={avg:.4f} ETA={eta_str}"
                    )
            else:
                epoch_fail += 1
                print(f"  [{epoch}][{i+1}/{num_sequences}] {seq_name} SKIP/FAIL ({dt:.1f}s)")

            # Periodic checkpoint
            if global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"registry_step{global_step:07d}.pt")
                registry.save_checkpoint(ckpt_path, extra={
                    "epoch": epoch, "global_step": global_step,
                })
                latest_path = os.path.join(ckpt_dir, "registry_latest.pt")
                registry.save_checkpoint(latest_path, extra={
                    "epoch": epoch, "global_step": global_step,
                })
                print(f"  [Save] {ckpt_path}")
                if wandb_run:
                    import wandb
                    wandb.save(ckpt_path)

            # Periodic 3D visualization
            if global_step > 0 and global_step % args.vis_interval == 0:
                try:
                    recon_dir = export_sequence_3d(
                        seq_name, seq_dir, registry, device, output_dir,
                    )
                    print(f"  [Vis] Exported 3D: {recon_dir}")
                    if wandb_run:
                        import wandb
                        for ply_name in ["human.ply", "object.ply", "scene.ply"]:
                            ply_path = os.path.join(recon_dir, ply_name)
                            if os.path.isfile(ply_path):
                                wandb.save(ply_path)
                except Exception as e:
                    print(f"  [Vis] Export failed: {e}")

        # End of epoch
        avg_epoch = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"\n  Epoch {epoch} done: {epoch_done} ok, {epoch_fail} fail, "
              f"avg_loss={avg_epoch:.4f}")

        if wandb_run:
            import wandb
            wandb.log({
                "epoch": epoch,
                "epoch/avg_loss": avg_epoch,
                "epoch/done": epoch_done,
                "epoch/fail": epoch_fail,
            }, step=global_step)

        # Save epoch checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"registry_epoch{epoch:03d}.pt")
        registry.save_checkpoint(ckpt_path, extra={
            "epoch": epoch, "global_step": global_step,
        })
        latest_path = os.path.join(ckpt_dir, "registry_latest.pt")
        registry.save_checkpoint(latest_path, extra={
            "epoch": epoch, "global_step": global_step,
        })
        print(f"  [Save] {ckpt_path}")

    # --- Final export ---
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Total time: {total_time/3600:.1f}h")
    print(f"  Global steps: {global_step}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
