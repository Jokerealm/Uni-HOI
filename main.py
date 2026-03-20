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
import glob
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
      - BEHAVE:       <input_dir>/<video_name>/t*.000/k1.color.jpg
    """
    input_dir = cfg.data_prep.input_dir
    video_name = cfg.data_prep.video_name
    base_dir = os.path.join(input_dir, video_name)
    is_behave = cfg.get("dataset") == "behave" or bool(
        glob.glob(os.path.join(base_dir, "t*.000"))
    )

    return {
        "base_dir": base_dir,
        "frames_dir": base_dir if is_behave else os.path.join(base_dir, "frames"),
        "is_behave": is_behave,
        "processed_dir": os.path.join(base_dir, cfg.data_prep.output_subdir),
        "cropped_dir": os.path.join(base_dir, cfg.data_prep.output_subdir, "cropped"),
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
    pipeline_cfg.data_prep.behave_cam_id = int(cfg.get("behave_cam_id", 1))
    pipeline_cfg.data_prep.device = dp.device
    pipeline_cfg.data_prep.scale_ratio = int(cfg.get("scale_ratio", 1))
    pipeline_cfg.data_prep.bbox_expand = float(cfg.get("bbox_expand", 1.0))
    pipeline_cfg.data_prep.crop_size = (
        int(cfg.get("step4", {}).get("image_height", 256)),
        int(cfg.get("step4", {}).get("image_width", 256)),
    )

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

    pipeline = None
    try:
        pipeline = Step1Pipeline(pipeline_cfg)
        pipeline.run()
    finally:
        if pipeline is not None:
            del pipeline
        cleanup_gpu()
    print("[Step 1] Done.\n")


def run_step2(cfg: DictConfig):
    """Step 2: Amodal video generation."""
    from configs.step2_config import Step2PipelineConfig, ProPainterConfig
    from configs.step3_config import FlowMatchingInferenceConfig

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
    fm_cfg = FlowMatchingInferenceConfig(
        checkpoint=cfg.fm.checkpoint,
        model_type=cfg.fm.model_type,
        video_channels=cfg.fm.video_channels,
        video_input_channels=cfg.fm.video_input_channels,
        point_channels=cfg.fm.point_channels,
        mask_channels=cfg.fm.mask_channels,
        dim=cfg.fm.dim,
        depth=cfg.fm.depth,
        num_heads=cfg.fm.num_heads,
        cond_dim=cfg.fm.cond_dim,
        num_ode_steps=cfg.fm.num_ode_steps,
        num_frames=cfg.fm.num_frames,
        video_h=cfg.fm.video_h,
        video_w=cfg.fm.video_w,
        num_points=cfg.fm.num_points,
        prior_noise_std=cfg.fm.prior_noise_std,
        clamp_visible_rgb=cfg.fm.clamp_visible_rgb,
        save_frames=cfg.fm.save_frames,
        save_fps=cfg.fm.save_fps,
    )
    dp = cfg.data_prep
    input_dir = dp.input_dir

    pipeline_cfg = Step2PipelineConfig(
        base_weights_dir=cfg.base_weights_dir,
        project_root=cfg.project_root,
        backend=cfg.get("amodal", {}).get("method", "propainter"),
        input_dir=input_dir,
        video_name=dp.video_name,
        processed_subdir=dp.output_subdir,
        output_subdir=cfg.amodal.output_subdir,
        gs_output_subdir=cfg.get("step3", {}).get("output_subdir", "gs_init"),
        max_frames=dp.get("max_frames", None),
        behave_cam_id=int(cfg.get("behave_cam_id", 1)),
        device=dp.device,
        propainter=pp_cfg,
        fm=fm_cfg,
    )
    pipeline = None
    try:
        if pipeline_cfg.backend == "dual_branch_flow_matching":
            from pipeline.step2_dual_branch_flow_matching import DualBranchFlowMatchingPipeline
            pipeline = DualBranchFlowMatchingPipeline(pipeline_cfg)
        elif pipeline_cfg.backend == "joint_flow_matching":
            from pipeline.step2_joint_flow_matching import JointFlowMatchingPipeline
            pipeline = JointFlowMatchingPipeline(pipeline_cfg)
        else:
            from pipeline.step2_amodal_completion import Step2Pipeline as ProPainterPipeline
            pipeline = ProPainterPipeline(pipeline_cfg)
        pipeline.run()
    finally:
        if pipeline is not None:
            del pipeline
        cleanup_gpu()
    print("[Step 2] Done.\n")


def run_step3(cfg: DictConfig):
    """Step 3: Zero-shot 3D lifting via Hunyuan3D-2 + metric alignment."""
    if cfg.get("amodal", {}).get("method", "propainter") in {"joint_flow_matching", "dual_branch_flow_matching"}:
        gs_dir = os.path.join(
            cfg.data_prep.input_dir,
            cfg.data_prep.video_name,
            cfg.get("step3", {}).get("output_subdir", "gs_init"),
        )
        combined_path = os.path.join(gs_dir, "gs_init_combined.pt")
        if os.path.isfile(combined_path):
            print("[Step 3] Skipped: joint Flow Matching backend already generated 3DGS in Step 2.\n")
            return

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
        image_height=int(cfg.get("step4", {}).get("image_height", 256)),
        image_width=int(cfg.get("step4", {}).get("image_width", 256)),
        focal=float(cfg.get("step4", {}).get("focal", 500.0)),
        frame_selection=cfg.get("step3", {}).get("frame_selection", "middle"),
        hy3d=hy3d_cfg,
        alignment=OmegaConf.to_container(cfg.get("alignment", {}), resolve=True),
        run_alignment=cfg.get("alignment", {}).get("enabled", True),
    )
    pipeline = None
    try:
        pipeline = Step3Pipeline(pipeline_cfg)
        pipeline.run()
    finally:
        if pipeline is not None:
            del pipeline
        cleanup_gpu()
    print("[Step 3] Done.\n")


def load_training_data(paths: dict, cfg: DictConfig, device: torch.device) -> dict:
    """Load all pre-computed data from Steps 1-3 for Step 4 training.

    Prefers cropped frames from the offline spatial preprocessing pipeline
    (CARI4D-style: 2x downsample → bbox crop → 224×224). Falls back to
    raw frames with naive resize if cropped data is not available.
    """
    import cv2
    import glob
    from dataset.video_transforms import (
        infer_camera_intrinsics,
        normalize_imagenet_tensor,
        resize_intrinsics_to_image,
        resize_keypoints_to_image,
        validate_pixel_keypoints,
    )

    H = cfg.step4.image_height
    W = cfg.step4.image_width

    data = {
        "frames": [], "frames_normalized": [],
        "masks_visible": [], "masks_primary_occ": [],
        "masks_secondary_occ": [], "keypoints_2d": [], "kp_confidence": [],
        "camera_fx": [], "camera_fy": [], "camera_cx": [], "camera_cy": [],
        "smpl_params": None, "human_gs": None, "object_gs": None,
    }
    frame_sizes_hw = []

    # --- Try loading cropped frames first (from offline preprocessing) ---
    processed_dir = paths["processed_dir"]
    cropped_dir = os.path.join(processed_dir, "cropped")
    use_cropped = os.path.isdir(os.path.join(cropped_dir, "rgb"))

    if use_cropped:
        frame_paths = sorted(
            glob.glob(os.path.join(cropped_dir, "rgb", "*.png"))
            + glob.glob(os.path.join(cropped_dir, "rgb", "*.jpg"))
        )
        print(f"[Data] Using pre-cropped frames from {cropped_dir}")
    else:
        raise FileNotFoundError(
            f"Missing cropped Step-1 assets under {cropped_dir}/rgb. "
            "Step 4 now requires the offline cropped training patches and will no longer "
            "fall back to raw-frame resizing."
        )

    assert len(frame_paths) > 0, f"No frames found"

    for p in frame_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_sizes_hw.append(tuple(int(v) for v in img.shape[:2]))
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3, H, W) in [0,1]
        data["frames"].append(t.to(device))
        # ImageNet-normalized version for feature extraction
        t_norm = normalize_imagenet_tensor(t)
        data["frames_normalized"].append(t_norm.to(device))

    num_frames = len(data["frames"])
    print(f"[Data] Loaded {num_frames} frames at {H}x{W} "
          f"({'cropped' if use_cropped else 'raw resize'})")

    # --- ROI camera parameters for cropped training patches ---
    if use_cropped:
        meta_path = os.path.join(cropped_dir, "meta.npz")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Missing ROI intrinsics metadata: {meta_path}. "
                "Step 4 requires cropped intrinsics from Step 1."
            )
        meta = np.load(meta_path)
        for key_src, key_dst in [
            ("fx", "camera_fx"),
            ("fy", "camera_fy"),
            ("cx", "camera_cx"),
            ("cy", "camera_cy"),
        ]:
            vals = meta[key_src]
            data[key_dst] = [float(v) for v in vals[:num_frames]]
        print(f"[Data] Loaded ROI intrinsics from {meta_path}")
    else:
        for src_h, src_w in frame_sizes_hw[:num_frames]:
            fx_src, fy_src, cx_src, cy_src = infer_camera_intrinsics(
                image_width=src_w,
                image_height=src_h,
                scale_ratio=1,
            )
            fx_dst, fy_dst, cx_dst, cy_dst = resize_intrinsics_to_image(
                fx_src, fy_src, cx_src, cy_src,
                src_size_hw=(src_h, src_w),
                dst_size_hw=(H, W),
            )
            data["camera_fx"].append(float(fx_dst))
            data["camera_fy"].append(float(fy_dst))
            data["camera_cx"].append(float(cx_dst))
            data["camera_cy"].append(float(cy_dst))
    for key, default in [
        ("camera_fx", float(cfg.step4.focal)),
        ("camera_fy", float(cfg.step4.focal)),
        ("camera_cx", W / 2.0),
        ("camera_cy", H / 2.0),
    ]:
        if len(data[key]) < num_frames:
            data[key].extend([default] * (num_frames - len(data[key])))

    # --- Load Step 1 outputs (masks, keypoints, SMPL) ---
    processed_dir = paths["processed_dir"]

    region_masks_path = os.path.join(processed_dir, "region_masks.npz")
    if use_cropped:
        cropped_region_masks_path = os.path.join(cropped_dir, "region_masks.npz")
        if os.path.isfile(cropped_region_masks_path):
            region_masks_path = cropped_region_masks_path
    if os.path.isfile(region_masks_path):
        import cv2 as _cv2
        rm = np.load(region_masks_path)
        # Load arrays once to avoid repeated npz decompression
        arr_obj = np.array(rm["M_object"], dtype=np.float32)
        arr_p = np.array(rm["M_p"], dtype=np.float32)
        arr_s = np.array(rm["M_s"], dtype=np.float32)
        n_masks = arr_obj.shape[0]
        for i in range(num_frames):
            idx = min(i, n_masks - 1)
            m_obj = _cv2.resize(arr_obj[idx], (W, H), interpolation=_cv2.INTER_NEAREST)
            m_p = _cv2.resize(arr_p[idx], (W, H), interpolation=_cv2.INTER_NEAREST)
            m_s = _cv2.resize(arr_s[idx], (W, H), interpolation=_cv2.INTER_NEAREST)
            data["masks_visible"].append(torch.from_numpy(m_obj).to(device))
            data["masks_primary_occ"].append(torch.from_numpy(m_p).to(device))
            data["masks_secondary_occ"].append(torch.from_numpy(m_s).to(device))
        del arr_obj, arr_p, arr_s
        print(f"[Data] Loaded region masks from {region_masks_path}")
    else:
        raise FileNotFoundError(
            f"Missing region masks for Step 4. Expected {region_masks_path}. "
            "Training no longer falls back to uniform masks."
        )

    # Keypoints
    kp_path = os.path.join(processed_dir, "keypoints_2d.npz")
    if use_cropped:
        cropped_kp_path = os.path.join(cropped_dir, "keypoints_2d.npz")
        if os.path.isfile(cropped_kp_path):
            kp_path = cropped_kp_path
    if os.path.isfile(kp_path):
        kp_data = np.load(kp_path)
        kps = kp_data["keypoints"]  # (T, J, 3) — x, y, confidence
        for i in range(num_frames):
            idx = min(i, kps.shape[0] - 1)
            kp_np = np.asarray(kps[idx], dtype=np.float32)
            if use_cropped:
                kp_np = validate_pixel_keypoints(
                    kp_np,
                    image_size_hw=(H, W),
                    context=f"{kp_path} frame {idx}",
                )
            else:
                kp_np = resize_keypoints_to_image(
                    kp_np,
                    src_size_hw=frame_sizes_hw[i],
                    dst_size_hw=(H, W),
                    context=f"{kp_path} frame {idx}",
                )
            kp = torch.from_numpy(kp_np).float().to(device)
            data["keypoints_2d"].append(kp[:, :2])
            data["kp_confidence"].append(kp[:, 2])
        print(f"[Data] Loaded 2D keypoints: {kps.shape}")
    else:
        legacy_kp_path = os.path.join(processed_dir, "keypoints", "openpose_2d.npz")
        if os.path.isfile(legacy_kp_path):
            kp_data = np.load(legacy_kp_path)
            kps = kp_data["keypoints"]
            for i in range(num_frames):
                idx = min(i, kps.shape[0] - 1)
                kp_np = np.asarray(kps[idx], dtype=np.float32)
                if use_cropped:
                    kp_np = validate_pixel_keypoints(
                        kp_np,
                        image_size_hw=(H, W),
                        context=f"{legacy_kp_path} frame {idx}",
                    )
                else:
                    kp_np = resize_keypoints_to_image(
                        kp_np,
                        src_size_hw=frame_sizes_hw[i],
                        dst_size_hw=(H, W),
                        context=f"{legacy_kp_path} frame {idx}",
                    )
                kp = torch.from_numpy(kp_np).float().to(device)
                data["keypoints_2d"].append(kp[:, :2])
                if kp.shape[1] > 2:
                    data["kp_confidence"].append(kp[:, 2])
                else:
                    data["kp_confidence"].append(torch.ones(kp.shape[0], device=device))
            print(f"[Data] Loaded legacy 2D keypoints: {kps.shape}")
        else:
            raise FileNotFoundError(
                f"Missing 2D keypoints for Step 4. Expected {kp_path} or {legacy_kp_path}."
            )

    # SMPL params
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        data["smpl_params"] = dict(np.load(smpl_path, allow_pickle=True))
        print(f"[Data] Loaded SMPL params from {smpl_path}")
    else:
        legacy_smpl_path = os.path.join(processed_dir, "poses", "smplh_aligned.npz")
        if os.path.isfile(legacy_smpl_path):
            data["smpl_params"] = dict(np.load(legacy_smpl_path, allow_pickle=True))
            print(f"[Data] Loaded legacy SMPL params from {legacy_smpl_path}")
        else:
            raise FileNotFoundError(
                f"Missing SMPL parameters for Step 4. Expected {smpl_path} or {legacy_smpl_path}."
            )

    # Load separate joints_3d if not already in smpl_params
    joints_3d_path = os.path.join(processed_dir, "joints_3d.npz")
    if os.path.isfile(joints_3d_path):
        j3d = np.load(joints_3d_path)
        if "joints_3d" in j3d:
            if data["smpl_params"] is None:
                data["smpl_params"] = {}
            if "joints_3d" not in data["smpl_params"] and "keypoints_3d" not in data["smpl_params"]:
                data["smpl_params"]["joints_3d"] = j3d["joints_3d"]
                print(f"[Data] Loaded 3D joints from {joints_3d_path}: {j3d['joints_3d'].shape}")

    # Log what SMPL data is available for loss computation
    if data["smpl_params"] is not None:
        sp_keys = list(data["smpl_params"].keys())
        has_joints = "joints_3d" in sp_keys or "keypoints_3d" in sp_keys
        has_verts = "vertices" in sp_keys
        has_faces = "faces" in sp_keys
        print(f"[Data] SMPL keys: {sp_keys}")
        print(f"[Data] Loss availability: contact={'yes' if has_joints else 'NO (missing joints_3d)'}, "
              f"j2d={'yes' if has_joints else 'NO'}, "
              f"pen={'yes' if has_verts and has_faces else 'NO (missing vertices/faces)'}")

    # --- Load Step 3 outputs (initial 3DGS) ---
    gs_init_dir = paths["gs_init_dir"]
    from scripts.joint_3dgs_optimization import GaussianModel
    aligned_candidates = []
    if gs_init_dir.endswith("gs_init"):
        aligned_candidates.append(os.path.join(os.path.dirname(gs_init_dir), "gs_aligned"))
        aligned_candidates.append(gs_init_dir + "_aligned")
    else:
        aligned_candidates.append(gs_init_dir + "_aligned")
        aligned_candidates.append(os.path.join(os.path.dirname(gs_init_dir), "gs_aligned"))

    combined_path = os.path.join(gs_init_dir, "gs_init_combined.pt")
    g_h_path = os.path.join(gs_init_dir, "G_h.pt")
    g_o_path = os.path.join(gs_init_dir, "G_o.pt")

    for aligned_dir in aligned_candidates:
        hum_aligned = os.path.join(aligned_dir, "human_gaussians_metric.npz")
        obj_aligned = os.path.join(aligned_dir, "object_gaussians_metric.npz")
        if os.path.isfile(hum_aligned) and os.path.isfile(obj_aligned):
            h_raw = torch.from_numpy(np.load(hum_aligned)["raw"]).float()
            o_raw = torch.from_numpy(np.load(obj_aligned)["raw"]).float()
            data["human_gs"] = GaussianModel.from_phase2(h_raw).to(device)
            data["object_gs"] = GaussianModel.from_phase2(o_raw).to(device)
            print(f"[Data] Loaded metric-aligned GS init from {aligned_dir}")
            break
    else:
        if os.path.isfile(combined_path):
            ckpt = torch.load(combined_path, map_location="cpu", weights_only=False)
            if "raw" not in ckpt.get("G_h", {}) or "raw" not in ckpt.get("G_o", {}):
                raise KeyError(
                    f"Combined GS checkpoint {combined_path} is missing `raw` Gaussian tensors."
                )
            data["human_gs"] = GaussianModel.from_phase2(ckpt["G_h"]["raw"]).to(device)
            data["object_gs"] = GaussianModel.from_phase2(ckpt["G_o"]["raw"]).to(device)
            print(f"[Data] Loaded GS init from {combined_path}")
        elif os.path.isfile(g_h_path) and os.path.isfile(g_o_path):
            g_h = torch.load(g_h_path, map_location="cpu", weights_only=False)
            g_o = torch.load(g_o_path, map_location="cpu", weights_only=False)
            if "raw" not in g_h or "raw" not in g_o:
                raise KeyError(
                    f"Separate GS checkpoints {g_h_path} / {g_o_path} are missing `raw` Gaussian tensors."
                )
            data["human_gs"] = GaussianModel.from_phase2(g_h["raw"]).to(device)
            data["object_gs"] = GaussianModel.from_phase2(g_o["raw"]).to(device)
            print(f"[Data] Loaded GS init from {g_h_path}, {g_o_path}")
        else:
            raise FileNotFoundError(
                f"Missing Step-3 Gaussian initialization under {gs_init_dir}. "
                "Step 4 no longer falls back to random Gaussian initialization."
            )

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

    # --- WandB init ---
    use_wandb = cfg.get("wandb", {}).get("enabled", True)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            video_name = cfg.data_prep.video_name
            wandb_run = wandb.init(
                project=cfg.get("wandb", {}).get("project", "uni-hoi-4d"),
                name=f"{video_name}",
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=[cfg.get("dataset", "unknown"), video_name],
                reinit=True,
            )
            print(f"[WandB] Initialized: {wandb_run.url}")
        except Exception as e:
            print(f"[WandB] Init failed: {e}, continuing without wandb")
            use_wandb = False

    # Load data from Steps 1-3
    data = load_training_data(paths, cfg, device)

    if cfg.step4.contact.enabled:
        smpl_params = data.get("smpl_params") or {}
        if "joints_3d" not in smpl_params and "keypoints_3d" not in smpl_params:
            raise RuntimeError(
                "Step 4 contact loss is enabled, but SMPL joints are missing from Step-1 outputs."
            )
    if cfg.step4.proj2d.enabled:
        smpl_params = data.get("smpl_params") or {}
        if not data["keypoints_2d"]:
            raise RuntimeError("Step 4 projection loss is enabled, but 2D keypoints are missing.")
        if "joints_3d" not in smpl_params and "keypoints_3d" not in smpl_params:
            raise RuntimeError(
                "Step 4 projection loss is enabled, but 3D joints are missing from SMPL outputs."
            )
    if cfg.step4.penetration.enabled:
        smpl_params = data.get("smpl_params") or {}
        if "vertices" not in smpl_params or "faces" not in smpl_params:
            raise RuntimeError(
                "Step 4 penetration loss is enabled, but SMPL vertices/faces are missing."
            )

    H = cfg.step4.image_height
    W = cfg.step4.image_width
    num_frames = len(data["frames"])
    num_iters = cfg.step4.num_iters

    human_gs = data["human_gs"]
    object_gs = data["object_gs"]

    # SE(3) transforms (canonical → world)
    se3_human = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_human", [0., 0., 2.])),
        num_frames=num_frames,
    ).to(device)
    se3_object = SE3Transform(
        init_translation=tuple(cfg.step4.se3.get("init_translation_object", [0., 0., 2.])),
        num_frames=num_frames,
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
    batch_size = cfg.get("step4", {}).get("batch_size", 4)

    # Print model summary
    total_params = (sum(p.numel() for p in human_gs.parameters())
                    + sum(p.numel() for p in object_gs.parameters())
                    + sum(p.numel() for p in se3_human.parameters())
                    + sum(p.numel() for p in se3_object.parameters()))
    print(f"[Step 4] Parameters: {total_params:,}")
    print(f"[Step 4] Device: {device}")
    print(f"[Step 4] {num_iters} iters, {num_frames} frames, batch_size={batch_size}")

    vis_interval = cfg.get("wandb", {}).get("vis_interval", 500)

    # --- Helper: export 3D point clouds ---
    def _resolve_frame_idx(frame_idx):
        if num_frames <= 1:
            return None
        if frame_idx is None:
            return 0
        return int(max(0, min(num_frames - 1, int(frame_idx))))

    def export_3d(step_label: str, frame_idx: int = 0):
        """Export human/object point clouds as PLY files and return paths."""
        import trimesh
        recon_dir = os.path.join(output_dir, "reconstructions")
        os.makedirs(recon_dir, exist_ok=True)
        frame_idx = _resolve_frame_idx(frame_idx)

        with torch.no_grad():
            xyz_h = se3_human(human_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
            xyz_o = se3_object(object_gs.get_xyz, frame_idx=frame_idx).cpu().numpy()
            col_h = (human_gs.get_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            col_o = (object_gs.get_colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        # Human point cloud (blue tint)
        pc_h = trimesh.PointCloud(xyz_h, colors=np.hstack([col_h, np.full((len(col_h), 1), 255, dtype=np.uint8)]))
        path_h = os.path.join(recon_dir, f"human_{step_label}.ply")
        pc_h.export(path_h)

        # Object point cloud (green tint)
        pc_o = trimesh.PointCloud(xyz_o, colors=np.hstack([col_o, np.full((len(col_o), 1), 255, dtype=np.uint8)]))
        path_o = os.path.join(recon_dir, f"object_{step_label}.ply")
        pc_o.export(path_o)

        # Combined scene
        xyz_all = np.concatenate([xyz_h, xyz_o], axis=0)
        col_all = np.concatenate([col_h, col_o], axis=0)
        pc_all = trimesh.PointCloud(xyz_all, colors=np.hstack([col_all, np.full((len(col_all), 1), 255, dtype=np.uint8)]))
        path_all = os.path.join(recon_dir, f"scene_{step_label}.ply")
        pc_all.export(path_all)

        return path_h, path_o, path_all, xyz_h, xyz_o, col_h, col_o

    # --- Helper: render 2D visualization ---
    def render_vis(gt_image_t, frame_idx: int = 0):
        """Render current state and return as numpy RGB image."""
        frame_idx = _resolve_frame_idx(frame_idx)
        with torch.no_grad():
            xyz_h = se3_human(human_gs.get_xyz, frame_idx=frame_idx)
            xyz_o = se3_object(object_gs.get_xyz, frame_idx=frame_idx)
            xyz_all = torch.cat([xyz_h, xyz_o], 0)
            col_all = torch.cat([human_gs.get_colors, object_gs.get_colors], 0)
            opa_all = torch.cat([human_gs.get_opacity, object_gs.get_opacity], 0)
            scl_all = torch.cat([human_gs.get_scaling, object_gs.get_scaling], 0)
            rendered = base_renderer(xyz_all, col_all, opa_all, scl_all)
            rendered_np = (rendered.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_np = (gt_image_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return rendered_np, gt_np

    # --- Helper: log to wandb ---
    def log_wandb(global_step, log_dict, gt_image_t=None, frame_idx=None):
        if not use_wandb or wandb_run is None:
            return
        import wandb

        # Scalar losses
        wandb_log = {f"loss/{k}": v for k, v in log_dict.items() if isinstance(v, (int, float))}
        wandb_log["step"] = global_step

        if global_step % vis_interval == 0:
            # 2D rendering
            if gt_image_t is not None:
                rendered_np, gt_np = render_vis(gt_image_t, frame_idx=frame_idx)
                import cv2
                overlay = cv2.addWeighted(gt_np, 0.5, rendered_np, 0.5, 0)
                wandb_log["vis/rendered"] = wandb.Image(rendered_np, caption=f"step {global_step}")
                wandb_log["vis/gt"] = wandb.Image(gt_np, caption="GT")
                wandb_log["vis/overlay"] = wandb.Image(overlay, caption="overlay")

            # 3D point cloud
            _, _, _, xyz_h, xyz_o, col_h, col_o = export_3d(
                f"step{global_step:06d}",
                frame_idx=0 if frame_idx is None else frame_idx,
            )
            # WandB 3D point cloud
            xyz_all = np.concatenate([xyz_h, xyz_o], axis=0)
            col_all_f = np.concatenate([col_h, col_o], axis=0).astype(np.float32) / 255.0
            # Label: 0=human, 1=object
            labels = np.array([0] * len(xyz_h) + [1] * len(xyz_o))
            wandb_log["vis/3d_scene"] = wandb.Object3D({
                "type": "lidar/beta",
                "points": np.column_stack([xyz_all, col_all_f, labels]),
            })

        wandb.log(wandb_log, step=global_step)

    # --- Training loop ---
    from tqdm import tqdm
    all_losses = []
    t_start = time.time()

    pbar = tqdm(range(1, num_iters + 1), desc="[Step 4]", dynamic_ncols=True)
    for step in pbar:
        # Sample a batch of frame indices
        if num_frames >= batch_size:
            batch_indices = np.random.choice(num_frames, batch_size, replace=False).tolist()
        else:
            batch_indices = list(range(num_frames))

        gt_images = torch.stack([data["frames"][i] for i in batch_indices])
        m_vis = torch.stack([data["masks_visible"][i] for i in batch_indices])
        m_pri = torch.stack([data["masks_primary_occ"][i] for i in batch_indices])
        m_sec = torch.stack([data["masks_secondary_occ"][i] for i in batch_indices])

        # Optional per-frame data — batch or None
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
            kps = []
            confs = []
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
            se3_human_module=se3_human if cfg.step4.temporal.enabled else None,
            se3_object_module=se3_object if cfg.step4.temporal.enabled else None,
            w_visible=cfg.step4.region_loss.weight_visible,
            w_primary=cfg.step4.region_loss.weight_primary_occ,
            w_secondary=cfg.step4.region_loss.weight_secondary_occ,
            lambda_ssim=cfg.step4.region_loss.lambda_ssim,
            lambda_contact=cfg.step4.contact.lambda_contact if cfg.step4.contact.enabled else 0.0,
            lambda_j2d=cfg.step4.proj2d.lambda_j2d if cfg.step4.proj2d.enabled else 0.0,
            lambda_pen=cfg.step4.penetration.lambda_pen if cfg.step4.penetration.enabled else 0.0,
            lambda_acc=cfg.step4.temporal.lambda_acc if cfg.step4.temporal.enabled else 0.0,
            focal=focal, fx=fx_batch, fy=fy_batch, cx=cx_batch, cy=cy_batch,
        )

        all_losses.append(log)

        if not np.isfinite(log["loss_total"]):
            if use_wandb and wandb_run:
                import wandb
                wandb.finish(exit_code=1)
            raise FloatingPointError(
                f"[Step 4] Non-finite loss detected at step {step}: {log['loss_total']}"
            )

        # Log to wandb every step (scalars), vis every vis_interval
        log_wandb(step, log, gt_images[0], frame_idx=batch_indices[0])

        # Save checkpoint at vis_interval
        if step % vis_interval == 0:
            ckpt = {
                "step": step,
                "human_gs": human_gs.state_dict(),
                "object_gs": object_gs.state_dict(),
                "se3_human": se3_human.state_dict(),
                "se3_object": se3_object.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            step_ckpt_path = os.path.join(output_dir, f"checkpoint_step{step:06d}.pt")
            torch.save(ckpt, step_ckpt_path)
            latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
            torch.save(ckpt, latest_path)
            print(f"  [Save] {step_ckpt_path}")
            if use_wandb and wandb_run:
                import wandb
                wandb.save(step_ckpt_path)

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            eta = elapsed / step * (num_iters - step)
            pbar.set_postfix_str(
                f"loss={log['loss_total']:.4f} render={log['loss_render']:.4f} "
                f"ETA={eta:.0f}s"
            )
            tqdm.write(
                f"  [step {step}/{num_iters}] "
                f"total={log['loss_total']:.4f} render={log['loss_render']:.4f} "
                f"contact={log['loss_contact']:.4f} j2d={log['loss_j2d']:.5f} "
                f"pen={log['loss_penetration']:.5f} acc={log['loss_temporal']:.5f}"
            )

    # Save final checkpoint
    ckpt = {
        "step": num_iters,
        "human_gs": human_gs.state_dict(),
        "object_gs": object_gs.state_dict(),
        "se3_human": se3_human.state_dict(),
        "se3_object": se3_object.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(ckpt, latest_path)
    print(f"  [Save] {latest_path}")

    avg_loss = np.mean([l["loss_total"] for l in all_losses])
    print(f"  Avg loss: {avg_loss:.4f}")

    # --- Final 3D reconstruction export ---
    print("\n[Step 4] Exporting final 3D reconstruction...")
    path_h, path_o, path_all, _, _, _, _ = export_3d("final", frame_idx=0)
    print(f"  Human:  {path_h}")
    print(f"  Object: {path_o}")
    print(f"  Scene:  {path_all}")

    if use_wandb and wandb_run:
        import wandb
        wandb.save(path_h)
        wandb.save(path_o)
        wandb.save(path_all)
        wandb.finish()

    total_time = time.time() - t_start
    print(f"\n[Step 4] Training complete in {total_time:.1f}s")
    cleanup_gpu()
    return all_losses


def run_step5(cfg: DictConfig, output_dir: str, device: torch.device):
    """Step 5: End-to-end evaluation."""
    from test import (
        compute_all_metrics, find_checkpoint,
        get_se3_num_frames, instantiate_se3_from_state_dict,
        load_visualization_inputs, render_and_save_visualization,
        se3_pose_sequence_to_numpy, transform_points_for_frame,
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

    se3_human = instantiate_se3_from_state_dict(ckpt["se3_human"])
    se3_object = instantiate_se3_from_state_dict(ckpt["se3_object"])

    human_gs = human_gs.to(device)
    object_gs = object_gs.to(device)
    se3_human = se3_human.to(device)
    se3_object = se3_object.to(device)

    print(f"[Step 5] Loaded: Human {n_h} pts, Object {n_o} pts, step {ckpt.get('step', ckpt.get('epoch', '?'))}")

    # --- Compute 3D point clouds in world space ---
    with torch.no_grad():
        xyz_h_world = transform_points_for_frame(se3_human, human_gs.get_xyz, frame_idx=0).cpu().numpy()
        xyz_o_world = transform_points_for_frame(se3_object, object_gs.get_xyz, frame_idx=0).cpu().numpy()

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

    object_pose_seq = None
    if get_se3_num_frames(se3_object) >= 3:
        object_pose_seq = se3_pose_sequence_to_numpy(se3_object)
        print(f"[Step 5] Object pose trajectory: {object_pose_seq.shape}")
    else:
        print("[Step 5] Acc-o skipped: checkpoint does not contain a per-frame object pose trajectory.")

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
        frames, kp_data, camera_params = load_visualization_inputs(
            base_dir=paths["base_dir"],
            processed_dir=processed_dir,
            H=H,
            W=W,
            device=device,
            is_behave=paths.get("is_behave", False),
            max_frames=5,
            default_focal=float(cfg.step4.focal),
        )

        render_and_save_visualization(
            human_gs, object_gs, se3_human, se3_object,
            renderer, frames, kp_data, output_dir, device,
            focal=cfg.step4.focal, H=H, W=W,
            camera_params=camera_params,
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

    # Resolve dataset shorthand, but preserve explicit custom input_dir overrides.
    if cfg.get("dataset") == "sample" and cfg.data_prep.input_dir in {"./sample_data", "sample_data"}:
        cfg.data_prep.input_dir = "./sample_data"
    elif cfg.get("dataset") == "behave" and cfg.data_prep.input_dir in {"./sample_data", "sample_data"}:
        cfg.data_prep.input_dir = "/data4/guanz/data/Behave/sequences"

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
        cleanup_gpu()
        run_step2(cfg)
        cleanup_gpu()
        run_step3(cfg)
        cleanup_gpu()
        run_step4(cfg, paths, output_dir, device)
        cleanup_gpu()
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

    elif job == "preprocess":
        # Steps 1-3 only: offline prior extraction (no GPU training)
        print("\n[Main] Running offline preprocessing (Steps 1-3)...\n")
        run_step1(cfg)
        cleanup_gpu()
        run_step2(cfg)
        cleanup_gpu()
        run_step3(cfg)

    elif job == "preprocess_and_train":
        # Steps 1-4 (skip evaluation)
        run_step1(cfg)
        cleanup_gpu()
        run_step2(cfg)
        cleanup_gpu()
        run_step3(cfg)
        cleanup_gpu()
        run_step4(cfg, paths, output_dir, device)

    else:
        raise ValueError(f"Unknown job type: {job}. "
                         f"Choose from: full, train, eval, preprocess, step1, step2, step3, step4, step5, preprocess_and_train")

    print(f"\n[Main] Outputs saved to: {output_dir}")
    cleanup_gpu()


if __name__ == '__main__':
    main()
