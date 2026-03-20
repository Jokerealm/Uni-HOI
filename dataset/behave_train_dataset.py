#!/usr/bin/env python3
"""
BehaveTrainDataset — Dataset-level training over all BEHAVE sequences.

Instead of training each sequence independently for N iters then moving on,
this dataset treats ALL preprocessed sequences as one unified dataset.

1 epoch = 1 pass over all sequences (each sequence visited once).
Each __getitem__ returns all preprocessed data for one sequence,
so the training loop can do K optimization steps per sequence per epoch.

This enables proper dataset-level training where the model sees all
sequences in each epoch, rather than overfitting to one at a time.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class BehaveTrainDataset(Dataset):
    """
    Dataset where each item is one BEHAVE sequence (all its frames + priors).

    Parameters
    ----------
    behave_dir : str
        Path to /data4/guanz/data/Behave/sequences/
    image_size : tuple
        (H, W) for resizing frames and masks.
    max_frames_per_seq : int or None
        Cap frames per sequence (None = all).
    """

    def __init__(
        self,
        behave_dir: str,
        image_size: tuple = (256, 256),
        max_frames_per_seq: int = None,
        cam_id: int = 1,
    ):
        super().__init__()
        self.behave_dir = behave_dir
        self.image_size = image_size
        self.max_frames_per_seq = max_frames_per_seq
        self.cam_id = cam_id

        # Discover all preprocessed sequences (must have gs_init_combined.pt)
        self.sequences = []
        all_dirs = sorted(os.listdir(behave_dir))
        for seq_name in all_dirs:
            seq_dir = os.path.join(behave_dir, seq_name)
            gs_init = os.path.join(seq_dir, "gs_init", "gs_init_combined.pt")
            if os.path.isdir(seq_dir) and os.path.isfile(gs_init):
                self.sequences.append(seq_name)

        print(f"[BehaveTrainDataset] Found {len(self.sequences)} preprocessed sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns all data for one sequence: frames, masks, keypoints,
        SMPL params, and GS init path.

        We return numpy/paths rather than full tensors to avoid
        loading everything into memory at once. The training loop
        will move data to GPU as needed.
        """
        seq_name = self.sequences[idx]
        seq_dir = os.path.join(self.behave_dir, seq_name)
        H, W = self.image_size

        return {
            "seq_name": seq_name,
            "seq_dir": seq_dir,
            "seq_idx": idx,
        }


def load_sequence_data(
    seq_dir: str,
    image_size: tuple = (256, 256),
    cam_id: int = 1,
    max_frames: int = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Load all preprocessed data for a single sequence into tensors.

    Returns dict with:
        frames: list of (3, H, W) tensors
        masks_visible, masks_primary_occ, masks_secondary_occ: lists of (H, W) tensors
        keypoints_2d: list of (J, 2) tensors
        kp_confidence: list of (J,) tensors
        smpl_params: dict of numpy arrays or None
        gs_init_path: str path to gs_init_combined.pt
    """
    H, W = image_size
    data = {
        "frames": [],
        "masks_visible": [],
        "masks_primary_occ": [],
        "masks_secondary_occ": [],
        "keypoints_2d": [],
        "kp_confidence": [],
        "camera_fx": [],
        "camera_fy": [],
        "camera_cx": [],
        "camera_cy": [],
        "smpl_params": None,
    }

    processed_dir = os.path.join(seq_dir, "processed")
    cropped_dir = os.path.join(processed_dir, "cropped")
    rgb_dir = os.path.join(cropped_dir, "rgb")
    use_cropped = os.path.isdir(rgb_dir)
    frame_sizes_hw = []

    # Prefer Step1-cropped assets so dataset-level training matches main.py Step4.
    if use_cropped:
        frame_paths = sorted(
            glob.glob(os.path.join(rgb_dir, "*.png"))
            + glob.glob(os.path.join(rgb_dir, "*.jpg"))
        )
    else:
        from dataset.behave_paths import DataPaths
        frame_paths = DataPaths.get_image_paths_seq(seq_dir, tid=cam_id)
        if not frame_paths:
            for cid in [0, 1, 2, 3]:
                frame_paths = DataPaths.get_image_paths_seq(seq_dir, tid=cid)
                if frame_paths:
                    break
    if not frame_paths:
        return None

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_sizes_hw.append(tuple(int(v) for v in img.shape[:2]))
        if not use_cropped:
            # CARI4D-style: downsample raw 2K image before resize to target
            oh, ow = img.shape[:2]
            if ow > 1024:
                img = cv2.resize(img, (ow // 2, oh // 2), interpolation=cv2.INTER_LINEAR)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        data["frames"].append(t.to(device))

    num_frames = len(data["frames"])
    if num_frames == 0:
        return None

    # --- Load masks (decompress once, resize on CPU with cv2) ---
    region_masks_path = os.path.join(processed_dir, "region_masks.npz")
    if use_cropped:
        cropped_region_masks_path = os.path.join(cropped_dir, "region_masks.npz")
        if os.path.isfile(cropped_region_masks_path):
            region_masks_path = cropped_region_masks_path
    if os.path.isfile(region_masks_path):
        rm = np.load(region_masks_path)
        # Load arrays once to avoid repeated npz decompression
        arr_obj = np.array(rm["M_object"], dtype=np.float32)
        arr_p = np.array(rm["M_p"], dtype=np.float32)
        arr_s = np.array(rm["M_s"], dtype=np.float32)
        n_masks = arr_obj.shape[0]
        for i in range(num_frames):
            idx = min(i, n_masks - 1)
            m_obj = cv2.resize(arr_obj[idx], (W, H), interpolation=cv2.INTER_NEAREST)
            m_p = cv2.resize(arr_p[idx], (W, H), interpolation=cv2.INTER_NEAREST)
            m_s = cv2.resize(arr_s[idx], (W, H), interpolation=cv2.INTER_NEAREST)
            data["masks_visible"].append(torch.from_numpy(m_obj).to(device))
            data["masks_primary_occ"].append(torch.from_numpy(m_p).to(device))
            data["masks_secondary_occ"].append(torch.from_numpy(m_s).to(device))
        del arr_obj, arr_p, arr_s
    else:
        for _ in range(num_frames):
            data["masks_visible"].append(torch.ones(H, W, device=device))
            data["masks_primary_occ"].append(torch.zeros(H, W, device=device))
            data["masks_secondary_occ"].append(torch.zeros(H, W, device=device))

    # --- Load keypoints ---
    from dataset.video_transforms import (
        infer_camera_intrinsics,
        resize_intrinsics_to_image,
        resize_keypoints_to_image,
        validate_pixel_keypoints,
    )

    if use_cropped:
        meta_path = os.path.join(cropped_dir, "meta.npz")
        if os.path.isfile(meta_path):
            meta = np.load(meta_path)
            for key_src, key_dst in [
                ("fx", "camera_fx"),
                ("fy", "camera_fy"),
                ("cx", "camera_cx"),
                ("cy", "camera_cy"),
            ]:
                vals = meta[key_src]
                data[key_dst] = [float(v) for v in vals[:num_frames]]
    else:
        for src_h, src_w in frame_sizes_hw[:num_frames]:
            fx_src, fy_src, cx_src, cy_src = infer_camera_intrinsics(
                image_width=src_w,
                image_height=src_h,
                scale_ratio=1,
            )
            fx_dst, fy_dst, cx_dst, cy_dst = resize_intrinsics_to_image(
                fx_src,
                fy_src,
                cx_src,
                cy_src,
                src_size_hw=(src_h, src_w),
                dst_size_hw=(H, W),
            )
            data["camera_fx"].append(float(fx_dst))
            data["camera_fy"].append(float(fy_dst))
            data["camera_cx"].append(float(cx_dst))
            data["camera_cy"].append(float(cy_dst))
    for key, default in [
        ("camera_fx", 500.0),
        ("camera_fy", 500.0),
        ("camera_cx", W / 2.0),
        ("camera_cy", H / 2.0),
    ]:
        if len(data[key]) < num_frames:
            data[key].extend([default] * (num_frames - len(data[key])))

    kp_path = os.path.join(processed_dir, "keypoints_2d.npz")
    if use_cropped:
        cropped_kp_path = os.path.join(cropped_dir, "keypoints_2d.npz")
        if os.path.isfile(cropped_kp_path):
            kp_path = cropped_kp_path
    if os.path.isfile(kp_path):
        kp_data = np.load(kp_path)
        kps = kp_data["keypoints"]
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

    # --- Load SMPL params ---
    smpl_path = os.path.join(processed_dir, "smpl_params.npz")
    if os.path.isfile(smpl_path):
        data["smpl_params"] = dict(np.load(smpl_path, allow_pickle=True))

    # Load separate joints_3d
    joints_3d_path = os.path.join(processed_dir, "joints_3d.npz")
    if os.path.isfile(joints_3d_path):
        j3d = np.load(joints_3d_path)
        if "joints_3d" in j3d:
            if data["smpl_params"] is None:
                data["smpl_params"] = {}
            if "joints_3d" not in data["smpl_params"] and "keypoints_3d" not in data["smpl_params"]:
                data["smpl_params"]["joints_3d"] = j3d["joints_3d"]

    data["num_frames"] = num_frames
    return data


class SequenceModelRegistry:
    """
    Manages per-sequence GaussianModel + SE3 parameters.

    All models are kept in CPU memory. The active sequence's model
    is moved to GPU for training, then moved back to CPU after.
    This keeps GPU memory constant regardless of dataset size.
    """

    def __init__(self, num_points_human=4096, num_points_object=2048):
        self.models = {}  # seq_name -> dict of state_dicts
        self.num_points_human = num_points_human
        self.num_points_object = num_points_object

    def get_or_init(
        self,
        seq_name: str,
        gs_init_dir: str,
        device: torch.device,
        num_frames: int = 1,
    ):
        """
        Get models for a sequence. If first time, initialize from GS init.
        Returns (human_gs, object_gs, se3_human, se3_object, optimizer_state).
        """
        from scripts.joint_3dgs_optimization import GaussianModel
        from scripts.step4_joint_optimization import SE3Transform

        if seq_name in self.models:
            # Restore from saved state
            state = self.models[seq_name]
            n_h = state["human_gs"]["xyz"].shape[0]
            n_o = state["object_gs"]["xyz"].shape[0]
            num_pose_frames = (
                int(state["se3_human"]["translation"].shape[0])
                if state["se3_human"]["translation"].ndim == 2
                else 1
            )
            human_gs = GaussianModel(num_points=n_h)
            object_gs = GaussianModel(num_points=n_o)
            human_gs.load_state_dict(state["human_gs"])
            object_gs.load_state_dict(state["object_gs"])
            se3_human = SE3Transform(num_frames=num_pose_frames)
            se3_object = SE3Transform(num_frames=num_pose_frames)
            se3_human.load_state_dict(state["se3_human"])
            se3_object.load_state_dict(state["se3_object"])
            opt_state = state.get("optimizer", None)
        else:
            # Initialize from GS init
            human_gs, object_gs = self._load_gs_init(gs_init_dir)
            se3_human = SE3Transform(init_translation=(0., 0., 0.), num_frames=num_frames)
            se3_object = SE3Transform(init_translation=(0., 0., 0.), num_frames=num_frames)
            opt_state = None

        return (
            human_gs.to(device),
            object_gs.to(device),
            se3_human.to(device),
            se3_object.to(device),
            opt_state,
        )

    def save(self, seq_name, human_gs, object_gs, se3_human, se3_object, optimizer=None):
        """Save model state back to CPU registry."""
        state = {
            "human_gs": {k: v.cpu().clone() for k, v in human_gs.state_dict().items()},
            "object_gs": {k: v.cpu().clone() for k, v in object_gs.state_dict().items()},
            "se3_human": {k: v.cpu().clone() for k, v in se3_human.state_dict().items()},
            "se3_object": {k: v.cpu().clone() for k, v in se3_object.state_dict().items()},
        }
        if optimizer is not None:
            # Deep copy optimizer state to CPU
            opt_sd = optimizer.state_dict()
            cpu_state = {"state": {}, "param_groups": opt_sd["param_groups"]}
            for k, v in opt_sd["state"].items():
                cpu_state["state"][k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        cpu_state["state"][k][kk] = vv.cpu().clone()
                    else:
                        cpu_state["state"][k][kk] = vv
            state["optimizer"] = cpu_state
        self.models[seq_name] = state

    def _load_gs_init(self, gs_init_dir: str):
        """Load initial GaussianModels from Step 3 output."""
        from scripts.joint_3dgs_optimization import GaussianModel

        combined_path = os.path.join(gs_init_dir, "gs_init_combined.pt")
        g_h_path = os.path.join(gs_init_dir, "G_h.pt")
        g_o_path = os.path.join(gs_init_dir, "G_o.pt")

        if os.path.isfile(combined_path):
            ckpt = torch.load(combined_path, map_location="cpu", weights_only=False)
            h_raw = ckpt["G_h"].get("raw", torch.randn(self.num_points_human, 14))
            o_raw = ckpt["G_o"].get("raw", torch.randn(self.num_points_object, 14))
            return GaussianModel.from_phase2(h_raw), GaussianModel.from_phase2(o_raw)
        elif os.path.isfile(g_h_path) and os.path.isfile(g_o_path):
            g_h = torch.load(g_h_path, map_location="cpu", weights_only=False)
            g_o = torch.load(g_o_path, map_location="cpu", weights_only=False)
            h_raw = g_h.get("raw", torch.randn(self.num_points_human, 14))
            o_raw = g_o.get("raw", torch.randn(self.num_points_object, 14))
            return GaussianModel.from_phase2(h_raw), GaussianModel.from_phase2(o_raw)
        else:
            return (
                GaussianModel(num_points=self.num_points_human, init_extent=0.5),
                GaussianModel(num_points=self.num_points_object, init_extent=0.3),
            )

    def save_checkpoint(self, path: str, extra: dict = None):
        """Save entire registry to disk."""
        ckpt = {"models": self.models}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load registry from disk. Returns extra metadata."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.models = ckpt.pop("models", {})
        return ckpt

    @property
    def num_initialized(self):
        return len(self.models)
