"""
Preprocessed Dataset — Pure file-read DataLoader
=================================================
This Dataset reads *only* from the serialised outputs of ``preprocess.py``.
No model inference or heavy geometry computation happens in ``__getitem__``.

Expected directory layout (created by preprocess.py):
    <root>/<video_name>/processed/
        masks/human/        *.png   (uint8 binary)
        masks/object/       *.png   (uint8 binary)
        masks/multi_region/ *.npz   (float16 soft masks)
        depth/              *.npz   (float32 aligned depth)
        poses/              smplh_aligned.npz
        keypoints/          openpose_2d.npz
    <root>/<video_name>/frames/     *.jpg / *.png  (original RGB)
"""
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Pure file-read dataset that loads precomputed priors from disk.

    Parameters
    ----------
    root_dir : str
        Path to ``<input_dir>/<video_name>`` (parent of ``frames/`` and ``processed/``).
    processed_subdir : str
        Name of the processed output folder (default ``"processed"``).
    image_size : tuple[int, int]
        (H, W) to which RGB and masks are resized before returning.
    max_frames : int | None
        Cap the number of frames (useful for debugging).
    """

    def __init__(
        self,
        root_dir: str,
        processed_subdir: str = "processed",
        image_size: tuple = (256, 256),
        max_frames: int = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        self.frames_dir = os.path.join(root_dir, "frames")
        self.proc_dir = os.path.join(root_dir, processed_subdir)

        # --- Discover frame stems (sorted) ---
        frame_paths = sorted(
            glob.glob(os.path.join(self.frames_dir, "*.jpg"))
            + glob.glob(os.path.join(self.frames_dir, "*.png"))
            + glob.glob(os.path.join(self.frames_dir, "*.jpeg"))
        )
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        self.frame_paths = frame_paths
        self.stems = [Path(p).stem for p in frame_paths]

        # --- Pre-load lightweight sequence-level data ---
        smplh_path = os.path.join(self.proc_dir, "poses", "smplh_aligned.npz")
        kp_path = os.path.join(self.proc_dir, "keypoints", "openpose_2d.npz")

        self.smplh_data = dict(np.load(smplh_path, allow_pickle=True)) if os.path.isfile(smplh_path) else {}
        self.keypoints_all = (
            np.load(kp_path, allow_pickle=True)["keypoints"] if os.path.isfile(kp_path) else None
        )

        # --- Sub-directory shortcuts ---
        self._mask_hum_dir = os.path.join(self.proc_dir, "masks", "human")
        self._mask_obj_dir = os.path.join(self.proc_dir, "masks", "object")
        self._mask_mr_dir = os.path.join(self.proc_dir, "masks", "multi_region")
        self._depth_dir = os.path.join(self.proc_dir, "depth")

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.stems)

    # -----------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        H, W = self.image_size

        # 1. RGB frame
        rgb = cv2.imread(self.frame_paths[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0  # (H, W, 3)

        # 2. Binary masks (human / object)
        mask_hum = self._load_mask(self._mask_hum_dir, stem, H, W)
        mask_obj = self._load_mask(self._mask_obj_dir, stem, H, W)

        # 3. Multi-region soft masks
        mr_path = os.path.join(self._mask_mr_dir, f"{stem}.npz")
        if os.path.isfile(mr_path):
            mr = np.load(mr_path)
            soft_M_p = cv2.resize(mr["soft_M_p"].astype(np.float32), (W, H))
            soft_M_s = cv2.resize(mr["soft_M_s"].astype(np.float32), (W, H))
            soft_M_obj = cv2.resize(mr["soft_M_obj"].astype(np.float32), (W, H))
        else:
            soft_M_p = np.zeros((H, W), dtype=np.float32)
            soft_M_s = np.zeros((H, W), dtype=np.float32)
            soft_M_obj = np.zeros((H, W), dtype=np.float32)

        # 4. Aligned depth
        depth_path = os.path.join(self._depth_dir, f"{stem}.npz")
        if os.path.isfile(depth_path):
            depth = np.load(depth_path)["depth"]
            depth = cv2.resize(depth, (W, H)).astype(np.float32)
        else:
            depth = np.zeros((H, W), dtype=np.float32)

        # 5. SMPL-H params for this frame
        smplh_frame = {}
        for k, v in self.smplh_data.items():
            if idx < len(v):
                smplh_frame[k] = torch.from_numpy(np.asarray(v[idx], dtype=np.float32))

        # 6. 2D keypoints for this frame
        if self.keypoints_all is not None and idx < len(self.keypoints_all):
            kps = self.keypoints_all[idx].astype(np.float32)  # (J, 3)
        else:
            kps = np.zeros((25, 3), dtype=np.float32)

        # --- Pack into tensors ---
        data = {
            "idx": idx,
            "stem": stem,
            # Images & masks  (C, H, W)
            "images": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "mask_human": torch.from_numpy(mask_hum).unsqueeze(0).float(),
            "mask_object": torch.from_numpy(mask_obj).unsqueeze(0).float(),
            "masks": torch.from_numpy(
                np.stack([mask_hum, mask_obj], axis=0)
            ).float(),
            # Multi-region soft masks  (3, H, W)
            "soft_masks": torch.from_numpy(
                np.stack([soft_M_p, soft_M_s, soft_M_obj], axis=0)
            ).float(),
            # Depth  (1, H, W)
            "depth": torch.from_numpy(depth).unsqueeze(0).float(),
            # 2D keypoints  (J, 3)
            "keypoints_2d": torch.from_numpy(kps).float(),
        }
        # Merge SMPL-H per-frame params with prefix
        for k, v in smplh_frame.items():
            data[f"smplh_{k}"] = v

        return data

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _load_mask(mask_dir: str, stem: str, H: int, W: int) -> np.ndarray:
        """Load a single-channel mask PNG, resize, and binarise to [0, 1]."""
        path = os.path.join(mask_dir, f"{stem}.png")
        if os.path.isfile(path):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (W, H))
            return (m > 127).astype(np.float32)
        return np.zeros((H, W), dtype=np.float32)
