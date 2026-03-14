"""
I/O utilities for Step 1 pipeline: frame loading, mask saving, etc.
"""
import glob
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


def load_frames(input_dir: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load RGB frames from a directory.

    Supports:
      - Flat image directory (*.jpg, *.png)
      - BEHAVE-style: input_dir/t*.000/k*.color.jpg

    Returns
    -------
    frames : list[np.ndarray]  (H, W, 3) BGR uint8
    paths  : list[str]         corresponding file paths
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files: List[str] = []

    # Try flat directory first
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        # BEHAVE-style
        behave_files = sorted(glob.glob(os.path.join(input_dir, "t*.000", "k*.color.jpg")))
        if behave_files:
            from collections import Counter
            cam_counts = Counter(os.path.basename(p).split(".")[0] for p in behave_files)
            best_cam = cam_counts.most_common(1)[0][0]
            image_files = sorted(p for p in behave_files if os.path.basename(p).startswith(best_cam + "."))
            print(f"[IO] Detected BEHAVE sequence, camera={best_cam}")

    if not image_files:
        # Try frames/ subdirectory
        frames_dir = os.path.join(input_dir, "frames")
        if os.path.isdir(frames_dir):
            for ext in exts:
                image_files.extend(glob.glob(os.path.join(frames_dir, ext)))

    image_files = sorted(image_files)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    if max_frames is not None:
        image_files = image_files[:max_frames]

    frames = []
    valid_paths = []
    for p in image_files:
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
            valid_paths.append(p)

    print(f"[IO] Loaded {len(frames)} frames from {input_dir}")
    return frames, valid_paths


def save_masks_png(masks: List[np.ndarray], out_dir: str, prefix: str = "mask") -> None:
    """Save binary masks as numbered PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    for idx, m in enumerate(masks):
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{idx:06d}.png"), m)


def save_npz(out_path: str, **arrays) -> None:
    """Save multiple arrays to a single .npz file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"[IO] Saved {out_path}")
