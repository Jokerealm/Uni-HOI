"""
Reusable spatial preprocessing utilities for CARI4D-style frame handling.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

_BEHAVE_INTRINSICS = (979.7844, 979.8400, 1018.9520, 779.4860)
_INTERCAP_INTRINSICS = (
    918.457763671875,
    918.4373779296875,
    956.9661865234375,
    555.944580078125,
)


def normalize_imagenet_tensor(image: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB tensor in [0, 1] with ImageNet statistics."""
    mean = _IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = _IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


def spatial_downsample(
    array: np.ndarray,
    scale_ratio: int = 1,
    is_mask: bool = False,
) -> np.ndarray:
    """Downsample an image-like array by an integer factor."""
    if scale_ratio <= 1:
        return array.copy()

    h, w = array.shape[:2]
    new_w = max(1, int(round(w / float(scale_ratio))))
    new_h = max(1, int(round(h / float(scale_ratio))))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(array, (new_w, new_h), interpolation=interp)


def scale_intrinsics(K: np.ndarray, scale_ratio: float) -> np.ndarray:
    """Scale a 3x3 camera intrinsic matrix after image downsampling."""
    K_scaled = np.array(K, dtype=np.float32, copy=True)
    K_scaled[0, :] /= scale_ratio
    K_scaled[1, :] /= scale_ratio
    return K_scaled


def compute_bbox_from_masks(
    mask_human: np.ndarray,
    mask_object: np.ndarray,
    bbox_expand: float = 1.1,
) -> np.ndarray:
    """Compute a square xywh crop from the union of human + object masks."""
    union = (mask_human > 0.5) | (mask_object > 0.5)
    if mask_human.dtype == np.uint8 or mask_object.dtype == np.uint8:
        union = (mask_human > 127) | (mask_object > 127)

    h, w = mask_human.shape[:2]
    ys, xs = np.where(union)
    if len(xs) == 0 or len(ys) == 0:
        size = max(h, w)
        x = (w - size) / 2.0
        y = (h - size) / 2.0
        return np.array([x, y, size, size], dtype=np.float32)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    size = max(x_max - x_min + 1.0, y_max - y_min + 1.0)
    size = max(2.0, size * float(bbox_expand))
    if int(round(size)) % 2 == 1:
        size = float(int(round(size)) + 1)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    x = cx - size / 2.0
    y = cy - size / 2.0
    return np.array([x, y, size, size], dtype=np.float32)


def crop_and_resize(
    array: np.ndarray,
    bbox_xywh: np.ndarray,
    out_size: Tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Crop a possibly out-of-bounds square ROI and resize it to the target size."""
    out_h, out_w = int(out_size[0]), int(out_size[1])
    x, y, bw, bh = [float(v) for v in bbox_xywh]
    size = int(round(max(bw, bh)))
    if size <= 0:
        raise ValueError(f"Invalid crop size from bbox {bbox_xywh}")

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + size
    y1 = y0 + size

    h, w = array.shape[:2]
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    if array.ndim == 3:
        padded = np.pad(array, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
    else:
        padded = np.pad(array, ((pad_t, pad_b), (pad_l, pad_r)))

    x0_pad = x0 + pad_l
    y0_pad = y0 + pad_t
    cropped = padded[y0_pad:y0_pad + size, x0_pad:x0_pad + size]

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def infer_camera_intrinsics(
    image_width: int,
    image_height: int,
    scale_ratio: int = 1,
) -> Tuple[float, float, float, float]:
    """Infer default intrinsics for BEHAVE/InterCap-like frames."""
    aspect = float(image_width) / max(float(image_height), 1.0)
    behave_like = abs(aspect - (2048.0 / 1536.0)) < 0.05
    intercap_like = abs(aspect - (1920.0 / 1080.0)) < 0.05

    if behave_like or image_width >= 1900:
        fx, fy, cx, cy = _BEHAVE_INTRINSICS
    elif intercap_like:
        fx, fy, cx, cy = _INTERCAP_INTRINSICS
    else:
        fx = fy = float(max(image_width, image_height))
        cx = image_width / 2.0
        cy = image_height / 2.0

    ds = max(float(scale_ratio), 1.0)
    return fx / ds, fy / ds, cx / ds, cy / ds


def compute_roi_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    bbox_xywh: np.ndarray,
    out_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Convert full-image intrinsics to crop-resized ROI intrinsics in pixel space."""
    out_h, out_w = int(out_size[0]), int(out_size[1])
    x, y, bw, bh = [float(v) for v in bbox_xywh]
    sx = out_w / max(bw, 1e-6)
    sy = out_h / max(bh, 1e-6)
    fx_roi = fx * sx
    fy_roi = fy * sy
    cx_roi = (cx - x) * sx
    cy_roi = (cy - y) * sy
    return fx_roi, fy_roi, cx_roi, cy_roi


def validate_pixel_keypoints(
    keypoints_2d: np.ndarray,
    image_size_hw: Optional[Tuple[int, int]] = None,
    context: str = "keypoints_2d",
    conf_threshold: float = 0.05,
) -> np.ndarray:
    """
    Validate that keypoints are expressed in pixel coordinates.

    Pipeline convention:
      - `processed/keypoints_2d.npz`: full-image pixel coordinates
      - `processed/cropped/keypoints_2d.npz`: crop-resized ROI pixel coordinates
    """
    kp = np.array(keypoints_2d, dtype=np.float32, copy=True)
    if kp.size == 0:
        return kp
    if kp.ndim < 2 or kp.shape[-1] < 2:
        raise ValueError(f"{context} must have shape (..., J, 2+) but got {kp.shape}")

    valid = np.isfinite(kp[..., 0]) & np.isfinite(kp[..., 1])
    if kp.shape[-1] > 2:
        valid &= np.isfinite(kp[..., 2]) & (kp[..., 2] > conf_threshold)
    else:
        valid &= (np.linalg.norm(kp[..., :2], axis=-1) > 1e-6)
    if not np.any(valid):
        return kp

    coords = kp[..., :2][valid]
    max_abs = float(np.max(np.abs(coords)))
    if coords.shape[0] >= 3 and max_abs <= 4.0:
        raise ValueError(
            f"{context} appears to use normalized coordinates (max abs={max_abs:.3f}). "
            "Expected pixel coordinates consistent with the active camera intrinsics."
        )

    if image_size_hw is not None:
        image_h, image_w = int(image_size_hw[0]), int(image_size_hw[1])
        max_reasonable = 8.0 * max(image_h, image_w, 1)
        if max_abs > max_reasonable:
            raise ValueError(
                f"{context} has implausibly large pixel values (max abs={max_abs:.1f}) "
                f"for image size {(image_h, image_w)}."
            )

    return kp


def resize_intrinsics_to_image(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_size_hw: Tuple[int, int],
    dst_size_hw: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Resize pixel-space intrinsics when an image is resized from src to dst."""
    src_h, src_w = int(src_size_hw[0]), int(src_size_hw[1])
    dst_h, dst_w = int(dst_size_hw[0]), int(dst_size_hw[1])
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    return fx * sx, fy * sy, cx * sx, cy * sy


def resize_keypoints_to_image(
    keypoints_2d: np.ndarray,
    src_size_hw: Tuple[int, int],
    dst_size_hw: Tuple[int, int],
    context: str = "keypoints_2d",
) -> np.ndarray:
    """Resize pixel-space keypoints when an image is resized from src to dst."""
    kp = validate_pixel_keypoints(
        keypoints_2d,
        image_size_hw=src_size_hw,
        context=context,
    )
    if kp.size == 0:
        return kp

    src_h, src_w = int(src_size_hw[0]), int(src_size_hw[1])
    dst_h, dst_w = int(dst_size_hw[0]), int(dst_size_hw[1])
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    kp[..., 0] *= sx
    kp[..., 1] *= sy
    return kp


def transform_keypoints_to_crop(
    keypoints_2d: np.ndarray,
    bbox_xywh: np.ndarray,
    out_size: Tuple[int, int],
    scale_ratio: int = 1,
) -> np.ndarray:
    """Transform pixel-space keypoints into crop-resized pixel coordinates."""
    kp = validate_pixel_keypoints(
        keypoints_2d,
        context="transform_keypoints_to_crop",
    )
    if kp.size == 0:
        return kp

    x, y, bw, bh = [float(v) for v in bbox_xywh]
    out_h, out_w = int(out_size[0]), int(out_size[1])
    sx = out_w / max(bw, 1e-6)
    sy = out_h / max(bh, 1e-6)

    kp[..., 0] = (kp[..., 0] / max(float(scale_ratio), 1.0) - x) * sx
    kp[..., 1] = (kp[..., 1] / max(float(scale_ratio), 1.0) - y) * sy
    return kp


def preprocess_frame_offline(
    frame: np.ndarray,
    mask_human: np.ndarray,
    mask_object: np.ndarray,
    depth: Optional[np.ndarray] = None,
    keypoints_2d: Optional[np.ndarray] = None,
    extra_maps: Optional[Dict[str, np.ndarray]] = None,
    scale_ratio: int = 2,
    bbox_expand: float = 1.1,
    out_size: Tuple[int, int] = (256, 256),
) -> Dict[str, np.ndarray]:
    """
    Apply CARI4D-style spatial preprocessing to a single frame.

    The source frame is first downsampled, then a square human+object crop is
    extracted and resized to the target size.
    """
    frame_ds = spatial_downsample(frame, scale_ratio=scale_ratio, is_mask=False)
    mask_h_ds = spatial_downsample(mask_human, scale_ratio=scale_ratio, is_mask=True)
    mask_o_ds = spatial_downsample(mask_object, scale_ratio=scale_ratio, is_mask=True)

    bbox_xywh = compute_bbox_from_masks(mask_h_ds, mask_o_ds, bbox_expand=bbox_expand)

    fx, fy, cx, cy = infer_camera_intrinsics(
        image_width=frame.shape[1],
        image_height=frame.shape[0],
        scale_ratio=scale_ratio,
    )
    fx_roi, fy_roi, cx_roi, cy_roi = compute_roi_intrinsics(
        fx, fy, cx, cy, bbox_xywh, out_size
    )

    out: Dict[str, np.ndarray] = {
        "rgb": crop_and_resize(frame_ds, bbox_xywh, out_size, is_mask=False),
        "mask_human": crop_and_resize(mask_h_ds, bbox_xywh, out_size, is_mask=True).astype(np.float32) / 255.0,
        "mask_object": crop_and_resize(mask_o_ds, bbox_xywh, out_size, is_mask=True).astype(np.float32) / 255.0,
        "bbox_xywh": bbox_xywh.astype(np.float32),
        "orig_size_hw": np.array(frame.shape[:2], dtype=np.int32),
        "downsampled_size_hw": np.array(frame_ds.shape[:2], dtype=np.int32),
        "fx": np.float32(fx_roi),
        "fy": np.float32(fy_roi),
        "cx": np.float32(cx_roi),
        "cy": np.float32(cy_roi),
        "scale_ratio": np.int32(scale_ratio),
    }

    if depth is not None:
        depth_ds = spatial_downsample(depth.astype(np.float32), scale_ratio=scale_ratio, is_mask=False)
        out["depth"] = crop_and_resize(depth_ds, bbox_xywh, out_size, is_mask=False).astype(np.float32)

    if keypoints_2d is not None:
        out["keypoints_2d"] = transform_keypoints_to_crop(
            keypoints_2d, bbox_xywh, out_size, scale_ratio=scale_ratio
        )

    if extra_maps:
        extra_out = {}
        for key, value in extra_maps.items():
            value_ds = spatial_downsample(value.astype(np.float32), scale_ratio=scale_ratio, is_mask=False)
            extra_out[key] = crop_and_resize(value_ds, bbox_xywh, out_size, is_mask=False).astype(np.float32)
        out["extra_maps"] = extra_out

    return out
