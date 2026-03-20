"""
Joint differentiable rendering and training losses for video-conditioned 3DGS.

This module assumes the production training environment: CUDA is available,
`diff-gaussian-rasterization` is installed, and LPIPS uses a cached pretrained
torchvision backbone.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

def _resolve_gs_param(gs_params: Dict[str, Tensor], names: Sequence[str]) -> Tensor:
    for name in names:
        if name in gs_params:
            return gs_params[name]
    raise KeyError(f"Missing Gaussian parameter. Expected one of: {tuple(names)}.")


def _expand_param_over_batch_time(param: Tensor, batch_size: int, num_frames: int, name: str) -> Tensor:
    if param.ndim < 2:
        raise ValueError(f"`{name}` must have at least 2 dims, got shape {tuple(param.shape)}.")

    if param.ndim == 2:
        return param.unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, *param.shape)

    if param.ndim == 3:
        if param.shape[0] != batch_size:
            raise ValueError(
                f"`{name}` with shape {tuple(param.shape)} must use batch-first layout [B, N, C] when 3D."
            )
        return param.unsqueeze(1).expand(batch_size, num_frames, *param.shape[1:])

    if param.ndim == 4:
        if param.shape[0] != batch_size or param.shape[1] != num_frames:
            raise ValueError(
                f"`{name}` with shape {tuple(param.shape)} must match [B, T, ...] = "
                f"[{batch_size}, {num_frames}, ...]."
            )
        return param

    raise ValueError(
        f"`{name}` with shape {tuple(param.shape)} is unsupported. Expected [N, C], [B, N, C], or [B, T, N, C]."
    )


def _expand_intrinsics(camera_intrinsics: Tensor, batch_size: int, num_frames: int) -> Tensor:
    if camera_intrinsics.ndim == 3:
        if camera_intrinsics.shape[0] != batch_size:
            raise ValueError(
                f"`camera_intrinsics` with shape {tuple(camera_intrinsics.shape)} must match batch size {batch_size}."
            )
        return camera_intrinsics.unsqueeze(1).expand(batch_size, num_frames, 3, 3)

    if camera_intrinsics.ndim == 4:
        if camera_intrinsics.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                f"`camera_intrinsics` with shape {tuple(camera_intrinsics.shape)} must match [B, T, 3, 3] = "
                f"[{batch_size}, {num_frames}, 3, 3]."
            )
        return camera_intrinsics

    raise ValueError(
        f"`camera_intrinsics` must have shape [B, 3, 3] or [B, T, 3, 3], got {tuple(camera_intrinsics.shape)}."
    )


def _homogenize(points: Tensor) -> Tensor:
    return torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)


def _transform_points(object_to_camera: Tensor, points: Tensor) -> Tensor:
    points_h = _homogenize(points)
    return (object_to_camera @ points_h.transpose(0, 1)).transpose(0, 1)[..., :3]


def _rotation_matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """
    Convert a single 3x3 rotation matrix to a quaternion in `[w, x, y, z]`.

    The pose matrix is treated as constant conditioning, so a scalar branch-based
    implementation is acceptable here.
    """

    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = float((matrix[2, 1] - matrix[1, 2]) / s)
        qy = float((matrix[0, 2] - matrix[2, 0]) / s)
        qz = float((matrix[1, 0] - matrix[0, 1]) / s)
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(float(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        qw = float(matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = float(matrix[0, 1] + matrix[1, 0]) / s
        qz = float(matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(float(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        qw = float(matrix[0, 2] - matrix[2, 0]) / s
        qx = float(matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = float(matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(float(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        qw = float(matrix[1, 0] - matrix[0, 1]) / s
        qx = float(matrix[0, 2] + matrix[2, 0]) / s
        qy = float(matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s

    quat = torch.tensor([qw, qx, qy, qz], device=matrix.device, dtype=matrix.dtype)
    return F.normalize(quat, dim=-1)


def _quaternion_multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def _compose_gaussian_rotations(rotations: Tensor, object_to_camera: Tensor) -> Tensor:
    pose_quaternion = _rotation_matrix_to_quaternion(object_to_camera[:3, :3])
    pose_quaternion = pose_quaternion.unsqueeze(0).expand_as(rotations)
    return F.normalize(_quaternion_multiply(pose_quaternion, rotations), dim=-1)


def _infer_sh_degree(num_coefficients: int) -> int:
    degree = int(round(math.sqrt(num_coefficients) - 1))
    if (degree + 1) ** 2 != num_coefficients:
        raise ValueError(
            f"Cannot infer spherical-harmonic degree from {num_coefficients} coefficients per Gaussian."
        )
    return degree


class DiffRasterizationLayer(nn.Module):
    """
    Render static or batched 3DGS parameters into a video sequence.

    Parameters
    ----------
    image_height, image_width:
        Target render resolution.
    near_plane, far_plane:
        Depth range for projection.
        bg_color:
            White by default, matching the rest of the repository.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        *,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        scale_modifier: float = 1.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.scale_modifier = float(scale_modifier)
        self.debug = bool(debug)

        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except ImportError as exc:
            raise ImportError(
                "`diff-gaussian-rasterization` is required for `DiffRasterizationLayer`."
            ) from exc

        self._raster_settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

    @property
    def backend(self) -> str:
        return "diff-gaussian-rasterization"

    def _build_projection_matrix(self, intrinsics: Tensor) -> Tensor:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        near = self.near_plane
        far = self.far_plane

        projection = intrinsics.new_zeros(4, 4)
        projection[0, 0] = 2.0 * fx / self.image_width
        projection[1, 1] = 2.0 * fy / self.image_height
        projection[0, 2] = (2.0 * cx / self.image_width) - 1.0
        projection[1, 2] = (2.0 * cy / self.image_height) - 1.0
        projection[2, 2] = far / (far - near)
        projection[2, 3] = -(far * near) / (far - near)
        projection[3, 2] = 1.0
        return projection.transpose(0, 1).contiguous()

    def _prepare_sh_inputs(self, shs: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], int]:
        if shs.ndim == 2 and shs.shape[-1] == 3:
            return None, shs.clamp(0.0, 1.0), 0

        if shs.ndim == 2 and shs.shape[-1] % 3 == 0:
            shs = shs.view(shs.shape[0], -1, 3)

        if shs.ndim != 3 or shs.shape[-1] != 3:
            raise ValueError(
                f"`shs` must have shape [N, 3], [N, K, 3], or [N, 3*K], got {tuple(shs.shape)}."
            )

        sh_degree = _infer_sh_degree(shs.shape[1])
        return shs, None, sh_degree

    def _extract_rgb_from_rasterizer_output(self, raster_output) -> Tensor:
        if torch.is_tensor(raster_output):
            candidate = raster_output
            if candidate.ndim == 3 and candidate.shape[0] == 3:
                return candidate

        if isinstance(raster_output, tuple):
            for item in raster_output:
                if torch.is_tensor(item) and item.ndim == 3 and item.shape[0] == 3:
                    return item

        raise RuntimeError("Unable to extract RGB image from rasterizer output.")

    def _render_frame_with_cuda(
        self,
        means_camera: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacities: Tensor,
        shs: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        tanfovx = float(self.image_width / (2.0 * intrinsics[0, 0]))
        tanfovy = float(self.image_height / (2.0 * intrinsics[1, 1]))

        sh_coeffs, colors_precomp, sh_degree = self._prepare_sh_inputs(shs)

        raster_settings = self._raster_settings_cls(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color.to(device=means_camera.device, dtype=means_camera.dtype),
            scale_modifier=self.scale_modifier,
            viewmatrix=torch.eye(4, device=means_camera.device, dtype=means_camera.dtype),
            projmatrix=self._build_projection_matrix(intrinsics),
            sh_degree=sh_degree,
            campos=torch.zeros(3, device=means_camera.device, dtype=means_camera.dtype),
            prefiltered=False,
            debug=self.debug,
        )
        rasterizer = self._rasterizer_cls(raster_settings=raster_settings)
        means2d = torch.zeros_like(means_camera, device=means_camera.device, requires_grad=means_camera.requires_grad)

        raster_output = rasterizer(
            means3D=means_camera,
            means2D=means2d,
            shs=sh_coeffs,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return self._extract_rgb_from_rasterizer_output(raster_output).clamp(0.0, 1.0)

    def _render_frame(
        self,
        means_camera: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacities: Tensor,
        shs: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        valid = (
            torch.isfinite(means_camera).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacities.squeeze(-1))
            & (means_camera[:, 2] > self.near_plane)
        )

        if not valid.any():
            return self.bg_color.to(device=means_camera.device, dtype=means_camera.dtype)[:, None, None].expand(
                3, self.image_height, self.image_width
            )

        means_camera = means_camera[valid]
        scales = scales[valid].clamp(min=1e-6)
        rotations = F.normalize(rotations[valid], dim=-1)
        opacities = opacities[valid].clamp(0.0, 1.0)
        shs = shs[valid]

        if not means_camera.is_cuda:
            raise RuntimeError("`DiffRasterizationLayer` requires CUDA tensors for rendering.")

        return self._render_frame_with_cuda(
            means_camera=means_camera,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            shs=shs,
            intrinsics=intrinsics,
        )

    def forward(
        self,
        gs_params: Dict[str, Tensor],
        object_poses: Tensor,
        camera_intrinsics: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        gs_params:
            Dict containing Gaussian parameters. The module accepts both the
            requested names (`means`, `scales`, `rotations`, `opacities`, `shs`)
            and the repository's aliases (`xyz`, `scaling`, `rotation`, `opacity`).
        object_poses:
            `[B, T, 4, 4]` object-to-camera / world-to-camera transforms.
        camera_intrinsics:
            `[B, 3, 3]` or `[B, T, 3, 3]` camera intrinsics.

        Returns
        -------
        v_render:
            `[B, T, 3, H, W]`.
        """

        if object_poses.ndim != 4 or object_poses.shape[-2:] != (4, 4):
            raise ValueError(f"`object_poses` must have shape [B, T, 4, 4], got {tuple(object_poses.shape)}.")

        batch_size, num_frames = object_poses.shape[:2]
        camera_intrinsics = _expand_intrinsics(camera_intrinsics, batch_size, num_frames)

        means = _expand_param_over_batch_time(
            _resolve_gs_param(gs_params, ("means", "xyz")),
            batch_size,
            num_frames,
            "means",
        )
        scales = _expand_param_over_batch_time(
            _resolve_gs_param(gs_params, ("scales", "scaling")),
            batch_size,
            num_frames,
            "scales",
        )
        rotations = _expand_param_over_batch_time(
            _resolve_gs_param(gs_params, ("rotations", "rotation")),
            batch_size,
            num_frames,
            "rotations",
        )
        opacities = _expand_param_over_batch_time(
            _resolve_gs_param(gs_params, ("opacities", "opacity")),
            batch_size,
            num_frames,
            "opacities",
        )
        shs = _expand_param_over_batch_time(
            _resolve_gs_param(gs_params, ("shs", "colors")),
            batch_size,
            num_frames,
            "shs",
        )

        rendered_frames = []
        for batch_idx in range(batch_size):
            video_frames = []
            for frame_idx in range(num_frames):
                pose = object_poses[batch_idx, frame_idx]
                intrinsics = camera_intrinsics[batch_idx, frame_idx]

                means_camera = _transform_points(pose, means[batch_idx, frame_idx])
                rotations_camera = _compose_gaussian_rotations(rotations[batch_idx, frame_idx], pose)
                frame = self._render_frame(
                    means_camera=means_camera,
                    scales=scales[batch_idx, frame_idx],
                    rotations=rotations_camera,
                    opacities=opacities[batch_idx, frame_idx],
                    shs=shs[batch_idx, frame_idx],
                    intrinsics=intrinsics,
                )
                video_frames.append(frame)
            rendered_frames.append(torch.stack(video_frames, dim=0))

        return torch.stack(rendered_frames, dim=0)


class JointVideo3DLoss(nn.Module):
    """
    Combined Flow Matching + video reconstruction loss.

    The forward method returns the total weighted loss and caches the component
    values in `self.last_loss_dict` for logging.
    """

    def __init__(
        self,
        *,
        flow_matching_weight: float = 1.0,
        video_l1_weight: float = 1.0,
        video_perceptual_weight: float = 0.1,
        lpips_backbone: str = "vgg",
    ) -> None:
        super().__init__()
        self.flow_matching_weight = float(flow_matching_weight)
        self.video_l1_weight = float(video_l1_weight)
        self.video_perceptual_weight = float(video_perceptual_weight)

        self.perceptual = None
        if self.video_perceptual_weight > 0.0:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "`lpips` is required when `video_perceptual_weight > 0`. "
                    "Install it with `pip install lpips`."
                ) from exc

            try:
                self.perceptual = lpips.LPIPS(
                    net=lpips_backbone,
                    spatial=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "LPIPS initialization failed. Cache the required torchvision backbone weights locally "
                    "before training in an offline environment."
                ) from exc
            self.perceptual.eval()
            for parameter in self.perceptual.parameters():
                parameter.requires_grad_(False)

        self.last_loss_dict: Dict[str, Tensor] = {}

    def _compute_lpips(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.perceptual is None:
            return pred.new_zeros(())

        batch_size, num_frames, channels, height, width = pred.shape
        pred = pred.reshape(batch_size * num_frames, channels, height, width).clamp(0.0, 1.0)
        target = target.reshape(batch_size * num_frames, channels, height, width).clamp(0.0, 1.0)

        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.perceptual(pred, target).mean()

    def forward(
        self,
        v_pred: Tensor,
        v_target: Tensor,
        v_render: Tensor,
        v_gt: Tensor,
    ) -> Tensor:
        if v_pred.shape != v_target.shape:
            raise ValueError(f"`v_pred` and `v_target` must match, got {tuple(v_pred.shape)} vs {tuple(v_target.shape)}.")
        if v_render.shape != v_gt.shape:
            raise ValueError(f"`v_render` and `v_gt` must match, got {tuple(v_render.shape)} vs {tuple(v_gt.shape)}.")
        if v_render.ndim != 5:
            raise ValueError(f"`v_render` and `v_gt` must have shape [B, T, 3, H, W], got {tuple(v_render.shape)}.")

        flow_matching_loss = F.mse_loss(v_pred, v_target)
        video_l1_loss = F.l1_loss(v_render, v_gt)
        video_perceptual_loss = self._compute_lpips(v_render, v_gt)

        total_loss = (
            self.flow_matching_weight * flow_matching_loss
            + self.video_l1_weight * video_l1_loss
            + self.video_perceptual_weight * video_perceptual_loss
        )

        self.last_loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_flow_matching": flow_matching_loss.detach(),
            "loss_video_l1": video_l1_loss.detach(),
            "loss_video_lpips": video_perceptual_loss.detach(),
        }
        return total_loss


__all__ = [
    "DiffRasterizationLayer",
    "JointVideo3DLoss",
]
