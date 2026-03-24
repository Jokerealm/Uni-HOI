"""
Dual-branch co-generative Flow Matching modules for 4D human-object interaction.

This module replaces the previous object-centric Hunyuan wrapper with a joint
state-video denoiser:

1. A video branch predicts latent velocities for human/object amodal videos.
2. A state branch predicts latent velocities for 4D HOI state tokens:
   canonical human/object Gaussian sets, human joint trajectories, object
   motion, and contact signatures.
3. Bidirectional interaction happens in every fusion block.
   - 2D -> 3D: projected video features update dynamic state tokens.
   - 3D -> 2D: projected geometry maps and state tokens update video tokens.

The implementation is intentionally self-contained so training/inference can be
driven without relying on the previous single-branch Hunyuan codepath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_2d_sincos_positions(height: int, width: int, dim: int, device: torch.device) -> Tensor:
    if dim % 4 != 0:
        raise ValueError(f"`dim` must be divisible by 4 for 2D sin/cos positions, got {dim}.")
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=-1)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}.")
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half - 1, 1)
    freqs = exponent.exp()
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _apply_gaussian_activation(raw_tokens: Tensor) -> Tensor:
    if raw_tokens.shape[-1] != 14:
        raise ValueError(f"Expected 14D Gaussian tokens, got {raw_tokens.shape[-1]}.")
    xyz = raw_tokens[..., 0:3]
    rotation = F.normalize(raw_tokens[..., 3:7], dim=-1)
    scaling = F.softplus(raw_tokens[..., 7:10]) + 1e-6
    opacity = raw_tokens[..., 10:11].sigmoid()
    shs = raw_tokens[..., 11:14].sigmoid()
    return torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)


def _rotation_matrix_to_6d(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"`matrix` must have shape [..., 3, 3], got {tuple(matrix.shape)}.")
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"`rotation_6d` must have shape [..., 6], got {tuple(rotation_6d.shape)}.")
    first = rotation_6d[..., 0:3]
    second = rotation_6d[..., 3:6]
    basis_x = F.normalize(first, dim=-1)
    second = second - (basis_x * second).sum(dim=-1, keepdim=True) * basis_x
    basis_y = F.normalize(second, dim=-1)
    basis_z = F.normalize(torch.cross(basis_x, basis_y, dim=-1), dim=-1)
    basis_y = F.normalize(torch.cross(basis_z, basis_x, dim=-1), dim=-1)
    return torch.stack([basis_x, basis_y, basis_z], dim=-1)


def _flatten_object_transforms(transforms: Tensor) -> Tensor:
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise ValueError(f"`transforms` must have shape [B, T, 4, 4], got {tuple(transforms.shape)}.")
    rotation_6d = _rotation_matrix_to_6d(transforms[:, :, :3, :3])
    translation = transforms[:, :, :3, 3]
    return torch.cat([rotation_6d, translation], dim=-1)


def _unflatten_object_transforms(flattened: Tensor) -> Tensor:
    if flattened.ndim != 3 or flattened.shape[-1] != 9:
        raise ValueError(f"`flattened` must have shape [B, T, 9], got {tuple(flattened.shape)}.")
    batch_size, num_frames = flattened.shape[:2]
    transforms = flattened.new_zeros(batch_size, num_frames, 4, 4)
    transforms[:, :, :3, :3] = _rotation_6d_to_matrix(flattened[:, :, :6])
    transforms[:, :, :3, 3] = flattened[:, :, 6:9]
    transforms[:, :, 3, 3] = 1.0
    return transforms


def _project_points(points: Tensor, intrinsics: Tensor) -> Tuple[Tensor, Tensor]:
    z = points[..., 2].clamp(min=1e-3)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    x = (points[..., 0] / z) * fx + cx
    y = (points[..., 1] / z) * fy + cy
    return torch.stack([x, y], dim=-1), z


def _make_heatmap(
    coords: Tensor,
    values: Tensor,
    height: int,
    width: int,
    sigma: float,
) -> Tensor:
    batch_size, num_points = coords.shape[:2]
    grid_y = torch.arange(height, device=coords.device, dtype=coords.dtype).view(1, 1, height, 1)
    grid_x = torch.arange(width, device=coords.device, dtype=coords.dtype).view(1, 1, 1, width)
    cx = coords[..., 0].view(batch_size, num_points, 1, 1)
    cy = coords[..., 1].view(batch_size, num_points, 1, 1)
    sq_dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    weights = torch.exp(-sq_dist / (2.0 * sigma * sigma))
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    values = values.view(batch_size, num_points, values.shape[-1], 1, 1)
    heatmap = (weights.unsqueeze(2) * values).sum(dim=1)
    return heatmap


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.hidden_dim)

    def patchify(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, T, C, H, W], got {tuple(video.shape)}.")
        batch_size, num_frames, channels, height, width = video.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Height/width must be divisible by patch_size={self.patch_size}, got {(height, width)}."
            )
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        patches = video.view(
            batch_size,
            num_frames,
            channels,
            patch_h,
            self.patch_size,
            patch_w,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            batch_size,
            num_frames * patch_h * patch_w,
            channels * self.patch_size * self.patch_size,
        )
        return patches

    def unpatchify(self, patches: Tensor, *, num_frames: int, height: int, width: int) -> Tensor:
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        patches = patches.view(
            patches.shape[0],
            num_frames,
            patch_h,
            patch_w,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            patches.shape[0],
            num_frames,
            self.in_channels,
            height,
            width,
        )
        return patches

    def forward(self, video: Tensor) -> Tensor:
        return self.proj(self.patchify(video))


class VideoLatentCodec(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        hidden_dim: int,
        patch_size: int,
        num_frames: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_h = self.image_height // self.patch_size
        self.patch_w = self.image_width // self.patch_size
        self.num_patches_per_frame = self.patch_h * self.patch_w
        self.embed = PatchEmbed3D(
            in_channels=self.channels,
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
        )
        self.decode_proj = nn.Linear(self.hidden_dim, self.channels * self.patch_size * self.patch_size)
        self.temporal_embedding = nn.Parameter(torch.zeros(self.num_frames, self.hidden_dim))
        self.spatial_embedding = nn.Parameter(torch.zeros(self.num_patches_per_frame, self.hidden_dim))
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.spatial_embedding, std=0.02)

    def _position_bias(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        temporal = self.temporal_embedding.to(device=device, dtype=dtype).unsqueeze(1)
        spatial = self.spatial_embedding.to(device=device, dtype=dtype).unsqueeze(0)
        bias = temporal + spatial
        return bias.reshape(1, self.num_frames * self.num_patches_per_frame, self.hidden_dim).expand(
            batch_size, -1, -1
        )

    def encode(self, video: Tensor) -> Tensor:
        batch_size = video.shape[0]
        tokens = self.embed(video)
        return tokens + self._position_bias(batch_size, device=tokens.device, dtype=tokens.dtype)

    def decode(self, tokens: Tensor) -> Tensor:
        batch_size = tokens.shape[0]
        tokens = tokens - self._position_bias(batch_size, device=tokens.device, dtype=tokens.dtype)
        patches = self.decode_proj(tokens)
        video = self.embed.unpatchify(
            patches,
            num_frames=self.num_frames,
            height=self.image_height,
            width=self.image_width,
        )
        return video


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
        num_frames: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.codec = VideoLatentCodec(
            channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, condition_video: Tensor) -> Tensor:
        return self.out_norm(self.codec.encode(condition_video))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `num_heads` ({num_heads}).")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if context is None:
            context = query
        batch_size, query_len, _ = query.shape
        key_len = context.shape[1]
        q = self.to_q(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(batch_size, query_len, self.dim)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ZeroInitCrossAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        return self.out(self.attn(self.norm_q(query), self.norm_ctx(context)))


class FactorizedVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        num_frames: int,
        token_h: int,
        token_w: int,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.token_h = int(token_h)
        self.token_w = int(token_w)
        self.num_spatial_tokens = self.token_h * self.token_w

        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def _reshape(self, tokens: Tensor) -> Tensor:
        expected = self.num_frames * self.num_spatial_tokens
        if tokens.shape[1] != expected:
            raise ValueError(
                f"Expected video tokens of length {expected}, got {tokens.shape[1]}."
            )
        return tokens.reshape(tokens.shape[0], self.num_frames, self.num_spatial_tokens, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        video = self._reshape(x)

        spatial = self.spatial_norm(video).reshape(batch_size * self.num_frames, self.num_spatial_tokens, self.dim)
        spatial = self.spatial_attn(spatial)
        video = video + spatial.view(batch_size, self.num_frames, self.num_spatial_tokens, self.dim)

        temporal_in = video.transpose(1, 2)
        temporal = self.temporal_norm(temporal_in).reshape(batch_size * self.num_spatial_tokens, self.num_frames, self.dim)
        temporal = self.temporal_attn(temporal)
        temporal = temporal.view(batch_size, self.num_spatial_tokens, self.num_frames, self.dim).transpose(1, 2)
        video = video + temporal

        flat = video.reshape(batch_size, self.num_frames * self.num_spatial_tokens, self.dim)
        flat = flat + self.ffn(self.ffn_norm(flat))
        return flat


class FactorizedVideoLatticeCrossAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        *,
        num_frames: int,
        token_h: int,
        token_w: int,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.token_h = int(token_h)
        self.token_w = int(token_w)
        self.num_spatial_tokens = self.token_h * self.token_w

        self.query_spatial_norm = nn.LayerNorm(dim)
        self.context_spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.query_temporal_norm = nn.LayerNorm(dim)
        self.context_temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _reshape(self, tokens: Tensor) -> Tensor:
        expected = self.num_frames * self.num_spatial_tokens
        if tokens.shape[1] != expected:
            raise ValueError(f"Expected lattice tokens length {expected}, got {tokens.shape[1]}.")
        return tokens.view(tokens.shape[0], self.num_frames, self.num_spatial_tokens, self.dim)

    def forward(self, query_tokens: Tensor, context_tokens: Tensor) -> Tensor:
        batch_size = query_tokens.shape[0]
        query = self._reshape(query_tokens)
        context = self._reshape(context_tokens)

        query_spatial = self.query_spatial_norm(query).reshape(batch_size * self.num_frames, self.num_spatial_tokens, self.dim)
        context_spatial = self.context_spatial_norm(context).reshape(batch_size * self.num_frames, self.num_spatial_tokens, self.dim)
        spatial = self.spatial_attn(query_spatial, context_spatial)
        spatial = spatial.view(batch_size, self.num_frames, self.num_spatial_tokens, self.dim)

        query_temporal = self.query_temporal_norm(query.transpose(1, 2)).reshape(
            batch_size * self.num_spatial_tokens, self.num_frames, self.dim
        )
        context_temporal = self.context_temporal_norm(context.transpose(1, 2)).reshape(
            batch_size * self.num_spatial_tokens, self.num_frames, self.dim
        )
        temporal = self.temporal_attn(query_temporal, context_temporal)
        temporal = temporal.view(batch_size, self.num_spatial_tokens, self.num_frames, self.dim).transpose(1, 2)

        fused = spatial + temporal
        return self.out(fused.reshape(batch_size, self.num_frames * self.num_spatial_tokens, self.dim))


class PerFrameVideoStateCrossAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        *,
        num_frames: int,
        token_h: int,
        token_w: int,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.token_h = int(token_h)
        self.token_w = int(token_w)
        self.num_spatial_tokens = self.token_h * self.token_w
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, video_tokens: Tensor, state_tokens: Tensor) -> Tensor:
        batch_size = video_tokens.shape[0]
        expected = self.num_frames * self.num_spatial_tokens
        if video_tokens.shape[1] != expected:
            raise ValueError(f"Expected video token length {expected}, got {video_tokens.shape[1]}.")
        query = video_tokens.view(batch_size, self.num_frames, self.num_spatial_tokens, self.dim)
        query = self.query_norm(query).reshape(batch_size * self.num_frames, self.num_spatial_tokens, self.dim)
        context = self.context_norm(state_tokens).unsqueeze(1).expand(batch_size, self.num_frames, -1, -1)
        context = context.reshape(batch_size * self.num_frames, state_tokens.shape[1], self.dim)
        attended = self.attn(query, context)
        return self.out(attended.reshape(batch_size, expected, self.dim))


@dataclass
class DecodedHOIState:
    human_gaussians: Tensor
    object_gaussians: Tensor
    joints_3d: Tensor
    object_transforms: Tensor
    contact_signature: Tensor


class HOIStateCodec(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_human_gaussians: int,
        num_object_gaussians: int,
        num_frames: int,
        num_joints: int,
        contact_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_human_gaussians = int(num_human_gaussians)
        self.num_object_gaussians = int(num_object_gaussians)
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.contact_dim = int(contact_dim)
        self.num_joint_tokens = self.num_frames * self.num_joints
        self.num_motion_tokens = self.num_frames
        self.num_contact_tokens = self.num_frames

        self.human_in = nn.Linear(14, hidden_dim)
        self.object_in = nn.Linear(14, hidden_dim)
        self.joint_in = nn.Linear(3, hidden_dim)
        self.motion_in = nn.Linear(9, hidden_dim)
        self.contact_in = nn.Linear(self.contact_dim, hidden_dim)

        self.human_out = nn.Linear(hidden_dim, 14)
        self.object_out = nn.Linear(hidden_dim, 14)
        self.joint_out = nn.Linear(hidden_dim, 3)
        self.motion_out = nn.Linear(hidden_dim, 9)
        self.contact_out = nn.Linear(hidden_dim, self.contact_dim)

        self.human_pos = nn.Parameter(torch.zeros(self.num_human_gaussians, hidden_dim))
        self.object_pos = nn.Parameter(torch.zeros(self.num_object_gaussians, hidden_dim))
        self.joint_pos = nn.Parameter(torch.zeros(self.num_joint_tokens, hidden_dim))
        self.motion_pos = nn.Parameter(torch.zeros(self.num_motion_tokens, hidden_dim))
        self.contact_pos = nn.Parameter(torch.zeros(self.num_contact_tokens, hidden_dim))
        self.frame_embedding = nn.Parameter(torch.zeros(self.num_frames, hidden_dim))
        self.joint_embedding = nn.Parameter(torch.zeros(self.num_joints, hidden_dim))
        self.type_embedding = nn.Parameter(torch.zeros(5, hidden_dim))
        nn.init.normal_(self.human_pos, std=0.02)
        nn.init.normal_(self.object_pos, std=0.02)
        nn.init.normal_(self.joint_pos, std=0.02)
        nn.init.normal_(self.motion_pos, std=0.02)
        nn.init.normal_(self.contact_pos, std=0.02)
        nn.init.normal_(self.frame_embedding, std=0.02)
        nn.init.normal_(self.joint_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    @property
    def total_tokens(self) -> int:
        return (
            self.num_human_gaussians
            + self.num_object_gaussians
            + self.num_joint_tokens
            + self.num_motion_tokens
            + self.num_contact_tokens
        )

    @property
    def num_global_tokens(self) -> int:
        return self.num_human_gaussians + self.num_object_gaussians

    @property
    def num_dynamic_tokens(self) -> int:
        return self.total_tokens - self.num_global_tokens

    def _split(self, tokens: Tensor) -> Dict[str, Tensor]:
        offset = 0
        human = tokens[:, offset : offset + self.num_human_gaussians]
        offset += self.num_human_gaussians
        obj = tokens[:, offset : offset + self.num_object_gaussians]
        offset += self.num_object_gaussians
        joints = tokens[:, offset : offset + self.num_joint_tokens]
        offset += self.num_joint_tokens
        motion = tokens[:, offset : offset + self.num_motion_tokens]
        offset += self.num_motion_tokens
        contact = tokens[:, offset : offset + self.num_contact_tokens]
        return {
            "human": human,
            "object": obj,
            "joints": joints,
            "motion": motion,
            "contact": contact,
        }

    def encode_targets(
        self,
        *,
        human_gaussians: Tensor,
        object_gaussians: Tensor,
        joints_3d: Tensor,
        object_transforms: Tensor,
        contact_signature: Tensor,
    ) -> Tensor:
        if human_gaussians.shape[1] != self.num_human_gaussians:
            raise ValueError(
                f"Expected {self.num_human_gaussians} human Gaussian tokens, got {human_gaussians.shape[1]}."
            )
        if object_gaussians.shape[1] != self.num_object_gaussians:
            raise ValueError(
                f"Expected {self.num_object_gaussians} object Gaussian tokens, got {object_gaussians.shape[1]}."
            )
        if joints_3d.shape[1:3] != (self.num_frames, self.num_joints):
            raise ValueError(
                f"Expected joints shape [B, {self.num_frames}, {self.num_joints}, 3], got {tuple(joints_3d.shape)}."
            )
        if object_transforms.shape[1] != self.num_frames:
            raise ValueError(
                f"Expected object motion shape [B, {self.num_frames}, 4, 4], got {tuple(object_transforms.shape)}."
            )
        if contact_signature.shape[1:] != (self.num_frames, self.contact_dim):
            raise ValueError(
                f"Expected contact shape [B, {self.num_frames}, {self.contact_dim}], got {tuple(contact_signature.shape)}."
            )

        batch_size = human_gaussians.shape[0]
        joints_flat = joints_3d.reshape(batch_size, self.num_joint_tokens, 3)
        motion_flat = _flatten_object_transforms(object_transforms)

        joint_frame_bias = self.frame_embedding.unsqueeze(1).expand(self.num_frames, self.num_joints, self.hidden_dim)
        joint_frame_bias = joint_frame_bias.reshape(self.num_joint_tokens, self.hidden_dim)
        joint_joint_bias = self.joint_embedding.unsqueeze(0).expand(self.num_frames, self.num_joints, self.hidden_dim)
        joint_joint_bias = joint_joint_bias.reshape(self.num_joint_tokens, self.hidden_dim)

        human_tokens = self.human_in(human_gaussians) + self.human_pos.unsqueeze(0) + self.type_embedding[0]
        object_tokens = self.object_in(object_gaussians) + self.object_pos.unsqueeze(0) + self.type_embedding[1]
        joint_tokens = (
            self.joint_in(joints_flat)
            + self.joint_pos.unsqueeze(0)
            + joint_frame_bias.unsqueeze(0)
            + joint_joint_bias.unsqueeze(0)
            + self.type_embedding[2]
        )
        motion_tokens = (
            self.motion_in(motion_flat)
            + self.motion_pos.unsqueeze(0)
            + self.frame_embedding.unsqueeze(0)
            + self.type_embedding[3]
        )
        contact_tokens = (
            self.contact_in(contact_signature)
            + self.contact_pos.unsqueeze(0)
            + self.frame_embedding.unsqueeze(0)
            + self.type_embedding[4]
        )
        return torch.cat([human_tokens, object_tokens, joint_tokens, motion_tokens, contact_tokens], dim=1)

    def decode_tokens(self, tokens: Tensor) -> DecodedHOIState:
        chunks = self._split(tokens)
        joints = self.joint_out(chunks["joints"]).view(tokens.shape[0], self.num_frames, self.num_joints, 3)
        object_motion = _unflatten_object_transforms(self.motion_out(chunks["motion"]))
        contact = self.contact_out(chunks["contact"])
        return DecodedHOIState(
            human_gaussians=_apply_gaussian_activation(self.human_out(chunks["human"])),
            object_gaussians=_apply_gaussian_activation(self.object_out(chunks["object"])),
            joints_3d=joints,
            object_transforms=object_motion,
            contact_signature=contact,
        )

    def split_global_dynamic(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        return tokens[:, : self.num_global_tokens], tokens[:, self.num_global_tokens :]

    def merge_global_dynamic(self, global_tokens: Tensor, dynamic_tokens: Tensor) -> Tensor:
        return torch.cat([global_tokens, dynamic_tokens], dim=1)


class GeometryProjector(nn.Module):
    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
        joint_sigma: float = 1.5,
        object_sigma: float = 1.2,
    ) -> None:
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_size = int(patch_size)
        self.token_h = self.image_height // self.patch_size
        self.token_w = self.image_width // self.patch_size
        self.joint_sigma = float(joint_sigma)
        self.object_sigma = float(object_sigma)

    def _scale_intrinsics_to_token_grid(self, intrinsics: Tensor) -> Tensor:
        scaled = intrinsics.clone()
        scaled[..., 0, 0] = scaled[..., 0, 0] / self.patch_size
        scaled[..., 1, 1] = scaled[..., 1, 1] / self.patch_size
        scaled[..., 0, 2] = scaled[..., 0, 2] / self.patch_size
        scaled[..., 1, 2] = scaled[..., 1, 2] / self.patch_size
        return scaled

    def forward(
        self,
        decoded_state: DecodedHOIState,
        camera_intrinsics: Tensor,
    ) -> Dict[str, Tensor]:
        batch_size, num_frames = decoded_state.joints_3d.shape[:2]
        intrinsics = self._scale_intrinsics_to_token_grid(camera_intrinsics)
        joints_maps = []
        object_silhouettes = []
        object_depth_maps = []
        contact_maps = []
        object_center_coords = []
        joint_coords_all = []

        object_gaussians = decoded_state.object_gaussians
        object_xyz = object_gaussians[..., 0:3]
        object_opacity = object_gaussians[..., 10:11]

        for frame_idx in range(num_frames):
            joints_frame = decoded_state.joints_3d[:, frame_idx]
            intrinsics_frame = intrinsics[:, frame_idx]
            joint_coords, joint_depth = _project_points(joints_frame, intrinsics_frame.unsqueeze(1))
            joint_heat = _make_heatmap(
                joint_coords,
                torch.ones(batch_size, joints_frame.shape[1], 1, device=joints_frame.device, dtype=joints_frame.dtype),
                self.token_h,
                self.token_w,
                sigma=self.joint_sigma,
            )
            joint_depth_map = _make_heatmap(
                joint_coords,
                joint_depth.unsqueeze(-1),
                self.token_h,
                self.token_w,
                sigma=self.joint_sigma,
            )

            transform = decoded_state.object_transforms[:, frame_idx]
            ones = torch.ones_like(object_xyz[..., :1])
            object_h = torch.cat([object_xyz, ones], dim=-1)
            object_world = torch.matmul(transform.unsqueeze(1), object_h.unsqueeze(-1)).squeeze(-1)[..., :3]
            object_coords, object_depth = _project_points(object_world, intrinsics_frame.unsqueeze(1))
            object_heat = _make_heatmap(
                object_coords,
                object_opacity,
                self.token_h,
                self.token_w,
                sigma=self.object_sigma,
            )
            object_depth_map = _make_heatmap(
                object_coords,
                object_depth.unsqueeze(-1) * object_opacity,
                self.token_h,
                self.token_w,
                sigma=self.object_sigma,
            )
            object_depth_norm = object_depth_map / object_heat.clamp(min=1e-5)

            contact_value = decoded_state.contact_signature[:, frame_idx].mean(dim=-1, keepdim=True)
            left_right = joint_coords[:, -2:].mean(dim=1, keepdim=True)
            contact_heat = _make_heatmap(
                left_right,
                contact_value.unsqueeze(1),
                self.token_h,
                self.token_w,
                sigma=self.joint_sigma * 1.5,
            )

            object_center = object_coords.mean(dim=1)
            joints_maps.append(torch.cat([joint_heat, joint_depth_map], dim=1))
            object_silhouettes.append(object_heat)
            object_depth_maps.append(object_depth_norm)
            contact_maps.append(contact_heat)
            object_center_coords.append(object_center)
            joint_coords_all.append(joint_coords)

        geometry_maps = torch.cat(
            [
                torch.stack(joints_maps, dim=1),
                torch.stack(object_silhouettes, dim=1),
                torch.stack(object_depth_maps, dim=1),
                torch.stack(contact_maps, dim=1),
            ],
            dim=2,
        )
        return {
            "geometry_maps": geometry_maps,
            "joint_coords": torch.stack(joint_coords_all, dim=1),
            "object_centers": torch.stack(object_center_coords, dim=1),
        }


class GeometryMapEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
        num_frames: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.codec = VideoLatentCodec(
            channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=1,
            num_frames=num_frames,
            image_height=image_height // patch_size,
            image_width=image_width // patch_size,
        )

    def forward(self, geometry_maps: Tensor) -> Tensor:
        return self.codec.encode(geometry_maps)


class ProjectedVideoSampler(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_frames: int,
        token_h: int,
        token_w: int,
        num_joints: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.token_h = int(token_h)
        self.token_w = int(token_w)
        self.num_joints = int(num_joints)
        self.dynamic_out = nn.Linear(hidden_dim, hidden_dim)
        self.global_out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.dynamic_out.weight)
        nn.init.zeros_(self.dynamic_out.bias)
        nn.init.zeros_(self.global_out.weight)
        nn.init.zeros_(self.global_out.bias)

    def _sample_from_map(self, feature_map: Tensor, coords: Tensor) -> Tensor:
        norm_x = 2.0 * coords[..., 0] / max(self.token_w - 1, 1) - 1.0
        norm_y = 2.0 * coords[..., 1] / max(self.token_h - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).view(coords.shape[0], 1, coords.shape[1], 2)
        sampled = F.grid_sample(
            feature_map.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            align_corners=True,
        )
        return sampled.squeeze(2).transpose(1, 2)

    def forward(
        self,
        video_tokens: Tensor,
        *,
        state_codec: HOIStateCodec,
        geometry_aux: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        batch_size = video_tokens.shape[0]
        feature_map = video_tokens.view(batch_size, self.num_frames, self.token_h, self.token_w, self.hidden_dim)
        frame_summaries = feature_map.mean(dim=(2, 3))

        joint_coords = geometry_aux["joint_coords"]
        joint_features = []
        for frame_idx in range(self.num_frames):
            sampled = self._sample_from_map(feature_map[:, frame_idx], joint_coords[:, frame_idx])
            joint_features.append(sampled)
        joint_features = torch.stack(joint_features, dim=1).reshape(batch_size, state_codec.num_joint_tokens, self.hidden_dim)

        motion_features = frame_summaries
        contact_features = frame_summaries
        dynamic = torch.cat([joint_features, motion_features, contact_features], dim=1)
        global_context = frame_summaries.mean(dim=1, keepdim=True)
        global_context = global_context.expand(batch_size, state_codec.num_global_tokens, self.hidden_dim)
        return self.global_out(global_context), self.dynamic_out(dynamic)


class DualBranchFusionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_frames: int,
        token_h: int,
        token_w: int,
    ) -> None:
        super().__init__()
        self.video_block = FactorizedVideoTransformerBlock(
            hidden_dim,
            num_heads,
            mlp_ratio,
            dropout,
            num_frames=num_frames,
            token_h=token_h,
            token_w=token_w,
        )
        self.state_block = TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.video_from_condition = FactorizedVideoLatticeCrossAdapter(
            hidden_dim,
            num_heads,
            dropout,
            num_frames=num_frames,
            token_h=token_h,
            token_w=token_w,
        )
        self.video_from_geometry = FactorizedVideoLatticeCrossAdapter(
            hidden_dim,
            num_heads,
            dropout,
            num_frames=num_frames,
            token_h=token_h,
            token_w=token_w,
        )
        self.video_from_state = PerFrameVideoStateCrossAdapter(
            hidden_dim,
            num_heads,
            dropout,
            num_frames=num_frames,
            token_h=token_h,
            token_w=token_w,
        )
        self.state_from_video = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout=dropout)
        self.global_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.dynamic_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(
        self,
        video_tokens: Tensor,
        state_tokens: Tensor,
        *,
        condition_tokens: Tensor,
        geometry_tokens: Tensor,
        global_video_context: Tensor,
        dynamic_video_context: Tensor,
        state_codec: HOIStateCodec,
    ) -> Tuple[Tensor, Tensor]:
        video_tokens = self.video_block(video_tokens)
        state_tokens = self.state_block(state_tokens)

        video_tokens = video_tokens + self.video_from_condition(video_tokens, condition_tokens)
        video_tokens = video_tokens + self.video_from_geometry(video_tokens, geometry_tokens)
        video_tokens = video_tokens + self.video_from_state(video_tokens, state_tokens)

        global_tokens, dynamic_tokens = state_codec.split_global_dynamic(state_tokens)
        global_tokens = global_tokens + self.global_gate(global_tokens) * global_video_context
        dynamic_tokens = dynamic_tokens + self.dynamic_gate(dynamic_tokens) * dynamic_video_context
        state_tokens = state_codec.merge_global_dynamic(global_tokens, dynamic_tokens)
        state_tokens = state_tokens + self.state_from_video(state_tokens, video_tokens)
        return video_tokens, state_tokens


@dataclass
class DualBranchFMOutput:
    video_velocity: Tensor
    state_velocity: Tensor
    geometry_maps: Tensor
    decoded_state: DecodedHOIState


class DualBranchCoGenerativeFlowMatching(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        condition_channels: int,
        video_channels: int,
        patch_size: int,
        num_frames: int,
        image_height: int,
        image_width: int,
        num_human_gaussians: int,
        num_object_gaussians: int,
        num_joints: int,
        contact_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_size = int(patch_size)
        self.token_h = self.image_height // self.patch_size
        self.token_w = self.image_width // self.patch_size
        self.video_codec = VideoLatentCodec(
            channels=video_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
        )
        self.condition_encoder = ConditionEncoder(
            in_channels=condition_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
        )
        self.state_codec = HOIStateCodec(
            hidden_dim=hidden_dim,
            num_human_gaussians=num_human_gaussians,
            num_object_gaussians=num_object_gaussians,
            num_frames=num_frames,
            num_joints=num_joints,
            contact_dim=contact_dim,
        )
        self.geometry_projector = GeometryProjector(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
        )
        self.geometry_encoder = GeometryMapEncoder(
            in_channels=5,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
        )
        self.video_sampler = ProjectedVideoSampler(
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            token_h=self.token_h,
            token_w=self.token_w,
            num_joints=num_joints,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DualBranchFusionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    num_frames=num_frames,
                    token_h=self.token_h,
                    token_w=self.token_w,
                )
                for _ in range(depth)
            ]
        )
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.video_velocity_head = nn.Linear(hidden_dim, hidden_dim)
        self.state_velocity_head = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.video_velocity_head.weight)
        nn.init.zeros_(self.video_velocity_head.bias)
        nn.init.zeros_(self.state_velocity_head.weight)
        nn.init.zeros_(self.state_velocity_head.bias)

    def encode_video_target(self, video_target: Tensor) -> Tensor:
        return self.video_codec.encode(video_target)

    def decode_video_tokens(self, video_tokens: Tensor) -> Tensor:
        return self.video_codec.decode(video_tokens)

    def encode_state_target(
        self,
        *,
        human_gaussians: Tensor,
        object_gaussians: Tensor,
        joints_3d: Tensor,
        object_transforms: Tensor,
        contact_signature: Tensor,
    ) -> Tensor:
        return self.state_codec.encode_targets(
            human_gaussians=human_gaussians,
            object_gaussians=object_gaussians,
            joints_3d=joints_3d,
            object_transforms=object_transforms,
            contact_signature=contact_signature,
        )

    def decode_state_tokens(self, state_tokens: Tensor) -> DecodedHOIState:
        return self.state_codec.decode_tokens(state_tokens)

    def project_geometry(self, decoded_state: DecodedHOIState, camera_intrinsics: Tensor) -> Dict[str, Tensor]:
        return self.geometry_projector(decoded_state, camera_intrinsics)

    def forward(
        self,
        *,
        video_xt: Tensor,
        state_xt: Tensor,
        timesteps: Tensor,
        condition_video: Tensor,
        camera_intrinsics: Tensor,
    ) -> DualBranchFMOutput:
        if video_xt.ndim != 3:
            raise ValueError(f"`video_xt` must have shape [B, L_v, D], got {tuple(video_xt.shape)}.")
        if state_xt.ndim != 3:
            raise ValueError(f"`state_xt` must have shape [B, L_s, D], got {tuple(state_xt.shape)}.")
        if video_xt.shape[-1] != self.hidden_dim or state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Latent dims must equal hidden_dim={self.hidden_dim}, got {video_xt.shape[-1]} and {state_xt.shape[-1]}."
            )

        time_cond = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        video_tokens = video_xt + time_cond
        state_tokens = state_xt + time_cond
        condition_tokens = self.condition_encoder(condition_video) + time_cond

        decoded_state = self.decode_state_tokens(state_tokens)
        geometry_aux = self.project_geometry(decoded_state, camera_intrinsics)
        geometry_tokens = self.geometry_encoder(geometry_aux["geometry_maps"]) + time_cond

        for block in self.blocks:
            global_context, dynamic_context = self.video_sampler(
                video_tokens,
                state_codec=self.state_codec,
                geometry_aux=geometry_aux,
            )
            video_tokens, state_tokens = block(
                video_tokens,
                state_tokens,
                condition_tokens=condition_tokens,
                geometry_tokens=geometry_tokens,
                global_video_context=global_context,
                dynamic_video_context=dynamic_context,
                state_codec=self.state_codec,
            )
            decoded_state = self.decode_state_tokens(state_tokens)
            geometry_aux = self.project_geometry(decoded_state, camera_intrinsics)
            geometry_tokens = self.geometry_encoder(geometry_aux["geometry_maps"]) + time_cond

        video_velocity = self.video_velocity_head(self.video_norm(video_tokens))
        state_velocity = self.state_velocity_head(self.state_norm(state_tokens))
        return DualBranchFMOutput(
            video_velocity=video_velocity,
            state_velocity=state_velocity,
            geometry_maps=geometry_aux["geometry_maps"],
            decoded_state=decoded_state,
        )


__all__ = [
    "DecodedHOIState",
    "DualBranchCoGenerativeFlowMatching",
    "DualBranchFMOutput",
    "HOIStateCodec",
    "VideoLatentCodec",
]
