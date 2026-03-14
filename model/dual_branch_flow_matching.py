"""
Dual-Branch Flow Matching Transformer
======================================
A two-stream architecture for joint video (2D) and 3D Gaussian proxy
generation, trained with Continuous Normalizing Flows (Flow Matching /
Optimal Transport) instead of DDPM.

Changes from original:
  - Issue 1: Added temporal self-attention (Latte / Video DiT style)
  - Issue 5: Mask conditioning via cross-attention (separate encoder)
  - Issue 6: Video output is RGB-only (3 channels), no mask in loss target
  - Issue 7: Default dim=384, depth=8 for ~60-65M params with temporal attn

Tensor shape conventions (annotated inline):
    Video branch  : (B, T, C_v, H, W)  – spatiotemporal latent features
    3D branch     : (B, T, N, C_3d)    – per-frame point / Gaussian tokens
    Timestep      : (B,)               – continuous flow time in [0, 1]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Map continuous scalar t ∈ [0, 1] to a fixed-dim sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale & shift are predicted from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Standard multi-head cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        H = self.num_heads
        q = rearrange(self.q_proj(query),   'B Sq (H d) -> B H Sq d', H=H)
        k = rearrange(self.k_proj(context), 'B Sc (H d) -> B H Sc d', H=H)
        v = rearrange(self.v_proj(context), 'B Sc (H d) -> B H Sc d', H=H)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = rearrange(attn @ v, 'B H Sq d -> B Sq (H d)')
        return self.proj_drop(self.out_proj(out))


class SelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        H = self.num_heads
        qkv = rearrange(self.qkv(x), 'B S (three H d) -> three B H S d', three=3, H=H)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = rearrange(attn @ v, 'B H S d -> B S (H d)')
        return self.proj_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# DualBranchDiTBlock (with temporal attention — Issue 1)
# ---------------------------------------------------------------------------

class DualBranchDiTBlock(nn.Module):
    """
    One transformer block of the dual-branch architecture.

    Processing order per block:
        1. Ada-LN + Spatial Self-Attention (video branch)
        2. Temporal Self-Attention (video branch) — Issue 1
        3. Ada-LN + Spatial Self-Attention (3D branch)
        4. Temporal Self-Attention (3D branch) — Issue 1
        5. Cross-Attention: 3D(Q) ← Video(KV)
        6. Cross-Attention: Video(Q) ← 3D(KV)
        7. Mask Cross-Attention: Video(Q) ← Mask(KV) — Issue 5
        8. FFN for each branch
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int = 192,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        # ---- Video branch spatial self-attention ----
        self.norm_v_sa = AdaLayerNorm(dim, cond_dim)
        self.sa_video = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- Video branch temporal self-attention (Issue 1) ----
        self.norm_v_ta = AdaLayerNorm(dim, cond_dim)
        self.ta_video = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- 3D branch spatial self-attention ----
        self.norm_3d_sa = AdaLayerNorm(dim, cond_dim)
        self.sa_3d = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- 3D branch temporal self-attention (Issue 1) ----
        self.norm_3d_ta = AdaLayerNorm(dim, cond_dim)
        self.ta_3d = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- Cross-attention: 3D(Q) ← Video(KV) ----
        self.norm_3d_ca_q = AdaLayerNorm(dim, cond_dim)
        self.norm_v_ca_kv = AdaLayerNorm(dim, cond_dim)
        self.ca_3d_from_video = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- Cross-attention: Video(Q) ← 3D(KV) ----
        self.norm_v_ca_q = AdaLayerNorm(dim, cond_dim)
        self.norm_3d_ca_kv = AdaLayerNorm(dim, cond_dim)
        self.ca_video_from_3d = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- Cross-attention: Video(Q) ← Mask(KV) (Issue 5) ----
        self.norm_v_mask_q = AdaLayerNorm(dim, cond_dim)
        self.norm_mask_kv = nn.LayerNorm(dim)
        self.ca_video_from_mask = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # ---- FFN (video) ----
        self.norm_v_ff = AdaLayerNorm(dim, cond_dim)
        self.ffn_video = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim), nn.Dropout(proj_drop),
        )

        # ---- FFN (3D) ----
        self.norm_3d_ff = AdaLayerNorm(dim, cond_dim)
        self.ffn_3d = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim), nn.Dropout(proj_drop),
        )

    def forward(
        self,
        x_video: Tensor,
        x_3d: Tensor,
        t_emb: Tensor,
        mask_tokens: Optional[Tensor] = None,
        B: int = 1,
        T: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        双分支 Transformer 块前向传播
        
        Parameters
        ----------
        x_video     : (B*T, S_v, D)   — 展平的视频 token, S_v = nH*nW (patch 数)
        x_3d        : (B*T, S_3d, D)  — 3D Gaussian token, S_3d = N (点数)
        t_emb       : (B*T, cond_dim) — 时间步条件嵌入, 例如 cond_dim=192
        mask_tokens : (B*T, S_m, D)   — 掩码条件 token (Issue 5), S_m = S_v
        B, T        : batch 和时间维度, 用于 reshape
        
        Returns
        -------
        x_video : (B*T, S_v, D)
        x_3d    : (B*T, S_3d, D)
        """
        S_v = x_video.shape[1]   # 视频空间 token 数 = nH * nW
        S_3d = x_3d.shape[1]     # 3D 点 token 数 = N

        # 1. Spatial self-attention (video): 每帧内 patch 间注意力
        x_video = x_video + self.sa_video(self.norm_v_sa(x_video, t_emb))
        # x_video: (B*T, S_v, D)

        # 2. Temporal self-attention (video) — Issue 1: 帧间注意力
        # Reshape: (B*T, S_v, D) → (B*S_v, T, D) 让每个空间位置跨帧做注意力
        x_video = rearrange(x_video, '(B T) S D -> (B S) T D', B=B, T=T)  # (B*S_v, T, D)
        # BUG FIX: t_emb 布局为 [b0,b0,..,b0, b1,b1,..,b1], 取每个 batch 的第一个
        t_emb_temporal_v = repeat(t_emb[::T], 'B D -> (B S) D', S=S_v)    # (B*S_v, cond_dim)
        x_video = x_video + self.ta_video(self.norm_v_ta(x_video, t_emb_temporal_v))
        x_video = rearrange(x_video, '(B S) T D -> (B T) S D', B=B, S=S_v)  # (B*T, S_v, D)

        # 3. Spatial self-attention (3D): 每帧内点间注意力
        x_3d = x_3d + self.sa_3d(self.norm_3d_sa(x_3d, t_emb))
        # x_3d: (B*T, S_3d, D)

        # 4. Temporal self-attention (3D) — Issue 1: 帧间注意力
        x_3d = rearrange(x_3d, '(B T) S D -> (B S) T D', B=B, T=T)       # (B*S_3d, T, D)
        # BUG FIX: t_emb 布局为 [b0,b0,..,b0, b1,b1,..,b1], 取每个 batch 的第一个
        t_emb_temporal_3d = repeat(t_emb[::T], 'B D -> (B S) D', S=S_3d)  # (B*S_3d, cond_dim)
        x_3d = x_3d + self.ta_3d(self.norm_3d_ta(x_3d, t_emb_temporal_3d))
        x_3d = rearrange(x_3d, '(B S) T D -> (B T) S D', B=B, S=S_3d)    # (B*T, S_3d, D)

        # 5. Cross-attention: 3D queries attend to video keys/values
        # query: (B*T, S_3d, D), context: (B*T, S_v, D) → output: (B*T, S_3d, D)
        x_3d = x_3d + self.ca_3d_from_video(
            query=self.norm_3d_ca_q(x_3d, t_emb),
            context=self.norm_v_ca_kv(x_video, t_emb),
        )

        # 6. Cross-attention: video queries attend to 3D keys/values
        # query: (B*T, S_v, D), context: (B*T, S_3d, D) → output: (B*T, S_v, D)
        x_video = x_video + self.ca_video_from_3d(
            query=self.norm_v_ca_q(x_video, t_emb),
            context=self.norm_3d_ca_kv(x_3d, t_emb),
        )

        # 7. Cross-attention: video queries attend to mask condition (Issue 5)
        # query: (B*T, S_v, D), context: (B*T, S_m, D) → output: (B*T, S_v, D)
        if mask_tokens is not None:
            x_video = x_video + self.ca_video_from_mask(
                query=self.norm_v_mask_q(x_video, t_emb),
                context=self.norm_mask_kv(mask_tokens),
            )

        # 8. Feed-forward: 逐 token MLP
        x_video = x_video + self.ffn_video(self.norm_v_ff(x_video, t_emb))  # (B*T, S_v, D)
        x_3d = x_3d + self.ffn_3d(self.norm_3d_ff(x_3d, t_emb))            # (B*T, S_3d, D)

        return x_video, x_3d


# ---------------------------------------------------------------------------
# Full Dual-Branch Flow Matching Transformer
# ---------------------------------------------------------------------------

class DualBranchFlowMatchingTransformer(nn.Module):
    """
    End-to-end model that:
      1. Patchifies / tokenises both inputs
      2. Encodes mask condition separately (Issue 5)
      3. Passes them through N DualBranchDiTBlocks (with temporal attn)
      4. Predicts the vector field v_θ for each branch (RGB only output — Issue 6)

    Issue 7: Default dim=384, depth=8 → ~60-65M params with temporal attention.
    """

    def __init__(
        self,
        # Video branch — Issue 6: output is RGB only (3ch), input can be 4ch (RGB+mask)
        video_channels: int = 3,       # output channels (RGB only)
        video_input_channels: int = 4, # input channels per stream (RGB + mask)
        video_patch_size: int = 2,
        # 3D branch
        point_channels: int = 14,
        # Mask conditioning (Issue 5: separate encoder + cross-attention)
        mask_channels: int = 2,
        # Shared — Issue 7: smaller defaults
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int = 192,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.video_patch_size = video_patch_size
        self.mask_channels = mask_channels
        self.video_channels = video_channels
        self.video_input_channels = video_input_channels

        # ---- Timestep embedding ----
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ---- Video tokeniser ----
        P = video_patch_size
        video_input_dim = video_input_channels * P * P  # input: RGB + mask per stream
        video_output_dim = video_channels * P * P       # output: RGB only (Issue 6)
        self.video_input_proj = nn.Linear(video_input_dim, dim)
        self.video_output_proj = nn.Linear(dim, video_output_dim)
        self._video_input_dim = video_input_dim
        self._video_output_dim = video_output_dim

        # ---- Mask condition encoder (Issue 5: separate encoder) ----
        mask_patch_dim = mask_channels * P * P
        self.mask_encoder = nn.Sequential(
            nn.Linear(mask_patch_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # ---- 3D tokeniser ----
        self.point_input_proj = nn.Linear(point_channels, dim)
        self.point_output_proj = nn.Linear(dim, point_channels)

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            DualBranchDiTBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                cond_dim=cond_dim, attn_drop=attn_drop, proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])

        # ---- Final norms ----
        self.final_norm_video = nn.LayerNorm(dim)
        self.final_norm_3d = nn.LayerNorm(dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_init)
        nn.init.zeros_(self.video_output_proj.weight)
        nn.init.zeros_(self.video_output_proj.bias)
        nn.init.zeros_(self.point_output_proj.weight)
        nn.init.zeros_(self.point_output_proj.bias)

    def patchify_video(self, x: Tensor) -> Tensor:
        """
        将视频张量分割为 patch token
        (B, T, C, H, W) → (B*T, nH*nW, C*P*P)
        
        例如 P=2: (B, T, 4, 16, 16) → (B*T, 64, 16)
            nH = H/P = 8, nW = W/P = 8, S_v = 64
            每个 token 维度 = C * P * P = 4 * 2 * 2 = 16
        """
        P = self.video_patch_size
        tokens = rearrange(
            x, 'B T C (nH P1) (nW P2) -> (B T) (nH nW) (C P1 P2)',
            P1=P, P2=P,
        )
        return tokens  # (B*T, nH*nW, C*P*P)

    def unpatchify_video(self, tokens: Tensor, T: int, H: int, W: int) -> Tensor:
        """
        将 patch token 还原为视频张量
        (B*T, nH*nW, C*P*P) → (B, T, C, H, W)
        
        例如 P=2: (B*T, 64, 12) → (B, T, 3, 16, 16)
            C = 12 / (2*2) = 3 (RGB only)
        """
        P = self.video_patch_size
        nH, nW = H // P, W // P
        C = tokens.shape[-1] // (P * P)
        x = rearrange(
            tokens,
            '(B T) (nH nW) (C P1 P2) -> B T C (nH P1) (nW P2)',
            T=T, nH=nH, nW=nW, P1=P, P2=P, C=C,
        )
        return x  # (B, T, C, H, W)

    def forward(
        self,
        x_video: Tensor,
        x_3d: Tensor,
        t: Tensor,
        mask_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        预测双分支的向量场 v_θ (Flow Matching)
        
        Parameters
        ----------
        x_video       : (B, T, C_in, H, W)  — 噪声视频, C_in=4 (RGB+mask)
        x_3d          : (B, T, N, C_3d)      — 噪声 3D Gaussian token, C_3d=14
        t             : (B,)                  — flow time ∈ [0, 1]
        mask_features : (B, T, C_m, H, W)    — Phase 1 分割掩码条件, C_m=2
        
        Returns
        -------
        v_video : (B, T, C_out, H, W)  — 预测的视频向量场, C_out=3 (仅 RGB, Issue 6)
        v_3d    : (B, T, N, C_3d)      — 预测的 3D 向量场
        
        维度流转示例 (B=2, T=4, C_in=4, H=W=16, P=2, N=128, C_3d=14, dim=384):
            视频 patchify: (B, T, 4, 16, 16) → (B*T, 64, 16) → proj → (B*T, 64, 384)
            3D tokenize:   (B, T, 128, 14) → (B*T, 128, 14) → proj → (B*T, 128, 384)
            Transformer:   8 blocks of DualBranchDiTBlock
            视频 output:   (B*T, 64, 384) → proj → (B*T, 64, 12) → unpatchify → (B, T, 3, 16, 16)
            3D output:     (B*T, 128, 384) → proj → (B*T, 128, 14) → reshape → (B, T, 128, 14)
        """
        B, T, C_in, H, W = x_video.shape  # 例如 (2, 4, 4, 16, 16)
        _, _, N, C_3d = x_3d.shape          # 例如 (2, 4, 128, 14)

        # ---- Timestep conditioning ----
        t_emb = self.time_embed(t)  # (B,) → SinusoidalEmbed → MLP → (B, cond_dim), 例如 (B, 192)

        # ---- Tokenise video ----
        # patchify: (B, T, C_in, H, W) → (B*T, nH*nW, C_in*P*P)
        vid_tokens = self.patchify_video(x_video)  # (B*T, S_v, C_in*P*P), 例如 (8, 64, 16)
        vid_tokens = self.video_input_proj(vid_tokens)  # (B*T, S_v, dim), 例如 (8, 64, 384)

        # ---- Encode mask condition separately (Issue 5) ----
        mask_tokens = None
        if mask_features is not None:
            # patchify: (B, T, C_m, H, W) → (B*T, S_v, C_m*P*P)
            mask_patches = self.patchify_video(mask_features)  # (B*T, S_v, C_m*P*P), 例如 (8, 64, 8)
            mask_tokens = self.mask_encoder(mask_patches)  # (B*T, S_v, dim), 例如 (8, 64, 384)

        # ---- Tokenise 3D ----
        pt_tokens = rearrange(x_3d, 'B T N C -> (B T) N C')  # (B*T, N, C_3d), 例如 (8, 128, 14)
        pt_tokens = self.point_input_proj(pt_tokens)  # (B*T, N, dim), 例如 (8, 128, 384)

        # ---- Expand t_emb to match B*T ----
        t_emb_expanded = repeat(t_emb, 'B D -> (B T) D', T=T)  # (B*T, cond_dim), 例如 (8, 192)

        # ---- Transformer blocks (with temporal attention) ----
        for block in self.blocks:
            vid_tokens, pt_tokens = block(
                vid_tokens, pt_tokens, t_emb_expanded,
                mask_tokens=mask_tokens, B=B, T=T,
            )
            # vid_tokens: (B*T, S_v, dim), pt_tokens: (B*T, N, dim)

        # ---- Project back ----
        vid_tokens = self.final_norm_video(vid_tokens)  # (B*T, S_v, dim)
        pt_tokens = self.final_norm_3d(pt_tokens)       # (B*T, N, dim)

        # Issue 6: Output only RGB channels (video_channels * P * P)
        v_video_tokens = self.video_output_proj(vid_tokens)  # (B*T, S_v, C_out*P*P), 例如 (8, 64, 12)
        v_3d_flat = self.point_output_proj(pt_tokens)        # (B*T, N, C_3d), 例如 (8, 128, 14)

        # ---- Reshape outputs ----
        v_video = self.unpatchify_video(v_video_tokens, T, H, W)  # (B, T, C_out, H, W), 例如 (2, 4, 3, 16, 16)
        v_3d = rearrange(v_3d_flat, '(B T) N C -> B T N C', B=B, T=T)  # (B, T, N, C_3d), 例如 (2, 4, 128, 14)

        return v_video, v_3d  # (B, T, 3, H, W), (B, T, N, 14)


# ---------------------------------------------------------------------------
# Flow Matching Training Objective (OT-CFM)
# ---------------------------------------------------------------------------

class FlowMatchingTrainer:
    """
    Implements the Optimal Transport Conditional Flow Matching (OT-CFM)
    training loop logic.

    Issue 6: Loss is computed only on RGB channels (video_channels=3).
    """

    def __init__(
        self,
        model: DualBranchFlowMatchingTransformer,
        sigma_min: float = 1e-4,
        lambda_video: float = 1.0,
        lambda_3d: float = 1.0,
    ):
        self.model = model
        self.sigma_min = sigma_min
        self.lambda_video = lambda_video
        self.lambda_3d = lambda_3d

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.rand(batch_size, device=device)

    def compute_interpolant(
        self, x_0: Tensor, x_1: Tensor, t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        计算 OT-CFM 插值和目标向量场
        
        Parameters
        ----------
        x_0 : 任意形状, 例如 (B, T, C, H, W) 或 (B, T, N, C) — 源 (噪声)
        x_1 : 同 x_0 形状 — 目标 (数据)
        t   : (B,) — 插值时间 ∈ [0, 1]
        
        Returns
        -------
        x_t : 同 x_0 形状 — 插值结果: (1-(1-σ_min)*t)*x_0 + t*x_1
        u_t : 同 x_0 形状 — 目标向量场: x_1 - (1-σ_min)*x_0
        """
        # t_shape: (B, 1, 1, 1, 1) 或 (B, 1, 1, 1) — 自动广播到 x_0 的维度
        t_shape = t.reshape(-1, *([1] * (x_0.dim() - 1)))
        x_t = (1 - (1 - self.sigma_min) * t_shape) * x_0 + t_shape * x_1
        u_t = x_1 - (1 - self.sigma_min) * x_0
        return x_t, u_t

    def compute_loss(
        self,
        x_0_video: Tensor,
        x_1_video: Tensor,
        x_0_3d: Tensor,
        x_1_3d: Tensor,
        mask_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        完整的 Flow Matching 训练步骤
        
        Parameters
        ----------
        x_0_video     : (B, T, C_in, H, W)  — 源噪声 (与输入通道数相同, C_in=4)
        x_1_video     : (B, T, C_in, H, W)  — 目标数据
        x_0_3d        : (B, T, N, C_3d)     — 3D 源噪声, C_3d=14
        x_1_3d        : (B, T, N, C_3d)     — 3D 目标数据
        mask_features : (B, T, C_m, H, W)   — 掩码条件, C_m=2
        
        Returns
        -------
        loss : scalar — 加权总损失
        log  : dict   — 各项损失的标量值
        
        维度流转:
            t: (B,) ∈ [0, 1]
            x_t_video: (B, T, C_in, H, W) — 插值后的噪声视频
            u_t_video: (B, T, C_in, H, W) — 目标向量场
            v_video:   (B, T, C_out, H, W) — 模型预测, C_out=3 (仅 RGB)
            损失仅在 RGB 通道上计算: u_t_video[:, :, :3] vs v_video
        """
        B = x_0_video.shape[0]
        device = x_0_video.device

        t = self.sample_time(B, device)  # (B,) ∈ [0, 1]

        # 计算 OT 插值和目标向量场
        x_t_video, u_t_video = self.compute_interpolant(x_0_video, x_1_video, t)
        # x_t_video: (B, T, C_in, H, W), u_t_video: (B, T, C_in, H, W)
        x_t_3d, u_t_3d = self.compute_interpolant(x_0_3d, x_1_3d, t)
        # x_t_3d: (B, T, N, C_3d), u_t_3d: (B, T, N, C_3d)

        # 模型预测向量场
        v_video, v_3d = self.model(x_t_video, x_t_3d, t, mask_features=mask_features)
        # v_video: (B, T, C_out, H, W), C_out=3 (RGB only)
        # v_3d:    (B, T, N, C_3d)

        # Issue 6: v_video is (B, T, 3, H, W) — RGB only output
        # u_t_video may be (B, T, 4, H, W) if input has mask channel
        # Only compute loss on the RGB channels of the target
        C_out = v_video.shape[2]  # should be 3 (RGB)
        u_t_video_rgb = u_t_video[:, :, :C_out]  # (B, T, 3, H, W) — 仅取 RGB 通道

        loss_video = F.mse_loss(v_video, u_t_video_rgb)  # MSE on RGB
        loss_3d = F.mse_loss(v_3d, u_t_3d)               # MSE on 3D
        loss = self.lambda_video * loss_video + self.lambda_3d * loss_3d

        log = {
            "loss_total": loss.item(),
            "loss_video": loss_video.item(),
            "loss_3d": loss_3d.item(),
            "t_mean": t.mean().item(),
        }
        return loss, log


# ---------------------------------------------------------------------------
# ODE Sampling (inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_ode_sample(
    model: DualBranchFlowMatchingTransformer,
    x_0_video: Tensor,
    x_0_3d: Tensor,
    num_steps: int = 50,
    mask_features: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    通过 Euler 方法积分学习到的 ODE 来生成样本
    
    Parameters
    ----------
    x_0_video     : (B, T, C_in, H, W) — 初始噪声视频, C_in=4 (RGB+mask)
    x_0_3d        : (B, T, N, C_3d)    — 初始噪声 3D token, C_3d=14
    num_steps     : int                 — Euler 积分步数
    mask_features : (B, T, C_m, H, W)  — 掩码条件, C_m=2
    
    Returns
    -------
    x_video : (B, T, C_in, H, W) — 生成的视频 (仅 RGB 通道被更新, mask 通道保持不变)
    x_3d    : (B, T, N, C_3d)    — 生成的 3D Gaussian 参数
    
    注意: 模型输出 v_video 仅有 3 通道 (RGB), 但 x_video 可能有 4 通道 (RGB+mask)
    积分时仅更新前 3 个通道, mask 通道保持初始值不变
    """
    B = x_0_video.shape[0]
    device = x_0_video.device
    dt = 1.0 / num_steps  # 步长

    x_video = x_0_video.clone()  # (B, T, C_in, H, W)
    x_3d = x_0_3d.clone()        # (B, T, N, C_3d)

    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((B,), t_val, device=device)  # (B,)

        v_video, v_3d = model(x_video, x_3d, t, mask_features=mask_features)
        # v_video: (B, T, C_out, H, W), C_out=3 (RGB only)
        # v_3d:    (B, T, N, C_3d)

        # v_video is RGB only (3ch), but x_video may be 4ch (RGB + mask).
        # Only update the RGB channels; keep the mask channel fixed.
        C_out = v_video.shape[2]  # 3
        x_video[:, :, :C_out] = x_video[:, :, :C_out] + v_video * dt  # 仅更新 RGB
        x_3d = x_3d + v_3d * dt  # 更新全部 3D 通道

    return x_video, x_3d  # (B, T, C_in, H, W), (B, T, N, C_3d)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, T = 2, 4
    C_in, H, W = 4, 16, 16   # input: RGB + mask
    C_out = 3                  # output: RGB only
    C_m = 2
    N, C_3d = 128, 14
    dim = 384
    depth = 4

    model = DualBranchFlowMatchingTransformer(
        video_channels=C_out, video_input_channels=C_in,
        video_patch_size=2, point_channels=C_3d,
        mask_channels=C_m, dim=dim, depth=depth, num_heads=6, cond_dim=192,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    # ---- Training smoke test ----
    trainer = FlowMatchingTrainer(model)

    x_0_video = torch.randn(B, T, C_in, H, W, device=device)
    x_1_video = torch.randn(B, T, C_in, H, W, device=device)
    x_0_3d = torch.randn(B, T, N, C_3d, device=device)
    x_1_3d = torch.randn(B, T, N, C_3d, device=device)
    mask_feat = torch.randn(B, T, C_m, H, W, device=device)

    loss, log = trainer.compute_loss(x_0_video, x_1_video, x_0_3d, x_1_3d, mask_features=mask_feat)
    print(f"Training loss (with masks): {log}")

    # ---- Sampling smoke test ----
    noise_v = torch.randn(1, T, C_in, H, W, device=device)
    noise_3d = torch.randn(1, T, N, C_3d, device=device)
    mask_cond = torch.randn(1, T, C_m, H, W, device=device)
    gen_v, gen_3d = euler_ode_sample(model, noise_v, noise_3d, num_steps=10, mask_features=mask_cond)
    print(f"Generated video shape: {gen_v.shape}")
    print(f"Generated 3D shape:    {gen_3d.shape}")
    print("Smoke test passed.")
