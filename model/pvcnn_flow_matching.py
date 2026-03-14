"""
PVCNN-based Flow Matching Model with Cross-Attention
=====================================================
Uses baseline's PVCNN2 for 3D branch + lightweight 2D ConvNet for video branch.
Adds bidirectional cross-attention, temporal self-attention, and mask cross-attention
between branches — matching the DualBranchDiTBlock architecture.

Drop-in replacement for DualBranchFlowMatchingTransformer with same interface.
"""
import os
import sys
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add baseline to path for PVCNN imports
_baseline_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baseline')
if _baseline_dir not in sys.path:
    sys.path.insert(0, _baseline_dir)

from model.pvcnn.pvcnn import PVCNN2


def get_timestep_embedding(embed_dim, timesteps, device):
    """Sinusoidal timestep embedding (same as baseline PVCNN)."""
    half = embed_dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# Cross-Attention & Temporal Attention modules
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention: query attends to context."""

    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True,
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
        """query: (B, Sq, D), context: (B, Sc, D) -> (B, Sq, D)"""
        B, Sq, D = query.shape
        Sc = context.shape[1]
        h = self.num_heads

        q = self.q_proj(query).reshape(B, Sq, h, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, Sc, h, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, Sc, h, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Sq, D)
        return self.proj_drop(self.out_proj(out))


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, D) -> (B, S, D)"""
        B, S, D = x.shape
        h = self.num_heads
        qkv = self.qkv(x).reshape(B, S, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, S, D)
        return self.proj_drop(self.out_proj(out))



class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep embedding."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)  # scale + shift

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """x: (B, S, D), cond: (B, cond_dim) -> (B, S, D)"""
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class CrossAttentionBlock(nn.Module):
    """
    One block of bidirectional cross-attention + temporal self-attention.
    Mirrors DualBranchDiTBlock but operates on pre-extracted token features.

    Processing order:
        1. Temporal self-attention (video tokens)
        2. Temporal self-attention (3D tokens)
        3. Cross-attention: 3D(Q) ← Video(KV)
        4. Cross-attention: Video(Q) ← 3D(KV)
        5. Mask cross-attention: Video(Q) ← Mask(KV)
        6. FFN for each branch
    """

    def __init__(self, dim: int, num_heads: int = 4, cond_dim: int = 128,
                 mlp_ratio: float = 2.0, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        # Temporal self-attention (video)
        self.norm_v_ta = AdaLayerNorm(dim, cond_dim)
        self.ta_video = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # Temporal self-attention (3D)
        self.norm_3d_ta = AdaLayerNorm(dim, cond_dim)
        self.ta_3d = SelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # Cross-attention: 3D(Q) ← Video(KV)
        self.norm_3d_ca_q = AdaLayerNorm(dim, cond_dim)
        self.norm_v_ca_kv = AdaLayerNorm(dim, cond_dim)
        self.ca_3d_from_video = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # Cross-attention: Video(Q) ← 3D(KV)
        self.norm_v_ca_q = AdaLayerNorm(dim, cond_dim)
        self.norm_3d_ca_kv = AdaLayerNorm(dim, cond_dim)
        self.ca_video_from_3d = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # Mask cross-attention: Video(Q) ← Mask(KV)
        self.norm_v_mask_q = AdaLayerNorm(dim, cond_dim)
        self.norm_mask_kv = nn.LayerNorm(dim)
        self.ca_video_from_mask = CrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # FFN (video)
        self.norm_v_ff = AdaLayerNorm(dim, cond_dim)
        self.ffn_video = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim), nn.Dropout(proj_drop),
        )

        # FFN (3D)
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
        x_video     : (B*T, S_v, D)
        x_3d        : (B*T, S_3d, D)
        t_emb       : (B*T, cond_dim)
        mask_tokens : (B*T, S_m, D) or None
        B, T        : batch and temporal dims
        """
        S_v = x_video.shape[1]
        S_3d = x_3d.shape[1]

        # 1. Temporal self-attention (video)
        # Reshape: (B*T, S_v, D) → (B*S_v, T, D)
        x_video = x_video.reshape(B, T, S_v, -1).permute(0, 2, 1, 3).reshape(B * S_v, T, -1)
        # BUG FIX: t_emb 布局为 [b0,b0,..,b0, b1,b1,..,b1], 取每个 batch 的第一个
        t_emb_v = t_emb[::T].unsqueeze(1).expand(B, S_v, -1).reshape(B * S_v, -1)  # (B*S_v, cond_dim)
        x_video = x_video + self.ta_video(self.norm_v_ta(x_video, t_emb_v))
        x_video = x_video.reshape(B, S_v, T, -1).permute(0, 2, 1, 3).reshape(B * T, S_v, -1)

        # 2. Temporal self-attention (3D)
        x_3d = x_3d.reshape(B, T, S_3d, -1).permute(0, 2, 1, 3).reshape(B * S_3d, T, -1)
        # BUG FIX: 同上
        t_emb_3d = t_emb[::T].unsqueeze(1).expand(B, S_3d, -1).reshape(B * S_3d, -1)  # (B*S_3d, cond_dim)
        x_3d = x_3d + self.ta_3d(self.norm_3d_ta(x_3d, t_emb_3d))
        x_3d = x_3d.reshape(B, S_3d, T, -1).permute(0, 2, 1, 3).reshape(B * T, S_3d, -1)

        # 3. Cross-attention: 3D(Q) ← Video(KV)
        x_3d = x_3d + self.ca_3d_from_video(
            query=self.norm_3d_ca_q(x_3d, t_emb),
            context=self.norm_v_ca_kv(x_video, t_emb),
        )

        # 4. Cross-attention: Video(Q) ← 3D(KV)
        x_video = x_video + self.ca_video_from_3d(
            query=self.norm_v_ca_q(x_video, t_emb),
            context=self.norm_3d_ca_kv(x_3d, t_emb),
        )

        # 5. Mask cross-attention: Video(Q) ← Mask(KV)
        if mask_tokens is not None:
            x_video = x_video + self.ca_video_from_mask(
                query=self.norm_v_mask_q(x_video, t_emb),
                context=self.norm_mask_kv(mask_tokens),
            )

        # 6. FFN
        x_video = x_video + self.ffn_video(self.norm_v_ff(x_video, t_emb))
        x_3d = x_3d + self.ffn_3d(self.norm_3d_ff(x_3d, t_emb))

        return x_video, x_3d


# ---------------------------------------------------------------------------
# Video U-Net (same as before, but returns bottleneck features for cross-attn)
# ---------------------------------------------------------------------------

class SimpleVideoUNet(nn.Module):
    """Minimal 2D U-Net for video branch. Processes each frame independently.
    Conditioned on timestep via AdaGN (adaptive group norm).
    Returns both output and bottleneck features for cross-attention."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int = 128, base_ch: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.base_ch = base_ch
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim),
        )
        # Encoder
        self.enc1 = self._block(in_ch, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        # Bottleneck
        self.mid = self._block(base_ch * 4, base_ch * 4)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = self._block(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = self._block(base_ch * 2 + base_ch * 2, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = self._block(base_ch + base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        # AdaGN projections
        self.t_proj_enc1 = nn.Linear(t_dim, base_ch * 2)
        self.t_proj_enc2 = nn.Linear(t_dim, base_ch * 2 * 2)
        self.t_proj_enc3 = nn.Linear(t_dim, base_ch * 4 * 2)
        self.t_proj_mid = nn.Linear(t_dim, base_ch * 4 * 2)

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.GroupNorm(min(8, in_c), in_c) if in_c >= 8 else nn.Identity(),
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.SiLU(),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.SiLU(),
        )

    def _ada_gn(self, x, t_proj):
        s, b = t_proj.chunk(2, dim=-1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x: Tensor, t_emb: Tensor, return_bottleneck: bool = False):
        """x: (B, C_in, H, W), t_emb: (B, t_dim) -> (B, C_out, H, W)
        If return_bottleneck=True, also returns bottleneck features (B, base_ch*4, h, w)."""
        t = self.t_mlp(t_emb)
        e1 = self._ada_gn(self.enc1(x), self.t_proj_enc1(t))
        e2 = self._ada_gn(self.enc2(F.avg_pool2d(e1, 2)), self.t_proj_enc2(t))
        e3 = self._ada_gn(self.enc3(F.avg_pool2d(e2, 2)), self.t_proj_enc3(t))
        m = self._ada_gn(self.mid(F.avg_pool2d(e3, 2)), self.t_proj_mid(t))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        if return_bottleneck:
            return out, m  # m is bottleneck: (B, base_ch*4, H/8, W/8)
        return out


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class PVCNNFlowMatchingModel(nn.Module):
    """
    Video branch: SimpleVideoUNet (2D conv, per-frame)
    3D branch: PVCNN2 (baseline architecture)
    Cross-attention bridge: bidirectional cross-attn + temporal self-attn + mask cross-attn

    Architecture:
        1. Video U-Net produces per-frame output + bottleneck features
        2. PVCNN produces per-frame 3D output + intermediate features
        3. Both sets of features are projected to a shared token dim
        4. N cross-attention blocks run bidirectional attention between them
        5. Cross-attended features are projected back and added as residuals

    Same interface as DualBranchFlowMatchingTransformer.
    """

    def __init__(
        self,
        video_channels: int = 3,
        video_input_channels: int = 4,
        video_patch_size: int = 2,  # unused, compat
        point_channels: int = 14,
        mask_channels: int = 2,
        dim: int = 384,       # unused, compat
        depth: int = 8,       # unused
        num_heads: int = 6,   # unused
        mlp_ratio: float = 4.0,  # unused
        cond_dim: int = 192,  # unused
        attn_drop: float = 0.0,  # unused
        proj_drop: float = 0.0,  # unused
        embed_dim: int = 64,
        # Cross-attention config
        cross_attn_dim: int = 128,
        cross_attn_heads: int = 4,
        cross_attn_depth: int = 2,
    ):
        super().__init__()
        self.video_channels = video_channels
        self.video_input_channels = video_input_channels
        self.point_channels = point_channels
        self.mask_channels = mask_channels

        t_dim = 128
        self._t_dim = t_dim
        base_ch = 64  # video U-Net base channels

        # Video: 2D U-Net
        self.video_net = SimpleVideoUNet(
            in_ch=video_input_channels + mask_channels,
            out_ch=video_channels,
            t_dim=t_dim,
            base_ch=base_ch,
        )

        # 3D: PVCNN2
        self.pvcnn_3d = PVCNN2(
            num_classes=point_channels,
            embed_dim=embed_dim,
            extra_feature_channels=point_channels - 3,
        )
        self.pvcnn_3d.classifier[-1].bias.data.zero_()
        self.pvcnn_3d.classifier[-1].weight.data.zero_()

        # --- Cross-attention bridge ---
        # Video bottleneck features: (B*T, base_ch*4, h, w) -> flatten to tokens
        video_feat_dim = base_ch * 4  # 256
        # PVCNN last decoder features: channels_fp_features (64 for PVCNN2)
        pvcnn_feat_dim = self.pvcnn_3d.channels_fp_features  # 64

        # Project both to shared cross-attention dimension
        self.video_to_tokens = nn.Linear(video_feat_dim, cross_attn_dim)
        self.tokens_to_video = nn.Linear(cross_attn_dim, video_feat_dim)
        self.pvcnn_to_tokens = nn.Linear(pvcnn_feat_dim, cross_attn_dim)
        self.tokens_to_pvcnn = nn.Linear(cross_attn_dim, pvcnn_feat_dim)

        # Mask encoder: project mask features to cross-attn token space
        self.mask_encoder = nn.Sequential(
            nn.Linear(mask_channels, cross_attn_dim),
            nn.GELU(),
            nn.Linear(cross_attn_dim, cross_attn_dim),
        )

        # Timestep conditioning for cross-attention blocks
        self.cross_t_embed = nn.Sequential(
            nn.Linear(t_dim, cross_attn_dim),
            nn.SiLU(),
            nn.Linear(cross_attn_dim, cross_attn_dim),
        )

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=cross_attn_dim,
                num_heads=cross_attn_heads,
                cond_dim=cross_attn_dim,
                mlp_ratio=2.0,
            )
            for _ in range(cross_attn_depth)
        ])

        # Residual projections: from cross-attn feature space back to output space
        # Video: bottleneck channels -> output video channels
        self.vid_residual_proj = nn.Conv2d(video_feat_dim, video_channels, 1)
        # 3D: PVCNN feature channels -> point channels
        self.pvcnn_residual_proj = nn.Conv1d(pvcnn_feat_dim, point_channels, 1)

        # Zero-init all residual projections so cross-attn starts as identity
        nn.init.zeros_(self.tokens_to_video.weight)
        nn.init.zeros_(self.tokens_to_video.bias)
        nn.init.zeros_(self.tokens_to_pvcnn.weight)
        nn.init.zeros_(self.tokens_to_pvcnn.bias)
        nn.init.zeros_(self.vid_residual_proj.weight)
        nn.init.zeros_(self.vid_residual_proj.bias)
        nn.init.zeros_(self.pvcnn_residual_proj.weight)
        nn.init.zeros_(self.pvcnn_residual_proj.bias)

    def forward(
        self,
        x_video: Tensor,
        x_3d: Tensor,
        t: Tensor,
        mask_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        PVCNN + 2D UNet 双分支 Flow Matching 前向传播
        
        Parameters
        ----------
        x_video       : (B, T, C_in, H, W)  — 噪声视频, C_in=4 (RGB+mask)
        x_3d          : (B, T, N, C_3d)      — 噪声 3D token, C_3d=14
        t             : (B,)                  — 连续时间 ∈ [0, 1]
        mask_features : (B, T, C_m, H, W)    — 掩码条件, C_m=2
        
        Returns
        -------
        v_video : (B, T, C_out, H, W) — 预测的视频向量场, C_out=3 (RGB)
        v_3d    : (B, T, N, C_3d)     — 预测的 3D 向量场
        
        维度流转:
            Video UNet: (B*T, C_in+C_m, H, W) → UNet → (B*T, C_out, H, W) + bottleneck (B*T, 256, H/8, W/8)
            PVCNN:      (B*T, C_3d, N) → PVCNN → (B*T, C_3d, N) + feats (B*T, 64, N)
            Cross-attn: vid_tokens (B*T, h*w, 128) ↔ pvcnn_tokens (B*T, N, 128)
            Residual:   vid_residual → upsample → (B*T, C_out, H, W), pvcnn_residual → (B*T, C_3d, N)
        """
        B, T, C_in, H, W = x_video.shape  # 例如 (2, 4, 4, 32, 32)
        _, _, N, C_3d = x_3d.shape          # 例如 (2, 4, 128, 14)
        device = x_video.device

        # === Timestep embeddings ===
        t_emb_vid = get_timestep_embedding(self._t_dim, t * 999, device)  # (B, t_dim=128)
        t_emb_vid_bt = t_emb_vid.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)  # (B*T, 128)

        t_disc = (t * 999).long()  # (B,) 离散化时间步 for PVCNN
        t_disc_bt = t_disc.unsqueeze(1).expand(B, T).reshape(B * T)  # (B*T,)

        # === Video branch (2D U-Net per frame) — get output + bottleneck ===
        vid_in = x_video.reshape(B * T, C_in, H, W)  # (B*T, C_in, H, W)
        if mask_features is not None:
            mask_in = mask_features.reshape(B * T, self.mask_channels, H, W)  # (B*T, C_m, H, W)
            vid_in = torch.cat([vid_in, mask_in], dim=1)  # (B*T, C_in+C_m, H, W), 例如 (8, 6, 32, 32)
        else:
            vid_in = torch.cat([vid_in, torch.zeros(B * T, self.mask_channels, H, W, device=device)], dim=1)

        v_video_raw, vid_bottleneck = self.video_net(vid_in, t_emb_vid_bt, return_bottleneck=True)
        # v_video_raw:    (B*T, C_out, H, W), 例如 (8, 3, 32, 32)
        # vid_bottleneck: (B*T, base_ch*4, H/8, W/8), 例如 (8, 256, 4, 4)

        # === 3D branch (PVCNN per frame) — get output + features ===
        pts_in = x_3d.reshape(B * T, N, C_3d).float().transpose(1, 2)  # (B*T, C_3d, N), 例如 (8, 14, 128)
        with torch.cuda.amp.autocast(enabled=False):
            v_3d_raw, pvcnn_feats_list = self.pvcnn_3d(pts_in, t_disc_bt, ret_feats=True)
        # v_3d_raw:       (B*T, C_3d, N), 例如 (8, 14, 128)
        # pvcnn_feats_list[-1] = (features, coords), features: (B*T, 64, N)
        pvcnn_last_feats = pvcnn_feats_list[-1][0]  # (B*T, feat_dim, N), feat_dim=64

        # === Project to cross-attention token space ===
        # Video: flatten spatial dims of bottleneck -> tokens
        BT, C_bot, h_bot, w_bot = vid_bottleneck.shape  # (B*T, 256, h, w)
        vid_tokens = vid_bottleneck.reshape(BT, C_bot, h_bot * w_bot).permute(0, 2, 1)  # (B*T, h*w, 256)
        vid_tokens = self.video_to_tokens(vid_tokens)  # (B*T, h*w, cross_attn_dim=128)

        # 3D: transpose point features -> tokens
        pvcnn_tokens = pvcnn_last_feats.permute(0, 2, 1)  # (B*T, N, 64)
        pvcnn_tokens = self.pvcnn_to_tokens(pvcnn_tokens)  # (B*T, N, cross_attn_dim=128)

        # Mask tokens for cross-attention conditioning
        mask_tokens = None
        if mask_features is not None:
            # Flatten mask spatially and project
            mask_flat = mask_features.reshape(B * T, self.mask_channels, H * W).permute(0, 2, 1)  # (B*T, H*W, C_m)
            mask_tokens = self.mask_encoder(mask_flat)  # (B*T, H*W, cross_attn_dim=128)

        # Cross-attention timestep conditioning
        t_emb_cross = self.cross_t_embed(t_emb_vid_bt)  # (B*T, cross_attn_dim=128)

        # === Run cross-attention blocks ===
        for block in self.cross_attn_blocks:
            vid_tokens, pvcnn_tokens = block(
                vid_tokens, pvcnn_tokens, t_emb_cross,
                mask_tokens=mask_tokens, B=B, T=T,
            )
            # vid_tokens: (B*T, h*w, 128), pvcnn_tokens: (B*T, N, 128)

        # === Project back and add as residuals ===
        # Video: tokens -> spatial bottleneck shape -> upsample to output resolution
        vid_residual = self.tokens_to_video(vid_tokens)  # (B*T, h*w, 256)
        vid_residual = vid_residual.permute(0, 2, 1).reshape(BT, C_bot, h_bot, w_bot)  # (B*T, 256, h, w)
        vid_residual_up = F.interpolate(vid_residual, size=(H, W), mode='bilinear', align_corners=False)
        # vid_residual_up: (B*T, 256, H, W)
        v_video = v_video_raw + self.vid_residual_proj(vid_residual_up)
        # vid_residual_proj: Conv2d(256 → C_out), v_video: (B*T, C_out, H, W)

        # 3D: tokens -> point features -> add to PVCNN output
        pvcnn_residual = self.tokens_to_pvcnn(pvcnn_tokens)  # (B*T, N, 64)
        pvcnn_residual = pvcnn_residual.permute(0, 2, 1)     # (B*T, 64, N)
        v_3d = v_3d_raw + self.pvcnn_residual_proj(pvcnn_residual)
        # pvcnn_residual_proj: Conv1d(64 → C_3d), v_3d: (B*T, C_3d, N)

        # === Reshape outputs ===
        v_video = v_video.reshape(B, T, self.video_channels, H, W)  # (B, T, C_out, H, W)
        v_3d = v_3d.transpose(1, 2).reshape(B, T, N, C_3d)          # (B, T, N, C_3d)

        return v_video, v_3d  # (B, T, 3, H, W), (B, T, N, 14)
