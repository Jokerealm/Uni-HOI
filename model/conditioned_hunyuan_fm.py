"""
Conditioned Hunyuan Flow Matching modules.

This file adds a lightweight ControlNet-style conditioning path for
Hunyuan3D-2's DiT backbone. The base Hunyuan DiT stays frozen, while a
trainable interaction encoder and zero-initialized cross-attention adapters
inject human-object interaction cues into the latent stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from torch import Tensor, nn


def _group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


def _zero_module(module: nn.Linear) -> nn.Linear:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def _normalize_block_indices(
    indices: Optional[Sequence[int]],
    num_blocks: int,
    *,
    default_all: bool,
) -> Tuple[int, ...]:
    if indices is None:
        resolved = tuple(range(num_blocks)) if default_all else ()
    else:
        resolved = tuple(sorted(set(int(idx) for idx in indices)))

    for idx in resolved:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"Block index {idx} is out of range for {num_blocks} blocks.")
    return resolved


def hunyuan_timestep_embedding(
    timesteps: Tensor,
    dim: int,
    *,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> Tensor:
    """
    Match the public Hunyuan3D-2 sinusoidal timestep embedding.

    Parameters
    ----------
    timesteps:
        Shape `[B]`, continuous flow-matching timesteps in `[0, 1]`.
    dim:
        Embedding dimension.
    max_period:
        Controls the minimum frequency.
    time_factor:
        Hunyuan rescales normalized flow time before sinusoidal encoding.
    """

    timesteps = time_factor * timesteps
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(timesteps):
        embedding = embedding.to(dtype=timesteps.dtype)
    return embedding


def sinusoidal_position_embedding(sequence_length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a standard 1D sinusoidal position embedding with shape `[1, L, D]`."""

    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1))
    angles = positions[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros(sequence_length, 1, device=device, dtype=torch.float32)], dim=-1)
    return embedding.unsqueeze(0).to(dtype=dtype)


class InteractionConditionEncoder(nn.Module):
    """
    Encode masked interaction video + human pose into conditioning tokens.

    Inputs
    ------
    v_masked:
        `[B, T, 3, H, W]` masked RGB video.
    m_human:
        `[B, T, 1, H, W]` human occlusion mask.
    h_pose:
        `[B, T, 144]` SMPL pose parameters.

    Returns
    -------
    cond_tokens:
        `[B, T, output_dim]` temporal conditioning tokens.
    """

    def __init__(
        self,
        pose_dim: int = 144,
        output_dim: int = 1024,
        video_base_channels: int = 96,
        pose_hidden_dim: int = 512,
        num_attention_heads: int = 8,
        num_transformer_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        video_hidden_dim = output_dim // 2
        pose_output_dim = output_dim // 2

        self.video_encoder = nn.Sequential(
            nn.Conv3d(4, video_base_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.GroupNorm(_group_norm_groups(video_base_channels), video_base_channels),
            nn.SiLU(),
            nn.Conv3d(
                video_base_channels,
                video_base_channels * 2,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.GroupNorm(_group_norm_groups(video_base_channels * 2), video_base_channels * 2),
            nn.SiLU(),
            nn.Conv3d(video_base_channels * 2, video_hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_norm_groups(video_hidden_dim), video_hidden_dim),
            nn.SiLU(),
        )

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.SiLU(),
            nn.Linear(pose_hidden_dim, pose_hidden_dim),
            nn.SiLU(),
            nn.Linear(pose_hidden_dim, pose_output_dim),
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(video_hidden_dim + pose_output_dim),
            nn.Linear(video_hidden_dim + pose_output_dim, output_dim),
            nn.SiLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_attention_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, v_masked: Tensor, m_human: Tensor, h_pose: Tensor) -> Tensor:
        if v_masked.ndim != 5:
            raise ValueError(f"`v_masked` must have shape [B, T, 3, H, W], got {tuple(v_masked.shape)}.")
        if m_human.ndim != 5:
            raise ValueError(f"`m_human` must have shape [B, T, 1, H, W], got {tuple(m_human.shape)}.")
        if h_pose.ndim != 3:
            raise ValueError(f"`h_pose` must have shape [B, T, 144], got {tuple(h_pose.shape)}.")

        if v_masked.shape[:2] != m_human.shape[:2] or v_masked.shape[:2] != h_pose.shape[:2]:
            raise ValueError("`v_masked`, `m_human`, and `h_pose` must share the same batch and time dimensions.")

        video = torch.cat([v_masked, m_human], dim=2)
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        video_tokens = self.video_encoder(video).mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        pose_tokens = self.pose_mlp(h_pose.to(dtype=video_tokens.dtype))

        fused = self.fuse(torch.cat([video_tokens, pose_tokens], dim=-1))
        fused = fused + sinusoidal_position_embedding(
            sequence_length=fused.shape[1],
            dim=fused.shape[2],
            device=fused.device,
            dtype=fused.dtype,
        )
        fused = self.temporal_transformer(fused)
        return self.output_norm(fused)


class ZeroInitCrossAttention(nn.Module):
    """
    Multi-head cross-attention with a zero-initialized output projection.

    This ensures the adapter starts as an exact no-op, similar to ControlNet's
    zero-conv initialization strategy.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim or query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        if self.head_dim * num_heads != query_dim:
            raise ValueError(f"query_dim={query_dim} must be divisible by num_heads={num_heads}.")

        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(self.context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(self.context_dim, query_dim, bias=qkv_bias)
        self.to_out = _zero_module(nn.Linear(query_dim, query_dim, bias=True))
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        batch_size, query_length, _ = hidden_states.shape
        context_length = encoder_hidden_states.shape[1]

        query = self.to_q(self.query_norm(hidden_states))
        key = self.to_k(self.context_norm(encoder_hidden_states))
        value = self.to_v(self.context_norm(encoder_hidden_states))

        query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attn_probs = self.attention_dropout(attn_scores.softmax(dim=-1))
            attended = torch.matmul(attn_probs, value)

        attended = attended.transpose(1, 2).reshape(batch_size, query_length, self.query_dim)
        return self.proj_dropout(self.to_out(attended))


class _ZeroInitResidualMLPBlock(nn.Module):
    """Residual MLP block initialized as a no-op."""

    def __init__(self, dim: int, expansion_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))
        return residual + x


class GaussianLatentBridge(nn.Module):
    """
    Trainable bridge between 14D 3DGS tokens and Hunyuan's native 64D latent tokens.

    The bridge starts from a partial identity mapping so optimization begins from
    a stable state, then learns a richer latent alignment through residual MLP
    blocks and a lightweight global set-conditioning path.
    """

    def __init__(
        self,
        token_dim: int = 14,
        latent_dim: int = 64,
        *,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        dropout: float = 0.0,
        expansion_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)

        self.encoder_norm = nn.Identity()
        self.encoder_proj = nn.Linear(self.token_dim, self.latent_dim)
        self.encoder_global = _zero_module(nn.Linear(self.latent_dim, self.latent_dim, bias=True))
        self.encoder_blocks = nn.ModuleList(
            [
                _ZeroInitResidualMLPBlock(
                    self.latent_dim,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_out = nn.Identity()

        self.decoder_in = nn.Identity()
        self.decoder_global = _zero_module(nn.Linear(self.latent_dim, self.latent_dim, bias=True))
        self.decoder_blocks = nn.ModuleList(
            [
                _ZeroInitResidualMLPBlock(
                    self.latent_dim,
                    expansion_ratio=expansion_ratio,
                    dropout=dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_proj = nn.Linear(self.latent_dim, self.token_dim)

        self._init_partial_identity()

    def _init_partial_identity(self) -> None:
        nn.init.zeros_(self.encoder_proj.weight)
        nn.init.zeros_(self.encoder_proj.bias)
        nn.init.zeros_(self.decoder_proj.weight)
        nn.init.zeros_(self.decoder_proj.bias)

        shared_dim = min(self.token_dim, self.latent_dim)
        with torch.no_grad():
            self.encoder_proj.weight[:shared_dim, :shared_dim] = torch.eye(shared_dim)
            self.decoder_proj.weight[:shared_dim, :shared_dim] = torch.eye(shared_dim)

    def _add_global_context(self, x: Tensor, projection: nn.Linear) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        return x + projection(pooled)

    def encode(self, gs_tokens: Tensor) -> Tensor:
        if gs_tokens.ndim != 3 or gs_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"`gs_tokens` must have shape [B, N, {self.token_dim}], got {tuple(gs_tokens.shape)}."
            )
        latent = self.encoder_proj(self.encoder_norm(gs_tokens))
        latent = self._add_global_context(latent, self.encoder_global)
        for block in self.encoder_blocks:
            latent = block(latent)
        return self.encoder_out(latent)

    def decode(self, latent_tokens: Tensor) -> Tensor:
        if latent_tokens.ndim != 3 or latent_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"`latent_tokens` must have shape [B, N, {self.latent_dim}], got {tuple(latent_tokens.shape)}."
            )
        gs_tokens = self.decoder_in(latent_tokens)
        gs_tokens = self._add_global_context(gs_tokens, self.decoder_global)
        for block in self.decoder_blocks:
            gs_tokens = block(gs_tokens)
        return self.decoder_proj(gs_tokens)

    def reconstruct(self, gs_tokens: Tensor) -> Tensor:
        return self.decode(self.encode(gs_tokens))

    def latent_prior_loss(self, latent_tokens: Tensor) -> Tensor:
        mean = latent_tokens.mean(dim=(1, 2))
        std = latent_tokens.std(dim=(1, 2), unbiased=False)
        return mean.square().mean() + (std - 1.0).square().mean()


@dataclass
class ConditionedHunyuanFMOutput(BaseOutput):
    sample: Tensor
    cond_tokens: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None


class Hunyuan3D2ControlNet(ModelMixin):
    """
    Frozen Hunyuan3D-2 DiT with ControlNet-style interaction conditioning.

    The wrapper mirrors the public `Hunyuan3DDiT` forward path and injects
    interaction tokens as residual latent updates through zero-initialized
    cross-attention adapters.
    """

    def __init__(
        self,
        frozen_hunyuan_dit: nn.Module,
        condition_dim: Optional[int] = None,
        *,
        interaction_condition_encoder: Optional[nn.Module] = None,
        inject_double_blocks: Optional[Sequence[int]] = None,
        inject_single_blocks: Optional[Sequence[int]] = (),
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        required_attributes = (
            "latent_in",
            "time_in",
            "cond_in",
            "double_blocks",
            "single_blocks",
            "final_layer",
            "hidden_size",
            "num_heads",
        )
        missing = [name for name in required_attributes if not hasattr(frozen_hunyuan_dit, name)]
        if missing:
            raise ValueError(
                "The provided `frozen_hunyuan_dit` does not match Hunyuan3D-2's public DiT interface. "
                f"Missing attributes: {missing}"
            )

        self.frozen_hunyuan_dit = frozen_hunyuan_dit
        self.hidden_size = int(frozen_hunyuan_dit.hidden_size)
        self.num_heads = int(frozen_hunyuan_dit.num_heads)
        self.time_factor = float(getattr(frozen_hunyuan_dit, "time_factor", 1000.0))
        self.condition_dim = int(condition_dim or self.hidden_size)
        self.input_dim = int(getattr(self.frozen_hunyuan_dit.latent_in, "in_features", self.hidden_size))
        self.context_in_dim = int(getattr(self.frozen_hunyuan_dit.cond_in, "in_features", self.hidden_size))

        self.condition_encoder = interaction_condition_encoder or InteractionConditionEncoder(
            output_dim=self.condition_dim,
            num_attention_heads=max(1, min(8, self.num_heads)),
        )

        self.double_block_ids = _normalize_block_indices(
            inject_double_blocks,
            len(self.frozen_hunyuan_dit.double_blocks),
            default_all=True,
        )
        self.single_block_ids = _normalize_block_indices(
            inject_single_blocks,
            len(self.frozen_hunyuan_dit.single_blocks),
            default_all=False,
        )

        self.double_injections = nn.ModuleDict(
            {
                str(block_idx): ZeroInitCrossAttention(
                    query_dim=self.hidden_size,
                    context_dim=self.condition_dim,
                    num_heads=self.num_heads,
                    attention_dropout=attention_dropout,
                    proj_dropout=proj_dropout,
                )
                for block_idx in self.double_block_ids
            }
        )
        self.single_injections = nn.ModuleDict(
            {
                str(block_idx): ZeroInitCrossAttention(
                    query_dim=self.hidden_size,
                    context_dim=self.condition_dim,
                    num_heads=self.num_heads,
                    attention_dropout=attention_dropout,
                    proj_dropout=proj_dropout,
                )
                for block_idx in self.single_block_ids
            }
        )

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.frozen_hunyuan_dit.requires_grad_(False)
        self.frozen_hunyuan_dit.eval()

    def train(self, mode: bool = True) -> "Hunyuan3D2ControlNet":
        super().train(mode)
        # Keep the pretrained DiT frozen in eval mode, while adapters remain trainable.
        self.frozen_hunyuan_dit.eval()
        return self

    def encode_condition(self, v_masked: Tensor, m_human: Tensor, h_pose: Tensor) -> Tensor:
        return self.condition_encoder(v_masked=v_masked, m_human=m_human, h_pose=h_pose)

    def _build_time_embedding(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
        time_embedding = hunyuan_timestep_embedding(
            timesteps,
            dim=256,
            time_factor=self.time_factor,
        ).to(dtype=dtype)
        return self.frozen_hunyuan_dit.time_in(time_embedding)

    def _inject(
        self,
        hidden_states: Tensor,
        cond_tokens: Tensor,
        injections: nn.ModuleDict,
        block_idx: int,
    ) -> Tensor:
        key = str(block_idx)
        if key not in injections:
            return hidden_states
        injection = injections[key]
        return hidden_states + injection(hidden_states, cond_tokens)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        contexts: Optional[Dict[str, Tensor]] = None,
        *,
        v_masked: Optional[Tensor] = None,
        m_human: Optional[Tensor] = None,
        h_pose: Optional[Tensor] = None,
        cond_tokens: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ConditionedHunyuanFMOutput, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tuple[Tensor, ...]]]:
        """
        Forward pass matching `Hunyuan3DDiT.forward(x, t, contexts, ...)`.

        Parameters
        ----------
        x:
            `[B, N_latents, C]` latent shape tokens.
        t:
            `[B]` flow-matching timestep.
        contexts:
            Must contain `contexts["main"]` with the frozen Hunyuan image-condition tokens.
        v_masked / m_human / h_pose:
            Interaction inputs used when `cond_tokens` is not provided.
        cond_tokens:
            Optional precomputed interaction conditioning tokens of shape `[B, L, D_cond]`.
        """

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"`x.shape[-1]` ({x.shape[-1]}) must match the frozen Hunyuan input dim ({self.input_dim})."
            )

        if cond_tokens is None:
            if v_masked is None or m_human is None or h_pose is None:
                raise ValueError(
                    "Provide either `cond_tokens` directly or the triplet `(v_masked, m_human, h_pose)`."
                )
            cond_tokens = self.encode_condition(v_masked=v_masked, m_human=m_human, h_pose=h_pose)
        elif cond_tokens.ndim != 3:
            raise ValueError(f"`cond_tokens` must have shape [B, L, D], got {tuple(cond_tokens.shape)}.")

        if cond_tokens.shape[-1] != self.condition_dim:
            raise ValueError(
                f"`cond_tokens.shape[-1]` ({cond_tokens.shape[-1]}) must match `condition_dim` ({self.condition_dim})."
            )

        latent = self.frozen_hunyuan_dit.latent_in(x)
        vec = self._build_time_embedding(t, dtype=latent.dtype)
        if contexts is None or "main" not in contexts:
            raise ValueError("`contexts['main']` is required for Hunyuan3D2ControlNet forward.")
        main_context = contexts["main"].to(device=latent.device, dtype=latent.dtype)
        text_context = self.frozen_hunyuan_dit.cond_in(main_context)
        cond_tokens = cond_tokens.to(device=latent.device, dtype=latent.dtype)

        hidden_states = [] if output_hidden_states else None
        pe = None

        for block_idx, block in enumerate(self.frozen_hunyuan_dit.double_blocks):
            latent, text_context = block(img=latent, txt=text_context, vec=vec, pe=pe)
            latent = self._inject(latent, cond_tokens, self.double_injections, block_idx)
            if hidden_states is not None:
                hidden_states.append(latent)

        latent = torch.cat((text_context, latent), dim=1)
        for block_idx, block in enumerate(self.frozen_hunyuan_dit.single_blocks):
            latent = block(latent, vec=vec, pe=pe)
            latent = self._inject(latent, cond_tokens, self.single_injections, block_idx)
            if hidden_states is not None:
                hidden_states.append(latent)

        latent = latent[:, text_context.shape[1] :, ...]
        sample = self.frozen_hunyuan_dit.final_layer(latent, vec)

        if not return_dict:
            if hidden_states is None:
                return sample, cond_tokens
            return sample, cond_tokens, tuple(hidden_states)

        return ConditionedHunyuanFMOutput(
            sample=sample,
            cond_tokens=cond_tokens,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )


__all__ = [
    "ConditionedHunyuanFMOutput",
    "GaussianLatentBridge",
    "Hunyuan3D2ControlNet",
    "InteractionConditionEncoder",
    "ZeroInitCrossAttention",
]
