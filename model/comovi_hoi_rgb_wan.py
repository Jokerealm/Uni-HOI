from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .cointeract_hoi_wan import (
    DecodedHOIState,
    FirstFramePatchEncoder,
    FrozenWanTI2VImageStream,
    HOIStateCodec,
    MultiHeadAttention,
    SharedStreamTransformerBlock,
    ZeroInitCrossAdapter,
    timestep_embedding,
)


class CoMoViCrossAttentionLayer(nn.Module):
    """Self-attention + cross-attention + FFN layer used for 3D-2D alignment."""

    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(hidden_dim * mlp_ratio))
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: Tensor, context_tokens: Tensor) -> Tensor:
        query_tokens = query_tokens + self.self_attn(self.self_norm(query_tokens))
        query_tokens = query_tokens + self.cross_attn(
            self.cross_norm(query_tokens),
            self.context_norm(context_tokens),
        )
        return query_tokens + self.mlp(self.mlp_norm(query_tokens))


class CoMoViVisualPriorResampler(nn.Module):
    """Compress dense Wan/image tokens into a compact RGB prior branch."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_frames: int,
        num_global_tokens: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.num_global_tokens = int(num_global_tokens)
        self.frame_queries = nn.Parameter(torch.zeros(1, self.num_frames, hidden_dim))
        self.global_queries = nn.Parameter(torch.zeros(1, self.num_global_tokens, hidden_dim))
        self.layers = nn.ModuleList(
            [
                CoMoViCrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.frame_queries, std=0.02)
        nn.init.normal_(self.global_queries, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self.num_frames + self.num_global_tokens

    def forward(self, *, rgb_tokens: Tensor, image_tokens: Tensor, time_cond: Tensor) -> Tensor:
        batch_size = rgb_tokens.shape[0]
        queries = torch.cat(
            [
                self.frame_queries.expand(batch_size, -1, -1),
                self.global_queries.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        queries = queries.to(device=rgb_tokens.device, dtype=rgb_tokens.dtype) + time_cond
        context = torch.cat([rgb_tokens, image_tokens], dim=1)
        for layer in self.layers:
            queries = layer(queries, context)
        return self.norm(queries)


class CoMoViDualBranchBlock(nn.Module):
    """Shared dual-branch block with zero-init mutual feature interaction."""

    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.shared_block = SharedStreamTransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.rgb_to_hoi = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout)
        self.hoi_to_rgb = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout)
        self.rgb_to_hoi_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.hoi_to_rgb_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(
        self,
        *,
        hoi_tokens: Tensor,
        rgb_prior_tokens: Tensor,
        time_cond: Tensor,
        rgb_to_hoi_scale: float,
        hoi_to_rgb_scale: float,
        run_hoi_to_rgb: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        hoi_tokens = self.shared_block(hoi_tokens, time_cond=time_cond, stream="hoi")
        rgb_prior_tokens = self.shared_block(rgb_prior_tokens, time_cond=time_cond, stream="rgb")

        hoi_tokens = hoi_tokens + float(rgb_to_hoi_scale) * self.rgb_to_hoi_gate(hoi_tokens) * self.rgb_to_hoi(
            hoi_tokens,
            rgb_prior_tokens,
        )
        if run_hoi_to_rgb or float(hoi_to_rgb_scale) != 0.0:
            rgb_prior_tokens = rgb_prior_tokens + float(hoi_to_rgb_scale) * self.hoi_to_rgb_gate(
                rgb_prior_tokens
            ) * self.hoi_to_rgb(rgb_prior_tokens, hoi_tokens)
        return hoi_tokens, rgb_prior_tokens


class CoMoVi3D2DRefiner(nn.Module):
    """HOI-query refiner that attends to RGB prior tokens after branch coupling."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CoMoViCrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, *, hoi_tokens: Tensor, rgb_prior_tokens: Tensor) -> Tensor:
        refined = hoi_tokens
        for layer in self.layers:
            refined = layer(refined, rgb_prior_tokens)
        return self.out(self.norm(refined))


@dataclass
class CoMoViHOIRGBOutput:
    rgb_velocity: Tensor
    state_velocity: Tensor
    decoded_state: DecodedHOIState
    rgb_hidden_tokens: Tensor
    rgb_prior_tokens: Tensor
    hoi_tokens: Tensor
    predicted_clean_state: Tensor


class CoMoViHOIRGBModel(nn.Module):
    """CoMoVi-like HOI-primary RGB-guided dual-branch model.

    RGB/Wan stays as the visual-prior branch. HOI tokens are the primary denoising
    branch and final supervised output. The CoMoVi-style parts are the full dual
    branch update, zero-init mutual feature interaction, and the 3D-2D
    cross-attention refiner that lets HOI query compact RGB prior tokens.
    """

    video_backend = "wan2.2-ti2v-5b"
    input_mode = "comovi_hoi_primary_rgb_prior"

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        num_frames: int,
        image_height: int,
        image_width: int,
        image_patch_size: int,
        num_human_gaussians: int,
        num_object_gaussians: int,
        num_joints: int,
        contact_dim: int,
        human_shape_dim: int,
        human_pose_dim: int,
        wan_model_id: str,
        wan_dtype: str = "bf16",
        wan_hidden_dim: int = 3072,
        wan_prompt_max_sequence_length: int = 512,
        wan_local_files_only: bool = True,
        freeze_wan: bool = True,
        detach_rgb_context: bool = True,
        enable_hoi_to_rgb: bool = False,
        visual_prior_num_global_tokens: int = 8,
        visual_resampler_depth: int = 2,
        cross_3d2d_depth: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.detach_rgb_context = bool(detach_rgb_context)
        self.enable_hoi_to_rgb = bool(enable_hoi_to_rgb)

        self.state_codec = HOIStateCodec(
            hidden_dim=hidden_dim,
            num_human_gaussians=num_human_gaussians,
            num_object_gaussians=num_object_gaussians,
            num_frames=num_frames,
            num_joints=num_joints,
            contact_dim=contact_dim,
            human_shape_dim=human_shape_dim,
            human_pose_dim=human_pose_dim,
        )
        self.rgb_stream = FrozenWanTI2VImageStream(
            model_id=wan_model_id,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
            torch_dtype=wan_dtype,
            prompt_max_sequence_length=wan_prompt_max_sequence_length,
            local_files_only=wan_local_files_only,
            freeze=freeze_wan,
        )
        self.rgb_token_adapter = nn.Sequential(
            nn.LayerNorm(int(wan_hidden_dim)),
            nn.Linear(int(wan_hidden_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.first_frame_encoder = FirstFramePatchEncoder(
            in_channels=3,
            hidden_dim=hidden_dim,
            patch_size=image_patch_size,
            image_height=image_height,
            image_width=image_width,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.visual_resampler = CoMoViVisualPriorResampler(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_frames=num_frames,
            num_global_tokens=visual_prior_num_global_tokens,
            depth=visual_resampler_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                CoMoViDualBranchBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.cross_3d2d_refiner = CoMoVi3D2DRefiner(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=cross_3d2d_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        if not self.enable_hoi_to_rgb:
            for block in self.blocks:
                block.hoi_to_rgb.requires_grad_(False)
                block.hoi_to_rgb_gate.requires_grad_(False)

        self.state_norm = nn.LayerNorm(hidden_dim)
        self.state_velocity_head = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.state_velocity_head.weight)
        nn.init.zeros_(self.state_velocity_head.bias)

    def ensure_wan_loaded(self, device: torch.device) -> None:
        self.rgb_stream.ensure_loaded(device)

    def encode_state_target(self, **kwargs) -> Tensor:
        return self.state_codec.encode_targets(**kwargs)

    def decode_state_tokens(self, state_tokens: Tensor) -> DecodedHOIState:
        return self.state_codec.decode_tokens(state_tokens)

    @torch.no_grad()
    def encode_video_target(self, video: Tensor) -> Tensor:
        return self.rgb_stream.encode_video(video)

    @torch.no_grad()
    def decode_video_latents(self, latents: Tensor) -> Tensor:
        return self.rgb_stream.decode_video(latents)

    @torch.no_grad()
    def encode_first_frame(self, first_frame: Tensor) -> Tensor:
        return self.rgb_stream.encode_first_frame(first_frame)

    def build_noisy_video_latents(
        self,
        target_latents: Tensor,
        first_frame_latents: Tensor,
        timesteps: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.rgb_stream.build_noisy_latents(target_latents, first_frame_latents, timesteps, generator=generator)

    def forward(
        self,
        *,
        video_xt: Tensor,
        state_xt: Tensor,
        timesteps: Tensor,
        first_frame: Tensor,
        rgb_to_hoi_scale: float = 1.0,
        hoi_to_rgb_scale: float = 0.0,
        cross_3d2d_scale: float = 1.0,
    ) -> CoMoViHOIRGBOutput:
        if state_xt.ndim != 3 or state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(f"`state_xt` must have shape [B, L, {self.hidden_dim}], got {tuple(state_xt.shape)}.")

        rgb_output = self.rgb_stream(video_xt=video_xt, timesteps=timesteps)
        rgb_hidden_tokens = rgb_output.hidden_tokens.detach() if self.detach_rgb_context else rgb_output.hidden_tokens
        rgb_tokens = self.rgb_token_adapter(rgb_hidden_tokens.to(dtype=state_xt.dtype))
        image_tokens = self.first_frame_encoder(first_frame.to(dtype=state_xt.dtype))
        time_cond = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1).to(dtype=state_xt.dtype)

        hoi_tokens = state_xt + time_cond
        rgb_tokens = rgb_tokens + time_cond
        image_tokens = image_tokens + time_cond
        rgb_prior_tokens = self.visual_resampler(
            rgb_tokens=rgb_tokens,
            image_tokens=image_tokens,
            time_cond=time_cond,
        )

        for block in self.blocks:
            hoi_tokens, rgb_prior_tokens = block(
                hoi_tokens=hoi_tokens,
                rgb_prior_tokens=rgb_prior_tokens,
                time_cond=time_cond,
                rgb_to_hoi_scale=rgb_to_hoi_scale,
                hoi_to_rgb_scale=hoi_to_rgb_scale if self.enable_hoi_to_rgb else 0.0,
                run_hoi_to_rgb=self.enable_hoi_to_rgb,
            )
        hoi_tokens = hoi_tokens + float(cross_3d2d_scale) * self.cross_3d2d_refiner(
            hoi_tokens=hoi_tokens,
            rgb_prior_tokens=rgb_prior_tokens,
        )

        state_velocity = self.state_velocity_head(self.state_norm(hoi_tokens))
        state_velocity = state_velocity + hoi_tokens.float().sum().to(dtype=state_velocity.dtype) * 0.0
        t_view = timesteps.to(device=state_xt.device, dtype=state_xt.dtype).view(state_xt.shape[0], 1, 1)
        predicted_clean_state = state_xt + (1.0 - t_view) * state_velocity.to(dtype=state_xt.dtype)
        return CoMoViHOIRGBOutput(
            rgb_velocity=rgb_output.velocity,
            state_velocity=state_velocity,
            decoded_state=self.decode_state_tokens(predicted_clean_state),
            rgb_hidden_tokens=rgb_output.hidden_tokens,
            rgb_prior_tokens=rgb_prior_tokens,
            hoi_tokens=hoi_tokens,
            predicted_clean_state=predicted_clean_state,
        )


__all__ = [
    "CoMoVi3D2DRefiner",
    "CoMoViDualBranchBlock",
    "CoMoViHOIRGBModel",
    "CoMoViHOIRGBOutput",
    "CoMoViVisualPriorResampler",
]
