from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .hoi_state_codec import DecodedHOIState, HOIStateCodec
from .wan_rgb_stream import FrozenWanTI2VImageStream, WanRGBStreamOutput


@dataclass
class WanBackboneHOI4DOutput:
    """Wan-backbone HOI output with optional RGB velocity for auxiliary FM loss."""

    rgb_velocity: Optional[Tensor]
    state_velocity: Tensor
    decoded_state: DecodedHOIState
    rgb_hidden_tokens: Optional[Tensor]
    rgb_context_tokens: Optional[Tensor]
    hoi_tokens: Tensor
    hoi_router_loss: Tensor


class WanBackboneHOI4DModel(nn.Module):
    """CoInteract-style HOI denoiser that reuses the pretrained Wan TI2V DiT blocks.

    RGB video latents are patchified by the Wan transformer. Explicit HOI state tokens
    are concatenated to that sequence and passed through the same pretrained DiT
    backbone, so the HOI branch can attend through Wan's video prior instead of a
    small randomly initialized transformer.
    """

    video_backend = "wan2.2-ti2v-5b"
    input_mode = "wan_backbone_rgb_video_guided_hoi"

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
    ) -> None:
        super().__init__()
        del hidden_dim, num_heads, depth, mlp_ratio, dropout
        self.hidden_dim = int(wan_hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.detach_rgb_context = bool(detach_rgb_context)

        self.state_codec = HOIStateCodec(
            hidden_dim=self.hidden_dim,
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
        self.hoi_type_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.state_velocity_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.state_velocity_head.weight)
        nn.init.zeros_(self.state_velocity_head.bias)

    def ensure_wan_loaded(self, device: torch.device) -> None:
        self.rgb_stream.ensure_loaded(device)
        actual_dim = int(
            self.rgb_stream.transformer.config.num_attention_heads
            * self.rgb_stream.transformer.config.attention_head_dim
        )
        if actual_dim != self.hidden_dim:
            raise RuntimeError(
                "Wan backbone hidden dim mismatch: "
                f"configured wan_hidden_dim={self.hidden_dim}, transformer dim={actual_dim}."
            )

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

    def _hoi_token_timesteps(self, timesteps: Tensor, token_count: int, *, dtype: torch.dtype) -> Tensor:
        scalar = (1.0 - timesteps.float()).clamp(0.0, 1.0)
        scalar = scalar * float(self.rgb_stream.scheduler.config.num_train_timesteps)
        return scalar.unsqueeze(1).expand(-1, int(token_count)).to(device=timesteps.device, dtype=dtype)

    @staticmethod
    def _append_identity_rotary(rotary_emb, token_count: int, *, device: torch.device):
        if int(token_count) <= 0:
            return rotary_emb
        cos, sin = rotary_emb
        extra_cos = torch.ones(
            cos.shape[0],
            int(token_count),
            *cos.shape[2:],
            device=device,
            dtype=cos.dtype,
        )
        extra_sin = torch.zeros(
            sin.shape[0],
            int(token_count),
            *sin.shape[2:],
            device=device,
            dtype=sin.dtype,
        )
        return torch.cat([cos.to(device=device), extra_cos], dim=1), torch.cat([sin.to(device=device), extra_sin], dim=1)

    def _identity_rotary(self, token_count: int, *, device: torch.device):
        transformer = self.rgb_stream.transformer
        head_dim = int(transformer.config.attention_head_dim)
        cos = torch.ones(1, int(token_count), 1, head_dim, device=device, dtype=transformer.rope.freqs_cos.dtype)
        sin = torch.zeros(1, int(token_count), 1, head_dim, device=device, dtype=transformer.rope.freqs_sin.dtype)
        return cos, sin

    def _project_video_velocity(self, video_hidden: Tensor, video_temb: Tensor, video_xt: Tensor) -> Tensor:
        transformer = self.rgb_stream.transformer
        batch_size, _, num_frames, height, width = video_xt.shape
        p_t, p_h, p_w = transformer.config.patch_size
        post_f = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w

        shift, scale = (transformer.scale_shift_table.unsqueeze(0).to(video_temb.device) + video_temb.unsqueeze(2)).chunk(
            2,
            dim=2,
        )
        shift = shift.squeeze(2).to(video_hidden.device)
        scale = scale.squeeze(2).to(video_hidden.device)
        tokens = (transformer.norm_out(video_hidden.float()) * (1 + scale) + shift).type_as(video_hidden)
        tokens = transformer.proj_out(tokens)
        tokens = tokens.reshape(batch_size, post_f, post_h, post_w, p_t, p_h, p_w, -1)
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _forward_wan_backbone(
        self,
        *,
        video_xt: Optional[Tensor],
        state_xt: Tensor,
        timesteps: Tensor,
        use_rgb_prior: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        self.ensure_wan_loaded(state_xt.device)
        transformer = self.rgb_stream.transformer
        if self.rgb_stream.freeze:
            self.rgb_stream.vae.eval()
            transformer.eval()
            self.rgb_stream.text_encoder.eval()

        prompt_embeds = self.rgb_stream._encode_empty_prompt(batch_size=state_xt.shape[0], device=state_xt.device)
        video_tokens = None
        video_temb = None
        num_video_tokens = 0
        rotary_emb = None

        if use_rgb_prior:
            if video_xt is None:
                raise ValueError("`video_xt` is required when use_rgb_prior=True.")
            video_xt = video_xt.to(device=state_xt.device, dtype=transformer.dtype)
            rotary_emb = transformer.rope(video_xt)
            video_tokens = transformer.patch_embedding(video_xt).flatten(2).transpose(1, 2)
            if self.detach_rgb_context:
                video_tokens = video_tokens.detach()
            num_video_tokens = int(video_tokens.shape[1])
            video_token_timesteps = self.rgb_stream._build_token_timesteps(timesteps, video_xt)
        else:
            video_token_timesteps = None

        hoi_tokens = state_xt.to(device=state_xt.device, dtype=transformer.dtype) + self.hoi_type_embedding.to(
            device=state_xt.device,
            dtype=transformer.dtype,
        )
        hoi_token_timesteps = self._hoi_token_timesteps(timesteps, hoi_tokens.shape[1], dtype=transformer.dtype)
        if video_tokens is not None:
            tokens = torch.cat([video_tokens, hoi_tokens], dim=1)
            token_timesteps = torch.cat([video_token_timesteps, hoi_token_timesteps], dim=1)
            rotary_emb = self._append_identity_rotary(rotary_emb, hoi_tokens.shape[1], device=tokens.device)
        else:
            tokens = hoi_tokens
            token_timesteps = hoi_token_timesteps
            rotary_emb = self._identity_rotary(hoi_tokens.shape[1], device=tokens.device)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            token_timesteps.flatten(),
            prompt_embeds,
            None,
            timestep_seq_len=token_timesteps.shape[1],
        )
        if encoder_hidden_states_image is not None:
            raise RuntimeError("Wan2.2-TI2V-5B should not require image encoder hidden states.")
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in transformer.blocks:
            tokens = block(tokens, encoder_hidden_states, timestep_proj, rotary_emb)

        if num_video_tokens > 0:
            video_hidden = tokens[:, :num_video_tokens]
            hoi_hidden = tokens[:, num_video_tokens:]
            video_temb = temb[:, :num_video_tokens]
        else:
            video_hidden = None
            hoi_hidden = tokens
        return hoi_hidden, video_hidden, video_temb, video_xt

    def forward(
        self,
        *,
        video_xt: Optional[Tensor],
        state_xt: Tensor,
        timesteps: Tensor,
        use_rgb_prior: bool = True,
    ) -> WanBackboneHOI4DOutput:
        if state_xt.ndim != 3 or state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(f"`state_xt` must have shape [B, L, {self.hidden_dim}], got {tuple(state_xt.shape)}.")

        hoi_hidden, video_hidden, video_temb, video_xt = self._forward_wan_backbone(
            video_xt=video_xt,
            state_xt=state_xt,
            timesteps=timesteps,
            use_rgb_prior=use_rgb_prior,
        )
        state_velocity = self.state_velocity_head(self.state_norm(hoi_hidden))
        state_velocity = state_velocity + hoi_hidden.float().sum().to(dtype=state_velocity.dtype) * 0.0
        t_view = timesteps.to(device=state_xt.device, dtype=state_xt.dtype).view(state_xt.shape[0], 1, 1)
        predicted_clean_state = state_xt + (1.0 - t_view) * state_velocity.to(dtype=state_xt.dtype)

        rgb_velocity = None
        if video_hidden is not None and video_temb is not None and video_xt is not None:
            rgb_velocity = self._project_video_velocity(video_hidden, video_temb, video_xt)

        return WanBackboneHOI4DOutput(
            rgb_velocity=rgb_velocity,
            state_velocity=state_velocity,
            decoded_state=self.decode_state_tokens(predicted_clean_state),
            rgb_hidden_tokens=video_hidden,
            rgb_context_tokens=video_hidden,
            hoi_tokens=hoi_hidden,
            hoi_router_loss=state_velocity.new_zeros(()),
        )

    def forward_hoi_only(self, *, state_xt: Tensor, timesteps: Tensor) -> WanBackboneHOI4DOutput:
        return self.forward(video_xt=None, state_xt=state_xt, timesteps=timesteps, use_rgb_prior=False)


__all__ = [
    "WanBackboneHOI4DModel",
    "WanBackboneHOI4DOutput",
]
