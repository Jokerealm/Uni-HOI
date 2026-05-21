from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .cointeract_hoi_wan import DecodedHOIState, HOIStateCodec, timestep_embedding


def _resolve_torch_dtype(name: str) -> torch.dtype:
    text = str(name).strip().lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}. Expected bf16/fp16/fp32.")


def _retrieve_latents(encoder_output, sample_mode: str = "argmax") -> Tensor:
    if hasattr(encoder_output, "latent_dist"):
        if sample_mode == "sample":
            return encoder_output.latent_dist.sample()
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    if isinstance(encoder_output, (tuple, list)) and encoder_output:
        return encoder_output[0]
    raise AttributeError("Could not retrieve latents from Wan VAE output.")


def _as_3tuple(value: int | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"Expected a 3D patch size, got {value!r}.")
    return result


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FrozenWanVAEEncoder(nn.Module):
    """Frozen Wan2.2 VAE encoder used as the visual feature front-end."""

    def __init__(
        self,
        *,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        subfolder: str = "vae",
        torch_dtype: str = "bf16",
        local_files_only: bool = True,
        sample_mode: str = "argmax",
        load_on_init: bool = False,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.subfolder = str(subfolder)
        self.torch_dtype_name = str(torch_dtype)
        self.local_files_only = bool(local_files_only)
        self.sample_mode = str(sample_mode)
        self._loaded = False
        if load_on_init:
            self.ensure_loaded(torch.device("cpu"))

    @property
    def loaded(self) -> bool:
        return bool(self._loaded)

    def _load(self, device: torch.device) -> None:
        if self._loaded:
            self.to(device=device)
            return
        from diffusers import AutoencoderKLWan

        dtype = _resolve_torch_dtype(self.torch_dtype_name)
        try:
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder=self.subfolder,
                torch_dtype=dtype,
                local_files_only=self.local_files_only,
            )
        except OSError:
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                local_files_only=self.local_files_only,
            )
        self.vae = vae.requires_grad_(False).eval()
        self.latent_channels = int(getattr(self.vae.config, "z_dim", getattr(self.vae.config, "latent_channels", 16)))

        latents_mean = getattr(self.vae.config, "latents_mean", None)
        latents_std = getattr(self.vae.config, "latents_std", None)
        if latents_mean is None:
            latents_mean = [0.0] * self.latent_channels
        if latents_std is None:
            latents_std = [1.0] * self.latent_channels
        self.register_buffer(
            "latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32).view(1, self.latent_channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std_inv",
            (1.0 / torch.tensor(latents_std, dtype=torch.float32)).view(1, self.latent_channels, 1, 1, 1),
            persistent=False,
        )
        self._loaded = True
        self.to(device=device)

    def train(self, mode: bool = True) -> "FrozenWanVAEEncoder":
        super().train(mode)
        if self._loaded:
            self.vae.eval()
        return self

    def ensure_loaded(self, device: torch.device) -> None:
        self._load(device)

    def _normalize_latents(self, latents: Tensor) -> Tensor:
        mean = self.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std_inv = self.latents_std_inv.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) * std_inv

    @torch.no_grad()
    def forward(self, video: Tensor) -> Tensor:
        self.ensure_loaded(video.device)
        if video.ndim == 4:
            video = video.unsqueeze(1)
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Wan VAE expects image/video [B, 3, H, W] or [B, T, 3, H, W], got {tuple(video.shape)}.")
        vae_dtype = next(self.vae.parameters()).dtype
        video = video.float().clamp(0.0, 1.0).mul(2.0).sub(1.0)
        video = video.permute(0, 2, 1, 3, 4).contiguous().to(device=video.device, dtype=vae_dtype)
        latents = _retrieve_latents(self.vae.encode(video), sample_mode=self.sample_mode)
        return self._normalize_latents(latents).float()


class WanVAELatentPatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        hidden_dim: int,
        num_frames: int,
        image_height: int,
        image_width: int,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 16,
        patch_size: int | Sequence[int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.vae_scale_factor_temporal = int(vae_scale_factor_temporal)
        self.vae_scale_factor_spatial = int(vae_scale_factor_spatial)
        self.patch_size = _as_3tuple(patch_size)

        self.latent_frames = (self.num_frames - 1) // self.vae_scale_factor_temporal + 1
        self.latent_height = self.image_height // self.vae_scale_factor_spatial
        self.latent_width = self.image_width // self.vae_scale_factor_spatial
        if self.image_height % self.vae_scale_factor_spatial or self.image_width % self.vae_scale_factor_spatial:
            raise ValueError("Image size must be divisible by the Wan VAE spatial scale factor.")
        if self.latent_frames % self.patch_size[0] or self.latent_height % self.patch_size[1] or self.latent_width % self.patch_size[2]:
            raise ValueError(
                "Latent grid must be divisible by latent patch size. "
                f"grid=({self.latent_frames}, {self.latent_height}, {self.latent_width}), patch={self.patch_size}."
            )

        self.proj = nn.Conv3d(
            self.latent_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        num_tokens = (
            self.latent_frames // self.patch_size[0]
            * self.latent_height // self.patch_size[1]
            * self.latent_width // self.patch_size[2]
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    @property
    def num_tokens(self) -> int:
        return int(self.pos_embed.shape[1])

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Wan VAE latents must have shape [B, C, T, H, W], got {tuple(latents.shape)}.")
        expected = (self.latent_channels, self.latent_frames, self.latent_height, self.latent_width)
        if tuple(latents.shape[1:]) != expected:
            raise ValueError(f"Expected latent shape [B, {expected}], got {tuple(latents.shape)}.")
        tokens = self.proj(latents).flatten(2).transpose(1, 2).contiguous()
        return tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """A minimal AdaLN-Zero DiT block over visual and HOI tokens."""

    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim}/{num_heads}.")
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        attn_in = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)


@dataclass
class UniModelOutput:
    decoded_state: DecodedHOIState
    hoi_tokens: Tensor
    visual_tokens: Tensor
    all_tokens: Tensor
    state_velocity: Optional[Tensor] = None
    predicted_clean_state_tokens: Optional[Tensor] = None
    vae_latents: Optional[Tensor] = None


class UniModel(nn.Module):
    """Single-stream Wan-VAE + DiT baseline for single-image RGB-to-HOI reconstruction."""

    video_backend = "wan2.2-vae"
    input_mode = "single_image"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_frames: int = 9,
        image_height: int = 256,
        image_width: int = 256,
        latent_patch_size: int | Sequence[int] = (1, 2, 2),
        num_human_gaussians: int = 850,
        num_object_gaussians: int = 850,
        num_joints: int = 22,
        contact_dim: int = 4,
        human_shape_dim: int = 10,
        human_pose_dim: int = 72,
        wan_vae_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        wan_vae_subfolder: str = "vae",
        wan_vae_dtype: str = "bf16",
        wan_vae_local_files_only: bool = True,
        vae_latent_channels: int = 48,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 16,
        load_vae_on_init: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        self.vae_encoder = FrozenWanVAEEncoder(
            model_id=wan_vae_model_id,
            subfolder=wan_vae_subfolder,
            torch_dtype=wan_vae_dtype,
            local_files_only=wan_vae_local_files_only,
            load_on_init=load_vae_on_init,
        )
        self.visual_embed = WanVAELatentPatchEmbed(
            latent_channels=vae_latent_channels,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=latent_patch_size,
        )
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

        self.visual_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hoi_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hoi_query_tokens = nn.Parameter(torch.zeros(1, self.state_codec.total_tokens, hidden_dim))
        nn.init.normal_(self.visual_type_embed, std=0.02)
        nn.init.normal_(self.hoi_type_embed, std=0.02)
        nn.init.normal_(self.hoi_query_tokens, std=0.02)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.state_velocity_head = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.state_velocity_head.weight)
        nn.init.zeros_(self.state_velocity_head.bias)

    def ensure_vae_loaded(self, device: torch.device) -> None:
        self.vae_encoder.ensure_loaded(device)

    def encode_state_target(self, **kwargs) -> Tensor:
        return self.state_codec.encode_targets(**kwargs)

    def decode_state_tokens(self, state_tokens: Tensor) -> DecodedHOIState:
        return self.state_codec.decode_tokens(state_tokens)

    @torch.no_grad()
    def encode_video_latents(self, rgb: Tensor) -> Tensor:
        return self.vae_encoder(rgb)

    def encode_visual_tokens_from_latents(self, vae_latents: Tensor) -> Tensor:
        return self.visual_embed(vae_latents) + self.visual_type_embed.to(
            device=vae_latents.device,
            dtype=vae_latents.dtype,
        )

    def _default_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, device=device, dtype=torch.float32)

    def forward_from_latents(
        self,
        *,
        vae_latents: Tensor,
        timesteps: Optional[Tensor] = None,
        state_xt: Optional[Tensor] = None,
    ) -> UniModelOutput:
        visual_tokens = self.encode_visual_tokens_from_latents(vae_latents)
        batch_size = visual_tokens.shape[0]
        if timesteps is None:
            timesteps = self._default_timesteps(batch_size, visual_tokens.device)
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}.")

        if state_xt is None:
            hoi_tokens = self.hoi_query_tokens.expand(batch_size, -1, -1)
        else:
            expected = (batch_size, self.state_codec.total_tokens, self.hidden_dim)
            if tuple(state_xt.shape) != expected:
                raise ValueError(f"`state_xt` must have shape {expected}, got {tuple(state_xt.shape)}.")
            hoi_tokens = state_xt
        hoi_tokens = hoi_tokens.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        hoi_tokens = hoi_tokens + self.hoi_type_embed.to(device=visual_tokens.device, dtype=visual_tokens.dtype)

        cond = self.time_embed(timestep_embedding(timesteps.to(visual_tokens.device), self.hidden_dim)).to(
            dtype=visual_tokens.dtype
        )
        tokens = torch.cat([visual_tokens, hoi_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.final_norm(tokens)

        visual_len = visual_tokens.shape[1]
        visual_tokens_out = tokens[:, :visual_len]
        hoi_tokens_out = tokens[:, visual_len:]

        state_velocity: Optional[Tensor] = None
        predicted_clean: Optional[Tensor] = None
        decode_tokens = hoi_tokens_out
        if state_xt is not None:
            state_velocity = self.state_velocity_head(hoi_tokens_out)
            t_view = timesteps.to(device=state_xt.device, dtype=state_xt.dtype).view(batch_size, 1, 1)
            predicted_clean = state_xt + (1.0 - t_view) * state_velocity.to(dtype=state_xt.dtype)
            decode_tokens = predicted_clean

        return UniModelOutput(
            decoded_state=self.decode_state_tokens(decode_tokens),
            hoi_tokens=hoi_tokens_out,
            visual_tokens=visual_tokens_out,
            all_tokens=tokens,
            state_velocity=state_velocity,
            predicted_clean_state_tokens=predicted_clean,
            vae_latents=vae_latents,
        )

    def forward(
        self,
        *,
        rgb: Tensor,
        timesteps: Optional[Tensor] = None,
        state_xt: Optional[Tensor] = None,
    ) -> UniModelOutput:
        vae_latents = self.encode_video_latents(rgb)
        return self.forward_from_latents(
            vae_latents=vae_latents.to(device=rgb.device),
            timesteps=timesteps,
            state_xt=state_xt,
        )


__all__ = [
    "DiTBlock",
    "FrozenWanVAEEncoder",
    "UniModel",
    "UniModelOutput",
    "WanVAELatentPatchEmbed",
]
