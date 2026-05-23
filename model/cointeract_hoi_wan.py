from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}.")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


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


def _retrieve_video_sample(decoder_output) -> Tensor:
    if hasattr(decoder_output, "sample"):
        return decoder_output.sample
    if isinstance(decoder_output, (tuple, list)) and decoder_output:
        return decoder_output[0]
    if isinstance(decoder_output, Tensor):
        return decoder_output
    raise AttributeError("Could not retrieve video sample from Wan VAE output.")


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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` must be divisible by `num_heads`, got {dim}/{num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // int(num_heads)
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        if context is None:
            context = query
        batch_size, query_len, dim = query.shape
        context_len = context.shape[1]

        q = self.q(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=query.device)
            if attn_mask.dtype == torch.bool:
                additive_mask = torch.zeros(attn_mask.shape, device=query.device, dtype=q.dtype)
                additive_mask = additive_mask.masked_fill(~attn_mask, -torch.finfo(q.dtype).max)
            else:
                additive_mask = attn_mask.to(dtype=q.dtype)
            if additive_mask.ndim == 2:
                additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
            elif additive_mask.ndim == 3:
                additive_mask = additive_mask.unsqueeze(1)
            elif additive_mask.ndim != 4:
                raise ValueError(f"`attn_mask` must have 2, 3, or 4 dims, got {additive_mask.ndim}.")
        else:
            additive_mask = None
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_len, dim)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ZeroInitCrossAdapter(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        return self.out(self.attn(self.query_norm(query), self.context_norm(context)))


class FirstFramePatchEncoder(nn.Module):
    def __init__(self, *, in_channels: int, hidden_dim: int, patch_size: int, image_height: int, image_width: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.token_h = int(image_height) // self.patch_size
        self.token_w = int(image_width) // self.patch_size
        if int(image_height) % self.patch_size or int(image_width) % self.patch_size:
            raise ValueError("Image size must be divisible by patch_size.")
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos = nn.Parameter(torch.zeros(self.token_h * self.token_w, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"`image` must have shape [B, C, H, W], got {tuple(image.shape)}.")
        tokens = self.proj(image).flatten(2).transpose(1, 2)
        return tokens + self.pos.unsqueeze(0).to(device=tokens.device, dtype=tokens.dtype)


@dataclass
class DecodedHOIState:
    human_shape: Tensor
    human_pose: Tensor
    human_translation: Tensor
    human_gaussians: Tensor
    object_gaussians: Tensor
    joints_3d: Tensor
    object_transforms: Tensor
    contact_signature: Tensor


class HOIStateCodec(nn.Module):
    """Explicit HOI tokenization used by the HOI-primary denoising stream."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_human_gaussians: int,
        num_object_gaussians: int,
        num_frames: int,
        num_joints: int,
        contact_dim: int = 4,
        human_shape_dim: int = 10,
        human_pose_dim: int = 72,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_human_gaussians = int(num_human_gaussians)
        self.num_object_gaussians = int(num_object_gaussians)
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.contact_dim = int(contact_dim)
        self.human_shape_dim = int(human_shape_dim)
        self.human_pose_dim = int(human_pose_dim)

        self.num_context_tokens = 1
        self.num_shape_tokens = 1
        self.num_pose_tokens = self.num_frames
        self.num_translation_tokens = self.num_frames
        self.num_object_motion_tokens = self.num_frames
        self.num_contact_tokens = self.num_frames
        self.num_joint_tokens = self.num_frames * self.num_joints

        self.context_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.frame_embedding = nn.Parameter(torch.zeros(self.num_frames, hidden_dim))
        self.type_embedding = nn.Parameter(torch.zeros(8, hidden_dim))
        self.human_gaussian_pos = nn.Parameter(torch.zeros(self.num_human_gaussians, hidden_dim))
        self.object_gaussian_pos = nn.Parameter(torch.zeros(self.num_object_gaussians, hidden_dim))
        self.joint_pos = nn.Parameter(torch.zeros(self.num_joints, hidden_dim))

        self.shape_in = nn.Linear(self.human_shape_dim, hidden_dim)
        self.pose_in = nn.Linear(self.human_pose_dim, hidden_dim)
        self.translation_in = nn.Linear(3, hidden_dim)
        self.object_motion_in = nn.Linear(9, hidden_dim)
        self.contact_in = nn.Linear(self.contact_dim, hidden_dim)
        self.gaussian_in = nn.Linear(14, hidden_dim)
        self.joint_in = nn.Linear(3, hidden_dim)

        self.shape_out = nn.Linear(hidden_dim, self.human_shape_dim)
        self.pose_out = nn.Linear(hidden_dim, self.human_pose_dim)
        self.translation_out = nn.Linear(hidden_dim, 3)
        self.object_motion_out = nn.Linear(hidden_dim, 9)
        self.contact_out = nn.Linear(hidden_dim, self.contact_dim)
        self.human_gaussian_out = nn.Linear(hidden_dim, 14)
        self.object_gaussian_out = nn.Linear(hidden_dim, 14)
        self.joint_out = nn.Linear(hidden_dim, 3)

        for param in (
            self.context_token,
            self.frame_embedding,
            self.type_embedding,
            self.human_gaussian_pos,
            self.object_gaussian_pos,
            self.joint_pos,
        ):
            nn.init.normal_(param, std=0.02)
        self._freeze_target_encoder()

    def _freeze_target_encoder(self) -> None:
        encode_modules = (
            self.shape_in,
            self.pose_in,
            self.translation_in,
            self.object_motion_in,
            self.contact_in,
            self.gaussian_in,
            self.joint_in,
        )
        for module in encode_modules:
            module.requires_grad_(False)
        for param in (
            self.context_token,
            self.frame_embedding,
            self.type_embedding,
            self.human_gaussian_pos,
            self.object_gaussian_pos,
            self.joint_pos,
        ):
            param.requires_grad_(False)

    @property
    def total_tokens(self) -> int:
        return (
            self.num_context_tokens
            + self.num_shape_tokens
            + self.num_pose_tokens
            + self.num_translation_tokens
            + self.num_object_motion_tokens
            + self.num_contact_tokens
            + self.num_human_gaussians
            + self.num_object_gaussians
            + self.num_joint_tokens
        )

    def _slices(self) -> Dict[str, slice]:
        offset = 0
        result = {}
        for name, count in (
            ("context", self.num_context_tokens),
            ("shape", self.num_shape_tokens),
            ("pose", self.num_pose_tokens),
            ("translation", self.num_translation_tokens),
            ("object_motion", self.num_object_motion_tokens),
            ("contact", self.num_contact_tokens),
            ("human_gaussians", self.num_human_gaussians),
            ("object_gaussians", self.num_object_gaussians),
            ("joints", self.num_joint_tokens),
        ):
            result[name] = slice(offset, offset + count)
            offset += count
        return result

    def encode_targets(
        self,
        *,
        human_shape: Tensor,
        human_pose: Tensor,
        human_translation: Tensor,
        object_transforms: Tensor,
        contact_signature: Tensor,
        human_gaussians: Tensor,
        object_gaussians: Tensor,
        joints_3d: Tensor,
    ) -> Tensor:
        batch_size = human_shape.shape[0]
        if human_pose.shape[1:] != (self.num_frames, self.human_pose_dim):
            raise ValueError(f"Expected human_pose [B, {self.num_frames}, {self.human_pose_dim}].")
        if joints_3d.shape[1:] != (self.num_frames, self.num_joints, 3):
            raise ValueError(f"Expected joints_3d [B, {self.num_frames}, {self.num_joints}, 3].")

        context = self.context_token.unsqueeze(0).expand(batch_size, -1, -1) + self.type_embedding[0]
        shape = self.shape_in(human_shape).unsqueeze(1) + self.type_embedding[1]
        pose = self.pose_in(human_pose) + self.frame_embedding.unsqueeze(0) + self.type_embedding[2]
        translation = self.translation_in(human_translation) + self.frame_embedding.unsqueeze(0) + self.type_embedding[3]
        object_motion = (
            self.object_motion_in(_flatten_object_transforms(object_transforms))
            + self.frame_embedding.unsqueeze(0)
            + self.type_embedding[4]
        )
        contact = self.contact_in(contact_signature) + self.frame_embedding.unsqueeze(0) + self.type_embedding[5]
        human_g = self.gaussian_in(human_gaussians) + self.human_gaussian_pos.unsqueeze(0) + self.type_embedding[6]
        object_g = self.gaussian_in(object_gaussians) + self.object_gaussian_pos.unsqueeze(0) + self.type_embedding[7]
        joints = (
            self.joint_in(joints_3d)
            + self.frame_embedding.view(1, self.num_frames, 1, self.hidden_dim)
            + self.joint_pos.view(1, 1, self.num_joints, self.hidden_dim)
            + self.type_embedding[5]
        ).reshape(batch_size, self.num_joint_tokens, self.hidden_dim)
        return torch.cat([context, shape, pose, translation, object_motion, contact, human_g, object_g, joints], dim=1)

    def decode_tokens(self, tokens: Tensor) -> DecodedHOIState:
        parts = self._slices()
        shape = tokens[:, parts["shape"]].squeeze(1)
        pose = tokens[:, parts["pose"]]
        translation = tokens[:, parts["translation"]]
        object_motion = tokens[:, parts["object_motion"]]
        contact = tokens[:, parts["contact"]]
        human_g = tokens[:, parts["human_gaussians"]]
        object_g = tokens[:, parts["object_gaussians"]]
        joints = tokens[:, parts["joints"]].reshape(tokens.shape[0], self.num_frames, self.num_joints, self.hidden_dim)

        return DecodedHOIState(
            human_shape=self.shape_out(shape),
            human_pose=self.pose_out(pose),
            human_translation=self.translation_out(translation),
            human_gaussians=_apply_gaussian_activation(self.human_gaussian_out(human_g)),
            object_gaussians=_apply_gaussian_activation(self.object_gaussian_out(object_g)),
            joints_3d=self.joint_out(joints),
            object_transforms=_unflatten_object_transforms(self.object_motion_out(object_motion)),
            contact_signature=self.contact_out(contact),
        )


@dataclass
class WanRGBStreamOutput:
    velocity: Tensor
    hidden_tokens: Tensor


class FrozenWanTI2VImageStream(nn.Module):
    """Wan2.2-TI2V RGB stream with empty text and first-frame-only image input."""

    def __init__(
        self,
        *,
        model_id: str,
        num_frames: int,
        image_height: int,
        image_width: int,
        torch_dtype: str = "bf16",
        prompt_max_sequence_length: int = 512,
        local_files_only: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.torch_dtype_name = str(torch_dtype)
        self.prompt_max_sequence_length = int(prompt_max_sequence_length)
        self.local_files_only = bool(local_files_only)
        self.freeze = bool(freeze)
        self._loaded = False
        self._empty_prompt_cache: Optional[Tensor] = None

    @property
    def loaded(self) -> bool:
        return bool(self._loaded)

    def _load(self, device: torch.device) -> None:
        if self._loaded:
            self.to(device=device)
            return
        from diffusers import WanPipeline

        dtype = _resolve_torch_dtype(self.torch_dtype_name)
        pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer

        self.latent_channels = int(self.vae.config.z_dim)
        self.latent_height = self.image_height // int(self.vae.config.scale_factor_spatial)
        self.latent_width = self.image_width // int(self.vae.config.scale_factor_spatial)
        self.num_latent_frames = (self.num_frames - 1) // int(self.vae.config.scale_factor_temporal) + 1
        self.token_dim = int(
            self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        )
        self.patch_size_t, self.patch_size_h, self.patch_size_w = tuple(
            int(v) for v in self.transformer.config.patch_size
        )
        if (self.num_frames - 1) % int(self.vae.config.scale_factor_temporal) != 0:
            raise ValueError(f"Wan TI2V requires num_frames = 4k + 1, got {self.num_frames}.")
        if int(self.transformer.config.in_channels) != self.latent_channels:
            raise RuntimeError(
                "This wrapper expects Wan2.2-TI2V Diffusers latent channels to match transformer input channels. "
                f"Got vae.z_dim={self.latent_channels}, transformer.in_channels={self.transformer.config.in_channels}."
            )

        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, self.latent_channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std_inv",
            (1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32)).view(
                1, self.latent_channels, 1, 1, 1
            ),
            persistent=False,
        )
        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()
        self.transformer.requires_grad_(not self.freeze)
        if self.freeze:
            self.transformer.eval()
        self._loaded = True
        del pipe
        self.to(device=device)

    def train(self, mode: bool = True) -> "FrozenWanTI2VImageStream":
        super().train(mode)
        if self._loaded and self.freeze:
            self.vae.eval()
            self.transformer.eval()
            self.text_encoder.eval()
        return self

    def ensure_loaded(self, device: torch.device) -> None:
        self._load(device)

    def _normalize_latents(self, latents: Tensor) -> Tensor:
        return (latents - self.latents_mean.to(latents.device, latents.dtype)) * self.latents_std_inv.to(
            latents.device, latents.dtype
        )

    def _denormalize_latents(self, latents: Tensor) -> Tensor:
        return latents / self.latents_std_inv.to(latents.device, latents.dtype) + self.latents_mean.to(
            latents.device, latents.dtype
        )

    @torch.no_grad()
    def decode_video(self, latents: Tensor) -> Tensor:
        self.ensure_loaded(latents.device)
        if latents.ndim != 5:
            raise ValueError(f"Wan latents must have shape [B, C, T, H, W], got {tuple(latents.shape)}.")
        latents = self._denormalize_latents(latents.to(device=self.latents_mean.device, dtype=self.vae.dtype))
        video = _retrieve_video_sample(self.vae.decode(latents))
        video = video.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def encode_video(self, video: Tensor) -> Tensor:
        self.ensure_loaded(video.device)
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Wan expects video [B, T, 3, H, W], got {tuple(video.shape)}.")
        if video.shape[1] not in {1, self.num_frames}:
            raise ValueError(f"Wan expects either 1 image frame or {self.num_frames} video frames, got {video.shape[1]}.")
        video = video.mul(2.0).sub(1.0).permute(0, 2, 1, 3, 4)
        video = video.to(device=self.latents_mean.device, dtype=self.vae.dtype)
        latents = _retrieve_latents(self.vae.encode(video), sample_mode="argmax")
        return self._normalize_latents(latents).to(dtype=self.transformer.dtype)

    @torch.no_grad()
    def encode_first_frame(self, first_frame: Tensor) -> Tensor:
        self.ensure_loaded(first_frame.device)
        if first_frame.ndim != 4 or first_frame.shape[1] != 3:
            raise ValueError(f"`first_frame` must have shape [B, 3, H, W], got {tuple(first_frame.shape)}.")
        video = first_frame.unsqueeze(1)
        latents = self.encode_video(video)
        if latents.shape[2] != 1:
            raise RuntimeError(f"Expected one first-frame latent, got {latents.shape[2]}.")
        return latents

    def build_noisy_latents(
        self,
        target_latents: Tensor,
        first_frame_latents: Tensor,
        timesteps: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        noise = torch.randn(
            target_latents.shape,
            generator=generator,
            device=target_latents.device,
            dtype=target_latents.dtype,
        )
        noise[:, :, :1] = first_frame_latents.to(device=target_latents.device, dtype=target_latents.dtype)
        target = target_latents.clone()
        target[:, :, :1] = first_frame_latents.to(device=target.device, dtype=target.dtype)
        t_view = timesteps.to(device=target.device, dtype=target.dtype).view(target.shape[0], 1, 1, 1, 1)
        xt = t_view * target + (1.0 - t_view) * noise
        xt[:, :, :1] = target[:, :, :1]
        velocity = target - noise
        velocity[:, :, :1] = 0.0
        return xt, velocity

    def _encode_empty_prompt(self, *, batch_size: int, device: torch.device) -> Tensor:
        self.ensure_loaded(device)
        if self._empty_prompt_cache is None:
            text_inputs = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.prompt_max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state.to(
                device=device,
                dtype=self.transformer.dtype,
            )
            self._empty_prompt_cache = prompt_embeds.detach().cpu()
        return self._empty_prompt_cache.to(device=device, dtype=self.transformer.dtype).expand(batch_size, -1, -1)

    def _build_token_timesteps(self, timesteps: Tensor, latents: Tensor) -> Tensor:
        scalar = (1.0 - timesteps.float()).clamp(0.0, 1.0) * float(self.scheduler.config.num_train_timesteps)
        tokens_per_frame = (latents.shape[3] // self.patch_size_h) * (latents.shape[4] // self.patch_size_w)
        mask = torch.ones(latents.shape[0], latents.shape[2], tokens_per_frame, device=latents.device)
        mask[:, 0] = 0.0
        return (mask.reshape(latents.shape[0], -1) * scalar.unsqueeze(1)).to(dtype=self.transformer.dtype)

    def _forward_transformer_with_hidden(
        self,
        *,
        hidden_states: Tensor,
        token_timesteps: Tensor,
        prompt_embeds: Tensor,
    ) -> WanRGBStreamOutput:
        transformer = self.transformer
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = transformer.config.patch_size
        post_f = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w

        rotary_emb = transformer.rope(hidden_states)
        tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        flat_timestep = token_timesteps.flatten()
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            flat_timestep,
            prompt_embeds,
            None,
            timestep_seq_len=token_timesteps.shape[1],
        )
        if encoder_hidden_states_image is not None:
            raise RuntimeError("Wan2.2-TI2V-5B should not require image encoder hidden states.")
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in transformer.blocks:
            tokens = block(tokens, encoder_hidden_states, timestep_proj, rotary_emb)

        hidden_tokens = tokens
        shift, scale = (transformer.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_tokens.device)
        scale = scale.squeeze(2).to(hidden_tokens.device)
        tokens = (transformer.norm_out(hidden_tokens.float()) * (1 + scale) + shift).type_as(hidden_tokens)
        tokens = transformer.proj_out(tokens)
        tokens = tokens.reshape(batch_size, post_f, post_h, post_w, p_t, p_h, p_w, -1)
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        velocity = tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return WanRGBStreamOutput(velocity=velocity, hidden_tokens=hidden_tokens)

    def forward(self, *, video_xt: Tensor, timesteps: Tensor) -> WanRGBStreamOutput:
        self.ensure_loaded(video_xt.device)
        if self.freeze:
            self.vae.eval()
            self.transformer.eval()
            self.text_encoder.eval()
        prompt_embeds = self._encode_empty_prompt(batch_size=video_xt.shape[0], device=video_xt.device)
        token_timesteps = self._build_token_timesteps(timesteps, video_xt)
        return self._forward_transformer_with_hidden(
            hidden_states=video_xt.to(dtype=self.transformer.dtype),
            token_timesteps=token_timesteps,
            prompt_embeds=prompt_embeds,
        )


class CoInteractHOIBlock(nn.Module):
    """HOI-primary dual-stream block.

    The HOI stream is the denoising stream that produces the final state. RGB tokens
    stay auxiliary: they get a lightweight DiT-style update and guide HOI through
    zero-init cross adapters, while the optional reverse adapter is off by default.
    """

    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.hoi_block = TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.rgb_block = TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.rgb_to_hoi = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout)
        self.image_to_hoi = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout)
        self.hoi_to_rgb = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout)
        self.hoi_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.rgb_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(
        self,
        hoi_tokens: Tensor,
        *,
        rgb_tokens: Tensor,
        image_tokens: Tensor,
        rgb_to_hoi_scale: float,
        hoi_to_rgb_scale: float,
    ) -> Tuple[Tensor, Tensor]:
        hoi_tokens = self.hoi_block(hoi_tokens)
        rgb_tokens = self.rgb_block(rgb_tokens)
        hoi_tokens = hoi_tokens + float(rgb_to_hoi_scale) * self.hoi_gate(hoi_tokens) * self.rgb_to_hoi(
            hoi_tokens,
            rgb_tokens,
        )
        hoi_tokens = hoi_tokens + self.image_to_hoi(hoi_tokens, image_tokens)
        if float(hoi_to_rgb_scale) != 0.0:
            rgb_tokens = rgb_tokens + float(hoi_to_rgb_scale) * self.rgb_gate(rgb_tokens) * self.hoi_to_rgb(
                rgb_tokens,
                hoi_tokens,
            )
        return hoi_tokens, rgb_tokens


@dataclass
class CoInteractHOI4DOutput:
    rgb_velocity: Tensor
    state_velocity: Tensor
    decoded_state: DecodedHOIState
    rgb_hidden_tokens: Tensor
    rgb_context_tokens: Tensor


class CoInteractHOI4DModel(nn.Module):
    video_backend = "wan2.2-ti2v-5b"
    input_mode = "rgb_video_guided_hoi"

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
        self.blocks = nn.ModuleList(
            [
                CoInteractHOIBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        if not self.enable_hoi_to_rgb:
            for block in self.blocks:
                block.hoi_to_rgb.requires_grad_(False)
                block.rgb_gate.requires_grad_(False)
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
    ) -> CoInteractHOI4DOutput:
        if state_xt.ndim != 3 or state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(f"`state_xt` must have shape [B, L, {self.hidden_dim}], got {tuple(state_xt.shape)}.")
        rgb_output = self.rgb_stream(video_xt=video_xt, timesteps=timesteps)
        rgb_hidden_tokens = rgb_output.hidden_tokens.detach() if self.detach_rgb_context else rgb_output.hidden_tokens
        rgb_tokens = self.rgb_token_adapter(rgb_hidden_tokens.to(dtype=state_xt.dtype))
        image_tokens = self.first_frame_encoder(first_frame.to(dtype=state_xt.dtype))
        time_cond = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1).to(dtype=state_xt.dtype)
        state_tokens = state_xt + time_cond
        rgb_tokens = rgb_tokens + time_cond
        image_tokens = image_tokens + time_cond
        for block in self.blocks:
            state_tokens, rgb_tokens = block(
                state_tokens,
                rgb_tokens=rgb_tokens,
                image_tokens=image_tokens,
                rgb_to_hoi_scale=rgb_to_hoi_scale,
                hoi_to_rgb_scale=hoi_to_rgb_scale if self.enable_hoi_to_rgb else 0.0,
            )
        state_velocity = self.state_velocity_head(self.state_norm(state_tokens))
        t_view = timesteps.to(device=state_xt.device, dtype=state_xt.dtype).view(state_xt.shape[0], 1, 1)
        predicted_clean_state = state_xt + (1.0 - t_view) * state_velocity.to(dtype=state_xt.dtype)
        return CoInteractHOI4DOutput(
            rgb_velocity=rgb_output.velocity,
            state_velocity=state_velocity,
            decoded_state=self.decode_state_tokens(predicted_clean_state),
            rgb_hidden_tokens=rgb_output.hidden_tokens,
            rgb_context_tokens=rgb_tokens,
        )


__all__ = [
    "CoInteractHOI4DModel",
    "CoInteractHOI4DOutput",
    "DecodedHOIState",
    "FrozenWanTI2VImageStream",
    "HOIStateCodec",
    "timestep_embedding",
]
