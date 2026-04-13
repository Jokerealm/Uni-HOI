from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _normalize_object_category(text: str) -> str:
    tokens = [token.strip().lower() for token in str(text).replace("-", "_").split("_") if token.strip()]
    filtered = [
        token
        for token in tokens
        if not token.startswith("date") and not token.startswith("sub") and not token.startswith("synzv")
    ]
    if not filtered:
        return "object"
    return filtered[0]


def _build_object_condition_prompt(object_category: str, *, branch: str = "joint") -> str:
    category = _normalize_object_category(object_category)
    if branch == "human":
        if category == "object":
            return "a clean amodal video of only the person interacting with an object"
        return f"a clean amodal video of only the person interacting with a {category}"
    if branch == "object":
        if category == "object":
            return "a clean amodal video of only the object being handled by a person"
        return f"a clean amodal video of only the {category} being handled by a person"
    if branch == "joint":
        if category == "object":
            return "a person interacting with an object"
        return f"a person interacting with a {category}"
    raise ValueError(f"Unsupported Wan prompt branch {branch!r}.")


def _resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported Wan dtype {name!r}. Expected one of bf16/fp16/fp32.")


def _retrieve_latents(encoder_output, sample_mode: str = "argmax") -> Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from Wan VAE output.")


@dataclass
class WanTeacherOutput:
    velocity: Tensor
    hidden_tokens: Tensor
    human_velocity: Optional[Tensor] = None
    object_velocity: Optional[Tensor] = None
    human_hidden_tokens: Optional[Tensor] = None
    object_hidden_tokens: Optional[Tensor] = None


class FrozenWanVideoTeacher(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        num_frames: int,
        image_height: int,
        image_width: int,
        torch_dtype: str = "bf16",
        prompt_max_sequence_length: int = 512,
        prompt_override: str = "",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()

        from diffusers import WanPipeline

        dtype = _resolve_torch_dtype(torch_dtype)
        pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=bool(local_files_only),
        )

        self.model_id = str(model_id)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.prompt_max_sequence_length = int(prompt_max_sequence_length)
        self.prompt_override = str(prompt_override).strip()
        self.local_files_only = bool(local_files_only)

        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer

        self.latent_channels = int(self.vae.config.z_dim)
        self.latent_height = self.image_height // int(self.vae.config.scale_factor_spatial)
        self.latent_width = self.image_width // int(self.vae.config.scale_factor_spatial)
        self.num_latent_frames = (self.num_frames - 1) // int(self.vae.config.scale_factor_temporal) + 1
        self.token_dim = int(self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim)
        self.patch_size_t, self.patch_size_h, self.patch_size_w = tuple(int(v) for v in self.transformer.config.patch_size)

        if self.transformer.config.image_dim is not None:
            raise RuntimeError(
                f"{self.model_id} exposes image encoder inputs, but TI2V-5B should not require them. "
                "Use the text+image-to-video Wan pipeline without CLIP image embeddings."
            )
        if not bool(getattr(pipe.config, "expand_timesteps", False)):
            raise RuntimeError(
                f"{self.model_id} does not enable `expand_timesteps`; this wrapper expects Wan2.2 TI2V behavior."
            )
        if (self.num_frames - 1) % int(self.vae.config.scale_factor_temporal) != 0:
            raise ValueError(
                f"`num_frames` must satisfy num_frames = 4k + 1 for {self.model_id}. Got {self.num_frames}."
            )
        if self.image_height % 16 != 0 or self.image_width % 16 != 0:
            raise ValueError(
                f"Wan TI2V expects height/width divisible by 16, got {(self.image_height, self.image_width)}."
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

        self._prompt_cache: Dict[str, Tensor] = {}

        self.vae.requires_grad_(False).eval()
        self.transformer.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()

        del pipe

    def train(self, mode: bool = True) -> "FrozenWanVideoTeacher":
        super().train(mode)
        self.vae.eval()
        self.transformer.eval()
        self.text_encoder.eval()
        return self

    def sample_prior_latents(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.randn(
            batch_size,
            self.latent_channels,
            self.num_latent_frames,
            self.latent_height,
            self.latent_width,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _normalize_latents(self, latents: Tensor) -> Tensor:
        return (latents - self.latents_mean.to(latents.device, latents.dtype)) * self.latents_std_inv.to(
            latents.device, latents.dtype
        )

    def _denormalize_latents(self, latents: Tensor) -> Tensor:
        latents_std = 1.0 / self.latents_std_inv.to(latents.device, latents.dtype)
        return latents / self.latents_std_inv.to(latents.device, latents.dtype) + self.latents_mean.to(
            latents.device, latents.dtype
        )

    @torch.no_grad()
    def encode_video(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Wan expects RGB video with shape [B, T, 3, H, W], got {tuple(video.shape)}.")
        if video.shape[1] != self.num_frames:
            raise ValueError(f"Wan expects {self.num_frames} frames, got {video.shape[1]}.")
        if video.shape[-2:] != (self.image_height, self.image_width):
            raise ValueError(
                f"Wan expects spatial size {(self.image_height, self.image_width)}, got {tuple(video.shape[-2:])}."
            )
        video_5d = video.permute(0, 2, 1, 3, 4).to(dtype=self.vae.dtype, device=self.latents_mean.device)
        latents = _retrieve_latents(self.vae.encode(video_5d), sample_mode="argmax")
        return self._normalize_latents(latents).to(dtype=self.transformer.dtype)

    @torch.no_grad()
    def decode_video(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Wan latents must have shape [B, {self.latent_channels}, T', H', W'], got {tuple(latents.shape)}."
            )
        latents = self._denormalize_latents(latents.to(dtype=self.vae.dtype, device=self.latents_mean.device))
        video = self.vae.decode(latents, return_dict=False)[0]
        return video.permute(0, 2, 1, 3, 4).to(dtype=latents.dtype)

    def _extract_rgb_and_masks(self, condition_video: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if condition_video.ndim != 5 or condition_video.shape[2] < 5:
            raise ValueError(
                f"`condition_video` must have shape [B, T, C>=5, H, W] for Wan conditioning, got {tuple(condition_video.shape)}."
            )
        rgb = condition_video[:, :, :3].contiguous()
        masks_human = condition_video[:, :, 3:4].clamp(0.0, 1.0)
        masks_object = condition_video[:, :, 4:5].clamp(0.0, 1.0)
        return rgb, masks_human, masks_object

    @torch.no_grad()
    def _encode_condition_latents(self, rgb_video: Tensor, condition_latents: Optional[Tensor] = None) -> Tensor:
        if condition_latents is not None:
            expected_shape_suffix = (self.num_latent_frames, self.latent_height, self.latent_width)
            if (
                condition_latents.ndim != 5
                or condition_latents.shape[1] != self.latent_channels
                or tuple(condition_latents.shape[-3:]) != expected_shape_suffix
            ):
                raise ValueError(
                    f"`condition_latents` must have shape [B, {self.latent_channels}, "
                    f"{self.num_latent_frames}, {self.latent_height}, {self.latent_width}], "
                    f"got {tuple(condition_latents.shape)}."
                )
            return condition_latents.to(device=rgb_video.device, dtype=self.transformer.dtype)
        return self.encode_video(rgb_video).to(device=rgb_video.device, dtype=self.transformer.dtype)

    def _downsample_mask_video(self, mask_video: Tensor) -> Tensor:
        if mask_video.ndim != 5 or mask_video.shape[2] != 1:
            raise ValueError(f"`mask_video` must have shape [B, T, 1, H, W], got {tuple(mask_video.shape)}.")
        batch_size, num_frames, _, height, width = mask_video.shape
        spatial_mask = F.interpolate(
            mask_video.reshape(batch_size * num_frames, 1, height, width),
            size=(self.latent_height, self.latent_width),
            mode="nearest",
        ).view(batch_size, num_frames, 1, self.latent_height, self.latent_width)
        spatial_mask = spatial_mask.permute(0, 2, 1, 3, 4).contiguous()
        if spatial_mask.shape[2] == self.num_latent_frames:
            return spatial_mask.to(dtype=self.transformer.dtype)

        temporal_scale = int(self.vae.config.scale_factor_temporal)
        pooled_frames = []
        for latent_index in range(self.num_latent_frames):
            start = min(latent_index * temporal_scale, spatial_mask.shape[2] - 1)
            end = min(spatial_mask.shape[2], start + temporal_scale + 1)
            pooled_frames.append(spatial_mask[:, :, start:end].amax(dim=2, keepdim=True))
        latent_mask = torch.cat(pooled_frames, dim=2)
        return latent_mask.clamp(0.0, 1.0).to(dtype=self.transformer.dtype)

    @torch.no_grad()
    def _encode_condition(
        self,
        condition_video: Tensor,
        *,
        condition_latents: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        rgb, masks_human, masks_object = self._extract_rgb_and_masks(condition_video)
        latent_condition = self._encode_condition_latents(rgb, condition_latents=condition_latents)
        human_inpaint_mask = self._downsample_mask_video(masks_object).to(device=condition_video.device)
        object_inpaint_mask = self._downsample_mask_video(masks_human).to(device=condition_video.device)
        return latent_condition.to(device=condition_video.device), human_inpaint_mask, object_inpaint_mask

    def _encode_prompts(
        self,
        *,
        sequence_names: Sequence[str],
        object_categories: Optional[Sequence[str]],
        device: torch.device,
        branch: str = "joint",
    ) -> Tensor:
        prompts = []
        missing = []
        for index, name in enumerate(sequence_names):
            category = None
            if object_categories is not None and index < len(object_categories):
                category = object_categories[index]
            prompt = self.prompt_override or _build_object_condition_prompt(category or name, branch=branch)
            prompts.append(prompt)
            if prompt not in self._prompt_cache:
                missing.append(prompt)

        if missing:
            text_inputs = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.prompt_max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            seq_lens = attention_mask.gt(0).sum(dim=1).long()
            prompt_embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state.to(
                device=device,
                dtype=self.transformer.dtype,
            )
            padded = []
            for embed, seq_len in zip(prompt_embeds, seq_lens):
                active = embed[:seq_len]
                pad_len = self.prompt_max_sequence_length - active.shape[0]
                if pad_len > 0:
                    active = torch.cat([active, active.new_zeros(pad_len, active.shape[1])], dim=0)
                padded.append(active)
            for prompt, embed in zip(missing, padded):
                self._prompt_cache[prompt] = embed.detach().cpu()

        stacked = [self._prompt_cache[prompt].to(device=device, dtype=self.transformer.dtype) for prompt in prompts]
        return torch.stack(stacked, dim=0)

    def _build_timestep_input(self, timesteps: Tensor, inpaint_mask: Tensor) -> Tensor:
        sigma = (1.0 - timesteps.float()).clamp(min=0.0, max=1.0)
        scalar_timestep = sigma * float(self.scheduler.config.num_train_timesteps)
        token_mask = inpaint_mask[:, 0, :: self.patch_size_t, :: self.patch_size_h, :: self.patch_size_w]
        token_mask = token_mask.reshape(inpaint_mask.shape[0], -1)
        return token_mask * scalar_timestep.unsqueeze(1)

    def _forward_single_branch(
        self,
        *,
        video_xt: Tensor,
        timesteps: Tensor,
        condition_latents: Tensor,
        inpaint_mask: Tensor,
        prompt_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        timestep_input = self._build_timestep_input(timesteps.to(video_xt.device), inpaint_mask)
        latent_model_input = (1.0 - inpaint_mask) * condition_latents + inpaint_mask * video_xt.to(
            dtype=self.transformer.dtype
        )
        return self._forward_transformer_with_hidden(
            hidden_states=latent_model_input,
            timestep=timestep_input.to(device=video_xt.device, dtype=self.transformer.dtype),
            prompt_embeds=prompt_embeds,
        )

    def _forward_transformer_with_hidden(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        transformer = self.transformer

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = transformer.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = transformer.rope(hidden_states)

        tokens = transformer.patch_embedding(hidden_states)
        tokens = tokens.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            flat_timestep = timestep.flatten()
        else:
            ts_seq_len = None
            flat_timestep = timestep

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
            flat_timestep,
            prompt_embeds,
            None,
            timestep_seq_len=ts_seq_len,
        )
        if encoder_hidden_states_image is not None:
            raise RuntimeError("Wan TI2V-5B wrapper does not expect image encoder hidden states.")
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in transformer.blocks:
            tokens = block(tokens, encoder_hidden_states, timestep_proj, rotary_emb)

        hidden_tokens = tokens

        if temb.ndim == 3:
            shift, scale = (transformer.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
                2, dim=2
            )
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_tokens.device)
        scale = scale.to(hidden_tokens.device)
        tokens = (transformer.norm_out(hidden_tokens.float()) * (1 + scale) + shift).type_as(hidden_tokens)
        tokens = transformer.proj_out(tokens)

        tokens = tokens.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        velocity = tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return velocity, hidden_tokens

    @torch.no_grad()
    def forward(
        self,
        *,
        video_xt: Tensor,
        timesteps: Tensor,
        condition_video: Tensor,
        sequence_names: Sequence[str],
        object_categories: Optional[Sequence[str]] = None,
        condition_latents: Optional[Tensor] = None,
        video_xt_human: Optional[Tensor] = None,
        video_xt_object: Optional[Tensor] = None,
    ) -> WanTeacherOutput:
        if video_xt.ndim != 5 or video_xt.shape[1] != self.latent_channels:
            raise ValueError(
                f"Wan video latents must have shape [B, {self.latent_channels}, T', H', W'], got {tuple(video_xt.shape)}."
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != video_xt.shape[0]:
            raise ValueError(f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}.")
        if len(sequence_names) != video_xt.shape[0]:
            raise ValueError(
                f"Expected one sequence name per batch item, got batch={video_xt.shape[0]} and names={len(sequence_names)}."
            )

        branch_video_xt_human = video_xt if video_xt_human is None else video_xt_human
        branch_video_xt_object = video_xt if video_xt_object is None else video_xt_object
        if branch_video_xt_human.shape != video_xt.shape or branch_video_xt_object.shape != video_xt.shape:
            raise ValueError(
                "Wan branch latents must match the shared video latent shape. "
                f"Shared={tuple(video_xt.shape)} | human={tuple(branch_video_xt_human.shape)} "
                f"| object={tuple(branch_video_xt_object.shape)}."
            )

        condition_latents, human_inpaint_mask, object_inpaint_mask = self._encode_condition(
            condition_video,
            condition_latents=condition_latents,
        )
        human_prompt_embeds = self._encode_prompts(
            sequence_names=sequence_names,
            object_categories=object_categories,
            device=video_xt.device,
            branch="human",
        )
        object_prompt_embeds = self._encode_prompts(
            sequence_names=sequence_names,
            object_categories=object_categories,
            device=video_xt.device,
            branch="object",
        )
        human_velocity, human_hidden_tokens = self._forward_single_branch(
            video_xt=branch_video_xt_human,
            timesteps=timesteps,
            condition_latents=condition_latents,
            inpaint_mask=human_inpaint_mask,
            prompt_embeds=human_prompt_embeds,
        )
        object_velocity, object_hidden_tokens = self._forward_single_branch(
            video_xt=branch_video_xt_object,
            timesteps=timesteps,
            condition_latents=condition_latents,
            inpaint_mask=object_inpaint_mask,
            prompt_embeds=object_prompt_embeds,
        )
        velocity = 0.5 * (human_velocity + object_velocity)
        hidden_tokens = torch.cat([human_hidden_tokens, object_hidden_tokens], dim=1)
        return WanTeacherOutput(
            velocity=velocity.to(device=video_xt.device, dtype=video_xt.dtype),
            hidden_tokens=hidden_tokens.to(device=video_xt.device, dtype=self.transformer.dtype),
            human_velocity=human_velocity.to(device=video_xt.device, dtype=video_xt.dtype),
            object_velocity=object_velocity.to(device=video_xt.device, dtype=video_xt.dtype),
            human_hidden_tokens=human_hidden_tokens.to(device=video_xt.device, dtype=self.transformer.dtype),
            object_hidden_tokens=object_hidden_tokens.to(device=video_xt.device, dtype=self.transformer.dtype),
        )
