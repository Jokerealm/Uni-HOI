from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """把配置里的 dtype 字符串转换成 PyTorch dtype。"""
    text = str(name).strip().lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}. Expected bf16/fp16/fp32.")


def _retrieve_latents(encoder_output, sample_mode: str = "argmax") -> Tensor:
    """兼容不同 diffusers VAE encode 返回格式，取出 latent tensor。"""
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
    """兼容不同 diffusers VAE decode 返回格式，取出视频样本。"""
    if hasattr(decoder_output, "sample"):
        return decoder_output.sample
    if isinstance(decoder_output, (tuple, list)) and decoder_output:
        return decoder_output[0]
    if isinstance(decoder_output, Tensor):
        return decoder_output
    raise AttributeError("Could not retrieve video sample from Wan VAE output.")

@dataclass
class WanRGBStreamOutput:
    """Wan RGB 分支输出：latent velocity 以及 Transformer hidden tokens。"""

    velocity: Tensor
    hidden_tokens: Tensor


class FrozenWanTI2VImageStream(nn.Module):
    """冻结/可选训练的 Wan2.2-TI2V RGB 分支。

    该封装使用空文本提示，把首帧保持为条件帧，并复用 Wan Transformer 预测视频 latent velocity。
    """

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
        """延迟加载 Wan Pipeline，避免构造模型时立即占用显存。"""
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

        # 从 Wan 配置推导 latent 网格和 patch token 形状，后续构造 per-token timestep 要用到。
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
        # VAE 和文本编码器始终冻结；Transformer 可由 freeze 控制是否参与训练。
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
        """使用 Wan VAE 配置中的均值/方差归一化 latent。"""
        return (latents - self.latents_mean.to(latents.device, latents.dtype)) * self.latents_std_inv.to(
            latents.device, latents.dtype
        )

    def _denormalize_latents(self, latents: Tensor) -> Tensor:
        """把归一化 latent 还原到 VAE decode 所需尺度。"""
        return latents / self.latents_std_inv.to(latents.device, latents.dtype) + self.latents_mean.to(
            latents.device, latents.dtype
        )

    @torch.no_grad()
    def decode_video(self, latents: Tensor) -> Tensor:
        """将 [B, C, T, H, W] Wan latent 解码成 [B, T, 3, H, W] RGB 视频。"""
        self.ensure_loaded(latents.device)
        if latents.ndim != 5:
            raise ValueError(f"Wan latents must have shape [B, C, T, H, W], got {tuple(latents.shape)}.")
        latents = self._denormalize_latents(latents.to(device=self.latents_mean.device, dtype=self.vae.dtype))
        video = _retrieve_video_sample(self.vae.decode(latents))
        video = video.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def encode_video(self, video: Tensor) -> Tensor:
        """把 RGB 视频或单帧图像编码到 Wan latent 空间。"""
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
        """只编码首帧，作为 TI2V 任务中需要固定的条件 latent。"""
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
        """构造 flow-matching 风格的 noisy latent 与监督 velocity。

        首帧 latent 被强制替换为条件帧，并且首帧 velocity 置零，保证模型不修改输入首帧。
        """
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
        # t=1 接近目标视频，t=0 接近噪声；训练目标是 target - noise。
        xt = t_view * target + (1.0 - t_view) * noise
        xt[:, :, :1] = target[:, :, :1]
        velocity = target - noise
        velocity[:, :, :1] = 0.0
        return xt, velocity

    def _encode_empty_prompt(self, *, batch_size: int, device: torch.device) -> Tensor:
        """缓存空文本提示的 embedding，减少重复调用文本编码器。"""
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
        """为 Wan patch token 构造逐 token timestep；首帧 token 的 timestep 固定为 0。"""
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
        """调用 Wan Transformer，并同时保留中间 hidden tokens 供 HOI 分支使用。"""
        transformer = self.transformer
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = transformer.config.patch_size
        post_f = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w

        rotary_emb = transformer.rope(hidden_states)
        tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        # diffusers Wan 的 condition_embedder 需要展平后的 per-token timestep。
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
        # Transformer 输出 token 需要经过 Wan 原生 norm/proj/unpatchify 才回到 latent velocity。
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
        """RGB 分支前向：输入 noisy video latent，输出 Wan velocity 与上下文 token。"""
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


__all__ = [
    "FrozenWanTI2VImageStream",
    "WanRGBStreamOutput",
]
