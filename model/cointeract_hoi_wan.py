from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .hoi_state_codec import DecodedHOIState, HOIStateCodec
from .hoi_transformer_blocks import SharedStreamTransformerBlock, timestep_embedding
from .wan_rgb_stream import FrozenWanTI2VImageStream, WanRGBStreamOutput


class CoInteractHOIBlock(nn.Module):
    """以 HOI 为主任务的共享双流 block。

    DiT 的注意力/MLP 权重在 HOI 和 RGB token 间共享；模态差异来自输入投影以及各自的
    AdaLN scale/shift/gate。HOI/RGB 交互通过全注意力完成，最终预测头仍只服务 HOI。
    """

    def __init__(self,*,hidden_dim: int,num_heads: int,mlp_ratio: float,dropout: float,
        enable_hoi_token_moe: bool = False,
        hoi_token_moe_expert_dim: int = 256,hoi_token_moe_router_hidden_dim: int = 0,
        hoi_token_moe_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.shared_block = SharedStreamTransformerBlock(
            hidden_dim,num_heads,mlp_ratio,dropout,
            enable_hoi_token_moe=enable_hoi_token_moe,hoi_token_moe_expert_dim=hoi_token_moe_expert_dim,
            hoi_token_moe_router_hidden_dim=hoi_token_moe_router_hidden_dim,
            hoi_token_moe_residual_scale=hoi_token_moe_residual_scale,
        )

    def forward(self,hoi_tokens: Tensor,*,rgb_tokens: Optional[Tensor],
        time_cond: Tensor,hoi_token_expert_targets: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        if rgb_tokens is not None:
            # 有 RGB 先验时，将 HOI/RGB token 拼接后做全注意力，实现跨模态交互。
            hoi_tokens, rgb_tokens, router_loss = self.shared_block.forward_full_pair(
                hoi_tokens,
                rgb_tokens,
                time_cond=time_cond,
                hoi_token_expert_targets=hoi_token_expert_targets,
            )
        else:
            # HOI-only 路径用于消融或无视觉先验训练。
            hoi_tokens, router_loss = self.shared_block(
                hoi_tokens,
                time_cond=time_cond,
                stream="hoi",
                hoi_token_expert_targets=hoi_token_expert_targets,
            )
        return hoi_tokens, rgb_tokens, router_loss


@dataclass
class CoInteractHOI4DOutput:
    """CoInteract 模型输出，包括 HOI velocity、可选 RGB velocity 和路由损失。"""

    rgb_velocity: Optional[Tensor]
    state_velocity: Tensor
    decoded_state: DecodedHOIState
    rgb_hidden_tokens: Optional[Tensor]
    rgb_context_tokens: Optional[Tensor]
    hoi_router_loss: Tensor


class CoInteractHOI4DModel(nn.Module):
    """HOI-primary 的 Wan RGB 引导模型。

    RGB 分支复用 Wan TI2V Transformer 提供上下文 token；HOI 分支负责显式状态去噪和解码。
    """

    video_backend = "wan2.2-ti2v-5b"
    input_mode = "rgb_video_guided_hoi"

    def __init__(self,*,hidden_dim: int,num_heads: int,
        depth: int,mlp_ratio: float,dropout: float,
        num_frames: int,image_height: int,
        image_width: int, num_human_gaussians: int,
        num_object_gaussians: int,num_joints: int,
        contact_dim: int, human_shape_dim: int,
        human_pose_dim: int, wan_model_id: str,
        wan_dtype: str = "bf16",
        wan_hidden_dim: int = 3072,
        wan_prompt_max_sequence_length: int = 512,
        wan_local_files_only: bool = True,
        freeze_wan: bool = True,
        detach_rgb_context: bool = True,
        enable_hoi_token_moe: bool = False,
        hoi_token_moe_expert_dim: int = 256,
        hoi_token_moe_router_hidden_dim: int = 0,
        hoi_token_moe_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.detach_rgb_context = bool(detach_rgb_context)
        # HOIStateCodec 决定显式状态 token 的固定布局，也是 decode 的唯一入口。
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
        self.register_buffer(
            "hoi_token_expert_targets",
            self.state_codec.build_token_expert_targets(),
            persistent=False,
        )
        # Wan RGB 分支输出视频 latent velocity，同时暴露 hidden token 作为 HOI 上下文。
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
        # timestep 条件会加到 HOI/RGB token 上，并传给每个共享 block 的 AdaLN。
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
                    enable_hoi_token_moe=enable_hoi_token_moe,
                    hoi_token_moe_expert_dim=hoi_token_moe_expert_dim,
                    hoi_token_moe_router_hidden_dim=hoi_token_moe_router_hidden_dim,
                    hoi_token_moe_residual_scale=hoi_token_moe_residual_scale,
                )
                for _ in range(depth)
            ]
        )
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
        video_xt: Optional[Tensor],
        state_xt: Tensor,
        timesteps: Tensor,
        use_rgb_prior: bool = True,
    ) -> CoInteractHOI4DOutput:
        """执行 HOI token 去噪；可选使用 Wan RGB token 作为视觉先验。"""
        if state_xt.ndim != 3 or state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(f"`state_xt` must have shape [B, L, {self.hidden_dim}], got {tuple(state_xt.shape)}.")

        rgb_output: Optional[WanRGBStreamOutput] = None
        rgb_hidden_tokens: Optional[Tensor] = None
        rgb_tokens: Optional[Tensor] = None
        use_rgb = bool(use_rgb_prior)
        if use_rgb:
            if video_xt is None:
                raise ValueError("`video_xt` is required when use_rgb_prior=True.")
            rgb_output = self.rgb_stream(video_xt=video_xt, timesteps=timesteps)
            # detach_rgb_context=True 时，HOI 损失不会回传到 Wan RGB 主干。
            rgb_hidden_tokens = rgb_output.hidden_tokens.detach() if self.detach_rgb_context else rgb_output.hidden_tokens
            rgb_tokens = self.rgb_token_adapter(rgb_hidden_tokens.to(dtype=state_xt.dtype))

        time_cond = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1).to(dtype=state_xt.dtype)
        state_tokens = state_xt + time_cond
        if rgb_tokens is not None:
            rgb_tokens = rgb_tokens + time_cond
        router_losses = []
        for block in self.blocks:
            # 每层返回 HOI token、可选 RGB token，以及 HOI token-aware MoE 的路由监督损失。
            state_tokens, rgb_tokens, block_router_loss = block(
                state_tokens,
                rgb_tokens=rgb_tokens,
                time_cond=time_cond,
                hoi_token_expert_targets=self.hoi_token_expert_targets,
            )
            router_losses.append(block_router_loss.to(device=state_xt.device, dtype=torch.float32))
        hoi_router_loss = torch.stack(router_losses).mean()
        state_velocity = self.state_velocity_head(self.state_norm(state_tokens))
        # 这一项数值为 0，用于保持图连接/避免某些分布式训练场景下的 unused parameter 问题。
        state_velocity = state_velocity + state_tokens.float().sum().to(dtype=state_velocity.dtype) * 0.0
        t_view = timesteps.to(device=state_xt.device, dtype=state_xt.dtype).view(state_xt.shape[0], 1, 1)
        # flow-matching 反推 clean state token，再交给 codec 解码成显式 HOI 状态。
        predicted_clean_state = state_xt + (1.0 - t_view) * state_velocity.to(dtype=state_xt.dtype)
        return CoInteractHOI4DOutput(
            rgb_velocity=rgb_output.velocity if rgb_output is not None else None,
            state_velocity=state_velocity,
            decoded_state=self.decode_state_tokens(predicted_clean_state),
            rgb_hidden_tokens=rgb_output.hidden_tokens if rgb_output is not None else None,
            rgb_context_tokens=rgb_tokens,
            hoi_router_loss=hoi_router_loss,
        )

    def forward_hoi_only(self, *, state_xt: Tensor, timesteps: Tensor) -> CoInteractHOI4DOutput:
        return self.forward(
            video_xt=None,
            state_xt=state_xt,
            timesteps=timesteps,
            use_rgb_prior=False,
        )


__all__ = [
    "CoInteractHOIBlock",
    "CoInteractHOI4DModel",
    "CoInteractHOI4DOutput",
]
