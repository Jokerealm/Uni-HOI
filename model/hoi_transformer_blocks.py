from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """生成 DiT 常用的正弦/余弦时间步嵌入。"""
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


class MultiHeadAttention(nn.Module):
    """轻量多头注意力封装，支持自注意力和跨注意力。"""

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
            # bool mask 表示可见位置；非 bool mask 直接当作 additive attention bias。
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
    """标准 PreNorm Transformer block，用作简单 token 建模基线。"""

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


HOI_TOKEN_MOE_EXPERT_NAMES = (
    "human_pose_joints",
    "object_motion_gaussians",
    "contact",
    "surface_gaussian",
    "base_shared",
)


class LightweightFFN(nn.Module):
    """MoE 中的轻量残差专家，默认零初始化输出以保持初始行为稳定。"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, *, zero_init_output: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        if zero_init_output:
            nn.init.zeros_(self.net[-2].weight)
            nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HOITokenAwareMoE(nn.Module):
    """面向 HOI token 的残差 MoE。

    原始 DiT FFN 始终作为共享/base 专家；router 根据 stop-gradient 的 hidden state
    选择轻量残差专家，因此路由监督不会反向干扰共享表示。
    """

    num_experts = len(HOI_TOKEN_MOE_EXPERT_NAMES)

    def __init__(
        self,
        *,
        dim: int,
        expert_hidden_dim: int,
        router_hidden_dim: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        router_hidden = int(router_hidden_dim) if int(router_hidden_dim) > 0 else int(dim)
        expert_hidden = max(int(expert_hidden_dim), 1)
        self.residual_scale = float(residual_scale)
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, self.num_experts),
        )
        self.residual_experts = nn.ModuleList(
            [
                LightweightFFN(dim, expert_hidden, dropout, zero_init_output=True),
                LightweightFFN(dim, expert_hidden, dropout, zero_init_output=True),
                LightweightFFN(dim, expert_hidden, dropout, zero_init_output=True),
                LightweightFFN(dim, expert_hidden, dropout, zero_init_output=True),
            ]
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(
        self,
        tokens: Tensor,
        *,
        shared_output: Tensor,
        token_expert_targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # router 使用 detach 后的 token，避免专家标签损失直接改变主干表示。
        logits = self.router(tokens.detach())
        probs = F.softmax(logits.float(), dim=-1).to(dtype=tokens.dtype)

        # 四个专用残差专家 + 一个全零 base 槽位；最终按 router 概率加权求和。
        residual_outputs = [expert(tokens) for expert in self.residual_experts]
        residual_outputs.append(shared_output.new_zeros(shared_output.shape))
        residual_stack = torch.stack(residual_outputs, dim=2)
        residual = (residual_stack * probs.unsqueeze(-1)).sum(dim=2)

        if token_expert_targets.ndim != 1 or token_expert_targets.shape[0] != tokens.shape[1]:
            raise ValueError(
                "`token_expert_targets` must have shape "
                f"[{tokens.shape[1]}], got {tuple(token_expert_targets.shape)}."
            )
        targets = token_expert_targets.to(device=tokens.device, dtype=torch.long)
        targets = targets.view(1, -1).expand(tokens.shape[0], -1).reshape(-1)
        router_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets)
        return shared_output + self.residual_scale * residual, router_loss


class StreamAdaptiveModulation(nn.Module):
    """为 HOI/RGB 两条流分别生成 AdaLN 的 shift/scale/gate。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )
        self.to_gates = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)
        nn.init.zeros_(self.to_gates[-1].weight)
        nn.init.zeros_(self.to_gates[-1].bias)

    def forward(self, time_cond: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # time_cond 既支持 [B, D]，也支持广播友好的 [B, 1, D]。
        if time_cond.ndim == 3:
            if time_cond.shape[1] != 1:
                time_cond = time_cond.mean(dim=1, keepdim=True)
            cond = time_cond.squeeze(1)
        elif time_cond.ndim == 2:
            cond = time_cond
        else:
            raise ValueError(f"`time_cond` must have shape [B, D] or [B, 1, D], got {tuple(time_cond.shape)}.")
        shift_attn, scale_attn, shift_mlp, scale_mlp = self.to_scale_shift(cond).chunk(4, dim=-1)
        gate_attn, gate_mlp = self.to_gates(cond).chunk(2, dim=-1)
        return (
            shift_attn.unsqueeze(1),
            scale_attn.unsqueeze(1),
            shift_mlp.unsqueeze(1),
            scale_mlp.unsqueeze(1),
            gate_attn.unsqueeze(1),
            gate_mlp.unsqueeze(1),
        )


class SharedStreamTransformerBlock(nn.Module):
    """共享权重的 DiT block。

    注意力和 MLP 权重在 HOI/RGB 间共享，但 AdaLN 的调制参数按流区分，从而保留模态差异。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        *,
        enable_hoi_token_moe: bool = False,
        hoi_token_moe_expert_dim: int = 256,
        hoi_token_moe_router_hidden_dim: int = 0,
        hoi_token_moe_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.hoi_token_moe = (
            HOITokenAwareMoE(
                dim=dim,
                expert_hidden_dim=hoi_token_moe_expert_dim,
                router_hidden_dim=hoi_token_moe_router_hidden_dim,
                dropout=dropout,
                residual_scale=hoi_token_moe_residual_scale,
            )
            if enable_hoi_token_moe
            else None
        )
        self.hoi_modulation = StreamAdaptiveModulation(dim)
        self.rgb_modulation = StreamAdaptiveModulation(dim)

    def _modulation_for_stream(self, stream: str) -> StreamAdaptiveModulation:
        """根据流名称选择对应的调制器。"""
        if stream == "hoi":
            return self.hoi_modulation
        if stream == "rgb":
            return self.rgb_modulation
        raise ValueError(f"Unknown stream {stream!r}; expected 'hoi' or 'rgb'.")

    def _modulate(self, x: Tensor, *, time_cond: Tensor, stream: str, mlp: bool = False) -> Tuple[Tensor, Tensor]:
        """执行 AdaLN 调制，并返回调制后的输入和残差门控。"""
        shift_attn, scale_attn, shift_mlp, scale_mlp, gate_attn, gate_mlp = self._modulation_for_stream(stream)(
            time_cond
        )
        if mlp:
            shift = shift_mlp
            scale = scale_mlp
            gate = gate_mlp
            normed = self.norm2(x)
        else:
            shift = shift_attn
            scale = scale_attn
            gate = gate_attn
            normed = self.norm1(x)
        shift = shift.to(device=x.device, dtype=x.dtype)
        scale = scale.to(device=x.device, dtype=x.dtype)
        gate = (1.0 + gate).to(device=x.device, dtype=x.dtype)
        return normed * (1.0 + scale) + shift, gate

    def _apply_mlp(
        self,
        x: Tensor,
        *,
        stream: str,
        hoi_token_expert_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """应用共享 FFN；HOI 流可额外叠加 token-aware MoE 残差。"""
        shared_output = self.mlp(x)
        if stream == "hoi" and self.hoi_token_moe is not None:
            return self.hoi_token_moe(
                x,
                shared_output=shared_output,
                token_expert_targets=hoi_token_expert_targets,
            )
        return shared_output, shared_output.new_zeros(())

    def forward(
        self,
        x: Tensor,
        *,
        time_cond: Tensor,
        stream: str,
        hoi_token_expert_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """单流前向：用于只更新 HOI 或只更新 RGB token 的场景。"""
        attn_input, attn_gate = self._modulate(x, time_cond=time_cond, stream=stream, mlp=False)
        x = x + attn_gate * self.attn(attn_input)
        mlp_input, mlp_gate = self._modulate(x, time_cond=time_cond, stream=stream, mlp=True)
        mlp_out, router_loss = self._apply_mlp(
            mlp_input,
            stream=stream,
            hoi_token_expert_targets=hoi_token_expert_targets,
        )
        x = x + mlp_gate * mlp_out
        return x, router_loss

    def forward_full_pair(
        self,
        hoi_tokens: Tensor,
        rgb_tokens: Tensor,
        *,
        time_cond: Tensor,
        hoi_token_expert_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """拼接 HOI/RGB 后做全注意力，实现两条流的直接交互。"""

        hoi_len = int(hoi_tokens.shape[1])
        hoi_attn_input, hoi_attn_gate = self._modulate(hoi_tokens, time_cond=time_cond, stream="hoi", mlp=False)
        rgb_attn_input, rgb_attn_gate = self._modulate(rgb_tokens, time_cond=time_cond, stream="rgb", mlp=False)
        attn_input = torch.cat([hoi_attn_input, rgb_attn_input], dim=1)
        attn_gate = torch.cat([hoi_attn_gate.expand_as(hoi_tokens), rgb_attn_gate.expand_as(rgb_tokens)], dim=1)
        attended = self.attn(attn_input)
        attended = attn_gate * attended
        hoi_tokens = hoi_tokens + attended[:, :hoi_len]
        rgb_tokens = rgb_tokens + attended[:, hoi_len:]

        hoi_mlp_input, hoi_mlp_gate = self._modulate(hoi_tokens, time_cond=time_cond, stream="hoi", mlp=True)
        rgb_mlp_input, rgb_mlp_gate = self._modulate(rgb_tokens, time_cond=time_cond, stream="rgb", mlp=True)
        if self.hoi_token_moe is None:
            mlp_input = torch.cat([hoi_mlp_input, rgb_mlp_input], dim=1)
            mlp_gate = torch.cat([hoi_mlp_gate.expand_as(hoi_tokens), rgb_mlp_gate.expand_as(rgb_tokens)], dim=1)
            mlp_out = self.mlp(mlp_input)
            mlp_out = mlp_gate * mlp_out
            hoi_tokens = hoi_tokens + mlp_out[:, :hoi_len]
            rgb_tokens = rgb_tokens + mlp_out[:, hoi_len:]
            router_loss = mlp_out.new_zeros(())
        else:
            hoi_mlp_out, router_loss = self._apply_mlp(
                hoi_mlp_input,
                stream="hoi",
                hoi_token_expert_targets=hoi_token_expert_targets,
            )
            rgb_mlp_out, _ = self._apply_mlp(rgb_mlp_input, stream="rgb")
            hoi_tokens = hoi_tokens + hoi_mlp_gate.expand_as(hoi_tokens) * hoi_mlp_out
            rgb_tokens = rgb_tokens + rgb_mlp_gate.expand_as(rgb_tokens) * rgb_mlp_out
        return hoi_tokens, rgb_tokens, router_loss


class ZeroInitCrossAdapter(nn.Module):
    """零初始化跨注意力适配器，初始时不改变主干输出，训练后逐步学习跨流注入。"""

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
    """把首帧 RGB 图像切成 patch token，作为静态外观先验。"""

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


__all__ = [
    "FirstFramePatchEncoder",
    "HOITokenAwareMoE",
    "HOI_TOKEN_MOE_EXPERT_NAMES",
    "MultiHeadAttention",
    "SharedStreamTransformerBlock",
    "TransformerBlock",
    "ZeroInitCrossAdapter",
    "timestep_embedding",
]
