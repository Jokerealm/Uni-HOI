from __future__ import annotations

"""HOI/Wan 相关模型组件的公共导出入口。

训练脚本可以从这里统一导入 codec、共享 Transformer block、Wan RGB 分支等基础模块。
"""

from .hoi_state_codec import DecodedHOIState, HOIStateCodec
from .hoi_transformer_blocks import (
    FirstFramePatchEncoder,
    HOITokenAwareMoE,
    HOI_TOKEN_MOE_EXPERT_NAMES,
    MultiHeadAttention,
    SharedStreamTransformerBlock,
    TransformerBlock,
    ZeroInitCrossAdapter,
    timestep_embedding,
)
from .wan_rgb_stream import FrozenWanTI2VImageStream, WanRGBStreamOutput

__all__ = [
    "DecodedHOIState",
    "FirstFramePatchEncoder",
    "FrozenWanTI2VImageStream",
    "HOITokenAwareMoE",
    "HOIStateCodec",
    "HOI_TOKEN_MOE_EXPERT_NAMES",
    "MultiHeadAttention",
    "SharedStreamTransformerBlock",
    "TransformerBlock",
    "WanRGBStreamOutput",
    "ZeroInitCrossAdapter",
    "timestep_embedding",
]
