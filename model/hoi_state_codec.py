from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _apply_gaussian_activation(raw_tokens: Tensor) -> Tensor:
    """把网络输出的 14 维高斯参数约束到可用范围。"""
    if raw_tokens.shape[-1] != 14:
        raise ValueError(f"Expected 14D Gaussian tokens, got {raw_tokens.shape[-1]}.")
    # 高斯 token 的布局: xyz(3) + quaternion(4) + scale(3) + opacity(1) + SH/RGB(3)。
    xyz = raw_tokens[..., 0:3]
    rotation = F.normalize(raw_tokens[..., 3:7], dim=-1)
    scaling = F.softplus(raw_tokens[..., 7:10]) + 1e-6
    opacity = raw_tokens[..., 10:11].sigmoid()
    shs = raw_tokens[..., 11:14].sigmoid()
    return torch.cat([xyz, rotation, scaling, opacity, shs], dim=-1)


def _rotation_matrix_to_6d(matrix: Tensor) -> Tensor:
    """将旋转矩阵转换成连续的 6D 旋转表示，避免欧拉角/四元数的不连续问题。"""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"`matrix` must have shape [..., 3, 3], got {tuple(matrix.shape)}.")
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """使用 Gram-Schmidt 正交化把 6D 旋转表示还原成 3x3 矩阵。"""
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
    """把每帧物体位姿 [R|t] 展平成 9 维 token 输入。"""
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise ValueError(f"`transforms` must have shape [B, T, 4, 4], got {tuple(transforms.shape)}.")
    rotation_6d = _rotation_matrix_to_6d(transforms[:, :, :3, :3])
    translation = transforms[:, :, :3, 3]
    return torch.cat([rotation_6d, translation], dim=-1)


def _unflatten_object_transforms(flattened: Tensor) -> Tensor:
    """把 9 维物体运动 token 解码回齐次变换矩阵。"""
    if flattened.ndim != 3 or flattened.shape[-1] != 9:
        raise ValueError(f"`flattened` must have shape [B, T, 9], got {tuple(flattened.shape)}.")
    batch_size, num_frames = flattened.shape[:2]
    transforms = flattened.new_zeros(batch_size, num_frames, 4, 4)
    transforms[:, :, :3, :3] = _rotation_6d_to_matrix(flattened[:, :, :6])
    transforms[:, :, :3, 3] = flattened[:, :, 6:9]
    transforms[:, :, 3, 3] = 1.0
    return transforms

@dataclass
class DecodedHOIState:
    """HOI token 解码后的显式人体-物体交互状态。"""

    human_shape: Tensor
    human_pose: Tensor
    human_translation: Tensor
    human_gaussians: Tensor
    object_gaussians: Tensor
    joints_3d: Tensor
    object_transforms: Tensor
    contact_signature: Tensor


class HOIStateCodec(nn.Module):
    """HOI 状态编解码器。

    该模块负责在“显式 HOI 状态”和 Transformer 使用的 token 序列之间转换：
    人体形状/姿态/平移、人体与物体高斯、关节点、物体运动和接触签名都会被映射到同一 hidden_dim。
    """

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

        # 不同语义 token 的数量。顺序需要和 _slices()/encode/decode 保持一致。
        self.num_context_tokens = 1
        self.num_shape_tokens = 1
        self.num_pose_tokens = self.num_frames
        self.num_translation_tokens = self.num_frames
        self.num_object_motion_tokens = self.num_frames
        self.num_contact_tokens = self.num_frames
        self.num_joint_tokens = self.num_frames * self.num_joints

        # 位置/类型嵌入只用于把监督目标编码到固定 token 布局，不作为可训练目标编码器。
        self.context_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.frame_embedding = nn.Parameter(torch.zeros(self.num_frames, hidden_dim))
        self.type_embedding = nn.Parameter(torch.zeros(8, hidden_dim))
        self.human_gaussian_pos = nn.Parameter(torch.zeros(self.num_human_gaussians, hidden_dim))
        self.object_gaussian_pos = nn.Parameter(torch.zeros(self.num_object_gaussians, hidden_dim))
        self.joint_pos = nn.Parameter(torch.zeros(self.num_joints, hidden_dim))

        # 输入投影负责把显式状态编码成 token；输出投影负责从 token 解码回原始物理量。
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
        """冻结目标编码端，保证训练主要学习 denoising/decoder，而不是移动监督坐标系。"""
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
        """返回各类 token 在拼接序列中的切片位置。"""
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

    def build_token_expert_targets(self) -> Tensor:
        """为 HOI-token-aware MoE 构造每个 token 的专家监督标签。"""
        parts = self._slices()
        # 默认专家 4 是共享 base；更明确的 HOI 子任务会路由到对应专家。
        targets = torch.full((self.total_tokens,), 4, dtype=torch.long)
        targets[parts["shape"]] = 0
        targets[parts["pose"]] = 0
        targets[parts["translation"]] = 0
        targets[parts["joints"]] = 0
        targets[parts["object_motion"]] = 1
        targets[parts["object_gaussians"]] = 1
        targets[parts["contact"]] = 2
        targets[parts["human_gaussians"]] = 3
        return targets

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
        """把一批显式 HOI 监督值编码成 [B, L, D] token 序列。"""
        batch_size = human_shape.shape[0]
        if human_pose.shape[1:] != (self.num_frames, self.human_pose_dim):
            raise ValueError(f"Expected human_pose [B, {self.num_frames}, {self.human_pose_dim}].")
        if joints_3d.shape[1:] != (self.num_frames, self.num_joints, 3):
            raise ValueError(f"Expected joints_3d [B, {self.num_frames}, {self.num_joints}, 3].")

        context = self.context_token.unsqueeze(0).expand(batch_size, -1, -1) + self.type_embedding[0]
        shape = self.shape_in(human_shape).unsqueeze(1) + self.type_embedding[1]
        # 帧级 token 叠加 frame embedding；高斯/关节 token 叠加自身的位置 embedding。
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
        """把模型输出 token 按固定切片解码回显式 HOI 状态。"""
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


__all__ = [
    "DecodedHOIState",
    "HOIStateCodec",
]
