from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from model.dual_branch_cogenerative_fm import (
    ConditionEncoder,
    DecodedHOIState,
    DualBranchFMOutput,
    GeometryProjector,
    HOIStateCodec,
    TransformerBlock,
    ZeroInitCrossAdapter,
    timestep_embedding,
)
from model.wan_video_teacher import FrozenWanVideoTeacher


def _resolve_wan_num_latent_frames(num_frames: int) -> int:
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be > 0, got {num_frames}.")
    return (num_frames - 1) // 4 + 1


class GeometrySummaryEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int, num_frames: int, num_channels: int = 5) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.num_channels = int(num_channels)
        self.in_proj = nn.Linear(self.num_channels * 2, self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.frame_embedding = nn.Parameter(torch.zeros(self.num_frames, self.hidden_dim))
        nn.init.normal_(self.frame_embedding, std=0.02)

    def forward(self, geometry_maps: Tensor) -> Tensor:
        if geometry_maps.ndim != 5:
            raise ValueError(f"`geometry_maps` must have shape [B, T, C, H, W], got {tuple(geometry_maps.shape)}.")
        if geometry_maps.shape[1] != self.num_frames or geometry_maps.shape[2] != self.num_channels:
            raise ValueError(
                f"Expected geometry maps shape [B, {self.num_frames}, {self.num_channels}, H, W], "
                f"got {tuple(geometry_maps.shape)}."
            )
        mean_features = geometry_maps.mean(dim=(-2, -1))
        max_features = geometry_maps.amax(dim=(-2, -1))
        summary = torch.cat([mean_features, max_features], dim=-1)
        tokens = self.in_proj(summary) + self.frame_embedding.unsqueeze(0)
        return self.out_norm(tokens)


class WanStateFusionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_block = TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
        self.global_from_teacher = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout=dropout)
        self.dynamic_from_teacher = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout=dropout)
        self.dynamic_from_condition = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout=dropout)
        self.dynamic_from_geometry = ZeroInitCrossAdapter(hidden_dim, num_heads, dropout=dropout)

    def forward(
        self,
        state_tokens: Tensor,
        *,
        state_codec: HOIStateCodec,
        teacher_tokens: Tensor,
        condition_tokens: Tensor,
        geometry_tokens: Tensor,
        video_to_state_scale: float,
    ) -> Tensor:
        state_tokens = self.state_block(state_tokens)
        global_tokens, dynamic_tokens = state_codec.split_global_dynamic(state_tokens)
        global_tokens = global_tokens + float(video_to_state_scale) * self.global_from_teacher(global_tokens, teacher_tokens)
        dynamic_tokens = dynamic_tokens + float(video_to_state_scale) * self.dynamic_from_teacher(
            dynamic_tokens,
            teacher_tokens,
        )
        dynamic_tokens = dynamic_tokens + self.dynamic_from_condition(dynamic_tokens, condition_tokens)
        dynamic_tokens = dynamic_tokens + self.dynamic_from_geometry(dynamic_tokens, geometry_tokens)
        return state_codec.merge_global_dynamic(global_tokens, dynamic_tokens)


class WanTeacherTokenAdapter(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.out_norm(self.proj(self.in_norm(tokens)))


class DualBranchWanTeacherFlowMatching(nn.Module):
    video_backend = "wan_ti2v_5b"
    video_output_mode = "full_rgb"

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        condition_channels: int,
        patch_size: int,
        condition_patch_size: int,
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
        wan_prompt_max_sequence_length: int = 512,
        wan_prompt_override: str = "",
        wan_local_files_only: bool = True,
        wan_hidden_dim: int = 3072,
        teacher_num_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frames = int(num_frames)
        self.teacher_num_frames = int(teacher_num_frames or num_frames)
        if self.teacher_num_frames < self.num_frames:
            raise ValueError(
                f"`teacher_num_frames` must be >= `num_frames`, got {self.teacher_num_frames} < {self.num_frames}."
            )
        self.valid_num_latent_frames = _resolve_wan_num_latent_frames(self.num_frames)
        self.teacher_num_latent_frames = _resolve_wan_num_latent_frames(self.teacher_num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_size = int(patch_size)
        self.condition_patch_size = int(condition_patch_size)
        if self.condition_patch_size <= 0:
            raise ValueError(f"`condition_patch_size` must be > 0, got {self.condition_patch_size}.")
        if self.image_height % self.condition_patch_size != 0 or self.image_width % self.condition_patch_size != 0:
            raise ValueError(
                f"Condition patch size {self.condition_patch_size} must divide image size "
                f"{(self.image_height, self.image_width)}."
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
        self.condition_encoder = ConditionEncoder(
            in_channels=condition_channels,
            hidden_dim=hidden_dim,
            patch_size=self.condition_patch_size,
            num_frames=num_frames,
            image_height=image_height,
            image_width=image_width,
        )
        self.geometry_projector = GeometryProjector(
            image_height=image_height,
            image_width=image_width,
            patch_size=self.patch_size,
        )
        self.geometry_encoder = GeometrySummaryEncoder(hidden_dim=hidden_dim, num_frames=num_frames, num_channels=5)
        self.teacher_token_adapter = WanTeacherTokenAdapter(input_dim=int(wan_hidden_dim), hidden_dim=hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                WanStateFusionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.state_velocity_head = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.state_velocity_head.weight)
        nn.init.zeros_(self.state_velocity_head.bias)

        self.wan_model_id = str(wan_model_id).strip()
        self.wan_dtype = str(wan_dtype).strip()
        self.wan_prompt_max_sequence_length = int(wan_prompt_max_sequence_length)
        self.wan_prompt_override = str(wan_prompt_override).strip()
        self.wan_local_files_only = bool(wan_local_files_only)
        self._wan_hidden_dim = int(wan_hidden_dim)
        object.__setattr__(self, "_video_teacher", None)
        self._resolved_wan_model_id = self.wan_model_id

    @property
    def resolved_wan_model_id(self) -> str:
        return self._resolved_wan_model_id

    def train(self, mode: bool = True) -> "DualBranchWanTeacherFlowMatching":
        super().train(mode)
        teacher = self.__dict__.get("_video_teacher")
        if teacher is not None:
            teacher.train(False)
        return self

    def _set_video_teacher(self, teacher: Optional[FrozenWanVideoTeacher]) -> None:
        self._modules.pop("_video_teacher", None)
        object.__setattr__(self, "_video_teacher", teacher)

    def _candidate_model_ids(self) -> list[str]:
        model_id = self.wan_model_id
        if model_id.endswith("-Diffusers"):
            return [model_id]
        return [f"{model_id}-Diffusers", model_id]

    def _get_video_teacher(self, device: torch.device) -> FrozenWanVideoTeacher:
        teacher = self.__dict__.get("_video_teacher")
        if teacher is None:
            errors: list[str] = []
            for candidate in self._candidate_model_ids():
                try:
                    teacher = FrozenWanVideoTeacher(
                        model_id=candidate,
                        num_frames=self.teacher_num_frames,
                        image_height=self.image_height,
                        image_width=self.image_width,
                        torch_dtype=self.wan_dtype,
                        prompt_max_sequence_length=self.wan_prompt_max_sequence_length,
                        prompt_override=self.wan_prompt_override,
                        local_files_only=self.wan_local_files_only,
                    )
                except Exception as exc:
                    errors.append(f"{candidate}: {exc}")
                    continue
                teacher = teacher.to(device=device)
                teacher.train(False)
                self._set_video_teacher(teacher)
                self._resolved_wan_model_id = candidate
                break
            teacher = self.__dict__.get("_video_teacher")
            if teacher is None:
                raise RuntimeError(
                    "Failed to initialize the frozen Wan teacher. Tried: "
                    + " | ".join(errors)
                )
        else:
            teacher.to(device=device)
        if int(teacher.num_latent_frames) != self.teacher_num_latent_frames:
            raise RuntimeError(
                f"Wan latent frame count mismatch: expected {self.teacher_num_latent_frames}, "
                f"got {teacher.num_latent_frames}."
            )
        return teacher

    def ensure_video_teacher(self, device: torch.device) -> None:
        self._get_video_teacher(device)

    def build_video_valid_mask(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mask = torch.zeros(
            int(batch_size),
            1,
            self.teacher_num_latent_frames,
            1,
            1,
            device=device,
            dtype=dtype,
        )
        mask[:, :, : self.valid_num_latent_frames] = 1
        return mask

    def apply_video_valid_mask(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"`latents` must have shape [B, C, T', H', W'], got {tuple(latents.shape)}.")
        if latents.shape[2] == self.valid_num_latent_frames and self.valid_num_latent_frames < self.teacher_num_latent_frames:
            pad_frames = self.teacher_num_latent_frames - self.valid_num_latent_frames
            latents = torch.cat(
                [
                    latents,
                    latents.new_zeros(
                        latents.shape[0],
                        latents.shape[1],
                        pad_frames,
                        latents.shape[3],
                        latents.shape[4],
                    ),
                ],
                dim=2,
            )
        elif latents.shape[2] != self.teacher_num_latent_frames:
            raise ValueError(
                f"Expected Wan latents with {self.valid_num_latent_frames} or {self.teacher_num_latent_frames} "
                f"frames, got {latents.shape[2]}."
            )

        if self.valid_num_latent_frames == self.teacher_num_latent_frames:
            return latents
        mask = self.build_video_valid_mask(
            latents.shape[0],
            device=latents.device,
            dtype=latents.dtype,
        )
        return latents * mask

    def _pad_video_frames_for_teacher(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, T, C, H, W], got {tuple(video.shape)}.")
        if video.shape[1] == self.teacher_num_frames:
            return video
        if video.shape[1] != self.num_frames:
            raise ValueError(
                f"Expected video with {self.num_frames} frames before Wan padding, got {video.shape[1]}."
            )
        pad_frames = self.teacher_num_frames - self.num_frames
        if pad_frames <= 0:
            return video
        padding = video.new_zeros(video.shape[0], pad_frames, video.shape[2], video.shape[3], video.shape[4])
        return torch.cat([video, padding], dim=1)

    @torch.no_grad()
    def sample_video_prior(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator],
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        teacher = self._get_video_teacher(device)
        if dtype is None:
            dtype = teacher.transformer.dtype
        prior = teacher.sample_prior_latents(
            batch_size,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return self.apply_video_valid_mask(prior)

    @torch.no_grad()
    def encode_video_target(self, video_target: Tensor) -> Tensor:
        teacher = self._get_video_teacher(video_target.device)
        encoded = teacher.encode_video(self._pad_video_frames_for_teacher(video_target))
        return self.apply_video_valid_mask(encoded)

    @torch.no_grad()
    def decode_video_tokens(self, video_tokens: Tensor) -> Tensor:
        teacher = self._get_video_teacher(video_tokens.device)
        return teacher.decode_video(video_tokens)

    def encode_state_target(
        self,
        *,
        human_shape: Tensor,
        human_pose: Tensor,
        human_translation: Tensor,
        object_transforms: Tensor,
        contact_signature: Tensor,
        human_gaussians: Optional[Tensor] = None,
        object_gaussians: Optional[Tensor] = None,
        joints_3d: Optional[Tensor] = None,
    ) -> Tensor:
        return self.state_codec.encode_targets(
            human_shape=human_shape,
            human_pose=human_pose,
            human_translation=human_translation,
            object_transforms=object_transforms,
            contact_signature=contact_signature,
            human_gaussians=human_gaussians,
            object_gaussians=object_gaussians,
            joints_3d=joints_3d,
        )

    def decode_state_tokens(self, state_tokens: Tensor) -> DecodedHOIState:
        return self.state_codec.decode_tokens(state_tokens)

    def project_geometry(self, decoded_state: DecodedHOIState, camera_intrinsics: Tensor):
        return self.geometry_projector(decoded_state, camera_intrinsics)

    def forward(
        self,
        *,
        video_xt: Tensor,
        state_xt: Tensor,
        timesteps: Tensor,
        condition_video: Tensor,
        camera_intrinsics: Tensor,
        sequence_names: Sequence[str],
        object_categories: Optional[Sequence[str]] = None,
        condition_latents: Optional[Tensor] = None,
        video_xt_human: Optional[Tensor] = None,
        video_xt_object: Optional[Tensor] = None,
        cross_branch_scale: Optional[float] = None,
        state_to_video_scale: float = 0.0,
        video_to_state_scale: float = 1.0,
    ) -> DualBranchFMOutput:
        if cross_branch_scale is not None:
            video_to_state_scale = float(cross_branch_scale)
        if abs(float(state_to_video_scale)) > 1e-6:
            raise ValueError(
                f"`state_to_video_scale` must stay at 0 for the fixed Wan teacher backend, got {state_to_video_scale}."
            )
        if video_xt.ndim != 5:
            raise ValueError(f"`video_xt` must have shape [B, C, T', H', W'], got {tuple(video_xt.shape)}.")
        if state_xt.ndim != 3:
            raise ValueError(f"`state_xt` must have shape [B, L, D], got {tuple(state_xt.shape)}.")
        if state_xt.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"State latent dim must equal hidden_dim={self.hidden_dim}, got {state_xt.shape[-1]}."
            )
        if condition_video.ndim != 5:
            raise ValueError(f"`condition_video` must have shape [B, T, C, H, W], got {tuple(condition_video.shape)}.")
        if condition_video.shape[1] != self.num_frames:
            raise ValueError(
                f"Expected condition video with {self.num_frames} frames, got {condition_video.shape[1]}."
            )
        if len(sequence_names) != state_xt.shape[0]:
            raise ValueError(
                f"Expected one sequence name per batch item, got batch={state_xt.shape[0]} and names={len(sequence_names)}."
            )

        teacher = self._get_video_teacher(video_xt.device)
        padded_condition_video = self._pad_video_frames_for_teacher(condition_video)
        masked_video_xt = self.apply_video_valid_mask(video_xt)
        if condition_latents is None:
            condition_latents = self.encode_video_target(condition_video[:, :, :3])
        else:
            condition_latents = self.apply_video_valid_mask(condition_latents)
        teacher_output = teacher(
            video_xt=masked_video_xt,
            timesteps=timesteps,
            condition_video=padded_condition_video,
            sequence_names=sequence_names,
            object_categories=object_categories,
            condition_latents=condition_latents,
            video_xt_human=video_xt_human,
            video_xt_object=video_xt_object,
        )
        if teacher_output.hidden_tokens.shape[-1] != self._wan_hidden_dim:
            raise RuntimeError(
                f"Wan teacher token dim changed from configured {self._wan_hidden_dim} to "
                f"{teacher_output.hidden_tokens.shape[-1]}."
            )

        time_cond = self.time_embed(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1)
        teacher_tokens = self.teacher_token_adapter(teacher_output.hidden_tokens.to(dtype=state_xt.dtype))
        state_tokens = state_xt + time_cond
        condition_tokens = self.condition_encoder(condition_video.to(dtype=state_xt.dtype)) + time_cond

        decoded_state = self.decode_state_tokens(state_tokens)
        geometry_aux = self.project_geometry(decoded_state, camera_intrinsics)
        geometry_tokens = self.geometry_encoder(geometry_aux["geometry_maps"].to(dtype=state_xt.dtype)) + time_cond

        for block in self.blocks:
            state_tokens = block(
                state_tokens,
                state_codec=self.state_codec,
                teacher_tokens=teacher_tokens,
                condition_tokens=condition_tokens,
                geometry_tokens=geometry_tokens,
                video_to_state_scale=video_to_state_scale,
            )
            decoded_state = self.decode_state_tokens(state_tokens)
            geometry_aux = self.project_geometry(decoded_state, camera_intrinsics)
            geometry_tokens = self.geometry_encoder(geometry_aux["geometry_maps"].to(dtype=state_xt.dtype)) + time_cond

        state_velocity = self.state_velocity_head(self.state_norm(state_tokens))
        video_velocity = self.apply_video_valid_mask(teacher_output.velocity)
        human_video_velocity = None
        if teacher_output.human_velocity is not None:
            human_video_velocity = self.apply_video_valid_mask(teacher_output.human_velocity)
        object_video_velocity = None
        if teacher_output.object_velocity is not None:
            object_video_velocity = self.apply_video_valid_mask(teacher_output.object_velocity)
        return DualBranchFMOutput(
            video_velocity=video_velocity,
            state_velocity=state_velocity,
            geometry_maps=geometry_aux["geometry_maps"],
            decoded_state=decoded_state,
            human_video_velocity=human_video_velocity,
            object_video_velocity=object_video_velocity,
        )
