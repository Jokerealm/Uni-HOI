from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor

from .feature_model import FeatureModel
from .model_utils import compute_distance_transform

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class PointCloudProjectionModel(ModelMixin):
    
    def __init__(
        self,
        image_size: int,
        image_feature_model: str,
        use_local_colors: bool = True,
        use_local_features: bool = True,
        use_global_features: bool = False,
        use_mask: bool = True,
        use_distance_transform: bool = True,
        predict_shape: bool = True,
        predict_color: bool = False,
        process_color: bool = False,
        image_color_channels: int = 3,  # for the input image, not the points
        color_channels: int = 3,  # for the points, not the input image
        colors_mean: float = 0.5,
        colors_std: float = 0.5,
        scale_factor: float = 1.0,
        # Rasterization settings
        raster_point_radius: float = 0.0075,  # point size
        raster_points_per_pixel: int = 1,  # a single point per pixel, for now
        bin_size: int = 0,
            model_name=None,
            # additional arguments added by XH
            load_sample_init=False,
            sample_init_scale=1.0,
            test_init_with_gtpc=False,
            consistent_center=False, # from https://arxiv.org/pdf/2308.07837.pdf
            voxel_resolution_multiplier: int=1,
            predict_binary: bool=False, # predict a binary class label
            lw_binary: float=1.0,
            binary_training_noise_std: float=0.1,
            dm_pred_type: str='epsilon', # diffusion prediction type
            self_conditioning=False,
            **kwargs,

    ):
        super().__init__()
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.use_local_colors = use_local_colors
        self.use_local_features = use_local_features
        self.use_global_features = use_global_features
        self.use_mask = use_mask
        self.use_distance_transform = use_distance_transform
        self.predict_shape = predict_shape # default False
        self.predict_color = predict_color # default True
        self.process_color = process_color
        self.image_color_channels = image_color_channels
        self.color_channels = color_channels
        self.colors_mean = colors_mean
        self.colors_std = colors_std
        self.model_name = model_name
        print("PointCloud Model scale factor:", self.scale_factor, 'Model name:', self.model_name)
        self.predict_binary = predict_binary
        self.lw_binary = lw_binary
        self.self_conditioning = self_conditioning

        # Types of conditioning that are used
        self.use_local_conditioning = self.use_local_colors or self.use_local_features or self.use_mask
        self.use_global_conditioning = self.use_global_features
        self.kwargs = kwargs

        # Create feature model
        self.feature_model = FeatureModel(image_size, image_feature_model)

        # Input size
        self.in_channels = 3  # 3 for 3D point positions
        if self.use_local_colors: # whether color should be an input
            self.in_channels += self.image_color_channels
        if self.use_local_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_global_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_mask:
            self.in_channels += 2 if self.use_distance_transform else 1
        if self.process_color:
            self.in_channels += self.color_channels # point color added to input or not, default False
        if self.self_conditioning:
            self.in_channels += 3 # add self conditioning

        self.in_channels = self.add_extra_input_chennels(self.in_channels)

        if self.model_name == 'pc2-diff-ho-sepsegm':
            self.in_channels += 2 if self.use_distance_transform else 1

        # Output size
        self.out_channels = 0
        if self.predict_shape:
            self.out_channels += 3
        if self.predict_color:
            self.out_channels += self.color_channels
        if self.predict_binary:
            print("Output binary classification score!")
            self.out_channels += 1

        # Save rasterization settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(image_size, image_size),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )

    def add_extra_input_chennels(self, input_channels):
        return input_channels
    
    def denormalize(self, x: Tensor, /, clamp: bool = True):
        x = x * self.colors_std + self.colors_mean
        return torch.clamp(x, 0, 1) if clamp else x

    def normalize(self, x: Tensor, /):
        x = (x - self.colors_mean) / self.colors_std
        return x

    def get_global_conditioning(self, image_rgb: Tensor):
        global_conditioning = []
        if self.use_global_features:
            global_conditioning.append(self.feature_model(image_rgb, 
                return_cls_token_only=True))  # (B, D)
        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning

    def get_local_conditioning(self, image_rgb: Tensor, mask: Tensor):
        """
        compute per-point conditioning (像素对齐的局部条件特征)
        
        Parameters
        ----------
        image_rgb: (B, 3, H, W), 例如 (B, 3, 224, 224), 值域 0-1, 背景被 mask 遮蔽
        mask: (B, 1, H, W) 或 (B, 2, H, W) for h+o
        
        Returns
        -------
        local_conditioning: (B, D_cond, H, W)
            D_cond = 3(RGB) + D_feat(ViT特征,384) + C_mask(1或2) + C_dt(1或2) 
            例如: 3 + 384 + 1 + 1 = 389 (单mask), 或 3 + 384 + 2 + 2 = 391 (h+o mask)
        """
        local_conditioning = []
        # import pdb; pdb.set_trace()

        if self.use_local_colors: # XH: default True
            local_conditioning.append(self.normalize(image_rgb))  # (B, 3, H, W) ImageNet 标准化后的 RGB
        if self.use_local_features: # XH: default True
            local_conditioning.append(self.feature_model(image_rgb))  # (B, D_feat, H, W), 例如 (B, 384, 224, 224)
        if self.use_mask: # default True
            local_conditioning.append(mask.float())  # (B, C_mask, H, W), C_mask=1 或 2
        if self.use_distance_transform: # default True
            if not self.use_mask:
                raise ValueError('No mask for distance transform?')
            if mask.is_floating_point():
                mask = mask > 0.5
            local_conditioning.append(compute_distance_transform(mask))  # (B, C_mask, H, W), 距离变换
        local_conditioning = torch.cat(local_conditioning, dim=1)  # (B, D_cond, H, W)
        return local_conditioning

    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection(
        self, points: Tensor, camera: CamerasBase, local_features: Tensor,
    ):
        """
        将 2D 局部特征通过光栅化投影到 3D 点上 (pixel-aligned feature projection)
        
        Parameters
        ----------
        points         : (B, N, 3)       — 3D 点坐标
        camera         : CamerasBase     — 相机参数
        local_features : (B, D_cond, H, W) — 2D 局部条件特征
        
        Returns
        -------
        local_features_proj : (B, N, D_cond) — 每个 3D 点对应的 2D 特征
        """
        B, C, H, W, device = *local_features.shape, local_features.device
        # C = D_cond (例如 389), H = W = 224
        R = self.raster_settings.points_per_pixel  # 默认 1
        N = points.shape[1]  # 点数
        
        # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
        camera = camera.clone()
        camera.T = camera.T * self.scale_factor  # (B, 3)

        # Create rasterizer
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings)

        # Associate points with features via rasterization
        fragments = rasterizer(Pointclouds(points))  # fragments.idx: (B, H, W, R)
        fragments_idx: Tensor = fragments.idx.long()  # (B, H, W, R), 每个像素对应的点索引, -1 表示无点
        visible_pixels = (fragments_idx > -1)  # (B, H, W, R), bool mask
        points_to_visible_pixels = fragments_idx[visible_pixels]  # (num_visible,), 可见点的索引

        # Reshape local features to (B, H, W, R, C)
        local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, D_cond)

        # Get local features corresponding to visible points
        local_features_proj = torch.zeros(B * N, C, device=device)  # (B*N, D_cond)
        # local feature includes: raw RGB color, image features, mask, distance transform
        local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]  # 填充可见点的特征
        local_features_proj = local_features_proj.reshape(B, N, C)  # (B, N, D_cond)
        
        return local_features_proj  # (B, N, D_cond)
    
    def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
        """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
        points = pc.points_padded() * (self.scale_factor if scale else 1)
        if self.predict_color and pc.features_padded() is not None: # normalize color, not point locations
            colors = self.normalize(pc.features_padded()) if normalize else pc.features_padded()
            return torch.cat((points, colors), dim=2)
        else:
            return points
    
    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            assert x.shape[2] == 3
            return Pointclouds(points=points)

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        t: Optional[Tensor],
    ):
        """
        提取图像局部特征并投影到 3D 点上, 拼接为扩散模型的输入
        
        Parameters
        ----------
        x_t       : (B, N, 3) 或 (B, N, 3+C_color) — 当前时间步的噪声点云
        camera    : CamerasBase — 相机参数
        image_rgb : (B, 3, H, W) — 输入 RGB 图像 (背景已 mask)
        mask      : (B, C_mask, H, W) — 分割掩码, C_mask=1 或 2
        t         : (B,) — 扩散时间步
        
        Returns
        -------
        x_t_input : (B, N, D_total)
            D_total = 3(坐标) + D_cond(局部特征) [+ D_global(全局特征)]
            典型值: 3 + 389 = 392 (单mask), 或 3 + 391 = 394 (h+o mask)
            对于 ho-sepsegm 模型: 额外 +2(距离变换) 或 +1(mask)
        """
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations (and colors if and only if predicting color)
        x_t_input = self.get_coord_feature(x_t)  # [(B, N, 3)] 或 [(B, N, 3+C_color)]

        # Local conditioning
        if self.use_local_conditioning:

            # Get local features and check that they are the same size as the input image
            local_features = self.get_local_conditioning(image_rgb=image_rgb, mask=mask)  # (B, D_cond, H, W)
            if local_features.shape[-2:] != image_rgb.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {image_rgb.shape=}')
            
            # Project local features. Here that we only need the point locations, not colors
            local_features_proj = self.surface_projection(points=x_t[:, :, :3],
                camera=camera, local_features=local_features)  # (B, N, D_cond)

            x_t_input.append(local_features_proj)  # 追加 (B, N, D_cond)

        # Global conditioning
        if self.use_global_conditioning: # False

            # Get and repeat global features
            global_features = self.get_global_conditioning(image_rgb=image_rgb)  # (B, D_global)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, D_global)

            x_t_input.append(global_features)

        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D_total)

        return x_t_input  # (B, N, D_total)

    def get_coord_feature(self, x_t):
        """get coordinate feature, for model that uses separate model to predict binary, we use first 3 channels only"""
        x_t_input = [x_t]
        return x_t_input

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()
