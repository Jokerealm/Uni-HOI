# Dual-Branch Co-Generative Flow Matching 方法与代码文档

## 1. 文档范围

这份文档记录当前仓库里“新主干”方法的真实代码落点，而不是旧版 HDM / 级联式 `Step2 -> Step3 -> Step4` 的论文路径。

本文档覆盖的主实现文件：

| 文件 | 角色 | 主要内容 |
|---|---|---|
| `configs/config.yaml` | 统一配置入口 | 配置 dual-branch 训练默认值与 legacy pipeline 默认值 |
| `legacy_pipeline.py` | legacy 全流程入口 | Step1~Step5 调度；Step2 可走双分支 FM；Step3 在已有 `gs_init_combined.pt` 时会自动跳过 |
| `configs/step2_config.py` | Step2 配置 dataclass | 定义 Step2 wrapper 的输入参数 |
| `configs/step3_config.py` | FM 推理配置 dataclass | 提供 Step2 FM 推理时共用的配置容器 |
| `pipeline/step2_dual_branch_flow_matching.py` | Step2 wrapper | 把 Hydra 配置转成 `infer_dual_branch_fm.py` 的参数 |
| `dataset/dual_branch_fm_dataset.py` | 训练数据契约 | 读取 Step1 结果、teacher 3DGS、组装 clip、返回 batch 字典 |
| `model/dual_branch_cogenerative_fm.py` | 主模型 | 双分支视频/状态 Flow Matching、几何投影、跨分支交互 |
| `model/joint_renderer_loss.py` | 可微渲染器 | 用 object Gaussian + pose 渲染 teacher object video |
| `train_dual_branch_fm.py` | 训练入口 | 数据加载、curriculum、FM 训练、loss 计算、checkpoint |
| `infer_dual_branch_fm.py` | 单序列推理 | ODE 采样、输出 amodal video 与 4D HOI state |
| `scripts/train.sh` | 训练脚本 | 预处理 ProciGen GT + 启动训练 |
| `scripts/test.sh` | 测试脚本 | 预处理单序列 + 单序列推理 |

下面默认描述的“模型方法”，指的是：

`Step1 预处理资产 -> Step2 Dual-Branch Co-Generative Flow Matching -> 可选 Step4 refinement`

其中 Step2 是当前主生成器。

## 2. 方法总览

### 2.1 核心思想

当前方法不再把“视频补全”和“3D lifting”拆成完全串联的两个生成器，而是把它们统一进一个双分支 Flow Matching 模型：

1. 视频分支
   生成 2 路 amodal video：
   - human amodal video
   - object amodal video
2. 状态分支
   生成 4D HOI state：
   - human Gaussian set
   - object Gaussian set
   - joints 3D trajectory
   - object pose trajectory
   - contact signature
3. 双向耦合
   - `3D -> 2D`：把当前状态分支解码出的 3D 几何投影回视频 token 网格
   - `2D -> 3D`：从视频 token 网格采样动态特征去更新状态 token

### 2.2 与主流程的关系

- Step1：负责输出稳定条件，不负责生成最终结果
- Step2：双分支 FM 主模型，直接输出 amodal video + 4D state
- Step3：如果 Step2 已经写出了 `gs_init_combined.pt`，`legacy_pipeline.py` 会跳过 Step3
- Step4：继续用联合 3DGS 优化做 refinement，而不是主生成

## 3. 数据契约

### 3.1 训练依赖的文件

`dataset/dual_branch_fm_dataset.py` 约定每个 sequence 至少有以下文件：

| 路径 | 作用 | 主要 shape |
|---|---|---|
| `processed/cropped/rgb/*.png` | 裁剪后的 RGB 序列 | 每帧 `[3,H,W]` |
| `processed/cropped/masks_raw.npz` | human/object mask | `human:[T,H,W]`，`object:[T,H,W]` |
| `processed/cropped/region_masks.npz` | 区域 mask | `M_p/M_s/M_object:[T,H,W]` |
| `processed/cropped/depth_aligned.npz` | 对齐后的深度 | `depth:[T,H,W]` |
| `processed/cropped/meta.npz` | ROI 相机内参 | `fx/fy/cx/cy:[T]` |
| `processed/cropped/keypoints_2d.npz` | 2D 关键点 | `keypoints:[T,J,2/3]` |
| `processed/smpl_params.npz` | SMPL/SMPL-H 伪 GT | 常用 `body_pose/cam_t/joints_3d/vertices/faces` |
| `processed/object_poses.npz` 或 `t*.000/*/fit01/*_fit.pkl` | 物体每帧位姿 | `[T,4,4]` |
| `gs_init/G_o.pt` 或 `gs_init/gs_init_combined.pt` | object Gaussian teacher | `[No,14]` |
| `gs_init/G_h.pt` 或 `processed/smpl_params.npz` | human Gaussian teacher / SMPL mesh | `[Nh,14]` |

### 3.2 Gaussian token 的 14 维定义

当前仓库统一把一个 Gaussian token 表示为：

`[xyz(3), quat(4), scale(3), opacity(1), sh_rgb(3)]`

所以单个 Gaussian token shape 是 `[14]`。

### 3.3 `load_dual_branch_sequence_bundle()` 输出

`dataset/dual_branch_fm_dataset.py:load_dual_branch_sequence_bundle()` 返回一个 sequence 级别的 bundle，关键字段如下：

| 字段 | Shape | 说明 |
|---|---|---|
| `masks_human` | `[T,1,H,W]` | human visible mask |
| `masks_object` | `[T,1,H,W]` | object visible mask |
| `m_primary` | `[T,1,H,W]` | primary occlusion region |
| `m_secondary` | `[T,1,H,W]` | secondary occlusion region |
| `m_object_region` | `[T,1,H,W]` | object region |
| `depth` | `[T,1,H,W]` | aligned depth |
| `intrinsics` | `[T,3,3]` | ROI camera intrinsics |
| `keypoints_2d` | `[T,J,3]` | `(x,y,conf)` |
| `keypoint_heatmaps` | `[T,1,H,W]` | 2D joints rasterized heatmap |
| `joints_3d` | `[T,J,3]` | 3D joints |
| `object_poses` | `[T,4,4]` | object transform per frame |
| `human_gaussians` | `[Nh,14]` | human Gaussian teacher / pseudo-GT |
| `object_gaussians` | `[No,14]` | object Gaussian teacher |
| `contact_signature` | `[T,Cc]` | 默认 `Cc=4` |

### 3.4 `DualBranchHOIDataset.__getitem__()` 输出

`DualBranchHOIDataset` 会把一个 sequence 切成 clip。`__getitem__()` 返回的是“无 batch 维”的 clip：

| 字段 | Shape |
|---|---|
| `rgb` | `[T,3,H,W]` |
| `human_visible` | `[T,3,H,W]` |
| `masks_human` | `[T,1,H,W]` |
| `masks_object` | `[T,1,H,W]` |
| `m_primary` | `[T,1,H,W]` |
| `m_secondary` | `[T,1,H,W]` |
| `m_object_region` | `[T,1,H,W]` |
| `depth` | `[T,1,H,W]` |
| `keypoints_2d` | `[T,J,3]` |
| `keypoint_heatmaps` | `[T,1,H,W]` |
| `joints_3d` | `[T,J,3]` |
| `camera_intrinsics` | `[T,3,3]` |
| `object_poses` | `[T,4,4]` |
| `contact_signature` | `[T,Cc]` |
| `human_gaussians` | `[Nh,14]` |
| `object_gaussians` | `[No,14]` |

经 `DataLoader` 默认 collate 后，训练时会变成 batch-first：

- `rgb -> [B,T,3,H,W]`
- `human_gaussians -> [B,Nh,14]`
- `object_poses -> [B,T,4,4]`

## 4. 训练时的真实输入定义

### 4.1 condition video

`train_dual_branch_fm.py` 里训练时的条件输入为：

```text
condition_video = concat(
  rgb[3],
  masks_human[1],
  masks_object[1],
  depth[1],
  m_primary[1],
  m_secondary[1],
  m_object_region[1],
  keypoint_heatmaps[1]
)
```

因此：

- `condition_video.shape = [B,T,10,H,W]`
- `condition_channels = 10`

### 4.2 视频分支 target

训练时视频目标不是直接拿两路 RGB GT，而是：

1. `human_visible`
   - 由 `rgb * masks_human + background * (1 - masks_human)` 构造
2. `teacher_object_video`
   - 由 `DiffRasterizationLayer` 把 `object_gaussians + object_poses + intrinsics` 渲染出来

然后拼成：

- `video_target = cat([human_visible, teacher_object_video], dim=2)`
- `video_target.shape = [B,T,6,H,W]`

其中 6 个通道的语义是：

- `0:3` human branch RGB
- `3:6` object branch RGB

### 4.3 状态分支 target

状态分支 target 来自：

| 变量 | Shape |
|---|---|
| `human_gaussians` | `[B,Nh,14]` |
| `object_gaussians` | `[B,No,14]` |
| `joints_3d` | `[B,T,J,3]` |
| `object_poses` | `[B,T,4,4]` |
| `contact_signature` | `[B,T,Cc]` |

## 5. 默认 shape 记号

默认参数来自 `train_dual_branch_fm.py`：

- `B`: batch size
- `T = 8`
- `H = W = 256`
- `P = 16`
- `h = H / P = 16`
- `w = W / P = 16`
- `D = 512`
- `Nh = 1024`
- `No = 1024`
- `J = 22`
- `Cc = 4`

所以：

- 视频 token 长度  
  `L_v = T * h * w = 8 * 16 * 16 = 2048`
- 状态 token 长度  
  `L_s = Nh + No + T*J + T + T = 1024 + 1024 + 176 + 8 + 8 = 2240`

## 6. 主模型结构与每层输入输出

主模型文件：`model/dual_branch_cogenerative_fm.py`

### 6.1 `VideoLatentCodec`

作用：

- 把视频 patchify 成 token
- 或者把 token unpatchify 回视频

#### `encode(video)`

输入：

- `video: [B,T,C,H,W]`

输出：

- `tokens: [B,L_v,D]`

其中：

- `C=6` 时用于视频 target 编码
- `C=10` 时通过 `ConditionEncoder` 用于条件编码

#### `decode(tokens)`

输入：

- `tokens: [B,L_v,D]`

输出：

- `video: [B,T,C,H,W]`

### 6.2 `ConditionEncoder`

本质上是一个 `VideoLatentCodec(channels=10)` 加 `LayerNorm`：

- 输入：`condition_video [B,T,10,H,W]`
- 输出：`condition_tokens [B,L_v,D]`

### 6.3 `HOIStateCodec`

作用：

- 把结构化 HOI state 编成 token
- 或把 state token 解码回结构化状态

#### `encode_targets(...)`

输入：

| 输入 | Shape |
|---|---|
| `human_gaussians` | `[B,Nh,14]` |
| `object_gaussians` | `[B,No,14]` |
| `joints_3d` | `[B,T,J,3]` |
| `object_transforms` | `[B,T,4,4]` |
| `contact_signature` | `[B,T,Cc]` |

输出：

- `state_tokens: [B,L_s,D]`

内部 token 拆分顺序固定为：

1. human Gaussian tokens：`[B,Nh,D]`
2. object Gaussian tokens：`[B,No,D]`
3. joint tokens：`[B,T*J,D]`
4. object motion tokens：`[B,T,D]`
5. contact tokens：`[B,T,D]`

#### `decode_tokens(tokens)`

输入：

- `tokens: [B,L_s,D]`

输出：`DecodedHOIState`

| 字段 | Shape |
|---|---|
| `human_gaussians` | `[B,Nh,14]` |
| `object_gaussians` | `[B,No,14]` |
| `joints_3d` | `[B,T,J,3]` |
| `object_transforms` | `[B,T,4,4]` |
| `contact_signature` | `[B,T,Cc]` |

注意：

- quaternion 会被归一化
- scale 会经过 `softplus + 1e-6`
- opacity 与 `sh_rgb` 会经过 `sigmoid`
- `object_transforms` 内部只预测前 3x4，共 12 维，再补最后一行 `[0,0,0,1]`

### 6.4 `GeometryProjector`

作用：

- 把当前解码出的 3D state 投影到视频 token 网格上

输入：

| 输入 | Shape |
|---|---|
| `decoded_state` | 结构化状态 |
| `camera_intrinsics` | `[B,T,3,3]` |

输出：

| 输出 | Shape | 说明 |
|---|---|---|
| `geometry_maps` | `[B,T,5,h,w]` | 3D -> 2D 几何特征图 |
| `joint_coords` | `[B,T,J,2]` | joints 投影坐标 |
| `object_centers` | `[B,T,2]` | object 投影中心 |

`geometry_maps` 的 5 个通道顺序：

1. `joint_heat`
2. `joint_depth`
3. `object_silhouette`
4. `object_depth`
5. `contact_heat`

### 6.5 `GeometryMapEncoder`

作用：

- 把 `geometry_maps [B,T,5,h,w]` 映射到和视频分支同长度的 token lattice

输入：

- `geometry_maps: [B,T,5,h,w]`

输出：

- `geometry_tokens: [B,L_v,D]`

这里虽然名字仍复用了 `VideoLatentCodec`，但它的 patch size 固定为 1，因为输入已经在 token 网格分辨率上。

### 6.6 `ProjectedVideoSampler`

作用：

- 从视频 token 网格里采样特征，反哺 state 分支

输入：

| 输入 | Shape |
|---|---|
| `video_tokens` | `[B,L_v,D]` |
| `geometry_aux["joint_coords"]` | `[B,T,J,2]` |

内部先 reshape：

- `feature_map = [B,T,h,w,D]`
- `frame_summaries = mean(feature_map over h,w) = [B,T,D]`

输出两类上下文：

| 输出 | Shape | 含义 |
|---|---|---|
| `global_video_context` | `[B,Nh+No,D]` | 给全局 Gaussian token |
| `dynamic_video_context` | `[B,T*J + T + T,D]` | 给 joints / motion / contact token |

其中：

- `joint_features -> [B,T*J,D]`
- `motion_features -> [B,T,D]`
- `contact_features -> [B,T,D]`

### 6.7 `FactorizedVideoTransformerBlock`

这是视频分支的主 block。它不直接在 `L_v = T*h*w` 个 token 上做全局 self-attention，而是分两步：

1. spatial attention  
   输入视角：`[B*T, h*w, D]`
2. temporal attention  
   输入视角：`[B*h*w, T, D]`

默认形状下：

- spatial attention: `[B*8, 256, 512]`
- temporal attention: `[B*256, 8, 512]`

这种设计比直接对 `[B,2048,512]` 做全自注意力更省显存。

### 6.8 `DualBranchFusionBlock`

一个 fusion block 里同时更新视频和状态：

输入：

| 输入 | Shape |
|---|---|
| `video_tokens` | `[B,L_v,D]` |
| `state_tokens` | `[B,L_s,D]` |
| `condition_tokens` | `[B,L_v,D]` |
| `geometry_tokens` | `[B,L_v,D]` |
| `global_video_context` | `[B,Nh+No,D]` |
| `dynamic_video_context` | `[B,T*J+T+T,D]` |

执行顺序：

1. `video_block`：视频分支自注意力
2. `state_block`：状态分支自注意力
3. `video_from_condition`：条件到视频
4. `video_from_geometry`：几何到视频
5. `video_from_state`：状态到视频
6. `global/dynamic gate`：视频上下文更新状态 token
7. `state_from_video`：视频回写状态

输出：

- `video_tokens: [B,L_v,D]`
- `state_tokens: [B,L_s,D]`

### 6.9 `DualBranchCoGenerativeFlowMatching.forward`

这是训练和推理共用的核心接口。

输入：

| 输入 | Shape |
|---|---|
| `video_xt` | `[B,L_v,D]` |
| `state_xt` | `[B,L_s,D]` |
| `timesteps` | `[B]` |
| `condition_video` | `[B,T,10,H,W]` |
| `camera_intrinsics` | `[B,T,3,3]` |

输出：`DualBranchFMOutput`

| 字段 | Shape |
|---|---|
| `video_velocity` | `[B,L_v,D]` |
| `state_velocity` | `[B,L_s,D]` |
| `geometry_maps` | `[B,T,5,h,w]` |
| `decoded_state.human_gaussians` | `[B,Nh,14]` |
| `decoded_state.object_gaussians` | `[B,No,14]` |
| `decoded_state.joints_3d` | `[B,T,J,3]` |
| `decoded_state.object_transforms` | `[B,T,4,4]` |
| `decoded_state.contact_signature` | `[B,T,Cc]` |

forward 流程：

1. `video_xt/state_xt` 加 time embedding
2. `condition_video -> condition_tokens`
3. `state_xt -> decode_state_tokens`
4. `decoded_state + K -> geometry_maps -> geometry_tokens`
5. 进入多个 `DualBranchFusionBlock`
6. 每个 block 后重新 decode state，再重新投影 geometry
7. 最后输出两路 velocity

## 7. Flow Matching 训练公式

训练入口：`train_dual_branch_fm.py`

### 7.1 噪声插值

对视频和状态分支都采用同样的 FM 目标。

设：

- `x1_v = video_target_tokens`
- `x1_s = state_target_tokens`
- `z_v ~ N(0, I)`
- `z_s ~ N(0, I)`
- `t ~ Uniform(0,1)`

则：

- `video_xt = t * x1_v + (1 - t) * z_v`
- `state_xt = t * x1_s + (1 - t) * z_s`

目标速度：

- `video_velocity_target = x1_v - z_v`
- `state_velocity_target = x1_s - z_s`

### 7.2 从速度恢复 `x1_hat`

代码里恢复方式为：

- `video_x1_hat = video_xt + (1 - t) * video_velocity_pred`
- `state_x1_hat = state_xt + (1 - t) * state_velocity_pred`

然后再 decode 出：

- `decoded_video`
- `decoded_state`

### 7.3 Loss 组成

`LOSS_NAMES` 里定义了 17 个 loss：

| loss 名称 | 监督对象 | shape 级别 |
|---|---|---|
| `video_fm` | 视频 velocity | `[B,L_v,D]` |
| `state_fm` | 状态 velocity | `[B,L_s,D]` |
| `video_latent` | 视频 latent 重建 | `[B,L_v,D]` |
| `state_latent` | 状态 latent 重建 | `[B,L_s,D]` |
| `human_visible` | human branch RGB | `[B,T,3,H,W]` |
| `object_video` | object branch RGB | `[B,T,3,H,W]` |
| `object_render` | 3DGS 渲染 object | `[B,T,3,H,W]` |
| `branch_coupling` | object video vs render 一致性 | `[B,T,3,H,W]` |
| `human_gaussian` | human Gaussian 参数 | `[B,Nh,14]` |
| `object_gaussian` | object Gaussian 参数 | `[B,No,14]` |
| `joints` | 3D joints | `[B,T,J,3]` |
| `object_motion` | object transforms | `[B,T,4,4]` |
| `contact` | contact signature | `[B,T,Cc]` |
| `joint_heat` | geometry channel 0 vs keypoint heatmap | `[B,T,1,h,w]` |
| `object_silhouette` | geometry channel 2 vs object mask | `[B,T,1,h,w]` |
| `object_depth` | geometry channel 3 vs depth | `[B,T,1,h,w]` |
| `geometry_distill` | 预测 geometry maps vs teacher geometry maps | `[B,T,5,h,w]` |

### 7.4 Curriculum

`build_curriculum_loss_weights()` 把训练分成三段：

1. Stage 0
   只开基础项：
   - `video_fm`
   - `state_fm`
   - `video_latent`
   - `state_latent`
   - `human_visible`
   - `object_video`
   - `joints`
   - `object_motion`
2. Stage 1
   从 `fusion_start_ratio` 开始逐步打开：
   - `object_render`
   - `branch_coupling`
   - `human_gaussian`
   - `object_gaussian`
   - `joint_heat`
   - `object_silhouette`
   - `geometry_distill`
3. Stage 2
   从 `full_start_ratio` 开始逐步打开：
   - `contact`
   - `object_depth`

此外视频 backbone 支持先冻结、后解冻顶部若干 block。

## 8. 推理流程

推理入口：`infer_dual_branch_fm.py`

### 8.1 输入

`load_inference_clip()` 会读取单个 sequence，并构造：

| 变量 | Shape |
|---|---|
| `condition_video` | `[1,T,10,H,W]` |
| `human_visible` | `[1,T,3,H,W]` |
| `object_visible` | `[1,T,3,H,W]` |
| `masks_human` | `[1,T,1,H,W]` |
| `masks_object` | `[1,T,1,H,W]` |
| `camera_intrinsics` | `[1,T,3,3]` |

### 8.2 潜变量初始化

推理时直接从高斯噪声开始：

- `video_latents: [1,L_v,D]`
- `state_latents: [1,L_s,D]`

### 8.3 ODE 更新

代码里使用简单的 Euler 形式离散积分：

```text
for k in range(num_ode_steps):
    v_dot, s_dot = model(...)
    video_latents = video_latents + dt * v_dot
    state_latents = state_latents + dt * s_dot
```

### 8.4 解码后的输出

- `decoded_video -> [1,T,6,H,W]`
- `decoded_state -> structured state`

再拆分：

- `pred_human = decoded_video[:,:,0:3]`
- `pred_object = decoded_video[:,:,3:6]`

若 `clamp_visible_rgb=True`，推理输出会保留可见区域的原图，只在遮挡区域使用预测：

- human branch 只在 `masks_object == 1` 的区域被替换
- object branch 只在 `masks_human == 1` 的区域被替换

## 9. 推理输出文件

`infer_dual_branch_fm.py` 会写出：

| 路径 | 内容 |
|---|---|
| `amodal/human_amodal/frames/*.png` | human amodal 每帧 |
| `amodal/human_amodal/inpaint_out.mp4` | human amodal 视频 |
| `amodal/object_amodal/frames/*.png` | object amodal 每帧 |
| `amodal/object_amodal/inpaint_out.mp4` | object amodal 视频 |
| `gs_init/G_h.pt` | human Gaussian tokens |
| `gs_init/G_o.pt` | object Gaussian tokens |
| `gs_init/gs_init_combined.pt` | 合并后的 4D HOI state |
| `gs_init/dual_branch_inference.json` | 序列名、帧数、ODE 步数等元信息 |

`gs_init_combined.pt` 的结构：

| key | 内容 |
|---|---|
| `G_h.raw` | `[Nh,14]` |
| `G_o.raw` | `[No,14]` |
| `motion.joints_3d` | `[T,J,3]` |
| `motion.object_poses` | `[T,4,4]` |
| `motion.contact_signature` | `[T,Cc]` |

## 10. 关键实现备注

### 10.1 训练和推理的 shape 来源不完全一样

- 训练 shape 由 `train_dual_branch_fm.py` 的命令行参数决定
- 推理时模型结构参数以 checkpoint 内保存的 `args` 为准
- `configs/step3_config.py` 里的 `FlowMatchingInferenceConfig` 更像 wrapper 配置容器，不是最终的结构真值

换句话说，Step2 wrapper 里传入的 `fm.num_frames / fm.hidden_dim / fm.patch_size` 并不会强制覆盖 checkpoint 的网络结构。

### 10.2 当前条件通道数固定为 10

`DualBranchHOIDataset.condition_channels` 直接返回 10，训练和推理都默认遵守下面这个顺序：

`rgb + human_mask + object_mask + depth + M_p + M_s + M_object + keypoint_heatmap`

如果以后增删条件通道，至少需要同时改：

- dataset 组装逻辑
- `condition_channels`
- checkpoint 兼容性

### 10.3 当前 object video teacher 来自可微渲染，不是直接 RGB GT

这意味着 object 分支学的是：

- “在当前 camera pose 下的 object amodal appearance/render”

而不是简单地复原被遮挡前的原图像素。

### 10.4 Step3 在双分支 FM 路径里是可跳过的

`legacy_pipeline.py` 的逻辑是：

- 如果 Step2 已经在 `gs_init/gs_init_combined.pt` 写出了联合 4D 状态
- 那么 Step3 的 Hunyuan3D lifting 可以直接跳过

这说明当前双分支 FM 已经被代码层面视为可直接产出 3D 初始化结果的主生成器。
