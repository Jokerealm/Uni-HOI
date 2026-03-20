# Pipeline

## 1. 系统角色划分

新的 Uni-HOI pipeline 不再是：

`2D 补全 -> 单帧 lifting -> 对齐 -> 3DGS 优化`

而是：

`Step1 几何先验 -> Step2 Dual-Branch Joint FM -> Step4 residual refinement`

其中：

- `Step1`
  - 负责输出稳定条件，不负责最终生成
  - 条件包括 `rgb / human mask / object mask / depth / M_p / M_s / M_object / keypoints`
- `Step2`
  - 是唯一主生成器
  - 联合生成 video latent 和 4D HOI state latent
- `Step3`
  - 不再是推理必经路径
  - 更适合作为训练期 teacher bootstrap 来源
- `Step4`
  - 只保留 residual refinement
  - 主要补 contact、projection、render consistency

## 2. 训练图

```text
Step1 assets + teacher GS
  |
  |-- condition_video --------------------> ConditionEncoder
  |-- human_visible + teacher object video -> VideoLatentCodec -> video_target_tokens
  |-- G_h / G_o / joints / object_pose / contact
       -> HOIStateCodec ------------------> state_target_tokens
  |
  |-- noise interpolation ----------------> video_xt / state_xt
  |
  |-- DualBranchCoGenerativeFlowMatching
        |
        |-- video branch: factorized spatial -> temporal attention
        |-- state branch: token transformer
        |-- 3D -> 2D: GeometryProjector -> geometry_tokens
        |-- 2D -> 3D: ProjectedVideoSampler -> dynamic/global state updates
        |
        --> video_velocity / state_velocity
  |
  |-- x1 reconstruction ------------------> decoded video / decoded state
  |
  |-- staged losses
        warmup: FM + latent + visible/object video + motion
        fusion: + render / gaussian / joint heat / silhouette / geometry distill
        full  : + contact / masked depth
```

## 3. 推理图

```text
Step1 assets only
  |
  |-- condition_video
  |-- camera_intrinsics
  |
  |-- sample video noise z_v ~ N(0, I)
  |-- sample state noise z_s ~ N(0, I)
  |
  |-- ODE integration in shared flow field
        dz_v / dt = F_v(z_v, z_s, cond, t)
        dz_s / dt = F_s(z_s, z_v, cond, t)
  |
  |-- decode
        -> human amodal video
        -> object amodal video
        -> G_h
        -> G_o
        -> joints_3d / object poses / contact
```

## 4. 当前代码落点

- 主模型：`model/dual_branch_cogenerative_fm.py`
- 训练入口：`train_dual_branch_fm.py`
- 推理入口：`infer_dual_branch_fm.py`
- Step2 pipeline wrapper：`pipeline/step2_dual_branch_flow_matching.py`
- 数据集：`dataset/dual_branch_fm_dataset.py`

## 5. 训练阶段建议

阶段 0，warmup：

- 只让模型先学会条件补全和基础状态回归
- 冻结 `video_block` 主干
- 仍然训练：
  - `video head`
  - `video/state codecs`
  - cross-branch adapters
  - state branch
- 激活：
  - `video_fm`
  - `state_fm`
  - `video_latent`
  - `state_latent`
  - `human_visible`
  - `object_video`
  - `joints`
  - `object_motion`

阶段 1，fusion：

- 开启跨分支几何约束
- 解冻顶层 `video_block`
- 当前默认只解冻最后 `2` 个 video blocks
- 逐步 ramp：
  - `object_render`
  - `branch_coupling`
  - `human_gaussian`
  - `object_gaussian`
  - `joint_heat`
  - `object_silhouette`
  - `geometry_distill`

阶段 2，full：

- 最后再加高不稳定项
- 逐步 ramp：
  - `contact`
  - `object_depth`
