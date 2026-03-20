# Innovation Points

## 1. 从级联系统改成联合流场

旧路径的问题：

- 2D 补全缺少 3D 约束，容易模糊
- 3D lifting 缺少视频时序约束，容易漂
- Step3/Step4 承担过多生成责任，Step2 只是前处理

新路径的核心变化：

- 把 video latent 和 4D state latent 放进同一个 Flow Matching 主干
- 联合定义：
  - `dz_v / dt = F_v(z_v, z_s, c, t)`
  - `dz_s / dt = F_s(z_s, z_v, c, t)`

## 2. 显式 4D HOI state 参数化

状态分支不再只预测单个 `G_o.pt`，而是显式建模：

- `G_h^can`
- `G_o^can`
- `J_h[1:T]`
- `T_o[1:T]`
- `C[1:T]`

好处：

- 3D branch 不再只是静态 object completion
- contact 和 motion 进入主干，不再只能在 Step4 里补救

## 3. 双向交互，不是单向条件

### 3D -> 2D

- 当前 state 先 decode 成显式几何
- 再经 `GeometryProjector` 投到 token grid
- 变成 `geometry_tokens` 去更新视频分支

这让视频补全知道：

- joints 在哪里
- object 轮廓在哪里
- object depth 如何分布
- 哪些区域存在 contact

### 2D -> 3D

- `ProjectedVideoSampler` 从 video token lattice 采样关节和帧级特征
- 这些特征回写到 dynamic/global state token

这让 3D state 能从视频里拿到：

- 时序外观证据
- 遮挡补全证据
- 帧间 motion cue

## 4. 时空分离注意力

默认 `P=16` 后：

- 每帧只有 `16 x 16 = 256` 个 patch token
- 总视频 token 为 `2048`

当前实现不在 `2048` 个 token 上直接做全 attention，而是：

- 先 spatial
- 再 temporal

好处：

- 显存压力更可控
- 更接近视频扩散模型里常见的时空解耦设计
- 更容易继续往更长序列扩展

## 5. FlashAttention 路径约束

注意力统一使用 `torch.nn.functional.scaled_dot_product_attention`。

原因：

- 这是当前代码里接入 PyTorch FlashAttention-2 路径的稳定方式
- 后面切换 `fp16/bf16` 时，不需要再重写 attention kernel

## 6. 条件补全而不是自由生成

Uni-HOI 本质是重建，不是无条件生成。

因此当前设计坚持：

- visible 区域由 Step1 先验锚定
- model 重点学习 occluded / amodal 部分
- 3D 几何和视频观测在联合流场里互相收缩

## 7. 课程式 loss，不把 15 个项同时砸进去

当前训练显式分三阶段：

- warmup
  - 先学 FM 主目标和基础 video/state 回归
- fusion
  - 再加几何蒸馏、render、一致性
- full
  - 最后加 depth 和 contact

## 8. 两阶段优化而不是一步全量联合训练

当前训练脚本里，视频主干被定义为每层 fusion block 内部的 `video_block`。

阶段 1：

- 冻结全部 `video_block`
- 保持这些模块可训练：
  - `video_codec`
  - `condition_encoder`
  - cross adapters
  - `video_velocity_head`
  - 全部 state modules

这样做的原因：

- 避免一开始 video denoiser 和 3D branch 同时大幅漂移
- 先让新加入的 3D/state/cross modules 学会围绕视频分支工作

阶段 2：

- 到设定步数后，只解冻顶层若干个 `video_block`
- 默认解冻最后 `2` 层

这样更接近 “先冻结视频 backbone，再部分联合训练” 的策略。

这比从第 0 步同时开全部 loss 更稳定，也更容易 debug。
