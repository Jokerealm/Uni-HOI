# Loss 设计方案

## 总体原则

将训练拆分为两个阶段，各阶段的 loss 职责严格分离：

- **Stage 1（Dual-Branch Flow Matching）**：学习从噪声到真实数据分布的向量场。只保留 flow matching 主目标 + decode head 的轻量像素监督 + 关键 3D 结构回归。不引入任何可微渲染、几何投影或物理约束。
- **Stage 2（Interaction-Aware Residual Refinement）**：在 Stage 1 生成的高质量初始解上，通过 test-time optimization 进行可微渲染、接触约束、穿模惩罚等精细调整。

---

## Stage 1：Dual-Branch Flow Matching

### 保留的 6 项 Loss

#### 1. Video Flow Matching Loss $\mathcal{L}_{fm}^{v}$

$$\mathcal{L}_{fm}^{v} = \|\hat{u}_v - u_v^{\star}\|_2^2$$

- **作用**：video branch 的核心训练目标。监督网络预测的 velocity field 与 ground-truth velocity 对齐，驱动 video latent 从噪声流向真实分布。
- **权重**：`lambda_video_fm = 1.0`
- **全程参与训练**

#### 2. State Flow Matching Loss $\mathcal{L}_{fm}^{s}$

$$\mathcal{L}_{fm}^{s} = \|\hat{u}_s - u_s^{\star}\|_2^2$$

- **作用**：state branch 的核心训练目标。与 video FM loss 对称，监督 HOI state latent 的向量场学习。
- **权重**：`lambda_state_fm = 1.0`
- **全程参与训练**

#### 3. Human Visible Loss $\mathcal{L}_{h}^{vis}$

$$\mathcal{L}_{h}^{vis} = \|\hat{V}_h - V_h\|_{1, M^h}$$

- **作用**：监督 video branch decode head 输出的 human stream 在可见区域与 visibility-preserving target 对齐。这是 decode head 的直接梯度来源——没有它，token → pixel 的映射只能靠 FM loss 的间接梯度，收敛慢且容易出现 pixel artifacts。
- **权重**：`lambda_human_visible = 1.0`
- **全程参与训练**
- **为什么不能砍**：本模型的 decode head 不是预训练冻结的 VAE decoder（如 Stable Diffusion），而是模型的可训练组件，需要显式像素级监督。

#### 4. Object Video Loss $\mathcal{L}_{o}^{img}$

$$\mathcal{L}_{o}^{img} = \|\hat{V}_o - V_o\|_{1, \bar{M}}$$

其中 $\bar{M} = \text{clip}(M^h + M^o, 0, 1)$。

- **作用**：监督 video branch decode head 输出的 object stream 与 teacher 渲染的 object video 对齐。与 human_visible 对称，为 object 侧的 decode head 提供直接梯度。
- **权重**：`lambda_object_video = 1.0`
- **全程参与训练**
- **为什么不能砍**：同上，decode head 需要显式监督。注意这里的 target $V_o$ 是预计算的 teacher rendering，训练时不调用可微渲染器，所以没有渲染梯度的问题。

#### 5. Joints Loss $\mathcal{L}_{joint}$

$$\mathcal{L}_{joint} = \text{SmoothL1}(\hat{J}^{1:T}, J^{1:T})$$

- **作用**：监督 state branch 预测的 3D human joints 与 ground-truth 对齐。这是 state branch 中人体运动的核心约束，也是 3D-to-2D geometry injection 的上游信号——joints 准确了，投影到 video lattice 的 geometry tensor 才有意义。
- **权重**：`lambda_joints = 1.0`
- **全程参与训练**

#### 6. Object Motion Loss $\mathcal{L}_{motion}$

$$\mathcal{L}_{motion} = \|\hat{\mathbf{t}} - \mathbf{t}\|_1 + d_{geo}(\hat{R}, R)$$

其中 $d_{geo}(\hat{R}, R) = \frac{1}{\pi}\arccos\left(\frac{\text{tr}(\hat{R}^\top R) - 1}{2}\right)$。

- **作用**：监督 state branch 预测的 object rigid trajectory（平移 + 旋转）。平移用 L1，旋转用 geodesic distance，避免了欧拉角/四元数的不连续性问题。
- **权重**：`lambda_object_motion = 0.5`（旋转 geodesic 和平移 L1 数量级不同，0.5 是合理的初始值）
- **全程参与训练**

### 砍掉的 11 项 Loss 及理由

| Loss | 公式 | 砍掉理由 |
|------|------|---------|
| $\mathcal{L}_{lat}^{v}$ (video latent recon) | $\text{SmoothL1}(\hat{z}_1^v, z_1^v)$ | 与 $\mathcal{L}_{fm}^v$ 重复约束（double-dipping）。$t \to 0$ 时 $\hat{z}_1$ 误差极大，产生 misleading gradients |
| $\mathcal{L}_{lat}^{s}$ (state latent recon) | $\text{SmoothL1}(\hat{z}_1^s, z_1^s)$ | 同上 |
| $\mathcal{L}_{o}^{render}$ (object render) | $\|\mathcal{R}_{gs}(\hat{G}_o, \hat{T}_o, K) - V_o\|_{1,\bar{M}}$ | 在 FM 前向中嵌入 diff rasterization，显存爆炸且早期去噪时 3D 参数混乱，渲染梯度无效 |
| $\mathcal{L}_{couple}$ (branch coupling) | $\|\hat{V}_o - \text{sg}(\mathcal{R}_{gs}(\cdot))\|_{1,M^h}$ | 依赖 $\mathcal{L}_{o}^{render}$ 的渲染结果，同上 |
| $\mathcal{L}_{G_h}$ (human gaussian) | $\text{SmoothL1}(\hat{G}_h, G_h)$ | 高维回归（14D × $N_h$ 点），容易与 FM velocity loss 梯度冲突。留给 Stage 2 photometric loss 隐式对齐 |
| $\mathcal{L}_{G_o}$ (object gaussian) | $\text{SmoothL1}(\hat{G}_o, G_o)$ | 同上 |
| $\mathcal{L}_{contact}$ | $\text{SmoothL1}(\hat{\mathbf{c}}, \mathbf{c})$ | 接触信号对噪声极度敏感，属于 Stage 2 的物理约束。如果 Stage 2 收敛困难，可考虑以极小权重（~0.05）加回作为 soft hint |
| $\mathcal{L}_{heat}$ (joint heatmap) | $\|M_{joint}^{pred} - M_{joint}^{gt}\|_1$ | 需要 geometry projection，属于显式 2D 投影 loss，应交给 Stage 2 |
| $\mathcal{L}_{sil}$ (object silhouette) | $\|M_{sil}^{pred} - M_{sil}^{gt}\|_1$ | 同上 |
| $\mathcal{L}_{depth}$ (object depth) | masked L1 on projected depth | 同上 |
| $\mathcal{L}_{distill}$ (geometry distill) | $\|M_{geo}^{pred} - M_{geo}^{teacher}\|_1$ | 同上 |

### Stage 1 总 Loss

$$\mathcal{L}_{Stage1} = \mathcal{L}_{fm}^{v} + \mathcal{L}_{fm}^{s} + \lambda_1 \mathcal{L}_{h}^{vis} + \lambda_2 \mathcal{L}_{o}^{img} + \lambda_3 \mathcal{L}_{joint} + \lambda_4 \mathcal{L}_{motion}$$

默认权重：$\lambda_1 = \lambda_2 = \lambda_3 = 1.0$，$\lambda_4 = 0.5$。

### Curriculum 策略

**不再需要 curriculum。** 6 项 loss 全部从第 0 步开始参与训练，没有分阶段引入的必要。原来的 20%/60% curriculum 是为了应对渲染和几何 loss 的尖锐梯度，现在这些 loss 已经全部移除。

### 训练时的计算节省

由于 `render_active = False` 且 `geometry_active = False`：
- 不调用 `DiffRasterizationLayer`（省显存 + 计算）
- 不调用 `model.project_geometry()`（省 geometry projection 计算）
- 不需要 downsample keypoint heatmaps / depth / object masks

---

## Stage 2：Interaction-Aware Residual Refinement

Stage 2 是 test-time optimization，在 Stage 1 生成的初始解上进行逐帧/逐序列的梯度优化。

### 保留的 5 项 Loss

#### 1. Multi-Region Photometric Loss $\mathcal{L}_{mr}$

$$\mathcal{L}_{mr} = (1 - \lambda_{ssim}) \frac{\sum_{\mathbf{p}} w(\mathbf{p}) \|R(\mathbf{p}) - I(\mathbf{p})\|_1}{\sum_{\mathbf{p}} w(\mathbf{p})} + \lambda_{ssim}(1 - \text{SSIM}(R, I))$$

其中 $w(\mathbf{p}) = w_v M^{obj}(\mathbf{p}) + w_p M^p(\mathbf{p}) + w_s M^s(\mathbf{p}) + \epsilon$。

- **作用**：通过可微渲染将 3D Gaussian 渲染到图像空间，与输入视频计算光度误差。三区域加权（visible / primary occlusion / secondary occlusion）让优化器聚焦于交互关键区域。
- **权重**：$w_v = 1.0$，$w_p = 0.3$，$w_s = 0.05$，$\lambda_{ssim} = 0.2$
- **此时可微渲染是安全的**：Stage 1 已经提供了高质量的 3D 初始解，渲染结果有意义，梯度有效。

#### 2. 3D Contact Loss $\mathcal{L}_{contact}^{3D}$

$$\mathcal{L}_{contact}^{3D} = \frac{1}{|\mathcal{H}|} \sum_{j \in \mathcal{H}} \min_i \|J_j^{world} - \mathbf{x}_{o,i}^{world}\|_2$$

- **作用**：最小化手部关节到最近物体 Gaussian 中心的距离，微调接触精度。
- **权重**：`lambda_contact = 0.5`

#### 3. 2D Projection Loss $\mathcal{L}_{proj}$

$$\mathcal{L}_{proj} = \frac{\sum_j \gamma_j \|\pi_K(J_j^{world}) - \hat{\mathbf{u}}_j\|_2^2}{\sum_j \gamma_j + \epsilon}$$

- **作用**：将 3D joints 投影到 2D，与 2D 检测器的关键点对齐。利用 Stage 1 中砍掉的几何先验，在这里强制 3D-2D 一致性。
- **权重**：`lambda_j2d = 0.1`

#### 4. Penetration Loss $\mathcal{L}_{pen}$

$$\mathcal{L}_{pen} = \frac{1}{N_o} \sum_{i=1}^{N_o} \max(0, -\text{SDF}_{body}(\mathbf{x}_{o,i}^{world}))$$

- **作用**：构建 SMPL body 的 volumetric SDF，惩罚穿入人体内部的 object Gaussians。解决穿模问题。
- **权重**：`lambda_pen = 1.0`

#### 5. Temporal Smoothness Loss $\mathcal{L}_{temp}$

$$\mathcal{L}_{temp} = \frac{1}{T-2} \sum_{t=2}^{T-1} \|\boldsymbol{\xi}_{t-1} - 2\boldsymbol{\xi}_t + \boldsymbol{\xi}_{t+1}\|_2^2$$

- **作用**：对 SE(3) pose 序列施加加速度惩罚，抑制高频抖动，保证时序平滑。
- **权重**：`lambda_acc = 0.5`

### Stage 2 总 Loss

$$\mathcal{L}_{Stage2} = \mathcal{L}_{mr} + \lambda_c \mathcal{L}_{contact}^{3D} + \lambda_{2d} \mathcal{L}_{proj} + \lambda_p \mathcal{L}_{pen} + \lambda_t \mathcal{L}_{temp}$$

---

## 与代码实现的对应关系

| 设计 | 代码位置 | 状态 |
|------|---------|------|
| Stage 1 core preset 定义 | `train_dual_branch_fm.py` L64 `CORE_LOSS_NAMES` | ✅ 已实现 |
| 非核心项权重置零 | `train_dual_branch_fm.py` L472 `build_curriculum_loss_weights()` | ✅ 已实现 |
| 渲染/几何计算短路 | `train_dual_branch_fm.py` L570 `render_active` / L578 `geometry_active` | ✅ 已实现 |
| 默认 preset = core | `scripts/train_dual_branch_fm.opt` L52 | ✅ 已实现 |
| train.sh 传递 --loss_preset | `scripts/train.sh` L211 | ✅ 已实现 |
| 启动时打印 active losses | `train_dual_branch_fm.py` L937 | ✅ 已实现 |
| Stage 2 refinement losses | `scripts/step4_joint_optimization.py` L579 `step4_training_step()` | ✅ 已实现 |

## 权重调优建议

训练初期（前 100 步）重点观察 wandb 中 6 项 loss 的 raw 值（不乘 lambda）：

- 如果 FM velocity loss 在 ~0.01 量级，pixel L1 在 ~0.1 量级 → 需要提高 `lambda_video_fm` / `lambda_state_fm`（如 5~10）
- 如果 joints smooth L1 很小（3D 坐标归一化后可能在 1e-3 量级）→ 需要提高 `lambda_joints`
- 如果 object_motion 中 geodesic rotation loss 远大于 translation L1 → 需要降低 `lambda_object_motion` 或分别设权

目标是让 6 项 loss 对 total loss 的贡献在同一数量级（不需要完全相等，但不应差超过 10x）。
