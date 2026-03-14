### 补充文档 A1：度量对齐桥接模块 (Metric Alignment Bridge)

**模块定位**：插入于 **Step 3**（Flow Matching 生成规范化 3DGS）与 **Step 4**（联合优化）之间的显式数学对齐层。本模块为确定性前向计算（非可学习），在 CPU/GPU 上均可执行，耗时可忽略。

**核心动机**：Step 3 的 Flow Matching 模型在归一化规范空间 (Normalized Canonical Space) 中生成 3DGS，其几何中心约在原点 $(0,0,0)$，包围球半径约为 $1.0$。而 Step 4 的联合优化需要在 Preprocess 阶段确立的真实物理度量空间 (Metric Space) 中工作——该空间的尺度为米、坐标原点为相机光心。若不做显式对齐，3DGS 优化器需要在巨大的参数空间中同时搜索正确的尺度、宏观平移和细节纹理，极易导致：
- 梯度爆炸或梯度消失（尺度差异可达 $10^1 \sim 10^2$ 倍）
- 陷入局部最优（物体坍缩为一个点或被拉扯成碎片）
- $L_{pen}$（穿模 Loss）和 $L_c$（接触 Loss）因空间错位而完全失效

**核心输入**：

| 数据 | 来源 | 格式 | 维度/说明 |
|---|---|---|---|
| 物体归一化 3DGS $G_o^{norm}$ | Step 3 输出 | `.npz` | `(N_o, 14)` — 规范空间 |
| 人体归一化 3DGS $G_h^{norm}$ | Step 3 输出 | `.npz` | `(N_h, 14)` — 规范空间 |
| 对齐深度图 $D_{align}$ | Preprocess 输出 | `.npz` | `(H, W)` float32, 单位：米 |
| 物体掩码 $M_{object}$ | Preprocess 输出 | `.png` | `(H, W)` uint8 二值 |
| 人物掩码 $M_{human}$ | Preprocess 输出 | `.png` | `(H, W)` uint8 二值 |
| 相机内参 $K$ | Preprocess 输出 | `.npz` | $3 \times 3$, 含 $f_x, f_y, c_x, c_y$ |
| SMPL-H 参数 $\mathcal{H}_i$ | Preprocess 输出 | `.npz` | body_pose, betas, transl 等 |

**模块输出**：

| 数据 | 格式 | 维度/说明 |
|---|---|---|
| 物体度量 3DGS $G_o^{metric}$ | `.npz` | `(N_o, 14)` — 度量空间 |
| 人体度量 3DGS $G_h^{metric}$ | `.npz` | `(N_h, 14)` — 度量空间 |
| 对齐元数据 `alignment_meta.npz` | `.npz` | 包含 $s, t, R_{obs}, R_{norm}$ 等中间量，供调试与回退 |

---

#### 第一阶段：提取真实物理观测表面的点云 (Observed Surface Unprojection)

仅依靠 2D Mask 的像素中心无法获得正确的 3D 定位。必须利用相机内参矩阵 $K$ 将掩码内的有效度量深度反投影为 3D 点云。

对于每一个属于掩码 $M$ 的像素 $(u, v)$，其真实 3D 坐标为：

$$Z = D_{align}(u, v)$$

$$X = \frac{(u - c_x) \cdot Z}{f_x}$$

$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$

分别对 $M_{object}$ 和 $M_{human}$ 执行上述反投影，得到：

- 物体观测点云 $\mathcal{P}_{obs}^{obj} \in \mathbb{R}^{N_1 \times 3}$
- 人体观测点云 $\mathcal{P}_{obs}^{hum} \in \mathbb{R}^{N_2 \times 3}$

**关键约束**：这些点云仅代表物体/人体的可见前表面，并非几何中心。后续的平移估算必须补偿这一偏差。

**深度有效性过滤**：反投影前必须过滤无效深度值：
- 剔除 $D_{align}(u,v) \leq 0$ 或 $D_{align}(u,v) > D_{max}$（默认 $D_{max} = 10.0$ 米）的像素
- 剔除掩码内有效像素数 $< N_{min}$（默认 $N_{min} = 50$）的帧，标记为退化帧并回退到纯 SE(3) 可学习对齐

---

#### 第二阶段：鲁棒的尺度估算与深度补偿 (Robust Scale Estimation & Depth Compensation)

设 Flow Matching 生成的归一化 3DGS 均值点云为 $\mathcal{P}_{norm} = \{\mu_i^{norm}\}_{i=1}^{N}$，其几何中心约在原点，包围球半径约为 $1.0$。

##### 2.1 鲁棒尺度因子 $s$

为避免离群点（深度噪声、掩码边缘泄漏）干扰，使用 90% 分位数估算点云在 X-Y 平面的投影半径：

$$R_{obs} = \text{Percentile}_{90}\left(\left\| \mathcal{P}_{obs}^{x,y} - \text{median}(\mathcal{P}_{obs}^{x,y}) \right\|_2\right)$$

$$R_{norm} = \text{Percentile}_{90}\left(\left\| \mathcal{P}_{norm}^{x,y} - \text{median}(\mathcal{P}_{norm}^{x,y}) \right\|_2\right)$$

尺度因子：

$$s = \frac{R_{obs}}{R_{norm}}$$

**数值安全守卫**：
- 若 $R_{norm} < \epsilon_{scale}$（默认 $\epsilon_{scale} = 10^{-4}$），说明生成的 3DGS 已坍缩，标记为退化并回退到默认尺度 $s_{default}$
- 对 $s$ 施加合理范围裁剪：$s \in [s_{min}, s_{max}]$（默认 $[0.01, 100.0]$），防止极端尺度

##### 2.2 考虑"表面-中心"几何厚度的平移估算 $t$

这是最容易遗漏的逻辑漏洞。观测点云 $\mathcal{P}_{obs}$ 仅为物体的可见前表面，而 $\mathcal{P}_{norm}$ 的几何中心位于物体体积中心。若直接将 $\mathcal{P}_{norm}$ 的中心对齐到 $\mathcal{P}_{obs}$ 的中心，会导致生成的 3DGS 有一半穿透真实深度面（离相机过近）。

正确做法：将 3DGS 的中心沿相机视线方向（Z 轴正方向，即远离相机）推移一个物体半径的厚度。

$$t_x = \text{median}(X_{obs})$$

$$t_y = \text{median}(Y_{obs})$$

$$t_z = \text{median}(Z_{obs}) + s \cdot R_{norm}^{z}$$

其中 $R_{norm}^{z}$ 是归一化模型在 Z 轴上的半厚度估计：

$$R_{norm}^{z} = \text{Percentile}_{90}\left(\left| \mathcal{P}_{norm}^{z} - \text{median}(\mathcal{P}_{norm}^{z}) \right|\right)$$

**注意**：此处使用绝对值的 90% 分位数而非标准差，以保持与 $R_{obs}$、$R_{norm}$ 一致的鲁棒统计口径。

##### 2.3 人体分支的特殊处理

人体的对齐可以采用两种策略（通过配置切换）：

- **策略 A（默认）**：与物体相同的反投影对齐流程，使用 $M_{human}$ 和 $D_{align}$ 独立计算 $s_h, t_h$
- **策略 B**：利用 SMPL-H 参数 $\mathcal{H}_i$ 中的 `transl` 字段直接获取人体在度量空间中的根关节位置，仅需估算尺度 $s_h$，平移直接取 SMPL-H 的 translation。此策略在 SMPL-H 估计质量较高时更为精确

推荐在配置中默认启用策略 B（`human_align_strategy: "smplh"`），因为 SAM3D-Body 的 translation 估计已经过深度配准，可信度高于纯深度反投影。

---

#### 第三阶段：3DGS 属性的仿射变换 (Affine Transformation of 3DGS Attributes)

在进入 Step 4 的 PyTorch 优化器之前，对 Flow Matching 输出的 3DGS 14 维属性进行一次确定性前向变换。

假设 Flow Matching 的输出已经是 View-centric（视点对齐的），旋转 $R$ 为单位阵，仅需各向同性缩放 + 平移。

##### 3.1 均值坐标 (Means, $\mu$, 通道 0-2)

$$\mu_{metric} = s \cdot \mu_{norm} + t$$

##### 3.2 旋转四元数 (Rotation Quaternion, 通道 3-6)

由于仅做各向同性缩放和平移（无坐标系旋转），旋转四元数保持不变：

$$q_{metric} = q_{norm}$$

##### 3.3 缩放参数 (Scales, 通道 7-9)

3DGS 源码中 scales 通常存储为 log-space 值（经 `exp` 激活后得到实际尺度）。各向同性缩放 $s$ 在 log-space 中等价于加法：

$$S_{metric} = S_{norm} + \ln(s)$$

##### 3.4 不透明度 (Opacity, 通道 10)

不透明度是与尺度无关的属性，保持不变：

$$\alpha_{metric} = \alpha_{norm}$$

##### 3.5 球谐系数 (SH Coefficients, 通道 11-13)

由于没有进行坐标系旋转（仅平移 + 各向同性缩放），球谐系数的方向依赖性不受影响：

$$SH_{metric} = SH_{norm}$$

**变换完整性校验**：变换后应验证：
- $\mu_{metric}$ 的 Z 分量均为正值（物体在相机前方）
- $\exp(S_{metric})$ 的值在合理范围内（如 $[10^{-5}, 10^{1}]$ 米）
- 不透明度 $\sigma(\alpha_{metric}) \in [0, 1]$（sigmoid 激活后）

---

#### 第四阶段：与 Step 4 SE(3) 模块的协作关系

本模块提供的是一个高质量的初始化，而非最终对齐。Step 4 中的可学习 SE(3) 变换模块仍然保留，但其角色从"从零搜索对齐"降级为"精细微调残差"：

- **无本模块时**：SE(3) 需要从随机初始化搜索 $\sim O(10)$ 量级的平移和 $\sim O(10)$ 量级的缩放，搜索空间巨大
- **有本模块后**：SE(3) 仅需学习 $\sim O(10^{-2})$ 量级的残差修正，收敛速度提升数个数量级

具体协作方式：
1. 本模块输出 $G^{metric}$ 作为 Step 4 的初始 3DGS 参数
2. Step 4 的 SE(3) 模块初始化为单位变换（$R = I, t = 0$），在 $G^{metric}$ 基础上学习残差
3. Step 4 的 SE(3) 学习率可相应降低（建议 `lr_translation` 降至 $5 \times 10^{-4}$，`lr_rotation` 降至 $5 \times 10^{-5}$）

---

#### 配置设计 (Hydra Config Integration)

##### Dataclass 定义

```python
@dataclass
class MetricAlignmentConfig:
    """度量对齐桥接模块配置"""
    enabled: bool = True
    # 深度有效性过滤
    depth_max: float = 10.0           # 最大有效深度 (米)
    depth_min_pixels: int = 50        # 掩码内最少有效深度像素数
    # 鲁棒统计
    percentile: float = 90.0          # 用于半径估算的分位数
    # 数值安全
    scale_eps: float = 1e-4           # R_norm 退化阈值
    scale_default: float = 1.0        # 退化时的默认尺度
    scale_min: float = 0.01           # 尺度裁剪下界
    scale_max: float = 100.0          # 尺度裁剪上界
    # 人体对齐策略
    human_align_strategy: str = "smplh"  # "smplh" | "unproject"
    # 变换后校验
    validate_transform: bool = True
    z_positive_check: bool = True     # 检查变换后 Z > 0
    scale_range_check: tuple = (1e-5, 10.0)  # exp(S) 的合理范围
```

##### YAML 配置 (追加至 `conf/config.yaml` 的 Step 3 与 Step 4 之间)

```yaml
# ============================================================
# Step 3.5: Metric Alignment Bridge (between Step 3 and Step 4)
# ============================================================
alignment:
  enabled: true
  depth_max: 10.0
  depth_min_pixels: 50
  percentile: 90.0
  scale_eps: 1.0e-4
  scale_default: 1.0
  scale_min: 0.01
  scale_max: 100.0
  human_align_strategy: smplh    # "smplh" or "unproject"
  validate_transform: true
```

---

#### 退化处理与回退策略 (Fallback Policy)

| 退化条件 | 检测方式 | 回退行为 |
|---|---|---|
| 掩码内有效深度像素不足 | $N_{valid} < $ `depth_min_pixels` | 跳过本模块，标记帧为 `degraded`，Step 4 使用纯 SE(3) 从默认初始化开始 |
| 归一化点云坍缩 | $R_{norm} < $ `scale_eps` | 使用 `scale_default`，日志输出警告 |
| 尺度因子超出合理范围 | $s \notin [s_{min}, s_{max}]$ | 裁剪至边界值，日志输出警告 |
| 变换后 Z 分量出现负值 | $\exists \mu_z^{metric} < 0$ | 将负值 Gaussian 的不透明度置零（软剔除），日志输出警告 |
| 相机内参缺失 | $K$ 文件不存在 | 使用 Step 4 配置中的 `focal` 构造默认内参 $K_{default}$，假设光心在图像中心 |

---

#### 输出目录结构

```
<input_dir>/<video_name>/gs_aligned/
├── human_gaussians_metric.npz     (N_h, 14) — 度量空间人体 3DGS
├── object_gaussians_metric.npz    (N_o, 14) — 度量空间物体 3DGS
└── alignment_meta.npz             — 对齐元数据 (s_h, t_h, s_o, t_o, R_obs, R_norm, ...)
```

---

#### 验证清单 (Sanity Checks)

在 `./sample_data/` 上运行本模块后，应检查：

1. **尺度合理性**：$s$ 的值应在 $[0.1, 50.0]$ 范围内（典型的桌面物体在 $[0.5, 5.0]$）
2. **空间位置**：$\mu_{metric}$ 的 Z 均值应在 $[0.5, 5.0]$ 米范围内（典型的人-物交互距离）
3. **渲染一致性**：将 $G^{metric}$ 用 Step 4 的渲染器投影到 2D，叠加在原始帧上，轮廓应与 $M_{object}$ / $M_{human}$ 大致吻合
4. **数值稳定性**：所有输出值无 NaN / Inf
