### **Uni-HOI 重建方法**

#### **1. 核心目标与任务**

* **输入：** * **训练期：** 单目 RGB 视频 + 掩码（Masks） + 深度图（Depth） + 文本提示（Text Prompt）。

	* **推理期：** 单目 RGB 视频 + 掩码（Masks） + 自动生成的文本标签。

* **输出：** 精准的 3D 物理控制参数：人体 SMPL 参数（$\theta, \beta, t$）及物体 6DoF 位姿（$R, t$）。

* **核心难点：** 通过 2D 视频大模型的 Amodal（无模态）补全能力，解决人-物交互中的极端遮挡问题，消除穿模与空间歧义，实现高精度的接触点估计。

* **任务范围：** 当前方法仅面向 **已知相机内参的 BEHAVE / 同类标注数据集**，暂不考虑 wild 场景。

	**统一坐标系：** 所有的人体重建、物体位姿回归、深度重投影和物理约束，均在 **相机坐标系（Camera Coordinate System）** 下定义和优化。

*  **文本来源约定：** Wan 所需文本条件不从自由语言生成，而是由数据集目录或标注元数据中解析出物体类别，例如 `chair`、`box`、`toolbox` 等；当前不考虑 open-vocabulary / wild 文本场景。

#### **2. 网络架构设计：基于潜空间（Latent Space）的双分支扩散模型**

模型在 Wan2.2 的 VAE 潜空间内运行，以白嫖高性能的图像压缩与生成先验：

* **分支 A：2D Amodal 视频先验分支（冻结）**
	* **机制：** 利用 Wan2.2 的 Inpainting 能力。给定原始 Latent 和 Mask，分别生成“纯人体（擦除物体）”和“纯物体（擦除人体）”的干净 Amodal 前景视频流，将被遮挡的像素（如被物体挡住的手臂）在隐空间中“画”出来。
	*   **训练策略：** 分支 A 不参与训练，不单独引入优化损失；仅保留可视化与质量监控。
* **分支 B：3D 状态重建分支（可训练）**
	* **机制：** 采用 DiT 风格的可训练状态重建主干，在每一层接收：
		* 当前噪声状态 `x_t`
		* 原始输入 latent `x_orig`
		* 冻结分支 A 的 amodal 先验特征
		* 相机内参条件 `K`
	* **结构化 State Token 设计：**
		* `Human Shape Token`：建模人体形状 `beta`
		* `Human Pose Tokens`：逐帧建模人体姿态 `theta`
		* `Human Translation Tokens`：逐帧建模人体平移 `t`
		* `Object Pose Tokens`：逐帧建模物体 6DoF
		* `Contact Tokens`：逐帧建模接触状态、接触点或接触签名
		* 可选 `Global Context Token`：吸收整段视频的全局交互语义
	* **目标：** 让网络学习 “遮挡补全后的 2D 证据 -> 结构化 3D 状态 -> 可渲染/可重投影物理参数” 的闭环映射。

#### **3. 特征交互与 3D 参数提取机制（无 Decoder 架构）**

* **特征注入 (Zero-Linear)：** 分支 A 补全后的干净前景特征通过 Zero-Linear 注入分支 B。这为重建分支提供了“透视”后的视觉参考，消除了遮挡导致的信息缺失。
* **3D-2D 交叉注意力 (Cross-Attention)**：引入“结构化 State Token”，弃用单一的全局 Query： 不要只用一个 Human Query 和一个 Object Query, 让多组结构化 state tokens 与融合潜特征进行 cross-attention / mutual interaction。 每类 token 只负责自身的物理语义子空间，降低 “单 token 负担过大” 的问题。
* **参数头（Prediction Heads）：**
	* `Human Shape Head`：由 `Human Shape Token` 回归 `beta`
	* `Human Pose Head`：由 `Human Pose Tokens` 回归逐帧 `theta`
	* `Human Translation Head`：由 `Human Translation Tokens` 回归逐帧 `t`
	* `Object Pose Head`：由 `Object Pose Tokens` 回归逐帧物体 6DoF
	* `Contact Head`：由 `Contact Tokens` 回归接触状态 / 接触点签名
* **Decoder 使用原则：**
	* 最终参数输出不依赖 VAE Decoder。
	* VAE Decoder 仅在训练或调试时打开，用于可视化 amodal 结果与潜空间补全质量。

#### **4. 训练策略：两阶段流匹配（Flow Matching）**

##### Stage 1：唤醒 3D 几何与结构化状态先验

目标：在不训练分支 A 的前提下，先让分支 B 学会稳定的 3D 几何与结构化状态表达。

* **监督目标：**

	* 结构化 state token 的 flow matching
	* 基于状态投影得到的几何辅助监督，例如：
		* 关键点热图
		* 物体 silhouette
		* 深度投影图

* **Flow Matching 损失：**
	$$
	\mathcal{L}_{fm}
	=
	\mathbb{E}_{t, x_0, \epsilon}
	\left[
	\left\|
	\mathcal{D}^{state}(x_t, t, x_{orig}, K)
	-
	(x_0 - \epsilon)
	\right\|^2
	\right]
	$$

* **目标解释：**

	* 让分支 B 在 `x_orig` 和 `K` 的引导下，先学会 “什么样的 3D 状态与投影几何是合理的”。
	* 这一阶段重点是把状态空间学稳，而不是立即追求最强的接触物理效果。

##### Stage 2：Amodal 引导下的联合训练

目标：引入冻结分支 A 的 amodal 先验，进行最终的人体/物体参数回归与物理一致性优化。

* **参数回归损失 `L_reg`：**

	* 目标是使用 BEHAVE 官方真值对人体和物体状态进行强监督。
	* 人体部分直接监督 `theta`、`beta`、`t`。
	* 物体部分监督 6DoF。
	* 文档中记为：

	$$
	\mathcal{L}_{reg}
	=
	\lambda_{\theta}\|\theta - \hat{\theta}\|^2
	+
	\lambda_{\beta}\|\beta - \hat{\beta}\|^2
	+
	\lambda_{Rt}\|(R, t_{obj}) - (\hat{R}, \hat{t}_{obj})\|^2
	$$

	* **实现建议：** 物体旋转部分建议使用 `6D rotation parameterization + geodesic loss` 落地，而不是直接对旋转矩阵做 L2。

* **深度重投影损失 `L_depth`：**

	* 该项必须显式依赖相机内参 `K`：

	$$
	\mathcal{L}_{depth}
	=
	\left\|
	\mathcal{D}_{render}(\mathcal{M}_{pred}, K)
	-
	\mathcal{D}_{gt}
	\right\|_1
	$$

	* 其中 `M_pred` 表示由预测的人体/物体状态在相机坐标系下渲染得到的几何。
	* 该项用于约束单目场景中最容易漂移的 `Z` 轴关系。

* **时序损失 `L_temp`：**

	* 不再只约束人体参数的一阶/二阶差分。
	* 采用 **CARI4D 风格**的两类时序约束：
		* 基于物体顶点的运动学平滑
		* 接触点相对速度锁定（contact-relative velocity locking）
	* 目标是减少：
		* 物体位姿抖动
		* 接触点滑移
		* 接触状态闪烁

* **物理约束损失 `L_phys`：**

	* 采用 **CARI4D 风格**的 HOI 物理约束：
		* `Contact Loss`
		* `Penetration Loss`
	* 其中：
		* `Contact Loss` 用于鼓励接触点或接触顶点靠近合理的人物交互区域
		* `Penetration Loss` 用于抑制人和物体之间的穿透

* **关于分支 A 的 Loss：**

	* 分支 A 完全冻结，因此不单独设计训练损失。
	* 对分支 A 只做可视化和质量检查，不做反向传播优化。

#### **5. 推理阶段（Inference）：端到端极速前馈**

* **流程：**
	* RGB 视频 + Masks + `K` + 物体类别文本
	* `-> VAE Encoder`
	* `-> 冻结 Wan amodal 先验分支 A`
	* `-> 可训练结构化状态分支 B`
	* `-> 结构化 State Tokens`
	* `-> 参数头`
	* `-> SMPL(theta, beta, t) + Object 6DoF + Contact`

* **推理约束：**
	* 当前仅考虑已知物体类别、已知相机内参的受控数据集场景。
	* 不考虑 wild 下自动文本生成、相机自标定和开放词表物体分类。

#### 6. Loss 方案总结表

| 阶段    | Loss 项     | 核心监督信号                             | 目的                       |
| :------ | :---------- | :--------------------------------------- | :------------------------- |
| Stage 1 | `L_fm`      | 结构化 state token 的 flow matching      | 唤醒 3D 几何与状态先验     |
| Stage 1 | `L_geo_aux` | keypoint / silhouette / depth 投影监督   | 建立稳定的 3D-2D 几何映射  |
| Stage 2 | `L_reg`     | BEHAVE 官方 GT 的 SMPL + 物体位姿强监督  | 保证基础姿态与位姿精度     |
| Stage 2 | `L_depth`   | `D_render(M_pred, K)` 与 `D_gt` 的一致性 | 约束单目深度与空间对齐     |
| Stage 2 | `L_temp`    | CARI4D 风格运动学平滑 + 接触速度锁定     | 保证时序稳定，抑制接触滑移 |
| Stage 2 | `L_phys`    | CARI4D 风格 Contact + Penetration        | 提高接触真实性并防穿透     |

**训练建议：**

* `L_reg` 在 Stage 2 中保持主导地位。
* `L_phys` 建议在训练中后期逐步开启，避免训练初期因接触和穿透约束导致梯度不稳定。
* `L_depth` 在已知 `K` 的设置下应保持较高权重，因为它直接约束单目场景的深度合理性。

#### 7.训练过程可视化内容

以下是需要可视化并上传到wandb的内容

### 1. 2D 潜空间映射可视化（Amodal 质量监控）

* **可视化内容：** 调用那条“不参与训练的 VAE Decoder 支路”，渲染出 **Human-only Amodal 视频**和 **Object-only Amodal 视频**。
* **观察点：** 
	* 被遮挡的肢体（如被箱子挡住的手臂）是否补全了？
	* 补全的部分是否存在严重的闪烁（Temporal Jitter）？
	* 物体轮廓是否稳定

### 2. 3D 重建结果的“重投影”叠加图（关键指标）

这是判断 **3D 参数回归**是否准确的最直观方式。

* **可视化内容：** 将回归得到的 SMPL 和 6DoF 物体模型，通过相机参数投影回原始 2D 视频。
* **渲染方式：** * **叠加渲染（Overlay）：** 将半透明的 3D Mesh 叠在原图上。
	* **侧视图（Side-view）：** 渲染一个侧面的视角，看人在 $Z$ 轴上是否真的“踩”在物体上。
* **观察点：** * **空间对齐：** 人手在 2D 图像上是否对准了物体边缘？
	* **深度关系：** 从侧面看，人与物体之间是否有不合理的空隙（悬空）或严重的深层穿模？

### 3. **结构化 State Token 注意力图**

* * **目的：** 验证结构化 state tokens 是否在各自关注正确的视觉区域。
	* **可视化内容：**
		* `Human Pose Tokens` 的注意力热图
		* `Object Pose Tokens` 的注意力热图
		* `Contact Tokens` 的注意力热图
	* **观察点：**
		* 人体姿态 token 是否关注肢体区域
		* 物体位姿 token 是否关注物体边界与可见区域
		* 接触 token 是否聚焦在人手与物体接触附近

### 4. 3D 几何（网格/点云）

这是衡量 Uni-HOI 最终成败的“终极考核”。你不能只看重投影（那是 2D 眼光），必须把重建出的人体 Mesh 和物体 Mesh 和GT 去对比。

* **可视化内容：**
	* 人体 mesh
	* 物体 mesh
	* 接触区域
	* 侧视深度关系

### 5. Loss曲线

除了常规的总 Loss，你必须拆开观察以下细分项：

| 监控项                    | 异常表现（需调整点）                                     |
| :------------------------ | :------------------------------------------------------- |
| **$\mathcal{L}_{fm}$**    | 若不下降，说明结构化 state 空间未学稳                    |
| **$\mathcal{L}_{depth}$** | 若波动大，说明深度和空间对齐不稳定                       |
| **$\mathcal{L}_{temp}$**  | 若该项极低但画面仍抖动，说明时序约束未绑定到关键接触区域 |
| **$\mathcal{L}_{phys}$**  | 穿模率不降，说明接触/穿透约束未真正生效                  |
| **$\mathcal{L}_{reg }$**  | 降不下来，说明状态 token 到参数头的映射仍有问题          |


### 上传说明

每1000个 Step 就上传一段 amodal 视频、一段 3D overlay 视频、一段侧视图视频。训练初期，优先看 **Amodal 视频**（确认Stage 1 的结构化状态稳定性）；训练中后期，盯着 **3D 重投影图**和**侧视图**。

#### 工程性训练提示

1. 先设置stage1跑10w steps，确认结构化状态空间和几何投影已经收敛，然后确定Golden Checkpoint. Stage 2 以该 Checkpoint 为起点的无限次试错
2. Stage 2 开始时，不要加载 Stage 1 的 optimizer state；由于新引入了 `L_reg / L_depth / L_temp / L_phys`，旧动量通常会干扰新阶段优化。Stage 1 优化器里积累的动量（Momentum）和方差对 Stage 2 是有害的。在 Stage 2，必须从 0 初始化一个全新的 Optimizer。
3. Stage 2 加载权重时建议使用 `strict=False`，因为此时会激活新的分支连接、参数头或 contact 相关模块。
4. Stage 2 重置学习率调度器，并重新 warmup，避免新开启的结构化 token 交互层和物理约束头在初期破坏已有几何先验。