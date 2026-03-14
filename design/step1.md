### 文档 1：空间对齐与多区域掩码提取 (Data Prep & Perception)

**模块目标**：处理原始视频帧序列，利用最新的 2D/3D 基础模型（SAM 3 家族等）全自动提取先验信息，将人体和深度统一到真实的物理尺度（Metric Scale），并计算出用于 3DGS 联合优化的多区域权重掩码。代码需严格遵循 Hydra 配置标准与 OOP 设计。

**核心输入**：

- 预先切分好的单视角 RGB 视频帧序列目录 $I_{orig}$ (开发验证期严格使用 `./sample_data/video_name/frames/`)

**预训练模型与权重路径 (Pretrained Models & Weights)**：

所有的模型权重必须从全局变量指定的本地统一目录加载。在 Hydra 配置文件中，需设定基准路径 `BASE_WEIGHTS_DIR: "/data4/guanz/coding/HDM/model"`：

- **SAM 3**: `$BASE_WEIGHTS_DIR/sam3/` (负责 2D 文本提示分割与时序追踪)
- **SAM 3D Body**: `$BASE_WEIGHTS_DIR/sam3d/` (包含 `sam3d-body-dinov3` 权重，负责 3D 人体参数估计)
- **UniDepth V2**: `$BASE_WEIGHTS_DIR/unidepth/` (负责单目度量深度估计)
- **OpenPose**: `$BASE_WEIGHTS_DIR/openpose/` (负责 2D 关键点检测)
- **SMPL-H Body Model**: `$BASE_WEIGHTS_DIR/smpl_models/smplh/` (人体参数化模型基础文件)

**开发任务单**：

**1. 基础配置与架构搭建 (Hydra & OOP)**

- 编写 `conf/config.yaml` 顶层配置，建立 `DataPrepConfig`、`SAM3Config`、`SAM3DConfig` 等强类型数据类 (Dataclasses)。
- 建立数据预处理的基类或 Pipeline 类，确保输入路径可以通过命令行轻松从 `./sample_data/` 切换到 `/data4/guanz/data/Behave`。

**2. 基础视觉感知 (2D Perception)**

- **实例化 SAM 3**：编写追踪脚本，利用文本 Prompt（例如 `["human", "object"]`）自动在第一帧初始化，随后遍历输出全视频的人物可见掩码 $M_{human}$ 和物体可见掩码 $M_{object}$。
- **实例化 OpenPose**：提取全序列人体 2D 关键点 $\hat{J}_i$。

**3. 度量尺度与 3D 参数对齐 (Metric Alignment & 3D Estimation)**

- **获取绝对深度**：调用 UniDepth V2 获取每帧的物理绝对深度 $D_{pred}$。
- **获取 3D 人体**：调用 SAM3d-body-dinov3 获取每帧极具鲁棒性的 SMPL-H 姿态与形状参数 $\mathcal{H}_i$。
- **深度尺度配准脚本**：计算掩码区域内（人和物体合并掩码）深度的中位数。求解尺度 $s$ 和平移 $t$，使得对齐后的深度 $D_{align} = s \cdot D_{pred} + t$。将 SAM3d 预测的深度平移在三维空间中强制对齐到 $D_{align}$。

**4. 多区域接触感知掩码 (Multi-Regional Masking)**

- 将对齐后的 SMPL-H 3D 关节点投影到 2D 提取接触点掩码 $M_{contact}$。
- **逐帧执行几何掩码划分**：
	- 交互边界：$M_{boundary} = \text{dilate}(M_{human}) \cap \text{dilate}(M_{object})$
	- 交互凸包：$M_{hull} = \text{ConvexHull}(M_{boundary} \cup M_{contact})$
	- 主遮挡区 (高置信度)：$M_p = M_{human} \cap M_{hull}$
	- 次遮挡区 (低置信度)：$M_s = M_{human} \setminus M_p$
- **软边缘处理**：对生成的 $M_p$、$M_s$ 以及 $M_{object}$ 应用高斯滤波 (Gaussian Blur)，平滑掩码边缘以防止后续渲染出现断层。

**模块输出**：

将处理好的核心数据（对齐后的 $\mathcal{H}_i$、绝对深度图序列 $D_{align}$、软边缘掩码序列 $M_p, M_s, M_{object}$）统一序列化保存为 `.npy` 或 `.npz` 格式，存放至对应的 `./sample_data/video_name/processed/` 目录下，作为下一阶段（视频补全与 3DGS 初始化）的完美输入。