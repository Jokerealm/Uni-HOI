### 文档 1：离线数据预处理与高精度先验提取 (Offline Data Preprocessing)

**模块目标**：作为整个系统的前置独立模块，负责调用所有重量级的 2D/3D 基础模型（SAM 3, SAM 3D Body, UniDepth V2, OpenPose）。将原始视频帧序列转化为高精度的 2D/3D 先验特征，进行度量尺度对齐与多区域掩码计算，并将所有结果**持久化落盘存储**。

**架构变更指令**：此脚本独立运行（例如命名为 `preprocess.py`）。后续的 `train.py` 和 DataLoader 中**必须彻底移除**这些模型的推理代码，仅保留对预处理结果的文件读取逻辑。

**核心输入**：

- 视频帧图像目录：`./sample_data/video_name/frames/`（或通过 Hydra 切换至完整数据集路径）。

**预训练模型与权重路径 (固定配置)**：

- `BASE_WEIGHTS_DIR = "/data4/guanz/coding/HDM"`
- **SAM 3**: `$BASE_WEIGHTS_DIR/sam3/`
- **SAM 3D Body**: `$BASE_WEIGHTS_DIR/sam3d/`
- **UniDepth V2**: `$BASE_WEIGHTS_DIR/unidepth/`
- **OpenPose**: `$BASE_WEIGHTS_DIR/openpose/`
- **SMPL-H**: `$BASE_WEIGHTS_DIR/smpl_models/smplh/`

**开发任务单**：

**1. 预处理专属配置 (Hydra Config)**

- 在 `conf/` 下新建 `preprocess.yaml`，编写 `PreprocessConfig` 数据类，包含各类基础模型的加载参数、输入路径以及**统一的输出根目录**（如 `./sample_data/video_name/processed/`）。

**2. 模型加载与前向推理 (Heavy Inference)**

- 编写独立脚本 `preprocess.py`，加载上述所有预训练模型。
- **SAM 3 分割追踪**：利用文本 Prompt 自动提取全视频的人物掩码 $M_{human}$ 和物体掩码 $M_{object}$。
- **2D 关键点**：调用 OpenPose 提取人体关节 $\hat{J}_i$。
- **度量深度**：调用 UniDepth V2 提取绝对物理深度图 $D_{pred}$。
- **3D 人体参数**：调用 SAM3d-body-dinov3 提取高精度的 SMPL-H 初始参数 $\mathcal{H}_i$。

**3. CPU 端的几何与尺度对齐 (Metric Alignment & Math)**

- **深度配准**：计算掩码区域内（$M_{human} \cup M_{object}$）深度的中位数，求解 $s$ 和 $t$，将 SAM 3D 预测的深度平移统一对齐到度量深度 $D_{align} = s \cdot D_{pred} + t$。
- **多区域掩码计算**：
	- 投影 SMPL-H 关节获取接触点 $M_{contact}$。
	- 计算交互边界 $M_{boundary}$ 与凸包 $M_{hull} = \text{ConvexHull}(M_{boundary} \cup M_{contact})$。
	- 划分主遮挡区 $M_p = M_{human} \cap M_{hull}$ 和次遮挡区 $M_s = M_{human} \setminus M_p$。
	- 对 $M_p, M_s, M_{object}$ 应用高斯滤波 (Gaussian Blur) 生成软边缘连续权重。

**4. 高效持久化序列化 (Serialization & I/O)**

- 将计算结果规范地保存至 `processed/` 目录下，建立清晰的子目录结构。例如：
	- `/processed/masks/human/` 和 `/processed/masks/object/` (推荐格式：无损 `.png`，节省空间)
	- `/processed/masks/multi_region/` (软掩码，推荐 `.npz` 或 `float16` 格式的 `.png`)
	- `/processed/depth/` (深度图，推荐 `.npz` 或 `.exr`)
	- `/processed/poses/smplh_aligned.npz` (包含所有帧的 $\mathcal{H}_i$ 字典)
	- `/processed/keypoints/openpose_2d.npz` (包含所有帧的 $\hat{J}_i$)

**5. 约束与重构要求**

- **重构 DataLoader**：在完成此预处理脚本后，需修改原有的 Dataset/DataLoader 类。新的 DataLoader **只允许**根据帧索引 `idx` 从磁盘读取对应的 `.png` 和 `.npz` 文件，并转换为 PyTorch Tensor 发送到 GPU。禁止在 DataLoader 的 `__getitem__` 中进行任何模型推理或复杂的几何运算。