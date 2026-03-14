### 文档 5：全链路集成与端到端验证 (Full Pipeline Integration & End-to-End Validation)

**模块目标**：核实文档 1 至 4 模块的连通性，确保整个系统在面向生产环境的标准 `train.py` 和 `test.py` 脚本下，能够在微型子数据集 (`sample_data`) 上成功完成“数据加载 $\rightarrow$ 训练优化 $\rightarrow$ 指标评估 $\rightarrow$ 结果输出”的完整闭环。验证代码逻辑的健壮性与评估指标的正确性。

**核心输入**：

- 完整的项目代码库（包含 Step 1-4 的所有类和函数）。
- 微型验证子集目录 `./sample_data/`（结构与 `/data4/guanz/data/Behave` 完全一致）。
- Hydra 全局配置文件 `conf/config.yaml`。

**开发与验证任务单**：

**1. 配置路由与环境核验 (Configuration & Env Check)**

- 确保处于 `hdm` conda 环境。
- 检查 Hydra 配置体系，确保数据路径由顶层参数控制。此时默认配置或命令行覆写应指向 `./sample_data/`，但 `train.py` 内部**没有任何写死的相对或绝对路径**。

**2. 端到端训练测试 (End-to-End Training Execution)**

- 使用标准的 GPU 调度命令和 Hydra 覆写启动训练：

	`CUDA_VISIBLE_DEVICES=1 python train.py dataset=sample model.epochs=2`

- **核实点**：

	- **日志输出**：控制台是否清晰打印了模型结构、参数量、当前使用的设备 (GPU ID)。
	- **Loss 收敛**：监控多区域渲染 Loss ($L_{render}$)、接触 Loss ($L_c$)、投影 Loss ($L_{j2d}$)、穿模 Loss ($L_{pen}$) 等是否正常计算并完成反向传播（Loss 值不能为 NaN 或 Inf）。
	- **系统开销**：检查显存占用是否符合预期，没有发生显存泄漏 (OOM)。

**3. 评估指标与推理验证 (Evaluation & Metrics Verification)**

- 使用标准测试脚本启动评估：

	`CUDA_VISIBLE_DEVICES=1 python test.py dataset=sample checkpoint.run_id=latest`

- **核实点（对齐学术标准）**：

	- 脚本必须能够正确计算并输出核心的定量指标，即使在 Sample 数据集上数值不好看，但计算逻辑必须无误：
		- **Chamfer Distance (倒角距离)**：验证是否正确计算了 SMPL 人体 (CD-h)、物体 (CD-o) 以及联合网格 (CD-c) 的三维重建误差 。
		-**Acceleration Error (时序加速度误差)**：验证是否正确计算了人体关节 (Acc-h) 和物体位姿 (Acc-o) 的运动平滑度 。
	- **尺度检查**：确认输出的 3D 结果保持在 Metric Scale（度量尺度），没有发生尺度退化或坍缩。

**4. 结果可视化与文件输出 (Output Inspection)**

- **检查 Checkpoint 目录**：确认模型权重 (`.pt` 或 `.pth`)、优化器状态和 Hydra 运行配置被正确保存在类似 `outputs/runs/YYYY-MM-DD_HH-MM-SS/` 的规范目录下。
- **检查可视化输出**：确认 `test.py` 输出了验证用的渲染图像或视频片段，叠加了 2D 关键点与接触点的高亮提示，以及 3DGS 渲染出的新视角结果，以便进行定性分析。

**模块验收标准**：

一旦该任务单在 `./sample_data/` 上无任何 Bug 跑通，且各 Loss 和 Metric 均能正常浮动，即可直接将启动命令修改为 `python train.py dataset=behave`（指向 `/data4/guanz/data/Behave`），无缝开启大规模集群训练。