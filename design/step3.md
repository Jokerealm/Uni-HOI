### 文档 3：基于 Flow Matching 的 3D 初始化 (3D Lifting)

**模块目标**：将 2D 的补全视频拉升为 3D Gaussian Splatting (3DGS) 的初始表达。这是从 2D 到 3D 的跃迁层。

**核心输入**：

- 来自step 2 的补全视频 $V_{o\_amodal}$ 和 $V_{h\_amodal}$

**开发任务单**：

1. **Flow Matching 模型接入**：加载你计划使用的、基于 Flow Matching 架构的图像/视频到 3D 的生成模型代码库（如果是开源的或者你之前准备好的 Baseline）。
2. **独立生成与表达转换**：
	- 输入 $V_{o\_amodal}$，通过常微分方程 (ODE) 采样，生成物体的 3D 表示，并转换为初始的 3D Gaussians 属性参数集合 $G_o$（包括均值、协方差、球谐系数、不透明度）。
	- 输入 $V_{h\_amodal}$，执行相同的操作，获取人体的初始 3D Gaussians 属性参数集合 $G_h$。

**模块输出**：两组初始的 3DGS 参数文件 $G_o$ 和 $G_h$（位于各自的规范化坐标系中）。