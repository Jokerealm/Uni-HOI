# Dual-Branch Co-Generative Flow Matching Design

这个目录用于固定 Uni-HOI 新主干的设计约束，避免代码继续回退到旧的级联路径。

当前主目标：

- 用一个统一的条件补全式 Flow Matching 主干，同时生成：
  - 人体/物体 amodal video
  - 4D HOI state
- 让视频分支持续感知 3D 结构先验
- 让 3D 分支持续吸收视频时空证据
- 把 Step3/Step4 从“主生成器”降级为 bootstrap / refinement

目录说明：

- `pipeline.md`
  - 训练图、推理图、模块职责、阶段划分
- `model_method_reference.md`
  - 当前代码实现对照版的方法文档，包含文件职责、训练/推理接口和张量 shape
- `tensor_shapes.md`
  - 默认维度、token 数量、每个变量的输入来源和输出去向
- `innovations.md`
  - 相对旧级联代码的核心创新点和取舍
- `runtime_and_scripts.md`
  - 运行环境、脚本入口、关键配置、显存与调试说明

当前默认实现约束：

- 图像尺寸：`256 x 256`
- Patch Size：`16`
- 视频 token 网格：`T x 16 x 16`
- 视频注意力：时空分离 attention，不做 `T*h*w` 的全展开自注意力
- Attention 内核：统一走 `torch.nn.functional.scaled_dot_product_attention`
- 训练策略：显式三阶段 loss curriculum
