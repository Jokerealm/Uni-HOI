# Uni-HOI 项目全局开发规范与配置总纲 (Master Guidelines)

## 1. AI 协作核心准则 (AI Workflow Rules)

- **上下文重置要求：** 本项目工程量巨大，被拆分为多个子阶段 (Steps)。每次开启新的开发会话 (Session) 或处理新的 Step 时，AI 必须首先读取并严格遵循本文档中的所有环境、配置与架构规范。
- **渐进式交付：** AI 在生成代码时，应专注于当前请求的子模块，避免过度实现未要求的后续流程。

## 2. 运行环境与硬件分配 (Environment & Hardware)

- **Python 环境：** 所有代码必须兼容并运行在名为 `cari4d` 的 conda 环境中, AI 提供的任何运行脚本或提示，均需默认激活该环境 (`conda activate cari4d`)。
- **GPU 资源调度：** 服务器为多卡环境，严禁代码默认占用所有 GPU。
	- 执行训练或推理脚本时，必须通过环境变量显式指定 GPU 编号
	- 标准执行命令示例：`CUDA_VISIBLE_DEVICES=0 python train.py ...`。代码内部不应硬编码特定的 GPU ID。

## 3. 数据集与本地路径配置 (Dataset Configurations)

- **真实数据集路径：**
	- Behave Dataset: `/data4/guanz/data/Behave`
	- ProciGen Dataset: `/data4/guanz/data/ProciGen`
- **微型验证子集 (Sample Dataset) 机制：**
	- **目的：** 极速验证代码逻辑是否走通，不对模型输出质量做任何要求。
	- **存放位置：** 项目根目录下的 `./sample_data/` 文件夹。
	- **一致性要求（核心约束）：** 子数据集在目录结构、文件命名规范、标注格式必须与完整版 Behave/ProciGen **100% 保持一致**。代码只要在 `./sample_data/` 上能够无 Error 跑通，切换到完整的 `/data4/guanz/data/` 路径后，必定能直接运行。
- **项目路径：**
	- 项目的根路径为：/data4/guanz/coding/HDM
	- 本项目用到的其它路径下的权重和模型都需要正确地放在本项目下。

## 4. 架构设计与配置管理 (Architecture & Configuration Standards)

本项目采用高度模块化、强类型约束的设计模式，AI 生成代码时必须遵循以下规范：

- **Hydra 框架驱动：** 使用 **Hydra** 作为全局配置管理工具。所有超参数、路径和模型切换均需通过 Hydra 的 `.yaml` 配置文件组织，并支持通过 Hydra 的命令行覆盖机制 (Command-line Overrides) 动态修改（例如：`python train.py model.lr=1e-4`）。
- **强类型配置类 (Dataclasses for Configs)：** * 必须使用 `dataclass` 为代码中的核心组件编写结构化的配置类。
	- 全局运行的顶层配置必须封装在 `class RunConfig` 中。
	- 具体的模型、优化器等组件需拥有各自独立的配置类。例如：`AdamOptimizerConfig`, `CrossAttnHOModelConfig`, `FlowMatchingConfig` 等。
- **动态模型与 Trainer 路由：**
	- 通过配置文件中的 `model.model_name` 参数来控制当前实例化的模型。
	- 系统需实现工厂模式或注册表机制：根据解析到的模型类型，自动选择并实例化对应的 Trainer 类。
- **面向对象的 Trainer 体系 (Trainer Inheritance)：**
	- 必须实现一个基础的基类 `BaseTrainer`，封装通用的逻辑（如设备挂载、梯度清零、模型保存/加载等）。
	- 具体的训练逻辑必须通过继承 `BaseTrainer` 来实现（例如 `class FlowMatchingTrainer(BaseTrainer)`, `class Joint3DGSTrainer(BaseTrainer)`），在子类中重写核心的 `training_step` 等方法。

## 5. 日志与监控 (Logging & Console Outputs)

- **日志规范：** 训练脚本需包含简洁但有效的基础训练日志。
- **控制台输出：** 拒绝冗长无用的打印。控制台输出应重点呈现：当前 Step/Epoch、Loss（分类拆解，如 $L_{render}$, $L_{pen}$ 等）、学习率、当前step/epoch花费的总时间以及预计剩余时间 (ETA)。