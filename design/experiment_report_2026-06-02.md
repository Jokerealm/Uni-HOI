# Uni-HOI 实验报告与下一步模型改进建议

生成时间：2026-06-02（Asia/Shanghai）

本文整理当前仓库中已经落盘的实验配置、评估 JSON、训练日志和已有设计记录，目标是回答三个问题：

1. 目前哪些实验结果可以直接比较？
2. 每类模型改动带来了什么趋势？
3. 下一步应该优先改哪些模型或训练策略？

## 1. 结论摘要

当前最强的可比结果是 `outputs/cointeract_stage1_geometry_lr1e4_70k` 的 **high-LR stage1-only**，在 BEHAVE heldout 全量 720 样本、ODE12、batch 4 的协议下，step 10k 达到：

| 当前最佳 | Step | CD-mean | CD-h | CD-o | CD-c | supervised |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high-LR stage1-only | 10000 | **0.484711** | **0.411127** | 0.672766 | **0.370240** | 0.736174 |

主要判断：

- **优先保留 high-LR stage1-only 作为当前 Chamfer 基线。** 它比 only-stage1 低 0.012561 CD-mean，比 MoE 低 0.020165 CD-mean。
- **supervised loss 与 Chamfer 不完全一致。** MoE 和 group-balanced FM 的 supervised 更低，但 CD-mean 更差；下一步不要只按 supervised loss 选模型。
- **stage2 constrained refine 没有带来稳定提升。** 它从 only-stage1 best 初始化后，全量 CD-mean 为 0.502635，反而差于初始化对应的 0.497272，并且 supervised 指标异常大（2.986581），说明 refine 阶段的目标或权重设置需要重做。
- **group-balanced state FM 当前失败。** 虽然它降低 supervised 到 0.574851，但 CD-mean 到 0.627721，明显劣化。
- **MoE 当前不值得继续放大。** MoE best 为 0.504876，低于 group-balanced，但仍差于 high-LR 和 only-stage1。
- **RGB/HOI full interaction 在当前实现中强于 asymmetric/drop-RGB 评估。** stage1 100k 的 full eval CD-mean 约 0.525，而 asymmetric 或 drop RGB 约 0.566-0.572。

下一步最建议做的是：围绕 high-LR stage1-only 做受控消融，而不是继续堆 MoE 或 group-balanced FM。优先级为：

1. 在 high-LR stage1-only 上做 5k/7.5k/10k/12.5k/15k/20k 全量评估，确认最优窗口是否稳定。
2. 针对 object CD-o 仍偏高的问题，做 object Gaussian / object pose 方向的局部损失重加权，而不是全局 group-balanced。
3. 重新设计 stage2 refine：从 high-LR 10k 初始化，用更温和的 loss 权重和更短训练窗口，避免 supervised loss 爆大。
4. 对 high-LR 10k 做 full/asymmetric/drop-RGB 三种推理评估，确认模型是否真正依赖 RGB prior。

## 2. 数据、方法与评价协议

### 2.1 数据

主要训练数据：

- `sample_data/WAI_prepared/sequences`
- 当前 CoInteract 训练日志显示约 159 个 sequences、1431 个 samples。

主要评估数据：

- `sample_data/BEHAVE_heldout_prepared/sequences`
- 当前全量评估协议为 720 samples、batch size 4、ODE steps 12。

早期和 smoke 实验还使用过：

- `sample_data/behave_1pct/sequences`
- ProciGen smoke 数据和并行预处理输出
- CARI4D supervised smoke

这些早期结果主要用于验证流程是否跑通，不应与当前 720-sample heldout 结果直接比较。

### 2.2 模型与训练方法

当前实验围绕 HOI-primary RGB-guided 显式状态预测展开：

- **HOI 状态 token**：SMPL shape、SMPL pose、人相对平移、object SE(3)、contact、3D joints、human Gaussians、object Gaussians。
- **RGB stream**：使用冻结的 Wan visual prior 产生 RGB hidden tokens，默认 `freeze_wan=true` 且 `detach_rgb_context=true`。
- **主干**：8 层、hidden dim 512、8 heads 的 DiT/CoInteract 变体。
- **融合方式**：
  - single-stream：视觉 token 与 HOI token 直接拼接。
  - dual-stream/dense-contact：早期双流并加入 dense contact。
  - independent HOI/RGB dual branch：HOI block 和 RGB block 相对独立。
  - shared CoInteract：HOI/RGB token 共享 attention/MLP 主干，通过 stream-specific AdaLN 区分模态。
  - full interaction：HOI/RGB token 拼接后全注意力交互。
  - asymmetric：RGB->HOI 保留，HOI->RGB 关闭。
- **损失**：
  - state flow matching / explicit state supervised loss
  - shape、pose、translation、object pose、contact、joints
  - human/object Gaussian Chamfer
  - Gaussian xyz / attr L1
  - physical contact / penetration
  - 可选 HOI-token-aware MoE router loss

### 2.3 指标

主指标：

```text
CD-mean = (CD-h + CD-o + CD-c) / 3
```

其中：

- `CD-h`：human Chamfer，越低越好。
- `CD-o`：object Chamfer，越低越好。
- `CD-c`：contact/combined contact Chamfer，越低越好。
- `supervised`：训练/评估记录中的显式状态监督损失，越低不一定代表 Chamfer 更好。

报告中将结果分成三类：

- **全量可比**：720 samples、ODE12、batch 4，可直接排序。
- **scan5 趋势**：5 batches、20 samples，适合选 checkpoint，不适合作最终结论。
- **早期周期/历史/smoke**：协议不同，只用于判断趋势或流程可用性。

## 3. 全量可比实验结果

以下结果均来自 BEHAVE heldout 全量 720 samples、ODE12、batch 4，除备注中明确说明的历史 flat metric 外，均可直接比较。

| 排名 | 实验 | 核心改动 | Step | CD-mean | CD-h | CD-o | CD-c | supervised | 结果文件 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | high-LR stage1-only | `lr=1e-4`，stage1 geometry，850/850 Gaussians，无 MoE | 10000 | **0.484711** | **0.411127** | 0.672766 | **0.370240** | 0.736174 | `outputs/cointeract_stage1_geometry_lr1e4_70k/eval_wai_test_full_ode12_b4_current_best10000.json` |
| 2 | only stage1 | `lr=2e-5`，stage1 geometry，850/850 Gaussians，无 MoE | 30000 | 0.497272 | 0.437633 | **0.672753** | 0.381429 | 0.699590 | `outputs/cointeract_stage1_geometry_50k/eval_wai_test_full_ode12_b4_best.json` |
| 3 | stage2 constrained refine | 从 only-stage1 30k 初始化，`lr=5e-6`，更强几何/物理约束 | 12500 | 0.502635 | 0.435626 | 0.691194 | 0.381084 | 2.986581 | `outputs/cointeract_stage2_constrained_refine_50k/eval_wai_test_full_ode12_b4_step12500_current_best.json` |
| 4 | MoE | HOI-token-aware MoE，router loss=1，uniform state FM | 27500 | 0.504876 | 0.440246 | 0.688721 | 0.385661 | **0.676226** | `outputs/cointeract_moe/eval_wai_test_full_ode12_b4_steps27500_70000.json` |
| 5 | shared stage1 full 100k, full eval | shared CoInteract，全程 full attention | 100000 | 0.524995 | 0.408125 | 0.773684 | 0.393177 | 0.585116 | `outputs/cointeract_shared_stage1_full_100k/eval_exp1_stage1_ckpt100k_full_ode12_b4_all.json` |
| 6 | shared stage1 100k, no RGB prior | 同 checkpoint，drop RGB branch | 100000 | 0.566143 | 0.459310 | 0.815339 | 0.423781 | 0.656450 | `outputs/cointeract_shared_stage1_full_100k/eval_stage1_ckpt100k_C_asym_no_rgb_prior_ode12_b4_all.json` |
| 7 | shared stage1 100k, asymmetric | 同 checkpoint，asymmetric RGB->HOI | 100000 | 0.571599 | 0.497293 | 0.773751 | 0.443752 | 0.650356 | `outputs/cointeract_shared_stage1_full_100k/eval_exp1_stage1_ckpt100k_asym_updated_ode12_b4_all.json` |
| 8 | shared two-stage selected | 10k full-attention stage1 + 5k stage2；全量选 14.5k | 14500 | 0.597181 | 0.517962 | 0.820358 | 0.453222 | 0.672538 | `outputs/cointeract_shared_hoi_wan_two_stage/eval_wai_test_full_ode12_b4_selected_11500_12500_14500.json` |
| 9 | group-balanced FM | MoE + group-balanced state FM | 40000 | 0.627721 | 0.506887 | 0.905506 | 0.470771 | 0.574851 | `outputs/cointeract_group_balanced_fm/eval_wai_test_full_ode12_b4_step40000.json` |
| 10 | dense-contact dual stream final full | 256/256 Gaussians，dense contact 早期基线 | 35000 | 0.785307 | 0.821474 | 0.878777 | 0.655671 | 0.466278 | `outputs/HOI_contact/test_metrics.json` |
| 11 | single-stream final full | 256/256 Gaussians，single-stream DiT | 35000 | 0.813020 | 0.877061 | 0.891454 | 0.670546 | 0.468048 | `outputs/unimodel_wai_real_smoke/test_metrics.json` |
| 12 | exp2 full aux0.05 | 从 stage1 初始化，短程 aux 实验 | 5000 | 2.056885 | 2.012987 | 2.363105 | 1.794564 | 1.842237 | `outputs/exp2_full_aux005_from_stage1_5k/eval_exp2_full_aux005_ckpt5000_asym_updated_ode12_b4_all.json` |

### 3.1 对全量结果的直接解释

**high-LR stage1-only 是当前应该保留的主基线。** 它在 CD-h 和 CD-c 上最强，CD-o 与 only-stage1 基本持平。因此它的收益主要来自 human/contact 几何质量，而不是 object 几何。

**object CD-o 是主要瓶颈。** 当前最佳 CD-o 为 0.672766，明显高于 CD-h 0.411127 和 CD-c 0.370240。继续优化时应重点关注 object pose、object Gaussian、object-local representation，而不是只压低总体 supervised loss。

**MoE 和 group-balanced FM 暂时不继续扩大。** MoE 虽然 supervised 更低，但 CD-mean 差 0.020165；group-balanced supervised 最低，但 CD-o 劣化到 0.905506，是所有当前大模型里最差的一组。

**stage2 refine 的损失设计有问题。** 它的 CD-mean 只比 MoE 略好，但差于初始化来源 only-stage1；supervised 到 2.986581，说明加权目标和评估 supervised 不在同一尺度，或者 refine 阶段过度优化了不匹配的项。

## 4. Checkpoint 扫描与消融趋势

这些结果多为 5-batch scan（20 samples）或单 checkpoint scan，用于选择 checkpoint 和判断趋势。

| 实验 | 协议 | Best step | CD-mean | CD-h | CD-o | CD-c | supervised | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| high-LR stage1-only | scan5, 22 ckpts | 10000 | 0.556268 | 0.442764 | 0.808498 | 0.417542 | 0.777483 | scan5 与全量都选到 10k，说明最优窗口较稳定。 |
| only stage1 | scan5, 20 ckpts | 30000 | 0.540031 | 0.463707 | 0.740577 | 0.415809 | 0.705690 | scan5 和全量都选到 30k。 |
| stage2 constrained refine | scan5, 5 ckpts | 12500 | 0.566189 | 0.469479 | 0.824103 | 0.404986 | 3.130984 | scan5 选 12.5k，但全量仍弱于 only-stage1。 |
| stage2 constrained refine | scan5, late ckpts | 25000 | 0.553375 | 0.499054 | 0.743261 | 0.417809 | 3.179884 | late scan5 看似略好，但没有全量验证。 |
| MoE | scan5, 23 ckpts | 27500 | 0.546945 | 0.479235 | 0.744970 | 0.416631 | 0.694021 | 与全量同样选 27.5k。 |
| MoE early | full 720, step 14500 | 14500 | 0.593770 | 0.514045 | 0.816553 | 0.450711 | 0.672691 | 早期 MoE 不如后续 27.5k。 |
| group-balanced FM | scan5 selected | 40000 | 0.586321 | 0.412370 | 0.911583 | 0.435010 | 0.584284 | human 好但 object 明显坏。 |
| group-balanced FM | scan5 step 100k | 100000 | 0.649761 | 0.532864 | 0.891689 | 0.524731 | 0.543593 | 继续训练 supervised 下降，但 CD 继续变差。 |
| shared stage1 full 100k | scan5, 40 ckpts | 40000 | 0.526598 | 0.454248 | 0.744485 | 0.381062 | 0.673733 | scan5 选 40k；100k 全量 full CD-mean 0.524995。 |
| shared stage2 from 40k | scan5, 24 ckpts | 37500 | 0.571835 | 0.468703 | 0.812665 | 0.434138 | 0.598122 | 从 40k 继续 stage2 没有优于 stage1。 |
| shared two-stage | scan5, 30 ckpts | 14500 | 0.591195 | 0.504334 | 0.827653 | 0.441599 | 0.699146 | selected full 也选 14.5k。 |
| independent HOI/RGB dual branch | 5-batch, 35 ckpts, batch 1 | 25000 | 0.804897 | 0.752070 | 1.075914 | 0.586708 | 1.175909 | 只评估 5 samples，协议较弱；趋势是 24k-33k 优于最终 35k。 |
| independent HOI/RGB dual branch | 1-batch, 35 ckpts | 21000 | 0.354031 | 0.185578 | 0.632588 | 0.243926 | 1.050157 | 1 sample 波动过大，不能作为最终结论。 |

趋势判断：

- **scan5 选点对 high-LR、only-stage1、MoE 是一致的**，说明这些模型的 checkpoint 选择相对可信。
- **group-balanced FM 越训 supervised 越低，但 CD-mean 越差**，说明它在优化不对应最终几何指标的方向。
- **full interaction 比 asymmetric/drop-RGB 更强**，至少在 stage1 100k 这个 checkpoint 上成立。
- **早期 1-batch 结果不能用于模型选择**，independent dual branch 的 1-batch best 过于乐观。

### 4.1 重复评估与协议差异

部分 checkpoint 被多次评估过，数值有小幅差异，主要来自 eval mode、候选 checkpoint 集合或评估脚本版本差异：

| 实验 | 评估文件 | Step | CD-mean | 说明 |
| --- | --- | ---: | ---: | --- |
| shared stage1 100k full | `eval_stage1_ckpt100k_A_full_ode12_b4_all.json` | 100000 | 0.518946 | 早一版 full eval 记录。 |
| shared stage1 100k full | `eval_exp1_stage1_ckpt100k_full_ode12_b4_all.json` | 100000 | 0.524995 | 本报告主表采用这一版 full eval。 |
| shared stage1 100k asymmetric | `eval_stage1_ckpt100k_B_asym_rgb1_hoi0_ode12_b4_all.json` | 100000 | 0.571574 | 早一版 asymmetric 记录。 |
| shared stage1 100k asymmetric | `eval_exp1_stage1_ckpt100k_asym_updated_ode12_b4_all.json` | 100000 | 0.571599 | 与 pre-update 记录一致，说明 asymmetric 结论稳定。 |
| shared two-stage final | `eval_wai_test_full_ode12_b4_step15000.json` | 15000 | 0.614557 | 单独评估 final checkpoint。 |
| shared two-stage selected | `eval_wai_test_full_ode12_b4_selected_11500_12500_14500.json` | 14500 | 0.597181 | selected full eval 更好，因此主表采用 14.5k。 |
| independent dual branch candidates | `dual_stream_eval_heldout_5batch_candidates_ode12.json` | 33000 | 0.821365 | 只评估候选 checkpoint。 |
| independent dual branch all scan | `dual_stream_eval_heldout_5batch_ode12.json` | 25000 | 0.804897 | 5-batch all scan 更好，因此趋势表采用 25k。 |

这些差异不改变主结论：当前最佳仍是 high-LR stage1-only；shared/full interaction 优于 asymmetric/drop-RGB；two-stage 和 group-balanced 没有超过 stage1 geometry 基线。

## 5. 早期基线、历史结果与 smoke 实验

| 实验 | 结果 | 作用 | 备注 |
| --- | --- | --- | --- |
| legacy stage2 single stage1-500step | CD=0.038532，F-score@0.01=0.415990，num_images=4552 | 旧 surface reconstruction pipeline 的历史基线 | 结果在 `results/behave-test__outputs_stage2_single_stage1-500step_*.json`，指标尺度与当前 HOI token CD-h/o/c 不同。 |
| legacy stage1-new single sample | CD=0.041088，F-score@0.01=0.387258，num_images=4554 | 旧 stage1-new 历史结果 | 不与当前 ODE12 heldout 直接比较。 |
| legacy stage2-contact | CD=0.054429，F-score@0.01=0.320002，num_images=4552 | 旧 contact refine 尝试 | 比 legacy stage2 更差，说明旧 contact refine 未带来收益。 |
| single-stream periodic | best step 3000，CD-mean=0.937069；final periodic 35000 为 1.138422 | 早期单流 checkpoint 趋势 | 一批 heldout 周期评估，显示 3k-4k 后过拟合。 |
| dense-contact periodic | best step 400，CD-mean=0.869926；final periodic 35000 为 1.204004 | 早期 dense-contact 趋势 | 比 single periodic best 好 7.17%，但最优极早。 |
| `.tmp/cointeract_shared_smoke` | 1 step 成功，loss=3.00379 | 验证 shared CoInteract 单卡流程 | 不作为质量结果。 |
| `outputs/test_stage1_launch` | 1 step 成功，loss=2.08409 | 验证 stage1 full attention 多卡启动 | 不作为质量结果。 |
| CARI4D supervised smoke | global_step=1，training_exit | 验证 CARI4D supervised 路径 | `tmp/cari4d_supervised_smoke/debug_status/rank_00.json`。 |
| ProciGen smoke inference | 8 frames，ODE2/ODE50 metadata 生成 | 验证 ProciGen dual-branch inference 输出 | 无 quantitative metric。 |
| ProciGen parallel preprocess | processed_total=2，status=completed | 验证并行预处理 | `tmp/procigen_parallel_smoke/_preprocess_logs/progress.json`。 |

## 6. HOI token imbalance 诊断

诊断文件：`outputs/ablation_hoi_token_balance/diagnostic_state_fm_imbalance.json`

850/850 Gaussians 时，state token 总数为 1728：

| Group | Token count | Token share | Uniform contribution share |
| --- | ---: | ---: | ---: |
| shape | 1 | 0.000579 | 0.000585 |
| pose | 1 | 0.000579 | 0.000584 |
| translation | 1 | 0.000579 | 0.000654 |
| object_motion | 1 | 0.000579 | 0.000559 |
| contact | 1 | 0.000579 | 0.000661 |
| human_gaussians | 850 | 0.491898 | 0.492595 |
| object_gaussians | 850 | 0.491898 | 0.490826 |
| joints | 22 | 0.012731 | 0.013534 |

诊断结论：

- uniform state FM 几乎完全由 human/object Gaussian tokens 主导。
- group-balanced 能让各 group contribution share 接近 12%-14%，但真实训练结果显示它没有改善 Chamfer，反而显著损害 CD-o。
- 因此问题不是“要不要平衡”，而是“如何平衡”。直接把所有 group 等权可能让低维状态项过度影响优化，破坏 Gaussian/object 几何。

下一步更合理的方向：

- 不直接继续 `state_fm_loss_mode=group_balanced`。
- 记录每个 group 的 gradient norm，而不只看 loss contribution。
- 只针对瓶颈项做轻量重加权，例如 object pose / object Gaussian，而不是把 shape、pose、contact、Gaussians 全部拉到同一贡献水平。

## 7. 关键失败模式

### 7.1 Supervised loss 与几何指标错配

对比：

| 实验 | CD-mean | supervised |
| --- | ---: | ---: |
| high-LR stage1-only | **0.484711** | 0.736174 |
| MoE | 0.504876 | 0.676226 |
| group-balanced FM | 0.627721 | **0.574851** |

supervised 越低不一定 CD 越好。当前训练目标里，显式状态项、Gaussian attr/xyz、Chamfer 和物理项之间的权重仍未对齐最终几何质量。

### 7.2 Object 分支仍是主要短板

当前最佳的分项指标：

| Metric | Best current value | 来自 |
| --- | ---: | --- |
| CD-h | 0.411127 | high-LR stage1-only |
| CD-o | 0.672753 | only-stage1 |
| CD-c | 0.370240 | high-LR stage1-only |

CD-o 明显高于其他两项，并且 high-LR 并没有明显改善 object。因此下一步改进应优先围绕 object pose/object Gaussian，而不是继续只改 human/contact。

### 7.3 Stage2 refine 没有复用好 stage1 的优势

stage2 constrained refine 从 only-stage1 初始化，理论上应提升几何细节，但结果：

| 对比 | Step | CD-mean | supervised |
| --- | ---: | ---: | ---: |
| only-stage1 init/best | 30000 | **0.497272** | 0.699590 |
| stage2 constrained refine | 12500 | 0.502635 | 2.986581 |

这说明 refine 阶段的权重设置或优化目标不稳定。尤其是 `lambda_gaussian_chamfer=1.5`、`lambda_phys_contact=0.05`、`lr=5e-6` 的组合没有实际改善 CD。

### 7.4 RGB prior 使用方式还需要单独确认

stage1 100k 的 eval mode 消融：

| Eval mode | CD-mean | CD-h | CD-o | CD-c |
| --- | ---: | ---: | ---: | ---: |
| full | **0.524995** | **0.408125** | **0.773684** | **0.393177** |
| asymmetric | 0.571599 | 0.497293 | 0.773751 | 0.443752 |
| drop RGB | 0.566143 | 0.459310 | 0.815339 | 0.423781 |

full inference 目前更强。若目标是最终支持 HOI-only inference，需要在最强 high-LR checkpoint 上重复这个消融，否则不要直接牺牲 RGB prior。

## 8. 下一步实验计划

建议按下面顺序执行，每一步都使用同一个全量评估协议：BEHAVE heldout 720 samples、ODE12、batch 4，并统一记录 CD-mean、CD-h、CD-o、CD-c、supervised。

| 优先级 | 实验 | 改动 | 目的 | 成功标准 |
| ---: | --- | --- | --- | --- |
| P0 | high-LR window full scan | 对 `cointeract_stage1_geometry_lr1e4_70k` 的 5k/7.5k/10k/12.5k/15k/20k 做全量 eval | 确认 10k 是否真是稳定最优点 | 找到 CD-mean <= 0.484711，或确认 10k 可固定为基线。 |
| P0 | high-LR eval mode ablation | 在 high-LR 10k 上跑 full/asymmetric/drop-RGB | 判断当前最佳是否依赖 RGB prior | 若 drop-RGB 接近 full，可优化 HOI-only；若差距大，继续保留 RGB。 |
| P1 | object-focused loss | 仅小幅提高 object pose/object Gaussian 相关权重，保持 state FM uniform | 改善 CD-o，不破坏 CD-h/CD-c | CD-o 明显低于 0.672753，CD-mean 不高于 0.484711。 |
| P1 | refined stage2 from high-LR 10k | 从 high-LR 10k 初始化，短程 5k-15k，避免 supervised 爆大；先使用 stage1 同权重，再逐项加入物理项 | 验证 refine 是否能真正提升高基线 | CD-mean 低于 high-LR 10k，且 supervised 不再到 2.x-3.x。 |
| P2 | MoE light ablation | 保留 MoE 架构但降低/关闭 router loss，或只在后半程启用 | 判断 MoE 是架构有害还是 router 监督有害 | 至少接近 high-LR；若仍 >0.50，停止 MoE。 |
| P2 | token count/object representation ablation | 850/850 vs 512/512 或 object-local normalized Gaussian tokens | 降低 token imbalance 与 object 拟合难度 | CD-o 下降且训练速度提升。 |
| P3 | group-balanced alternative | 不用硬等权，改为 gradient norm monitor 或 capped group reweight | 保留诊断收益，避免 CD-o 崩坏 | supervised 和 CD 同时改善，或至少 CD 不劣化。 |

## 9. 推荐的模型修改方向

### 9.1 先改 object 表示，不先改整体容量

当前瓶颈是 object CD-o。建议优先检查：

- object Gaussian 是否应使用 object-local canonical frame 后再预测；
- object pose 与 object Gaussian 的损失是否重复或冲突；
- object Gaussian Chamfer、xyz L1、attr L1 的权重是否需要分开调；
- object-specific normalization 是否与 human Gaussian normalization 一致。

### 9.2 让 checkpoint selection 与最终指标对齐

每次训练至少保留：

- scan5 用于快速定位窗口；
- 对 scan5 top 3 checkpoints 做 full 720 eval；
- 只用 full CD-mean 作为主排序；
- 同时记录 supervised 与分项 CD，避免只看 loss。

### 9.3 重做 stage2，而不是继续当前 stage2

当前 stage2 constrained refine 的问题很明确：没有优于 init，且 supervised 尺度失控。下一版 stage2 应该：

- 从 high-LR 10k 而不是 only-stage1 30k 初始化；
- 第一版不增加 `lambda_gaussian_chamfer` 和 `lambda_phys_contact`，只验证低 LR refine 是否稳定；
- 第二版再逐项加入 object-focused loss；
- 每 2.5k 做 scan5，top checkpoints 做 full eval。

### 9.4 暂停 group-balanced FM 的正式训练

group-balanced 的诊断是有价值的，但当前实现对应的训练结果不成立。下一步应转向：

- per-group gradient norm logging；
- capped reweighting；
- object/contact-only targeted balancing；
- 或者把 low-dimensional state token 的 FM loss 与 Gaussian token loss 分成两个 optimizer/loss schedule。

### 9.5 不急于扩展 MoE

MoE 当前比 only-stage1 和 high-LR 都差。除非要研究专家路由本身，否则短期不建议扩大专家维度或继续长训。更合理的是先做 router loss ablation，确认 `lambda_hoi_token_router=1.0` 是否过强。

## 10. 结果来源索引

主要报告来源：

- `design/model_test_results.md`
- `design/single_stream_dit_35k_results.md`
- `design/hoi_token_fm_imbalance_ablation.md`
- `outputs/hoi_baselines_current.json`
- `outputs/cointeract_stage1_geometry_lr1e4_70k/eval_wai_test_full_ode12_b4_current_best10000.json`
- `outputs/cointeract_stage1_geometry_50k/eval_wai_test_full_ode12_b4_best.json`
- `outputs/cointeract_stage2_constrained_refine_50k/eval_wai_test_full_ode12_b4_step12500_current_best.json`
- `outputs/cointeract_moe/eval_wai_test_full_ode12_b4_steps27500_70000.json`
- `outputs/cointeract_group_balanced_fm/eval_wai_test_full_ode12_b4_step40000.json`
- `outputs/cointeract_shared_stage1_full_100k/eval_exp1_stage1_ckpt100k_full_ode12_b4_all.json`
- `outputs/cointeract_shared_stage1_full_100k/eval_exp1_stage1_ckpt100k_asym_updated_ode12_b4_all.json`
- `outputs/cointeract_shared_stage1_full_100k/eval_stage1_ckpt100k_C_asym_no_rgb_prior_ode12_b4_all.json`
- `outputs/cointeract_shared_hoi_wan_two_stage/eval_wai_test_full_ode12_b4_selected_11500_12500_14500.json`
- `outputs/HOI_contact/test_metrics.json`
- `outputs/unimodel_wai_real_smoke/test_metrics.json`
- `outputs/ablation_hoi_token_balance/diagnostic_state_fm_imbalance.json`
- `results/behave-test__outputs_stage2_single_stage1-500step_None_2026-02-20T18-11-43.json`
- `results/behave-test__outputs_stage1-new_single_sample_None_2026-02-28T16-45-43.json`
- `results/behave-test__outputs_stage2-contact_single_stage1-contact-100step_None_2026-03-09T08-37-59.json`
