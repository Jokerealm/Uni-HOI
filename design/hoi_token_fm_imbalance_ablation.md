# HOI Baseline Registry

Date: 2026-06-01

This file keeps only the current baseline set.

## Active Baselines

All rows below use the same heldout full-eval protocol:

```text
data_root: sample_data/BEHAVE_heldout_prepared/sequences
num_samples: 720
batch_size: 4
num_ode_steps: 12
selection: best checkpoint by heldout scan/full result available for that run
```

| Baseline | Output | Best step | CD-mean | CD-h | CD-o | CD-c | supervised |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high-LR stage1-only current best | `outputs/cointeract_stage1_geometry_lr1e4_70k` | 10k | **0.484711** | **0.411127** | 0.672766 | **0.370240** | 0.736174 |
| only stage1 | `outputs/cointeract_stage1_geometry_50k` | 30k | 0.497272 | 0.437633 | **0.672753** | 0.381429 | 0.699590 |
| MoE | `outputs/cointeract_moe` | 27.5k | 0.504876 | 0.440246 | 0.688721 | 0.385661 | **0.676226** |

## Baseline Decision

The current baseline to keep is **high-LR stage1-only current best** at step
10k. It is the best Chamfer result among the active baselines:

| Comparison | Delta CD-mean | Interpretation |
| --- | ---: | --- |
| high-LR stage1-only minus only stage1 | -0.012561 | high-LR early checkpoint improves heldout Chamfer. |
| high-LR stage1-only minus MoE | -0.020165 | MoE is not competitive under the matched full-eval protocol. |

The trade-off is that high-LR has worse supervised loss than MoE and only
stage1, so it should be treated as the current Chamfer baseline rather than a
strictly better semantic-state baseline.

## Artifacts

| Baseline | Full eval JSON |
| --- | --- |
| high-LR stage1-only current best | `outputs/cointeract_stage1_geometry_lr1e4_70k/eval_wai_test_full_ode12_b4_current_best10000.json` |
| only stage1 | `outputs/cointeract_stage1_geometry_50k/eval_wai_test_full_ode12_b4_best.json` |
| MoE | `outputs/cointeract_moe/eval_wai_test_full_ode12_b4_steps27500_70000.json` |
