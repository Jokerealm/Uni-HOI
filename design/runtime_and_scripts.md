# Runtime And Scripts

## 1. 推荐环境

当前代码实际验证使用：

- Python: `/data3/guanz/miniforge3/envs/cari4d/bin/python`
- CUDA: 需要可用 GPU 才能真正训练
- PyTorch: 需要支持 `scaled_dot_product_attention`

建议要求：

- PyTorch `2.1+`
- CUDA `11.8+`
- mixed precision 使用 `fp16` 或 `bf16`

如果想真正走 FlashAttention 路径，需要满足：

- GPU 支持 flash SDP
- 输入 dtype 为 half precision
- attention 调用走 `torch.nn.functional.scaled_dot_product_attention`

## 2. 主要脚本

| 脚本 | 作用 |
|---|---|
| `train_dual_branch_fm.py` | 训练双分支联合 FM 主干 |
| `infer_dual_branch_fm.py` | 对单个 sequence 做 Step2 联合推理 |
| `pipeline/step2_dual_branch_flow_matching.py` | Hydra Step2 包装 |
| `legacy_pipeline.py` | legacy Step1~Step5 主流程调度入口 |

## 3. 训练命令

```bash
scripts/train.sh
```

当前 ProciGen 双分支训练推荐统一走：

- 入口脚本：`scripts/train.sh`
- 配置文件：`configs/config.yaml`
- 实际启动器：`scripts/run_dual_branch_fm.py`

`scripts/train.sh` 里放了一组最常改的参数；更细的默认值仍放在 `configs/config.yaml` 的 `dual_branch_fm` 段里。
GPU 选择统一交给外部环境变量 `CUDA_VISIBLE_DEVICES`，多卡时内部自动走：

```bash
python -m torch.distributed.run --nproc_per_node N ...
```

常用覆盖参数可以直接写在命令行：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/train.sh \
  --dataset procigen_train \
  --run_name fm_debug \
  --max_steps 2000 \
  --lr 2e-4 \
  --num_processes 2
```

如果要改路径、缓存、模型宽度、loss 权重等，直接修改：

```bash
configs/config.yaml
```

`scripts/train.sh --help` 当前支持直接覆盖：

- `--lr`
- `--batch_size`
- `--max_steps`
- `--dataset`
- `--prepared_root`
- `--num_processes`
- `--prepare/--no-prepare`
- `--run_name`
- `--wandb/--no-wandb`
- `--split_file`

关键默认值位于 `configs/config.yaml` 的 `dual_branch_fm.train`：

- `patch_size=16`
- `hidden_dim=384`
- `depth=6`
- `num_heads=6`
- `human_gaussian_source=smpl_mesh`
- `num_human_gaussians=750`
- `num_object_gaussians=750`
- `lr=1.5e-4`
- `mixed_precision=no`
- `loss_preset=core`

训练实现本身不是旧的 `step1-5` 训练图，而是统一双分支模型，内部带课程式阶段：

- `loss_preset=core|stage0|full`
- `curriculum_fusion_start_ratio=0.2`
- `curriculum_full_start_ratio=0.6`
- `freeze_video_backbone=true`
- `video_unfreeze_start_ratio=-1`
- `video_stage2_num_top_blocks=2`

## 4. Step2 推理命令

直接脚本：

```bash
/data3/guanz/miniforge3/envs/cari4d/bin/python infer_dual_branch_fm.py \
  --input_dir sample_data/behave_1pct/sequences \
  --video_name Date03_Sub03_chairblack_sitstand \
  --checkpoint /abs/path/to/checkpoint.pt \
  --num_ode_steps 50
```

通过主 pipeline：

```bash
/data3/guanz/miniforge3/envs/cari4d/bin/python legacy_pipeline.py \
  run.job=step2 \
  amodal.method=dual_branch_flow_matching \
  fm.checkpoint=/abs/path/to/checkpoint.pt
```

## 5. 数据前提

训练需要：

- `processed/cropped/rgb`
- `processed/cropped/masks_raw.npz`
- `processed/cropped/region_masks.npz`
- `processed/cropped/depth_aligned.npz`
- `processed/cropped/meta.npz`
- `processed/smpl_params.npz`
- human 侧默认从 `smpl_params.npz` 的 `vertices/faces` 生成 SMPL-anchored 伪真值
- `gs_init/G_o.pt` 或 `gs_init_combined.pt`

推理只需要 Step1 资产：

- `processed/cropped/*`
- `processed/smpl_params.npz`
- object pose sequence

推理已不再强依赖 teacher GS。

## 6. 显存注意点

当前主显存来源：

- video branch spatial/temporal attention
- state branch token transformer
- object render consistency
- geometry distillation

如果显存不足，优先调整：

1. `batch_size`
2. `clip_length`
3. `hidden_dim`
4. `depth`

不要先把 `patch_size` 改回 `8`，因为这会把视频 token 从 `2048` 直接拉回 `8192`。

## 7. 调试建议

第一轮看这些日志：

- `loss_video_fm`
- `loss_state_fm`
- `loss_object_video`
- `loss_joints`
- `curriculum_stage`
- `video_optim_stage`
- `video_unfrozen_blocks`
- `curriculum_fusion_progress`
- `curriculum_full_progress`

如果 warmup 阶段已经发散，先不要开更多几何约束，优先检查：

- Step1 条件张量是否对齐
- teacher object render 是否正常
- camera intrinsics 是否在 resize 后同步缩放
