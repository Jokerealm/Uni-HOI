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
| `main.py` | 主流程调度入口 |

## 3. 训练命令

```bash
GPU_ID=0 BATCH_SIZE=4 LR=1.5e-4 scripts/train.sh
```

详细超参数默认从 `scripts/train_dual_branch_fm.opt` 读取。

常见改法：

```bash
GPU_ID=1 BATCH_SIZE=8 LR=2e-4 MIXED_PRECISION=fp16 scripts/train.sh
OPT_FILE=/abs/path/to/another_train.opt scripts/train.sh
```

关键默认值：

- `patch_size=16`
- `hidden_dim=384`
- `depth=6`
- `num_heads=6`
- `human_gaussian_source=smpl_mesh`
- `num_human_gaussians=750`
- `num_object_gaussians=750`
- `lr=1.5e-4`
- `mixed_precision=bf16`

课程式 loss 相关参数：

- `curriculum_fusion_start_ratio=0.2`
- `curriculum_full_start_ratio=0.6`

两阶段视频主干训练参数：

- `freeze_video_backbone=true`
- `video_unfreeze_start_ratio=-1`
- `video_stage2_num_top_blocks=2`

含义：

- 前 `20%` step 主要做 warmup
- `20% -> 60%` 逐渐打开 fusion losses
- `60% -> 100%` 逐渐打开 full losses
- `video_unfreeze_start_ratio=-1` 表示默认跟随 `curriculum_fusion_start_ratio`
- 到第二阶段时，解冻最顶层的 `2` 个 `video_block`

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
/data3/guanz/miniforge3/envs/cari4d/bin/python main.py \
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
