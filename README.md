# Uni-HOI 4D: CoInteract-Style RGB to HOI Wan Experiment

This branch keeps the existing HDM/ProciGen/BEHAVE dataloader and preprocessing assets, but replaces the old training code with a new experimental model:

- RGB stream: RGB frame/video latents through the frozen Wan visual prior.
- HOI stream: explicit per-clip HOI tokens for human/object Gaussians, joints, SMPL pose/shape/translation, object SE(3), and contact in a human-relative coordinate frame.
- Fusion: HOI-primary CoInteract-style shared DiT stream. HOI and RGB tokens share the attention/MLP block weights; modality separation is kept in patch/projection layers and stream-specific AdaLN scale/shift modulation. RGB hidden tokens stay auxiliary and guide HOI through zero-init cross-stream adapters. By default the Wan RGB stream is frozen/detached and the retained supervised output is HOI-side.
- Default schedule: short sample/smoke runs use constant-with-warmup learning rate and select step count from effective dataset passes.

## Train

```bash
DATA_ROOT=/path/to/prepared/sequences \
OUTPUT_DIR=outputs/unimodel_wai_real_smoke \
MAX_STEPS=800 \
LR=1e-4 \
WARMUP_STEPS=50 \
LR_SCHEDULER=constant \
./train.sh
```

The wrapper uses `/data3/guanz/miniforge3/envs/cari4d/bin/python` by default. Override with `PYTHON_BIN=/path/to/python`.

Useful overrides:

```bash
WAN_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
DATA_ROOT=/path/to/prepared/sequences \
LOG_WITH=wandb \
CLIP_LENGTH=5 \
scripts/train_cointeract.sh --batch_size 1 --num_human_gaussians 750 --num_object_gaussians 750
```

For the shared-parameter WAI run plus heldout/test evaluation:

```bash
scripts/run_wai_shared_train_test_once.sh
```

Dual-stream controls:

- CoInteract uses full RGB/HOI shared attention throughout training: RGB/main tokens and HOI-depth tokens are concatenated into one shared DiT sequence with full visibility.
- DiT blocks use stream-specific AdaLN scale/shift plus learned residual gates to modulate interaction strength by block depth.
- The HOI-token-aware MoE is enabled in the HOI FFN path. The shared/base expert reuses the original DiT FFN; lightweight residual experts specialize for human pose/joints, object motion/object Gaussians, contact, and human surface Gaussians. Router supervision uses stop-gradient hidden states before the router.
- `MODEL_VARIANT=wan_backbone` switches to an experimental CoInteract-style model that concatenates HOI state tokens directly into the Wan2.2-TI2V DiT token sequence and reuses the pretrained Wan transformer blocks instead of the local 8-layer 512-dim DiT. The HOI token dimension is `wan_hidden_dim` (3072 by default), and the number of Wan blocks is read from the loaded pretrained transformer.
- `train.sh` writes to `outputs/cointeract_shared_hoi_wan_full` by default.
- Inference can set `use_rgb_prior=False` in `CoInteractHOI4DModel.forward` or `--drop_rgb_branch` in the checkpoint eval script to delete the RGB/Wan branch and run the HOI stream alone.
- `--no-detach_rgb_context`: allow HOI loss to update the Wan RGB branch when `--no-freeze_wan` is also used. The default keeps the RGB video prior as a fixed collaborator.

Wan-backbone smoke command:

```bash
MODEL_VARIANT=wan_backbone \
OUTPUT_DIR=outputs/cointeract_wan_backbone_ti2v5b \
BATCH_SIZE=1 \
MAX_STEPS=1000 \
LR=1e-5 \
scripts/train_cointeract.sh
```

For single-image sample data, choose `MAX_STEPS` by effective passes rather than paper-scale step counts:

```text
effective_epochs ~= MAX_STEPS * (batch_size * num_gpus * grad_accum) / num_samples
```

As a starting point, use roughly 200-500 effective passes for smoke/sample runs, then inspect saved renders/checkpoints. For larger prepared datasets, move toward 3K-7K optimization steps, keeping `LR_SCHEDULER=constant` for exploratory runs or `LR_SCHEDULER=cosine MIN_LR_RATIO=0.1` for final convergence.

## Code Kept Intentionally

The previous task dataloader is preserved in `dataset/dual_branch_fm_dataset.py`. Preprocessing and data preparation code under `pipeline/`, `behave/`, `unidepth/`, and preprocessing scripts are left intact.

The active model and training entrypoint are:

- `model/cointeract_hoi_wan.py`
- `model/cointeract_wan_backbone_hoi.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract.sh`
- `train.sh`
