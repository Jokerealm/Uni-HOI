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

- `--rgb_to_hoi_scale`: strength of RGB auxiliary tokens guiding the HOI main stream.
- `--hoi_to_rgb_scale`: optional reverse adapter for symmetric experimentation; default `0.0` keeps the CoInteract-style asymmetric direction but with HOI as the retained main stream.
- `train.sh` defaults to a CoInteract-inspired two-stage schedule: `STAGE1_FULL_ATTENTION_STEPS=10000` with `STAGE1_HOI_TO_RGB_SCALE=1.0`, then asymmetric Stage 2 with `HOI_TO_RGB_SCALE=0.0` until `MAX_STEPS=15000`. It writes to `outputs/cointeract_shared_hoi_wan_two_stage`, uses offline W&B by default, and sets `SAVE_EVERY=500` and `TRAIN_VISUAL_EVERY=500`.
- `--no-detach_rgb_context`: allow HOI loss to update the Wan RGB branch when `--no-freeze_wan` is also used. The default keeps the RGB video prior as a fixed collaborator.

For single-image sample data, choose `MAX_STEPS` by effective passes rather than paper-scale step counts:

```text
effective_epochs ~= MAX_STEPS * (batch_size * num_gpus * grad_accum) / num_samples
```

As a starting point, use roughly 200-500 effective passes for smoke/sample runs, then inspect saved renders/checkpoints. For larger prepared datasets, move toward 3K-7K optimization steps, keeping `LR_SCHEDULER=constant` for exploratory runs or `LR_SCHEDULER=cosine MIN_LR_RATIO=0.1` for final convergence.

## Code Kept Intentionally

The previous task dataloader is preserved in `dataset/dual_branch_fm_dataset.py`. Preprocessing and data preparation code under `pipeline/`, `behave/`, `unidepth/`, and preprocessing scripts are left intact.

The active model and training entrypoint are:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract.sh`
- `train.sh`
