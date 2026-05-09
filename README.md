# Uni-HOI 4D: CoInteract-Style RGB to HOI Wan Experiment

This branch keeps the existing HDM/ProciGen/BEHAVE dataloader and preprocessing assets, but replaces the old training code with a new experimental model:

- RGB stream: Wan2.2-TI2V-5B, image-only input, empty text prompt.
- HOI stream: explicit 4D HOI tokens for human/object Gaussians, joints, SMPL pose/shape/translation, object SE(3), and contact.
- Fusion: CoInteract-style RGB hidden tokens guide the HOI stream. By default the Wan RGB stream is frozen and gradients update only the HOI stream.
- Default schedule: 7000 optimization steps for fast method validation.

## Train

```bash
scripts/train.sh \
  --data_root /path/to/prepared/sequences \
  --output_dir outputs/cointeract_hoi_wan_ti2v \
  --max_steps 7000
```

The wrapper uses `/data3/guanz/miniforge3/envs/cari4d/bin/python` by default. Override with `PYTHON_BIN=/path/to/python`.

Useful overrides:

```bash
WAN_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
DATA_ROOT=/path/to/prepared/sequences \
LOG_WITH=wandb \
scripts/train.sh --batch_size 1 --clip_length 9 --num_human_gaussians 750 --num_object_gaussians 750
```

## Code Kept Intentionally

The previous task dataloader is preserved in `dataset/dual_branch_fm_dataset.py`. Preprocessing and data preparation code under `pipeline/`, `behave/`, `unidepth/`, and preprocessing scripts are left intact.

The active model and training entrypoint are:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract_hoi.sh`
- `scripts/train.sh`
