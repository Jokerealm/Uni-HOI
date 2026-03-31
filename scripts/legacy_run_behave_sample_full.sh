#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
INPUT_DIR="${INPUT_DIR:-/data4/guanz/coding/HDM/sample_data/behave_1pct/sequences}"
VIDEO_NAME="${1:-Date03_Sub03_chairblack_sitstand}"
MAX_FRAMES="${MAX_FRAMES:-12}"
NUM_ITERS="${NUM_ITERS:-20}"
SAVE_EVERY="${SAVE_EVERY:-10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" /data4/guanz/coding/HDM/legacy_pipeline.py \
  dataset=custom \
  run.job=full \
  data_prep.input_dir="${INPUT_DIR}" \
  data_prep.video_name="${VIDEO_NAME}" \
  data_prep.max_frames="${MAX_FRAMES}" \
  step4.num_iters="${NUM_ITERS}" \
  step4.save_every="${SAVE_EVERY}" \
  wandb.enabled=false
