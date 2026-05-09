#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/accelerate}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/sample_data/behave_1pct/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cointeract_hoi_wan_ti2v}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

TRAIN_ARGS=(
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --wan_model_id "${WAN_MODEL_ID}" \
  --clip_length "${CLIP_LENGTH:-9}" \
  --clip_stride "${CLIP_STRIDE:-8}" \
  --max_steps "${MAX_STEPS:-7000}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --lr "${LR:-2e-4}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --log_with "${LOG_WITH:-none}" \
  --project_name "${WANDB_PROJECT:-uni-hoi-4d}" \
  --run_name "${RUN_NAME:-cointeract_rgb_to_hoi_wan_ti2v}" \
  --train_visual_every "${TRAIN_VISUAL_EVERY:-500}" \
  --num_workers "${NUM_WORKERS:-2}" \
  "$@"
)

if [[ "${NUM_PROCESSES:-1}" -gt 1 ]]; then
  exec "${ACCELERATE_BIN}" launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${REPO_ROOT}/train_cointeract_hoi.py" \
    "${TRAIN_ARGS[@]}"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/train_cointeract_hoi.py" "${TRAIN_ARGS[@]}"
