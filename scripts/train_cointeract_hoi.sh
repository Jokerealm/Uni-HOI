#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/sample_data/behave_1pct/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cointeract_hoi_wan_ti2v}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/train_cointeract_hoi.py" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --wan_model_id "${WAN_MODEL_ID}" \
  --clip_length "${CLIP_LENGTH:-9}" \
  --clip_stride "${CLIP_STRIDE:-8}" \
  --max_steps "${MAX_STEPS:-7000}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --lr "${LR:-2e-4}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --log_with "${LOG_WITH:-none}" \
  --num_workers "${NUM_WORKERS:-2}" \
  "$@"
