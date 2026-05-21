#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/accelerate}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LOG_FILE="${LOG_FILE:-logs/output.log}"
NOHUP="${NOHUP:-1}"
LOG_WITH="${LOG_WITH:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-uni-hoi-4d}"
RUN_NAME="${RUN_NAME:-unimodel_wai_dit}"

IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#CUDA_DEVICE_IDS[@]}}"
MODE="${MODE:-train_test}"
DATA_ROOT="${DATA_ROOT:-sample_data/WAI_prepared/sequences}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-sample_data/BEHAVE_heldout_prepared/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/unimodel_wai_real_smoke}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_STEPS="${MAX_STEPS:-35000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"
TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY:-500}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
TEST_EVERY="${TEST_EVERY:-100}"
TEST_MAX_BATCHES="${TEST_MAX_BATCHES:-0}"
PERIODIC_TEST_MAX_BATCHES="${PERIODIC_TEST_MAX_BATCHES:-1}"
TEST_SAVE_BATCHES="${TEST_SAVE_BATCHES:-1}"

MAIN_ARGS=(
  --mode "${MODE}"
  --data_root "${DATA_ROOT}"
  --test_data_root "${TEST_DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --max_sequences "${MAX_SEQUENCES}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --log_every "${LOG_EVERY}"
  --log_with "${LOG_WITH}"
  --project_name "${WANDB_PROJECT}"
  --run_name "${RUN_NAME}"
  --save_every "${SAVE_EVERY}"
  --train_visual_every "${TRAIN_VISUAL_EVERY}"
  --clip_length "${CLIP_LENGTH:-1}"
  --clip_stride "${CLIP_STRIDE:-1}"
  --coordinate_mode "${COORDINATE_MODE:-relative}"
  --lr "${LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --lr_scheduler "${LR_SCHEDULER}"
  --min_lr_ratio "${MIN_LR_RATIO}"
  --test_every "${TEST_EVERY}"
  --test_max_batches "${TEST_MAX_BATCHES}"
  --periodic_test_max_batches "${PERIODIC_TEST_MAX_BATCHES}"
  --test_save_batches "${TEST_SAVE_BATCHES}"
  --no-pin_memory
  --render_test_predictions
  "$@"
)

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  RUN_CMD=(
    "${ACCELERATE_BIN}" launch
    --multi_gpu
    --num_processes "${NUM_PROCESSES}"
    --num_machines 1
    --mixed_precision "${MIXED_PRECISION}"
    --dynamo_backend no
    main.py
    "${MAIN_ARGS[@]}"
  )
else
  RUN_CMD=("${PYTHON_BIN}" main.py "${MAIN_ARGS[@]}")
fi

mkdir -p "$(dirname "${LOG_FILE}")"

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  LAUNCH_MODE="accelerate"
else
  LAUNCH_MODE="python"
fi

{
  echo "============================================================"
  echo "launch_time=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "launch_mode=${LAUNCH_MODE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  printf 'command='
  printf '%q ' "${RUN_CMD[@]}"
  echo
} > "${LOG_FILE}"

if [[ "${NOHUP}" == "1" ]]; then
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "${RUN_CMD[@]}" >> "${LOG_FILE}" 2>&1 < /dev/null &
  else
    nohup "${RUN_CMD[@]}" >> "${LOG_FILE}" 2>&1 < /dev/null &
  fi
  PID=$!
  echo "started training with nohup | pid=${PID} | log=${LOG_FILE}"
else
  "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
