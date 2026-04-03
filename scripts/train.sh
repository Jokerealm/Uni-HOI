#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES_VALUE="2,3,4,5,6"
DATASET="procigen_train"
BATCH_SIZE="6"
LR="3e-4"
SPLIT_FILE="/data4/guanz/data/train-procigen-test-behave.pkl"
MAX_STEPS="150000"
NUM_PROCESSES="5"
PREPARE_NUM_WORKERS="10"
PROJECT_NAME="FM_model"
RUN_NAME="FM_model"
OUTPUT_DIR="${REPO_ROOT}/outputs/${RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-auto}"
HONEST_VAL_EVERY="5000"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
LOG_FILE="${OUTPUT_DIR}/train.log"
CONFIG_FILE="${OUTPUT_DIR}/config.json"
PID_FILE="${OUTPUT_DIR}/train.pid"

mkdir -p "${OUTPUT_DIR}"

if [ "${RESUME_CHECKPOINT}" = "auto" ]; then
  RESUME_CHECKPOINT="$(ls -1t "${OUTPUT_DIR}"/checkpoints/checkpoint_*.pt 2>/dev/null | head -n 1 || true)"
fi

if [ -n "${RESUME_CHECKPOINT}" ] && [ ! -f "${RESUME_CHECKPOINT}" ]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

cat > "${CONFIG_FILE}" <<EOF
{
  "python_bin": "${PYTHON_BIN}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES_VALUE}",
  "dataset": "${DATASET}",
  "batch_size": ${BATCH_SIZE},
  "lr": ${LR},
  "split_file": "${SPLIT_FILE}",
  "max_steps": ${MAX_STEPS},
  "num_processes": ${NUM_PROCESSES},
  "prepare_num_workers": ${PREPARE_NUM_WORKERS},
  "prepare": false,
  "wandb": true,
  "project_name": "${PROJECT_NAME}",
  "run_name": "${RUN_NAME}",
  "output_dir": "${OUTPUT_DIR}",
  "resume_checkpoint": "${RESUME_CHECKPOINT}",
  "mixed_precision": "${MIXED_PRECISION}",
  "wandb_init_timeout": ${WANDB_INIT_TIMEOUT},
  "honest_val_every": ${HONEST_VAL_EVERY},
  "log_file": "${LOG_FILE}"
}
EOF

cd "${REPO_ROOT}"

set -- \
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_dual_branch_fm.py" \
  --dataset "${DATASET}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --split_file "${SPLIT_FILE}" \
  --max_steps "${MAX_STEPS}" \
  --num_processes "${NUM_PROCESSES}" \
  --no-prepare \
  --wandb \
  --project_name "${PROJECT_NAME}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --honest_val_every "${HONEST_VAL_EVERY}"

if [ -n "${RESUME_CHECKPOINT}" ]; then
  set -- "$@" --resume_checkpoint "${RESUME_CHECKPOINT}"
fi

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" PREPARE_NUM_WORKERS="${PREPARE_NUM_WORKERS}" MIXED_PRECISION="${MIXED_PRECISION}" WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT}" \
  "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"

echo "Started training in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Config: ${CONFIG_FILE}"
echo "Tail with: tail -f ${LOG_FILE}"
