#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/config.yaml}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,5,6,7}"

DATASET="${DATASET:-procigen_train}"
TRAINING_STAGE="${TRAINING_STAGE:-stage1}"
LOSS_PRESET="${LOSS_PRESET:-${TRAINING_STAGE}}"
PROJECT_NAME="${PROJECT_NAME:-uni-hoi-4d}"
RUN_NAME="${RUN_NAME:-unihoi_${TRAINING_STAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"

RAW_ROOT="${RAW_ROOT:-/data4/guanz/data/ProciGen}"
PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/preprocessed/ProciGen_preprocessed_fixed}"
SPLIT_FILE="${SPLIT_FILE:-/data4/guanz/data/train-procigen-test-behave.pkl}"
SPLIT_KEY="${SPLIT_KEY:-train}"

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
MAX_STEPS="${MAX_STEPS:-150000}"
NUM_PROCESSES="${NUM_PROCESSES:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
BATCH_SAMPLER_MODE="${BATCH_SAMPLER_MODE:-sequence_grouped}"
SEQUENCE_BATCH_INTERLEAVE_WINDOW="${SEQUENCE_BATCH_INTERLEAVE_WINDOW:-1}"
SEQUENCE_BATCH_BURST="${SEQUENCE_BATCH_BURST:-1}"
FREEZE_VIDEO_BACKBONE="${FREEZE_VIDEO_BACKBONE:-true}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
BUILD_H5_CACHE="${BUILD_H5_CACHE:-false}"
H5_CACHE_CONTINUE_ON_ERROR="${H5_CACHE_CONTINUE_ON_ERROR:-true}"
H5_CACHE_OVERWRITE="${H5_CACHE_OVERWRITE:-false}"
H5_CACHE_MAX_SEQUENCES="${H5_CACHE_MAX_SEQUENCES:-}"
H5_CACHE_CHUNK_FRAMES="${H5_CACHE_CHUNK_FRAMES:-16}"

HONEST_VAL_EVERY="${HONEST_VAL_EVERY:-5000}"
HONEST_VAL_NUM_ODE_STEPS="${HONEST_VAL_NUM_ODE_STEPS:-50}"
HONEST_VAL_SEQUENCE="${HONEST_VAL_SEQUENCE:-}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-auto}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-unihoi_stage1}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-${REPO_ROOT}/outputs/${STAGE1_RUN_NAME}/checkpoints}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
INIT_CHECKPOINT_STRICT="${INIT_CHECKPOINT_STRICT:-false}"

LOG_FILE="${OUTPUT_DIR}/train.log"
CONFIG_SNAPSHOT="${OUTPUT_DIR}/launch_config.json"
PID_FILE="${OUTPUT_DIR}/train.pid"
H5_CACHE_LOG="${H5_CACHE_LOG:-${OUTPUT_DIR}/h5_cache_builder.log}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RESUME_CHECKPOINT}" == "auto" ]]; then
  RESUME_CHECKPOINT="$(ls -1t "${OUTPUT_DIR}"/checkpoints/checkpoint_*.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${INIT_CHECKPOINT}" && "${TRAINING_STAGE}" == "stage2" ]]; then
  INIT_CHECKPOINT="auto"
fi

if [[ "${INIT_CHECKPOINT}" == "auto" ]]; then
  INIT_CHECKPOINT="$(ls -1t "${INIT_CHECKPOINT_DIR}"/checkpoint_*.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "[train.sh] Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ -n "${INIT_CHECKPOINT}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "[train.sh] Init checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

cat > "${CONFIG_SNAPSHOT}" <<EOF
{
  "python_bin": "${PYTHON_BIN}",
  "config_file": "${CONFIG_FILE}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES_VALUE}",
  "dataset": "${DATASET}",
  "training_stage": "${TRAINING_STAGE}",
  "loss_preset": "${LOSS_PRESET}",
  "project_name": "${PROJECT_NAME}",
  "run_name": "${RUN_NAME}",
  "output_dir": "${OUTPUT_DIR}",
  "raw_root": "${RAW_ROOT}",
  "prepared_root": "${PREPARED_ROOT}",
  "split_file": "${SPLIT_FILE}",
  "split_key": "${SPLIT_KEY}",
  "batch_size": ${BATCH_SIZE},
  "lr": ${LR},
  "max_steps": ${MAX_STEPS},
  "num_processes": ${NUM_PROCESSES},
  "mixed_precision": "${MIXED_PRECISION}",
  "dataloader_num_workers": ${DATALOADER_NUM_WORKERS},
  "dataloader_prefetch_factor": ${DATALOADER_PREFETCH_FACTOR},
  "dataloader_pin_memory": ${DATALOADER_PIN_MEMORY},
  "dataloader_persistent_workers": ${DATALOADER_PERSISTENT_WORKERS},
  "batch_sampler_mode": "${BATCH_SAMPLER_MODE}",
  "sequence_batch_interleave_window": ${SEQUENCE_BATCH_INTERLEAVE_WINDOW},
  "sequence_batch_burst": ${SEQUENCE_BATCH_BURST},
  "freeze_video_backbone": ${FREEZE_VIDEO_BACKBONE},
  "wandb_enabled": ${WANDB_ENABLED},
  "skip_prepare": ${SKIP_PREPARE},
  "build_h5_cache": ${BUILD_H5_CACHE},
  "h5_cache_continue_on_error": ${H5_CACHE_CONTINUE_ON_ERROR},
  "h5_cache_overwrite": ${H5_CACHE_OVERWRITE},
  "h5_cache_max_sequences": "${H5_CACHE_MAX_SEQUENCES}",
  "h5_cache_chunk_frames": ${H5_CACHE_CHUNK_FRAMES},
  "h5_cache_log": "${H5_CACHE_LOG}",
  "honest_val_every": ${HONEST_VAL_EVERY},
  "honest_val_num_ode_steps": ${HONEST_VAL_NUM_ODE_STEPS},
  "honest_val_sequence": "${HONEST_VAL_SEQUENCE}",
  "resume_checkpoint": "${RESUME_CHECKPOINT}",
  "stage1_run_name": "${STAGE1_RUN_NAME}",
  "init_checkpoint_dir": "${INIT_CHECKPOINT_DIR}",
  "init_checkpoint": "${INIT_CHECKPOINT}",
  "init_checkpoint_strict": ${INIT_CHECKPOINT_STRICT}
}
EOF

cd "${REPO_ROOT}"

set -- \
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_dual_branch_fm.py" \
  --config "${CONFIG_FILE}" \
  --dataset "${DATASET}" \
  --run_name "${RUN_NAME}" \
  --project_name "${PROJECT_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --raw_root "${RAW_ROOT}" \
  --prepared_root "${PREPARED_ROOT}" \
  --split_file "${SPLIT_FILE}" \
  --split_key "${SPLIT_KEY}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --max_steps "${MAX_STEPS}" \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --batch_sampler_mode "${BATCH_SAMPLER_MODE}" \
  --sequence_batch_interleave_window "${SEQUENCE_BATCH_INTERLEAVE_WINDOW}" \
  --sequence_batch_burst "${SEQUENCE_BATCH_BURST}" \
  --honest_val_every "${HONEST_VAL_EVERY}" \
  --honest_val_num_ode_steps "${HONEST_VAL_NUM_ODE_STEPS}" \
  --training_stage "${TRAINING_STAGE}" \
  --loss_preset "${LOSS_PRESET}"

if [[ -n "${HONEST_VAL_SEQUENCE}" ]]; then
  set -- "$@" --honest_val_sequence "${HONEST_VAL_SEQUENCE}"
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  set -- "$@" --resume_checkpoint "${RESUME_CHECKPOINT}"
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  set -- "$@" --init_checkpoint "${INIT_CHECKPOINT}"
fi

if is_truthy "${WANDB_ENABLED}"; then
  set -- "$@" --wandb
else
  set -- "$@" --no-wandb
fi

if is_truthy "${SKIP_PREPARE}"; then
  set -- "$@" --no-prepare
else
  set -- "$@" --prepare
fi

if is_truthy "${BUILD_H5_CACHE}"; then
  set -- "$@" --build_h5_cache
else
  set -- "$@" --no-build_h5_cache
fi

if is_truthy "${DATALOADER_PIN_MEMORY}"; then
  set -- "$@" --dataloader_pin_memory
else
  set -- "$@" --no-dataloader_pin_memory
fi

if is_truthy "${DATALOADER_PERSISTENT_WORKERS}"; then
  set -- "$@" --dataloader_persistent_workers
else
  set -- "$@" --no-dataloader_persistent_workers
fi

if is_truthy "${H5_CACHE_CONTINUE_ON_ERROR}"; then
  set -- "$@" --h5_cache_continue_on_error
else
  set -- "$@" --no-h5_cache_continue_on_error
fi

if is_truthy "${H5_CACHE_OVERWRITE}"; then
  set -- "$@" --h5_cache_overwrite
else
  set -- "$@" --no-h5_cache_overwrite
fi

if [[ -n "${H5_CACHE_MAX_SEQUENCES}" ]]; then
  set -- "$@" --h5_cache_max_sequences "${H5_CACHE_MAX_SEQUENCES}"
fi

if [[ -n "${H5_CACHE_CHUNK_FRAMES}" ]]; then
  set -- "$@" --h5_cache_chunk_frames "${H5_CACHE_CHUNK_FRAMES}"
fi

if [[ -n "${H5_CACHE_LOG}" ]]; then
  set -- "$@" --h5_cache_log "${H5_CACHE_LOG}"
fi

if is_truthy "${INIT_CHECKPOINT_STRICT}"; then
  set -- "$@" --init_checkpoint_strict
else
  set -- "$@" --no-init_checkpoint_strict
fi

if is_truthy "${FREEZE_VIDEO_BACKBONE}"; then
  set -- "$@" --freeze_video_backbone
else
  set -- "$@" --no-freeze_video_backbone
fi

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" PYTHONUNBUFFERED=1 "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"

echo "[train.sh] Started training in background."
echo "[train.sh] PID            : ${PID}"
echo "[train.sh] Stage          : ${TRAINING_STAGE}"
echo "[train.sh] Output dir     : ${OUTPUT_DIR}"
echo "[train.sh] Log            : ${LOG_FILE}"
echo "[train.sh] Launch config  : ${CONFIG_SNAPSHOT}"
if is_truthy "${BUILD_H5_CACHE}"; then
  echo "[train.sh] H5 cache log   : ${H5_CACHE_LOG}"
fi
echo "[train.sh] Tail with      : tail -f ${LOG_FILE}"
