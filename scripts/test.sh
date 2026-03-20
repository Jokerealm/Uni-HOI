#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
RAW_ROOT="${RAW_ROOT:-/data4/guanz/data/ProciGen}"
PREPARED_ROOT="${PREPARED_ROOT:-$RAW_ROOT}"
SPLIT_FILE="${SPLIT_FILE:-/data4/guanz/data/train-procigen-test-behave.pkl}"
SPLIT_KEY="${SPLIT_KEY:-train}"
CAMERA_ID="${CAMERA_ID:-k1}"

VIDEO_NAME="${VIDEO_NAME:-Date04_Subxx_toolbox_synzv2-10}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/procigen_dual_branch_fm/checkpoints}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_ODE_STEPS="${NUM_ODE_STEPS:-50}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-amodal}"
GS_OUTPUT_SUBDIR="${GS_OUTPUT_SUBDIR:-gs_init_pred}"
SAVE_FRAMES="${SAVE_FRAMES:-1}"

PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed}"
GS_SUBDIR="${GS_SUBDIR:-gs_init}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
MAX_PREPARE_FRAMES="${MAX_PREPARE_FRAMES:-0}"

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls -1t "${CHECKPOINT_DIR}"/checkpoint_*.pt 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[test.sh] Could not resolve a checkpoint. Set CHECKPOINT or CHECKPOINT_DIR." >&2
  exit 1
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  PREP_ARGS=(
    scripts/preprocess_procigen_gt.py
    --raw_root "${RAW_ROOT}"
    --output_root "${PREPARED_ROOT}"
    --sequence_name "${VIDEO_NAME}"
    --camera_id "${CAMERA_ID}"
    --processed_subdir "${PROCESSED_SUBDIR}"
    --gs_subdir "${GS_SUBDIR}"
  )
  if [[ "${MAX_PREPARE_FRAMES}" != "0" ]]; then
    PREP_ARGS+=(--max_frames "${MAX_PREPARE_FRAMES}")
  fi
  if [[ "${OVERWRITE_PREPARE}" == "1" ]]; then
    PREP_ARGS+=(--overwrite)
  fi

  echo "[test.sh] Preparing sequence assets for ${VIDEO_NAME}..."
  "${PYTHON_BIN}" "${PREP_ARGS[@]}"
fi

echo "============================================================"
echo "  ProciGen New Pipeline Inference"
echo "  Sequence    : ${VIDEO_NAME}"
echo "  Prepared root: ${PREPARED_ROOT}"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "============================================================"

SAVE_FLAG="--save_frames"
if [[ "${SAVE_FRAMES}" == "0" ]]; then
  SAVE_FLAG="--no-save_frames"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" infer_dual_branch_fm.py \
  --input_dir "${PREPARED_ROOT}" \
  --video_name "${VIDEO_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --processed_subdir "${PROCESSED_SUBDIR}" \
  --gs_subdir "${GS_SUBDIR}" \
  --output_subdir "${OUTPUT_SUBDIR}" \
  --gs_output_subdir "${GS_OUTPUT_SUBDIR}" \
  --num_ode_steps "${NUM_ODE_STEPS}" \
  --device "${DEVICE}" \
  "${SAVE_FLAG}"
