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
RAW_ROOT="${RAW_ROOT:-/data4/guanz/data/ProciGen}"
PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/preprocessed/ProciGen_preprocessed_fixed}"
CAMERA_ID="${CAMERA_ID:-k1}"

VIDEO_NAME="${VIDEO_NAME:-Date04_Subxx_toolbox_synzv2-10}"
RUN_NAME="${RUN_NAME:-unihoi_stage1}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}/checkpoints}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_ODE_STEPS="${NUM_ODE_STEPS:-50}"
PRIOR_NOISE_STD="${PRIOR_NOISE_STD:-1.0}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-amodal}"
GS_OUTPUT_SUBDIR="${GS_OUTPUT_SUBDIR:-gs_init_pred}"
SAVE_FRAMES="${SAVE_FRAMES:-1}"
SAVE_FPS="${SAVE_FPS:-24}"
CLAMP_VISIBLE_RGB="${CLAMP_VISIBLE_RGB:-1}"

PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed}"
GS_SUBDIR="${GS_SUBDIR:-gs_init}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
MAX_PREPARE_FRAMES="${MAX_PREPARE_FRAMES:-0}"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-uni-hoi-4d}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ARTIFACT_NAME="${WANDB_ARTIFACT_NAME:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls -1t "${CHECKPOINT_DIR}"/checkpoint_*.pt 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[test.sh] Could not resolve a checkpoint. Set CHECKPOINT or CHECKPOINT_DIR." >&2
  exit 1
fi

if ! is_truthy "${SKIP_PREPARE}"; then
  PREP_ARGS=(
    "${REPO_ROOT}/scripts/preprocess_procigen_gt.py"
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
  if is_truthy "${OVERWRITE_PREPARE}"; then
    PREP_ARGS+=(--overwrite)
  fi

  echo "[test.sh] Preparing sequence assets for ${VIDEO_NAME}..."
  "${PYTHON_BIN}" "${PREP_ARGS[@]}"
fi

SAVE_FLAG="--save_frames"
if ! is_truthy "${SAVE_FRAMES}"; then
  SAVE_FLAG="--no-save_frames"
fi

CLAMP_FLAG="--clamp_visible_rgb"
if ! is_truthy "${CLAMP_VISIBLE_RGB}"; then
  CLAMP_FLAG="--no-clamp_visible_rgb"
fi

WANDB_FLAG="--no-wandb"
if is_truthy "${WANDB_ENABLED}"; then
  WANDB_FLAG="--wandb"
fi

echo "============================================================"
echo "  Uni-HOI Dual Amodal Inference"
echo "  Sequence      : ${VIDEO_NAME}"
echo "  Prepared root : ${PREPARED_ROOT}"
echo "  Checkpoint    : ${CHECKPOINT}"
echo "  ODE steps     : ${NUM_ODE_STEPS}"
echo "  Outputs       : ${OUTPUT_SUBDIR}/human_amodal + ${OUTPUT_SUBDIR}/object_amodal"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${REPO_ROOT}/infer_dual_branch_fm.py" \
  --input_dir "${PREPARED_ROOT}" \
  --video_name "${VIDEO_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --processed_subdir "${PROCESSED_SUBDIR}" \
  --gs_subdir "${GS_SUBDIR}" \
  --output_subdir "${OUTPUT_SUBDIR}" \
  --gs_output_subdir "${GS_OUTPUT_SUBDIR}" \
  --num_ode_steps "${NUM_ODE_STEPS}" \
  --prior_noise_std "${PRIOR_NOISE_STD}" \
  --save_fps "${SAVE_FPS}" \
  --device "${DEVICE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_name "${WANDB_NAME}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_artifact_name "${WANDB_ARTIFACT_NAME}" \
  "${SAVE_FLAG}" \
  "${CLAMP_FLAG}" \
  "${WANDB_FLAG}"
