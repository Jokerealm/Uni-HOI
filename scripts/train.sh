#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPT_FILE="${OPT_FILE:-${SCRIPT_DIR}/train_dual_branch_fm.opt}"

if [[ ! -f "${OPT_FILE}" ]]; then
  echo "[train.sh] Could not find OPT_FILE: ${OPT_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${OPT_FILE}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
RAW_ROOT="${RAW_ROOT:-/data4/guanz/data/ProciGen}"
PREPARED_ROOT="${PREPARED_ROOT:-$RAW_ROOT}"
SPLIT_FILE="${SPLIT_FILE:-/data4/guanz/data/train-procigen-test-behave.pkl}"
SPLIT_KEY="${SPLIT_KEY:-train}"
CAMERA_ID="${CAMERA_ID:-k1}"

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
if (( CPU_COUNT <= 2 )); then
  DEFAULT_PREPARE_WORKERS=1
elif (( CPU_COUNT <= 8 )); then
  DEFAULT_PREPARE_WORKERS=$((CPU_COUNT / 2))
else
  DEFAULT_PREPARE_WORKERS=8
fi

PROCESSED_SUBDIR="${PROCESSED_SUBDIR:-processed}"
GS_SUBDIR="${GS_SUBDIR:-gs_init}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
PREPARE_MAX_SEQUENCES="${PREPARE_MAX_SEQUENCES:-0}"
PREPARE_MAX_FRAMES="${PREPARE_MAX_FRAMES:-0}"
PREPARE_NUM_WORKERS="${PREPARE_NUM_WORKERS:-$DEFAULT_PREPARE_WORKERS}"
PREPARE_HEARTBEAT_INTERVAL="${PREPARE_HEARTBEAT_INTERVAL:-30}"
STATUS_DIR="${STATUS_DIR:-${PREPARED_ROOT}/_preprocess_logs}"
DETACH_PREPARE="${DETACH_PREPARE:-0}"

# Common runtime overrides. Detailed model/training defaults live in OPT_FILE.
RUN_NAME="${RUN_NAME:-procigen_dual_branch_fm}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
GPU_ID="${GPU_ID:-0}"
GPU_ID_CLEAN="${GPU_ID// /}"
BATCH_SIZE="${BATCH_SIZE:-${OPT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-${OPT_NUM_WORKERS}}"
MAX_STEPS="${MAX_STEPS:-${OPT_MAX_STEPS}}"
LR="${LR:-${OPT_LR}}"
MIXED_PRECISION="${MIXED_PRECISION:-${OPT_MIXED_PRECISION}}"
LOG_WITH="${LOG_WITH:-${OPT_LOG_WITH}}"
SEED="${SEED:-${OPT_SEED}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OPT_RESUME_CHECKPOINT}}"
DIST_NUM_PROCESSES="${DIST_NUM_PROCESSES:-0}"
DIST_MAIN_PROCESS_PORT="${DIST_MAIN_PROCESS_PORT:-29500}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PYTHON_BIN}")/accelerate}"

if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  if command -v accelerate >/dev/null 2>&1; then
    ACCELERATE_BIN="$(command -v accelerate)"
  fi
fi

echo "============================================================"
echo "  ProciGen New Pipeline Training"
echo "  Raw root      : ${RAW_ROOT}"
echo "  Prepared root : ${PREPARED_ROOT}"
echo "  Split         : ${SPLIT_FILE}:${SPLIT_KEY}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "  Opt file      : ${OPT_FILE}"
echo "  Prepare workers: ${PREPARE_NUM_WORKERS}"
echo "============================================================"

resolve_bool_flag() {
  local value="$1"
  local positive_flag="$2"
  local negative_flag="$3"

  case "${value}" in
    1|true|TRUE|yes|YES) printf '%s' "${positive_flag}" ;;
    0|false|FALSE|no|NO) printf '%s' "${negative_flag}" ;;
    *)
      echo "[train.sh] Invalid boolean value \`${value}\` for flags ${positive_flag}/${negative_flag}." >&2
      exit 1
      ;;
  esac
}

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  PREP_ARGS=(
    "${REPO_ROOT}/scripts/preprocess_procigen_gt.py"
    --raw_root "${RAW_ROOT}"
    --output_root "${PREPARED_ROOT}"
    --status_dir "${STATUS_DIR}"
    --split_file "${SPLIT_FILE}"
    --split_key "${SPLIT_KEY}"
    --camera_id "${CAMERA_ID}"
    --processed_subdir "${PROCESSED_SUBDIR}"
    --gs_subdir "${GS_SUBDIR}"
    --num_workers "${PREPARE_NUM_WORKERS}"
    --heartbeat_interval "${PREPARE_HEARTBEAT_INTERVAL}"
  )
  if [[ "${PREPARE_MAX_SEQUENCES}" != "0" ]]; then
    PREP_ARGS+=(--max_sequences "${PREPARE_MAX_SEQUENCES}")
  fi
  if [[ "${PREPARE_MAX_FRAMES}" != "0" ]]; then
    PREP_ARGS+=(--max_frames "${PREPARE_MAX_FRAMES}")
  fi
  if [[ "${OVERWRITE_PREPARE}" == "1" ]]; then
    PREP_ARGS+=(--overwrite)
  fi

  mkdir -p "${STATUS_DIR}"

  if [[ "${DETACH_PREPARE}" == "1" ]]; then
    if [[ "${PREPARE_ONLY}" != "1" ]]; then
      echo "[train.sh] DETACH_PREPARE=1 only supports PREPARE_ONLY=1." >&2
      exit 1
    fi
    STDOUT_LOG="${STATUS_DIR}/stdout_$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="${STATUS_DIR}/preprocess.pid"
    echo "[train.sh] Launching detached ProciGen preprocessing..."
    if command -v setsid >/dev/null 2>&1; then
      setsid env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${PREP_ARGS[@]}" >"${STDOUT_LOG}" 2>&1 < /dev/null &
    else
      nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${PREP_ARGS[@]}" >"${STDOUT_LOG}" 2>&1 < /dev/null &
    fi
    PREP_PID=$!
    echo "${PREP_PID}" > "${PID_FILE}"
    printf '%s\n' "${STDOUT_LOG}" > "${STATUS_DIR}/latest_stdout.log"
    printf '%s\n' "${PREP_PID}" > "${STATUS_DIR}/latest_pid"
    echo "[train.sh] Detached PID=${PREP_PID}"
    echo "[train.sh] Stdout log: ${STDOUT_LOG}"
    echo "[train.sh] Status dir: ${STATUS_DIR}"
    echo "[train.sh] Monitor with:"
    echo "  ${PYTHON_BIN} ${REPO_ROOT}/scripts/monitor_procigen_preprocess.py --status_dir ${STATUS_DIR} --follow"
    exit 0
  fi

  echo "[train.sh] Preparing ProciGen GT assets..."
  "${PYTHON_BIN}" "${PREP_ARGS[@]}"
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "[train.sh] PREPARE_ONLY=1, exiting after preprocessing."
  exit 0
fi

TRAIN_ARGS=(
  "${REPO_ROOT}/train_dual_branch_fm.py"
  --data_root "${PREPARED_ROOT}"
  --processed_subdir "${PROCESSED_SUBDIR}"
  --gs_subdir "${GS_SUBDIR}"
  --split_file "${SPLIT_FILE}"
  --split_key "${SPLIT_KEY}"
  --output_dir "${OUTPUT_DIR}"
  --project_name "${PROJECT_NAME:-${OPT_PROJECT_NAME}}"
  --seed "${SEED}"
  --log_with "${LOG_WITH}"
  --mixed_precision "${MIXED_PRECISION}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_steps "${MAX_STEPS}"
  --save_every "${SAVE_EVERY:-${OPT_SAVE_EVERY}}"
  --log_every "${LOG_EVERY:-${OPT_LOG_EVERY}}"
  --print_every "${PRINT_EVERY:-${OPT_PRINT_EVERY}}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-${OPT_GRAD_ACCUM_STEPS}}"
  --max_grad_norm "${MAX_GRAD_NORM:-${OPT_MAX_GRAD_NORM}}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY:-${OPT_WEIGHT_DECAY}}"
  --warmup_steps "${WARMUP_STEPS:-${OPT_WARMUP_STEPS}}"
  --clip_length "${CLIP_LENGTH:-${OPT_CLIP_LENGTH}}"
  --clip_stride "${CLIP_STRIDE:-${OPT_CLIP_STRIDE}}"
  --max_sequences "${TRAIN_MAX_SEQUENCES:-${OPT_TRAIN_MAX_SEQUENCES}}"
  --dataset_cache_sequences "${DATASET_CACHE_SEQUENCES:-${OPT_DATASET_CACHE_SEQUENCES}}"
  --index_progress_every "${INDEX_PROGRESS_EVERY:-${OPT_INDEX_PROGRESS_EVERY}}"
  --background_value "${BACKGROUND_VALUE:-${OPT_BACKGROUND_VALUE}}"
  --prefetch_factor "${PREFETCH_FACTOR:-${OPT_PREFETCH_FACTOR}}"
  --image_height "${IMAGE_HEIGHT:-${OPT_IMAGE_HEIGHT}}"
  --image_width "${IMAGE_WIDTH:-${OPT_IMAGE_WIDTH}}"
  --patch_size "${PATCH_SIZE:-${OPT_PATCH_SIZE}}"
  --hidden_dim "${HIDDEN_DIM:-${OPT_HIDDEN_DIM}}"
  --depth "${DEPTH:-${OPT_DEPTH}}"
  --num_heads "${NUM_HEADS:-${OPT_NUM_HEADS}}"
  --mlp_ratio "${MLP_RATIO:-${OPT_MLP_RATIO}}"
  --dropout "${DROPOUT:-${OPT_DROPOUT}}"
  --video_channels "${VIDEO_CHANNELS:-${OPT_VIDEO_CHANNELS}}"
  --human_gaussian_source "${HUMAN_GAUSSIAN_SOURCE:-${OPT_HUMAN_GAUSSIAN_SOURCE}}"
  --num_human_gaussians "${NUM_HUMAN_GAUSSIANS:-${OPT_NUM_HUMAN_GAUSSIANS}}"
  --num_object_gaussians "${NUM_OBJECT_GAUSSIANS:-${OPT_NUM_OBJECT_GAUSSIANS}}"
  --num_joints "${NUM_JOINTS:-${OPT_NUM_JOINTS}}"
  --contact_dim "${CONTACT_DIM:-${OPT_CONTACT_DIM}}"
  --curriculum_fusion_start_ratio "${CURRICULUM_FUSION_START_RATIO:-${OPT_CURRICULUM_FUSION_START_RATIO}}"
  --curriculum_full_start_ratio "${CURRICULUM_FULL_START_RATIO:-${OPT_CURRICULUM_FULL_START_RATIO}}"
  --video_unfreeze_start_ratio "${VIDEO_UNFREEZE_START_RATIO:-${OPT_VIDEO_UNFREEZE_START_RATIO}}"
  --video_stage2_num_top_blocks "${VIDEO_STAGE2_NUM_TOP_BLOCKS:-${OPT_VIDEO_STAGE2_NUM_TOP_BLOCKS}}"
)

TRAIN_ARGS+=("$(resolve_bool_flag "${CACHE_RGB:-${OPT_CACHE_RGB}}" "--cache_rgb" "--no-cache_rgb")")
TRAIN_ARGS+=("$(resolve_bool_flag "${FREEZE_VIDEO_BACKBONE:-${OPT_FREEZE_VIDEO_BACKBONE}}" "--freeze_video_backbone" "--no-freeze_video_backbone")")

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_ID_CLEAN}"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "[train.sh] Could not find an executable accelerate launcher. Checked: ${ACCELERATE_BIN}" >&2
    exit 1
  fi
  if [[ "${DIST_NUM_PROCESSES}" == "0" ]]; then
    DIST_NUM_PROCESSES="${GPU_COUNT}"
  fi
  echo "[train.sh] Launching dual-branch FM distributed training on GPUs ${GPU_ID_CLEAN}..."
  echo "[train.sh] Processes     : ${DIST_NUM_PROCESSES}"
  echo "[train.sh] Master port   : ${DIST_MAIN_PROCESS_PORT}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID_CLEAN}" \
    "${ACCELERATE_BIN}" launch \
      --multi_gpu \
      --num_machines 1 \
      --num_processes "${DIST_NUM_PROCESSES}" \
      --main_process_port "${DIST_MAIN_PROCESS_PORT}" \
      --mixed_precision "${MIXED_PRECISION}" \
      "${TRAIN_ARGS[@]}"
else
  echo "[train.sh] Launching dual-branch FM training on GPU ${GPU_ID_CLEAN}..."
  if (( BATCH_SIZE > 8 )); then
    echo "[train.sh] Warning: BATCH_SIZE=${BATCH_SIZE} is per-device. On a single GPU this is likely too large."
  fi
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID_CLEAN}" "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
fi
