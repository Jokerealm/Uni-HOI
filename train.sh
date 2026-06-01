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
export WANDB_MODE="${WANDB_MODE:-offline}"

IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#CUDA_DEVICE_IDS[@]}}"

MODEL_VARIANT="${MODEL_VARIANT:-cointeract}"
DATA_ROOT="${DATA_ROOT:-sample_data/WAI_prepared/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cointeract_uniform_fm}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
SPLIT_FILE="${SPLIT_FILE:-}"
SPLIT_KEY="${SPLIT_KEY:-train}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

MAX_STEPS="${MAX_STEPS:-100000}"
ENABLE_HOI_TOKEN_MOE="${ENABLE_HOI_TOKEN_MOE:-0}"
HOI_TOKEN_MOE_EXPERT_DIM="${HOI_TOKEN_MOE_EXPERT_DIM:-256}"
HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM="${HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM:-0}"
HOI_TOKEN_MOE_RESIDUAL_SCALE="${HOI_TOKEN_MOE_RESIDUAL_SCALE:-1.0}"
LAMBDA_HOI_TOKEN_ROUTER="${LAMBDA_HOI_TOKEN_ROUTER:-1.0}"
STATE_FM_LOSS_MODE="${STATE_FM_LOSS_MODE:-uniform}"
STATE_FM_GROUP_WEIGHTS="${STATE_FM_GROUP_WEIGHTS:-}"
RUN_NAME="${RUN_NAME:-cointeract_uniform_fm_${MAX_STEPS}}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET_CACHE_SEQUENCES="${DATASET_CACHE_SEQUENCES:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"
TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY:-500}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

CLIP_LENGTH="${CLIP_LENGTH:-1}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
COORDINATE_MODE="${COORDINATE_MODE:-relative}"
HUMAN_GAUSSIAN_SOURCE="${HUMAN_GAUSSIAN_SOURCE:-smpl_mesh}"
NUM_HUMAN_GAUSSIANS="${NUM_HUMAN_GAUSSIANS:-850}"
NUM_OBJECT_GAUSSIANS="${NUM_OBJECT_GAUSSIANS:-850}"
CONTACT_JOINT_INDICES="${CONTACT_JOINT_INDICES:-20,21}"

LAMBDA_STATE_FM="${LAMBDA_STATE_FM:-1.0}"
LAMBDA_RGB_FM="${LAMBDA_RGB_FM:-0.0}"
LAMBDA_SHAPE="${LAMBDA_SHAPE:-0.1}"
LAMBDA_POSE="${LAMBDA_POSE:-0.5}"
LAMBDA_TRANSLATION="${LAMBDA_TRANSLATION:-0.5}"
LAMBDA_OBJECT_POSE="${LAMBDA_OBJECT_POSE:-0.5}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-0.1}"
LAMBDA_JOINTS="${LAMBDA_JOINTS:-1.0}"
LAMBDA_HUMAN_GAUSSIAN="${LAMBDA_HUMAN_GAUSSIAN:-1.0}"
LAMBDA_OBJECT_GAUSSIAN="${LAMBDA_OBJECT_GAUSSIAN:-1.0}"
LAMBDA_GAUSSIAN_CHAMFER="${LAMBDA_GAUSSIAN_CHAMFER:-1.0}"
LAMBDA_GAUSSIAN_XYZ_L1="${LAMBDA_GAUSSIAN_XYZ_L1:-0.05}"
LAMBDA_GAUSSIAN_ATTR_L1="${LAMBDA_GAUSSIAN_ATTR_L1:-0.1}"
LAMBDA_PHYS_CONTACT="${LAMBDA_PHYS_CONTACT:-0.01}"
LAMBDA_PHYS_PENETRATION="${LAMBDA_PHYS_PENETRATION:-0.01}"
PHYS_LOSS_MAX_FRAMES="${PHYS_LOSS_MAX_FRAMES:-1}"
PHYS_LOSS_MAX_OBJECT_POINTS="${PHYS_LOSS_MAX_OBJECT_POINTS:-512}"

MAIN_ARGS=(
  --model_variant "${MODEL_VARIANT}"
  --data_root "${DATA_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --wan_model_id "${WAN_MODEL_ID}"
  --split_file "${SPLIT_FILE}"
  --split_key "${SPLIT_KEY}"
  --init_checkpoint "${INIT_CHECKPOINT}"
  --resume_checkpoint "${RESUME_CHECKPOINT}"
  --max_sequences "${MAX_SEQUENCES}"
  --max_steps "${MAX_STEPS}"
  --hoi_token_moe_expert_dim "${HOI_TOKEN_MOE_EXPERT_DIM}"
  --hoi_token_moe_router_hidden_dim "${HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM}"
  --hoi_token_moe_residual_scale "${HOI_TOKEN_MOE_RESIDUAL_SCALE}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --dataset_cache_sequences "${DATASET_CACHE_SEQUENCES}"
  --log_every "${LOG_EVERY}"
  --log_with "${LOG_WITH}"
  --project_name "${WANDB_PROJECT}"
  --run_name "${RUN_NAME}"
  --save_every "${SAVE_EVERY}"
  --train_visual_every "${TRAIN_VISUAL_EVERY}"
  --clip_length "${CLIP_LENGTH}"
  --clip_stride "${CLIP_STRIDE}"
  --coordinate_mode "${COORDINATE_MODE}"
  --human_gaussian_source "${HUMAN_GAUSSIAN_SOURCE}"
  --contact_joint_indices "${CONTACT_JOINT_INDICES}"
  --num_human_gaussians "${NUM_HUMAN_GAUSSIANS}"
  --num_object_gaussians "${NUM_OBJECT_GAUSSIANS}"
  --lambda_state_fm "${LAMBDA_STATE_FM}"
  --lambda_rgb_fm "${LAMBDA_RGB_FM}"
  --lambda_shape "${LAMBDA_SHAPE}"
  --lambda_pose "${LAMBDA_POSE}"
  --lambda_translation "${LAMBDA_TRANSLATION}"
  --lambda_object_pose "${LAMBDA_OBJECT_POSE}"
  --lambda_contact "${LAMBDA_CONTACT}"
  --lambda_joints "${LAMBDA_JOINTS}"
  --lambda_human_gaussian "${LAMBDA_HUMAN_GAUSSIAN}"
  --lambda_object_gaussian "${LAMBDA_OBJECT_GAUSSIAN}"
  --lambda_gaussian_chamfer "${LAMBDA_GAUSSIAN_CHAMFER}"
  --lambda_gaussian_xyz_l1 "${LAMBDA_GAUSSIAN_XYZ_L1}"
  --lambda_gaussian_attr_l1 "${LAMBDA_GAUSSIAN_ATTR_L1}"
  --lambda_phys_contact "${LAMBDA_PHYS_CONTACT}"
  --lambda_phys_penetration "${LAMBDA_PHYS_PENETRATION}"
  --lambda_hoi_token_router "${LAMBDA_HOI_TOKEN_ROUTER}"
  --state_fm_loss_mode "${STATE_FM_LOSS_MODE}"
  --state_fm_group_weights "${STATE_FM_GROUP_WEIGHTS}"
  --phys_loss_max_frames "${PHYS_LOSS_MAX_FRAMES}"
  --phys_loss_max_object_points "${PHYS_LOSS_MAX_OBJECT_POINTS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_steps "${WARMUP_STEPS}"
  --lr_scheduler "${LR_SCHEDULER}"
  --min_lr_ratio "${MIN_LR_RATIO}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --mixed_precision "${MIXED_PRECISION}"
  --no-pin_memory
  "$@"
)

case "${ENABLE_HOI_TOKEN_MOE}" in
  1|true|TRUE|yes|YES|on|ON)
    MAIN_ARGS+=(--enable_hoi_token_moe)
    ;;
  *)
    MAIN_ARGS+=(--no-enable_hoi_token_moe)
    ;;
esac

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  RUN_CMD=(
    "${ACCELERATE_BIN}" launch
    --multi_gpu
    --num_processes "${NUM_PROCESSES}"
    --num_machines 1
    --mixed_precision "${MIXED_PRECISION}"
    --dynamo_backend no
    train_cointeract_hoi.py
    "${MAIN_ARGS[@]}"
  )
else
  RUN_CMD=("${PYTHON_BIN}" train_cointeract_hoi.py "${MAIN_ARGS[@]}")
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
  echo "WANDB_MODE=${WANDB_MODE}"
  echo "MODEL_VARIANT=${MODEL_VARIANT}"
  echo "SPLIT_FILE=${SPLIT_FILE}"
  echo "SPLIT_KEY=${SPLIT_KEY}"
  echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
  echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "ENABLE_HOI_TOKEN_MOE=${ENABLE_HOI_TOKEN_MOE}"
  echo "DATASET_CACHE_SEQUENCES=${DATASET_CACHE_SEQUENCES}"
  echo "INTERACTION=full_attention"
  echo "HOI_TOKEN_MOE_EXPERT_DIM=${HOI_TOKEN_MOE_EXPERT_DIM}"
  echo "HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM=${HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM}"
  echo "HOI_TOKEN_MOE_RESIDUAL_SCALE=${HOI_TOKEN_MOE_RESIDUAL_SCALE}"
  echo "LAMBDA_HOI_TOKEN_ROUTER=${LAMBDA_HOI_TOKEN_ROUTER}"
  echo "STATE_FM_LOSS_MODE=${STATE_FM_LOSS_MODE}"
  echo "STATE_FM_GROUP_WEIGHTS=${STATE_FM_GROUP_WEIGHTS}"
  echo "LAMBDA_STATE_FM=${LAMBDA_STATE_FM}"
  echo "LAMBDA_POSE=${LAMBDA_POSE}"
  echo "LAMBDA_OBJECT_POSE=${LAMBDA_OBJECT_POSE}"
  echo "LAMBDA_CONTACT=${LAMBDA_CONTACT}"
  echo "LAMBDA_HUMAN_GAUSSIAN=${LAMBDA_HUMAN_GAUSSIAN}"
  echo "LAMBDA_OBJECT_GAUSSIAN=${LAMBDA_OBJECT_GAUSSIAN}"
  echo "SAVE_EVERY=${SAVE_EVERY}"
  echo "TRAIN_VISUAL_EVERY=${TRAIN_VISUAL_EVERY}"
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
  echo "started cointeract training with nohup | pid=${PID} | log=${LOG_FILE}"
else
  "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
