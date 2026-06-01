#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/accelerate}"
CONDA_ENV_BIN="$(dirname "${PYTHON_BIN}")"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#CUDA_DEVICE_IDS[@]}}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/output.log}"

MODEL_VARIANT="${MODEL_VARIANT:-cointeract}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/sample_data/behave_1pct/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${MODEL_VARIANT}_hoi_wan_ti2v}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

TRAIN_ARGS=(
  --model_variant "${MODEL_VARIANT}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --wan_model_id "${WAN_MODEL_ID}" \
  --hoi_token_moe_expert_dim "${HOI_TOKEN_MOE_EXPERT_DIM:-256}" \
  --hoi_token_moe_router_hidden_dim "${HOI_TOKEN_MOE_ROUTER_HIDDEN_DIM:-0}" \
  --hoi_token_moe_residual_scale "${HOI_TOKEN_MOE_RESIDUAL_SCALE:-1.0}" \
  --cross_3d2d_scale "${CROSS_3D2D_SCALE:-1.0}" \
  --visual_prior_num_global_tokens "${VISUAL_PRIOR_NUM_GLOBAL_TOKENS:-8}" \
  --visual_resampler_depth "${VISUAL_RESAMPLER_DEPTH:-2}" \
  --cross_3d2d_depth "${CROSS_3D2D_DEPTH:-6}" \
  --clip_length "${CLIP_LENGTH:-1}" \
  --clip_stride "${CLIP_STRIDE:-1}" \
  --coordinate_mode "${COORDINATE_MODE:-relative}" \
  --max_steps "${MAX_STEPS:-35000}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-2e-5}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --lr_scheduler "${LR_SCHEDULER:-constant}" \
  --min_lr_ratio "${MIN_LR_RATIO:-0.1}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --log_with "${LOG_WITH:-wandb}" \
  --project_name "${WANDB_PROJECT:-uni-hoi-4d}" \
  --run_name "${RUN_NAME:-hoi}" \
  --train_visual_every "${TRAIN_VISUAL_EVERY:-500}" \
  --num_human_gaussians "${NUM_HUMAN_GAUSSIANS:-850}" \
  --num_object_gaussians "${NUM_OBJECT_GAUSSIANS:-850}" \
  --lambda_gaussian_chamfer "${LAMBDA_GAUSSIAN_CHAMFER:-1.0}" \
  --lambda_gaussian_xyz_l1 "${LAMBDA_GAUSSIAN_XYZ_L1:-0.05}" \
  --lambda_gaussian_attr_l1 "${LAMBDA_GAUSSIAN_ATTR_L1:-0.1}" \
  --lambda_phys_contact "${LAMBDA_PHYS_CONTACT:-0.01}" \
  --lambda_phys_penetration "${LAMBDA_PHYS_PENETRATION:-0.01}" \
  --lambda_hoi_token_router "${LAMBDA_HOI_TOKEN_ROUTER:-1.0}" \
  --state_fm_loss_mode "${STATE_FM_LOSS_MODE:-uniform}" \
  --state_fm_group_weights "${STATE_FM_GROUP_WEIGHTS:-}" \
  --phys_loss_max_frames "${PHYS_LOSS_MAX_FRAMES:-1}" \
  --phys_loss_max_object_points "${PHYS_LOSS_MAX_OBJECT_POINTS:-512}" \
  --num_workers "${NUM_WORKERS:-8}" \
  "$@"
)

if [[ "${NUM_PROCESSES:-1}" -gt 1 ]]; then
  TRAIN_CMD=(
    "${ACCELERATE_BIN}" launch
    --num_processes "${NUM_PROCESSES}"
    --mixed_precision "${MIXED_PRECISION}"
    "${REPO_ROOT}/train_cointeract_hoi.py"
    "${TRAIN_ARGS[@]}"
  )
else
  TRAIN_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/train_cointeract_hoi.py" "${TRAIN_ARGS[@]}"
  )
fi

mkdir -p "$(dirname "${LOG_PATH}")"
echo "Starting training with nohup. Logs: ${LOG_PATH}"
echo "MODEL_VARIANT=${MODEL_VARIANT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
nohup "${TRAIN_CMD[@]}" > "${LOG_PATH}" 2>&1 &
echo "Training PID: $!"
