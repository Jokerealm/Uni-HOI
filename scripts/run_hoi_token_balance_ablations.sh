#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-sample_data/WAI_prepared/sequences}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-sample_data/BEHAVE_heldout_prepared/sequences}"
ABLATION_ROOT="${ABLATION_ROOT:-outputs/ablation_hoi_token_balance}"
LOG_ROOT="${LOG_ROOT:-logs/ablation_hoi_token_balance}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_DATASET_CACHE_SEQUENCES="${EVAL_DATASET_CACHE_SEQUENCES:-2}"
EVAL_NUM_ODE_STEPS="${EVAL_NUM_ODE_STEPS:-12}"
BUILD_H5_CACHE="${BUILD_H5_CACHE:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE

mkdir -p "${ABLATION_ROOT}" "${LOG_ROOT}"

echo "[hoi-token-ablation] diagnostic"
"${PYTHON_BIN}" scripts/analyze_hoi_token_fm_imbalance.py \
  --data_root "${TRAIN_DATA_ROOT}" \
  --token_budgets "850x850,128x128" \
  --max_batches "${DIAGNOSTIC_MAX_BATCHES:-2}" \
  --out "${ABLATION_ROOT}/diagnostic_state_fm_imbalance.json"

if [[ "${BUILD_H5_CACHE}" == "1" ]]; then
  for budget in "850 850" "128 128"; do
    read -r human_count object_count <<< "${budget}"
    echo "[hoi-token-ablation] build train H5 | H=${human_count} O=${object_count}"
    "${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
      --data_root "${TRAIN_DATA_ROOT}" \
      --num_human_gaussians "${human_count}" \
      --num_object_gaussians "${object_count}"
    echo "[hoi-token-ablation] build test H5 | H=${human_count} O=${object_count}"
    "${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
      --data_root "${TEST_DATA_ROOT}" \
      --num_human_gaussians "${human_count}" \
      --num_object_gaussians "${object_count}"
  done
fi

run_one() {
  local name="$1"
  local human_count="$2"
  local object_count="$3"
  local fm_mode="$4"
  local fm_weights="$5"
  local output_dir="${ABLATION_ROOT}/${name}"
  local train_log="${LOG_ROOT}/${name}.train.log"
  local eval_out="${output_dir}/eval_ode${EVAL_NUM_ODE_STEPS}_b${EVAL_BATCH_SIZE}_step${MAX_STEPS}.json"

  echo "[hoi-token-ablation] train ${name} | H=${human_count} O=${object_count} | state_fm=${fm_mode}"
  NOHUP=0 \
  LOG_FILE="${train_log}" \
  DATA_ROOT="${TRAIN_DATA_ROOT}" \
  OUTPUT_DIR="${output_dir}" \
  RUN_NAME="${name}_${MAX_STEPS}" \
  MAX_STEPS="${MAX_STEPS}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY}" \
  NUM_HUMAN_GAUSSIANS="${human_count}" \
  NUM_OBJECT_GAUSSIANS="${object_count}" \
  STATE_FM_LOSS_MODE="${fm_mode}" \
  STATE_FM_GROUP_WEIGHTS="${fm_weights}" \
  ./train.sh

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[hoi-token-ablation] eval ${name}"
    "${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
      --output_dir "${output_dir}" \
      --data_root "${TEST_DATA_ROOT}" \
      --steps "${MAX_STEPS}" \
      --max_batches 0 \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --num_workers "${EVAL_NUM_WORKERS}" \
      --dataset_cache_sequences "${EVAL_DATASET_CACHE_SEQUENCES}" \
      --num_ode_steps "${EVAL_NUM_ODE_STEPS}" \
      --out "${eval_out}"
  fi
}

run_one "a0_uniform_850x850" "850" "850" "uniform" ""
run_one "a2_uniform_128x128" "128" "128" "uniform" ""

echo "[hoi-token-ablation] done | root=${ABLATION_ROOT}"
