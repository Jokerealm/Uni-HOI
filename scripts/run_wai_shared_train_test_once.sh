#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-sample_data/WAI_prepared/sequences}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-sample_data/BEHAVE_heldout_prepared/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cointeract_shared_hoi_wan_wai_train_test}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-logs/cointeract_shared_wai_train_test.train.log}"

STAGE1_FULL_ATTENTION_STEPS="${STAGE1_FULL_ATTENTION_STEPS:-10000}"
STAGE2_ASYMMETRIC_STEPS="${STAGE2_ASYMMETRIC_STEPS:-5000}"
MAX_STEPS="${MAX_STEPS:-$((STAGE1_FULL_ATTENTION_STEPS + STAGE2_ASYMMETRIC_STEPS))}"
RUN_NAME="${RUN_NAME:-cointeract_shared_wai_s1_${STAGE1_FULL_ATTENTION_STEPS}_s2_${STAGE2_ASYMMETRIC_STEPS}}"

NUM_HUMAN_GAUSSIANS="${NUM_HUMAN_GAUSSIANS:-850}"
NUM_OBJECT_GAUSSIANS="${NUM_OBJECT_GAUSSIANS:-850}"
NUM_JOINTS="${NUM_JOINTS:-22}"
CONTACT_DIM="${CONTACT_DIM:-4}"
SAVE_EVERY="${SAVE_EVERY:-500}"
TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_DATASET_CACHE_SEQUENCES="${EVAL_DATASET_CACHE_SEQUENCES:-2}"
EVAL_NUM_ODE_STEPS="${EVAL_NUM_ODE_STEPS:-12}"
EVAL_TEST_OUT="${EVAL_TEST_OUT:-${OUTPUT_DIR}/eval_wai_test_full_ode${EVAL_NUM_ODE_STEPS}_b${EVAL_BATCH_SIZE}_step${MAX_STEPS}.json}"
BUILD_H5_CACHE="${BUILD_H5_CACHE:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${BUILD_H5_CACHE}" == "1" ]]; then
  echo "[wai-shared] build train H5 cache | root=${TRAIN_DATA_ROOT}"
  "${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
    --data_root "${TRAIN_DATA_ROOT}" \
    --num_human_gaussians "${NUM_HUMAN_GAUSSIANS}" \
    --num_object_gaussians "${NUM_OBJECT_GAUSSIANS}" \
    --num_joints "${NUM_JOINTS}" \
    --contact_dim "${CONTACT_DIM}"

  echo "[wai-shared] build test H5 cache | root=${TEST_DATA_ROOT}"
  "${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
    --data_root "${TEST_DATA_ROOT}" \
    --num_human_gaussians "${NUM_HUMAN_GAUSSIANS}" \
    --num_object_gaussians "${NUM_OBJECT_GAUSSIANS}" \
    --num_joints "${NUM_JOINTS}" \
    --contact_dim "${CONTACT_DIM}"
fi

echo "[wai-shared] train | output=${OUTPUT_DIR} | max_steps=${MAX_STEPS}"
NOHUP=0 \
LOG_FILE="${TRAIN_LOG_FILE}" \
DATA_ROOT="${TRAIN_DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_NAME="${RUN_NAME}" \
STAGE1_FULL_ATTENTION_STEPS="${STAGE1_FULL_ATTENTION_STEPS}" \
STAGE2_ASYMMETRIC_STEPS="${STAGE2_ASYMMETRIC_STEPS}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_EVERY="${SAVE_EVERY}" \
TRAIN_VISUAL_EVERY="${TRAIN_VISUAL_EVERY}" \
NUM_HUMAN_GAUSSIANS="${NUM_HUMAN_GAUSSIANS}" \
NUM_OBJECT_GAUSSIANS="${NUM_OBJECT_GAUSSIANS}" \
./train.sh

echo "[wai-shared] eval test | output=${OUTPUT_DIR} | test_root=${TEST_DATA_ROOT} | out=${EVAL_TEST_OUT}"
"${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${TEST_DATA_ROOT}" \
  --steps "${MAX_STEPS}" \
  --max_batches 0 \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${EVAL_NUM_WORKERS}" \
  --dataset_cache_sequences "${EVAL_DATASET_CACHE_SEQUENCES}" \
  --num_ode_steps "${EVAL_NUM_ODE_STEPS}" \
  --out "${EVAL_TEST_OUT}"

echo "[wai-shared] done | test_eval=${EVAL_TEST_OUT}"
