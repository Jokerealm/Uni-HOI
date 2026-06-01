#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
SPLIT_FILE="${SPLIT_FILE:-/data4/guanz/data/train-procigen-test-behave.pkl}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-preprocessed/ProciGen_preprocessed_fixed}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-preprocessed/behave_test_dual_branch}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cointeract_hoi_wan_procigen_train_behave_test_once}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-logs/procigen_train_test_once.train.log}"

MAX_STEPS="${MAX_STEPS:-15000}"
RUN_NAME="${RUN_NAME:-cointeract_procigen_train_behave_test_full_${MAX_STEPS}}"

NUM_HUMAN_GAUSSIANS="${NUM_HUMAN_GAUSSIANS:-850}"
NUM_OBJECT_GAUSSIANS="${NUM_OBJECT_GAUSSIANS:-850}"
NUM_JOINTS="${NUM_JOINTS:-22}"
CONTACT_DIM="${CONTACT_DIM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_DATASET_CACHE_SEQUENCES="${EVAL_DATASET_CACHE_SEQUENCES:-2}"
EVAL_NUM_ODE_STEPS="${EVAL_NUM_ODE_STEPS:-12}"
EVAL_OUT="${EVAL_OUT:-${OUTPUT_DIR}/eval_behave_test_full_ode${EVAL_NUM_ODE_STEPS}_b${EVAL_BATCH_SIZE}_step${MAX_STEPS}.json}"

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[procigen-once] build train H5 cache | root=${TRAIN_DATA_ROOT} | split=${SPLIT_FILE}:train"
"${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
  --data_root "${TRAIN_DATA_ROOT}" \
  --split_file "${SPLIT_FILE}" \
  --split_key train \
  --num_human_gaussians "${NUM_HUMAN_GAUSSIANS}" \
  --num_object_gaussians "${NUM_OBJECT_GAUSSIANS}" \
  --num_joints "${NUM_JOINTS}" \
  --contact_dim "${CONTACT_DIM}"

echo "[procigen-once] build test H5 cache | root=${TEST_DATA_ROOT} | split=${SPLIT_FILE}:test"
"${PYTHON_BIN}" scripts/build_dual_branch_h5_cache.py \
  --data_root "${TEST_DATA_ROOT}" \
  --split_file "${SPLIT_FILE}" \
  --split_key test \
  --num_human_gaussians "${NUM_HUMAN_GAUSSIANS}" \
  --num_object_gaussians "${NUM_OBJECT_GAUSSIANS}" \
  --num_joints "${NUM_JOINTS}" \
  --contact_dim "${CONTACT_DIM}"

echo "[procigen-once] train | output=${OUTPUT_DIR} | max_steps=${MAX_STEPS}"
NOHUP=0 \
LOG_FILE="${TRAIN_LOG_FILE}" \
DATA_ROOT="${TRAIN_DATA_ROOT}" \
SPLIT_FILE="${SPLIT_FILE}" \
SPLIT_KEY=train \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_NAME="${RUN_NAME}" \
MAX_STEPS="${MAX_STEPS}" \
NUM_HUMAN_GAUSSIANS="${NUM_HUMAN_GAUSSIANS}" \
NUM_OBJECT_GAUSSIANS="${NUM_OBJECT_GAUSSIANS}" \
./train.sh

echo "[procigen-once] eval | output=${OUTPUT_DIR} | test_root=${TEST_DATA_ROOT} | out=${EVAL_OUT}"
"${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${TEST_DATA_ROOT}" \
  --split_file "${SPLIT_FILE}" \
  --split_key test \
  --steps "${MAX_STEPS}" \
  --max_batches 0 \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${EVAL_NUM_WORKERS}" \
  --dataset_cache_sequences "${EVAL_DATASET_CACHE_SEQUENCES}" \
  --num_ode_steps "${EVAL_NUM_ODE_STEPS}" \
  --out "${EVAL_OUT}"

echo "[procigen-once] done | eval_out=${EVAL_OUT}"
