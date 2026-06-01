#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-sample_data/WAI_prepared/sequences}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-sample_data/BEHAVE_heldout_prepared/sequences}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cointeract_stage1_geometry_lr1e4_70k}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-logs/cointeract_stage1_geometry_lr1e4_baseline.train.log}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/hoi_baselines_current.json}"

BASELINE_STEP="${BASELINE_STEP:-10000}"
LR="${LR:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_DATASET_CACHE_SEQUENCES="${EVAL_DATASET_CACHE_SEQUENCES:-2}"
EVAL_NUM_ODE_STEPS="${EVAL_NUM_ODE_STEPS:-12}"

HIGH_LR_FULL_JSON="${HIGH_LR_FULL_JSON:-${OUTPUT_DIR}/eval_wai_test_full_ode12_b4_current_best10000.json}"
STAGE1_ONLY_FULL_JSON="${STAGE1_ONLY_FULL_JSON:-outputs/cointeract_stage1_geometry_50k/eval_wai_test_full_ode12_b4_best.json}"
MOE_FULL_JSON="${MOE_FULL_JSON:-outputs/cointeract_moe/eval_wai_test_full_ode12_b4_steps27500_70000.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
export NUM_PROCESSES="${NUM_PROCESSES:-${#CUDA_DEVICE_IDS[@]}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

train_baseline_if_needed() {
  local target="${OUTPUT_DIR}/checkpoints/checkpoint_$(printf "%07d" "${BASELINE_STEP}").pt"
  if [[ -s "${target}" ]]; then
    echo "[stage1-high-lr-baseline] checkpoint exists, skip training: ${target}"
    return
  fi

  echo "[stage1-high-lr-baseline] train to baseline step ${BASELINE_STEP} | lr=${LR}"
  NOHUP=0 \
  LOG_WITH=none \
  ENABLE_HOI_TOKEN_MOE=0 \
  LAMBDA_HOI_TOKEN_ROUTER=0.0 \
  STATE_FM_LOSS_MODE=uniform \
  LOG_FILE="${TRAIN_LOG_FILE}" \
  DATA_ROOT="${TRAIN_DATA_ROOT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RUN_NAME="cointeract_stage1_geometry_lr1e4_baseline_${BASELINE_STEP}" \
  MAX_STEPS="${BASELINE_STEP}" \
  LR="${LR}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  TRAIN_VISUAL_EVERY=0 \
  LAMBDA_STATE_FM=1.0 \
  LAMBDA_SHAPE=0.1 \
  LAMBDA_POSE=0.5 \
  LAMBDA_TRANSLATION=0.5 \
  LAMBDA_OBJECT_POSE=0.5 \
  LAMBDA_CONTACT=0.1 \
  LAMBDA_JOINTS=1.0 \
  LAMBDA_HUMAN_GAUSSIAN=1.0 \
  LAMBDA_OBJECT_GAUSSIAN=1.0 \
  LAMBDA_GAUSSIAN_CHAMFER=1.0 \
  LAMBDA_GAUSSIAN_XYZ_L1=0.05 \
  LAMBDA_GAUSSIAN_ATTR_L1=0.1 \
  LAMBDA_PHYS_CONTACT=0.01 \
  LAMBDA_PHYS_PENETRATION=0.01 \
  ./train.sh
}

eval_high_lr_if_needed() {
  if [[ -s "${HIGH_LR_FULL_JSON}" ]]; then
    echo "[stage1-high-lr-baseline] full eval exists, skip: ${HIGH_LR_FULL_JSON}"
    return
  fi

  echo "[stage1-high-lr-baseline] full eval step ${BASELINE_STEP}"
  env CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
    --output_dir "${OUTPUT_DIR}" \
    --data_root "${TEST_DATA_ROOT}" \
    --steps "${BASELINE_STEP}" \
    --max_batches 0 \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${EVAL_NUM_WORKERS}" \
    --dataset_cache_sequences "${EVAL_DATASET_CACHE_SEQUENCES}" \
    --num_ode_steps "${EVAL_NUM_ODE_STEPS}" \
    --out "${HIGH_LR_FULL_JSON}"
}

write_summary() {
  HIGH_LR_FULL_JSON="${HIGH_LR_FULL_JSON}" \
  STAGE1_ONLY_FULL_JSON="${STAGE1_ONLY_FULL_JSON}" \
  MOE_FULL_JSON="${MOE_FULL_JSON}" \
  SUMMARY_OUT="${SUMMARY_OUT}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def best(name, path):
    item = load(path)["summary"]["best"]
    metrics = item["metrics"]
    return {
        "name": name,
        "path": path,
        "step": int(item["step"]),
        "num_samples": int(item["num_samples"]),
        "CD-mean": float(metrics["CD-mean"]),
        "CD-h": float(metrics["CD-h"]),
        "CD-o": float(metrics["CD-o"]),
        "CD-c": float(metrics["CD-c"]),
        "supervised": float(metrics.get("supervised", 0.0)),
    }

rows = [
    best("high_lr_stage1_only_current_best", os.environ["HIGH_LR_FULL_JSON"]),
    best("only_stage1", os.environ["STAGE1_ONLY_FULL_JSON"]),
    best("moe", os.environ["MOE_FULL_JSON"]),
]
by_name = {row["name"]: row for row in rows}
high_lr = by_name["high_lr_stage1_only_current_best"]
comparisons = {
    "high_lr_minus_only_stage1": {
        key: high_lr[key] - by_name["only_stage1"][key]
        for key in ("CD-mean", "CD-h", "CD-o", "CD-c", "supervised")
    },
    "high_lr_minus_moe": {
        key: high_lr[key] - by_name["moe"][key]
        for key in ("CD-mean", "CD-h", "CD-o", "CD-c", "supervised")
    },
}
payload = {
    "protocol": {
        "data_root": "sample_data/BEHAVE_heldout_prepared/sequences",
        "num_samples": 720,
        "batch_size": 4,
        "num_ode_steps": 12,
    },
    "baselines": rows,
    "current_chamfer_baseline": "high_lr_stage1_only_current_best",
    "comparisons": comparisons,
}
out = Path(os.environ["SUMMARY_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print(f"[stage1-high-lr-baseline] wrote {out}")
PY
}

train_baseline_if_needed
eval_high_lr_if_needed
write_summary
