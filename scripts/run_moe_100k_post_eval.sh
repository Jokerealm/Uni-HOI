#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data3/guanz/miniforge3/envs/cari4d/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cointeract_moe}"
DATA_ROOT="${DATA_ROOT:-sample_data/BEHAVE_heldout_prepared/sequences}"
STAGE1_FULL_JSON="${STAGE1_FULL_JSON:-outputs/cointeract_shared_stage1_full_100k/eval_stage1_ckpt100k_A_full_ode12_b4_all.json}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"

TARGET_CKPT="${OUTPUT_DIR}/checkpoints/checkpoint_0100000.pt"
SCAN_OUT="${OUTPUT_DIR}/eval_wai_test_scan5_ode12_b4_moe_100k_all.json"
FULL_OUT_PREFIX="${OUTPUT_DIR}/eval_wai_test_full_ode12_b4_moe_best"
SUMMARY_OUT="${OUTPUT_DIR}/eval_moe_100k_vs_stage1_summary.json"
export OUTPUT_DIR SCAN_OUT STAGE1_FULL_JSON SUMMARY_OUT

echo "[post-eval] waiting for ${TARGET_CKPT}"
start_ts="$(date +%s)"
while [[ ! -s "${TARGET_CKPT}" ]]; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > WAIT_TIMEOUT_SECONDS )); then
    echo "[post-eval] timed out waiting for ${TARGET_CKPT}" >&2
    exit 1
  fi
  if ! pgrep -f "train_cointeract_hoi.py.*--output_dir ${OUTPUT_DIR}.*--max_steps 100000" >/dev/null; then
    echo "[post-eval] training process is not running and target checkpoint is absent" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

echo "[post-eval] target checkpoint found"
STEPS="$(OUTPUT_DIR="${OUTPUT_DIR}" "${PYTHON_BIN}" -c 'import glob, os, re
output_dir = os.environ["OUTPUT_DIR"]
paths = sorted(glob.glob(f"{output_dir}/checkpoints/checkpoint_*.pt"))
steps = []
for path in paths:
    m = re.search(r"checkpoint_(\d+)\.pt$", path)
    if m:
        steps.append(int(m.group(1)))
print(",".join(str(step) for step in sorted(set(steps))))')"

echo "[post-eval] scan steps: ${STEPS}"
env CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --steps "${STEPS}" \
  --max_batches 5 \
  --batch_size 4 \
  --num_workers 2 \
  --dataset_cache_sequences 2 \
  --num_ode_steps 12 \
  --out "${SCAN_OUT}"

BEST_STEP="$(SCAN_OUT="${SCAN_OUT}" "${PYTHON_BIN}" -c 'import json, os
d = json.load(open(os.environ["SCAN_OUT"]))
print(int(d["summary"]["best_step"]))')"
FULL_OUT="${FULL_OUT_PREFIX}_step${BEST_STEP}.json"
export FULL_OUT

echo "[post-eval] best scan step: ${BEST_STEP}"
env CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" scripts/eval_dual_stream_hoi_rgb_checkpoints.py \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --steps "${BEST_STEP}" \
  --max_batches 0 \
  --batch_size 4 \
  --num_workers 2 \
  --dataset_cache_sequences 2 \
  --num_ode_steps 12 \
  --out "${FULL_OUT}"

"${PYTHON_BIN}" -c 'import json
import os
from pathlib import Path

scan_path = Path(os.environ["SCAN_OUT"])
scan = json.load(scan_path.open())
best_step = int(scan["summary"]["best_step"])
full_path = Path(os.environ["FULL_OUT"])
moe_full = json.load(full_path.open())
stage1_path = Path(os.environ["STAGE1_FULL_JSON"])
stage1 = json.load(stage1_path.open())

def best_metrics(payload):
    best = payload["summary"]["best"]
    return {
        "step": int(best["step"]),
        "num_samples": int(best["num_samples"]),
        "metrics": best["metrics"],
    }

moe = best_metrics(moe_full)
base = best_metrics(stage1)
delta = {
    key: float(moe["metrics"][key] - base["metrics"][key])
    for key in ("CD-mean", "CD-h", "CD-o", "CD-c", "supervised")
    if key in moe["metrics"] and key in base["metrics"]
}
payload = {
    "moe_scan": {
        "path": str(scan_path),
        "best_step": best_step,
        "best_cd_mean": float(scan["summary"]["best_cd_mean"]),
    },
    "moe_full": {
        "path": str(full_path),
        **moe,
    },
    "stage1_full_100k": {
        "path": str(stage1_path),
        **base,
    },
    "delta_moe_minus_stage1": delta,
}
out = Path(os.environ["SUMMARY_OUT"])
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
print(f"[post-eval] wrote {out}")'
