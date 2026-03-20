#!/bin/bash
set -e
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================
# Batch Preprocessing: Steps 1-3 for all BEHAVE sequences
#
# Runs offline prior extraction (Step 1), amodal completion
# (Step 2), and Hunyuan3D-2 3D lifting (Step 3) for every
# sequence in the BEHAVE dataset.
#
# These steps are inference-only (no gradient training) and
# their outputs are cached per-sequence, so this only needs
# to run once.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 conda run -n cari4d bash scripts/preprocess_all.sh
# ============================================================

BEHAVE_DIR="/data4/guanz/data/Behave/sequences"
LOG_DIR="logs/preprocess"
mkdir -p "$LOG_DIR"

# Get all sequence names
SEQUENCES=($(ls -1 "$BEHAVE_DIR"))
TOTAL=${#SEQUENCES[@]}

echo "============================================================"
echo "  BEHAVE Batch Preprocessing — Steps 1-3"
echo "  Total sequences: $TOTAL"
echo "  Start time: $(date)"
echo "============================================================"

DONE=0
FAIL=0
SKIP=0

for SEQ in "${SEQUENCES[@]}"; do
    IDX=$((DONE + FAIL + SKIP + 1))

    # Skip if already preprocessed (gs_init exists = Steps 1-3 done)
    if [ -d "$BEHAVE_DIR/$SEQ/gs_init" ] && [ -f "$BEHAVE_DIR/$SEQ/gs_init/gs_init_combined.pt" ]; then
        echo "[$IDX/$TOTAL] SKIP $SEQ (already preprocessed)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[$IDX/$TOTAL] Processing: $SEQ"
    echo "  Time: $(date)"
    echo "============================================================"

    LOG_FILE="$LOG_DIR/${SEQ}.log"

    if python main.py \
        run.job=preprocess \
        dataset=behave \
        data_prep.video_name="$SEQ" \
        2>&1 | tee "$LOG_FILE"; then
        DONE=$((DONE + 1))
        echo "[$IDX/$TOTAL] DONE $SEQ"
    else
        FAIL=$((FAIL + 1))
        echo "[$IDX/$TOTAL] FAIL $SEQ — see $LOG_FILE"
    fi
done

echo ""
echo "============================================================"
echo "  Batch Preprocessing Complete"
echo "  Done: $DONE / $TOTAL"
echo "  Skipped: $SKIP"
echo "  Failed: $FAIL"
echo "  End time: $(date)"
echo "============================================================"
