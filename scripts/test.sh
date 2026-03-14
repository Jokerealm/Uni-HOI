#!/bin/bash
# test.sh — Launch evaluation & metrics (Step 5)
# Usage:
#   bash scripts/test.sh                                    # sample_data, latest checkpoint
#   bash scripts/test.sh --gpu 1 --dataset behave           # 完整数据集
#   bash scripts/test.sh --checkpoint outputs/runs/xxx/checkpoint_latest.pt
set -e

GPU_ID=0
DATASET="sample"
VIDEO_NAME="test_video"
CHECKPOINT=""
RUN_ID="latest"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)         GPU_ID="$2";      shift 2 ;;
        --dataset)     DATASET="$2";     shift 2 ;;
        --video)       VIDEO_NAME="$2";  shift 2 ;;
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --run-id)      RUN_ID="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,6p' "$0"; exit 0 ;;
        *)  EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

CMD="python test.py dataset=${DATASET} data_prep.video_name=${VIDEO_NAME}"
if [ -n "$CHECKPOINT" ]; then
    CMD="${CMD} checkpoint.path=${CHECKPOINT}"
else
    CMD="${CMD} checkpoint.run_id=${RUN_ID}"
fi
CMD="${CMD} ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}
