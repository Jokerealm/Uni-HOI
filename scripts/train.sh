#!/bin/bash
# train.sh — Launch end-to-end training (Step 5)
# Usage:
#   bash scripts/train.sh                          # sample_data 快速验证
#   bash scripts/train.sh --gpu 1 --dataset behave # 完整数据集
#   bash scripts/train.sh --gpu 0 --video Date03_Sub03_chairwood --epochs 5
set -e

GPU_ID=0
DATASET="sample"
VIDEO_NAME="test_video"
EPOCHS=""
ITERS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)      GPU_ID="$2";      shift 2 ;;
        --dataset)  DATASET="$2";     shift 2 ;;
        --video)    VIDEO_NAME="$2";  shift 2 ;;
        --epochs)   EPOCHS="$2";      shift 2 ;;
        --iters)    ITERS="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,6p' "$0"; exit 0 ;;
        *)  EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

CMD="python train.py dataset=${DATASET} data_prep.video_name=${VIDEO_NAME}"
[ -n "$EPOCHS" ] && CMD="${CMD} step5.num_epochs=${EPOCHS}"
[ -n "$ITERS" ]  && CMD="${CMD} step5.num_iters_per_epoch=${ITERS}"
CMD="${CMD} ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}
