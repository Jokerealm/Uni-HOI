#!/bin/bash
set -e
export HYDRA_FULL_ERROR=1

# ============================================================
# Uni-HOI 4.0 — Training Script
#
# Pipeline:
#   Step 1: Preprocess (offline prior extraction)
#   Step 2: Amodal Video Completion (ProPainter)
#   Step 3: 3D Lifting & Metric Alignment (zero-shot, no training)
#   Step 4: Joint 3DGS Optimization (per-video, ONLY gradient step)
#   Step 5: End-to-End Evaluation
# ============================================================

# ======== Quick smoke test on sample_data ========
# Step 4 only (assumes Steps 1-3 outputs exist):
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     run.job=train dataset=sample \
#     step5.num_epochs=2 step5.num_iters_per_epoch=500

# ======== Full pipeline on sample_data ========
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     run.job=full dataset=sample \
#     step5.num_epochs=2 step5.num_iters_per_epoch=500

# ======== Per-video optimization on BEHAVE (default) ========
CUDA_VISIBLE_DEVICES=0 python main.py \
    run.job=train \
    dataset=behave \
    data_prep.video_name=Date03_Sub03_chairwood_hand \
    step5.num_epochs=5 \
    step5.num_iters_per_epoch=1000 \
    step4.num_iters=5000

# ======== Full pipeline on BEHAVE ========
# CUDA_VISIBLE_DEVICES=1 python main.py \
#     run.job=full \
#     dataset=behave \
#     data_prep.video_name=Date03_Sub03_chairwood_hand \
#     step5.num_epochs=5 \
#     step5.num_iters_per_epoch=1000

# ======== Evaluation only ========
# CUDA_VISIBLE_DEVICES=1 python main.py \
#     run.job=eval dataset=behave \
#     data_prep.video_name=Date03_Sub03_chairwood_hand
