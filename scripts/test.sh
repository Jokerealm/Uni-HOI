#!/bin/bash

# Example command to run test on BEHAVE dataset with nohup and logging
# Tip: You can speed up inference by appending `run.diffusion_scheduler=ddim run.num_inference_steps=100` to the end of the command,
# same for both stage 1 and stage 2 inference.

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Stage 1: reconstruct H+O and segment, results will be saved to outputs/run.name/single/sample
# Using 1 GPU for inference
echo "Starting stage1 inference at $(date)"
echo "Log file: logs/test_stage1_${TIMESTAMP}.log"
nohup python -m torch.distributed.run --nproc_per_node 1 main.py \
run.name=stage1 model.consistent_center=True \
model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=32 \
model.model_name=pc2-diff-ho-sepsegm model.predict_binary=True model.lw_binary=3.0 \
dataset=behave dataset.max_points=16384 \
scheduler=linear optimizer.lr=3e-4 \
dataset.split_file=/data4/guanz/data/train-procigen-test-behave.pkl run.job=sample \
run.diffusion_scheduler=ddim run.num_inference_steps=100 \
> logs/test_stage1_${TIMESTAMP}.log 2>&1 &

STAGE1_PID=$!
echo "Stage1 inference started with PID: $STAGE1_PID"
echo $STAGE1_PID > logs/test_stage1.pid

# Wait for stage 1 to complete before starting stage 2
# Uncomment the following lines if you want to run stage 2 automatically after stage 1
# echo "Waiting for stage1 to complete..."
# wait $STAGE1_PID
# echo "Stage1 completed at $(date)"

# Uncomment below to run stage 2 inference
# Stage 2: load stage 1 results and refine human and object
# echo "Starting stage2 inference at $(date)"
# echo "Log file: logs/test_stage2_${TIMESTAMP}.log"
# nohup python -m torch.distributed.run --nproc_per_node 1 main.py \
# run.name=stage2 model.consistent_center=True \
# model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=32 \
# model=ho-attn model.attn_weight=1.0 model.attn_type=coord3d+posenc-learnable \
# dataset=behave dataset.type=behave-attn model.point_visible_test=combine \
# dataset.split_file=/data4/guanz/data/train-procigen-test-behave.pkl run.job=sample \
# run.save_name=stage1-500step run.sample_noise_step=500 run.sample_mode=interm-pred \
# dataset.ho_segm_pred_path=$PWD/outputs/stage1/single/sample/pred \
# run.diffusion_scheduler=ddim run.num_inference_steps=100 \
# > logs/test_stage2_${TIMESTAMP}.log 2>&1 &
# 
# STAGE2_PID=$!
# echo "Stage2 inference started with PID: $STAGE2_PID"
# echo $STAGE2_PID > logs/test_stage2.pid

echo ""
echo "Inference started in background. To monitor progress:"
echo "  tail -f logs/test_stage1_${TIMESTAMP}.log"
echo ""
echo "To check if inference is still running:"
echo "  ps -p $STAGE1_PID"
echo ""
echo "To stop inference:"
echo "  kill $STAGE1_PID"
