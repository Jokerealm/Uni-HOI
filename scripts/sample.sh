# ======== Quick smoke test on sample_data ========
# Step 4 only (assumes Steps 1-3 outputs exist):
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     run.job=train dataset=sample \
#     step4.num_iters=500

# ======== Full pipeline on sample_data ========
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     run.job=full dataset=sample \
#     step4.num_iters=500
