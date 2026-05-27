# Uni-HOI Current Method

Current experimental code follows an HOI-primary RGB-guided dual-stream DiT design:

- RGB stream consumes RGB frame/video latents with the frozen Wan visual prior. For video guidance, use Wan-compatible clip lengths such as 5, 9, 13, or 17 frames.
- HOI stream predicts explicit state tokens in a human-relative coordinate frame: SMPL shape, SMPL pose, human-relative translation, object-relative SE(3), contact, 3D joints, human Gaussian tokens, and object Gaussian tokens.
- Fusion is HOI-primary with shared CoInteract-style DiT blocks: HOI and RGB tokens share the attention/MLP backbone, while patch/projection layers and stream-specific AdaLN scale/shift modulation distinguish modalities. Stage 1 uses shared full RGB/HOI attention; Stage 2 and inference use asymmetric RGB->HOI co-attention. RGB DiT hidden tokens act only as an auxiliary visual prior, the retained prediction head and supervised losses stay HOI-side, and inference may drop the RGB/Wan branch to run HOI-only.
- The Wan stream is frozen and RGB context is detached by default, so RGB priors guide HOI without letting HOI supervision distort the video prior.
- Training uses the preserved `DualBranchHOIDataset` so existing GT/preprocessing assets remain compatible.

Active files:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract_hoi.sh`
