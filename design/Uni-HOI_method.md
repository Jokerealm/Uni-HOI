# Uni-HOI Current Method

Current experimental code follows an HOI-primary RGB-guided dual-stream DiT design:

- RGB stream consumes RGB frame/video latents with the frozen Wan visual prior. For video guidance, use Wan-compatible clip lengths such as 5, 9, 13, or 17 frames.
- HOI stream predicts explicit state tokens in a human-relative coordinate frame: SMPL shape, SMPL pose, human-relative translation, object-relative SE(3), contact, 3D joints, human Gaussian tokens, and object Gaussian tokens.
- Fusion is asymmetric by default with HOI as the main stream: RGB DiT hidden tokens form an auxiliary stream inside each HOI block, then guide HOI tokens through zero-init cross-stream adapters. `hoi_to_rgb_scale` can be enabled for symmetric ablations, but the retained prediction head and supervised losses remain HOI-side.
- The Wan stream is frozen and RGB context is detached by default, so RGB priors guide HOI without letting HOI supervision distort the video prior.
- Training uses the preserved `DualBranchHOIDataset` so existing GT/preprocessing assets remain compatible.

Active files:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract_hoi.sh`
