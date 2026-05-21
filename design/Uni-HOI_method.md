# Uni-HOI Current Method

Current experimental code follows a single-image RGB-to-HOI design:

- RGB stream consumes one RGB image. Video clips are not part of the model input.
- HOI stream predicts explicit state tokens in a human-relative coordinate frame: SMPL shape, SMPL pose, human-relative translation, object-relative SE(3), contact, 3D joints, human Gaussian tokens, and object Gaussian tokens.
- Fusion is one-way by default: RGB DiT hidden tokens and first-frame image tokens update the HOI stream. The Wan stream is frozen, so RGB priors guide HOI without letting HOI supervision distort the video prior.
- Training uses the preserved `DualBranchHOIDataset` so existing GT/preprocessing assets remain compatible.

Active files:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract_hoi.sh`
