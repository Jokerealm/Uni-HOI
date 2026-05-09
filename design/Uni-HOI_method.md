# Uni-HOI Current Method

Current experimental code follows a CoInteract-style dual-stream design for 4D HOI:

- RGB stream uses Wan2.2-TI2V-5B with image-only conditioning. The only external condition is the first RGB frame; text is encoded as an empty prompt.
- HOI stream predicts explicit 4D state tokens: SMPL shape, SMPL pose, human translation, object SE(3), contact, 3D joints, human Gaussian tokens, and object Gaussian tokens.
- Fusion is one-way by default: RGB DiT hidden tokens and first-frame image tokens update the HOI stream. The Wan stream is frozen, so RGB priors guide HOI without letting HOI supervision distort the video prior.
- Training uses the preserved `DualBranchHOIDataset` so existing GT/preprocessing assets remain compatible.

Active files:

- `model/cointeract_hoi_wan.py`
- `train_cointeract_hoi.py`
- `scripts/train_cointeract_hoi.sh`
