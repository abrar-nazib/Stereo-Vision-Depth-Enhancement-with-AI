# VKITTI2 1,000-pair head ablation plan — v2

**Question:** Can literature-inspired confidence gating, right-view evidence,
and edge-aware refinement improve *both* frozen outputs beyond the v1 late
residual head, without retraining the stereo or ADE20K predictors?

## Locked protocol

- Data: 1,000 deterministic VKITTI2 pairs (20 evenly spaced frames × 10
  variations × 5 scenes), 800 train from Scene01/02/06/18, 200 held-out
  from Scene20. No image resizing; 256×512 aligned training crop; native
  375×1242 held-out evaluation.
- Source models: exactly the v1 A09m 120k SceneFlow checkpoint and
  `yolo26m-sem-ade20k.pt`; all of their parameters frozen. Reuse the same
  9-class mapping and invalid-pixel policy as v1.
- Local RTX 3050, one arm at a time. Fixed seed, identical shuffled training
  record order and crop coordinates per arm. AdamW + per-step OneCycle,
  1,600 steps per arm, full held-out evaluation at 400/800/1200/1600.
- Check EPE, RMSE, bad-0.5/1/2/3, D1, mapped mIoU, pixel accuracy,
  edge/non-edge EPE, trainable parameters, latency, and VRAM. Save configs,
  loss trace, plots, and all four checkpoints per arm.

## Arms (add one idea at a time)

1. Frozen baseline and **v1 joint** on the enlarged subset — fair control.
2. **v2 gate:** compressed semantic embedding, separated geometry/semantic
   features, task-specific confidence-gated residuals.
3. **v3 warp:** v2 plus right image warped with the frozen disparity and a
   photometric disagreement/validity cue. Same train/held-out data.
4. **v4 edge:** only if the diagnostic shows a substantial edge EPE gap; v3
   plus shallow full-resolution edge residual and masked boundary-weighted
   disparity loss. Otherwise record why it was skipped rather than running an
   unjustified arm.

No arm is selected from training loss alone. Compare the same held-out split
and inspect per-class IoU; the split remains a one-scene feasibility test, not
a generalization benchmark. Prototype latency includes two instantiated
predictors and must not be labeled one-shot shared-encoder runtime.
