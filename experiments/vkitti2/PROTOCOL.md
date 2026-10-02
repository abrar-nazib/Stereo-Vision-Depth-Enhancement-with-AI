# VKITTI2 joint-residual feasibility ablation — v1

Purpose: test whether a small trainable head can improve disparity and mapped
semantic segmentation while both existing predictors remain frozen. This is a
feasibility ablation, not full-dataset training or a final benchmark.

## Locked data and provenance

- Source: Virtual KITTI 2 official RGB, depth, class-segmentation, and text-GT
  archives, previously MD5-verified in Modal volume `svde-vkitti2`.
- Subset: 500 deterministic pairs, ten evenly spaced frames in each of ten
  variations for each of five scenes. The selection function and file paths are
  in `subset.py`; exact records and intrinsics are in the downloaded
  `ablation500/manifest.json`.
- Train: Scene01, Scene02, Scene06, Scene18 (400 pairs). Held-out: Scene20
  (100 pairs). No scene location crosses the split.
- Each pair contains left and right RGB, left 16-bit depth, and left color-coded
  class mask. Depth is centimetres; disparity in pixels is `fx * 0.532725 / Z_m`.
  Invalid zero/sentinel values and disparity >=192 px are excluded.
- The subset tar is 302,448,640 bytes, SHA-256
  `8281497a60a7229e8043c80aefb40452004ec48b9b7ab88db177fc08c1eb3e8d`.

## Models and arms

All arms use the A09m plateau best checkpoint at step 120,000 and the local
`yolo26m-sem-ade20k.pt`. Both complete predictors are set to eval mode and
their parameters have `requires_grad=False`. The stereo checkpoint's encoder
is frozen too. The new head sees frozen full-resolution disparity, frozen
150-class stride-8 semantic logits, and the left image. A zero-initialized
last convolution starts each residual arm at exactly the frozen predictions.

1. **baseline**: no trainable residual; score the frozen predictors.
2. **depth_only**: train a small disparity residual; semantic logits unchanged.
3. **joint**: train disparity and nine ADE20K semantic-logit residuals together.

Mapped VKITTI → ADE20K IDs: building 1, sky 2, tree 4, road 6, car 20,
truck 83, pole 93, van 102, traffic light 136. Ambiguous or absent classes
(terrain, vegetation, guardrail, sign, misc, undefined) are ignored for
semantic training and evaluation. The semantic metric is macro IoU over mapped
classes with nonzero union, not a 150-class ADE20K benchmark.

## Training and validation

- Local RTX 3050 4 GB only; one arm at a time; one deterministic seed 260930.
- 800 steps per trainable arm, batch 1, randomly sampled 256×512 aligned crop.
  No resizing. Pixels whose right-camera correspondence falls outside the crop
  are excluded from the training disparity loss.
- AdamW, max LR 2e-4, weight decay 1e-4; per-step OneCycleLR, 10% warm-up;
  gradient-norm clipping 1.0. Loss: smooth L1 disparity + 0.5×cross-entropy
  over mapped semantic labels for the joint arm. No encoder/head fine-tuning.
- Every 200 steps, validate all 100 held-out pairs at native 375×1242,
  replicate-padded to a multiple of 16 and cropped back before scoring.
- Report valid-pixel EPE, RMSE, bad-0.5/1/2/3, D1 (>3 px **and** >5%),
  mapped-class mIoU, pixel accuracy, trainable parameters, and elapsed time.
- Save per-arm `history.jsonl`, `curves.png`, every-200-step checkpoint, and
  `best_epe.pth`. `config.json` hashes the exact manifest and source checkpoint.

The semantic and stereo predictors are independently instantiated for this
head feasibility test, even though the stereo encoder originated from the same
YOLO weights. Do **not** report this prototype's runtime or total parameter
count as a deployed shared-encoder one-shot model.
