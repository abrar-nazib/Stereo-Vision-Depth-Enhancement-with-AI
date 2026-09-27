# YOLO26s semantic / LAS2-S shared encoder — architecture v1.0

**Current protocol: A02 native v1.0.** A01 resized runs are superseded and
retained only for audit. Native full-image inference/training is the default.

This isolated experiment combines the frozen Cityscapes YOLO26s semantic model
with the pretrained LAS2-S stereo feature pyramid and disparity head. The
FasterNet image backbone is removed. Four learned 1x1 adapters map YOLO layers
2/4/6/10 (128/256/256/512 channels) to LAS2 inputs (40/80/160/320).
The original LAS2 FPN, attention/aggregation, regression and upsampling are
fine-tuned. The small image-guidance stem receives LAS2-normalized input;
YOLO receives RGB in [0,1]. Both images share the same YOLO weights. Only
left-image features pass through the semantic neck and head.

## Protocol

Run from the repository root with the project uv environment:

```sh
uv run python experiments/yolo_las2_v1/run.py --steps 500
```

- 100 distinct stereo pairs, evenly allocated across available Driving
  sequence directories and uniformly spaced over their sorted frame indices.
  Driving sequences are correlated; these are NOT 100 independent scenes.
- Full 540x960 finalpass images and unchanged native disparity. Symmetric
  replicated image padding to 544x960, removed before loss/metrics. No resize,
  crop, disparity rescaling, or augmentation. Original LAS2 and hybrid use
  identical padding. LightStereo audit uses its official top/right padding.
- Train and evaluate on the same 100 pairs: this is an overfit/capacity study,
  NOT a held-out validation or zero-shot benchmark.
- Seed 42, batch 1, 500 updates (five passes), AdamW, LR 1e-4, weight decay
  1e-4, gradient norm cap 1, FP16 autocast and GradScaler. No scheduler.
- Loss: smooth L1 at full resolution plus 0.3 times smooth L1 on the
  upsampled coarse disparity. Valid GT: finite, positive, less than 192 px.
- YOLO parameters and BatchNorm buffers frozen, checked after training.
  Semantic neck skipped during stereo-only training; included at evaluation
  and joint inference timing. No semantic labels: mIoU is not measured.
- Original LAS2-S, randomly initialized-adapter hybrid, and trained hybrid
  evaluated on identical pairs and masks. All metrics are macro-averaged
  across images. D1 requires BOTH >3 px and >5% GT; bad-t requires >t px.
- CUDA-synchronized timings: batch 1, 10 warm-ups, 50 measured forwards,
  GPU-resident input. Mean/median/p95/FPS and peak allocated memory recorded.
  Original LAS2 timing is disparity only; hybrid timing includes semantic
  logits resized to full resolution and argmax, excluding point clouds/I/O.

The `evaluate_native.py` audit independently evaluates LAS1, LAS2-S/M and
LightStereo-S in FP32 on the saved manifest. The training runner uses FP16
for both original LAS2-S and hybrid evaluation; compare precision-matched
rows when deciding whether adaptation helped. Native-image audit records
include valid-pixel coverage because the fixed 192-pixel search range masks
some Driving ground truth.

## Records and limitations

Each timestamped run has config/source/checkpoint hashes, upstream commit,
exact file manifest, train CSV, per-image error CSVs, metrics JSON, and final
checkpoint including optimizer/scaler state. Checkpoints are gitignored.
An adapter gradient smoke test verifies stereo learning reaches the adapters.
Original backbone weights are retained only in the separate baseline model;
baseline and hybrid execute sequentially to fit the RTX 3050.

This initial study tests whether the hybrid can learn. Five passes are a
short diagnostic budget, not evidence of convergence. A worse result than
original LAS2 after this budget cannot establish that the architecture is
incapable. Original LAS2 may already have seen Driving during pretraining.
Published zero-shot numbers must not be compared directly to this protocol.
Version any architecture/protocol changes as a new study and preserve A01.
