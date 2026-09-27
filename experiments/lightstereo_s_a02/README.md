# Ablation Test 2 — LightStereo-S control

This is the 200-pair native-resolution SceneFlow Driving benchmark for the
unmodified official `LightStereo-S-SceneFlow.ckpt` checkpoint. It is a control
benchmark, not a YOLO-backbone replacement experiment.

Run it with:

```bash
uv run python experiments/lightstereo_s_a02/run.py --count 200
```

It records the immutable pair manifest, checkpoint hash, FP32 native metrics,
per-image errors, and GPU latency in a timestamped directory under `runs/`.
Input frames are never resized; they receive only LightStereo's required
top/right replicate padding to a multiple of 32.

## A02B: frozen YOLO26s backbone replacement

`run_yolo_backbone.py` replaces LightStereo-S's MobileNetV2 feature encoder
with the frozen YOLO26s Cityscapes semantic encoder. Four learned 1×1 adapters
map YOLO feature channels `[128, 256, 256, 512]` to LightStereo's expected
`[24, 32, 96, 160]` pyramid. The pretrained LightStereo aggregation and
refinement head remains trainable so it can adapt to the new feature space.

```bash
uv run python experiments/lightstereo_s_a02/run_yolo_backbone.py --steps 1000
```

It creates a fixed 200-pair manifest, a deterministic 160/40 stratified
train/validation split, trains only on co-located native 384×640 crops, and
evaluates the 40 held-out pairs at their full 540×960 resolution. No operation
resizes an image or rescales disparity.
