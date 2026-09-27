# A02 native v1.0: 100-pair Driving capacity study

Full 540x960 images, padding only. Errors are in native-image pixels. Same 100 stereo pairs for training and evaluation; this measures fitting capacity, not generalization.

| Model | EPE | RMSE | bad-0.5 % | bad-1 % | bad-2 % | bad-3 % | D1 % | Mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lightstereo FP32 | 1.945 | 5.697 | 38.949 | 26.964 | 17.750 | 13.188 | 11.328 | 64.44 |
| las1 FP32 | 3.709 | 8.668 | 63.421 | 48.967 | 33.057 | 24.886 | 19.861 | 85.85 |
| las2s FP32 | 6.674 | 17.672 | 69.612 | 54.886 | 40.557 | 32.054 | 26.099 | 59.60 |
| las2m FP32 | 2.777 | 7.519 | 47.434 | 32.722 | 22.308 | 17.365 | 15.205 | 78.02 |
| Original LAS2-S FP16 | 6.675 | 17.668 | 69.618 | 54.892 | 40.560 | 32.059 | 26.110 | 57.28 |
| Hybrid before training FP16 | 88.645 | 102.492 | 99.316 | 98.796 | 98.027 | 97.320 | 94.733 | not measured |
| Hybrid after 500 updates FP16 | 3.805 | 8.621 | 65.429 | 46.729 | 31.024 | 23.864 | 20.782 | 58.13 |

Hybrid minus original LAS2-S FP16 EPE: -2.869 px (negative is better).

## Interpretation

The trained hybrid has seen these 100 pairs; the original checkpoints were not fine-tuned in this run. This is not evidence of generalization superiority. Original checkpoint training may also include Driving.
Five passes are a short feasibility budget, not a converged ablation. The original model timing produces disparity only; hybrid timing also produces full-resolution semantic labels. FP32 audit latencies and FP16 training-run latencies are separate comparisons.

## Configuration and verification

- Architecture: A01-v1.0; shared frozen YOLO26s-sem, four trainable adapters, retained LAS2-S FPN/head.
- Parameters: 10,559,662 total, 4,056,520 trainable.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU; batch 1; AdamW LR 1e-4; 500 updates; seed 42.
- Training time: 68.23 s; peak allocated GPU memory: 758.61 MiB.
- Frozen semantic weights and buffers unchanged: True.
- Separate forward equivalence check against original YOLO semantic output passed at 128x256 (FP32, max absolute difference 6.68e-6).
- No semantic ground truth or mIoU measurement. Class predictions are semantic categories, not separate object instances.
- Metric mask: finite GT, 0 < d < 192; D1 requires >3 px AND >5%; macro-average across images.
- Exact pairs and checkpoint/source fingerprints: manifest.json and config.json.
- A01 resize results are superseded; they must not be used as the native benchmark.

![Training loss](training_curve.png)
