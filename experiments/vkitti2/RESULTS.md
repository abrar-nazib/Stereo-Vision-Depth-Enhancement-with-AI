# VKITTI2 500-pair frozen-head ablation — v1 results

Completed locally on 2026-09-30 using the RTX 3050 Laptop GPU. The exact
selection, losses, mapping, and caveats are in [PROTOCOL.md](PROTOCOL.md).
The tar on the external SSD passed the Modal-generated SHA-256 check before
extraction. `config.json` in the run records the source checkpoint and manifest
hashes. All three arms use the same 100-pair held-out Scene20 set at native
375×1242 resolution; no images were resized.

| Arm / step | Added trainable parameters | EPE ↓ px | RMSE ↓ px | bad-0.5 ↓ % | bad-1 ↓ % | bad-2 ↓ % | bad-3 ↓ % | D1 ↓ % | mapped mIoU ↑ | mapped pixel acc. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 0 | 2.6314 | 8.0521 | 45.8403 | 28.5668 | 17.7876 | 13.6511 | 13.2157 | 0.4649 | 0.8903 |
| Depth-only / 800 | 53,921 | **2.5542** | **7.9584** | 46.2024 | 27.5869 | **16.4870** | **12.6885** | **12.2927** | 0.4649 | 0.8903 |
| Joint / 800 | 56,522 | 2.5748 | 8.0095 | **45.6458** | **27.4340** | 16.6861 | 12.9156 | 12.5288 | **0.5315** | **0.9479** |

The joint head improved both frozen predictions in this split: EPE fell by
0.0566 px (2.15%) and mapped mIoU rose by 0.0666 (6.66 percentage points).
Compared with depth-only, it gave up 0.0206 px EPE while raising mIoU by
0.0666. Depth-only has the best EPE, but its bad-0.5 score is slightly worse
than the frozen baseline, so the gain is not uniform across thresholds.

## Full held-out progression

| Step | Depth-only EPE | Depth-only mIoU | Joint EPE | Joint mIoU |
|---:|---:|---:|---:|---:|
| 0 (frozen) | 2.6314 | 0.4649 | 2.6314 | 0.4649 |
| 200 | 2.5818 | 0.4649 | 2.5833 | 0.5293 |
| 400 | 2.5590 | 0.4649 | 2.5954 | 0.5338 |
| 600 | 2.5584 | 0.4649 | 2.5884 | **0.5382** |
| 800 | **2.5542** | 0.4649 | **2.5748** | 0.5315 |

The best *joint* semantic checkpoint is step 600; the best joint disparity
checkpoint is step 800. Both files are retained, so this choice remains
explicit for later work. Each trained arm ran for roughly six minutes,
including four full validation passes. The baseline evaluation ran first.

## Semantic class detail at step 800

| ADE20K mapped class | Baseline IoU | Joint IoU |
|---|---:|---:|
| Building | 0.006 | 0.007 |
| Sky | 0.864 | 0.905 |
| Tree | 0.894 | 0.897 |
| Road | 0.830 | 0.968 |
| Car | **0.908** | 0.880 |
| Truck | 0.150 | 0.311 |
| Pole | 0.409 | 0.606 |
| Van | 0.123 | 0.208 |
| Traffic light | 0.000 | 0.000 |

The result supports the narrow feasibility claim that a small residual head
can learn to improve *both* frozen outputs on this held-out split. It does not
establish all-class semantic quality: building and traffic-light IoU remain
near zero, and car IoU worsens. The 100 validation frames are distinct frames
but all come from one scene location with ten variations, so they are not 100
independent geographic scenes. One seed and one train/validation split cannot
establish a generalization or significance claim for a paper. Also, this
prototype runs the YOLO semantic model and stereo model separately; its total
runtime and parameter count must not be reported as those of a one-shot
shared-encoder architecture.

Artifacts: [`runs/ablation_v1_20260930`](runs/ablation_v1_20260930/) contains
`config.json`, `baseline.json`, `summary.json`, per-arm `history.jsonl`,
`curves.png`, checkpoints at steps 200/400/600/800, and `best_epe.pth`.
