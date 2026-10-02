# B-series 1,000-pair ablation report

Run ID: `20261001_b_full_seed42`. 800 train / 100 validation / 100 test; source-frame grouped.
Both pretrained predictors were frozen; only B1–B3 disparity heads trained.
All disparity metrics are native-resolution; EPE and RMSE are in pixels, bad/D1 in percent.

| Arm | Selected step | Val EPE | Test EPE | Test RMSE | Test bad-1 | Test bad-3 | Test D1 | Test mIoU | Trainable params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 frozen | — | 2.5597 | 2.2730 | 6.9627 | 26.52 | 12.29 | 12.08 | 0.6780 | 0 |
| B_1_residual | 1200 | 2.4972 | 2.2159 | 6.8785 | 25.80 | 11.72 | 11.54 | 0.6780 | 5,345 |
| B_2_confidence | 1200 | 2.4993 | 2.2172 | 6.8789 | 25.83 | 11.72 | 11.53 | 0.6780 | 6,226 |
| B_3_classaware | 400 | 2.5027 | 2.2174 | 6.8796 | 25.66 | 11.78 | 11.59 | 0.6780 | 4,666 |

## Frozen semantic output on Scene20 test

IoU is reported only for classes with nonzero prediction/label union; an absent class is shown as a dash. The disparity heads cannot change these values.

| Class | IoU |
| --- | ---: |
| Terrain | 0.9463 |
| Sky | 0.9018 |
| Tree | 0.8726 |
| Vegetation | 0.0000 |
| Building | 0.4570 |
| Road | 0.9642 |
| GuardRail | 0.8216 |
| TrafficSign | 0.8483 |
| TrafficLight | — |
| Pole | 0.5328 |
| Misc | 0.0000 |
| Truck | 0.9386 |
| Car | 0.9417 |
| Van | 0.5894 |

![B-series training and validation curves](B_series_progress.png)

## Causal controls (validation EPE)

Uniform or spatially shifted semantic logits are passed to the same selected head.
A useful semantic head should lose its gain when guidance is invalid.

| Arm | Normal | Uniform | Shifted |
| --- | ---: | ---: | ---: |
| B_1_residual | 2.4972 | 2.6127 | 2.5973 |
| B_2_confidence | 2.4993 | 2.6176 | 2.5940 |
| B_3_classaware | 2.5027 | 2.6118 | 2.5813 |

## Paired frame-group uncertainty for B1 − B0 EPE

Ten Scene20 source frames per partition were resampled with replacement (10,000 draws); all ten variants of a frame stay together.

| Partition | EPE delta (px) | 95% bootstrap interval (px) |
| --- | ---: | ---: |
| val | -0.0625 | [-0.0913, -0.0384] |
| test | -0.0571 | [-0.0903, -0.0272] |

## End-to-end GPU latency

Native 375×1242 input; includes fusion and class-confidence output, excludes image I/O.
The extra photometric warp makes B2 substantially slower on this RTX 3050.

| Arm | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| B_0_fused_baseline | 64.55 | 66.02 |
| B_1_residual | 64.25 | 65.96 |
| B_2_confidence | 85.32 | 93.56 |
| B_3_classaware | 64.45 | 66.20 |

## Qualitative held-out examples

Each panel uses the same disparity color scale for GT, B0 and B1; error scales are shared too.
![Scene20 clone](figures/Scene20_clone_00000.png)
![Scene20 rain](figures/Scene20_rain_00000.png)
![Scene20 fog](figures/Scene20_fog_00000.png)

## Interpretation limits

The semantic checkpoint was fine-tuned using a random full-VKITTI split that likely overlaps this subset. This is an in-domain fusion feasibility test, not independent real-world generalization.
Scene20 validation and test each contain only ten independent source frames; their ten variations per frame are correlated.

Do not compare these EPE values directly to the previous 800/200 split or to the semantic checkpoint's full-dataset random-split mIoU.
