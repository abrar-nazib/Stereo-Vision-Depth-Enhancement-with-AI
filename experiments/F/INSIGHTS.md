# F-series KITTI 2015 transfer insight

Run `f1_f2_kitti2015_v1_20261003`, evaluated 2026-10-03 on one Modal T4
(detached app `ap-xCpWCVAzMaCzUZzPAXvv92`, function call
`fc-01M3Z9E9S1T1FXQ1AXN1MZ32GT`). F3 was evaluated by detached T4 app
`ap-KVIg46HNWNDvnB74i5ZsJX` (function call `fc-01M3ZA86TZKZV5AF26CJVB8W7F`).
All three completed on the same 200
public labeled KITTI 2015 training pairs, with no KITTI optimization or
checkpoint selection. See [F1](F_1_kitti2015_baseline/runs/f1_f2_kitti2015_v1_20261003/result.json)
and [F2](F_2_kitti2015_e3/runs/f1_f2_kitti2015_v1_20261003/result.json)
and [F3](F_3_kitti2015_e4_control/runs/f1_f2_kitti2015_v1_20261003/result.json)
for all per-pair values. The two-pair probe is not a research result.

| KITTI 2015 all-valid GT (`disp_occ_0`) | F1 frozen A09 stereo | F2 full E3 | F2 gain |
| --- | ---: | ---: | ---: |
| EPE ↓ (px) | 3.0822 | 2.5876 | 0.4946 px (16.0%) |
| RMSE ↓ (px) | 6.6024 | 5.5439 | 1.0585 px |
| bad-0.5 ↓ (%) | 81.1524 | 77.6213 | 3.5310 pp |
| bad-1 ↓ (%) | 63.6533 | 57.9741 | 5.6792 pp |
| bad-2 ↓ (%) | 38.6154 | 32.4519 | 6.1634 pp |
| bad-3 ↓ (%) | 25.3085 | 20.5277 | 4.7808 pp |
| D1 ↓ (%) | 24.9956 | 20.2017 | 4.7939 pp |
| Median T4 inference time ↓ (ms/pair) | 37.63 | 59.76 | F2 costs 22.13 ms |
| Peak allocated VRAM (GiB) | 0.360 | 0.597 | F2 costs 0.237 GiB |

F2 EPE improves on **196/200** individual stereo pairs and bad-3 on
**194/200**. The median per-pair EPE reduction is 0.3700 px. Four EPE
regressions are `000055_10`, `000144_10`, `000157_10`, and `000194_10`; retain
them for visual failure analysis. Both arms score the same 18,375,134 GT
pixels in the all-valid mask. Just two labeled pixels exceed the model's
192 px disparity range; both arms exclude those two. The KITTI non-occluded
mask gives EPE **3.0411 → 2.5332 px** and bad-3 **24.9630 → 20.0859%**.

## Equal-capacity F3 control

F3 runs the frozen E4 no-semantics checkpoint. It has the same 38,606
trainable head parameters and frozen predictor architecture as F2, selected
using the same VKITTI validation bad-3 rule. Its semantic tails still execute,
but semantic inputs to the gate/refiner are zeroed in both training and
evaluation. It therefore controls for a trained head of the same capacity.

| KITTI all-valid GT | F1 stereo only | F3 no semantics | F2 semantic | F2 vs F3 |
| --- | ---: | ---: | ---: | ---: |
| EPE ↓ (px) | 3.0822 | 2.7139 | **2.5876** | **-0.1263 px** |
| RMSE ↓ (px) | 6.6024 | 5.9532 | **5.5439** | -0.4093 px |
| bad-3 ↓ (%) | 25.3085 | 20.9902 | **20.5277** | -0.4624 pp |
| D1 ↓ (%) | 24.9956 | 20.6439 | **20.2017** | -0.4422 pp |
| Median T4 inference ↓ (ms/pair) | **37.63** | 58.70 | 59.76 | +1.06 ms |

F2 improves EPE versus F3 on **137/200** pairs and bad-3 on **121/200**;
median per-pair improvements are 0.0731 px and 0.3797 pp. The globally
valid-pixel-weighted gains are larger, so some difficult pairs account for
substantial benefit. A 20,000-resample paired image bootstrap (seed 42,
resampling whole pairs and weighting by each pair's valid GT pixels) gives
a 95% interval of **[0.074, 0.207] px** for F3 minus F2 EPE and
**[0.200, 0.728] percentage points** for bad-3. These intervals describe
uncertainty across these 200 pairs, not across all possible road domains.

The complete F2/E3 system transfers better than its frozen SceneFlow-trained
A09 stereo base to real KITTI 2015 road images, after the semantic model and
fusion head were trained on synthetic VKITTI2. F2 beating the matched F3
control is additional evidence that semantic guidance contributes beyond
head capacity on this external dataset. It remains a single real-road
benchmark, not a guarantee for arbitrary domains. KITTI segmentation mIoU
was not measured because the deployed 14-class VKITTI taxonomy differs from
KITTI's taxonomy. KITTI 2015 benchmark test images have no public disparity
ground truth; these are the 200 labeled *training-set* images used only as an
untuned external evaluation set here.

Protocol: left `image_2`, right `image_3`, 16-bit disparity PNG divided by
256; RGB 0–255, no resize, replicate padding to multiples of 32; valid GT
finite, greater than zero, and less than 192 px. Scores are globally
valid-pixel-weighted. Runtime excludes the first image as warm-up and includes
GPU inference only, not PNG decoding or model loading. F1 is the exact A09
`FusionStereoLite("V")` stereo checkpoint; its trained stereo-side veto block
is retained. F2 adds E3's selected step-16,000 semantic gate and residual,
with stereo and YOLO predictors frozen. F3 substitutes E4's selected
step-22,000 no-semantics head. See [README.md](README.md) for
reproduction and polling commands.
