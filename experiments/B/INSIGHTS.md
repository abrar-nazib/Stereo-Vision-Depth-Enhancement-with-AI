# B-series insights: late semantic-to-disparity heads

Run `20261001_b_full_seed42` used 800 train / 100 validation / 100 test
VKITTI2 pairs, grouped by source frame. The A09 stereo and 14-class semantic
predictors were frozen; only each small correction head trained. All disparity
metrics below are native-resolution test values, from the
[full B-series report](B_0_fused_baseline/runs/20261001_b_full_seed42/REPORT.md).

| Arm | EPE (px) | Bad-1 (%) | Bad-3 (%) | D1 (%) | Trainable params |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 frozen | 2.2730 | 26.52 | 12.29 | 12.08 | 0 |
| B1 residual | **2.2159** | 25.80 | **11.72** | 11.54 | 5,345 |
| B2 confidence | 2.2172 | 25.83 | **11.72** | **11.53** | 6,226 |
| B3 class-aware | 2.2174 | **25.66** | 11.78 | 11.59 | 4,666 |

B1 was selected by validation EPE. Its test improvement over B0 was
**0.0571 px**; a ten-source-frame paired bootstrap in the report gives a
95% interval of **[−0.0903, −0.0272] px** for B1 − B0. Its segmentation
output did not change: the frozen model's union-present 14-class mIoU on
this subset was **0.6780**. B1 also had the practical latency advantage:
median **64.25 ms** versus B2's **85.32 ms** on the local RTX 3050
(native 375 × 1242, model forward only). The report's training curve and
example disparity/error panels are the key visual figures.

**Interpretation:** B shows that a tiny frozen-predictor correction head can
improve disparity. It does **not** isolate semantic causality: there is no
B1-equivalent head trained without semantic input. Uniform/shifted guidance
at inference worsened validation EPE, but that post-training perturbation
is not a matched training control. The semantic teacher's random full-VKITTI
split may overlap the depth test frames; no real-domain claim follows.
