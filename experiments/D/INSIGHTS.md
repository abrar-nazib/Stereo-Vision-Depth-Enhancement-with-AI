# D-series insights: semantic guidance at the stereo cost volume

Status: all four local RTX 3050 runs in `d_v1_seed42_20261002` completed.
This is **in-domain VKITTI2 evidence**, not a cross-domain or real-camera
generalization result. The D implementation adapts SGNet's candidate
confidence and category-conditioned refinement ideas to the frozen A09
StereoLite-style predictor; it is not a reproduction of SGNet.

## Protocol and what was held fixed

- The same 800-train / 100-validation / 100-test VKITTI2 subset and seed 42
  frame-grouped Scene20 split as B/C. The 100 test images represent **ten
  independent source frames**, each under ten variations. Training uses paired
  256 × 512 native-pixel crops; validation/test use full images with padding,
  without image/disparity resizing.
- The A09 stereo predictor, shared YOLO26m encoder, and VKITTI 14-class
  semantic decoder are frozen. Both views share one encoder call; the semantic
  decoder processes left and right views separately. All arms do the same
  frozen inference work. Only the indicated D modules train.
- All arms use the same optimizer, OneCycle schedule, 10,000-step ceiling,
  outlier-aware loss, validation cadence and bad-3 checkpoint-selection rule.
  Selected checkpoints differ: D1/D2/D4 at step 6,000; D3 at step 8,000.
- D1/D3 each have **657** trainable parameters; D2/D4 each have **9,758**.
  D3/D4 replace semantic guidance with zeros throughout *training and test*,
  retaining the same module sizes and stereo-correlation inputs. They are
  matched-capacity controls, not inference-only perturbations.
- Checkpoint/data SHA-256 hashes, exact settings and GPU are in each arm's
  `runs/d_v1_seed42_20261002/manifest.json`. Each arm has its own validation
  trace, test metrics and per-pair test results. The queue completion record is
  `D_1_semantic_cost/queue/d_v1_seed42_20261002/status.json`.

## Best-checkpoint test results

Lower is better for every disparity metric. Bad-* and D1 are percentages;
EPE/RMSE are pixels. All four arms have the same frozen segmentation output:
GT-present 14-class mIoU **0.7345** on this test subset. This mIoU protocol
differs from B-series union-present mIoU, so do not compare those numbers.

| Arm | Guidance | EPE | RMSE | Bad-0.5 | Bad-1 | Bad-2 | Bad-3 | D1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [D1](D_1_semantic_cost/runs/d_v1_seed42_20261002/test.json) cost gate | aligned left/right semantics | 2.0350 | 6.5117 | 42.195 | 24.602 | 14.515 | 10.921 | 10.695 |
| [D3](D_3_no_semantic_cost/runs/d_v1_seed42_20261002/test.json) matched cost gate | no semantics | 2.2774 | 7.0540 | 43.307 | 25.803 | 15.621 | 11.956 | 11.737 |
| [D2](D_2_semantic_cost_residual/runs/d_v1_seed42_20261002/test.json) cost gate + residual | aligned left/right semantics | **2.0286** | 6.5145 | 43.171 | **24.359** | **13.926** | **10.512** | **10.291** |
| [D4](D_4_no_semantic_residual/runs/d_v1_seed42_20261002/test.json) matched gate + residual | no semantics | 2.2765 | 7.0797 | 44.461 | 24.975 | 15.144 | 11.708 | 11.509 |

For the direct semantic-input contrasts:

- **D1 − D3:** EPE −0.2424 px and bad-3 −1.0349 percentage points. EPE and
  bad-3 improved on **all ten** source frames.
- **D2 − D4:** EPE −0.2479 px and bad-3 −1.1959 points. EPE improved on
  **all ten** source frames; bad-3 improved on **nine of ten**.
- A paired, source-frame-cluster bootstrap with 20,000 resamples and seed 42,
  weighting frames by valid disparity pixels, gave D1 − D3 95% intervals of
  **[−0.370, −0.138] px** EPE and **[−1.541, −0.595] points** bad-3;
  D2 − D4 intervals were **[−0.378, −0.143] px** and
  **[−1.762, −0.689] points**. These quantify variation over ten test
  frame groups, not variation over training seeds or domains.

## Interpretation

This is the strongest *in-domain* semantic-specific disparity evidence so far.
The small D1 gate, which compares class distributions across each candidate
left/right match before the initial disparity regression, accounts for most
of the gain. Unlike C2's ~0.035 px test advantage over its no-semantics
control, D1's EPE/bad-3 improvement appears across the source frames and its
frame-group bootstrap intervals exclude zero. The class-conditioned residual
adds a more targeted outlier benefit: D2 versus D1 reduces bad-3 by **0.4092
points** (nine of ten frame groups; bootstrap interval
**[−0.623, −0.221]**), while EPE changes only **−0.0064 px** (five of ten;
interval **[−0.023, +0.009]**). Do not call the residual's EPE gain proven.

The frozen [B0 baseline](../B/B_0_fused_baseline/runs/20261001_b_full_seed42/test.json)
is 2.2730 EPE / 12.2888% bad-3 on the same depth test subset. Thus D2
improves both metrics over the original predictor, not only over a weak D4
control. The [C2 feature-fusion test](../C/C_2_feature_fusion/runs/c_v1_seed42_20261001/test.json)
has slightly better EPE (2.0024 versus D2's 2.0286), but worse bad-3
(10.8890% versus 10.5119%) and many more trainable parameters (369,137
versus D2's 9,758). D2 is the preferable D-series arm when **outlier
reduction** is the priority; C2 remains the lower-EPE in-domain result.

## Claim boundaries and next checks

1. There is only **one training seed** and only ten independent depth-test
   source frames. Repeating with fresh seeds/frame groups is needed before
   calling the effect stable beyond this subset.
2. The frozen semantic teacher was fine-tuned with a random, frame-grouped
   split *across all VKITTI scenes*. We have **not verified** that D's Scene20
   test frames were excluded from that teacher's training split. This is a
   semantic-label leakage risk even though the D head never trains on the
   held-out Scene20 disparity. Verify or eliminate overlap before a paper's
   independent-test claim. See
   [semantic run provenance](../vkitti2/SEMANTIC_MODAL_RUN.md).
3. Neither real-domain disparity accuracy nor model latency/VRAM overhead
   has been measured for D. Right-view semantic-tail inference adds compute;
   do not call the model real-time or real-world-generalizing yet.
4. Do not infer that semantics improved segmentation: its frozen mIoU is
   identical across arms. The supported claim is narrower: **aligned
   semantic input to the stereo candidate gate reduced in-domain disparity
   errors and outliers in this frozen-predictor architecture**.
