# C-series insights: larger late-fusion heads and causal controls

The C arms use the same frozen A09 stereo/YOLO26m semantic predictors and
800/100/100 frame-grouped VKITTI2 split as B. They train larger depth-only
heads with native-resolution validation/test and no image/disparity resize.
The original arms are [C1](C_1_large_control/runs/c_v1_seed42_20261001/test.json),
[C2](C_2_feature_fusion/runs/c_v1_seed42_20261001/test.json), and
[C3](C_3_semantic_match/runs/c_v1_seed42_20261001/test.json). C4 and C5
are equal-architecture C2 controls trained with absent or spatially
misaligned semantics throughout, not inference-only perturbations.

| Arm | Semantic guidance | EPE (px) | Bad-1 (%) | Bad-3 (%) | D1 (%) | Trainable params |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C1 larger output control | aligned logits | 2.1796 | 26.350 | 11.850 | 11.501 | 230,849 |
| C2 multi-scale feature fusion | aligned logits/features | **2.0024** | 24.695 | **10.889** | **10.688** | 369,137 |
| C3 right-feature mismatch | aligned guidance | 2.0596 | 25.255 | 11.530 | 11.335 | 395,281 |
| [C4](C_4_no_semantics/runs/c_causal_v1_seed42_20261001/test.json) | none; same C2 architecture | 2.0374 | **24.462** | 10.911 | 10.713 | 369,137 |
| [C5](C_5_misaligned_semantics/runs/c_causal_v1_seed42_20261001/test.json) | shifted; same C2 architecture | 2.0359 | 24.685 | 11.043 | 10.847 | 369,137 |

C2 is the lowest-EPE C arm, but its **0.0351 px** test advantage over C4
is too small and frame-dependent to present as an established semantic
effect. C2's bad-1 is actually **0.233 percentage points worse** than C4's.
Most of the gain over the frozen stereo baseline can be explained by the
extra multi-scale feature capacity; the no-semantics control still reaches
2.0374 px. C3's extra right-feature mismatch does not beat C2 on EPE or
outlier rates. Frozen segmentation quality is unchanged across these arms:
GT-present 14-class test mIoU **0.7345**. This differs from B's
union-present mIoU definition and must not be compared directly.

**Interpretation:** late feature fusion is useful for disparity, but the C
series does not demonstrate a robust semantic-specific gain. Its natural
follow-up was D's semantic agreement at the **disparity candidate volume**,
before regression, rather than another output-only residual. See
[D-series insights](../D/INSIGHTS.md). These remain in-domain results only.
