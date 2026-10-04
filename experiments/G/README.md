# G-series: visual edge refinement on frozen E3

This is a local RTX 3050 ablation, not another Modal run. It uses the B-series
VKITTI2 1,000-pair manifest with 800 train, 100 validation, and 100 test pairs;
the Scene20 holdout is grouped by source frame across ten variations. E3 and
its shared YOLO encoder, semantic decoder, stereo head and E3 fusion head are
frozen. Only each G head trains. Inputs are native pixels, with paired 256×512
training crops and top/right padding for full-resolution evaluation. No resize.

- `G_1_rgb_residual`: learned full-resolution RGB-guided disparity residual.
- `G_2_convex`: G1 plus learned nine-neighbor convex reconstruction from E3's
  half-resolution tile disparity.
- `G_3_stereo_correct`: G2 plus a small left/right warped-image residual.

All use seed 42, 10,000 steps, OneCycle/AdamW and the same boundary-aware loss.
Validation edge-region EPE selects the checkpoint; test is scored once after
selection. `test.json` reports EPE, RMSE, bad-0.5/1/2/3, D1 and the frozen E3
baseline on the identical pixels. `visuals/` contains full-native-size
left/ground-truth/E3/head comparison sheets at a fixed 0–80 px color scale.
The full VKITTI E3 teacher may already have seen these images; this study tests
appearance and in-domain head behavior, **not** independent generalization.

Queue status:

```bash
jq . experiments/G/queue/g_edge_native_seed42_10k_v2_20261004/status.json
tail -f experiments/G/queue/g_edge_native_seed42_10k_v2_20261004/G_1_rgb_residual.log
tail -f experiments/G/queue/g_edge_native_seed42_10k_v2_20261004/G_2_convex.log
tail -f experiments/G/queue/g_edge_native_seed42_10k_v2_20261004/G_3_stereo_correct.log
```

The queue stops on an arm failure and records the failing log in `status.json`.
