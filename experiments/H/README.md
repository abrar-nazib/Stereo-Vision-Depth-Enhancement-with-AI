# H-series: stereo-only final-upsampling comparison

This is the controlled follow-up to G. No semantic decoder, predicted class
map, or E3 fusion head runs. The current v2 comparison uses a **fresh random
A09 stereo-side initialization** in every arm; YOLO26m ADE20K layers 0–6 are
pretrained but remain frozen in evaluation mode. The same random seed yields
the same initial stereo tensors in all three arms. The stereo module then
trains for 25,000 steps in every arm. No SceneFlow disparity-head checkpoint
is loaded in v2.

The superseded v1 queue (`h_stereo_only_2k_25k_seed42_v1`) was stopped after
5,400 steps because it initialized from the fully SceneFlow-trained A09
disparity head. Its artifacts are retained for provenance, **not** included
in the scratch-head comparison.

| Arm | Difference at the 1/2-to-full stage |
|---|---|
| `H0_a09_scratch` | Original A09 plane-tile upsample; matched scratch-head control |
| `H1_rgb_guided_scratch` | Original plane upsample plus a bounded RGB-guided full-resolution residual |
| `H2_strict_convex_scratch` | Replace final plane result with a nonnegative, sum-to-one nine-neighbor reconstruction of half-resolution tile disparity, using RGB and plane output only to predict weights |

The local subset is `/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation2000`:
1,800 training pairs from Scene01/02/06/18, plus 100 validation and 100 test
pairs from disjoint Scene20 source-frame groups (ten variations per frame).
The first 1,000 pairs are unchanged from the earlier ablation. Training uses
native 256×512 paired crops and no resizing; validation/test use padded native
images with padding removed before scoring. All arms use AdamW + OneCycle,
identical A09 multiscale/gradient/hinge loss, 25k steps, and validation every
1,000 steps. Select the lowest validation **full-image EPE** checkpoint. The
test set is evaluated only after selection. Reports include EPE, RMSE,
bad-0.5/1/2/3, D1, edge EPE, per-pair metrics and three native-resolution
Scene20 visual sheets at a fixed 0–80 px color scale.

The G post-E3 result and the H stereo-only result answer different questions;
do not compare them as if their base predictors or trainability matched.
VKITTI is an in-domain screening dataset. A pretrained ADE20K YOLO encoder
still supplies features; only the disparity side starts fresh. The semantic
teacher's historical VKITTI exposure is irrelevant because its decoder is
not used.

## Monitoring

The sequential local queue writes `experiments/H/queue/h_stereo_only_2k_25k_scratchhead_seed42_v2/status.json`
and one log per arm. It can be restarted after an interruption; the current
arm resumes from `latest.pth` at its last validation step.

These commands work from any directory; run each `tail` in its own terminal.
Future-arm logs contain a queued marker until that arm starts.

```bash
cat /home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI/experiments/H/queue/h_stereo_only_2k_25k_scratchhead_seed42_v2/status.json
tail -F /home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI/experiments/H/queue/h_stereo_only_2k_25k_scratchhead_seed42_v2/H0_a09_scratch.log
tail -F /home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI/experiments/H/queue/h_stereo_only_2k_25k_scratchhead_seed42_v2/H1_rgb_guided_scratch.log
tail -F /home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI/experiments/H/queue/h_stereo_only_2k_25k_scratchhead_seed42_v2/H2_strict_convex_scratch.log
```

The full run and checkpoint files live under `experiments/H/runs/<arm>/<run-id>/`.
