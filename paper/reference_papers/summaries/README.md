# Paper-summary corpus

The summaries in this directory are arranged by research role, following the
structure of the curated `stero_research_claude` research library:

- `fusion/`: monocular-prior, foundation-model, and teacher-distillation stereo
  methods.
- `lightweight/`: efficient cost-volume, mobile, tile, and recurrent-pruning
  methods.
- `semantic_stereo/`: joint networks that output both semantic segmentation
  and stereo disparity; each summary labels whether its code and usable
  pretrained checkpoint are actually public.

## Evidence standard for results

For each paper, preserve the metrics exactly as reported and label a missing
metric as **not reported**. Do not convert thresholds across datasets or infer
an unreported score.

| Metric | Meaning |
| --- | --- |
| EPE | Mean absolute disparity end-point error, in pixels. |
| bad-0.5 / 1 / 2 / 3 | Percentage of valid pixels whose absolute disparity error exceeds the named pixel threshold. |
| D1-all | KITTI outlier rate: absolute error >3 px and relative error >5%, over all valid pixels. |
| D1-bg / D1-fg | The corresponding KITTI outlier rates on background / foreground pixels. |
| RMSE / MAE / median AE | Error statistics reported by some papers in addition to EPE. |
| Latency / FPS | Hardware-, resolution-, precision-, and batch-size-specific measurement; never compare directly without those conditions. |
| Parameters / FLOPs / MACs / memory | Architecture or runtime cost measures; record the paper's stated convention. |

Architecture images are stored in `../figures/`. Every source-derived image
must retain its PDF figure number and page provenance in the figure manifest.
