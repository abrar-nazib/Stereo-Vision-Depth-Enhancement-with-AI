# Ablation Test 2 — LightStereo-S native benchmark

## Scope

This is the unmodified official `LightStereo-S-SceneFlow.ckpt` control, tested
in FP32 on an RTX 3050 Laptop GPU. It is an inference benchmark: no model
parameters were trained or changed.

- Dataset: SceneFlow Driving, 200 unique 540×960 stereo pairs.
- Sampling: 25 evenly spaced frames from each of the eight available Driving
  sequences. These are 200 distinct frames, not 200 independent sequences.
- Input protocol: native full frames; top/right replicate padding to 544×960
  for the required stride-32 input; padding removed before scoring.
- Valid-pixel mask: finite ground truth with `0 < disparity < 192` pixels.
- Checkpoint SHA-256:
  `e34b6aaa646217d03abcaa4f438dd02404711fe2bcee10eb59a6aff71f2ca02f`.

## Results

| EPE px | RMSE px | bad-0.5 % | bad-1 % | bad-2 % | bad-3 % | D1 % |
|---:|---:|---:|---:|---:|---:|---:|
| 1.984 | 5.909 | 38.773 | 26.827 | 17.872 | 13.354 | 11.569 |

| Parameters | Mean latency | Median latency | P95 latency | Throughput | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|
| 3,441,360 | 60.63 ms | 59.09 ms | 68.12 ms | 16.49 FPS | 149.88 MiB |

The score is a macro-average of the per-image values in `per_image.csv`.
It is not directly comparable to the paper's 256×512 SceneFlow benchmark,
because this study uses native 540×960 Driving frames and a different fixed
subset. `config.json`, `manifest.json`, `per_image.csv`, and `results.json`
are the reproducibility artifacts.
