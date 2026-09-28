# Overfit ablation: v2_hitnet

**Run:** 2026-09-28T16:56:33 → 2026-09-28T17:16:09  (1156.0 s)
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU

## Configuration

- arch: `v2_hitnet`
- steps: 3000, lr: 0.0002, batch: 4
- input: 384×640, 20 fixed pairs (seed 42)
- model: 0.487 M total, 0.487 M trainable
- encoder out_channels (1/2, 1/4, 1/8, 1/16): [24, 48, 72, 96]
- peak GPU memory: 1.879 GB

## Result (full-res, all 20 pairs, eval mode)

| Metric | Value |
|---|---|
| EPE (px) | **0.7700** |
| RMSE (px) | 1.7916 |
| Median AE (px) | 0.2974 |
| bad-0.5 (%) | 33.49 |
| bad-1.0 (%) | **17.90** |
| bad-2.0 (%) | 8.44 |
| bad-3.0 (%) | 5.04 |
| D1-all (%) | 5.04 |

## Inference latency

- mean: **32.989 ms** (30.31 FPS)
- median: 32.467 ms
- p95: 38.69 ms
