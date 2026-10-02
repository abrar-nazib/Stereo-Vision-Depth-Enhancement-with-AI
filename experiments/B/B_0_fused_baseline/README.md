# B-series fused semantic-to-disparity ablations

The A09 V-arm stereo checkpoint and the 14-class VKITTI semantic checkpoint
share exactly equal YOLO26m layers 0–6 (186 tensors/buffers checked locally).
`model.py` executes this trunk on a batch of left/right RGB frames once, then
uses the cached left intermediate features to run the semantic tail. This
replaces the legacy stereo-left/stereo-right plus separate semantic-left trunk
execution. `predict.py` supplies native-resolution disparity in pixels, class
ID, and class confidence; `(u,v)` are the output map's column/row indices.

Local RTX 3050 (4 GiB), AMP FP16, warm-up five, 20 measured repeats at padded
384×1248 resolution on 2026-10-01:

| Execution | Median | p95 | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Legacy extra-trunk path | 69.56 ms | 70.34 ms | 452.7 MiB |
| Fused, one CUDA stream | 62.28 ms | 64.00 ms | 429.4 MiB |
| Fused, two branch streams | 62.86 ms | 64.11 ms | 462.1 MiB |

The two-stream branch implementation was correct but did not improve latency
on this GPU, so the B-series runner uses the one-stream path. These are model
forward measurements, excluding image decoding, CPU transfer, and the B1–B3
correction heads. CUDA AMP fused/legacy parity is covered in `tests/test_b_fused.py`.

From the repository root, the full 800/100/100 protocol runs with:

```bash
uv run python -m experiments.B.B_0_fused_baseline.run \
  --run-id 20261001_b_full_seed42 --steps 1600 --eval-every 400
```

The launch log is `B_series_20261001.log`. Its `train` records are optimizer
steps and `val`/`test` records are native-resolution full-set evaluations.
Each arm stores its chosen `best_epe.pth`, metrics, and trace under its own
`B_[N]_[short_identifier]/runs/<run-id>/` folder. The test set is Scene20's
ten held-out frame groups (ten variations per frame); semantic pretraining may
overlap it, so this is an in-domain feasibility experiment.
