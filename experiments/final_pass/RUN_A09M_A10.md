# A09 M encoder, full SceneFlow on Modal A10

Run ID: `a09m_fullsf_a10_v1_20260929`  
Modal app: `ap-l81AlA7tyv4OssaRxbfD8r`  
Modal call: `fc-01M3P7KSXPPVQGS1M8YEPMHKXY`  
Resume point: step 1,000 from the first detached app

The architecture is `FusionStereoLite(arm="V")` with the frozen
`yolo26m-sem-ade20k.pt` encoder. The head and loss match the successful A09
Driving-1000 experiment: five-scale disparity L1, gradient consistency, and
bad-1 hinge; AdamW at 1e-4 with fp16 AMP and gradient clipping. From step
1,000 onward, a validation-EPE plateau scheduler halves the learning rate
after four evaluations without a 0.5% improvement, with one-evaluation
cooldown and a 1e-6 floor. Input is a
co-located native 384×640 crop; there is no resize. The full official split
contains 22,390 FlyingThings3D train, 8,664 Monkaa, and 4,400 Driving pairs
(35,454 total). Validation uses 400 fixed FlyingThings3D test pairs at native
960×540, with top/right pad to a multiple of 16. At completion, the best
checkpoint is evaluated over all 4,370 FlyingThings3D test pairs. Twelve fixed
validation scenes get left/GT images and raw 16-bit plus color prediction PNGs
at every evaluation (resume baseline at 1,000, then steps 2,000, 4,000, ...). The color previews use the same
0–192 px scale for all steps.

This matches the earlier StereoLite full-run validation protocol: the 400
validation pairs are a fixed seed-42 subset of FlyingThings3D **TEST**. The
4,370-pair final test includes these 400 scenes, so selecting the best
checkpoint on this subset means the full-test score is not wholly independent.
State this overlap when reporting the paper result.

At the resume baseline (step 1,000), native validation EPE was 5.7080 px.
This is an early training measurement; it does not predict the final score.
After the logging update, each 50-step progress line and `train.csv` row shows
the last measured validation EPE and its step alongside crop EPE.

The A09 local run trained on 800 Driving pairs for 35,000 batch-1 steps,
roughly 44 sample passes. Its held-out Driving-200 EPE was 3.3265 px. The
full run uses 120,000 steps at batch 16, or 1.92 million sample draws, roughly 54
passes over the full training split. At the observed 3.6–3.7 steps/s, training
alone is about nine hours; validation and full testing add time. These validation datasets differ, so
their EPEs are not directly comparable.

The A10 probe `a09m_a10_probe_20260929` completed a real-data batch sweep,
50 training steps, validation, and checkpointing. A10 reports 22.06 GiB VRAM.

| Batch | ms/step | samples/s | Peak reserved | VRAM |
|---:|---:|---:|---:|---:|
| 8 | 129 | 62.2 | 4.48 GiB | 20.3% |
| 16 | 254 | 62.9 | 8.90 GiB | 40.3% |
| 24 | 409 | 58.7 | 13.49 GiB | 61.1% |
| 32 | 555 | 57.7 | 17.94 GiB | 81.3% |
| 48 | OOM | — | — | — |

Batch 16 gave the highest measured throughput. It also leaves ample memory
for validation and worker variance. The 50-step probe EPE is an early-run
smoke test, not a benchmark result.

Launch command (already submitted with `uv`):

```bash
uv run modal run -d experiments/final_pass/modal_full_pass.py::launch_a10 \
  --run-name a09m_fullsf_a10_v1_20260929 \
  --steps 120000 --batch 16 --eval-every 2000 --ckpt-every 1000
```

Check progress without holding a client connection:

```bash
uv run modal app list
uv run modal app logs ap-l81AlA7tyv4OssaRxbfD8r --since 1h
uv run modal volume ls svde-results /final_pass/a09m_fullsf_a10_v1_20260929/checkpoints
```

The results volume stores `config.json`, `train.csv`, `validation.csv`,
`learning_rate.csv`, `latest_eval.json`, `best_eval.json`, `full_test.json`,
the `visualizations/` directory, and resumable `checkpoints/latest.pth`.
Retries use the same run ID and resume from that checkpoint. The launcher
returns as soon as Modal accepts the input; the laptop may disconnect.
