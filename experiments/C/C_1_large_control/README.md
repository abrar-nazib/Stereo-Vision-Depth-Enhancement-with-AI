# C-series: larger frozen-predictor disparity heads

All three arms use the B-series 1,000-pair VKITTI subset and identical
800/100/100 Scene20 frame-grouped split (seed 42). The A09 stereo checkpoint,
YOLO26m VKITTI semantic checkpoint, and shared YOLO layers 0–6 are frozen.
Only the new head is optimized. No image/disparity resizing is used; training
uses aligned 256×512 random crops and evaluation uses native images padded
to a multiple of 32. C-series mIoU averages **GT-present classes**, matching
Ultralytics; the prior B-series report included predicted-only absent classes.

- `C_1_large_control`: wider output-only residual; tests parameter capacity.
- `C_2_feature_fusion`: frozen 1/8 and 1/4 left/stereo/semantic features with
  a learned semantic gate and 1/4-resolution correction.
- `C_3_semantic_match`: C2 plus disparity-warped right 1/8 feature mismatch,
  gated by left semantic probabilities. This is a *late guidance test*, not
  SGNet's pre-aggregation cost-volume confidence module.

The single detached queue executes arms sequentially to avoid RTX 3050 VRAM
contention. Each arm has `runs/<run-id>/history.jsonl`, `status.json`,
`manifest.json`, periodic `validation_step_*.json`, `checkpoints/step_*.pth`,
`checkpoints/best.pth`, `test.json`, and `controls.json`. The queue writes
`C_1_large_control/queue/<run-id>/status.json` and one text log per arm.
The max budget is 10,000 steps, validation every 1,000 steps, with early
stopping after three consecutive validation checks without at least 0.002 px
EPE improvement (after step 3,000). The selected checkpoint minimizes native
Scene20 validation EPE. The test split is evaluated only after selection.

Start from repository root:

```bash
nohup setsid uv run --no-sync python -m experiments.C.C_1_large_control.queue \
  --run-id c_v1_seed42_20261001 --steps 10000 --eval-every 1000 --patience 3 \
  > experiments/C/C_1_large_control/queue_launcher.log 2>&1 < /dev/null &
```

Monitor without connecting to the process:

```bash
tail -f experiments/C/C_1_large_control/queue/c_v1_seed42_20261001/C_1_large_control.log
tail -f experiments/C/C_1_large_control/queue/c_v1_seed42_20261001/C_2_feature_fusion.log
tail -f experiments/C/C_1_large_control/queue/c_v1_seed42_20261001/C_3_semantic_match.log
uv run --no-sync python -m json.tool experiments/C/C_1_large_control/queue/c_v1_seed42_20261001/status.json
uv run --no-sync python -m json.tool experiments/C/C_2_feature_fusion/runs/c_v1_seed42_20261001/status.json
```

Do not use a smoke run's one-image mIoU/EPE as a benchmark. The C-series
does not train the semantic head, so mIoU must remain unchanged across arms.
