# D-series: SGNet-inspired semantic guidance at stereo matching time

This is an adaptation of SGNet's semantic/disparity candidate-confidence and
category-conditioned residual mechanisms to the frozen A09 lightweight stereo
predictor. It is **not** an SGNet reproduction. The pretrained YOLO26m shared
encoder, stereo weights, and 14-class VKITTI semantic decoder are frozen.
Only the new gate/refiner train. The encoder runs once on a batch containing
both views. The semantic tail runs on each view using those already-computed
features; this adds one right semantic-tail pass, not a third encoder pass.

| Arm | Candidate gate | Native-resolution residual | Semantic input |
| --- | --- | --- | --- |
| `D_1_semantic_cost` | yes | no | aligned L/R class probabilities |
| `D_2_semantic_cost_residual` | yes | yes | aligned L/R class probabilities |
| `D_3_no_semantic_cost` | yes | no | zeros, same gate size as D1 |
| `D_4_no_semantic_residual` | yes | yes | zeros, same size as D2 |

The gate compares left class probabilities at `(x,y)` with right probabilities
at `(x-d,y)` for every 1/16-scale candidate. It combines that agreement with
the frozen groupwise stereo correlation before the pretrained 3-D aggregator's
final regression. Its initial weights make it an exact identity transform.
The residual arm uses class-wise depthwise convolutions on disparity masked
by class probabilities. Zero-semantic arms still run both tails and retain all
trainable parameters, so their compute and capacity match their semantic arm.

Protocol: B/C 800/100/100 Scene20 frame-grouped split, seed 42; native
full-resolution validation/test with replicate padding; paired 256x512
training crops **without resizing**. OneCycleLR, max 10,000 steps, full
validation every 1,000 steps, three-evaluation patience after 3,000 steps.
All arms use the same smooth-L1 plus 0.2 × >3-pixel hinge loss. The best
checkpoint is chosen by validation bad-3 (minimum 0.01 percentage-point
improvement). Report EPE, RMSE, bad-0.5/1/2/3, D1 and semantic mIoU together;
the smoke runs are not research results. Checkpoints and traces are versioned.

The D1–D3 and D2–D4 comparisons test whether **semantic information**, not
just added trainable capacity, contributes. D1–D2 tests the added refiner,
but its extra parameters mean that contrast alone is not causal evidence.
These are in-domain VKITTI tests, not real-world generalization tests.

Full queue, if not already launched:

```bash
setsid -f uv run --no-sync python -m experiments.D.D_1_semantic_cost.queue \
  --run-id d_v1_seed42_20261002 --steps 10000 --eval-every 1000 --patience 3 \
  > experiments/D/D_1_semantic_cost/queue_launcher.log 2>&1 < /dev/null
```

Polling (run from the repository root):

```bash
cat experiments/D/D_1_semantic_cost/queue/d_v1_seed42_20261002/status.json
tail -n 20 -F experiments/D/D_1_semantic_cost/queue/d_v1_seed42_20261002/D_1_semantic_cost.log
tail -n 20 -F experiments/D/D_1_semantic_cost/queue/d_v1_seed42_20261002/D_2_semantic_cost_residual.log
tail -n 20 -F experiments/D/D_1_semantic_cost/queue/d_v1_seed42_20261002/D_3_no_semantic_cost.log
tail -n 20 -F experiments/D/D_1_semantic_cost/queue/d_v1_seed42_20261002/D_4_no_semantic_residual.log
```

Each arm also writes `runs/d_v1_seed42_20261002/status.json`,
`history.jsonl`, `validation_step_*.json`, checkpoint files, and `test.json`.
The queue stops on the first failure and preserves the failing arm's log.
