# C2 causal controls (local RTX 3050)

`C_4_no_semantics` and `C_5_misaligned_semantics` instantiate the *same*
369,137-parameter `FeatureFusionHead` as the completed C2 run. They use the
same frozen checkpoints, 800/100/100 frame-grouped VKITTI split, seed 42,
aligned 256×512 crop protocol, native-resolution evaluation, AdamW/OneCycle,
10,000-step ceiling, validation every 1,000 steps, and early stopping. They
are **trained from scratch**, not perturbed only after C2 training.

- C4: zero semantic logits and decoder map for the trainable head during
  training, validation, and test. Shared/stereo feature maps remain intact.
- C5: circularly shift both semantic logits and decoder map together by
  independently drawn nonzero horizontal and vertical offsets per training
  crop. The frozen model outputs and stereo/shared maps stay aligned.
  Validation/test shifts are deterministic from seed 42. This preserves
  semantic input values while breaking their pixelwise alignment.

The frozen semantic output is still reported with normal 14-class,
GT-present mIoU; neither control trains or alters the segmentation model.
The controls cannot by themselves solve possible overlap between the semantic
teacher's random split and this depth test split. Smoke runs with one-image
evaluation are **not** benchmark results.

From the repository root, the detached sequential queue command is:

```bash
nohup setsid uv run --no-sync python -m experiments.C.C_1_large_control.queue \
  --controls --run-id c_causal_v1_seed42_20261001 \
  --steps 10000 --eval-every 1000 --patience 3 \
  > experiments/C/C_4_no_semantics/queue_launcher.log 2>&1 < /dev/null &
```

Poll queue state and follow the individual arms independently:

```bash
uv run --no-sync python -m json.tool experiments/C/C_4_no_semantics/queue/c_causal_v1_seed42_20261001/status.json
tail -n 30 -F experiments/C/C_4_no_semantics/queue/c_causal_v1_seed42_20261001/C_4_no_semantics.log
tail -n 30 -F experiments/C/C_4_no_semantics/queue/c_causal_v1_seed42_20261001/C_5_misaligned_semantics.log
uv run --no-sync python -m json.tool experiments/C/C_4_no_semantics/runs/c_causal_v1_seed42_20261001/status.json
uv run --no-sync python -m json.tool experiments/C/C_5_misaligned_semantics/runs/c_causal_v1_seed42_20261001/status.json
```

Each run records its manifest, progress JSONL, per-validation metrics,
periodic and best checkpoints, test metrics, and counterfactual validation
probes. Only the validation-selected best checkpoint is tested.
