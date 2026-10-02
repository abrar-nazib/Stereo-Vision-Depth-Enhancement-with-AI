# Full VKITTI2 semantic run on Modal A10

Status (2026-09-30): seeded 14-class preparation and the one-epoch A10 probe
finished. The batch-4 full run was stopped at the user's request on
2026-09-30 17:14 +06:00; after the A10 batch-size sweep, the user explicitly
requested its continuation at batch 32. A new detached app resumed from
`last.pt` at epoch 2/60. The first completed epoch used batch 4; subsequent
epochs used batch 32. At the user's request, training was stopped during
epoch 15 on 2026-09-30 23:29 +06:00, after validation mIoU had failed to
beat the epoch-9 best for five completed epochs. A separate test-set run of
`best.pt` completed; see below. This is **not** a 60-epoch completed run.

## Purpose and split

Fine-tune `models/segmentation/yolo26m-sem-ade20k.pt` for dense VKITTI
class labels while preserving the exact YOLO layers 0–6 used by the frozen
A09 stereo model. This is semantic, not instance, segmentation. No stereo
weights are updated in this run.

All 21,260 left-camera RGB/mask pairs from the five VKITTI scenes and ten
variations enter a seeded 80/10/10 random split (seed 42). The grouping key
is `(scene, frame)` so weather/lighting/camera variations of the same source
frame remain in one partition. `split_manifest.json` records the exact groups;
`audit.json` records counts and all fourteen class pixel supports. `Undefined`
is ignored as 255; **GuardRail stays a supervised class**. This split measures
semantic fitting for the fusion experiment, not scene-disjoint generalization.

The official VKITTI2 release does not prescribe a semantic train/test split.
Ultralytics' Scene20 validation convention is its own depth-training recipe,
not a required split for this experiment.

The initial 14-class scene-split archive remains on `svde-vkitti2` at
`/semantic/v1/dataset.tar` and is not modified. A CPU function extracts it
to ephemeral local disk, reshuffles paired images/masks, audits class support,
then saves the new tar under `/semantic/v3_random14_seed42/`. The GPU function
extracts that tar onto container-local disk rather than creating tens of
thousands of Volume inodes.

## Training configuration

- Modal A10, 12 CPU cores, 36 GiB RAM, Python 3.12 and project `uv.lock`
  through `modal.Image.uv_sync`.
- YOLO26m-sem ADE20K source; layers 0–6 frozen (`freeze=7`). Parameters and
  BatchNorm buffers are checked bit-exact against the source after each epoch
  and in the selected checkpoint. Later semantic layers remain trainable.
- `imgsz=1248`, rectangular batches, batch 4 for epoch 1 and 32 thereafter, AdamW
  `lr0=2e-4`, cosine schedule ending at `1e-5`, 60-epoch ceiling, patience 12,
  AMP, seed 42; no mosaic, mixup, scale, translation or vertical flip.
  Semantic images are resized by Ultralytics; **no disparity data are used
  or resized**.
- Validation each epoch; `best.pt`, `last.pt`, every-five-epoch checkpoint,
  plots, `results.csv`, metadata and model hashes on `svde-results`.
  A Modal retry resumes from `last.pt` after verifying the frozen trunk.

## Commands and run IDs

From the repository root:

```bash
uv run modal run -d experiments/vkitti2/modal_semantic.py::prepare_random_14
uv run modal volume ls svde-vkitti2 /semantic/v3_random14_seed42
uv run modal run -d experiments/vkitti2/modal_semantic.py::probe_a10 --batch 4
uv run modal run -d experiments/vkitti2/modal_semantic.py::launch_a10 \
  --run-name vkitti2sem_random14_a10_v1_20260930 --batch 4
```

Preparation app: `ap-eDoMQXFgH1kkvgYUFYJcm0`; probe app:
`ap-HrrHwBaydUQ4DZCTDVrFrP`; full-run app:
`ap-8dFfgvm6JedFxVkTfh66ow`; detached function call:
`fc-01M3RYCJK3N3E1YS3J2KEC07EG`. The launch entrypoint uses `spawn`, and
`-d` keeps its app alive after laptop disconnect. Reusing a run
name resumes its `last.pt` rather than overwriting it; a run with
`final_test.json` is returned as complete.

The first app stopped after saving `last.pt`, `best.pt`, and `epoch0.pt`.
Those files were preserved; the later user instruction explicitly authorized
resuming `last.pt` with batch 32. The resumed app is
`ap-ww7zhVzclsLmvCL13Wk9TT`, function call
`fc-01M3S1D3Z0Y6JRHM5Y1YPK6QB1`. Its log states
`Resuming ... from epoch 2 to 60 total epochs`, `batch=32`, `freeze=7`, and
`Using 17000 train, 2120 val images`; the epoch 2 progress rows show actual
32-image training batches. That app was detached while running and is now
stopped.

## Stopped-run best checkpoint and separate test

The best validation mIoU was **0.83168 at epoch 9**. Completed epochs 10–14
did not exceed it; the run stopped during epoch 15, so that incomplete epoch
has no validation result or checkpoint. The saved `best.pt` was evaluated in a
separate detached A10 app, `ap-KrMAFGhlGxG84bU3zLEesg` (function call
`fc-01M3SP170C0DNRTYYP3KRGY6WG`). This evaluation did not train.

On all 2,140 held-out random-split VKITTI test images, it achieved **14-class
mIoU 0.82664** and **pixel accuracy 0.95608**. The source-versus-checkpoint
comparison verified frozen shared layers 0–6 bit-exact. Full per-class IoU,
pixel support, checkpoint SHA-256, and the verification flag are stored at
`svde-results:/vkitti2_semantic/vkitti2sem_random14_a10_v1_20260930/final_test.json`.
The lowest IoUs were Pole 0.627, Misc 0.636, and TrafficLight 0.655; Road
was 0.972. This random-split test is not an independent-scene or real-world
generalization score.

Reproduce only the test evaluation (no training):

```bash
uv run modal run -d experiments/vkitti2/modal_semantic.py::launch_best_evaluation \
  --run-name vkitti2sem_random14_a10_v1_20260930 --batch 32
```

## A10 batch-size benchmark

`batch_benchmark.py` measured 40 steady-state training steps after eight
warm-up steps for each batch, starting fresh from the same source checkpoint.
It used the same 1248-pixel training recipe, a 25% training subset, no
validation, and no saved weights. Each result was committed to
`svde-results:/vkitti2_semantic/batch_benchmark_a10_20260930/benchmark.json`.
The detached sweep app was `ap-51LVP0biMklrEx0fL6Pz5j`.

| Batch | Images/s | Peak allocated GiB | Outcome |
| ---: | ---: | ---: | --- |
| 4 | 11.38 | 2.02 | Pass |
| 8 | 11.69 | 3.69 | Pass |
| 16 | 11.87 | 7.06 | Pass |
| 32 | 11.98 | 13.85 | Fastest passing batch |
| 64 | — | — | OOM; Ultralytics auto-shrank it to 32 |

Batch 32 improved measured training throughput by only 5.3% over batch 4.
The projected training-step time for 17,000 images × 60 epochs is about
24.9 hours at batch 4 versus 23.7 hours at batch 32, **excluding validation,
startup, checkpoint I/O, and final test**. The earlier 10–25% wall-clock
speedup estimate was too optimistic. This probe does not establish
convergence or mIoU equivalence across batch sizes, so full-run selection
still needs an explicit quality/time trade-off.

Reproduce the sweep, without launching a full run:

```bash
uv run modal run -d experiments/vkitti2/modal_semantic.py::launch_batch_benchmark
```

The prepared audit found **17,000 train / 2,120 val / 2,140 test** images,
representing **1,700 / 212 / 214** `(scene, frame)` groups. Every split
contains all fourteen classes. The A10 probe checkpoint is under
`/vkitti2_semantic/a10_probe_random14_v1/weights/`; its metadata reports
`trunk_verified: true` after the one-epoch run.

```bash
uv run modal app list
uv run modal app logs <app-id> --since 1h -n 100
uv run modal volume ls svde-results /vkitti2_semantic/<run-name>
uv run modal volume get svde-results /vkitti2_semantic/<run-name>/weights/best.pt \
  /path/on/local/disk/best.pt
```

The random VKITTI test score is a diagnostic for semantic quality, **not**
evidence that disparity generalizes. For the paper, hold the stereo baseline,
fusion architecture and depth evaluation set fixed; compare old and improved
semantics on an independent domain. Report class support, per-class IoU, mIoU,
stereo EPE/error metrics, frozen-trunk hashes, latency and the depth-fusion
controls separately.
