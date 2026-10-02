# E-series: D2 head-capacity sweep

Question: does making D2's **trainable** stereo-candidate gate and
class-conditioned residual wider improve disparity, especially bad-3, beyond
the matched extra-capacity no-semantics control? The frozen stereo predictor,
YOLO shared encoder and VKITTI semantic decoder are unchanged. D2/D4 are the
1× reference results; E does not retrain them.

| Arm | Gate hidden channels | Residual hidden channels | Semantics | Trainable head parameters |
| --- | ---: | ---: | --- | ---: |
| D2 reference | 8 | 32 | aligned left/right | 9,758 |
| D4 reference | 8 | 32 | zeroed | 9,758 |
| E1 | 16 | 64 | aligned left/right | 19,374 |
| E2 | 16 | 64 | zeroed | 19,374 |
| E3 | 32 | 128 | aligned left/right | 38,606 |
| E4 | 32 | 128 | zeroed | 38,606 |

E1–E2 and E3–E4 isolate semantic input at equal capacity. Compare E1/E3
against D2 for width sensitivity, but also compare their controls to D4:
extra capacity can improve disparity without using semantics. Gate and
residual architecture, loss, and checkpoint-selection rule are otherwise
identical to D. The shared two-view encoder is one batched call, not two sets
of encoder weights. Both semantic tails still run in every arm, so controls
have matched frozen inference work.

Protocol: the exact B/C/D VKITTI2 manifest and seed-42 800/100/100 split
(80/10/10 source-frame groups), paired 256×512 native-pixel training crops,
full-resolution padded validation/test, AdamW 2e-4, OneCycle, 10,000-step
ceiling, validation every 1,000, three-evaluation patience, and validation
bad-3 checkpoint selection. The local RTX 3050 is the only training GPU.
Each run records hashes, widths, trainable count, all disparity metrics,
segmentation mIoU, per-pair test results, status, and a best checkpoint.

Before the full queue, smoke every arm with `--steps 1 --eval-every 1
--eval-limit 1`. Smoke metrics are not research results. The queue stops at
the first failure and retains per-arm logs. Run the queue detached from the
repository root:

```bash
setsid -f uv run --no-sync python -m experiments.E.E_1_wide2_semantics.queue \
  --run-id e_capacity_v1_seed42_20261002 --steps 10000 --eval-every 1000 --patience 3 \
  > experiments/E/E_1_wide2_semantics/queue_launcher.log 2>&1 < /dev/null
```

Poll without keeping the launching terminal open:

```bash
cat experiments/E/E_1_wide2_semantics/queue/e_capacity_v1_seed42_20261002/status.json
tail -n 30 -F experiments/E/E_1_wide2_semantics/queue/e_capacity_v1_seed42_20261002/E_1_wide2_semantics.log
tail -n 30 -F experiments/E/E_1_wide2_semantics/queue/e_capacity_v1_seed42_20261002/E_2_wide2_control.log
tail -n 30 -F experiments/E/E_1_wide2_semantics/queue/e_capacity_v1_seed42_20261002/E_3_wide4_semantics.log
tail -n 30 -F experiments/E/E_1_wide2_semantics/queue/e_capacity_v1_seed42_20261002/E_4_wide4_control.log
```

The semantic teacher may have trained on the D/E depth-test RGB frames under
its random all-scene split. E is an **in-domain capacity screen**, not an
independent-test or real-domain result. Audit the teacher split before a
manuscript claim; only after reviewing E should a full VKITTI Modal run be
chosen and submitted.

## Full-VKITTI T4 follow-up

The follow-up compares E3 and E4 at 38,606 trainable parameters with D2 at
9,758. D2 is the compact *semantic* reference, **not D4**. It starts each
head at identity initialization on the same frozen A09 stereo and
VKITTI-14 semantic checkpoints; it does not warm-start from the local E/D
heads. The exact semantic teacher's seed-42 `(scene, frame)` split is reused
for depth, avoiding semantic-teacher training overlap with depth validation
or test frames. This remains a within-VKITTI experiment, not a scene-held-out
or real-world transfer test.

`full_vkitti_data.py` derives and audits the 17,000/2,120/2,140 stereo-pair
manifest. `modal_full_vkitti_t4.py::stage_full` is CPU-only and stores one
right-RGB/depth tar plus a manifest on `svde-vkitti2`. The semantic archive
already contains left RGB and indexed masks. The T4 worker extracts those
archives to container-local disk, uses native 256×512 paired training crops,
and validates/tests at full resolution with padding only. Frozen models run
once per image pair; only the gate and residual head update. OneCycle/AdamW,
validation bad-3 selection, EPE, bad-0.5/1/2/3, D1, RMSE, 14-class mIoU,
per-pair results, and optimizer/scheduler/RNG checkpoints are retained.

The long run is **always launched detached**. Real 30-step T4 probes measured
batch 8 at roughly 19 training images/s and 3.90 GiB peak allocated VRAM;
batch 4 was slower, and batches 16/24 gave no throughput gain. Use batch 8
and a 30,000-step ceiling (240,000 training-pair exposures, about 14 full
training-set passes), with validation every 2,000 steps and patience five.
Each arm has an independent run
directory and one detached Modal app. A retry resumes from `latest.pth`.
Outputs live under `svde-results:/E_full_vkitti/<run-id>/<arm>`.

Submitted detached 2026-10-02, run ID `e3_e4_d2_full_t4_v1_20261002`:

| Arm | Modal app | Function call |
| --- | --- | --- |
| E3 | `ap-Bdpl9WcywpBuY20evQveMI` | `fc-01M3YJ6YK9XYDT416Y66S25FB3` |
| E4 | `ap-ErrbVVosdOWB2jg9zRDIyT` | `fc-01M3YJ7ZPJ5RZZ9ZJCHB7261YT` |
| D2 | `ap-U0VNjHtnakvCUwp5S37PWc` | `fc-01M3YJ8R3E2C7S34YM1SVEFK36` |

The CPU staging app was `ap-7t6clPOmVlAJ04A8ADPqar`. The 30-step T4
probes are under `/E_full_vkitti/probes/` on `svde-results`; smoke metrics
are not research results. Use `modal app logs <app-id>` or inspect each
arm's `status.json` in the results Volume to monitor without a live laptop.

```bash
uv run --no-sync modal run -d experiments/E/modal_full_vkitti_t4.py::stage_full
uv run --no-sync modal run -d experiments/E/modal_full_vkitti_t4.py::probe_t4 \
  --run-id e_full_t4_probe_v1 --batch 4
# After verifying the probe:
uv run --no-sync modal run -d experiments/E/modal_full_vkitti_t4.py::launch \
  --arm E3 --run-id e3_e4_d2_full_t4_v1_20261002 --batch 8 --steps 30000 --eval-every 2000
uv run --no-sync modal run -d experiments/E/modal_full_vkitti_t4.py::launch \
  --arm E4 --run-id e3_e4_d2_full_t4_v1_20261002 --batch 8 --steps 30000 --eval-every 2000
uv run --no-sync modal run -d experiments/E/modal_full_vkitti_t4.py::launch \
  --arm D2 --run-id e3_e4_d2_full_t4_v1_20261002 --batch 8 --steps 30000 --eval-every 2000
uv run --no-sync modal app list
uv run --no-sync modal app logs <app-id> --since 1h -n 100
uv run --no-sync modal volume ls svde-results /E_full_vkitti/e3_e4_d2_full_t4_v1_20261002
```
