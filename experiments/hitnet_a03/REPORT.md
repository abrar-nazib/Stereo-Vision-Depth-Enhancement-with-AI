# A03 — encoder swap on the HITNet-style stereo head

## Question

Does a pretrained *semantic segmentation* encoder work as a stereo feature
encoder when it is frozen and only the matching head is trained? This follows
A02B, where a frozen YOLO26s-sem encoder on a **LightStereo** head stalled at
6.20 px EPE.

## Design

Two arms, same head, same data, same loss, same schedule. Only the encoder
differs.

| Arm | Encoder | Frozen | Encoder channels | Total / trainable |
| --- | --- | --- | --- | --- |
| `ghost` | `TileFeatureEncoder` (ghost conv, random init) | no — trained | 24 / 48 / 72 / 96 | 0.487 M / 0.487 M |
| `ade20k` | `yolo26s-sem-ade20k.pt`, layers 0-6 | **yes** — no grads, BN in eval | 32 / 128 / 256 / 256 | 1.647 M / 0.414 M |

The head is the author's own `StereoLite_v2_hitnet` (see `NOTICE.md`). It sizes
its convolutions from `fnet.out_channels`, so the encoder swaps with **no
adapter layers** — unlike A02B, where LightStereo's fixed 24/32/96/160 pyramid
forced 1x1 adapters and left an unused 160-channel branch.

Protocol, identical to A02: 200 fixed SceneFlow Driving pairs, deterministic
stratified 160 train / 40 held-out, native 384x640 co-located crops for
training, full native 540x960 evaluation with replicate padding to /16, no
resizing anywhere. 10,000 steps, batch 1, AdamW lr 2e-4, AMP fp16, seed 42.

## Results (held-out, 40 frames)

| | `ghost` | `ade20k` (frozen) |
| --- | ---: | ---: |
| Final EPE (px) | 5.686 | **5.660** |
| Best EPE (px) | 4.714 @ 8700 | **4.323 @ 9300** |
| RMSE (px) | 11.900 | **11.075** |
| Median AE (px) | — | — |
| bad-1 (%) | 52.95 | **46.03** |
| bad-3 (%) | 30.66 | **28.64** |
| D1-all (%) | 27.45 | **25.86** |
| Peak GPU | 289 MB | 314 MB |
| Training time | 29.6 min | **24.5 min** |

Held-out EPE through training (every 1,000 steps):

| step | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ghost` | 122.8 | 13.22 | 11.12 | 10.20 | 7.60 | 7.06 | 5.88 | 6.89 | 7.05 | 5.55 | 5.69 |
| `ade20k` | 129.6 | 10.44 | 6.77 | 6.40 | 6.00 | 5.70 | 6.82 | 6.08 | 5.22 | 5.26 | 5.66 |

## Reading

1. **The frozen pretrained encoder is ahead at every matched step**, by 1-4 px
   early on, and converges to the same final EPE with **fewer trainable
   parameters** (0.414 M vs 0.487 M) and **17% less wall time**.
2. This **contradicts the earlier prediction** in this project that freezing a
   pretrained encoder necessarily reproduces a ~6 px ceiling. That prediction
   was inferred from A02B, but A02B's ceiling came from the **LightStereo head
   plus its 1x1 adapter bottleneck**, not from freezing per se. On the
   HITNet-style head the same frozen-encoder idea works much better.
3. **The head dominates.** Both A03 arms beat A02B's final 6.204 px, and both
   reach a better best-EPE than A02B's 6.047. The head, not the encoder, is the
   larger lever.
4. **Error structure differs.** A02B has the *best* D1 (21.67%) but the *worst*
   EPE (6.204). The LightStereo head is more often approximately right while the
   HITNet-style head has a lower mean error but a higher fraction of gross
   outliers. Report both metrics; neither alone characterises the arms.

## Confound, stated explicitly

`ghost` and `ade20k` differ in **two** respects, not one: the encoder's weights
(from-scratch ghost vs pretrained semantic) **and** its trainability (trained vs
frozen). A third arm with the ADE20K encoder trainable would separate them; it
was considered and deliberately not run. Any conclusion here is therefore about
"frozen pretrained semantic encoder vs trained from-scratch encoder" as a
combination.

## Limitations

- **Not converged.** Both arms sit at ~5.7 px final. Reference points on the
  same data: official LightStereo-S control 1.958 px, and the author's
  full-SceneFlow StereoLite GEV4 0.79 px (60k steps, batch 32, A100, full split).
- **Single seed**, one run per arm.
- **Stripped recipe.** No augmentation, and a single-term final loss. The
  author's established recipe is materially different and is the most likely
  route to better absolute numbers:
  - loss `1.0*ms_l1(d_final) + 0.5*ms_l1(d_half) + 0.3*ms_l1(d4)
    + 0.2*ms_l1(d8) + 0.1*ms_l1(d16) + 0.5*grad_consistency + 0.2*bad1_hinge`
    (`hitnet_head.forward(..., aux=True)` already returns these scales);
  - `augment_batch`: colour jitter, right-image eraser, and random scale with
    anisotropic stretch (`D *= nW/W`);
  - batch 32, bf16, OneCycle schedule.
- The `ade20k` arm's head is also wider (encoder channels 32/128/256/256 vs
  24/48/72/96), so part of its advantage is head capacity, not the encoder.

## Reproduction

```bash
cd experiments/hitnet_a03
uv run python run.py --arm ghost  --steps 10000 --eval-every 100 --checkpoint-every 500
uv run python run.py --arm ade20k --steps 10000 --eval-every 100 --checkpoint-every 500
uv run python visualize.py --run runs/A03_ade20k_<timestamp>
```

Each run directory holds `config.json`, both manifests, `train.csv`,
`validation.csv`, `run.log`, `training_curve.png`, `results.json`, and
checkpoints every 500 steps.

## Related

- `experiments/lightstereo_s_a02/` — A02/A02B, the LightStereo-S control and the
  frozen-YOLO26s-Cityscapes adapter ablation.
- `NOTICE.md` — provenance of the ported head and the status of the parked
  HITNet-XL port in `model.py`.
