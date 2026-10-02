# A09 M encoder, full SceneFlow: OneCycle scheduler comparison

Run ID: `a09m_fullsf_a10_onecycle_v1_20260929`  
Modal app: `ap-mzZwpQ6CEZbIaVItN1xLYE`  
Modal call: `fc-01M3PBJM0MV7JF9NJSC32R18GX`  
Comparator: `a09m_fullsf_a10_v1_20260929` (`ReduceLROnPlateau`)

This is an independent fresh 120,000-step A10 run, not a continuation of the
plateau checkpoint. Both runs use the same A09 V-arm model, frozen
YOLO26m ADE20K encoder, seed 42, official 35,454-pair SceneFlow train split,
400 fixed FT3D-test validation pairs, batch 16, native co-located 384×640
crops without resizing, AdamW, fp16, loss, and 2,000-step validation cadence.
The scheduler is the only intended protocol difference. Both jobs read the
same immutable shard volume, but write to distinct directories under
`svde-results:/final_pass/`.

OneCycle configuration: peak LR `1e-4`, initial LR `4e-6` (`div_factor=25`),
1% warmup, cosine decay, final LR `1e-6` (`final_div_factor=4`), and no
momentum cycling. It steps after each successful optimizer update (skipped
AMP updates do not advance it). The scheduler state is saved in every
checkpoint and restored on retry. Validation does not change the LR.

Launch command used:

```bash
uv run modal run -d experiments/final_pass/modal_full_pass.py::launch_a10_onecycle \
  --run-name a09m_fullsf_a10_onecycle_v1_20260929 \
  --steps 120000 --batch 16 --eval-every 2000 --ckpt-every 1000
```

Monitor without holding a laptop connection:

```bash
uv run modal app list
uv run modal app logs ap-mzZwpQ6CEZbIaVItN1xLYE --since 1h
uv run modal volume ls svde-results /final_pass/a09m_fullsf_a10_onecycle_v1_20260929
```

The comparison should use `validation.csv` at matched steps and the final
full-test metrics, not random-crop EPE. The validation subset overlaps the
4,370-pair full test, as documented in `RUN_A09M_A10.md`.
