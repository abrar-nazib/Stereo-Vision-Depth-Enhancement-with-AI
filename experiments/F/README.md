# F-series: frozen real-domain KITTI 2015 transfer test

F1 runs the exact frozen A09 `FusionStereoLite("V")` SceneFlow-family stereo
checkpoint, including its trained stereo-side veto block but **without** a
semantic decoder or E3 fusion head. F2 uses the same frozen stereo checkpoint,
the frozen 14-class VKITTI semantic decoder, and E3's validation-selected
semantic gate/residual head (step 16,000). Nothing is trained or selected on
KITTI. Both arms use the same 200 public labeled KITTI 2015 training pairs.

The source archive is `stereo-datasets:/kitti/data_scene_flow.zip`. KITTI
`image_2` is left, `image_3` is right, and the 16-bit `disp_occ_0` and
`disp_noc_0` ground truths are divided by 256 to obtain disparity in pixels.
RGB is passed in 0–255 range at native resolution with replicate padding to a
multiple of 32; there is no resize. The identical evaluation mask for both
arms is finite GT disparity `>0` and `<192` px, matching the pretrained
stereo model's range. Counts excluded above the range are reported. Each run
records EPE, RMSE, bad-0.5/1/2/3, D1, per-pair metrics, runtime and peak VRAM
for all-valid and non-occluded GT separately. KITTI 2015 benchmark testing
images do not have public GT and are not scored here.

F1 versus F2 tests whether the **complete E3 model** transfers better than
the frozen stereo model on real road scenes. It does not isolate semantics
from head capacity; the separately trained E4 no-semantics control would be
required for that narrower causal claim. F3 now supplies that equal-capacity
E4 control on the exact same 200 pairs. Segmentation mIoU is not reported
because the VKITTI 14-class taxonomy is not KITTI's taxonomy.

The single detached T4 job evaluates F1, commits its results, then evaluates
F2. First run the two-pair smoke for both arms:

```bash
uv run --no-sync modal run experiments/F/modal_kitti_t4.py::probe_t4 \
  --run-id f_kitti2015_probe_v1 --limit 2
```

The smoke outputs are under `svde-results:/F_kitti2015/probes/` and must not
be quoted as research results. Launch the full job detached:

```bash
uv run --no-sync modal run -d experiments/F/modal_kitti_t4.py::launch \
  --run-id f1_f2_kitti2015_v1_20261003
```

Full results: `svde-results:/F_kitti2015/full/f1_f2_kitti2015_v1_20261003/F1`
and the sibling `F2` directory. Poll without keeping the laptop connected:

```bash
uv run --no-sync modal app list
uv run --no-sync modal app logs <app-id> --since 1h -n 100
uv run --no-sync modal volume ls svde-results /F_kitti2015/full/f1_f2_kitti2015_v1_20261003
```

The full run finished on 2026-10-03; detached app
`ap-xCpWCVAzMaCzUZzPAXvv92` is stopped. Both result/status JSON files were
copied into the two versioned local F-arm `runs/` folders. Read
[INSIGHTS.md](INSIGHTS.md) for the paired result and attribution limits.

F3 was added afterward using only the E4 head. Its two-pair smoke and full
evaluation commands are:

```bash
uv run --no-sync modal run experiments/F/modal_kitti_t4.py::probe_control_t4 \
  --run-id f3_kitti2015_probe_v1 --limit 2
uv run --no-sync modal run -d experiments/F/modal_kitti_t4.py::launch_control \
  --run-id f1_f2_kitti2015_v1_20261003
```

The detached F3 app was `ap-KVIg46HNWNDvnB74i5ZsJX`; its results live in
the sibling `F3` directory on `svde-results` and the versioned local
`F_3_kitti2015_e4_control/runs/` folder. Do not compare F2/F3 smoke scores
with the 200-pair results.
