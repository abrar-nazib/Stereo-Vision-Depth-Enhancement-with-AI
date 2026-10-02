# A-series checkpoint pruning

On 2026-10-02, the LightStereo **experiment-run** checkpoint files below
were permanently removed at the user's request to reclaim disk space.
Only `.pth`, `.pt`, `.ckpt`, and `.safetensors` files inside these exact run
directories were removed. Run reports, logs, manifests, metrics, plots, and
source code remain. These checkpoint files are not recoverable from this
workspace unless separately backed up or retrained.

| Run directory under `experiments/` | Files removed | Bytes removed |
| --- | ---: | ---: |
| `lightstereo_s_a02/runs/A02B_yolo26s_frozen_native_20260927T133921Z` | 12 | 557,341,387 |
| `lightstereo_s_a02/runs/A02B_yolo26s_frozen_native_20260927T154418Z` | 28 | 1,300,609,813 |
| `lightstereo_s_a02/runs/A02B_yolo26s_frozen_native_20260927T160319Z` | 102 | 4,737,907,807 |
| `lightstereo_m_a04/runs/A04a_lightstereo_m_ade20k_20260927T213605Z` | 32 | 3,004,194,124 |
| `a05_clean/runs/A05b_lightstereo_s_ade20k_clean_20260928T070120Z` | 32 | 1,490,675,635 |
| **Total** | **206** | **11,090,728,766 bytes (10.33 GiB)** |

The standalone downloaded reference weight
`models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt` was **not** removed.
It is separate from the ablation-run checkpoints. The boundary for pruning
other early, non-LightStereo A-series run checkpoints is still being
confirmed; A06/A07 and A09 tile-style results remain intact.
