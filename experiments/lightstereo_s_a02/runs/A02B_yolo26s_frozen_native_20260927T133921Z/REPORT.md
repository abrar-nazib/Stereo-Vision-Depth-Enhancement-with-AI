# Ablation Test 2B — frozen YOLO26s encoder + LightStereo-S head

- Fixed split: 160 train / 40 held-out validation frames from the 200-pair A02 manifest.
- Training: 1000 updates of co-located native 384×640 crops; no image or disparity resizing.
- Validation: full native 540×960 frames with only top/right stride-32 padding.
- Frozen YOLO semantic weights/buffers unchanged: True.

| Model | EPE | RMSE | bad-1 % | bad-3 % | D1 % |
|---|---:|---:|---:|---:|---:|
| Official LightStereo-S control | 1.958 | 5.983 | 27.298 | 13.096 | 11.069 |
| YOLO-adapted initial | 45.367 | 52.697 | 98.856 | 96.499 | 95.573 |
| YOLO-adapted after 1000 updates | 8.218 | 14.223 | 57.385 | 32.232 | 27.990 |

![Training curve](training_curve.png)

This is a capacity test, not a final SceneFlow generalization result: the 40 validation frames are held out from optimization, but all 200 pairs come from the Driving subset. See `config.json`, manifests, CSV traces, and `results.json` for reproducibility.
