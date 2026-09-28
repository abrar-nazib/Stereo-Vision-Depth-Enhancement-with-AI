# Ablation Test 2B — frozen YOLO26s encoder + LightStereo-S head

- Fixed split: 160 train / 40 held-out validation frames from the 200-pair A02 manifest.
- Training: 15000 updates of co-located native 384×640 crops; no image or disparity resizing.
- Validation: full native 540×960 frames with only top/right stride-32 padding.
- Frozen YOLO semantic weights/buffers unchanged: True.

| Model | EPE | RMSE | bad-1 % | bad-3 % | D1 % |
|---|---:|---:|---:|---:|---:|
| Official LightStereo-S control | 1.887 | 5.894 | 25.761 | 12.545 | 10.916 |
| YOLO-adapted initial | 43.528 | 51.456 | 99.077 | 97.200 | 96.818 |
| YOLO-adapted after 15000 updates | 4.005 | 8.248 | 53.278 | 28.362 | 24.159 |

![Training curve](training_curve.png)

This is a capacity test, not a final SceneFlow generalization result: the 40 validation frames are held out from optimization, but all 200 pairs come from the Driving subset. See `config.json`, manifests, CSV traces, and `results.json` for reproducibility.
