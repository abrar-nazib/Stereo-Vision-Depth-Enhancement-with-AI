# Segmentation model inventory

Downloaded 2026-09-27. SHA-256 values below were calculated locally after the
download; the byte sizes are the local file sizes.

> **Terminology:** Ultralytics `-seg` weights are **instance-segmentation**
> models. Ultralytics reserves `-sem` (for example, `yolo26n-sem.pt`) for
> semantic segmentation. The two `-seg` files below are the explicitly
> requested YOLO26 artifacts; no similarly named model was substituted.

| File | Source and version | Size (bytes) | SHA-256 | License stated by source |
| --- | --- | ---: | --- | --- |
| `yolo26n-seg.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt), release published 2026-01-13 | 6,719,965 | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |
| `yolo26s-seg.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt), release published 2026-01-13 | 23,467,933 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |
| `FastSAM-s.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt), release published 2026-01-13; upstream [FastSAM v0.0.2](https://github.com/CASIA-LMC-Lab/FastSAM/releases/tag/v0.0.2) | 23,851,578 | `c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844` | [AGPL-3.0](https://github.com/CASIA-LMC-Lab/FastSAM/blob/main/LICENSE) |
| `mobile_sam.pt` | [MobileSAM checkpoint](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt), upstream `master` resolved to [`f706ad9`](https://github.com/ChaoningZhang/MobileSAM/commit/f706ad9c4eb7f219c00d9050e46328518ffb65d2) (2026-05-05); no checkpoint release tag is stated | 40,728,226 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` | [Apache-2.0](https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE) |
| `yolo26n-sem-cityscapes.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt); semantic model pretrained on Cityscapes (19 classes) | 3,487,283 | `f3f293cca764de1f93044030d8d5612de9c5ffbf37c9c8ea1b69418b73038999` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |
| `yolo26s-sem-cityscapes.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem.pt); semantic model pretrained on Cityscapes (19 classes) | 13,252,403 | `bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |
| `yolo26n-sem-ade20k.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem-ade20k.pt); semantic model pretrained on ADE20K (150 classes) | 3,497,863 | `68ac71bef2868c987fff7cd7c49cb656922f6d8447ad21af3893305577435000` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |
| `yolo26s-sem-ade20k.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem-ade20k.pt); semantic model pretrained on ADE20K (150 classes) | 13,291,143 | `feb2bd47eed6f721a4588213c670ccdad69c0ba29b4b5f97892df222b18c6c3a` | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or [Enterprise](https://www.ultralytics.com/license) |

## Loading

Install the loader packages in the active project environment, then use paths
relative to this directory (or replace them with absolute paths).

```python
# YOLO26 instance segmentation
from ultralytics import YOLO
model = YOLO("models/segmentation/yolo26n-seg.pt")  # or yolo26s-seg.pt
results = model("image.jpg")

# YOLO26 semantic segmentation -- use the checkpoint matching the dataset.
cityscapes = YOLO("models/segmentation/yolo26n-sem-cityscapes.pt")  # 19 classes
ade20k = YOLO("models/segmentation/yolo26n-sem-ade20k.pt")          # 150 classes
result = cityscapes("image.jpg")
class_map = result[0].semantic_mask.data

# FastSAM instance segmentation
from ultralytics import FastSAM
fastsam = FastSAM("models/segmentation/FastSAM-s.pt")
results = fastsam("image.jpg", device="cpu", retina_masks=True, imgsz=1024)

# MobileSAM promptable segmentation (clone/install its upstream package first)
from mobile_sam import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_t"](checkpoint="models/segmentation/mobile_sam.pt")
predictor = SamPredictor(sam)
```

References: [Ultralytics YOLO26 segmentation documentation](https://docs.ultralytics.com/tasks/segment/), [Ultralytics FastSAM documentation](https://docs.ultralytics.com/models/fast-sam/), and the [MobileSAM README](https://github.com/ChaoningZhang/MobileSAM#mobilesam).
