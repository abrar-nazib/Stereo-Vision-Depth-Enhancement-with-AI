# Fast-FoundationStereo checkpoints

These are serialized PyTorch checkpoints from the official
[NVIDIA Fast-FoundationStereo release](https://github.com/NVlabs/Fast-FoundationStereo).
The upstream source needed to deserialize them is intentionally kept in the
ignored `external_models/Fast-FoundationStereo/` checkout.

| Preset | Checkpoint | SHA-256 |
| --- | --- | --- |
| `fastfs-23-36-37` | `23-36-37/model_best_bp2_serialize.pth` | `af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990` |
| `fastfs-20-26-39` | `20-26-39/model_best_bp2_serialize.pth` | `d147587849b3482dd2921eef60dc2cd9e12690735612e91ee2782a92f8d1c59e` |
| `fastfs-20-30-48` | `20-30-48/model_best_bp2_serialize.pth` | `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` |
| `fastfs-15-44-51` | `15-44-51/model_best_bp2_serialize.pth` | `7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214` |

`23-36-37`, `20-26-39`, and `20-30-48` are the three recommended variants
from the release folder. `15-44-51` was also published there and is retained
as an optional preset.

The live tool uses the left image from a horizontally rectified side-by-side
pair and renders FastFS disparity beside the semantic overlay. Start with the
balanced `20-30-48` preset:

```bash
uv run python src/live_semantic_segmentation.py \
  --model cityscapes-n --stereo-model fastfs-20-30-48 --fastfs-iters 4
```

FastFS predicts disparity, not metric depth. Use a matching stereo calibration
and verify the left/right order before interpreting it; `--swap-stereo` is an
explicit opt-in when the camera's packing order is reversed.
