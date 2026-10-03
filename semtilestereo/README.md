# SemTileStereo local inference tools

These tools run the released E3 semantic-guided stereo model. They use one
shared YOLO26m encoder for the stereo pair, frozen semantic and A09 predictors,
and the trained E3 fusion head. Input disparity is never resized. Left and right
images must be rectified, matched, and in the correct order.

## Checkpoints

The default paths are in `semtilestereo/core.py` (`ModelPaths`). The A09
checkpoint is staged locally at
`models/stereo/semtilestereo/a09m_sceneflow_best.pth`; it is ignored by Git.
Its SHA-256 is
`5ceaad4c84d941bd23be0d4c045f6d94934f6c17d0e6f5a9bc4cbc18f7f226b7`.
The other default checkpoints are:

| Role | Path | SHA-256 |
| --- | --- | --- |
| E3 fusion head | `experiments/E/E_3_wide4_semantics/runs/e3_e4_d2_full_t4_v1_20261002/checkpoints/best.pth` | `2493d46d17eceb51ab133b73f9b8fd54fb4ccc1b9b2d87a9a05810f15b727b87` |
| VKITTI 14-class semantic decoder | `models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt` | `bc3b9d334e3dc9d6535e8c9bea04cd2fad54b0dd36cf223c5313887c145807b9` |
| ADE20K-pretrained shared encoder | `models/segmentation/yolo26m-sem-ade20k.pt` | `099d977182f7b13fabf5d4ed254dda05e7c03d6405096572c356722e9ae8f374` |

The default A09 hash is verified at load. Other checkpoint paths can be
overridden on the CLI. If the A09 local copy is missing, retrieve the released
best checkpoint from the Modal results volume and verify its hash before using
it; model files are never auto-downloaded.

## Commands

Install from the project root with `uv sync`. This includes PySide6 and Open3D
as project dependencies. Run:

```bash
uv run python -m semtilestereo.infer pair left.jpg right.jpg outputs/pair --vkitti-camera
uv run python -m semtilestereo.infer live --camera /dev/video2 --width 1280 --height 480
uv run python -m semtilestereo.gui
```

`pair` writes `disparity.npy` (float32 pixels), `semantic_logits.npy` (raw
float32 model logits), `disparity_color.png`, `segmentation_overlay.png`, and
`result.npz`. The bundle is the GUI import format; its JSON metadata includes
schema version, image size, labels, camera parameters if supplied, source paths,
checkpoint identities, and the preview color range. Neither PNG encodes metric
depth. The disparity output is a scalar per left pixel; `(u,v)` are implicit
array indices. Invalid/nonpositive disparity is kept in the raw array and
excluded from the point cloud.

`live` expects the connected rectified rig's side-by-side V4L2 MJPEG frame,
with the left view in the first half. Use `--left-view right` if the halves are
reversed, `--rotate 180` if both views are inverted, or `--rectification MAP.xml`
only when that file matches the camera, ordering, and half-frame resolution.
The two windows are disparity and left semantic overlay. Close either window or
press `q`, Escape, or Ctrl+C to exit. Capture is released on exit. The live
mode does not save frames.

For real-camera point clouds, enter that camera's `fx`, `fy`, `cx`, `cy` in
pixels and baseline in metres in the GUI. The VKITTI preset is only for native
1242×375 VKITTI results; it must not be used as a substitute for rig
calibration. Camera-frame coordinates use `X` right, `Y` down, `Z` forward.
The GUI's class checkboxes only change the displayed cloud and do not rerun
inference. Left/right images can be selected with file pickers for asynchronous
inference. A newly inferred pair does not inherit a previously opened bundle's
calibration: select **Apply these camera parameters to next inference** explicitly
if they match the new pair, or enter calibration after loading the result.

## Reproducing examples and validation

Three ignored, ready-to-open bundles live under `examples/semtilestereo/`:
Scene01 clone frame 246, Scene06 clone frame 19, and Scene18 clone frame 259.
Their matching source JPEGs are the local files under
`paper/figures/semtilestereo/candidates/`; they are not redistributed via Git.
To regenerate with the released checkpoint:

```bash
uv run python -m semtilestereo.examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q tests/test_semtilestereo_core.py tests/test_semtilestereo_results.py tests/test_semtilestereo_cli.py tests/test_semtilestereo_geometry.py tests/test_semtilestereo_gui.py
SEMTILESTEREO_RUN_INTEGRATION=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q tests/test_semtilestereo_integration.py
```

The opt-in integration test requires the local assets and uses CUDA for the direct
inference parity check. Across independent CUDA processes, stereo aggregation
may differ slightly (observed maximum 0.048 px for Scene01); saved arrays
themselves are full-precision float32. Physical-camera capture was not
automatically exercised; run the `live` command above on the rectified rig and
visually verify left/right alignment before interpreting disparity.
