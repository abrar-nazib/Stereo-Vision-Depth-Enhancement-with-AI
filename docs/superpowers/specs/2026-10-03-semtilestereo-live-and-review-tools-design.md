# SemTileStereo inference CLI and point-cloud review app

Date: 2026-10-03. Status: design for user review.

## Intent and scope

Deliver two local tools for the released E3 SemTileStereo model: (1) a CLI for
live stereo-camera preview and saved-image-pair inference, and (2) a Qt6 desktop
app that runs the same inference or loads a saved result, reconstructs a metric
point cloud with supplied camera parameters, and filters visible points by
semantic class. The tools share preprocessing, output decoding, colors, class
names, and the saved-result schema. They do not train or modify checkpoints.

The current camera provides a rectified side-by-side frame. Rectification is
optional; no historical calibration XML is applied by default. VKITTI examples
use known camera parameters, but these must never silently become the real
camera's calibration.

## Selected architecture and alternatives

- **Selected:** PySide6 Widgets plus Open3D's offscreen renderer displayed in a
  Qt image viewport. Mouse drag/pan/wheel update the Open3D camera and rerender
  in the embedded area. This meets the single-window requirement and avoids
  reparenting a native Open3D window under Wayland.
- Native-window embedding via `QWidget.createWindowContainer` offers Open3D's
  own interaction, but depends on external window handles and is fragile on
  Wayland; not selected.
- An Open3D WebRTC visualizer embedded in Qt WebEngine adds a server, WebRTC,
  and a larger UI dependency; not selected for a local review tool.

The Qt viewport is an Open3D-rendered, mouse-interactive view, not a separate
Open3D window or a custom point renderer. Point generation and filtering use
Open3D geometry. Inference runs outside the GUI thread so the controls remain
responsive.

## Shared inference core

Load the full-SceneFlow A09 stereo checkpoint, the frozen YOLO26m ADE20K
encoder, the VKITTI2 14-class semantic checkpoint, and the full-VKITTI E3
fusion-head checkpoint. The local E3 head is
`experiments/E/E_3_wide4_semantics/runs/e3_e4_d2_full_t4_v1_20261002/checkpoints/best.pth`.
Copy the existing A09 best checkpoint from its current temporary download to a
stable, documented local model path; verify its known SHA-256 before use.
Expose checkpoint-path overrides, but refuse missing or mismatched components
with an actionable error. The core uses the existing `EModel` forward path,
which batches the two views through the shared encoder once.

Decode each rectified RGB pair in native pixels: equal shapes and left/right
order are required; pad both views on the right/bottom to a multiple of 32,
infer, then crop the outputs to the original dimensions. Never resize the
stereo input or disparity. Return full-resolution float32 disparity in pixels,
the model's native-resolution float32 left semantic logits, a full-resolution
14-class `uint8` argmax map, full-resolution class confidence, and the original
left RGB image. Upsampling logits for the class map is display/output decoding;
the E3 head does not refine semantic logits. CUDA is the default when present;
CPU can be selected explicitly, with a speed warning.

The canonical labels and semantic colors come from
`experiments/vkitti2/semantic_data.py`. Invalid/nonpositive disparity stays in
the raw array but is excluded from metric point-cloud generation; color PNGs
use an explicit, recorded display range and mark invalid pixels distinctly.

## CLI

Proposed entry point: `uv run python -m semtilestereo.infer` with subcommands:

- `live`: side-by-side V4L2 source, default `/dev/video2` at `1280x480`,
  configurable camera, half ordering, rotation, output display range, and
  optional rectification maps. Show **two** OpenCV windows: color-coded
  disparity and semantic segmentation overlaid on the left image. Stop on
  either window's close button, `q`, Escape, or Ctrl+C; always release capture
  and close windows. Do not save frames unless explicitly requested.
- `pair`: take left and right image paths and an output directory. Generate
  the versioned numeric bundle and PNGs below. Accept an optional camera
  preset or explicit camera parameters for metadata. Optional rectification
  is applied before inference when requested; do not assume an arbitrary
  calibration matches the input.

The CLI and GUI call the same inference and saving functions; they must not
have separate RGB/BGR, padding, palette, or class-decoding implementations.

## Saved-result contract

For each `pair` run, save:

- `disparity.npy`: raw full-resolution float32 disparity in pixels.
- `semantic_logits.npy`: raw model left semantic logits, float32 at their
  returned resolution. No interpretation as an instance mask.
- `disparity_color.png`: reproducible color visualization (not metric data).
- `segmentation_overlay.png`: left RGB blended with the decoded 14-class map.
- `result.npz`: portable, compressed single-file import for the GUI. Keys:
  `disparity_px`, `semantic_logits`, `class_id`, `class_confidence`, `left_rgb`,
  and `metadata_json`. The JSON includes schema version, original image size,
  class names, disparity display range, camera parameters if known,
  rectification provenance, and checkpoint identities. Load with
  `allow_pickle=False`, validate dtypes/shapes, and reject corrupt bundles.

Pixel coordinates are implicit array indices `(u,v)`; do not store redundant
coordinate planes. This preserves the exact float outputs and keeps the GUI's
file-open interaction to one file. The `.npy` files satisfy direct numerical
inspection; the `.npz` is the recommended interchange format.

## Qt6 review app

Proposed entry point: `uv run python -m semtilestereo.gui`.

The main window has an embedded Open3D viewport, a result selector with
`Open result…`, left/right image selectors plus an `Infer pair` button, a camera
panel, class filter checkboxes, and a status/provenance area. Inference loads
once and runs in a worker; UI changes do not reload the model. Errors appear
in the window without terminating the app.

Camera panel fields: `fx`, `fy`, `cx`, `cy` (pixels), baseline (metres), depth
range, and display sampling stride. Presets include `VKITTI2 Camera 0`:
`fx=fy=725.0087`, `cx=620.5`, `cy=187.0`, baseline `0.532725 m`, for the
native `1242×375` images. The preset is selectable for a loaded VKITTI result;
it is not applied to other image sizes without a warning. Custom fields remain
editable. An imported bundle's camera metadata populates the fields when
present. Changing camera parameters rebuilds the cloud.

The point cloud uses only valid `d>0` finite pixels within the selected depth
range, with `Z=fx·baseline/d`, `X=(u−cx)Z/fx`, `Y=(v−cy)Z/fy` in metres in the
left camera frame (`X` right, `Y` down, `Z` forward). Original left RGB colors
are the default; a semantic-color toggle is available. The class panel lists
only class IDs present in the result, with per-class valid-point counts.
Unchecking a class removes its points from the Open3D geometry; rechecking
restores them without rerunning the network. Sampling affects viewport density
only, not the saved numeric arrays. Mouse orbit, pan, zoom, and reset-view work
inside the Qt window.

Include at least three ready-to-open VKITTI bundles from different scenes,
generated with the released E3 model and matching left/right pairs already
available locally (for example Scene01, Scene06, Scene18). Show them in an
Examples menu/list and preset the corresponding VKITTI camera fields. Keep
raw dataset images and generated bundles out of routine Git commits unless
small fixtures are intentionally selected; document how to regenerate examples.

## Dependencies, placement, validation

Use the repository-wide `uv` project, `uv add`/`uv sync`, and wheels; never
`uv pip` or source builds. Add PySide6 and Open3D only after checking a
compatible CPython 3.12 Linux wheel. The local `open3d/` research directory
currently resolves as a namespace import when the distribution is absent;
verify that installed Open3D resolves to the actual package before GUI work.
Put the new package under `semtilestereo/`, tests under `tests/`, and generated
examples under an ignored `examples/semtilestereo/` output directory. Update
`AGENTS.md` with the new folder's role, not a file inventory.

Test geometry and bundle round-trips without a GPU or display. Test CLI
argument errors, left/right shape checks, zero/negative disparity filtering,
class checkbox filtering, and rectification-map shape mismatch. Run a one-pair
GPU inference smoke on the local RTX 3050 and verify saved arrays/PNGs against
the same model's direct output. Smoke the PySide6+Open3D viewport on the local
desktop, including a sample bundle and filter toggles. Camera capture remains
an operator-run check unless the connected rig is available and authorized;
provide its exact command. Do not claim a metric real-camera cloud until its
calibration fields are supplied and verified.

## Acceptance criteria

1. Both CLI modes use the released E3 model and produce correctly sized,
   aligned left-view disparity and semantic outputs without resizing.
2. Live preview exits cleanly by either close button, `q`, Escape, or Ctrl+C.
3. `pair` writes both raw `.npy` outputs, both requested PNGs, and a bundle
   that the GUI opens without asking for sibling files.
4. The GUI's Open3D viewport is inside Qt, and camera edits or class checkbox
   changes visibly update the cloud without rerunning inference.
5. Three VKITTI examples open with the correct preset and visibly different
   class sets/clouds. Missing checkpoints, bad bundles, or incompatible
   calibration produce actionable errors rather than silent incorrect output.
