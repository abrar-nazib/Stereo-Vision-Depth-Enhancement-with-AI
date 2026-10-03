# SemTileStereo Inference and Review Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a shared E3 inference core, a two-mode CLI, and a Qt6/Open3D review app with metric, class-filterable point clouds and VKITTI examples.

**Architecture:** A small `semtilestereo/` package owns model loading, native-resolution preprocessing, result decoding, save/load, visualization, and geometry. The CLI and Qt app import those same functions. The Qt viewport displays Open3D offscreen renders and maps mouse interaction to Open3D camera updates; it never opens a second 3D window.

**Tech Stack:** Python 3.12, project-wide `uv`, PyTorch/EModel, OpenCV, NumPy, PySide6, Open3D.

**Spec:** `docs/superpowers/specs/2026-10-03-semtilestereo-live-and-review-tools-design.md`

## Global Constraints

- Use only the released E3 gate/refiner plus the A09 stereo, VKITTI 14-class semantic, and ADE20K encoder checkpoints; no training.
- No image or disparity resizing; pad paired native inputs right/bottom to a multiple of 32, then crop prediction to the original size.
- Current live source is pre-rectified side-by-side `/dev/video2`, default `1280x480`; optional map application only when explicitly requested.
- Live mode shows exactly two OpenCV windows: color-coded disparity and segmentation overlaid on left RGB; close button, `q`, Escape, and Ctrl+C all stop cleanly.
- `pair` writes float32 disparity and native semantic logits `.npy`, two PNGs, and the `result.npz` bundle described in the spec.
- VKITTI Camera 0 preset: `fx=fy=725.0087`, `cx=620.5`, `cy=187.0`, baseline `0.532725 m`, valid at `1242×375` unless explicitly overridden.
- Install dependencies with `uv add`/`uv sync` from wheels; never `uv pip` or source builds. Preserve unrelated working-tree changes.

## Review Focus

- Odd-width, swapped, or unequal-size stereo input must fail before inference instead of silently pairing unrelated pixels (Tasks 1, 3).
- Low-resolution semantic logits must be preserved raw while only the class map/confidence are upsampled and cropped (Task 1).
- Missing or incompatible rectification maps must produce an error rather than falling back to unrectified inference (Task 3).
- Malformed `.npz` metadata or array shapes must not be accepted by the GUI (Task 2).
- Zero/negative/nonfinite disparity, out-of-range depth, or disabled classes must produce no invalid or hidden points (Task 4).

---

### Task 1: Shared released-E3 inference core

**Files:** Create `semtilestereo/__init__.py`, `semtilestereo/core.py`; test `tests/test_semtilestereo_core.py`.

**Interfaces:** `ModelPaths` (four checkpoint paths); `InferenceResult(disparity_px, semantic_logits, class_id, class_confidence, left_rgb)`; `load_model(paths: ModelPaths, device: str) -> EModel`; `infer_pair(model, left_bgr: np.ndarray, right_bgr: np.ndarray, device: str) -> InferenceResult`.

- [ ] **Step 1: Write failing tests.** Fake a model returning padded disparity and low-res logits; assert exact unpadded `H×W` disparity/class/confidence, native logits shape, left RGB conversion, and no resize. Assert unequal shapes, wrong channel count, and missing checkpoint paths fail clearly.
- [ ] **Step 2: Verify red.** `uv run --no-sync python -m pytest -q tests/test_semtilestereo_core.py` must fail for missing interfaces, not import/dependency errors.
- [ ] **Step 3: Implement core.** Reuse `EModel` with `gate_hidden=32`, `residual_hidden=128`, `use_semantics=True`; load `payload['trainable']`. Store the known A09 SHA-256 and default stable local path in `ModelPaths`; verify default model files and emit a retrieval hint when absent. Pad RGB float32 `[0,255]` exactly as the existing figure inference script, upsample logits only for class decoding.
- [ ] **Step 4: Verify green.** Run the focused tests and `python -m py_compile` on edited Python.
- [ ] **Step 5: Commit only this task's files.**

### Task 2: Versioned output contract and rendering helpers

**Files:** Create `semtilestereo/results.py`, `semtilestereo/visuals.py`; test `tests/test_semtilestereo_results.py`.

**Interfaces:** `CameraParameters(fx, fy, cx, cy, baseline_m)`; `save_result(result: InferenceResult, out_dir: Path, *, camera: CameraParameters | None, metadata: dict, display_max: float) -> Path`; `load_bundle(path: Path) -> tuple[InferenceResult, dict]`; `colorize_disparity(...)` and `segmentation_overlay(...)`.

- [ ] **Step 1: Write failing tests.** Round-trip exact arrays and schema metadata via `.npz` with `allow_pickle=False`; check both `.npy` and PNG artifacts. Assert missing keys, unsupported schema, wrong dtype/shape, nonfinite camera parameters, and invalid display range fail. Check colorization has a stable palette and marks invalid disparity distinctly.
- [ ] **Step 2: Verify red.** `uv run --no-sync python -m pytest -q tests/test_semtilestereo_results.py` fails for absent functions.
- [ ] **Step 3: Implement minimal output module.** Import 14 labels/colors from `semantic_data.py`; save raw logits unchanged; use JSON string metadata and schema version `1`; no pickle payloads. Keep all channel/color conversion in this module rather than duplicating it in front ends.
- [ ] **Step 4: Verify green.** Run the focused tests and syntax checks.
- [ ] **Step 5: Commit only this task's files.**

### Task 3: CLI `live` and `pair`

**Files:** Create `semtilestereo/infer.py`, `semtilestereo/camera.py`; test `tests/test_semtilestereo_cli.py`.

**Interfaces:** `parse_args(argv: list[str]) -> argparse.Namespace`; `split_stereo(frame, left_view) -> tuple[np.ndarray, np.ndarray]`; `load_rectification(path: Path, image_shape) -> RectificationMaps`; `run_pair(args) -> Path`; `run_live(args) -> None`.

- [ ] **Step 1: Write failing tests.** Assert subcommand parsing/defaults, valid/odd-width split, left-view swap, rotation, optional map shape rejection, two expected preview window names, and exit decision for either closed window/`q`/Escape. A fake capture verifies release/`destroyAllWindows` on normal stop and read failure.
- [ ] **Step 2: Verify red.** `uv run --no-sync python -m pytest -q tests/test_semtilestereo_cli.py` fails for missing behavior.
- [ ] **Step 3: Implement CLI.** Use one inference core and results module; configure V4L2 MJPG width/height, confirm actual captured dimensions, `cv2.remap` only when an explicit compatible XML is provided. Catch KeyboardInterrupt at the boundary and clean up in `finally`. `pair` takes explicit left/right/out paths and optional camera metadata.
- [ ] **Step 4: Verify green.** Run focused tests, syntax checks, and `--help` for `live` and `pair`.
- [ ] **Step 5: Commit only this task's files.**

### Task 4: Metric point-cloud conversion and filtering

**Files:** Create `semtilestereo/geometry.py`; test `tests/test_semtilestereo_geometry.py`.

**Interfaces:** `VKITTI_CAMERA: CameraParameters`; `points_from_result(result: InferenceResult, camera: CameraParameters, *, visible_classes: set[int], min_depth_m: float, max_depth_m: float, stride: int, semantic_colors: bool) -> tuple[np.ndarray, np.ndarray, dict[int, int]]`.

- [ ] **Step 1: Write failing tests.** With a tiny array and known intrinsics, assert `Z=fx·B/d`, `X=(u−cx)Z/fx`, `Y=(v−cy)Z/fy`; assert RGB versus semantic colors, class filtering/counts, sampling stride, zero/negative/NaN disparity removal, and invalid depth range.
- [ ] **Step 2: Verify red.** `uv run --no-sync python -m pytest -q tests/test_semtilestereo_geometry.py` fails for absent functions.
- [ ] **Step 3: Implement pure NumPy geometry.** Return finite point/color arrays in camera metres, ready to wrap in `open3d.geometry.PointCloud`; do not alter raw saved predictions.
- [ ] **Step 4: Verify green.** Run focused tests and syntax checks.
- [ ] **Step 5: Commit only this task's files.**

### Task 5: Qt6/Open3D app

**Files:** Create `semtilestereo/gui.py`, `semtilestereo/viewport.py`; modify `pyproject.toml`, `uv.lock`; test `tests/test_semtilestereo_gui.py`.

**Interfaces:** `Open3DViewport(QWidget).set_cloud(points, colors)`, `.reset_view()`; `ReviewWindow` with `open_result(path)`, `infer_pair(left_path, right_path)`, `apply_filters()`; QThread worker calls Task 1 and Task 2 interfaces.

- [ ] **Step 1: Add wheel dependencies via project-wide `uv add --no-build pyside6 open3d` and `uv sync`.** Verify `import open3d` resolves the installed distribution, not the repo's `open3d/` namespace; abort rather than build from source. Confirm `OffscreenRenderer` works on the local desktop/driver before wiring the viewport.
- [ ] **Step 2: Write failing tests.** Offscreen Qt test loads a synthetic bundle, checks only present class checkboxes, toggles one and verifies filtered point count, changes intrinsics and verifies rebuilt coordinates, rejects a VKITTI preset on a mismatched image size without warning, and checks that worker failure appears in the status area without a GUI crash. Inject a fake viewport for these tests; keep geometry tests independent of GPU/display.
- [ ] **Step 3: Verify red.** Run `QT_QPA_PLATFORM=offscreen uv run --no-sync python -m pytest -q tests/test_semtilestereo_gui.py`; it fails for absent UI interfaces.
- [ ] **Step 4: Implement GUI and viewport.** Embed Open3D-rendered `QImage` in a QWidget and map drag/pan/wheel to camera updates. Build controls, checked present-class list, preset/custom camera fields, file picker, example list, and asynchronous inference. Use `points_from_result`, not a second geometry implementation.
- [ ] **Step 5: Verify green.** Run GUI tests, `py_compile`, and a local desktop smoke of orbit/zoom/filtering; note environment failures precisely.
- [ ] **Step 6: Commit only this task's files and dependency lockfile.**

### Task 6: Real-checkpoint smoke, examples, and operator docs

**Files:** Add stable A09 weight under ignored `models/stereo/semtilestereo/`; generated bundles under ignored `examples/semtilestereo/`; create `semtilestereo/README.md`; modify `AGENTS.md`; test `tests/test_semtilestereo_integration.py`.

**Interfaces:** `make_examples` calls Task 1 `infer_pair` and Task 2 `save_result` for Scene01/Scene06/Scene18 pairs; GUI example menu reads those bundles.

- [ ] **Step 1: Write failing integration assertions.** Verify all four checkpoint paths/hashes, three source scenes, exact bundle keys, output dimensions, and a direct-model-vs-saved-array parity check for one pair. Check that a missing local A09 weight yields retrieval guidance.
- [ ] **Step 2: Verify red.** Run the focused integration test; expected failure is missing example/stable weight, not unrelated import errors.
- [ ] **Step 3: Populate local assets and docs.** Copy/checksum the A09 checkpoint to its documented model path, generate at least three different-scene VKITTI bundles with the released E3 model and preset metadata, record hashes/provenance and commands, and add folder roles to `AGENTS.md`. Do not overwrite user camera calibration files or commit routine raw datasets/large generated arrays.
- [ ] **Step 4: Verify green and full suite.** Run focused integration, project tests with plugin autoload disabled if needed, syntax checks, CLI `pair` on one real pair, GUI bundle/filter smoke, and record any unrun physical-camera check with its exact `uv run` command.
- [ ] **Step 5: Commit only task docs/tests/small intentional assets; verify `git diff --check` and report untracked ignored examples separately.**
