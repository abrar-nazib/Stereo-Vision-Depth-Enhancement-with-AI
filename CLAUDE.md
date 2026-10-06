# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Experimental PyTorch research workspace: stereo calibration, disparity/depth
estimation, YOLO-based semantic segmentation, and semantic-to-disparity fusion
heads that share one frozen encoder. The end goal is live YOLO-seg and
good-enough depth running together on shared layers. Disparity SOTA on its own
is not the goal. There is no single packaged app. `AGENTS.md` holds the
per-series results map with key metrics and caveats. Read the entry for a
series before touching or citing it, and keep that map current when a series
is added.

## Delegation rule (mandatory)

The main agent makes only the cognitively demanding decisions: architecture and
experiment design, interpreting results, adopt/kill verdicts, and ambiguous
tradeoffs. Delegate everything else to subagents with an explicit `model`:

- `haiku` handles directory listings, grep/find, file lookups, reading known
  files, fixed shell commands, and short extraction.
- `sonnet` handles code search that needs judgment, tracing imports across
  experiment folders, summarizing several files, and routine code edits.
- `opus` is only for adversarial verification, complex multi-file reasoning, or
  a task that already failed at Sonnet.

Prefix every spawn `description` with its tier (`[haiku] list runs/`), and name
the tier in the user-facing message announcing the dispatch. Do not
re-run a search yourself after delegating it.

## Skills

Project skills live in `.agents/skills/` (used by another agent tool) and are
symlinked into `.claude/skills/`. Edit them in `.agents/skills/`:

- `stereo-vision-expert`: design choices from 26 papers in `paper/reference_papers/`,
  covering stereo heads, losses, fusion and training. Load it before proposing an
  architecture or loss.
- `ml-arch-diagram` + `excalidraw-diagram`: architecture figures in published
  paper-figure grammar. Render the diagram, inspect it, fix it and re-render
  before delivering. The headless renderer's fonts are narrower than the
  Excalidraw app's, so leave text headroom.
  The renderer needs a one-time setup: `cd .claude/skills/excalidraw-diagram/references && uv sync && uv run playwright install chromium`.
- `modal-expert`: Modal CLI/SDK reference.
- `humanizer`: strips AI-writing tells from manuscript prose.

`.commandcode/taste/taste/taste.md` records the user's working preferences from
earlier sessions. Examples: real ablations are at least 10k steps, every finished
batch ends with a per-arm adopt/kill verdict and visual collages, frozen
pretrained encoders are preferred, and a severe regression against a trusted
baseline is treated as a probable implementation bug. Consult it for judgment
calls.

## Environment and commands

- Python 3.12 with `uv` and `uv.lock` (torch from the cu130 index). Never use `uv pip`
  or ad-hoc installs. Modal and pytest are in the `dev` group.
- **Tests must run as `python -m pytest` from the repo root.** Bare `pytest` cannot import
  `semtilestereo`/`experiments`. The machine's ROS `PYTHONPATH` injects a broken
  pytest plugin, so clear it:
  ```bash
  env -u PYTHONPATH uv run python -m pytest -q tests            # full suite (~20s)
  env -u PYTHONPATH uv run python -m pytest -q tests/test_h_split.py
  env -u PYTHONPATH uv run python -m pytest -q tests/test_b_protocol.py::test_split_is_800_100_100_without_source_frame_leakage
  ```
  Always pass `tests` (or `--ignore=external_models`), because OpenStereo's own tests break
  collection. Tests that need data, checkpoints or CUDA skip when those are
  missing. The exception is `test_b_protocol.py`, which hard-fails when the
  dataset SSD is unmounted. Set `SEMTILESTEREO_RUN_INTEGRATION=1` to run
  the semtilestereo integration tests.
- There is no linter or formatter config. The minimum check on edited files is `python -m py_compile <file>`.
- Do not run hardware capture, model downloads, or long training/point-cloud jobs as
  routine validation. State what they require and give the user the command.

## Architecture: how the experiment code fits together

`experiments/` is a namespace package (no `__init__.py`), imported as
`experiments.<S>.<arm>.<module>` and run with `python -m` from the repo root.
Later series import earlier ones, so changing an old series can break newer ones:

- **A09 stereo model** is `FusionStereoLite` in `experiments/a06_shallow/runners/model_a07v2.py`
  (loaded via `sys.path.insert`). `experiments/final_pass/modal_full_pass.py` trained it
  on full SceneFlow on Modal.
- **`experiments/B/B_0_fused_baseline/model.py`** defines `FusedStereoSemantic`. It
  runs the frozen YOLO26m layers 0–6 once per stereo pair and feeds them to both
  A09 and the VKITTI 14-class semantic decoder
  (`models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt`). Its
  `run.py` owns `DATA`, `STEREO`, `pad32` and `sha256`, and defines the
  800/100/100 frame-grouped split.
- **`C_1_large_control/run.py`** supplies `atomic_json` and the mIoU helpers. D, E, F, G and H all
  funnel through B_0's and C_1's runners.
- **D → E → F/G**: `D_1_semantic_cost/model.py` (`DModel` = `SemanticCostGate` + `ClassResidual`)
  → `E_1_wide2_semantics/model.py` (`EModel`; E3 is the wide4 arm and the released model)
  → F (KITTI 2015 transfer, Modal-only) and G (edge refinement on frozen E3).
  H is stereo-only and reuses G's `edge_mask` and B_0's helpers.
- **`semtilestereo/`** is the released E3 inference package:
  - `core.py` runs inference and pins the A09 checkpoint by SHA.
  - `geometry.py` turns disparity into metric points.
  - `results.py` defines the versioned bundle format.
  - `infer.py` is the `pair`/`live` CLI.
  - `gui.py`/`viewport.py` are the Qt6 + Open3D reviewer.
  ```bash
  uv run python -m semtilestereo.infer pair left.jpg right.jpg outputs/pair --vkitti-camera
  uv run python -m semtilestereo.infer live --camera /dev/video2 --width 1280 --height 480
  uv run python -m semtilestereo.gui
  ```
- `src/`, `combined_depth/`, `stereo_calibration/`, `camera_calibrate_rahi/`, `open3d/` and
  `maixcam/` are older standalone camera/SGBM/MiDaS prototypes. Run each from its own
  directory, because they load adjacent XML/PKL files and write outputs beside themselves.

Datasets and the A09 checkpoint are hardcoded under `/media/abrar/AbrarSSD/`:
- `Datasets/sceneflow_driving`
- `Datasets/VirtualKitti2/ablation{1000,2000}`
- `ResearchArtifacts/SVDE/...`

Most runners accept `--data`. This storage is external and unversioned.

### Per-series layout and the detached queue

Each series has `run.py` (protocol plus a per-arm CLI: `--arm --run-id --steps
--eval-every [--patience] [--train-limit/--eval-limit]`), `model.py` with `ARMS`, and
`queue.py`. The queue runs the arms sequentially as subprocesses, writes
`queue/<run-id>/status.json` atomically (`queued|running|failed|complete`) plus
one `<arm>.log` per arm, and stops at the first failure.
Arm outputs go to `<arm>/runs/<run-id>/` (`manifest.json`, `history.jsonl`, `best.pth`,
`test.json`, `summary.json`, `visuals/`).

```bash
# smoke every arm first (metrics are not results)
uv run --no-sync python -m experiments.E.E_1_wide2_semantics.run --arm <arm> --run-id <smoke-id> --steps 1 --eval-every 1 --eval-limit 1
# full detached queue (from repo root)
nohup setsid uv run --no-sync python -m experiments.C.C_1_large_control.queue --run-id <id> --steps 10000 --eval-every 1000 --patience 3 \
  > experiments/C/C_1_large_control/queue_launcher.log 2>&1 < /dev/null &
```

### Modal

Run Modal entrypoints as `uv run --no-sync modal run [-d] path.py::function`,
probing first (for example `experiments/F/modal_kitti_t4.py::probe_t4 --limit 2`,
then `::launch`).

Volumes:
- `svde-results` holds results (`/F_kitti2015/`, `/E_full_vkitti/`, `/final_pass/`).
- `stereo-datasets`, `sceneflow-shards` and `svde-vkitti2` hold the data.

Never spend Modal credit on local screening ablations. A full Modal run is a
separate decision made after the ablation has been reviewed.

## Ablation protocol rules

- Each series gets its own `experiments/<letter>/` folder with versioned `runs/<run-id>/`.
  Never rewrite old results. Newer folders do not supersede all code or metrics in older ones.
- Every manifest/report records:
  - arm architecture and trainable parameter count
  - frozen-weight and dataset hashes
  - seed, crop, and optimizer/schedule
  - the validation selection rule, GPU, metrics, and limitations
- VKITTI2 fusion-head comparisons:
  - Use the fixed-seed 1,000-pair subset, split 800/100/100. Group all
    ten variations of each `(scene, frame)` together, giving 80/10/10 source-frame groups.
  - Keep A09, YOLO layers 0–6 and the semantic decoder frozen, and train only the stated head.
  - Causal claims require a no-semantics control with the same architecture and capacity.
- Never resize stereo images or disparity. Train on paired native-pixel crops,
  and validate/test on padded full images. Preserve left/right ordering.
- Report:
  - valid-pixel EPE, RMSE, bad-0.5/1/2/3 and D1
  - the mIoU definition (B used union-present classes, C onward use GT-present classes)
  - trainable params, and latency/peak VRAM when relevant
- Select checkpoints on validation only. Test is for the final comparison.
- Before a long run, run the unit tests and a smoke of **every** arm. Then launch the
  arms sequentially on the local RTX 3050 in a detached queue.
- The queue handoff must include an **absolute-path** status command and an
  absolute-path `tail -F` command for **every** arm, including arms that have not
  started.
  - Create the log files at queue setup so `tail -F` works before an arm starts.
  - Verify the commands from `~`.
  - Explain that arms start automatically and that stopping `tail` does not
    stop training.
  - Do not keep polling on the user's behalf.
- B–H VKITTI evidence is in-domain only. The semantic teacher's random split may
  overlap the depth test frames. Never claim real-world generalization without an
  independent-domain evaluation. Read `experiments/vkitti2/RESEARCH_DIRECTION.md`
  before planning fusion work or manuscript claims, and treat it as a hypothesis.

## Calibration, data and git hygiene

- Each calibration workflow is self-contained, and their XML/PKL formats are
  not interchangeable.
- Recalibrate after any change to the cameras, lens/focus, baseline, camera
  order or resolution. Most live scripts assume left camera `2`, right camera
  `0`, and 640x480.
- Validate rectification with horizontal epipolar lines.
- Never silently swap left and right, because disparity sign depends on the
  order.
- `*.pkl`, `*.xml` and `*.npy` files are valuable artifacts. Inspect shape, dtype, units and
  provenance before overwriting them.
- Generated `.jpg/.png/.ply/.pcd/.npy`, videos, downloaded weights and
  `external_models/` are gitignored.
- Selected `best*.pt(h)` checkpoints from substantive runs may be committed
  after checking their size. Smoke/probe checkpoints stay ignored.
