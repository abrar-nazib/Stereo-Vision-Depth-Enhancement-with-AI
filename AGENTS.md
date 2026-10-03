# Repository guide

## Purpose and layout

This is an experimental Python workspace for stereo-camera calibration,
disparity/depth estimation, semantic segmentation, and point-cloud
visualization. It does not expose a single packaged application; the root
`pyproject.toml` and `uv.lock` manage the project environment.

- `src/` is the main stereo-vision prototype: capture, calibration,
  rectification, SGBM disparity/depth, PSMNet, RAFT Stereo, and point-cloud
  generation.
- `combined_depth/` fuses rectified SGBM metric depth with MiDaS relative
  depth. `main.py` is its live-camera entry point; `point_cloud_viewer.py`
  reads saved fusion results.
- `stereo_calibration/` and `camera_calibrate_rahi/` are independent
  calibration/capture workflows with their own parameter formats.
- `open3d/` contains point-cloud and surface-reconstruction experiments,
  primarily notebooks.
- `maixcam/` contains MaixSense RGB-D stream experiments.
- `training/` creates the EPE plot. The plotted data are synthetic in
  `training/plot_gen.py`, not recorded training metrics.

## Research and artifact map

Keep this map current when an important folder is added or its role changes.
Record a key result or entry-point filename when it prevents confusion, but
do not turn this guide into a file-by-file inventory. Historical experiments
are preserved as separate versions; do not assume a newer folder supersedes
all metrics or code in an older one.

- `experiments/yolo_las2_v1/`, `lightstereo_s_a02/`, `hitnet_a03/`,
  `lightstereo_m_a04/`, and `selective_raft_a04/` are earlier local stereo
  baselines and shared-YOLO-encoder ablations. Their `runs/` folders carry
  versioned manifests, configurations, measurements, and reports.
- `experiments/a05_clean/` and `a06_shallow/` hold the shallow stereo-head
  and architecture ablations. The A08 architecture diagram and its renderer
  live in `a06_shallow/`; check the implementation before treating a diagram
  as an exact architecture specification.
- `experiments/a09_encoder_m/` holds the YOLO26m-encoder A09 ablation and
  its run records. `experiments/final_pass/` holds the full SceneFlow-family
  A09 training/inference workflow and Modal run notes, including the plateau
  and OneCycle comparisons (`RUN_A09M_A10*.md`). The A run folders deliberately
  remain at their historical paths; `experiments/A/INSIGHTS.md` indexes their
  key results without relocating checkpoints or breaking old path references.
- `experiments/vkitti2/` holds VKITTI2 data conversion, semantic fine-tuning,
  fusion-head ablations, and Modal runners. Read `RESEARCH_DIRECTION.md`
  before planning claims and `SEMANTIC_MODAL_RUN.md` for the seeded 14-class
  split, stopped training history, and separate best-checkpoint test. The
  latter reports validation mIoU 0.83168 at epoch 9 and random-split test
  mIoU 0.82664; neither measures real-world depth generalization.
- `experiments/B/` groups the B-series and its `INSIGHTS.md`.
  `B_0_fused_baseline/`, `B_1_residual/`, `B_2_confidence/`, and
  `B_3_classaware/` are the local B-series semantic-to-disparity ablations.
  `B_0_fused_baseline/model.py` reuses the exact frozen YOLO layers 0–6 once
  for both stereo views and the left semantic branch; `run.py` owns the
  800/100/100 VKITTI2 frame-grouped protocol. `runs/` holds versioned traces,
  checkpoints and metrics. Run `20261001_b_full_seed42` selected B1 on
  validation EPE; test EPE was 2.2730 px frozen versus 2.2159 px with B1,
  with unchanged 14-class mIoU 0.6780 on its Scene20 test subset. Results are
  in-domain feasibility evidence, not independent real-world generalization.
- `experiments/C/` groups the C-series and its `INSIGHTS.md`.
  `C_1_large_control/`, `C_2_feature_fusion/`, and
  `C_3_semantic_match/` contain the next frozen-predictor depth-only head
  ablations: larger output-only control, multi-scale feature fusion, and
  right-feature matching guidance. `C_1_large_control/run.py` owns the shared
  training/evaluation protocol; `queue.py` launches arms sequentially and
  writes detached-run status and per-arm logs. See its `README.md` for tail
  commands. C-series mIoU uses only GT-present classes, unlike the earlier
  B-series union-present reporting; compare only after aligning protocols.
- `experiments/C/C_4_no_semantics/` and `experiments/C/C_5_misaligned_semantics/` hold the
  equal-architecture C2 causal-control runs. They zero or spatially misalign
  semantic guidance throughout training and evaluation, respectively, while
  retaining the same frozen stereo/shared features. The shared runner and
  detached queue remain in `C_1_large_control/`; C4's `README.md` has the
  control protocol and polling paths.
- `experiments/D/` groups the D-series and its result-focused `INSIGHTS.md`.
  `D_1_semantic_cost/`, `D_2_semantic_cost_residual/`,
  `D_3_no_semantic_cost/`, and `D_4_no_semantic_residual/` hold SGNet-inspired
  stereo-candidate semantic gating and class-conditioned refinement ablations,
  plus equal-architecture no-semantics controls. D1 owns `model.py`, `run.py`,
  the sequential detached `queue.py`, and the protocol/polling `README.md`.
  These preserve the B/C 800/100/100 split and frozen pretrained predictors.
  D1–D3 test EPE differed by -0.2424 px and D2–D4 by -0.2479 px, with
  lower bad-3 in both semantic arms; read the D insight caveats before a
  manuscript claim, especially semantic-teacher split overlap risk.
- `experiments/E/` holds the D2 head-capacity sweep: 2× and 4× gate/refiner
  widths, each paired with an equal-capacity no-semantics control. E1 owns
  the shared model, runner, sequential queue and polling `README.md`; D2/D4
  remain the 1× references. Treat E as an in-domain capacity ablation, not
  proof of cross-domain generalization. Its `full_vkitti_data.py`,
  `full_vkitti_train.py`, and `modal_full_vkitti_t4.py` implement the subsequent
  teacher-split-aligned full VKITTI2 T4 comparison of E3/E4/D2; results live
  on `svde-results:/E_full_vkitti/`, not in the local ablation run folders.
- `experiments/F/` holds the inference-only KITTI 2015 real-domain transfer
  test. `F1` is the frozen A09 stereo model without semantic fusion; `F2` is
  the frozen E3 model; `F3` is its matched-capacity E4 no-semantics control.
  `modal_kitti_t4.py` runs them on detached Modal T4 calls, and `README.md`
  defines the shared GT masks and polling protocol. Run
  `f1_f2_kitti2015_v1_20261003` found all-valid KITTI EPE 3.0822 / 2.5876 /
  2.7139 px for F1/F2/F3 across 200 pairs. Read `INSIGHTS.md` for paired
  semantic-control analysis and caveats.
  Results live locally in F-arm `runs/` and on `svde-results:/F_kitti2015/`.
- `models/segmentation/` stores local semantic/instance checkpoints and a
  provenance inventory. The frozen-shared-trunk VKITTI checkpoint is
  `yolo26m-sem-vkitti2-14class-freeze7-best.pt` (14 semantic classes); large
  weights stay gitignored. `models/stereo/` stores downloaded stereo-model
  weights and their model-specific notes.
- `paper/reference_papers/` contains source PDFs, structured summaries in
  `summaries/`, and cropped architecture/equation figures in `figures/`.
  `paper/figures/` contains manuscript figures, not model checkpoints.
  `paper/figures/semtilestereo/` holds the editable Excalidraw architecture,
  checked preview, same-frame VKITTI2 training thumbnails, and provenance;
  the inference helper is in `paper/figures/semtilestereo_infer_example.py`.
- `tests/` holds fast protocol and code tests; `docs/` holds research links
  and implementation plans. `.agents/skills/` contains local workflow/domain
  skills, including Modal and ML-architecture guidance. `external_models/`
  contains third-party source checkouts used to load research checkpoints.
- `/media/abrar/AbrarSSD/Datasets/` is the external raw/custom dataset store,
  outside this repository. Do not mistake it for a versioned experiment output.

## Running work safely

- Use Python 3 with the project-wide `uv` environment and `uv.lock`; do not
  use `uv pip` or ad-hoc package installs. Common workflow requirements are
  `numpy`, `opencv-contrib-python`, `torch`, `torchvision`, `scipy`,
  `scikit-learn`, `matplotlib`, `open3d`, `tqdm`, and (for MaixSense)
  `imageio` and `requests`.
- Run scripts from their own directories, or preserve their directory-relative
  paths. Several scripts load adjacent XML/PKL files and write outputs beside
  themselves.
- Live scripts require two accessible cameras and a desktop OpenCV display.
  Confirm camera indices before use: most current scripts assume left camera
  `2`, right camera `0`, and 640x480 input. `combined_depth/main.py` chooses
  V4L2 on Linux and DirectShow on Windows.
- MiDaS-based scripts download models through `torch.hub` on first use. Do not
  assume a network connection or GPU is available; `combined_depth/main.py`
  falls back to CPU.
- The fusion workflow relies on `cv2.ximgproc`, supplied by
  `opencv-contrib-python`, and on `combined_depth/stereoMap.xml` containing
  rectification maps plus `Q`.

## Calibration and data compatibility

- Treat each calibration workflow as self-contained. Their XML node names and
  pickle contents are not interchangeable without explicitly verifying map
  names, camera ordering, resolution, and units.
- Recalibrate after changing either camera, the lens/focus, baseline, camera
  order, or capture resolution. Validate rectification visually with horizontal
  epipolar lines before trusting disparity or depth.
- Preserve left/right pairing and timestamp/index alignment for captured
  images. Do not silently swap camera inputs: disparity sign and depth validity
  depend on that ordering.
- Calibration files (`*.pkl`, `*.xml`) and saved arrays (`*.npy`) are valuable
  experiment artifacts. Inspect shapes, dtypes, units, and provenance before
  overwriting or reusing them.

## Outputs and version-control hygiene

- The root `.gitignore` intentionally excludes generated `.jpg`, `.png`,
  `.ply`, and `.pcd` files. Keep raw captures, visualizations, and point clouds
  out of commits unless a small, intentional fixture is required.
- `combined_depth/main.py` saves timestamped result arrays, visualizations, and
  point-cloud data after pressing `s`. Its write location is relative to the
  process working directory; run it from `combined_depth/` when outputs should
  land in `combined_depth/images/`.
- Avoid committing routine model checkpoints, camera recordings, raw sensor
  streams, or large generated arrays. The intentionally selected `best*.pt`
  and `best*.pth` checkpoints in substantive experiment runs are eligible for
  Git after checking individual sizes; smoke/probe and downloaded weights stay
  ignored. Record reproducibility details (camera model,
  resolution, baseline, calibration source, model/checkpoint, and parameters)
  alongside any retained result.

## Local ablation protocol

- Give each series its own `experiments/<letter>/` folder and versioned
  `runs/<run-id>/` artifacts; preserve old series rather than rewriting their
  results. Record arm architecture, trainable parameter count, frozen-weight
  and dataset hashes, seed, crop, optimizer/schedule, validation selection
  rule, GPU, metrics, and limitations in each manifest/report.
- For the current VKITTI2 fusion-head comparison, keep the fixed-seed
  1,000-pair subset: 800 train, 100 validation, 100 test. Group all ten
  variations of a `(scene, frame)` together: 80/10/10 distinct source-frame
  groups. Keep the A09 stereo model, shared YOLO layers 0–6, and VKITTI
  semantic decoder frozen. Train only the stated fusion head, and include a
  same-architecture/same-capacity no-semantics control for causal claims.
- Never resize stereo images or disparity in these comparisons. Use paired
  native-pixel crops for training and padded full-image validation/test;
  preserve left/right ordering and report valid-pixel EPE, RMSE, bad-0.5/1/2/3,
  D1, segmentation mIoU definition, trainable parameters, and when relevant
  inference latency and peak VRAM. Select the checkpoint on validation only;
  the test split is for final comparison, not tuning.
- Before a long local ablation, run unit tests and a short smoke of **every**
  arm. Launch full arms sequentially on the RTX 3050 in a detached queue with
  a durable status JSON and separate arm logs; give the user copyable polling
  commands and do not keep polling on their behalf. Do not spend Modal credit
  on these local screening runs. A later full VKITTI Modal run is a separate
  decision after the ablation is reviewed.
- B/C/D/E VKITTI evidence is in-domain. The semantic teacher's random
  full-VKITTI split may overlap the depth test frames; audit or eliminate this
  overlap, repeat with independent source frames/seeds, and evaluate a real
  domain before claiming generalization in a manuscript.

## Editing and validation

- Prefer focused changes within one workflow; this repository intentionally
  keeps several historical prototypes side by side.
- Keep camera IDs, file paths, and algorithm parameters configurable where a
  change touches them. Preserve the existing 640x480 assumptions unless the
  matching calibration maps are regenerated.
- Before committing Python changes, at minimum run syntax checks on edited
  files, for example: `python -m py_compile path/to/edited_script.py`.
- Do not run hardware capture, model downloads, or long training/point-cloud
  jobs as a default validation step. State the required hardware/data and give
  the operator the command instead.

## Current research direction and manuscript reminder

- Before planning VKITTI semantic training, semantic-to-stereo fusion, or the
  manuscript, read `experiments/vkitti2/RESEARCH_DIRECTION.md`. It records the
  current **hypothesis**, existing mapped-nine-class results, the semantic vs
  instance choice, the shared-encoder freeze constraint, and the causal and
  cross-domain controls needed before making a paper claim.
- Treat that note as a working proposal rather than a demonstrated result.
  In particular, do not claim that VKITTI semantic improvements establish
  real-world disparity generalization without a separate evaluation.
