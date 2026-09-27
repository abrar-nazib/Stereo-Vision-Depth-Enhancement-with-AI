# Repository guide

## Purpose and layout

This is an experimental Python workspace for stereo-camera calibration,
disparity/depth estimation, and point-cloud visualization. It does not expose a
single packaged application or a centralized dependency manifest.

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

## Running work safely

- Use Python 3 and a project-local virtual environment. Install only the
  packages needed by the workflow you are running; common requirements are
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
- Avoid committing model checkpoints, camera recordings, raw sensor streams,
  or large generated arrays. Record reproducibility details (camera model,
  resolution, baseline, calibration source, model/checkpoint, and parameters)
  alongside any retained result.

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
