# Superseded — interrupted first 10k attempt

This run was stopped deliberately at ~step 2720 and has **no `results.json`,
`REPORT.md`, or training curve**; it is not a result.

It was interrupted to adopt an I/O-bound optimization in the runner:

- Every step previously decoded two full 540×960 PNGs from the external SSD
  (`cv2.imread` ≈ 128 ms/step) plus the PFM ground truth, synchronously at batch
  size 1, leaving the GPU at ~2–36% utilization.
- The runner now decodes all 200 pairs into RAM once (`preload`) and runs the
  frozen encoder over the left/right pair in a single call.

The replacement run keeps the identical protocol — same manifest, seed, split,
native crops, and step count — but feeds the GPU from RAM. See the sibling
`A02B_yolo26s_frozen_native_*` directory created after this one for the result.
The step checkpoints here are partial and can be deleted.
