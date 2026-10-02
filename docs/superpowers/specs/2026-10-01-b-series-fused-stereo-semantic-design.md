# B-series fused semantic-to-disparity ablation design

Status: design for review; no B-series training has begun. The purpose is to
test whether useful semantic predictions improve a frozen stereo predictor
while retaining a single shared YOLO trunk and fast CUDA inference.

## Fixed checkpoints and contract

- Stereo: A09 V-arm `best.pth` from the full SceneFlow-family run, using the
  frozen YOLO26m ADE20K layers 0–6 and the trained stereo/tile head.
- Semantics: `models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt`,
  with the same layers 0–6 and trained layers 7–17. Both checkpoints stay frozen.
- A local CPU comparison found all **186 tensors and buffers** in shared layers
  0–6 equal. The implementation must recheck this on load, fail closed on any
  mismatch, and record both checkpoint hashes.
- Input is one rectified RGB stereo pair in left/right order. Preserve native
  disparity pixel units; use only symmetric crop for training and replicate
  padding/unpadding for native-size evaluation. The result is a left-view
  disparity map, 14-class ID map, and class confidence/probability map.
  Coordinates `(u,v)` are implicit pixel indices, not extra network channels.

## Fused CUDA inference

1. Pad both views identically to a multiple of 32 and normalize once. The
   semantic neck includes a stride-32 stage; parity comparisons must give the
   legacy path the same padding. Stack
   left/right along the batch dimension; execute shared YOLO layers 0–6 **once
   on batch 2**. Cache the four pyramid outputs and split each into left/right.
2. Pass cached left/right pyramids directly into the existing A09 stereo head.
   Do not call its `forward` method if that would rerun `fnet`; expose a tested
   `forward_from_features` entry point that reproduces its disparity output.
3. Run semantic layers 7–17 using cached left layer-6 output and left layer-4
   and layer-6 skip tensors. Respect Ultralytics' `from` graph, specifically
   layers 12, 15, and 17. Do not run semantic layers 0–6 again or infer on the
   right view for the first B-series study.
4. Run the frozen stereo and semantic branches on one CUDA stream as the
   correctness/reference implementation. Optionally benchmark separate CUDA
   streams with explicit completion events before fusion; enable them only if
   median and p95 end-to-end latency improve on the RTX 3050 without excessive
   memory use. A batched shared trunk does not, by itself, imply concurrent
   downstream execution.
5. Feed disparity, left-view semantic probabilities, and optional confidence
   cues to the selected trainable depth-only head. The semantic logits are
   unchanged. Avoid transferring intermediate tensors to the CPU. Return
   native-size outputs after removing padding.

The fused result must match the current three-trunk-pass implementation within
documented FP32 and AMP tolerances on several real pairs, including one with
non-divisible dimensions. Compare full maps, not just final metrics. Benchmark
end-to-end inference after warm-up using CUDA events, synchronized boundaries,
the same input and dtype, and median/p95 over repeated runs; report separate
shared-trunk, branch, and fusion-head times plus peak allocated VRAM. The
single-stream fused path must be no slower than the old path within benchmark
noise; if dual streams regress, use single stream.

## B-series experiment folders and arms

Each folder under `experiments/` is independently versioned and contains its
own configuration, split/checkpoint/code hashes, metrics, traces, checkpoint,
and representative native-resolution disparity/semantic visualizations.

| Folder | Trainable part | Main question |
| --- | --- | --- |
| `B_0_fused_baseline/` | None | Does exact trunk reuse preserve frozen outputs and reduce inference cost? |
| `B_1_residual/` | Zero-initialized compact depth residual | Does left semantic guidance improve disparity at all? |
| `B_2_confidence/` | B1-style residual plus learned uncertainty gate | Can low-confidence semantics avoid harmful corrections? |
| `B_3_classaware/` | Class-conditioned residual with boundary-safe loss | Are region-specific disparity corrections useful? |

All heads produce `d = d0 + bounded_delta`; B2/B3 may multiply the delta by
a learned gate. Initialize the residual to zero so the first forward reproduces
the frozen stereo result. No semantic loss or semantic-head updates are allowed.
Keep parameters and runtime small and report both explicitly. Use one common
training budget, optimizer, crop distribution, and checkpoint-selection rule.

## Data and evaluation

- Reuse `/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation1000`. It has
  1,000 pairs: 200 per scene, with 20 source frames and ten variations per
  scene. Train on all 800 pairs from Scene01/02/06/18. Split Scene20's 20
  frame groups into ten validation and ten test groups, with all ten variations
  of a source frame in the same partition. Fix seed and hash the new manifest.
- Select head checkpoints by **validation EPE** only. Evaluate the untouched
  test partition once per frozen selection. Report EPE, RMSE, bad-0.5/1/2/3,
  D1, edge/interior and per-class disparity error, unchanged 14-class mIoU,
  parameter count, median/p95 latency, and peak VRAM. Summarize uncertainty
  across source-frame groups, not 100 purportedly independent images.
- Add a no-semantics control and shuffled-semantic control for trained heads;
  ground-truth labels are an analysis-only oracle, never a deployable input.
  In the first study, avoid a right-view semantic pass because that changes
  runtime and the question being asked.
- The semantic checkpoint was trained with a random full-VKITTI2 frame-group
  split. Some local val/test frames were likely in *semantic* training.
  Therefore this is an in-domain fusion feasibility study, **not** an
  independent system-generalization result. A later paper test needs a
  non-overlapping semantic teacher and an independent real-domain benchmark.

## Validation and launch gates

1. Unit tests: exact trunk parameter/buffer equality; branch graph skip
   routing; zero-residual identity; split disjointness by `(scene, frame)`;
   14-class output; invalid-disparity and crop correspondence masks.
2. Integration: fused versus legacy output parity in FP32 and AMP at crop
   and native resolution; verify the shared trunk executes once per pair.
3. CUDA benchmark: confirm the RTX 3050 is accessible, record driver/Torch
   versions and free VRAM, then select single versus dual stream empirically.
4. Run B0, then B1–B3 locally and sequentially so each has comparable GPU
   conditions. Use the project `uv` environment only. Do not use Modal.
5. Before declaring a winner, compare test EPE and bad-pixel rates with
   per-frame uncertainty and ensure 14-class mIoU is unchanged. Preserve
   all runs and note when an apparent gain comes with unacceptable latency.

At design time, `nvidia-smi` could not communicate with the driver and
`torch.cuda.is_available()` returned false; `/dev/nvidia*` nodes were absent
in this session despite PCI detection and loaded NVIDIA kernel modules.
CUDA training must not be launched until device access is restored.
