<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# VPEngine card

Source: paper/reference_papers/lightweight/VPEngine_Lucki_RAP2026.pdf (7 pp: 4 body, 1 refs, 2 appendix A-D). All pages read. No existing summary in summaries/. This is a SYSTEMS paper: no new network, no stereo, no accuracy metrics at all.

### 0. Meta
- Title: Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks. Lucki, Becktor, Georgakis, Royce, Khattak (JPL/Caltech, ETH Zurich).
- Venue: arXiv 2508.11584v3 (16 Sep 2026), cs.RO. "RAP2026" is only in the filename; venue not stated in the PDF.
- Code: https://github.com/nasa-jpl/visual-perception-engine (footnote 2, p.1). Python, ROS2 (Humble) C++ bindings.
- Domain: robotics (planetary rovers, field robots); hardware NVIDIA Jetson AGX Orin 64GB + AMD Ryzen 7735U host (Fig. 1).
- Datasets: none for accuracy. Latency test stream: 30,000 images 1920x1080 (Sec. IV); benchmark on 426 test images 1920x1080 (Figs. 3-4).

### 1. Problem & failure modes targeted
- Redundant computation and integration complexity when separate models per task each run their own backbone (Isaac ROS deploys ESS stereo depth, YOLOv8, SegFormer as independent nodes, p.1).
- Latency (R1: target ~one 30 Hz frame, ~33 ms, p.2 fn.3), memory predictability (R2, no OOM over long missions), extensibility (R3), per-task dynamic rate (R4).
- Jetson-specific: no native inter-process GPU memory sharing, so they build by-reference sharing (Sec. III-B).
- Not targeted: accuracy, stereo, task conflict (backbone is frozen/pretrained, heads independent).

### 2. Pipeline by stage (system pipeline, not a network)
- 2a Feature extraction: ONE foundation model run once per frame. Example: DINOv2 ViT-S (TensorRT). Outputs final features PLUS representations from three evenly spaced intermediate layers, kept in a "middle buffer" because the DAV2 depth head needs them (Sec. IV, p.3). Backbone shared by reference; frozen/pretrained (heads are not trained jointly; no training described). Feature resolution/channels: not stated.
- 2b Semantic/prior branch: the semantic head is a plain LINEAR layer on DINOv2 features (compiled to TensorRT) [4]; classes not stated.
- 2c-2g Cost volume / aggregation / disparity / refinement / upsampling: n/a (no stereo in the paper). Depth head is monocular DepthAnythingV2 using the TensorRT implementation from [10].
- 2h Fusion points: none between heads. Heads are independent consumers of the shared features; no head-to-head interaction (explicitly decoupled for fault isolation/rate control).
- Runtime structure (Fig. 2, Sec. III):
  - Foundation module = input buffer -> pre-processing (PyTorch module) -> foundation model (TensorRT kernel) -> middle buffer.
  - Each head module = async access to middle buffer -> preallocated memory slot -> head -> post-processing -> output queue (ROS2 publisher).
  - Every module runs in its OWN PROCESS; one CUDA MPS server, all processes share one CUDA context (Sec. III-A.4).
  - By-reference sharing via custom IPC built on low-level `cuMemImportFromShareableHandle` (Sec. III-B), since the standard Jetson path does not support it. Features never leave GPU; no GPU-CPU-GPU copies.
  - Copies occur only when popping from queue/buffer into the pre-allocated processing slot (to free the queue slot).
  - Two data structures (Sec. III-C): GPU Queues (FIFO, signal overflow) and GPU Buffers (circular, overwrite oldest). Freshness over completeness; outputs are labeled dictionaries so heads pick what they need.
  - All queues/buffers PREALLOCATED at start so memory is constant in steady state (R2).
  - Per-head runtime rate control; modules are unsynchronized and lossy by design (Sec. III-A.5).
  - Models compiled to TensorRT independently (FM and each head), or left as PyTorch (Sec. III-E; object-detection head is uncompiled PyTorch).

### 3. Block -> problem -> evidence table
| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| Shared backbone, 8 TRT heads vs 8 independent DAV2 models | redundant backbone compute | 8 tasks: ~39 ms vs ~89 ms = 2.3x (Fig. 3, values read off plot, text says 2.3x) | Orin AGX 64GB, 1080p, 426 imgs | GPU mem per extra task: DAV2 96 MB, seq. heads 48 MB, VPEngine 214 MB (Sec. V-A, Fig. 5a slopes .096/.048/.214 GB/task) |
| Multi-process (VPEngine) vs single-process sequential heads, TRT heads | parallelism | ~equal to sequential (Fig. 3: both ~39 ms at 8 tasks); overhead only 2 ms (Sec. V-A) | TRT kernels already ~100% GPU use, so no parallel gain | CPU avg 438% vs ~70% seq. vs 65% DAV2 (Fig. 5b, text) |
| Multi-process vs sequential with PyTorch (unoptimized) detection heads | utilization when kernels do not saturate GPU | up to 3.3x at 8 heads (~65 vs ~213 ms, Fig. 4) | same | 295 MB GPU/ head vs 5 MB sequential (App. B); each process loads own PyTorch instance ~110 MB; CPU 465% vs 125% |
| CUDA MPS | share GPU across processes | disabling MPS = 43.4% speed loss (Sec. V-B). Table I: 4 tasks 15.24 -> 21.23 Hz (+39.3%); 8 tasks 8.77 -> 15.52 Hz (+77.0%) | PyTorch-detection config | one more point of failure (MPS daemon) |
| Rate limiting (R4) | dynamic prioritization | Table II: delivered frames within +-1 of ideal at 5-30 Hz over 300 images/10 s | 3 heads at same rate | not stated |
| By-reference zero-copy GPU sharing | latency, memory copies | not ablated against a copy baseline | | not ablated |
| Preallocated buffers | OOM / fragmentation | not ablated; constant 1.5 GB observed (Sec. IV) | | not ablated |
| Pre/post transform modules | adapt shapes/dtypes | not ablated | | not ablated |
| Param sharing | memory | combined 27M params, 69% lower than 3 independent models (Sec. IV; footnote 5-6 says this is an approximation) | | |

### 4. Interactions & dependencies
- Multi-process pays off ONLY when heads underutilize the GPU (PyTorch heads). With fully TensorRT-optimized heads the single-process sequential version equals VPEngine within 2 ms (Sec. V-A).
- Needs CUDA MPS (otherwise -43%; and MPS also gives one CUDA context). MPS daemon failure slows but does not crash the pipeline; recommend watchdog (p.2).
- Cost of parallelism scales linearly in CPU and GPU memory with number of tasks (Figs. 5-6): ~214-295 MB/head, which they call "slightly higher" memory; for DINOv2-Giant 4.4 GB (fn.8).
- Freshness design means frames may be dropped (~0.02% in the untuned case, fn.4, "minimal losses ... especially when not compiled with TensorRT").
- Depth head needs intermediate backbone layers (three evenly spaced), so the middle buffer is not just final features.

### 5. Losses
n/a (no training).

### 6. Training recipe
n/a. Pretrained DINOv2 ViT-S [4]; DAV2 [9]; linear segmentation head [4]; Faster R-CNN (custom, PyTorch). Hardware for latency: Jetson AGX Orin 64GB (power mode not stated).

### 7. Results
- Example implementation (Sec. IV): 3 heads = DepthAnythingV2 (TRT), linear semantic seg (TRT), Faster R-CNN detection (PyTorch, NOT TRT). Sustained 30 Hz throughput over 30,000 1080p images.
- Median per-head inference: depth 20 ms, segmentation 18 ms, detection 32 ms (Sec. IV). The 30 Hz is input-rate limited, not a head limit.
- GPU memory constant ~1.5 GB (Sec. IV).
- Combined params 27M; 69% lower than three independent models.
- Scaling: Fig. 3 (8 TRT heads), Fig. 4 (8 PyTorch heads, 3.3x), Fig. 5/6 resource slopes, Table I MPS, Table II rate-limiter. Backbone-only latency is not stated separately. Single-task latency read off Fig. 3: VPEngine ~15 ms vs ~13 ms sequential at 1 task (plot read, approximate).
- No stereo results, no accuracy/mIoU/EPE anywhere.

### 8. Negative results & limitations
- Authors: higher CPU use (multi-process), slightly higher GPU memory per head, may not suit drones; effect on closed-loop behavior depends on controller (Sec. VI). Future: CUDA streams instead of processes, Triton.
- Mine: (1) no stereo head; "depth" is monocular DAV2, so the pattern's cost for a two-view input (shared encoder on left AND right images, correlation volume needs both views' features) is not addressed. (2) 1080p input but backbone resolution after preprocessing (e.g. 518 or 224 resizing) not stated, so per-head ms are not transferable. (3) Head latencies are medians of a 30 Hz-throttled stream; backbone ms not stated separately; power mode not stated. (4) The 2.3x headline compares against 8 full DAV2 models; against single-process TRT sequential heads the gain is ~0 (they say so, Sec. V-A). The real, honest win is memory/compute sharing, not speed. (5) Multi-process results use unoptimized PyTorch heads, i.e. the 3.3x is partly a symptom of non-TRT heads. (6) No accuracy check that TRT-compiled features equal PyTorch features (precision of FM/heads not stated; FP16 vs FP32 not stated). (7) Figure-read numbers above are approximate.
- Existing summary: none available to cross-check.

### 9. Relevance to OUR model
This is our deployment pattern: one frozen YOLO26m trunk (layers 0-6), shared by frozen A09 stereo predictor and frozen 14-class semantic decoder, with a small trained fusion head on top.
- Portable and cheap: (a) TensorRT-compile the trunk and each head as separate engines, FP16, keep trunk features GPU-resident. (b) Preallocate all feature/queue buffers at start (constant memory on a 4 GB RTX 3050 / shared-RAM Jetson). (c) Per-head rate control: run semantics (and the gate) at lower rate than stereo if needed, since semantics change slowly; the "freshness over completeness" policy matters for live demo.
- Do NOT port blindly: multi-process + MPS is meant for heads that underutilize GPU. If A09 + semantic decoder + fusion head are all TensorRT, expect ~0 gain over a single-process sequential graph (their own Fig. 3) and +214-295 MB per process on a 4 GB card. For our topology the better choice is ONE process, one TRT engine (or one CUDA graph / streams) for trunk -> {stereo head, semantic head} -> fusion. Their suggestion of CUDA streams is the right option for us.
- Our dependency graph differs: fusion head consumes BOTH stereo candidates and semantic logits, so the heads are not independent; asynchronous lossy rates risk pairing a stale semantic map with a fresh disparity (misalignment is the C5 failure mode we measured). Use timestamped buffers and only fuse matched frames, or run the semantic branch at the same frame index.
- Stereo-specific: backbone runs twice (left, right) or batched as 2; their paper has no precedent. Batch-2 trunk with one engine is the obvious approach; semantic decoder only on left features.
- Expected benefit: per-task memory/compute saving (they report 69% fewer parameters vs 3 independent models); cost: engineering; risk: TRT conversion of custom ops (correlation, tile/plane refinement, gate) which they did not face (their heads are linear/ViT-style). Novelty: a frozen shared encoder feeding stereo+segmentation+fusion in TensorRT on Jetson is not demonstrated here, so it would be a combination not yet done in this paper.

### 10. Key quotes/equations worth citing
- "zero-copy sharing of visual foundation features ... across multiple specialized task-specific model heads running in parallel ... achieving up to a 3x speedup compared to sequential execution" (Abstract, p.1).
- "disabling CUDA MPS resulted in a 43.4% loss in speed" (Sec. V-B, p.4).
- "VPEngine introduces a negligible overhead of only about 2 ms" (Sec. V-A, p.4).
- "using a shared foundation model provides a significant speedup of up to 2.3x" (p.4); "TensorRT-optimized kernels ... already achieve nearly 100% GPU utilization, leaving little room for parallel execution" (p.4).
- Median inference 20/18/32 ms depth/seg/detection, 30 Hz, ~1.5 GB GPU, 27M params (Sec. IV, p.3).
