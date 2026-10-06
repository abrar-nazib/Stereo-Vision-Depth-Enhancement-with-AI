# Deployment reference: real-time stereo + semantics on RTX 3050 4 GB and Jetson Orin

Scope: one frozen YOLO26m trunk (layers 0-6) run once per stereo pair, feeding a frozen A09 stereo predictor
(HITNet/LightStereo lineage, correlation volume) and a frozen 14-class semantic decoder, plus a small trained
fusion head (semantic cost gate + class residual). Every number carries its source in parentheses. "approx" =
read off a plot. "not stated" = the paper does not say. "derived" = computed by us from cited table values.
No paper below measures a frozen shared detector trunk feeding stereo + segmentation; every transfer is by analogy.

## Measured latency table

Protocol-gap flags: P = power mode not stated, B = batch not stated, W = warmup/runs not stated,
R = resolution not stated, F = framework/precision not stated. TRT = TensorRT.

### Embedded (TX2, Xavier AGX, Orin NX, Orin AGX)

| model | hardware | precision / TRT | resolution | latency / FPS | params | ref | gaps |
|---|---|---|---|---|---|---|---|
| RTS2Net c=8 (stereo+seg) | Jetson TX2 | not stated / not stated | not stated (KITTI-like) | 6.3 FPS (~159 ms derived) | not stated | RTS2Net Tab. II | P B W R F |
| RTS2Net c=1/4/8/16/32 | TX2 | not stated | not stated | 8.3 / 7.4 / 6.3 / 4.5 / 2.3 FPS | not stated | RTS2Net Tab. II | P B W R F |
| RTS2Net c=8 anytime stage 1/2/3 | TX2 | not stated | not stated | 17.2 / 10.9 / 6.3 FPS (58 / 92 / 159 ms derived) | not stated | RTS2Net Tab. IV | P B W R F; unclear if refinement runs before exit |
| RTS2Net, disparity-only c=8 | TX2 | not stated | not stated | 8.1 FPS | not stated | RTS2Net Tab. III | as above |
| DTPnet Setting 1 / 2 / 3 | Jetson AGX (Xavier per card) | not stated; TRT NOT stated for measurement | not stated | 104 / 44 / 16 ms | 2.04 / 0.95 / 0.64 M | DTPnet Tab. II | P B W R F |
| DTPnet (final) | Jetson AGX | not stated | not stated | 16.3 ms | 0.64 M (0.26 M in Tab. VII, unreconciled) | DTPnet Tab. III, Tab. VII | final latency may be pre- or post-prune |
| DTPnet competitors | AGX / TX2 | not stated | not stated | MSN2d 269 ms, StereoVAE 29.8, AnyNet 38.4 (AGX); MADnet 250 (TX2) | not stated | DTPnet Tab. III | TX2 ~4x slower than AGX (DTPnet Sec. V-C) |
| LAS2-S | Orin NX 8 GB | PyTorch eager, no TRT/FP16 stated | 384x1248 | 181 / 144 / 81 ms (10 W / 20 W / MAXN) | not stated | LAS2 Tab. XI | B W; torch.compile off |
| LAS2-M | Orin NX 8 GB | same | 384x1248 | 225 / 179 / 101 ms | not stated | LAS2 Tab. XI, Tab. II | B W |
| LAS2-L | Orin NX 8 GB | same | 384x1248 | 372 / 283 / 166 ms | not stated | LAS2 Tab. XI | B W |
| LAS2-H (4 GRU iters) | Orin NX 8 GB | same | 384x1248 | 689 / 594 / 344 ms | not stated | LAS2 Tab. XI | B W |
| LAS V1 (3D+2D) | Orin NX 8 GB | same | 384x1248 | 469 / 345 / 193 ms | not stated | LAS2 Tab. XI | B W |
| LightStereo-S | Orin NX 8 GB | same | 384x1248 | 229 / 155 / 89 ms | not stated | LAS2 Tab. XI | B W; retrained by LAS2 authors |
| Fast-ACVNet+ | Orin NX 8 GB | same | 384x1248 | 469 / 380 / 221 ms | not stated | LAS2 Tab. XI | B W |
| Fast-FoundationStereo (iter.) | Orin NX 8 GB | same | 384x1248 | 918 ms (MAXN presumed; power not given in Tab. II) | 14.6 M or 17.65 M (inconsistent) | LAS2 Tab. II; Fast-FS Tab. 5/6 | largest Fast-FS variant OOM on Orin (LAS2 Tab. II) |
| PipStereo (1 iter) | Orin NX 16 GB, 25 W | FP32 | 384x1344 | 0.44 s | not stated | Pip-Stereo Tab. 1 | B W |
| PipStereo (abstract headline) | Orin NX 16 GB | FP16 | 320x640 | 75 ms | not stated | Pip-Stereo Abstract | not comparable to Tab. 1 (different precision and size) |
| LightStereo-S / CoEx / BGNet+ / FastACV+ | Orin NX 16 GB, 25 W | FP32 | 384x1344 | 0.13 / 0.17 / 0.16 / 0.27 s | not stated | Pip-Stereo Tab. 1 | B W |
| HITNet(L) / AANet / RT-IGEV++ / IINet | Orin NX 16 GB, 25 W | FP32 | 384x1344 | 0.44 / 0.48 / 0.39 / 0.29 s | not stated | Pip-Stereo Tab. 1 | B W |
| RT-MonSter++ / IGEV / Sel-IGEV | Orin NX 16 GB, 25 W | FP32 | 384x1344 | 0.79 / 1.29 / 1.61 s | not stated | Pip-Stereo Tab. 1 | B W |
| DEFOM / MonSter / FoundationStereo-L | Orin NX 16 GB, 25 W | FP32 | 384x1344 | 5.05 / 7.63 / 14.02 s | not stated | Pip-Stereo Tab. 1 | B W |
| ESMStereo-S / M / L | Orin AGX 64 GB | TRT FP16, trtexec 200 runs | not stated | 91 / 29 / 8.4 FPS (L ~119 ms derived) | 1.8 / 6.3 / 6.8 M | ESMStereo Tab. 7, Tab. 6 | P R B; FP16 = no observable EPE loss (Sec. 5.5) |
| VPEngine: depth / seg / detection heads | Orin AGX 64 GB | TRT (depth, seg), PyTorch (detection); FP16/32 not stated | 1080p input; backbone input size not stated | median 20 / 18 / 32 ms; 30 Hz is input-rate limited | 27 M combined | VPEngine Sec. IV | P; backbone ms not stated; no stereo |
| VPEngine 8 TRT heads, shared vs 8 DAV2 | Orin AGX 64 GB | TRT | 1080p | ~39 vs ~89 ms (approx, Fig. 3; text says 2.3x) | not stated | VPEngine Fig. 3 | P; plot-read |
| RT-MonSter++ | GPU not stated (4090 used for experiments) | not stated | 1248x384 | 47 ms | not stated | MonSter++ Tab. IV | not an embedded measurement |

### Desktop

| model | hardware | precision / TRT | resolution | latency / FPS | params | ref | gaps |
|---|---|---|---|---|---|---|---|
| LightStereo-S / M / L / H | RTX 3090 | not stated | not stated (SceneFlow likely, unverified) | 17 / 23 / 37 / 54 ms | 3.44 / 7.64 / 24.29 / 45.63 M | LightStereo Tab. I | R F B W |
| CoEx | RTX 2080 Ti | not stated | not stated | 27 ms (feature 10 + cost/agg 17) | 2.7 M | CoEx Tab. I, II | R F |
| BGNet / BGNet+ | RTX 2080 Ti | not stated | KITTI15 | 25.4 / 32.3 ms | not stated | BGNet Tab. 4, 5 | F |
| HITNet (KITTI frame) | Titan V | custom CUDA ops | 0.5 Mpix | 19 ms; L 54 ms; XL 114 ms | 0.66 / 0.97 / 2.07 M | HITNet p.16, p.15, Tab. 7 | R for L/XL; CPU TF default 3.3 s/Mpix |
| ESMStereo-S / M / L | RTX 4070 Super | PyTorch 8.6 / 14 / 26 ms; TRT FPS 116 / 71 / 38 | not stated | see left | 1.8 / 6.3 / 6.8 M | ESMStereo Tab. 6, Tab. 7 | R |
| Fast-FoundationStereo | RTX 3090 / 4090 / A100 | PyTorch 49 / 30 / 41 ms; TRT 21 ms (3090) | Midd-Q for timing | see left | 14.6 M (Tab. 5) / 17.65 M (Tab. 6) | Fast-FS Tab. 1, Tab. 5 | no Jetson measured; peak mem 0.63 GB (3090) |
| LAS2-S / M / L / H | RTX 4090 | PyTorch eager | 384x1248 | 11.5 / 16.8 / 23.2 / 26.2 ms | not stated | LAS2 Tab. XI | B W |
| LAS2-S / M / L / H | A5000 | same | 384x1248 | 16.3 / 21.4 / 29.2 / 37.2 ms | not stated | LAS2 Tab. XI | B W |
| LAS2-S / M / L / H | A100 | same | 384x1248 | 22.9 / 29.2 / 41.8 / 65.9 ms | not stated | LAS2 Tab. XI | column order as printed |
| LAS2-S / M / L / H | H200 | same | 384x1248 | 6.6 / 8.1 / 11.4 / 15.1 ms | not stated | LAS2 Tab. XI, Tab. II | B W |
| LAS V1 | GTX 1080 / 4090 / A5000 / A100 | not stated | not stated | 21 / 19 / 23 / 17 ms | not stated | LiteAnyStereo Tab. 7 | R F; the "4K at 21 ms" headline is inconsistent with 33 GMACs at 1242x375 |
| GGEV (ViT-S) | GPU not stated (3090 for training) | not stated | 1248x384 | 47 ms; baseline 30 ms; ViT-L 110 ms | 3.68 M trainable, EXCLUDES frozen DA-V2-S | GGEV Tab. 4 | hardware, precision |
| TwInS Tiny / Base / Large | RTX 4090 | not stated | 640x320 | 22 / 20 / 18 FPS | 67.77 / 137.97 / 275.89 M | TwInS Tab. VI | not real-time on edge; no ms |
| MobileStereoNet 2D / 3D | none | n/a | MACs at 256x512 | NO latency measured; 32.2 / 153.14 GMac | 2.32 / 1.77 M | MobileStereoNet Tab. 5 | MACs only |
| RTX 3050 LAS V1 (local note, NOT from the paper) | RTX 3050 | FP16 | 384x640; 480x768 | 70 ms; 57 ms | not stated | LiteAnyStereo card, errata note | unverified; larger input faster is odd, treat as suspect |

## Operator compatibility checklist (TensorRT / edge SDKs)

Support is SDK-version dependent. DTPnet (2024) says TensorRT lacks good 3D conv, slicing, iterative conv and
trilinear support (DTPnet Sec. I), yet ESMStereo ran a 3D hourglass, top-k and pixel-shuffle under TRT FP16 on
Orin AGX (ESMStereo Tab. 7) and Fast-FS reports TRT 49 -> 21 ms (Fast-FS Tab. 1). Convert a minimal graph early.

| op / pattern | risk | what the paper did instead | evidence |
|---|---|---|---|
| 3D conv on the cost volume | slow or unsupported on edge; dominates compute | disparity as channels, 2D-only aggregation: DTPnet channel-to-disparity 3 convs (0.11 vs 7.16 GFLOPs, EPE +0.16 1.57 -> 1.73); LightStereo inverted residuals on D/4 channels; LAS2 2D-only U-Net | DTPnet Tab. II/IV; LightStereo Tab. II; LAS2 Tab. XI (V1 193 ms -> M 101 ms Orin MAXN; architecture and training changed together) |
| 3D conv kept small | accuracy vs cost | LAS V1 kept ~4.8% 3D proportion; larger 3D kernels did not help | LiteAnyStereo Tab. 2d, 2b |
| Slicing / strided gather | listed unsupported by DTPnet | DTPnet avoids slice and shift loops; BGNet's slice layer (0.74 ms on 2080 Ti) was never tested under TRT | DTPnet Sec. I; BGNet Sec. 3 |
| Trilinear upsampling | listed unsupported | D/4 -> D 1x1 conv then bilinear (DTPnet); BGNet's bilateral-grid slicing is a 4D linear interpolation, TRT status unknown | DTPnet Sec. III-B; BGNet Eq. 1 |
| Shifted correlation / group-wise volume | many small kernels, memory | Fast-FS: left-pad then unfold, one fused multiply-sum, ~6x faster and 3x less memory (3090, Midd-Q, no ablation table) | Fast-FS supp p.15 |
| Gather-heavy warp (per-tile slanted warp) | custom op needed | HITNet used custom CUDA ops (fused init 0.25 ms; warp >100 elementwise ops per tile; without custom op runtime x3). TRT portability "nontrivial" (card) | HITNet p.16 |
| Local cost volume around previous disparity (cascaded search) | gather per pixel | RT-MonSter++ uses it at 1/16 -> 1/8 -> 1/4, no RT ablation | MonSter++ Sec. III-C |
| RNN / ConvGRU loops | control flow blocks fusion, quantization-sensitive, memory-bound | Pip-Stereo: prune 32 -> 1 iteration with regularized init; FlashGRU fused operator (CUDA 11.4 / JetPack 5.3 on Orin NX); GGEV, LAS2-H keep 4-8 iterations | Pip-Stereo Sec. 1, Tab. 3, Tab. 4; LAS2 Tab. XI (H 3.4x M on Orin) |
| Sorting / top-k at full res | prohibitive | CoEx regresses top-k (k=2) at 1/4 res on 48 levels, then learned 3x3 upsample; full-res top-k on PSMNet 292 -> 405 ms | CoEx Sec. 3, Tab. III |
| Top-k swapped in at test time | accuracy loss | must train with the same k (48 -> 2 swap 0.7782 vs 0.6854 trained) | CoEx Tab. III |
| Depthwise / MobileNetV1 blocks | memory-bound, slow despite low MACs | LightStereo chose inverted residual (V2) over V1; V2 is still slower than a regular conv at equal GF | LightStereo Tab. II |
| Attention / reshape-matmul (DDCA, disparity transformer) | memory-bound reshapes, TRT behavior unknown | GGEV DDCA +9 ms (HW x S^2 affinity, S not stated); Fast-FS transformer block replaced by searched blocks | GGEV Sec. Ablation; Fast-FS Sec. 3.2 |
| Pixel shuffle / transposed conv upsample | generally supported | ESMStereo pixel-shuffle + FMBlocks + hourglass converted to TRT FP16 on Orin AGX | ESMStereo Tab. 7 |
| Re-parameterizable blocks | train-only branches | RepViT blocks fused to plain conv at inference | Pip-Stereo Fig. 2 |
| FP16 | numerics | ESMStereo: "no observable degradation in EPE" (SceneFlow) and same KITTI15 zero-shot D1 as FP32 (10.8 / 8.2 / 5.5) | ESMStereo Sec. 5.5 |
| INT8 | not evidenced | not done: DTPnet and Fast-FS list quantization as future work; Pip-Stereo flags GRU sensitivity | DTPnet Sec. VI; Fast-FS p.8; Pip-Stereo Sec. 1 |

Concrete actions for A09 + gate: (1) export trunk, A09 and head as separate ONNX graphs and run trtexec on each
before any integration; (2) inventory A09 for tile warp, gather, argmin-over-d and per-tile loops; (3) keep the
gate to 1x1 conv + sigmoid + broadcast over candidates (CoEx GCE form, CoEx Eq. 1); (4) test FP16 against FP32 on
the held-out split before trusting latency.

## Shared-backbone multi-head runtime engineering

VPEngine (JPL, Orin AGX 64 GB; VPEngine Sec. III-V) is the only systems paper on this pattern. It has no stereo
head, no accuracy metric, and heads are independent (no head-to-head fusion).

| lesson | evidence | transfer to our model |
|---|---|---|
| Shared backbone saves compute and memory, not wall time vs single-process TRT | 8 TRT heads ~39 ms vs 8 full DAV2 ~89 ms (2.3x, approx Fig. 3); multi-process vs single-process sequential TRT: ~equal, overhead ~2 ms (Sec. V-A) | Win is amortising the trunk, not parallelism |
| Multi-process helps only when heads under-utilise the GPU | PyTorch heads: up to 3.3x at 8 heads (~65 vs ~213 ms, approx Fig. 4); TRT kernels "already ~100% GPU utilization" | A09 + decoder + head all in TRT: expect ~0 gain from processes |
| CUDA MPS is needed for multi-process | disabling MPS = 43.4% speed loss; Tab. I: 4 tasks 15.24 -> 21.23 Hz (+39.3%), 8 tasks 8.77 -> 15.52 Hz (+77.0%), PyTorch-detection config | MPS only matters if we keep PyTorch heads in separate processes; MPS daemon is an extra failure point |
| Per-process memory cost | extra GPU memory per task: DAV2 96 MB, sequential heads 48 MB, VPEngine 214 MB (Fig. 5a, Sec. V-A); PyTorch heads 295 MB each vs 5 MB sequential (App. B); CPU 438% vs ~70% sequential (Fig. 5b) | On 4 GB (3050) or shared-RAM Jetson, one process; do not pay ~214-295 MB per head |
| Single process + streams is the authors' own future direction | "CUDA streams instead of processes, Triton" (Sec. VI) | Trunk -> {stereo, semantic} on two CUDA streams in one process, one TRT engine each |
| Zero-copy by-reference GPU sharing, features never leave GPU | custom IPC on `cuMemImportFromShareableHandle` because Jetson lacks native sharing (Sec. III-B); not ablated against a copy baseline | In one process this is free: keep trunk features as device tensors, no host round trip |
| Preallocate all queues/buffers | GPU memory constant ~1.5 GB for 3 heads at 1080p over 30,000 frames (Sec. IV); not ablated | Preallocate trunk-feature, candidate, logit buffers at start; constant memory for 4 GB |
| Freshness over completeness | circular buffers overwrite oldest; ~0.02% frames dropped in untuned case (fn. 4); rate limiter within +-1 frame of ideal 5-30 Hz (Tab. II) | Fine for independent heads; unsafe for ours (below) |
| Backbone exposes intermediate layers | depth head needs three evenly spaced intermediate DINOv2 layers in a middle buffer (Sec. IV) | Fusion head and A09 may need different trunk taps; keep every tap resident |
| Combined-parameter accounting | 27 M combined, 69% lower than 3 independent models (approximation, fn. 5-6) | Report the same way: total incl. frozen parts |

Stale-semantics and timestamp alignment (our topology, not VPEngine's):
- The fusion head consumes BOTH stereo candidates and semantic logits, so lossy asynchronous rates can pair a stale semantic map with fresh disparity. VPEngine's own design is explicitly unsynchronised (VPEngine Sec. III-A.5). Our matched-misalignment control (C5) is the failure mode to avoid (VPEngine card, Sec. 9).
- Rule: tag every trunk-feature buffer with a frame index; the head fuses only matched indices. If the semantic branch runs at a lower rate, pass its map through with an age counter and fall back to the head-off path above an age threshold (threshold: our decision, no paper value).
- Left/right ordering and trunk batch order must be fixed: trunk runs on both views (batch 2); the semantic decoder consumes left features only. No paper has precedent for a two-view shared trunk (VPEngine card, Sec. 8).

RTS2Net per-branch cost split (shared siamese encoder, joint training; RTS2Net Tab. II-IV, derived from FPS):

| component | TX2 (ms, derived) | RTX 2080 Ti (ms, derived) | note |
|---|---|---|---|
| disparity-only net, c=8 | ~123 (8.1 FPS) | ~10.4 (96.2 FPS) | RTS2Net text: disparity subnet ~120 ms of ~160 ms (Sec. IV-B) |
| + semantic decoder, joint training | +~28 (6.6 FPS) | +~2.6 (76.9 FPS) | ~18% of TX2 total; stereo dominates |
| + synergy refinement | +~7-8 (6.3 FPS) | +~3.6 (60.4 FPS) | costs 1.99 mIoU (62.22 vs 64.21) for -0.58 D1; no no-semantics control (RTS2Net Tab. III) |
| total | ~159 (6.3 FPS) | ~16.6 (60.4 FPS) | params / FLOPs / memory not stated |

Anytime / early-exit designs (all supervise intermediate outputs so exits are usable):
- RTS2Net: 3 coarse-to-fine stages, D1 8.00 / 4.70 / 3.33 at 17.2 / 10.9 / 6.3 FPS on TX2 (Tab. IV); stage weights 1/4, 1/2, 1 (Eq. 2).
- Iteration count knobs: Pip-Stereo 32 -> 1 (see Where the time goes); GGEV 2 / 4 / 6 / 8 iterations EPE 0.54 / 0.49 / 0.47 / 0.46 at 35 / 39 / 44 / 47 ms (GGEV Tab. 6); LAS2 S/M/L/H and ESMStereo S/M/L are size knobs, not run-time exits.
- Ours: the fusion head is the last stage, so "head off" is a free exit that returns A09's disparity. Expose it as a runtime flag; do not re-architect A09 to emit intermediate outputs.

## Where the time goes

| paper | split (ms unless noted) | shares | ref |
|---|---|---|---|
| LightStereo-S (3090) | feature 10.39, cost 1.98, aggregation 3.98, regression 1.48 = 17.83 | backbone 58% (derived) | LightStereo Tab. VI |
| LightStereo-H (3090) | backbone 27.2 of 54 | ~50% (derived) | LightStereo Sec. IV (card interactions) |
| ESMStereo S / M / L (4070S) | feature 6.85 / 10.86 / 11.80; cost volume 0.08 / 0.62 / 6.22; aggregation 0.10 / 0.25 / 3.18; ESM upsampler 1.57 / 2.27 / 4.81; total 8.60 / 14.00 / 26.01 | feature 80% / 78% / 45% (derived); ESM 18% / 16% / 18% (derived) | ESMStereo Tab. 5 |
| CoEx (2080 Ti) | feature 10, cost volume + aggregation + regression 17, refine none, total 27 | feature ~37% (derived) | CoEx Tab. II |
| BGNet+ (2080 Ti, KITTI15) | feature 8.8, cost build + aggregation 12.2, bilateral grid 4.3, refinement 7.0 = 32.3 | feature 27% (derived) | BGNet Tab. 5 |
| HITNet (Titan V, 0.5 Mpix) | extractor 6, init 0.25, last 3 propagation steps 7.5, total 19 | propagation ~39% (derived) | HITNet p.16 |
| RTS2Net c=8 (TX2) | disparity subnet ~120 of ~159 ms | ~75% | RTS2Net Sec. IV-B |
| Fast-FoundationStereo (3090) | teacher ~0.50 s: feature ~0.10, cost filtering ~0.10, refinement ~0.29 (approx, Fig. 10); student ~0.05 s, each stage >10x faster; per-stage student ms not stated | refinement ~58% of teacher (approx) | Fast-FS Fig. 10, p.8 |
| Fast-FS refinement (3090, Midd-Q) | 1 iteration ~7 -> ~3.5 ms after 0.6 pruning; ratio-0.6 model at 2 / 4 / 8 / 16 iterations ~33 / 38 / 46 / 62 ms (approx) | marginal ~2 ms per extra iteration (derived, approx) | Fast-FS Fig. 9, Fig. 12 |
| GGEV (hardware not stated, 1248x384) | baseline 30 ms; +DFE 7 ms; +SCF 1 ms; +DDCA 9 ms = 47 ms; ViT-L 110 ms | frozen depth prior +8 ms, aggregation +9 ms | GGEV Tab. 4, p.6 |
| GGEV iterations | 2 / 4 / 6 / 8 iterations = 35 / 39 / 44 / 47 ms, i.e. ~2 ms per iteration (derived); card quotes ~3.7 ms, not reproduced from Tab. 6 | | GGEV Tab. 6 |
| Pip-Stereo (Orin NX 25 W FP32, 384x1344) | 32 / 16 / 8 / 4 / 2 / 1 iterations = 3.40 / 1.87 / 1.02 / 0.68 / 0.51 / 0.44 s with SceneFlow EPE 0.38 / 0.38 / 0.40 / 0.41 / 0.43 / 0.45 | fixed cost (encoder + volume + 1 iter) ~0.44 s; ~0.095 s per extra iteration (derived) | Pip-Stereo Tab. 3 rows 6-11 |
| Pip-Stereo baseline | Selective-IGEV 1 / 12 / 32 iterations: EPE 0.64 / 0.44 / 0.44 at 0.60 / 1.61 / 3.57 s | direct cut 32 -> 1 EPE 0.56 vs PIP 0.45 | Pip-Stereo Tab. 3 rows 1-3, 5 |
| FlashGRU operator (RTX 4090, 4x feature map) | 320x736: 28.8 -> 14.8 ms; 640x1472: 45.2 -> 15.2; 1280x2944: 122.4 -> 16.8; peak mem 743 -> 561 / 1431 -> 715 / 4105 -> 957 MiB | benefit grows with resolution, marginal at 1 iteration; no Orin profiling | Pip-Stereo Tab. 4 |
| MonSter++ | 356.1 M params = 335.3 M frozen mono ViT-L + 12.6 M stereo + 8.2 M SGA/MGR; 0.64 s (32 it) vs IGEV 0.37 s; 0.34 s at 4 iterations | frozen prior dominates | MonSter++ Sec. IV-G, Tab. X |
| LAS2 power scaling (Orin NX 8 GB) | MAXN -> 20 W -> 10 W: S 81 -> 144 -> 181; M 101 -> 179 -> 225; L 166 -> 283 -> 372; H 344 -> 594 -> 689 | 20 W is 1.7-1.8x MAXN (H 1.7x); 10 W is 2.0-2.2x (derived) | LAS2 Tab. XI |
| LAS2 iterations on Orin | H (4 iterations) 344 vs M 101 ms MAXN; H200 15.1 vs 8.1 ms | 3.4x on Orin vs 1.9x on H200 (derived): loops hurt edge more | LAS2 Tab. II |

Reading: for 2D real-time nets the feature extractor is 27-80% of latency, so amortising a frozen trunk over
two views plus the semantic branch targets the largest block. Our A09 stage and head costs are not measured here.

## Compression recipes

| recipe | steps and settings | evidence | caveats |
|---|---|---|---|
| DTPnet distill then prune | logits KD: L1 on softmax(logits/t) over 192 bins, t 0.5 -> 1.0, teacher-only supervision best (EPE GT 1.73 / GT+teacher 1.61 / teacher-only 1.56, Tab. V); DepGraph structural pruning, L2 importance, r=0.1 x E=5 steps = 50% params, finetune after each step (Algorithm 1, Sec. V-A) | architecture cuts 104 -> 44 -> 16 ms AGX for EPE 1.27 -> 1.48 -> 1.73 (Tab. II); hourglass removal +3.31 EPE (Tab. IV); KL 1.86 vs L1 1.78 (Tab. VI) | pruning is NOT ablated (no pruned vs unpruned EPE or latency); params 0.64 / 0.63 / 0.26 M and FLOPs 6.25 / 4.00 / 6.30 / 3.677 G inconsistent across tables; text claims lowest EPE but Tab. VII has MSN3d 0.80, DeepPruner 0.97 lower; TRT not stated for latency; feature-level KD called infeasible, no numbers |
| Fast-FS distill, NAS, prune, retrain | (1) backbone MSE distill ViT+CNN to one CNN; (2) blockwise NAS: each candidate block trained to mimic the teacher block on teacher features, ILP picks per latency budget (space 5.5e24, 2584 blocks trained, search 14 days on 128 A100, Delta tau -0.04 s); (3) refinement pruning ratio 0.6 by first-order Taylor + end-to-end retrain with feature distill (gamma 0.9, lambda 0.1); (4) 1.4M pseudo-labels; (5) end-to-end | 496 -> 49 ms (3090), TRT 21 ms; MSE distill vs none Midd-H BP-2 2.20 vs 2.87 (Tab. 3); searched blocks beat 10 random at all budgets (Fig. 8); prune-only BP-2 ~2% -> ~11% at 0.8 vs prune+retrain stays ~2% (Fig. 9, approx) | direct pruning of cost filtering fails (BP-2 2% -> ~55% at 0.2, Fig. 11, approx); error roughly doubles vs teacher (Midd-H BP-2 1.10 -> 2.20); no Jetson measurement; quantization not done; Tab. 5 vs 6 params inconsistent |
| Pip-Stereo iteration pruning | PIP halves iterations per stage (32 -> 16 -> 8 -> 4 -> 2 -> 1), update block only finetuned 50K steps, loss L_cum + L_final + L_hid, Fi-RNN initialised from Mi-RNN | 1 iteration 0.45 EPE vs 0.56 by direct cut (Tab. 3); updates <1% of pixels by iteration 32 (Fig. 1) | needs regularized (non-zero) initial disparity; RAFT zero-init fails (EPE 2.16 at 1 iteration, Tab. 3 row 18-19); appendix missing in PDF; headline latency mixes FP16 320x640 and FP32 384x1344 |
| FlashGRU | fused sparse-aware GRU operator | 1.94x / 2.97x / 7.28x at 320x736 / 640x1472 / 1280x2944 on 4090; Orin NX end-to-end ~1.23x (Tab. 3 rows 12-13) | irrelevant for a single-pass head; CUDA 11.4 JetPack 5.3 build |
| Fused GWC volume | left-pad + unfold + single fused multiply-sum | ~6x runtime, 3x memory (Midd-Q, 3090) | supplement only, no table; zero accuracy cost claimed |
| FP16 evidence | TRT FP16 on Orin AGX | no observable EPE degradation on SceneFlow; same KITTI15 D1 as FP32 (ESMStereo Sec. 5.5) | conv/correlation + 3D hourglass net only; VPEngine does not state precision |
| MACs do not equal latency | | LightStereo (3090): regular 3x3 36.27 GF 16.7 ms; V1 depthwise 34.9 GF 54.2 ms; V2 inverted-residual 35.82 GF 22.9 ms; EfficientViT 34.5 GF 51.1 ms (Tab. II); EffNetV2 103 GF 46.8 ms vs RepViT 50.5 GF 28.7 ms (Tab. III). LAS2 (Orin NX 8 GB): ConvNeXt 31.2 G 162 ms; MobileNetV2 33.9 G 107; FasterNet 47.6 G 101 (Tab. V). MobileStereoNet reports MACs only, 7-19x block MAC cut, no latency | LAS2 "MACs alone do not reliably reflect real inference speed"; LAS2 ranking changes between 3090-class and Orin |
| Pseudo-label adaptation (training-only, zero inference cost) | LAS2: M_LR (1 px) x M_edge x M_sky mask, clamp tau=10; Fast-FS: normal-consistency mask, >60% valid | LAS2 Stage 3 K12 D1 4.21 -> 2.88 (Tab. IX); Fast-FS pseudo-labels help SceneFlow-only baselines most (Tab. 4) | LAS2 changed architecture and recipe together; teacher bias; useful for the head only |

## Budget rules for our model

1. Measure on device, in the lowest power mode you ship. LAS2 latency is 1.7-1.8x worse at 20 W and 2.0-2.2x at 10 W than MAXN (LAS2 Tab. XI, derived); Pip-Stereo reports 25 W FP32 (Pip-Stereo Tab. 1). VPEngine and ESMStereo do not state power mode, so their Orin AGX numbers are optimistic upper bounds for an NX or Nano.
2. Never port a desktop number to the RTX 3050. LAS2-M ranges 8.1 ms (H200) to 21.4 ms (A5000) (LAS2 Tab. XI); a 3050 is slower than the A5000 (our inference). The only 3050 figures are an unverified local note (LiteAnyStereo card errata). Budget by measured per-stage ms of trunk, A09, decoder, head.
3. Keep the fusion head single pass, no recurrent loop. Iteration is the costliest edge pattern: LAS2-H 3.4x M on Orin (344 vs 101 ms), Pip-Stereo 0.44 s fixed + ~0.095 s per iteration at 384x1344 FP32 (derived), GGEV ~2 ms per iteration on a desktop GPU (derived, Tab. 6). RTS2Net's final refinement added only ~7-8 ms on TX2 (derived, Tab. III).
4. Use pointwise 1x1 fusion convs and broadcast gates. GGEV: 1x1 fusion K12 D1 4.11 vs 3x3 4.59 (4.18 M params, 47 ms) vs DW+PW 4.73 (GGEV Tab. 7); CoEx 1x1 + sigmoid broadcast over disparity adds ~1 ms (CoEx Sec. 6); LightStereo MSCA +0.2 ms (LightStereo Tab. IV). Prior-keyed spatial kernels (GGEV DDCA) cost +9 ms and have unknown TRT behavior.
5. Avoid 3D convs on the 1/4 volume on Jetson. LAS2 dropped them for 2x on Orin NX (LAS2 Tab. XI); DTPnet replaced them for 7.05 GFLOPs saved at +0.16 EPE (DTPnet Tab. IV). If a 3D block is unavoidable, keep it narrow (ESMStereo 1/4 volume, j=16 channels, L variant 8.4 FPS on Orin AGX, ESMStereo Tab. 7).
6. Amortise the trunk across both views and the semantic branch: one TRT engine at batch 2, semantic decoder on left features only, one process, two streams, preallocated GPU buffers (VPEngine Sec. III, V; 214-295 MB per extra process). Target the largest block: feature extraction is 27-80% of 2D real-time nets (table above), but our own trunk share is unmeasured.
7. Ship a head-off path and per-stage timers. The head is the cheap stage (RTS2Net +~28 ms for the whole semantic branch, stereo ~120 ms, on TX2); the frozen stereo model dominates. Report fused vs head-off ms, peak VRAM and frame-index alignment, so the gate cost can be judged against the 1/4-volume cost (RTS2Net Tab. III, Tab. IV).
8. Fuse only frame-matched inputs. VPEngine's freshness-over-completeness policy suits independent heads (VPEngine Sec. III-A.5); our gate needs matched semantic and disparity indices (misalignment is our C5 failure).
9. Report parameters and memory including frozen parts. GGEV's 3.68 M counts trainable only and excludes the frozen DA-V2-S (GGEV card errata); MonSter++ 356.1 M is 335.3 M frozen prior + 20.8 M trained (MonSter++ Sec. IV-G); VPEngine reports 27 M combined (VPEngine Sec. IV). Give trainable, frozen and total, plus peak VRAM (references: Fast-FS 0.63 GB at Midd-Q on 3090, VPEngine ~1.5 GB for 3 heads at 1080p, LiteAnyStereo 2.5 GB for 2K input).
10. State the evaluation protocol with every latency: resolution, precision, TRT yes/no, batch, power mode, warmup/runs. Pip-Stereo's FP16 320x640 75 ms and FP32 384x1344 0.44 s are not interchangeable. If deployment resolution differs from the native-pixel comparison protocol in AGENTS.md, label it a separate deployment config and never mix its numbers with the ablation tables.
