<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# Pip-Stereo card

Source: paper/reference_papers/lightweight/Pip-Stereo_Zheng_CVPR2026.pdf (10 pp: 8 body + refs). The PDF contains NO appendix, although the text repeatedly says "refer to Appendix" (supernet search configs, block sequence, more training details). Those are "not stated" below. Fig. 2 viewed as image.

### 0. Meta
- Title: Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching. Zheng*, Liu, Xu, Chen (ARIDGE / XPENG).
- Venue: CVPR 2026 (arXiv 2602.20496v1, 24 Feb 2026).
- Code: "will be updated at https://github.com/XPENG-Aridge-AI" (Abstract). No stated release.
- Domain: general/driving; train BTS (MonSter protocol): SceneFlow, CREStereo, TartanAir, Sintel, FallingThings, InStereo2K (Sec. 4.1); supernet search set 10% FoundationStereo Dataset (FSD). Eval: SceneFlow, ETH3D, KITTI 2012/2015, zero-shot DrivingStereo weather (Sec. 4.1).

### 1. Problem & failure modes targeted
- Iterative (GRU) stereo cannot be deployed on edge: control-flow loops hinder operator fusion, RNN is sensitive to quantization noise, high memory-bandwidth demand at high-res (Sec. 1). FLOPs/params do not capture this.
- Non-iterative real-time methods have poor generalization/robustness when fine-tuned to subdomains (claim, Sec. 1).
- Mono-prior stereo (MonSter, DEFOM, FoundationStereo) needs a DA encoder at inference: MonSter ~7.6 s per 384x1344 frame on Orin NX (Sec. 2).
- Observation (Fig. 1, Middlebury): GRU updates are spatially sparse (<1% of pixels updated by iter 32) and temporally redundant; hit ratio vs previous iteration >0.99 from iter ~10 for IGEV, ~15 for RAFT.

### 2. Pipeline by stage
Built on Selective-IGEV (Sec. 4.1) with RepViT student encoder.
- 2a Features: student encoder = RepViT-style re-parameterizable blocks (training branches 3x3 DW, 1x1 DW, SE, 1x1; fused to a plain conv at inference; Fig. 2). Initialized from RepViT M2-3, expanded to a supernet (blocks cloned, with/without SE, at four resolutions), 10K-step warmup, then genetic-algorithm search (10K generations, early-exit) on 10% FSD to pick per-resolution block counts (details in missing appendix). The search gives 5% additional EPE reduction (0.40 -> 0.38 -type gain; Tab. 3 rows 4 vs 6). Multi-scale features incl. 1/4 used for cost volume and context net. Not frozen during stage 1; frozen during stage 2 (only GRU trained).
- 2b Semantic/prior branch: monocular teacher = Depth Anything V2-L (frozen, Fig. 2 snowflake) inside a teacher stereo pipeline (teacher also has cost volume, 3D regularization, GRU update branch; Fig. 2 top row). Student has NO mono encoder at inference; teacher is dropped. Stage 1 length 200K steps.
- 2c Cost volume: inherited from Selective-IGEV (geometry encoding volume: group-wise correlation + 3D regularization, "3D Regularization Features"); resolution/levels/channel counts not stated in this paper.
- 2d Aggregation: 3D regularization network of the IGEV family (not detailed). Stage-1 alignment taps its output embeddings.
- 2e Disparity: initial disparity from regularized volume (IGEV-style "regularized initialization", non-zero) plus GRU residual updates Delta d. Pruned model: 1 iteration.
- 2f Refinement: ConvGRU (Selective GRU in Selective-IGEV) with T = 24 used in the PIP derivation example (Sec. 3.3), 32 iterations for the unpruned PipStereo-i32. PIP halves iterations per stage: 32 -> 16 -> 8 -> 4 -> 2 -> 1 (p1..p5; "pk has 32/2^k iterations", with k=1..4 named, PipStereo = p5). FlashGRU is an alternative for non-regularized-init cases.
- 2g Upsampling: "before 4x upsampling" (Sec. 4.4) -- method not stated.
- 2h Fusion points:
  (i) Mono prior -> student at TRAINING only, as MSE feature alignment: (a) multi-resolution contextual features of the encoders, (b) cost-volume/3D-regularization embeddings. Direction teacher -> student. No inference-time fusion.
  (ii) Stage 2 iteration distillation: unpruned GRU (Mi-RNN) -> pruned GRU (Fi-RNN) by output-trajectory, final-disparity and hidden-state losses (Eq. 3-7).
  Semantic cue could enter: as a third feature-alignment teacher (semantic features -> student encoder, same MSE form) or by adding semantic features to the GRU hidden state init; the paper does neither.

### 3. Block -> problem -> evidence table
Tab. 3 (SceneFlow EPE; latency in seconds on Orin NX FP32 384x1344 per Tab. 1 note; ETH3D Bad-1 Noc for rows 20-22):
| block | problem | evidence | conditions | cost |
|---|---|---|---|---|
| Selective-IGEV baseline | | 12 it: EPE 0.44, 1.61 s; 32 it: 0.44, 3.57 s; 1 it: 0.64, 0.60 s (rows 1-3) | | |
| MPT (mono prior transfer), no search | ZERO-iter/ill-posed accuracy | 32 it 0.44 -> 0.40 (row 4) ; overall "13.6% EPE reduction" (rows 1-5, text) | | 3.40 s at 32 it |
| MPT with supernet search (PipStereo-i32) | | 1 it: 0.56 vs baseline 0.64 (row 5 vs 3, -12.5%); 32 it: 0.38 (row 6); search = extra 5% (text) | | 1 it 0.44 s (vs 0.60 s: RepViT encoder is faster) |
| PIP progressive pruning, p1..p5 | accuracy loss from fewer iters | 16 it 0.38 (1.87 s); 8 it 0.40 (1.02 s); 4 it 0.41 (0.68 s); 2 it 0.43 (0.51 s); 1 it 0.45 (0.44 s) (rows 7-11); directly cutting i32 to 1 it: 0.56 (+0.18, +32.1%) vs PIP 0.45 (+0.07, +15.5%) | | latency 3.40 -> 0.44 s |
| FlashGRU P2/P3 | GRU memory-bound latency | P2-Flash 8 it 0.43 EPE 0.76 s (vs 1.02 s); P3-Flash 4 it 0.43, 0.55 s (vs 0.68 s) (rows 12-13), ~1.23x end-to-end | Orin NX FP32 | |
| PIP on RAFT-Stereo (zero-init) | | 32 it 0.72/2.95 s; 1 it 3.65/0.86 s; 4 it 2.22/1.07 s; Raft-p5 (1 it) 2.16; Raft-p3 (4 it) 1.43/1.07 s; Raft-p3-Flash 1.47/0.69 s (rows 14-19) | | PIP fails for zero-init |
| PIP on FoundationStereo | | ETH3D Bad-1: 32 it 0.38 (14.02 s); 1 it 0.94 (7.14 s); Founda-p5 0.70 (+0.32 vs +0.56 unfinetuned) (rows 20-22) | | |
| FlashGRU operator (RTX 4090, Tab. 4, 4x feature map, vs native ConvGRU of Selective-IGEV) | | 320x736: 28.8 -> 14.8 ms (1.94x), peak mem 743 -> 561 MiB (-24.4%); 640x1472: 45.2 -> 15.2 ms (2.97x), 1431 -> 715 (-50.0%); 1280x2944: 122.4 -> 16.8 ms (7.28x), 4105 -> 957 MiB (-76.6%), mem requests -80.9% | RTX 4090 CUDA 12.1 | no Orin profiling (unified memory) |
| Re-param RepViT blocks | edge latency | not ablated separately | | |
| Teacher choice DA-V2-L | | not ablated (only DA-V2-L tried) | | |
| Alignment layers (context vs cost-volume), MSE weight | | not ablated | | |
| 70% sparsity / importance-map thresholding | | not ablated (Fig. 3 shows 70% mask, top 30%) | | |

### 4. Interactions & dependencies
- PIP requires a regularized (non-zero) initial disparity (IGEV/Selective-IGEV/FoundationStereo). With zero-init (RAFT) it fails: EPE stays >2 at 1 iteration; needs ~4 iterations + FlashGRU (Sec. 4.3).
- MPT must come first: PIP finetunes ONLY the update block with the encoder frozen (Stage 2, 50K steps).
- Hit-ratio plateau (~10 iterations for IGEV) predicts the pruning degradation curve: 32->16 nearly free, further cuts cost progressively (rows 7-11).
- FlashGRU's speedup is "marginal when only a single iteration remains", so it is only relevant when >1 iteration is retained; benefit scales with resolution and is bounded by on-chip memory / L2 size.
- Supernet search and MPT are coupled (search is done with the teacher alignment).

### 5. Losses
- Stage 1: MSE feature alignment teacher->student on multi-resolution contextual features and cost-volume (3D regularization) embeddings; weights not stated. Plus standard stereo disparity supervision (implied, not stated; Selective-IGEV's sequence loss presumably).
- Stage 2 (PIP): L = L_cum + L_final + L_hid (Eq. 7), equal weights.
  - L_cum = sum_{s=1..S} || sum_{k<=s} d^Fi_k - sum_{k<=s} dbar^Mi_k ||^2 (Eq. 4), with block-aggregated Mi outputs dbar^Mi_s = (1/r) sum_{i=1..r} Psi(z^Mi_{r(s-1)+i}) (Eq. 3).
  - L_final = || d^Fi_S - Psi(z^Mi_T) ||^2 (Eq. 5).
  - L_hid = sum_s || z^Fi_s - z^Mi_{rs} ||^2 (Eq. 6).
  - r = 2 compression per stage, S = T/r (Eq. 1-2); Fi-RNN initialized from Mi-RNN weights.
- Whether GT disparity loss is also used in Stage 2: not stated.

### 6. Training recipe
- 8x RTX 4090 (24 GB). Stage 1: warm-up supernet 10K steps on BTS; freeze, GA search 10K generations on 10% FSD; transfer mono prior 200K steps, lr 2.5e-4, wd 1e-5, batch 24; then task finetune ETH3D/KITTI 100K steps. Stage 2: PIP finetune update block only, 50K steps, lr 2e-4, batch 64 (per halving stage? not stated). Zero-shot models: both stages on the full BTS (as MonSter++). Crop/augmentation/optimizer type: not stated.
- FlashGRU: CUDA 11.4 on JetPack 5.3 (Orin NX), CUDA 12.1 benchmark on 4090; Nsight Compute 2024.07.
- Efficiency profiling: Jetson Orin NX 16GB LPDDR5.

### 7. Results
- Orin NX: abstract: 75 ms at 320x640 FP16; 19 ms on RTX 4090. Tab. 1 latency (s) at 384x1344, Orin NX 25 W, FP32 (not FP16): PipStereo 0.44 s; LightStereo(S) 0.13; CoEx 0.17; FastACVNet+ 0.27; IINet 0.29; BGNet+ 0.16; HITNet(L) 0.44; AANet 0.48; RT-IGEV++ 0.39; RT-MonSter++ 0.79; IGEV 1.29; Selective-IGEV 1.61; DEFOM 5.05; MonSter 7.63; FoundationStereo-L 14.02. So Pip is NOT faster than most 2D real-time nets at that resolution/precision; claim is accuracy at similar latency.
- In-domain (Tab. 1), PipStereo 1 iter: SceneFlow EPE 0.45; ETH3D Bad-1 Noc 0.35 / All 0.67, RMSE 0.19; KITTI 2012 Out-2 Noc/All 1.60/1.65, Out-3 0.92/0.94, KITTI 2015 D1-bg/all Noc... shown as 1.20 1.49 (KITTI12 D1-bg Noc / D1-all Noc?) and 1.33 1.44 (KITTI15 D1-bg/D1-all All); exact column mapping garbled in the text extraction -- check the table image before quoting KITTI numbers. Comparison EPE SceneFlow: MonSter 0.37, DEFOM 0.42, SelectiveIGEV 0.44, IGEV 0.49, LightStereo(S) 0.73, HITNet(L) 0.55.
- Zero-shot DrivingStereo D1-all (Tab. 2) Sunny/Cloudy/Rainy/Foggy/Avg: PipStereo 3.27/2.69/7.71/3.76/4.35 (1 iter, 0.44 s); MonSter++ 2.60/2.12/3.08/2.94/2.69 (7.63 s); RT-MonSter++ 3.18/2.86/5.90/4.76/4.18 (0.79 s); RT-IGEV++ avg 8.04; IGEV 7.80; FoundationStereo 4.93; DEFOM 5.48; LightStereo(S) 13.08 (0.13 s); CoEx 22.95; IINet 27.70; HITNet(L) 93.52 (reported). Training data not identical across rows (non-iterative: SceneFlow-only; RT-* mixed; others copied from MonSter++ paper).
- Speed claims: "22x faster than MonSter, 14x than DEFOM, 41x than FoundationStereo-L": from Tab. 1 latencies the ratios are 17.3x, 11.5x, 31.9x -- not reproduced by the table (inconsistent, possibly different latency basis).

### 8. Negative results & limitations
- Authors: PIP degrades for zero-init (RAFT) methods; aggressive iteration cuts raise error; FlashGRU benefits vanish at 1 iteration; no Orin memory profiling; large resolution needed for FlashGRU gains.
- Weak/unfair: (1) headline latency mix: FP16 320x640 75 ms (abstract) vs Tab. 1 FP32 384x1344 0.44 s; (2) zero-shot comparison uses unequal training data per row (acknowledged in text); the claim "iteration is indispensable for generalization" is confounded by training data, not an ablation; (3) EPE improvements 13.5% over "second-best" mixes iterative and real-time; (4) no appendix in PDF so search results unverifiable; (5) no params/FLOPs reported; (6) 3D regularization + group-wise correlation volume of IGEV remains (cost volume cost not addressed; "beyond scope"); (7) the mono-prior benefit is not separated from the RepViT encoder change in latency (row 3 vs 5 latency 0.60 vs 0.44 s).

### 9. Relevance to OUR model
- Our A09/E3 setup already differs: no GRU, single pass; so PIP/FlashGRU are not applicable. But two lessons transfer.
  1. Prior transfer without inference-time prior encoder (MPT): align student features/cost embeddings to a frozen teacher via MSE. Our frozen shared YOLO trunk already provides semantics at zero extra encoder cost; the analogue for us is aligning the A09 cost/aggregation embeddings (or the fusion head's internal features) to a depth-foundation teacher (DA-V2) during head training -- a "mono-depth feature distillation into the head" with zero inference cost. Unclear benefit when stereo is frozen (we only train the head); a variation: distill DA-V2 relative depth into the ClassResidual branch as a loss-only coupling. Novel vs our E3? Yes (loss-only mono prior), moderate risk (scale ambiguity; they also had to co-update teacher/student).
  2. Deployment realism: FP32 Orin NX 384x1344 = 0.13-0.44 s for these networks; our frozen YOLO26m trunk + A09 + gate needs a measured Orin number; target "real-time" requires lower res or FP16/INT8. Use Pip's 320x640 FP16 75 ms point as a budget reference (for an iterative GRU model with 1 iteration).
- The FlashGRU sparse-update idea (importance map picks top 30% pixels; only these get updated) resembles gating of refinement to uncertain regions: our SemanticCostGate could provide a semantic/uncertainty mask to restrict ClassResidual to a sparse set of pixels (boundaries, small classes) to cut compute. Plausible but our residual is cheap; sparse gather/scatter may hurt TensorRT. Low priority.
- Observation that updates are <1% of pixels supports our finding that edge residual modules (G, H series) add little: dense refinement over mostly-converged disparity gives tiny gains.

### 10. Key quotes/equations worth citing
- "refinement activity is highly sparse and overwhelmingly redundant ... By the time the 32 iterations are reached, the set of updated pixels dwindles to less than 1%" (Sec. 1, p.2).
- "implicitly embeds depth priors without requiring a dedicated monocular encoder" (Abstract).
- Eq. 4-7: L = L_cum + L_final + L_hid.
- "non-iterative real-time methods exhibit markedly inferior generalization ... even when both are engineered under comparable latency constraints" (Sec. 1, engineering observation, not an ablation).

### Errata vs existing summary (summaries/lightweight/Pip-Stereo.md)
- Summary's "11.5x faster than DEFOM-Stereo" matches the Tab. 1 latencies (5.05/0.44) but not the paper's text claim of 14x (and 22x vs MonSter, 41x vs FoundationStereo-L, vs 17x and 32x from the table). The paper is internally inconsistent; use the table.
- Summary says teacher is "frozen"; only the DA-V2-L mono model is frozen (Fig. 2); the teacher stereo branch carries gradient-updated parts and the text says teacher and student "are co-updated in tandem".
- Appendix (supernet block config, search results) is absent from the PDF in the repo.
