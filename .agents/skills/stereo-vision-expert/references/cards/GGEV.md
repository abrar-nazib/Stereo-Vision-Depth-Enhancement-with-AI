<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# GGEV card

Source: paper/reference_papers/lightweight/GGEV_Liu_AAAI2026.pdf (11 pp: 7 body + refs + 2 pp supplementary, all read; Fig. 4 viewed as image). Page refs: body p.1-7, Supplement p.10-11 (supplement tables: Tab. 6-8, Figs. 9-10).

### 0. Meta
- Title: Generalized Geometry Encoding Volume for Real-time Stereo Matching. Liu, Xu, Wang, Zhang, Yang* (HUST).
- Venue: AAAI 2026 (arXiv 2512.06793v1, 7 Dec 2025).
- PDF: paper/reference_papers/lightweight/GGEV_Liu_AAAI2026.pdf
- Code: https://github.com/JiaxinLiu-A/GGEV (Abstract).
- Domain: general; driving + indoor zero-shot. Datasets: Scene Flow (Finalpass; 35,454/4,370), KITTI 2012/2015, ETH3D, Middlebury (quarter-res, Bad 2.0).

### 1. Problem & failure modes targeted
- Real-time stereo overfits in-domain and generalizes poorly (zero-shot), especially in occlusions, textureless, repetitive patterns, thin structures (Intro, Fig. 3).
- Stereo foundation models (MonSter, DEFOM, FoundationStereo) generalize but are slow, and need scale-shift handling for mono affine depth.
- Specific diagnosis of geometry-encoding volumes: (1) the critical regions differ per disparity hypothesis; (2) matching relations in those regions are fragile in unseen domains. Hourglass aggregation treats all hypotheses uniformly -> mismatches, edge blur, detail loss.
- Reflective regions (KITTI 2012 reflective, Tab. 5).

### 2. Pipeline by stage
Baseline is a simplified RT-IGEV (aggregation depth reduced to two downsampling stages, 8 iterations) (Sec. Ablation).
- 2a Features: (i) Texture encoder: ImageNet MobileNetV2, siamese (left and right), scales 1/4, 1/8, 1/16 (Ci channels not stated), trained. (ii) Depth Feature Encoder: FROZEN Depth Anything V2 Small, LEFT image only, multi-scale features at 1/2, 1/4, 1/8, 1/16 (which ViT layers / DPT head taps: not stated; C_i not stated). Uses ViT-S in the main model; ViT-L variant tested (Tab. 4).
- 2b Semantic/prior branch: the depth encoder above (depth, not semantics). Frozen. Left only. Prior enters as features, not as depth values, so no scale-shift problem.
- Selective Channel Fusion (SCF): concat(f_l,i, f_d,i) -> 1x1 conv -> depth-aware prior features f_da,i, i in {1/4,1/8,1/16}. "Selective" = a plain 1x1 conv (no explicit gating nonlinearity stated). Not frozen.
- 2c Cost volume: group-wise correlation at 1/4 between texture features: C(g,d,x,y) = 1/(Nc/Ng) <f_l^g(x,y), f_r^g(x-d,y)> (Eq. 1), Ng = 8 groups, d in {0..D/4-1}. D (max disparity): not stated in main text. Texture-only (depth features do not enter the correlation).
- 2d Aggregation (DDCA, Fig. 4, Eq. 2-6): operates per disparity hypothesis plane. Inputs: plane C_d in R^{G x H x W} (d = disparity index; G groups) and f_da in R^{C x H x W}. Q = Reshape(W_q C_d) in R^{C x HW} (1x1 conv); K = Reshape(W_k Pool(f_da)) in R^{C x S^2} (adaptive avg-pool of the depth-aware feature to S x S regional centers; S value not stated); affinity A = Q^T K in R^{HW x S^2} (Eq. 4). Split channels into G groups -> G affinity matrices A^g. Linear W_m maps A^g -> kernel weights M^g in R^{HW x K^2}, softmax-normalised (Eq. 5): a spatially varying KxK kernel per pixel per group; channels of the fused features [C_d ; f_da] are split into G groups sharing a kernel; group-wise dynamic convolution C'_d = C_d * M^g_dynamic(C_d, f_da) (Eq. 6). Combination of large and small kernels for low/high frequency (sizes K not stated). The independently aggregated planes are re-stacked -> "generalized GEV" C'. Inspired by OverLoCK context-mixing dynamic kernels. Cost: +0.03M params, +9 ms (Sec. Ablation).
- 2e Disparity: soft-argmin on C' at 1/4 -> d0 (Eq. 7).
- 2f Refinement: single-layer ConvGRU (Eq. 8-11). Hidden state h0 initialised from f_da,4 (depth prior injected into recurrence). Input x_k = concat(d_k, f_G) where f_G are geometry features looked up from C' at d_k. Residual decode with two convs: d_{k+1} = d_k + Delta d_k. 11 iterations in training, 8 at inference (Sec. Implementation).
- 2g Upsampling: conv(h_k) -> upsample to half-res -> concat with depth feature f_d (1/2 scale from left) -> weight map W in R^{H x W x 9} -> convex-style 9-neighbour weighted combination of low-res d_k.
- 2h FUSION POINTS (all depth prior -> stereo, one direction, only left image):
  1. SCF: feature level, concat+1x1 conv with texture features, at 1/4,1/8,1/16.
  2. DDCA: aggregation stage; depth features supply the KEY of an affinity with each disparity hypothesis plane (query) -> dynamic kernels weights; also concatenated into the filtered tensor (so depth also acts as content).
  3. GRU hidden-state initialisation from f_da,4.
  4. Upsampling weight prediction uses f_d at 1/2 scale.
  Cost volume itself is untouched by depth (no depth in correlation), and no loss coupling to the depth model (no distillation, no depth loss).

### 3. Block -> problem -> evidence table
Tab. 4 (200k steps on Scene Flow, 4 zero-shot datasets; D1 for KITTI, Bad2.0 Middlebury, Bad1.0 ETH3D; time at 1248x384; hardware not stated; params = trainable only). Columns: SF EPE / K12 D1 / K15 D1 / Mid / ETH3D / params / time.
| block | problem | evidence | conditions | cost |
|---|---|---|---|---|
| Baseline (simplified RT-IGEV) | | 0.54 / 6.63 / 8.01 / 7.84 / 7.54 | | 3.60M, 30 ms |
| +DFE (DA-V2-S, plain, no SCF/DDCA) | domain-invariant structure | 0.52 / 4.40 / 6.32 / 6.47 / 5.02 (K12 -2.23, ETH3D -2.52 abs.) (Tab. 4) | frozen ViT-S, 200k SF | 3.57M, 37 ms (+7) |
| +DFE+SCF | fusion texture+depth | 0.49 / 5.14 / 7.58 / 5.57 / 4.85: in-domain better, KITTI zero-shot WORSE than +DFE (K12 4.40->5.14, K15 6.32->7.58), Mid/ETH3D better; paper: "mixed effect on generalization" | | 3.65M, 38 ms |
| +DCA only (texture-guided dynamic aggregation, no depth) | hypothesis-specific aggregation | 0.47 / 6.80 / 6.75 / 7.73 / 5.61: in-domain gain, generalization limited ("texture features sensitive") | | 3.63M, 39 ms |
| Full ViT-S (DFE+SCF+DDCA) | | 0.46 / 4.11 / 5.56 / 6.53 / 2.84 | | 3.68M, 47 ms |
| Full ViT-L | prior quality | 0.45 / 4.20 / 5.65 / 4.41 / 1.69 | | 3.75M, 110 ms; Middlebury and ETH3D big gain, KITTI none |
| SCF conv type (Tab. 7, supp.) | | 3x3 conv: 0.47/4.59/6.04/6.35/3.31 (4.18M, 47 ms); 3x3 DW+1x1 PW: 0.46/4.73/5.84/6.87/3.04 (3.69M, 48 ms); 1x1 conv (chosen): 0.46/4.11/5.56/6.53/2.84 (3.68M, 47 ms) | | spatial mixing in fusion hurts generalization |
| MFM swap (Tab. 7) | | MoGe-2 ViT-S: 0.48/4.36/6.25/6.96/3.80, 75 ms; DA-V2 ViT-S 0.46/4.11/5.56/6.53/2.84, 47 ms | | all GGEV variants still beat RT-IGEV per text |
| Affinity construction (Tab. 8, supp.) | what conditions the kernels | Q=DH,K=DH, no depth: 0.47/4.58/3.52 (K12 D1 / ETH3D); Q=DF,K=DH: 0.46/4.33/3.04; Q=DH,K=DF (chosen): 0.46/4.11/2.84 | | |
| # GRU iterations (Tab. 6, supp.) SF test | | 2: 0.54 (35 ms); 4: 0.49 (39); 6: 0.47 (44); 8: 0.46 (47) vs RT-IGEV 0.50 (40), BANet-3D 0.51 (30) | | iteration cost ~3.7 ms each |
| Fig. 7 | | consistently better than RT-IGEV at same #iterations in-domain and zero-shot (curves only) | | |
| Reflective regions KITTI12 (Tab. 5) | | GGEV 2-noc/2-all/3-noc/3-all 7.33/9.27/4.04/5.34 vs RT-IGEV 9.56/11.54/5.76/7.26, BANet-3D 9.64/11.97/5.37/7.07, HITNet 9.75/11.85/5.91/7.54, RAFT-Stereo 8.41/9.87/5.40/6.48 | | |
| Depth-aware GRU hidden init, depth-aware upsampling, large+small kernel mix, group count Ng/G, pooling size S | | not ablated | | |
| Training with extra synthetic data (SF + CREStereo + TartanAir) | | GGEV 3.6/4.7/5.7/2.2 vs RT-IGEV 4.0/5.4/8.6/3.4 (Tab. 1) | | |

### 4. Interactions & dependencies
- DDCA's gain requires the depth prior: texture-guided dynamic kernels alone give in-domain gain but no generalization (Tab. 4, +DCA row) -- a built-in capacity/ prior-matched-ish control.
- SCF alone can hurt KITTI zero-shot; DDCA on top restores and surpasses (K12 5.14 -> 4.11 with DDCA). So the fusion is order/combination dependent. Pointwise (1x1) fusion generalizes better than 3x3.
- Prior quality drives indoor zero-shot (ViT-L: Middlebury 6.53->4.41, ETH3D 2.84->1.69) but not KITTI (4.11 -> 4.20, 5.56 -> 5.65): in driving, the S model already saturates; ~63 ms extra for L.
- Affinity query must come from the cost-plane (DH) and key from depth; swapping loses (K12 4.33 vs 4.11).
- The DFE trainable-param count (+0.05M) hides the frozen DA-V2-S size (not stated in this paper) and its compute (+7 ms); "trainable params 3.68M" understates the real model.
- Latency measured at 1248x384; GPU not stated.

### 5. Losses
- L = SmoothL1(d0 - d_gt) + sum_{i=1..N} gamma^{N-i} ||d_i - d_gt||_1, gamma = 0.9, N = 11 training iterations (Eq. 13). No depth/mono loss, no distillation, no regularizer.

### 6. Training recipe
- RTX 3090s; AdamW, gradient clipping to [-1, 1], one-cycle LR (peak LR, epochs/steps of main SF pretraining not stated; ablation variants 200k steps). SF crop 320x768, batch 12, asymmetric chromatic + spatial augmentation. 11 training iters / 8 at inference.
- KITTI finetune: SF-pretrained, 50k steps, batch 8, mixed KITTI12+15. ETH3D: 300k steps, batch 8, crop 384x512, mix of TartanAir, CREStereo, SF, Sintel, InStereo2k, ETH3D; then 100k more on CREStereo+InStereo2k+ETH3D.
- Frozen: DA-V2 encoder (always). Trained: MobileNetV2 (ImageNet init), SCF, DDCA, GRU, upsampler.
- Zero-shot protocol: train on SF only (following MonSter) and test on 4 real datasets; plus second setting SF + CREStereo + TartanAir.

### 7. Results
- Zero-shot (Tab. 1, SF-only; error % thresholds 3px KITTI, 2px Mid-quarter, 1px ETH3D) K12/K15/Mid/ETH3D: GGEV 4.1/5.5/6.5/2.8; RT-IGEV 5.8/6.6/7.8/5.8; BGNet+ 5.3/6.6/11.2/10.3; IINet 11.6/8.5/-/-; Fast-ACVNet 12.4/10.6/13.5/7.9; CoEx 13.5/10.6/14.5/9.0; DeepPrunerFast 16.8/15.9/18.3/11.0; accuracy models: RAFT 4.5/5.7/9.3/3.2; DEFOM 3.7/4.9/5.6/2.3; DEFOM ViT-S 4.2/5.3/6.3/2.6; FoundationStereo 3.2/4.9/-/1.8. Claimed reductions vs RT-IGEV 29%/16%/16%/51% (recomputed: 29%, 16%, 17%, 52% -- OK).
- KITTI 2015 online (Tab. 2): D1-all 1.70 (bg 1.38, fg 3.28), 47 ms; KITTI 2012 3-noc 1.10, 3-all 1.44, 2-noc 1.66. Compare RT-IGEV 1.79 @40 ms; BANet-3D 1.77 @30 ms; HITNet 1.98 @20 ms; CoEx 2.02 @27 ms. Accuracy: RAFT 1.82 @380, IGEV 1.59 @180, Selective-IGEV 1.55 @240, MonSter 1.41 @450.
- ETH3D (Tab. 3): GGEV Bad 0.5/1.0/2.0 3.70/1.19/0.34, AvgErr 0.14; HITNet 7.83/2.79/0.80/0.20; Fast-ACVNet 14.25/5.62/1.41/0.31; Selective-IGEV 3.06/1.23/0.22/0.12.
- Scene Flow EPE 0.46 at 8 iters, 47 ms (Tab. 6); 0.54 at 2 iters, 35 ms.
- Timing: "run-time measured at KITTI resolution 1248x384" (Tab. 4); Hardware for all times not stated (training on RTX 3090). No Jetson/Orin/TensorRT. "HITNet's real-time constraint of 100 ms" is used as the bar. GGEV with ViT-S is 81% faster than DEFOM ViT-S (255 ms).

### 8. Negative results & limitations
- Authors: SCF alone has mixed generalization effect; texture-guided DCA alone limited; ViT-L latency 110 ms; future work: metric depth foundation models, video.
- My assessment: (1) no Jetson/embedded latency despite "real-time" claim, and the frozen DA-V2-S (~25M params; not given in paper) dominates memory/compute -- trainable-param count (3.68M) is misleading; (2) the zero-shot table mixes the real-time and accuracy blocks with different training sets; (3) first 3 ablation rows lack a capacity-matched control for the DFE branch (extra trainable ViT-S-sized prior vs random frozen features); (4) 4 benchmark zero-shot tests share training on Scene Flow only -- good control but ETH3D/Middlebury are tiny (27/15 pairs) with high variance and no seeds; (5) depth-aware GRU init/upsampling untested; (6) S, K, Ng, G, D, C_i not stated; (7) Fig. 3/10 qualitative claims about "filtering mismatches" are single examples; (8) inference iterations (8) vs RT-IGEV settings: matched by iteration curve only.

### 9. Relevance to OUR model (exhaustive analogy to SemanticCostGate)
Our D2/E3: frozen shared YOLO trunk + frozen A09 candidates + frozen semantic decoder; trainable SemanticCostGate + ClassResidual.
- Direct structural analogues:
  (a) SCF <-> our fusion of trunk features and semantic features before gating. GGEV: 1x1 concat fusion, pointwise only; 3x3 spatial mixing was WORSE zero-shot (K12 4.59 vs 4.11; Tab. 7) and heavier. Action: ensure our gate's fusion layer is pointwise; our E-series 2x/4x widths may have added spatial convs -- check; if spatial, an equal-param 1x1 variant is a cheap ablation. Risk low.
  (b) DDCA <-> SemanticCostGate. Differences: DDCA conditions per-hypothesis planes through (query = cost plane, key = pooled prior features), outputs spatially-varying dynamic KxK kernels that filter the plane (edge/boundary-aware aggregation), cost +0.03M params / +9 ms. Our gate multiplies/selects candidates by semantic class conditioning (no spatial kernel). Port: replace our semantic multiplicative gate input with an affinity-driven dynamic kernel on candidate/disparity planes, with the prior = semantic features (YOLO decoder features or class-embedding map) pooled to SxS centers. Insertion: after the frozen A09 volume/candidates, before final selection. Expected benefit: boundary-respecting aggregation without RGB residuals (our G/H edge residuals failed, which acted post-hoc; DDCA is pre-regression), and generalization via hypothesis-specific context. Cost: HW x S^2 affinity (S small, e.g. 8) per group; at 1/4 res fits a RTX 3050, risk moderate (memory-bound reshape/matmul, TensorRT friendliness unknown). Novel combination: semantic-keyed DDCA is not done in this paper, and no semantic work in the lightweight set does it; it is novel relative to our D/E gate.
  (c) Query/key ablation (Tab. 8): conditioning on the prior is what generalizes (K12 4.58 -> 4.11, ETH3D 3.52 -> 2.84), and DH-query/prior-key beats the reverse. For our gate: ensure semantics act as KEY/context and the cost/candidate plane as query, and keep an equal-capacity no-semantics control (which the paper effectively had in "+DCA" row: K12 6.80 vs 4.11).
  (d) GRU init from f_da <-> initialise our ClassResidual state/queries with semantic features (we have no GRU; analog = feed semantic embedding as the initial residual feature). Not ablated in GGEV: no evidence.
  (e) Depth-aware upsampling with half-res prior features <-> we tested convex upsampling (no gain). GGEV gives no evidence either, skip.
- Second prior: GGEV shows a frozen depth-FM feature prior is a very strong generalizer (K12 D1 6.63 -> 4.40 with DFE alone) beyond our semantic gain (EPE 3.08 -> 2.59 KITTI15 for E3). A DA-V2-S feature branch as a second cue into the same gate (semantic + depth features, SCF-style) is the obvious extension. Cost: ~7-8 ms on a 3090 at 1248x384 (unknown on RTX 3050/Jetson) -> likely violates our real-time budget; mitigate with Pip-Stereo-style offline distillation of DA features into our shared trunk head (training-only), or run DA-V2-S at reduced res. Risk: DA-V2 domain mismatch plus compute; must be tested with matched controls.
- Iterations: our model has none; GGEV shows cost ~3.7 ms/iter, which is a reason to keep single-pass.
- Not portable: GRU-centric refinement; the group-wise correlation we may already have; MobileNetV2 texture encoder (we use frozen YOLO).

### 10. Key quotes/equations worth citing
- "treating all disparity hypotheses uniformly fails to fully leverage the potential of the depth features" (Ablation, p.6).
- "Selective Channel Fusion ... a lightweight 1x1 convolution ... preserving structural details and avoiding spatial blurring" (p.3).
- Eq. 4-6: A = Q^T K; M^g = softmax(A^g W_m); C'_d = C_d * M^g_dynamic(C_d, f_da).
- Eq. 13: L = |d0-d_gt|_smoothL1 + sum_i gamma^{N-i} ||d_i - d_gt||_1, gamma 0.9.
- "full model (ViT-S) introduces only a marginal increase of 0.08M (+2%) trainable parameters ... 17 ms over the baseline, including 8 ms from DFE+SCF and 9 ms from DDCA" (p.6).

### Errata vs existing summary (summaries/lightweight/GGEV.md)
- Summary says "Speed comparable to DEFOM-Stereo (ViT-S) while reducing ETH3D error by 81%": the paper says accuracy comparable to DEFOM ViT-S (255 ms) with 81% LESS INFERENCE TIME; the 81% is not an ETH3D error number.
- Summary says "~48 ms range on edge GPUs" and "inference time on 3090": the paper reports 47 ms at 1248x384 but does not state the GPU or any edge device.
- The 3.68M "trainable params" excludes the frozen DA-V2-S encoder; the summary should not be read as total model size.
