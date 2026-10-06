<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# Lite Any Stereo card

Source: paper/reference_papers/lightweight/LiteAnyStereo_Jing_arXiv2025.pdf (11 pp: 8 body + refs). NOTE: the PDF does NOT include the supplementary material, although the text refers to it (perturbation recipe for Stage 2, full/quarter-res Middlebury). Those items are "not stated" here.

### 0. Meta
- Title: Lite Any Stereo: Efficient Zero-Shot Stereo Matching. Jing, Luo, Mao, Mikolajczyk (Imperial College London).
- Venue: arXiv 2511.16555v2, 16 Mar 2026.
- Code/project: https://tomtomtommi.github.io/LiteAnyStereo/ (title page); repo link not stated separately.
- Domain: general zero-shot (driving, indoor, in-the-wild). Eval: KITTI 2012/2015, ETH3D, Middlebury (half-res, non-occ), DrivingStereo weather.

### 1. Problem & failure modes targeted
- Belief that efficient models cannot generalize zero-shot (Abstract/Intro). Target: ultra-light model with real-world zero-shot ability, avoiding the cost of DepthAnything-style priors (DA-S overhead "prohibitive", Sec. 3.1).
- Sim-to-real gap (supervised synthetic only), domain-specific KITTI overfitting of prior light nets.
- 2D-only aggregation lacks structured continuity along the disparity axis (Sec. 3.1).
- Qualitative claims: reflections, non-texture, repetitive texture, blurry boundaries (Fig. 7).

### 2. Pipeline by stage
- 2a Features: two weight-shared ImageNet-pretrained MobileNetV2 (channel config chosen over ConvNeXt/other, Sec. 3.1), pyramid 1/4,1/8,1/16,1/32, all upsampled to 1/4 with residual upsampling blocks "following [20]" (LightStereo). Trained, not frozen. No DA/prior features.
- 2b Semantic/prior branch at inference: n/a (explicitly rejected). Priors enter only at TRAINING time via the Stage-3 teacher (FoundationStereo) (see 2h).
- 2c Cost volume: correlation, C(d,h,w) = 1/Nc <F_L^{1/4}(h,w), F_R^{1/4}(h,w-d)> (Eq. 1), d in [0, Dmax/4], Dmax = 192 (Sec. 4.2) -> 48 disparity levels at 1/4 res.
- 2d Aggregation: hybrid 3D-then-2D serial: C_agg = G2D(G3D(C)) (Eq. 2). G3D = multi-scale 3D convs (kernel (3,3,3)), G2D = ConvNeXt layers (disparity folded to channels). Only a small 3D proportion kept: 4.8% of the aggregation compute (Tab. 2d). Exact block counts/channels not stated in main text. Four layouts compared (Fig. 5): bilateral (BANet), 2D+3D, 3D+2D (chosen, "(c)"), interleaved.
- 2e Disparity: soft-argmax at 1/4 (Eq. 3).
- 2f Refinement: none (feed-forward, no iterations).
- 2g Upsampling: convex upsampling 1/4 -> full (Fig. 3, Sec. 3.1).
- 2h Fusion points (all training-time, none at inference):
  (i) Stage 2 self-distillation: feature alignment L_feat = 1 - 1/HW sum cos(F_i, F'_i) between a FIXED teacher copy (clean input) and student (perturbed input); direction teacher->student, at feature level (which layer: not stated).
  (ii) Stage 3 pseudo-label distillation: frozen accurate stereo model (FoundationStereo, a depth-foundation-prior model) labels 0.5M unlabeled real pairs (even replaces sparse GT of DrivingStereo with dense pseudo labels). Loss-only coupling.
  A semantic cue could enter: (a) as extra channels into G2D (2D ConvNeXt block, cheap), or (b) as an additional distillation target; the paper has no such block.

### 3. Block -> problem -> evidence table
All Tab. 2 numbers: models trained 200K iters (text says 100K in 4.3 -- inconsistency) without augmentation on 1.4M synthetic subset; metrics K12 D1 / K15 D1 / ETH3D Bad1 / Mid Bad2; MACs at 1242x375.
| block | problem | evidence | conditions | cost |
|---|---|---|---|---|
| 2D only (BANet-2D-style default) | baseline | 5.02/5.01/6.48/11.29 (Tab. 2a) | | 32.9 GMACs |
| Bilateral (2D+3D parallel sum) | | 5.10/5.10/8.55/12.00 worse (Tab. 2a) | | 35.8 G |
| 2D-3D serial | | 4.73/4.44/28.85/11.52 (ETH3D collapses) | | 33.9 G |
| 3D-2D serial (chosen on cost-fixed budget; final model used ConvNeXt+3D 4.38/4.84/5.75/9.50) | structured disparity continuity + efficiency | 4.78/4.64/5.39/10.89 (MobileNetV2 2D layer) (Tab. 2a) | | 33.9 G |
| Interleaved | | 4.61/4.73/6.20/11.34 | | 35.2 G |
| 3D kernel (3,3,3) chosen | | (5,3,3) 4.65/4.88/6.29/10.36; (7,3,3) 4.84/4.85/6.37/9.64; (11,3,3) 4.86/4.87/6.87/10.39; (3,1,1) 4.70/4.85/6.63/10.06 (Tab. 2b) | | 32.1-32.8 G |
| 2D layer: ConvNeXt (chosen) | | 4.38/4.84/5.75/9.50 vs MobileNetV2 4.78/4.64/5.39/10.89 vs ConvNeXtV2 4.79/4.81/5.03/10.52 (Tab. 2c) | | 32.8 / 33.9 / 34.0 G |
| 3D proportion 4.8% (chosen) | 3D dominates compute | 4.8%: 4.38/4.84/5.75/9.50; 9.5%: 4.71/4.51/5.81/10.06; 15.6%: 4.49/4.68/5.54/10.34 (Tab. 2d) | fixed MAC budget | ~32.7 G |
| Stage-2 strategy: none / data aug / knowledge distillation | domain-invariant features | 4.38/4.84/5.75/9.50 ; 4.31/4.79/5.73/11.14 ; 3.64/4.63/6.82/8.86 (Tab. 2e) | | training-only |
| Teacher update: EMA / hard copy / fixed | | EMA 3.97/4.82/6.91/9.78; hard copy 4.22/4.71/6.35/10.07; fixed 3.64/4.63/6.82/8.86 (Tab. 2f) | | |
| Three-stage on full train set | | S1 4.05/4.55/4.43/8.49; S2 3.66/4.53/4.69/7.03; S3 3.04/3.87/3.53/7.51 (Tab. 2g) | | |
| Stage 3 applied to LightStereo-M | | S1 4.34/5.27/6.68/10.29; S2 3.80/4.62/5.44/8.96; S3 3.35/4.14/4.22/9.85 (Tab. 6) | | |
| Stage 3 applied to BANet-2D | | S1 4.34/4.78/7.71/10.54; S2 3.87/4.80/4.86/9.54; S3 3.28/4.08/4.05/10.30 (Tab. 6) | | |
| Residual-upsample + convex upsampling, soft-argmax, correlation volume | | not ablated | | |
| Backbone MobileNetV2 vs others | | claimed better matched channel config; no table | | not ablated here |
| Dataset curation (exclude Stereo4D, HRWSI, SCOD; use pseudo labels over sparse GT) | data quality > scale | text only, no table (Sec. 3.2) | | not ablated numerically |

### 4. Interactions & dependencies
- Self-distillation helps K12/Mid but slightly hurts ETH3D (4.43 -> 4.69) (Tab. 2g); the benefit of data aug vs KD flips on Middlebury (11.14 vs 8.86) -> distillation cleaner than augmentation.
- Teacher weights must be FIXED (EMA/hard-copy worse) (Tab. 2f).
- Stage 3 requires a strong frozen teacher (FoundationStereo); no self-distillation in Stage 3 ("no observable gains").
- Naive half-half 3D/2D blend is ineffective since 3D dominates compute; only ~5% 3D works (Tab. 2d). 2D-3D order collapses on ETH3D (28.85) -> order matters; serial 3D-first.
- Wider disparity-axis 3D kernels do not help here, contradicting FoundationStereo's findings (Sec. 4.3).
- Middlebury Bad2 gets worse in Stage 3 (8.49->7.51 in Tab. 2g improved; but for LightStereo-M/BANet-2D 8.96->9.85 and 9.54->10.30) -> real indoor data scarce.

### 5. Losses
- Stage 1 and 3: L_disp = smoothL1(D - D_gt) (Eq. 4) at full res; in Stage 3 D_gt = pseudo label from FoundationStereo. No weights listed.
- Stage 2: L_disp + L_feat, L_feat = 1 - 1/HW sum_i cos(F_i, F'_i) (Eq. 5); relative weight not stated in main text.
- No multi-scale / deep supervision stated.

### 6. Training recipe
- Hardware: NVIDIA A100s, total batch size 176. AdamW, one-cycle LR, peak 2e-4. Steps: Stage 1 150K, Stage 2 50K, Stage 3 100K (Sec. 4.2). Crop 256x512, then fine-tune at 320x736. Dmax 192.
- Stage 1 (1.8M synthetic labeled: SceneFlow 35K, FallingThings 30K, FSD 1.1M, CREStereo 0.2M, VKITTI2 21K, TartanAir 0.31M, Dynamic Replica 0.14M), NO augmentation. Excluded IRS, Sintel, Spring, InfinigenSV.
- Stage 2: teacher/student both init from Stage 1; teacher clean + FIXED, student gets strong perturbations (details in supplementary, not available).
- Stage 3: 0.5M unlabeled real pairs (Tab. 1: Flickr1024 1K, InStereo2k 2K, Holopix50K 49K, DrivingStereo 174K (weather subset excluded), SouthKenSV 113K, UASOL 156K); pseudo labels from FoundationStereo.
- Data counts: Fig. 4 says 1.9M in one panel and 1.8M elsewhere (diagram text inconsistency).

### 7. Results
- Zero-shot, SceneFlow-only block (Tab. 3; "stage 2 on SceneFlow"): Lite Any Stereo D1 K12 5.45 / K15 6.45 / ETH3D bad1 15.38 / Mid bad2 13.13 at 33 GMACs; LightStereo-M 6.76/6.79/13.93/16.99 at 33G; LightStereo-L 6.80/6.62/9.66/17.23 at 84G; Lite-CREStereo++ 5.93/7.37/8.95/14.91 at 101G. (Note: ETH3D worse than several baselines.)
- Million-scale block: Lite Any Stereo 3.04/3.87/3.53/7.51 (EPE 0.79/0.99/0.32/0.94); LightStereo-M* (retrained on same data) 4.10/4.97/5.33/10.85; BANet-2D* 3.90/4.71/5.92/10.05; StereoAnything-L (30M pseudo-labeled) 4.00/4.81/3.81/9.82 at 84G. Accurate refs: Selective-IGEV 3.20/4.50/3.40/7.50 at 3619G; FoundationStereo 2.51/2.83/0.49/1.12 at 12824G. Claim "< 1% MACs" is vs Selective-IGEV.
- DrivingStereo weather, overall D1/EPE (Tab. 4): 8.74/1.80 vs FoundationStereo 10.71/2.22 (teacher beaten; rainy 20.69 vs 27.01). Caveat: the student's Stage-3 data includes DrivingStereo (non-weather) pseudo-labelled, so partially in-domain.
- KITTI leaderboards (Tab. 5): KITTI 2012 3-all 1.49; KITTI 2015 D1-all 1.71 (bg 1.36, fg 3.45) at 33 GMACs, fine-tuned trained on [54] (KITTI depth completion) without original annotations. Best among efficient, vs BANet-3D 1.77 @78G, LightStereo-L 1.93.
- Runtime (Tab. 7, local, same setup; input size not stated): GTX 1080 21 ms, RTX 4090 19, A5000 23, A100 17. LightStereo-M 24/22/27/21; LightStereo-L 34/30/35/27; BANet-2D 58/30/63/27. Memory 2.5 GB for 2K input (Sec. 4.5). No Jetson/Orin, no TensorRT/FP16 statement.
- Params: not stated in main text.

### 8. Negative results & limitations
- Authors: still trails depth-prior-based approaches; limited high-quality real data; Middlebury accuracy drops in Stage 3; transparency/reflection robustness can improve (Sec. 4.5).
- Things tried and dropped: DA-S features (too costly); half-half 3D/2D hybrid; larger 3D kernels; EMA/hard copy teacher; self-distillation in Stage 3; Stereo4D (18M, 512x512), HRWSI (rectification), SCOD (narrow); IRS/Sintel/Spring/InfinigenSV.
- My concerns: (1) headline "4K Stereo Depth at 21 ms on GTX 1080" (Fig. 1) is inconsistent with 33 GMACs at 1242x375 (4K is ~19x more pixels) and Tab. 7 gives no resolution, treat as unverified; (2) first-block fairness: baselines trained on SceneFlow only vs a Stage-2 recipe; second block retrains baselines on their data, but their pseudo-label Stage 3 recipe is not applied to teachers' architecture-specific tuning; (3) Stage-3 uses FoundationStereo as teacher so "zero-shot" has a heavy-prior dependency (dependence moved to training); (4) the Tab. 2 text says 100K iters, caption says 200K; (5) supplementary missing from PDF; (6) no semantics, no params count; (7) ETH3D weak in the SceneFlow-only regime.

### 9. Relevance to OUR model
- The paper is a recipe for getting "free" priors without inference cost: Stage-3 distillation from a frozen foundation stereo model. For us, the frozen A09 + frozen semantic decoder are already trained; only the fusion head trains. A Stage-3-like step could train the D2/E3 head on unlabeled real stereo (e.g. KITTI raw / DrivingStereo) with pseudo labels from a big teacher, directly attacking our "VKITTI-only, KITTI zero-shot gap" (F-series: 2.59 px). Cost: zero at inference; risk: pseudo-label bias, need a teacher that runs offline (FoundationStereo is 12.8 TMACs, offline-only). Not done in our repo -> novel for our design. Also the observation that real pseudo-labels beat sparse GT.
- Self-distillation with perturbed student + cosine feature alignment (Stage 2, 50K steps) could be applied to the trainable fusion head's gate features, but our trunk is frozen so the representation-robustness gain is limited to the head. Low priority.
- 3D-then-2D tiny-3D-proportion aggregation is interesting for A09 refinement, but our gate acts on candidates after the cost volume; an inexpensive 3D conv pass at 1/4 with 48 levels is plausible but we have no ablation evidence for a frozen A09.
- Semantic cue entry points they leave open: concat semantic logits/features into G2D ConvNeXt layers (cheap, 2D). Our SemanticCostGate already does a gating form.
- Real-time budget: 33 GMACs / 17-21 ms on desktop GPUs; no embedded measurement, so cannot be used to predict RTX 3050/Jetson latency beyond ratio to LightStereo-M (24 vs 21 ms: ~12% faster).

### 10. Key quotes/equations worth citing
- "an ultra-light model can deliver strong generalization ... less than 1% computational cost" (Abstract).
- "A naive half-half hybrid design is ineffective, as 3D convolutions dominate the compute budget ... we retain only a small proportion of 3D component" (Sec. 3.1, p.4).
- Eq. 2 C_agg = G2D(G3D(C)); Eq. 5 L_feat = 1 - 1/HW sum cos(F_i, F'_i).
- "data quality is more critical than scale" (Sec. 3.2, p.5).

### Errata / notes vs existing summary (summaries/lightweight/LiteAnyStereo.md)
- Summary reproduces "21 ms at 4K on GTX 1080" (Fig. 1 caption); I flag it as inconsistent with 33 GMACs at 1242x375 and an unspecified Tab. 7 resolution.
- Summary contains a local RTX 3050 fp16 measurement (70 ms at 384x640, 57 ms at 480x768, tested 2026-05-02), which is NOT from the paper; useful for our budget but unverified here.
- Supplementary material is not in the repo PDF; any summary claim about the perturbation recipe cannot be verified from it.
