<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# MonSter / MonSter++ block card

IMPORTANT PROVENANCE NOTE (verified against the PDF): the file `MonSter_Cheng_CVPR2025.pdf` is arXiv 2501.08643**v2 (25 Sep 2025), titled "MonSter++: Unified Stereo Matching, Multi-view Stereo, and Real-time Stereo with Monodepth Priors"** (IEEE-style journal extension, 17 pp, no supplement). It is NOT the CVPR 2025 MonSter paper. The original CVPR MonSter ("Marry Monodepth to Stereo Unleashes Power", ref [7] in the PDF, pp. 6273-6282) appears only as a citation. Consequently every number below is MonSter++ (large) or RT-MonSter++; the paper never gives separate original-MonSter numbers, so "original MonSter numbers: not stated in this file". The summary in summaries/fusion/MonSter.md mixes these (see Sec. 8 errors list). MonSter++ adds over the original: multi-view-stereo variant, RT-MonSter++, 2M-pair FTS training, DrivingStereo test (all described as new/extended in the abstract/Secs. III-B,C, IV-E,F).

### 0. Meta
- Title: MonSter++ (v2 of "MonSter: Marry Monodepth to Stereo Unleashes Power"); authors Junda Cheng*, Wenjing Liao* et al. (HUST, Meta, Autel Robotics); venue as in file: IEEE-format arXiv preprint, v2 Sep 2025 (original MonSter = CVPR 2025).
- PDF: paper/reference_papers/fusion/MonSter_Cheng_CVPR2025.pdf. Code: https://github.com/Junda24/MonSter-plusplus (p.1, "will be released").
- Domain: general (driving KITTI/DDAD/DrivingStereo, indoor Middlebury/ETH3D). Datasets: Scene Flow, KITTI 2012/2015, Middlebury, ETH3D, DDAD, KITTI Eigen, DrivingStereo; BTS (Scene Flow, FSD, CREStereo, TartanAir, Sintel, FallingThings, InStereo2k) and 2M-pair FTS (+3DKenBurns, DynamicStereo, IRS, VA, Booster, Carla-Highres) (Sec. IV-A).

### 1. Problem & failure modes targeted
- Ill-posed matching: occlusion, textureless, repetitive/thin structures, distant objects (low pixel proportion), reflective regions (Sec. I; Tab. V reflective, Tab. VI edge/non-edge, D1-bg for distant).
- Scale-shift ambiguity of monocular depth: even after global least-squares scale+shift to GT, DepthAnythingV2 disparity is still noisy (Fig. 3(b)); unidirectional mono->stereo fusion injects noise on slanted/curved surfaces (p.3). Framing: "per-pixel scale-shift recovery of mono depth using stereo cues".
- Zero-shot generalization (Scene Flow -> real) and real-time deployment (RT variant).

### 2. Pipeline by stage
- 2a Feature extraction: mono ViT-L DepthAnythingV2 (DINOv2 encoder + DPT decoder) is the single frozen encoder shared with the stereo branch (Fig. 5 snowflakes on mono encoder and decoder). ViT yields single-scale features; a trainable "feature transfer network" (stack of 2D convs) down/up-samples into pyramid F={F0..F3}, F_k in R^{H/2^(5-k) x W/2^(5-k) x c_k}, i.e. 1/32,1/16,1/8,1/4 (p.5). Channel counts c_k: not stated. Stereo branch otherwise follows IGEV.
- 2b Semantic/prior branch: monocular depth branch = frozen DepthAnythingV2-ViT-L (335.3M params, Sec. IV-G Efficiency). Outputs relative inverse depth D_M. No semantics. Swappable (DAv1 ViT-L, MiDaS dpt_beit_large also tested, Tab. X).
- 2c Cost volume: IGEV Geometry Encoding Volume (GEV) at 1/4 (details inherited from IGEV, "not restated"; disparity levels not stated). For MVS: parameter-free variance cost volume with differentiable homography (Eqs. 8-9), depth bins uniform in log space.
- 2d Aggregation: IGEV's (3D-reg. GEV); not restated.
- 2e Disparity computation: initial stereo disparity D_S^0 from N1 IGEV ConvGRU iterations (soft-argmin init inside IGEV; not restated).
- 2f Iterative update: N1 stereo-only GRU iterations, then N2 rounds of dual-branch mutual refinement (SGA then MGR), each a "condition-guided ConvGRU" (Eq. 4: standard GRU where the gate pre-activations get context features c_z,c_r,c_h added, and the GRU input is the condition vector x). Full model N1=N2 with 32 total iterations in Tab. IX; 4-iteration setting N1=N2=2 (Sec. IV-G, Tab. X). Final output = stereo branch after N2 rounds.
- 2g Upsampling: not stated in this file (inherits IGEV).
- 2h FUSION POINTS (all at 1/4 res for the large model):
  1. Feature level: shared frozen ViT encoder -> trainable conv transfer net -> stereo pyramid (mono->stereo, feature). Ablation "Feature Sharing" EPE 0.39->0.37 (Tab. IX).
  2. Init: global scale-shift alignment (Eq. 1) of D_M to D_S^0: closed-form least squares over pixels Omega = disparities between the 20th and 90th percentile (excludes sky, far, near outliers); gives D_M^0 = s_G D_M + t_G. Resolves coarse scale-shift (global only).
  3. SGA (stereo -> mono, per-pixel shift): condition x_S = [Eng([G_S, F_S, D_S]), End(D_M), D_M] (Eq. 3) feeds a ConvGRU whose hidden state belongs to the mono branch; decoded by 2 convs to a residual shift Δt; D_M^{j+1} = D_M^j + Δt (Eq. 5). Only additive per-pixel shift is learned (scale stays global). G_S = lookup in GEV at D_S; F_S = feature-warp residual (below).
  4. MGR (mono -> stereo): x_M = [Eng([G_M, F_M, D_M]), End(D_M), D_M, Eng([G_S, F_S, D_S]), End(D_S), D_S] (Eq. 6), independent GRU parameters, same 2-conv head -> Δd added to stereo disparity. G_M = GEV lookup at the *mono* disparity D_M (so the mono disparity is also scored against the stereo cost volume).
  5. Loss coupling: both branches supervised (Eq. 7).
- Confidence mechanism (key for our gating analogy): there is NO explicit confidence map or multiplicative gate. "Confidence" = feature-warp residual F_S^j(x,y) = || F3^L(x,y) - F3^R(x - D_S^j, y) ||_1 (Eq. 2, quarter-res features, L1 norm over channels; verified on page image), concatenated into the GRU condition together with the GEV lookup G_S and D_S. The GRU learns how much to trust stereo from these cues. "Adaptively selects reliable stereo cues" (abstract) is thus implicit/learned selection, not a computed gate. The same residual for the mono disparity (F_M, Eq. 6) lets MGR see whether the mono disparity is photometrically/feature consistent.
- RT-MonSter++ (Sec. III-C, Fig. 6): fully coarse-to-fine 1/16 -> 1/8 -> 1/4, each stage a GRU on a *local* cost volume (full volume only at the coarsest): per pixel D samples at interval δ centered on previous-stage disparity (Eqs. 10-12). Single-layer ConvGRU, 1 update at each of the two lower scales + 2 at 1/4 = 4 updates ("4 update iterations at inference", Tab. IV caption); lightweight redesigned aggregation (details not stated). Per Fig. 6 as I read it, mono fusion (global align + SGA/MGR) happens at the final 1/4 stage after the stereo init disparity; the text calls it "multi-scale depth fusion" but gives no further formulation. Mono encoder shared and frozen as in large model (Fig. 6). Params/channels of RT variant: not stated. Memory "only 2 GB", ">20 FPS at 1K resolution" (p.7), 47 ms at KITTI res 1248x384 (Tab. IV; GPU for timing not explicitly stated; experiments on RTX 4090 per Sec. IV-A).

### 3. Block -> problem -> evidence table
All Scene Flow in-domain unless noted (Tab. IX, 32 iterations, runtime for the 32-iteration setting).
| block | problem it solves | evidence (ablation delta) | context/conditions | cost |
|---|---|---|---|---|
| Baseline IGEV | - | EPE 0.47, >1px 5.21 (Tab. IX) | Scene Flow, in-domain | 0.37 s |
| Mono depth + conv fusion ("Mono+Conv": equal-params hourglass on concatenated mono & stereo disp) | naive fusion | EPE 0.47->0.46 (-0.01), >1px 5.21->5.12 (Tab. IX) | in-domain | 0.64 s |
| MGR (mono-guided GRU refinement) vs Conv | stereo failure regions | 0.46->0.43 EPE (-6.52%), >1px 4.96 (Tab. IX, text p.13) | in-domain | 0.65 s |
| Scale-shift refinement = Conv on top of MGR | ambiguity (naive) | 0.43->0.42, >1px 4.82 (Tab. IX) | in-domain | 0.65 s |
| SGA instead of Conv (Mono+MGR+SGA) | per-pixel shift error of mono | 0.43->0.39 EPE (-9.30%), >1px 4.96->4.43 (-10.69%); vs Mono+MGR+Conv 1px -8.09% (Tab. IX, p.13) | in-domain | 0.66 s |
| Feature sharing (frozen mono ViT -> transfer net) | stereo features lack context/robustness | 0.39->0.37 EPE (-5.13%), 4.43->4.25 (Tab. IX) | in-domain | 0.64 s |
| Full MonSter++ | all | EPE 0.37 (-21.3% vs IGEV, -15.9% vs Selective-IGEV 0.44) (Tab. I, text p.9) | in-domain | 0.64 s |
| Full, 4 iterations (N1=N2=2) | latency | EPE 0.42 in 0.34 s vs IGEV@32 it 0.47 / 0.37 s (-10.64%) (Tab. X) | in-domain | 0.34 s |
| Mono model swap: DAv1 ViT-L / MiDaS dpt_beit_large / DAv2 | choice of prior | 0.39 / 0.41 / 0.37 EPE; run-time 0.64/0.51/0.64 s (Tab. X) | in-domain | - |
| Edge vs non-edge (SF) | boundaries | edge EPE 2.23->1.91, non-edge 0.41->0.31 vs IGEV; vs Selective-IGEV edge -12.39%, non-edge -18.42% (Tab. VI, p.11) | in-domain | - |
| Zero-shot (Scene Flow-only training) full model | domain shift | avg of 4 sets 5.03 (IGEV) -> 3.70 (-26.4%); KITTI12 4.8->3.6, KITTI15 5.5->4.0, ETH3D 3.6->2.0, Midd 6.2->5.2 (Tab. VII, bad >3/>3/>1/>2 px) | ZERO-SHOT, synthetic only | 450 ms (Tab. VII) |
| Zero-shot FTS (2M pairs) full | scale | IGEV 3.48 avg -> MonSter++ 2.30 (Midd -39.5%, ETH3D -60%) (Tab. VII, p.12) | zero-shot but large mixed training incl. real-ish sets (Booster etc.) | - |
| RT-MonSter++ zero-shot (Scene Flow-only) | RT generalization | avg 8.90 (RT-IGEV++) -> 7.03; Midd 17.3->11.4 (-34.1%); ETH3D 5.8->6.0 (slightly WORSE) (Tab. VII) | zero-shot | 47 ms |
| RT-MonSter++ zero-shot (FTS) | scale | avg 4.83->3.70; ETH3D 2.5->1.7, Midd 8.3->6.2 (Tab. VII) | zero-shot-ish | 47 ms |
| DrivingStereo weather zero-shot (>3px) | real-world robustness | avg 4.93 (FoundationStereo) -> 2.69; rainy 6.08->3.08 vs SMoEStereo (Tab. VIII); note MonSter++ here trained on FTS (assumed, not explicitly stated for this table) | zero-shot | - |
| Benchmarks (fine-tuned, in-domain) | leaderboard | KITTI15 D1-all(All) 1.37 / D1-bg 1.12 vs IGEV++ 1.51/1.31; KITTI12 Out-3 All 1.07 vs 1.36; ETH3D Bad1 All 0.45 vs 0.48 FoundationStereo; Midd Bad4 Noc 1.18 (Tab. II) | fine-tuned on target sets | - |
| Reflective regions KITTI12 | non-Lambertian | Out-4 All 2.29 vs Selective-IGEV 3.05, LoS 3.01 (Tab. V) | fine-tuned | - |
| MVS (DDAD, KITTI Eigen) | same prior for MVS | DDAD AbsRel 0.075 vs AFNet 0.088; KITTI Eigen SqRel 0.104 vs 0.127 (Tab. III) | in-domain MVS | - |
| Stereo-branch update count, GEV, cascaded local volumes (RT) | speed | RT variant has NO ablation table of its own: cascaded search, single-layer GRU, lightweight aggregation all "not ablated" | - | - |
| Global scale-shift alignment (Eq. 1) alone, percentile window 20-90% | coarse scale | not ablated (no row without it) | - | - |
| Frozen mono encoder vs fine-tuned | generalization | claimed as intent ("prevent stereo training from affecting generalization", Sec. III-A1) but not ablated | - | - |

### 4. Interactions & dependencies
- SGA only makes sense after global alignment (D_M^0 in disparity space); without it mono and stereo are not in the same space for the GRU conditions.
- MGR needs a decent mono disparity: Mono+MGR without SGA gains 0.43; the learned SGA shift gives a further 0.04 EPE, and the gain of SGA over Conv (-8.09% 1px) shows the confidence-conditioned GRU beats plain conv for shift refinement.
- N1 initial stereo iterations are required to get a usable D_S^0 for the alignment; too few means noisy alignment (not ablated).
- Frozen shared ViT + trainable transfer net: the stereo branch inherits mono robustness; Fig. 2 and Tab. VII attribute the zero-shot gain to this, but feature sharing alone contributes only -5.13% EPE in-domain; zero-shot attribution to each block is NOT ablated.
- Mono model quality matters mildly (V1 0.39, MiDaS 0.41 vs V2 0.37).
- Cost: mono ViT-L dominates latency; with 4 iterations the fusion approach is faster than IGEV at 32 iterations.

### 5. Losses
- L1 on both branches with exponentially increasing weights, gamma = 0.9 (Eq. 7): L_Stereo = sum_{i=0}^{N1-1} gamma^{N1+N2-i} ||d_i - d_gt||_1 + sum_{i=N1}^{N1+N2-1} gamma^{N1+N2-i} ||D_S^{i-N1} - d_gt||_1; L_Mono = sum_{i=N1}^{N1+N2-1} gamma^{N1+N2-i} ||D_m^{i-N1} - d_gt||_1; total = L_Stereo + L_Mono (weights equal, 1:1; not stated otherwise). Mono-branch supervision uses only the post-alignment iterates, not the raw D_M. (Note the exponent gamma^{N1+N2-i} as printed with 0.9; indexes follow the paper's formula.)
- No confidence loss; no semantic loss.

### 6. Training recipe
- AdamW, grad clip [-1,1], one-cycle LR 2e-4, batch 8, 200k steps for the Scene Flow pretrained model, RTX 4090 GPUs (Sec. IV-A). Mono branch ViT-L DepthAnythingV2 frozen throughout.
- ETH3D/Middlebury: two-stage (BTS pretrain, then fine-tune on target + BTS). KITTI: fine-tune the Scene Flow model on KITTI12+15 mix for 50k steps. FTS (2M pairs) used for the generalizable released models. Crop size, augmentation, RT-model training schedule: not stated.
- MVS: trained from scratch on DDAD (Sec. IV-C).

### 7. Results (MonSter++ unless stated)
- Scene Flow EPE 0.37 (Tab. I). KITTI15 (All): D1-bg 1.12, D1-fg 2.78 (Fig. 1), D1-all 1.37; KITTI12 Out-2/3 All 1.70/1.07 (Tab. II). ETH3D Bad1.0 All 0.45; RMSE Noc 0.18. Middlebury Bad4.0 Noc 1.18, RMSE Noc 6.03.
- Params: 356.1M total = 335.3M mono (frozen ViT-L DAv2) + 12.6M stereo + 8.2M SGA+MGR (Sec. IV-G; NOT 20M-RT as the summary claimed). Runtime 0.64 s (32 it) vs IGEV 0.37 s (hardware: not explicitly given for this table; 4090 used for experiments); 0.34 s at 4 iterations.
- RT-MonSter++ (KITTI res 1248x384, 4 iterations, 47 ms): KITTI12 Out-3 All 1.41 / Out-4 All 1.05 / Out-5 All 0.84; KITTI15 D1-all All 1.69, D1-fg Noc 2.49 (Tab. IV); vs RT-IGEV++ (48 ms) 1.68/1.30/1.06, D1-all 1.79; HITNet 20 ms Out-3 All 1.89, D1-all 1.98. All benchmark numbers are on fine-tuned KITTI.
- Zero-shot RT numbers in Sec. 3 table.

### 8. Negative results & limitations
- Authors admit: cost (ViT-L 335M params, 0.64 s vs 0.37 s), "further memory reduction through encoder quantization or distillation... future work" (p.14). Original large model not real-time.
- No semantic/class prior studied; mono prior is pixel-level metric-less depth.
- RT-MonSter++ ETH3D zero-shot slightly worse than RT-IGEV++ on Scene Flow-only training (6.0 vs 5.8, Tab. VII); only Middlebury and KITTI show large gains.
- RT variant has no ablations; RT params and memory breakdown missing; timing hardware unspecified in the table; the "47 ms" compares at KITTI res and is stated as IGEV++'s protocol.
- Fairness: the zero-shot Tab. VII top rows compare under the same Scene Flow-only training (good), but FTS rows use a 2M-pair mix with real-like data and the leaderboard numbers use extra data; headline ranks mix these regimes.
- Per-pixel *shift only* is learned; scale is global, so strongly non-affine mono errors (scale varying across the image, as DEFOM reports) are only partially handled; no explicit test.
- Timing note: intro quotes IGEV 180 ms and IGEV++ 280 ms; RT-IGEV++ 48 ms; GPU used for the ms figures is not stated beside them.
- Errors found in summaries/fusion/MonSter.md: (1) labelled CVPR MonSter but all numbers are MonSter++ v2; (2) "RT-MonSter ~20M params" is not stated anywhere in the PDF; (3) KITTI15 "D1-all 1.37" is All-pixels MonSter++ (Tab. II); (4) the paper does not give a confidence map: "confidence" is an input feature (F_S), the summary's phrase "confidence-based flow residual: low = high confidence" is the authors' intent, not an explicitly computed/normalized confidence; (5) (verified, not an error) "mutual refinement reduces iteration count ... 4 iterations beats 32" is verified (Tab. X: 0.42 vs 0.47).

### 9. Relevance to OUR model
Our D2/E3 = frozen YOLO trunk + frozen A09 stereo + frozen semantic decoder; trained: SemanticCostGate + ClassResidual.
- Most portable idea: use *feature-warp residual* F_S (L1 between left 1/4 feature and right feature warped by the current disparity, Eq. 2) as a free, label-free stereo-reliability channel and feed it (with the cost lookup and disparity) to our class-conditioned gate/residual. Insertion: inside SemanticCostGate and ClassResidual inputs, at the A09 resolution. Cost: one gather+abs per candidate/iteration, negligible. Benefit: lets the semantic gate learn "trust semantics when stereo is inconsistent" rather than gating by class alone; analogous to the SGA condition. Risk: the shared YOLO features are optimized for segmentation, so their warp residual may be less discriminative than IGEV-type matching features; must be measured. Novelty: semantic class gate + feature-residual reliability has not been done in this series (E/F/G used semantics or RGB edges, not residual-conditioned arbitration).
- Bidirectional (mono<->stereo) refinement and per-pixel shift refinement do NOT transfer: they need a metric-free but continuous depth prior to align; a categorical prior has no scale or shift to recover. The nearest categorical analogue of "per-pixel shift refinement" is our ClassResidual (class-conditioned additive residual), which already exists; SGA-style hidden-state-of-the-prior GRU would be a new, heavier design (2x GRUs, +8.2M params in MonSter++ scale) and is not justified for a 4GB real-time target.
- Equal-parameter conv-concat control: MonSter++'s Mono+Conv vs MGR (-6.5% EPE) and Conv vs SGA (-8.1% bad-1) mirrors our no-semantics equal-capacity control logic; it supports using a GRU/gate with explicit reliability inputs rather than plain concat if we ever try a heavier head. Do not expect their magnitude (they have a dense metric-ish prior, we have 14 classes).
- RT-MonSter++ tricks portable to Jetson latency: cascaded local cost volumes centered on previous-stage disparity (Eqs. 10-12) and ~4 total GRU updates; our tile/plane lineage (HITNet) already searches locally, so little new. Note RT mono fusion is only at the finest scale and was never ablated.
- Frozen shared encoder with a conv "transfer" adapter (feature sharing, -5.13% EPE) is the exact structure we already use (frozen YOLO layers 0-6 shared by stereo and semantics); the paper is mild evidence that sharing a prior-pretrained trunk helps in-domain; zero-shot attribution is not ablated.
- Caution on claims: gains are mono-prior-specific and in-domain Scene Flow for the ablation; do not borrow the zero-shot headline as support for a semantic prior.

### 10. Key quotes/equations worth citing
- "MonSter++ ... formulate multi-view depth estimation as a problem of leveraging multi-view matching information to achieve per-pixel scale-shift recovery based on monocular depth estimation" (p.3).
- Eq. 1 global alignment over pixels with disparity in 20%-90% rank range (p.6); Eq. 2 F_S^j = ||F3^L(x,y) - F3^R(x - D_S^j, y)||_1 (p.6); Eq. 3/6 conditions; Eq. 5 D_M^{j+1} = D_M^j + Δt.
- "SGA uses confidence based guidance. ... we compute the confidence using the flow residual map F_S^j" (p.6).
- "RT-MonSter++ requires only 47 ms per inference yet even surpasses IGEV++ (280 ms) particularly on edge regions" (Fig. 2 caption).
- Param breakdown: 335.3M + 12.6M + 8.2M = 356.1M (p.14).
