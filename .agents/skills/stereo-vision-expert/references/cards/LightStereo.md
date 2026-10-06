<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# LightStereo card

Source: paper/reference_papers/lightweight/LightStereo_Guo_ICRA2025.pdf (7 pp, no supplement). Page refs below: p.1-6 per ICRA layout; most tables on p.4-5.

### 0. Meta
- Title: LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation. Guo*, Zhang*, Zhang, Zheng, Nie, Poggi, Chen.
- Venue: ICRA 2025 (arXiv 2406.19833v3, 26 Feb 2025).
- PDF: paper/reference_papers/lightweight/LightStereo_Guo_ICRA2025.pdf
- Code: https://github.com/XiandaGuo/OpenStereo (abstract).
- Domain: general/driving stereo; datasets SceneFlow (35,454 train / 4,370 test, 960x540), KITTI 2012/2015 (194/195 train, 195/200 test), Middlebury 2014 (half-res) for zero-shot only (Sec. IV-A).

### 1. Problem & failure modes targeted
- Latency/memory on edge: 4D cost volume + 3D CNN aggregation is too slow; iterative methods >100 ms (Intro).
- Prior 2D-aggregation light nets (AANet, MobileStereoNet-2D, EPE 1.14 on SceneFlow) are inaccurate. Claim: put capacity on the disparity (channel) axis of a 3D cost volume (C = D/4 channels) rather than on H,W.
- Disparity discontinuities: MSCA claims "network halts propagation" at discontinuities (p.2-3; asserted, no boundary-specific metric).
- Cross-domain generalization of lightweight models (Table VII).

### 2. Pipeline by stage
- 2a Features: ImageNet-pretrained MobileNetV2 (also timm-style), 4 scales 1/4,1/8,1/16,1/32; upsampling blocks with skip connections restore to 1/4 for cost volume (Sec. III-C). Trained (not frozen; not stated otherwise). Left/right appear to share weights (not explicitly stated). Channel count at 1/4: not stated. Feature extraction costs 10.39 ms of 17.83 ms total for -S (Tab. VI).
- 2b Semantic/prior branch: n/a. Only an image-feature "context" branch (left-image features) used by MSCA.
- 2c Cost volume: single-channel-per-disparity correlation volume at 1/4: C_cor(d,h,w)=1/C * sum_c f_l(h,w) f_r(h,w-d), d in 0..D-1 (Eq. 5). Result H/4 x W/4 x D/4 as written in text (p.2: "H/4 x W/4 x Disp/4"; Fig. 2 shows D as channel dim). Max disparity D: not stated. Not group-wise, not concat.
- 2d Aggregation: pure 2D U-Net-like encoder-decoder over the volume, with the D/4 disparity axis as CHANNELS. Inverted residual (MobileNetV2 V2) blocks at 1/4, 1/8, 1/16: 1x1 expand (ReLU6) -> 3x3 depthwise (ReLU6) -> 1x1 project (linear), skip if shape matches (Eq. 1-4). Block counts per scale: S (1,2,4), M (4,8,14), L (8,16,32), H = L with EfficientNetV2 backbone; expansion factor 4 (S,M), 8 (L,H) (Sec. III-C). Strided (stride 2) blocks for down.
- MSCA (Fig. 4): input = left-image feature at 1/4,1/8,1/16; parallel depthwise strips 1x1, 7x1+1x7, 11x1+1x11, 21x1+1x21 -> concat -> 1x1 conv channel mixer -> MULTIPLY with the aggregated cost at same scale. Applied at each of the 3 aggregation resolutions.
- 2e Disparity: soft-argmax over D (Eq. 6), softmax over cost channels. No top-k, no tiles.
- 2f Refinement: none (single-pass, no GRU).
- 2g Upsampling: Fig. 2 labels "x4" and "x16" arrows toward final disparity; method (bilinear/learned) not stated. No convex upsampling mentioned.
- 2h Fusion points: (i) left-image multi-scale strip features gate cost channels multiplicatively at 3 scales (image->cost, feature level, resolution 1/4,1/8,1/16). That is the only second-cue injection. A semantic cue COULD enter here: replace/augment the MSCA input with semantic features (our shared YOLO layers 0-6 features) or class-conditioned masks; the multiplicative channel-gate hook is identical in form to our SemanticCostGate. Authors do say MSCA "incorporates semantic information embedded within the images" (p.2) but no semantic network is used.

### 3. Block -> problem -> evidence table
| block | problem | evidence | conditions | cost |
|---|---|---|---|---|
| Regular 3x3 conv aggregation (baseline family) | - | EPE 0.7652 (blocks 4,8,16) (Tab. II) | SceneFlow, 50 ep ablation | 36.27 GF, 8.04M, 16.7 ms |
| Larger kernels 5/7/11 | tests spatial extent | worse: 0.7979/0.8190/0.8672 (Tab. II) | same | 71/124/281 GF; 17.3/19.7/33.6 ms |
| V1 depthwise-separable block | cheap | 0.7801 (Tab. II) | blocks (30,60,120) | 34.9 GF, 54.2 ms (slow) |
| V2 inverted residual (chosen) | channel boost on disparity axis | 0.7144 vs 0.7652 regular (-0.051) (Tab. II) | same | 35.82 GF, 7.54M, 22.9 ms |
| ViT (EfficientViT) block | alt | 0.7149 (Tab. II) | blocks (3,6,9) | 34.5 GF, 6.5M, 51.1 ms |
| Backbone: MobileNetV2 (chosen) | speed | 0.7144 (Tab. III) | | 35.82 GF, 22.9 ms |
| Backbone MNv3 / StarNet / RepViT / EffNetV2 | | 0.7292 / 0.7247 / 0.6823 (text says 0.6876, inconsistency) / 0.6130 (Tab. III) | | EffNetV2 103 GF, 46.8 ms; RepViT 50.5 GF, 28.7 ms |
| Expansion factor 2->4->8->16 | more disparity-channel capacity | 0.7557->0.7144->0.6853->0.6650 (Tab. IV a-d) | blocks (4,8,16) | FLOPs 26->36->55->93 G; 22.4->36.7 ms |
| Block depth (1,2,4)/(2,4,8)/(4,8,16)/(8,16,32) | | 0.8317/0.7464/0.7144/0.6973 (Tab. IV e-h) | exp 4 | 22.2/26.7/35.8/54.0 GF |
| SE module | channel excite | 0.7036 (-0.011) (Tab. IV j) | | params 7.54->12.76M, 30.1 ms (+7 ms) |
| MSCA (chosen) | left-image-guided cost gating | 0.6809 (-0.034 vs 0.7144) (Tab. IV k) | | +0.54 GF, +0.1M, +0.2 ms (22.93->23.14) |
| SE + MSCA | | 0.6810 (no extra gain) (Tab. IV l) | | 12.86M, 30.8 ms |
| MSCA on small block (1,2,4) | | 0.8317->0.7899 (-0.042) (Tab. IV e vs m) | | 3.44M, 17.6 ms |
| Final S/M/L/H | | see Sec. 7 | | |
Not ablated: soft-argmax vs alternatives, upsampling method, cost-volume type (correlation vs group-wise), MSCA kernel set, MSCA at fewer scales, pretraining of backbone, 1x1 channel mixer.

### 4. Interactions & dependencies
- MSCA gain is larger on small aggregation nets (-0.042 on S) and nearly free; SE gain overlaps MSCA (SE+MSCA = MSCA only), so image-conditioned channel excitation saturates quickly (Tab. IV).
- Capacity must sit on disparity-channel expansion, not spatial kernel size (Tab. II).
- Real latency != FLOPs: V1 block and ViT block have similar FLOPs but 2x runtime (Tab. II), depthwise ops are memory bound on 3090.
- Backbone dominates latency for -S (10.4 of 17.8 ms; Tab. VI) and for -H (27.2 ms of 54 ms).

### 5. Losses
- Single smooth-L1 on final soft-argmax disparity, averaged over labeled pixels: L = 1/N sum smoothL1(d_i - d_hat_i) (Eq. 7). No deep supervision, no multi-scale loss, no weights.

### 6. Training recipe
- 8x RTX 3090; AdamW + OneCycleLR, max LR = 1e-4 x batch size; SceneFlow batch sizes S/M/L/H = 24/12/8/6; 90 epochs; only random crop 320x736 (Sec. IV-B). Ablations trained 50 epochs.
- KITTI: finetune SceneFlow-pretrained for 500 epochs on mixed KITTI12+15 train, batch 2, OneCycle max LR 2e-4.
- Generalization models: "trained on SceneFlow only"; augmentation color jitter, random erase, random scale, random crop were used "for the generalization experiments" (ambiguous whether in SF training). Nothing frozen; backbone ImageNet-init.

### 7. Results
- SceneFlow EPE / GFLOPs / Params / time (Tab. I): S 0.73 / 22.71 / 3.44M / 17 ms; M 0.62 / 36.36 / 7.64M / 23 ms; L 0.59 / 91.85 / 24.29M / 37 ms; H 0.51 / 159.26 / 45.63M / 54 ms. Competitors: HITNet 0.55 / 50.2 GF / 0.42M / 36 ms; CoEx 0.67 / 53.4 / 2.72M / 36 ms; Fast-ACVNet+ 0.59 / 93 / 27 ms.
- Timing hardware: RTX 3090 (Tab. V footnote star, "time measured on our RTX 3090"). No Jetson/Orin or TensorRT numbers anywhere. Input resolution for timing not stated (likely 960x540 SceneFlow; unverified). Runtime breakdown (Tab. VI): S = 10.39 feat + 1.98 cost + 3.98 agg + 1.48 regression = 17.83 ms.
- KITTI 2015 online D1-all (Tab. V): S 2.30 (17 ms), M 2.04 (23 ms), L 1.93 (34 ms), H 1.82 (49 ms). KITTI 2012 3-all: S 1.88, M 1.56, L 1.55, H 1.34. Compare HITNet 1.98 D1-all @54 ms; CoEx 2.13 @33 ms.
- Zero-shot, SceneFlow-only (Tab. VII), KITTI12 D1 / KITTI15 D1 / Middlebury 2px: CoEx 13.5/11.6/25.51; Fast-ACV 12.4/10.6/20.13; IINet 11.6/8.5/19.57; LS-S 11.6/9.0/19.63; LS-M 7.0/6.6/17.69; LS-L 6.4/6.4/17.51; LS-H 7.2/7.3/14.27.

### 8. Negative results & limitations
- Authors admit: larger spatial kernels hurt; ViT block similar accuracy but 2x slower; SE costs +5M params.
- Weak points: no Jetson numbers though embedded is motivation; timing protocol (resolution, precision, warmup) not stated; "detailed metrics not provided" for block ablation (text p.5); inconsistent RepViT EPE (0.6823 table vs 0.6876 text); abstract "17 ms" vs Tab. VI 17.83 ms; LS-S KITTI15 time 17 ms measured on 3090 while competitors' times are from leaderboards (different GPUs) -> unfair latency comparison. Ablations 50 ep vs final 90 ep. Zero-shot trained on SceneFlow only (augmentation ambiguity), generalization gain of M/L over S (D1 6.6 vs 9.0) suggests capacity matters more than design. MSCA "stops at discontinuities" is not tested with a boundary metric. No semantic/mono prior.

### 9. Relevance to OUR model
- Closest lineage to our A09 shallow head. Portable idea 1: MSCA-style multiplicative gating of cost channels by image-feature strips. Our SemanticCostGate does the same with semantic features; LightStereo gives the evidence that cheap multiplicative image-conditioned excitation yields -0.034 EPE (SceneFlow, 5% rel.) at +0.2 ms, and that SE-style channel-only excite is weaker/more expensive. Nothing new for our gate except a sanity check on the form, and the finding that stacking SE with MSCA saturates (a warning that semantic gate and a generic image gate may be partly redundant; our no-semantics equal-capacity control already handles this).
- Idea 2: disparity-axis channel expansion (inverted residuals on cost volume). Our trainable head is small; expanding the disparity-candidate channel inside the gate/refiner (expansion 4 -> 8 gave -0.03 EPE at +19 GF here) is a possible E-series-like capacity knob but E already tested width. Cheap, low risk, probably small gain on frozen A09.
- Idea 3: strip (1xk, kx1) depthwise kernels at 1/4,1/8,1/16 as the context input of the gate; elongated structures (poles, rails). Moderate novelty vs ours; risk: our frozen YOLO features at ~1/8 and 1/16 only; 1/4 not available.
- Not portable: backbone (we are frozen YOLO), no refinement, no semantic. For Jetson planning, its Tab. VI shows feature extraction dominates latency at ~58%; our shared trunk amortizes that.
- Novel combination: semantic-feature MSCA (class strips) as the gate input is not done in this paper.

### 10. Key quotes/equations worth citing
- "we utilize semantic information inherent in the images (such as object-level semantic details) to guide the cost aggregation process. When encountering discontinuities in disparity, the network halts propagation." (p.1-2)
- Eq. 5 correlation volume; Eq. 6 soft-argmax; Eq. 7 smooth L1.
- "The final output is combined with the aggregated cost by multiplication" (Sec. III-B).

### Errata vs existing summary (summaries/lightweight/LightStereo.md)
- Summary lists expansion t=8 EPE 0.6779; paper Tab. IV(c) gives 0.6853.
- Summary says LightStereo-S 10.39 ms feature extraction = "58%" (correct: 10.39/17.83). Summary labels all times RTX 3090: correct only for the "*" starred rows; other competitor times in Tab. V are leaderboard-reported on unknown GPUs.
- Summary says "diminishing returns after t=4" but Tab. IV shows EPE still dropping 0.7144 -> 0.6853 -> 0.6650 at t=8,16 (at +19 and +57 GFLOPs); choice of 4 is a cost choice.
