<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# ESMStereo card

Source: paper/reference_papers/lightweight/ESMStereo_Tahmasebi_JImaging2026.pdf (25 pp: 20 body, 5 refs; no appendix). All pages read. No existing summary in summaries/.

### 0. Meta
- Title: ESMStereo: Enhanced ShuffleMixer Disparity Upsampling for Real-Time and Accurate Stereo Matching. Tahmasebi, Huq, Meehan, McAfee (ATU Sligo / Bridgewater College).
- Venue: arXiv 2506.21091v2, 24 Jun 2026; "JImaging2026" is only in the filename.
- Code: https://github.com/M2219/ESMStereo (abstract).
- Domain: general/driving; datasets SceneFlow, KITTI 2012/2015, ETH3D, Middlebury 2014.

### 1. Problem & failure modes targeted
- Small-scale (1/4-1/16) cost volumes lose matching information; low-res disparity needs a good upsampler to recover details and thin structures (Intro, Sec. 3).
- Latency of large 3D cost volumes + aggregation.
- Prior upsamplers (bilinear/deconv, CoEx superpixel, pixel-shuffle in RTSMNet) fail on fine/thin structures; FBPGNet limited by missing refinement after deconv (Sec. 3).
- Not targeted: occlusion, textureless, domain shift as explicit mechanisms (only evaluated as zero-shot).

### 2. Pipeline by stage
- 2a Features: encoder-decoder; EfficientNet-B2 (L, M) or MobileNetV2 (S) encoder + transposed-conv decoder; features at 1/4, 1/8, 1/16. Backbone TRAINED FROM SCRATCH including encoder (Sec. 5.2); no ImageNet init stated. Channel counts: not stated.
- 2b Semantic/prior: n/a.
- 2c Cost volume: ONE volume, either group-wise correlation (Eq. 1, Ng groups not stated) or norm-correlation (Eq. 2). Resolution: L 1/4 (Dmax/4 levels), M 1/8, S 1/16 (Sec. 4.2; Dmax=192 -> 48/24/12 levels). Choice evaluated as separate models, not simultaneously.
- 2d Aggregation: single lightweight 3D hourglass (Table 2); channels j = 16 (L), 8 (M), 4 (S) with i = 8; GELU+BN, 3x3 3D convs, deconvs, skip concat. Output channel 1.
- 2e Disparity: top-k selection from the aggregated volume then soft regression following CoEx [5]: k=1 at 1/8 and 1/16, k=2 at 1/4 (Sec. 4.3).
- 2f Refinement: part of ESM (below); no iterations/GRU.
- 2g Upsampling: Enhanced ShuffleMixer (ESM) stages; progressive. Config (Tab. 1): L: 2 stages (1/4->1/2->1), input feature ch 16, 2 FMBlocks, 7x7 kernel, 2 pixel-shuffle x2; M: 3 stages (1/8->1/4->1/2->1), 8 ch, 3 FMBlocks, x2 shuffles; S: 2 stages (1/16->1/4->1), 8 ch, 4 FMBlocks, pixel-shuffle factor 4.
  Per-stage ESM (Fig. 3, Sec. 4.1): (1) 4 conv layers on the incoming low-res disparity to extract disparity features, concatenated with left-image feature at that scale (f_{1/4} in Fig. 3), 2 more conv; (2) two FMBlocks (ShuffleMixer FMBlock [45]: 2 Shuffle Mixing Layers = LayerNorm + channel split-point MLP w/ residual + depthwise kxk conv, then a bottleneck refine with 3x3 conv expanding to C+16, SiLU, 1x1 back to C, residuals); (3) pixel shuffle (x2 / x4); (4) feature-guided hourglass (encoder-decoder) with left-image features f_{1/8} and f_{1/4} concatenated inside; (5) output SUMMED with bilinear upsampled disparity (residual, Fig. 3). Fig. 3 is drawn for one stage; per-stage feature scales for M/S are not detailed.
- 2h Fusion points: left-image features enter at the disparity-feature fusion (concat, image->disparity, at each ESM scale) and inside the hourglass (concat). No semantic/mono/edge cue. Image-feature fusion cannot be ablated separately; "evaluated jointly with refinement" (Sec. 5.3).

### 3. Block -> problem -> evidence table
All ablation on SceneFlow (EPE px / D1 %), RTX 4070 Super.
| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost (ms/FLOPs) |
|---|---|---|---|---|
| Group-wise corr vs norm corr | matching quality at small volumes | S 1.21 -> 1.10, M 0.80 -> 0.77, L 0.55 -> 0.53 EPE (Tab. 3) | volume res 1/16, 1/8, 1/4 | ~same (Tab. 3: L-nc 24 ms/58 G vs L-gwc 26 ms/69 G) |
| Cost-volume res 1/4 -> 1/16 | speed | EPE 0.53 -> 1.10, runtime 26 -> 8.6 ms (3x) (Sec. 5.3/Tab. 3) | different backbones too (EffNet-B2 vs MNv2) | 69 G vs 9.4 G |
| Disparity feature extraction only (baseline in Tab. 4) | | EPE 1.20, 23.5 ms | L-gwc | |
| + FMBlock (ShuffleMixer) | local mixing | 1.20 -> 0.92 EPE, +0.6 ms (Tab. 4) | L-gwc | 24.1 ms |
| + feature-guided hourglass refinement (with left-image fusion) | detail/thin structure | 0.92 -> 0.53 EPE, +1.9 ms (Tab. 4) | L-gwc | 26 ms; ESM total 4.81 of 26.01 ms (Tab. 5) |
| Image-feature fusion alone | | not ablated separately ("inherently embedded", p.12) | | |
| Pixel shuffle vs bilinear/deconv as the upsampler | | NOT ablated inside ESMStereo. Cross-method only: RTSMNet-c8 0.71, FBPGNet 1.19 vs 0.53 (Sec. 5.4/Tab. 6) | different backbones, volumes, training | |
| ESM portability | | PSMNet 1.09 -> 1.02 EPE, 613 -> 756 GFLOPs, 5.22 -> 5.70M params; Fast-ACVNet+ 0.59 -> 0.51, 62 -> 70 G, 3.2 -> 3.5M (Tab. 10) | retrained on SceneFlow; ESM not retuned | +24% / +13% FLOPs |
| Top-k regression (k=1,1,2) | | not ablated | | |
| Multi-scale loss weights | | not ablated | | |
| Variants S/M/L | speed-accuracy | Tab. 6/7/8 | | |
| FP16 TensorRT | Jetson speed | "no observable degradation in EPE" on SceneFlow (Sec. 5.5) | | |

### 4. Interactions & dependencies
- ESM is meant to compensate for the small cost volume; the gain over compact volumes is conditional on the volume being weak (S/M variants lose accuracy vs L: 1.10/0.77/0.53).
- Image features must be fused; they say fusion improves training stability and edges but cannot be isolated (p.7, p.12).
- Pixel shuffle alone does not smooth; needs the hourglass after it (Sec. 4.1, Sec. 3).
- Tab. 4 baseline already has "disparity feature extraction" so the ladder does not show what a plain bilinear upsampler would score.
- Backbone trained from scratch; accuracy depends on SceneFlow pretraining (60 epochs).

### 5. Losses
- Multi-scale smooth L1: L = sum_{i=0..N} lambda_i * smoothL1(d_i - d_i^gt) (Eq. 3), lambda = {1, 1/6, 1/10} (reduction factors), GT interpolated to each scale; smoothL1 as Eq. 4 (0.5x^2 for |x|<1, |x|-0.5 else). Text: "N=3 ESM modules generates d_{1/4}, d_{1/2}, d_1" while Tab. 1 L has 2 stages; supervision count for each variant is ambiguous.
- No edge-specific, semantic or confidence loss.

### 6. Training recipe
- PyTorch, AdamW (b1 0.9, b2 0.999), batch 4, crop 256x512, Dmax 192, augmentation brightness/contrast/saturation only.
- SceneFlow: 60 epochs, LR 1e-3, halved at epochs 20, 32, 40, 48, 56; then "fine-tuning" 80 epochs, LR 2e-4 halved at 20, 30, 40, 50, 60, 70 (dataset of the fine-tune phase not stated, presumably SceneFlow).
- KITTI: from SceneFlow weights, 600 epochs on KITTI 2012+2015 combined, LR 1e-3, 1e-4 at epoch 300.
- Hardware: RTX 4070 SUPER 12GB; Jetson AGX Orin 64GB for inference only.

### 7. Results
- SceneFlow (Tab. 6): S-gwc EPE 1.10, 8.6 ms, 9.4 GFLOPs, 1.8M; M-gwc 0.77, 14 ms, 31 G, 6.3M; L-gwc 0.53, 26 ms, 69 G, 6.8M (RTX 4070S). Others (different GPUs): LightStereo-S 0.73/17 ms/22G, LightStereo-L 0.59/37, Fast-ACVNet+ 0.59/27, RT-IGEV++ 0.55/42, IINet 0.54/26/90G, CoEx 0.68/27, RTSMNet 0.71/28.
- Runtime breakdown (Tab. 5, ms): feature 6.85/10.86/11.80, cost volume .08/.62/6.22, aggregation .10/.25/3.18, ESM modules 1.57/2.27/4.81, total 8.60/14.00/26.01 (S/M/L).
- KITTI (Tab. 8, fine-tuned): L 3-noc 1.15, 3-all 1.52, 4-noc 0.88, 4-all 1.16, EPE noc/all 0.4/0.5, D1-bg 1.43, fg 3.80, all 1.82, 26 ms; M 1.95/2.37/1.36/1.69, D1-all 2.34, 14 ms; S 3.0/3.45/1.98/2.31, D1-all 3.30, 8.6 ms.
- Zero-shot, SceneFlow-only (Tab. 9; K12 D1, K15 D1, Mid bad2, ETH bad1): L 5.4/5.5/11.1/6.1; M 9.7/8.2/15.7/11.7; S 12.3/10.8/24.1/27.8.
- Jetson AGX Orin 64GB, TensorRT FP16, trtexec 200 runs (Tab. 7): S 91 FPS, M 29, L 8.4 FPS (L = ~119 ms, NOT real time); RTX 4070S: 116/71/38 FPS. Same D1 on KITTI 2015 training (zero-shot) as FP32: 10.8/8.2/5.5.
- No edge metric of any kind (no boundary EPE, no Bad-x at depth discontinuities, no thin-structure metric).

### 8. Negative results & limitations
- Authors: does not beat heavier models (IGEV, MoCha) on ETH3D/Middlebury in textureless, thin, occluded regions; low-res disparity limits high-frequency detail (Sec. 5.6). Runtime comparisons across GPUs caveated.
- Mine: (1) THE QUESTION ASKED: does the learned upsampler measurably improve edges? The paper offers NO edge-specific evidence. All support is global EPE on SceneFlow (Tab. 4: 1.20 -> 0.92 -> 0.53) and qualitative figures (Figs. 4-8: error maps showing thin-structure recovery). The base for upsampling is 1/4 (L), 1/8 (M), 1/16 (S). The Tab. 4 ladder removes the whole ESM (feature fusion + FMBlock + hourglass); bilinear-only baseline is not given. Only portability (Tab. 10: -0.07 and -0.08 px EPE) is a controlled comparison, ~6% and ~14% relative. A global EPE gain on SceneFlow is dominated by large-area errors, and is not proof of sharper edges. (2) The large 1.20 -> 0.53 gain comes mainly from the hourglass with left-image fusion, i.e. image-guided refinement, a design G1/H1 in our repo already tested in a stronger form and found not to help E3/A09 EPE; their gain is relative to a weak, from-scratch 1/4 base (EPE 1.20) where there is lots to recover. (3) From-scratch backbone, SceneFlow 0.53 vs LightStereo-L 0.59 uses different training budgets (60+80 epochs, batch 4) and GPUs; not an equal-protocol comparison. (4) Zero-shot K12 D1 5.4% from SceneFlow alone is better than several accuracy-oriented models in Tab. 9; the D1 definition for KITTI zero-shot is not restated and baseline rows mix paper-reported and self-run values (asterisk only for LightStereo). Treat as unverified. (5) FLOPs inconsistencies: Fast-ACVNet+ 93 GFLOPs in Tab. 6 vs 62 G in Tab. 10. (6) Abstract says 116 FPS RTX 4070S and 91 FPS Orin: both are the S variant; L variant is 8.4 FPS on Orin. (7) Dmax-limited zero-shot ETH3D S bad1 27.8%.

### 9. Relevance to OUR model
- The measurable-edge question for us is answered "no evidence" here, consistent with our G/H negative results; do not expect the ShuffleMixer upsampler to fix edges in a frozen A09 + fusion head. Our setting differs: ESM trains the full network from scratch to compensate for a weak 1/16-1/4 base; our base disparity is already good (E3 KITTI EPE 2.59, VKITTI 1.4 px).
- Portable at low cost: pixel-shuffle upsampling (parameter-free rearrangement; trivially TensorRT-friendly) as a replacement for bilinear/convex upsample of the fused disparity or of the class-residual map. Insertion point: after ClassResidual at 1/4 or 1/8. Expected benefit: small; cost: small (their ESM modules 1.6-4.8 ms on RTX 4070S, 6-18% of runtime, Tab. 5; on a 3050 more, uncompiled). Risk: we already saw RGB-guided residuals and convex upsampling not sharpen; this is another learned image-guided upsampler with only global-EPE support.
- Better use: their runtime breakdown shows backbone is 60-80% of runtime (Tab. 5); our shared frozen trunk amortises that. Their evidence supports shifting compute from 3D aggregation to cheap 2D refinement, which is consistent with our 2D gate head. Group-wise vs norm corr: +0.02-0.04 EPE (Tab. 3), tiny; we already use correlation.
- Deployment: FP16 TensorRT on Orin AGX shows no EPE loss for conv/correlation nets, supporting FP16 for our trunk and heads; but even S at 1/16 gets 91 FPS only on AGX Orin 64GB (not NX 8G / Nano). Note the L variant at 1/4 volume is 8.4 FPS on AGX Orin.
- Novel combination vs done: semantic-conditioned gating with a ShuffleMixer/pixel-shuffle upsampler is not in this paper; low priority given our negative results on guided upsamplers.

### 10. Key quotes/equations worth citing
- "ESM ... integrating primary features into the disparity upsampling unit ... mixed by shuffling and layer splitting then refined through a compact feature-guided hourglass network" (Abstract, p.1).
- Tab. 4 (p.12): disparity features only 1.20 EPE / 23.5 ms; +FMBlock 0.92 / 24.1; +hourglass refinement 0.53 / 26.
- "image-feature fusion is inherently embedded within the refinement process and cannot be cleanly isolated" (p.12).
- Tab. 10 (p.20): PSMNet 1.09 -> 1.02; Fast-ACVNet-Plus 0.59 -> 0.51 EPE.
- "no observable degradation in EPE" under FP16 TensorRT on AGX Orin (Sec. 5.5, p.14); Tab. 7: S/M/L 91/29/8.4 FPS.
