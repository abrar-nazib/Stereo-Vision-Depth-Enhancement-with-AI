<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# D-FUSE card

### 0. Meta
- Title: Diving into the Fusion of Monocular Priors for Generalized Stereo Matching. Authors: Yao, Yu, Liu, Zeng, Wu, Jia (BIT / Shenzhen MSU-BIT / NVIDIA). ICCV 2025 (arXiv 2505.14414v2). 32 pages: main p. 1-8, refs p. 9-11, supplement p. 12-32 (visualizations Fig. 5-23; tables 8-13 p. 14-16; failure cases p. 31-32).
- pdf: paper/reference_papers/fusion/D-FUSE_Yao_ICCV2025.pdf. Code: GitHub/HF links on title (URL text not given in extracted page).
- Domain: general, zero-shot generalization (train SceneFlow only). Eval: KITTI-12/15, Middlebury-H, ETH3D, Booster-Q, DrivingStereo (weather), Flickr1024 (qualitative).

### 1. Problem & failure modes targeted
- Ill-posed regions: occlusion, textureless, reflective/transparent (Booster) (p. 1-2). Mono priors learned on small stereo sets are domain-biased (p. 1) -> use frozen Depth Anything v2 (DA-V2).
- Three named fusion problems (p. 1-2): (1) affine-invariant relative mono depth vs absolute disparity misalignment; (2) implicit mono-FEATURE fusion in iterative update biases toward binocular info and the over-confident update -> local optima; (3) direct mono depth fusion is hurt by noisy disparity in the first iterations (misguides fusion, slows convergence, p. 4-5).

### 2. Pipeline by stage
- 2a Feature extraction: RAFT-Stereo siamese feature encoder (shared weights) -> cost volume; trained. Scales/channels: not stated beyond RAFT-Stereo.
- 2b Mono prior branch: frozen DA-V2 (ViT variant not stated; image resized so longest side = 512 px, p. 3). Intermediate features before DPT head + mono depth after DPT head, bilinearly resized to H/4 x W/4, then two-stream conv -> initial GRU hidden state and mono context features (replaces RAFT context net). "Mono depth" is inverse-depth/disparity-like (p. 3).
- 2c Cost volume: warped correlation (RAFT-Stereo); "reg" (precomputed) or "alt" (on-the-fly) modes (Tab.12). Disparity range limit removed vs volume methods (p. 3).
- 2d Aggregation: n/a (no 3D regularization).
- 2e Disparity: GRU delta update accumulation (Eq. 3). Zero-initialized disparity (Fig. 2).
- 2f Refinement: T iterations of multi-level GRU (RAFT-Stereo) + conv head gives Delta_d; T not stated in main text (Fig. 3 shows t up to 12, supp Fig. 14 ~12 steps). Update: D_d^t = D_d^{t-1} + Delta_d(1 + G * r * t/T) (Eq. 2-3), r=1.
- 2g Upsampling: not described (output "upsampled registered mono depth" in supp figs); details not stated.
- 2h FUSION POINTS:
  (A) Context/hidden init: DA-V2 features replace RAFT context features (implicit feature fusion, mono->disp, 1/4 res). Evidence it is needed: Tab.5.
  (B) Iterative Local Fusion: LBP-like fixed-weight conv blocks (window sizes 5,3, sigmoid after, Eq.1) make local ordering maps M_O from mono depth AND from current disparity D^{t-1}; concat -> conv block -> guidance G ~ Beta(alpha,beta) (reparameterized Gamma sampling in training, mean alpha/(alpha+beta) at test); G multiplies GRU update (Eq. 2). Bidirectional-ish: mono ordering vs stereo ordering difference tells where to trust update. Resolution H/4 presumably (not stated).
  (C) Global Fusion (after last iteration): conv net F(D_m, D_d^T) -> per-pixel (a,b), D~_m = a D_m + b (Eq.4); confidence c = sigmoid conv on [sampled cost volume, hidden state, last guidance] (Hybrid, Tab.7); D_f = c D_d^T + (1-c) D~_m (Eq.5). Operator: affine registration + confidence blend. Mono->disp output-level fusion.

### 3. Block -> problem -> evidence table
All Middlebury-H (all pixels), SceneFlow-trained, mean±std of last 100k/90k/80k steps, each ablation trained from scratch (p. 6-7). EPE / bad2.0 (%).
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Baseline RAFT-Stereo | - | 2.11±0.16 / 14.12±0.64 (Tab.5) | zero-shot | 0.32 s (p. 7; hardware/res not stated) |
| Baseline w/o mono (context) feature | tests domain bias | 1.83±0.11 / 12.45±0.86: REMOVING RAFT's own context features HELPS generalization | zero-shot | - |
| + ME (DA-V2 monocular encoder) | unbiased prior | 1.42±0.01 / 9.81±0.18 (-0.69 EPE vs baseline) | zero-shot | frozen ViT |
| + ME + IDF (iterative direct fusion, concat mono depth + disparity each iter) | direct depth fusion | 1.41±0.01 / 10.34±0.19: no EPE gain, bad2 WORSE (+0.53) | | |
| + ME + PF (post fusion, no registration) | | 1.41±0.00 / 9.71±0.00: no gain | | |
| + ME + ILF (iterative local fusion) | local optima/noise | 1.20±0.08 / 9.06±0.70 (-0.22 EPE vs ME) | | tiny |
| + ME + ILF + GF (full) | scale ambiguity | 1.15±0.01 / 8.35±0.04 | | total 0.40 s |
| FE-DepthAnything (replace stereo feature extractor by DA-V2) | | 3.26±0.03 / 28.73±0.28 (supp Tab.13, p. 16) -- terrible | | |
| FE-MASt3R | | 4.41±0.40 / 26.83±0.57 | | |
| LBP kernel 1 / 3 (no sigmoid) | Tab.6 | L1: 1.38 / 10.20; L2: 1.36 / 10.10 | | |
| LBP 3 + sigmoid | | L3: 1.44 / 9.65 | | |
| LBP 5,3 + sigmoid (chosen) | | L4: 1.20±0.08 / 9.06±0.70 | | |
| LBP 9,7,5,3 | larger windows | L5: 1.32 / 9.57; L8 (13,11,9,7,5,3) 1.31 / 9.89 | | |
| learnable conv instead of fixed LBP | | L6 conv 1.32 / 9.53; L7 deeper conv 1.39 / 9.71 -> worse than fixed | | |
| amplitude r = 2 / 3 | | L9 1.32 / 9.45; L10 1.26 / 9.42 (text: r not significant); S-column flags for L9/L10 illegible in render | | |
| Global fusion: no registration, cost-conf | Tab.7 | G1 1.23 / 9.69 | | |
| Registered mono depth alone (MonoDepth) | | 1.19 / 8.72 | | |
| Reg + cost-only confidence | | G2 1.18 / 8.77 | | |
| Reg + hybrid confidence (cost+hidden+guidance) | | G3 1.15 / 8.35 | | |
| Why mono depth alone insufficient | Tab.4 | DA-V2 metric-finetuned EPE 205 / bad2 99.99; DA-V2 aligned with GT-derived scale/shift 5.83 / 69.28; Metric3D 33.14 / 97.18; ours 1.15 / 8.39 | | |
| Fusion with extra synthetic transparent data (TranScene) | | Booster-Q ALL EPE 2.26->1.24; Trans 7.93->5.67; Class2 5.32->1.62 (Tab.10-11) | still zero-shot, but extra training data | |

### 4. Interactions & dependencies
- Mono features without a fusion guard are worse than good-prior alone? Specifically: DA-V2 context features alone (ME) give the biggest jump; ILF refines on top. Naive depth fusion (IDF/PF) adds nothing on top of ME => fusion needs ordering guidance AND registration. Fixed-weight LBP > learnable conv (limited data makes learnable mono-related modules unreliable, p. 7). Guidance ramp r*t/T because early disparities are noisy (ordering maps from early t are wrong, Fig. 3 shows binocular ordering maps gradually resemble monocular ones). Global fusion needs the staged training (registration trained before confidence). Confidence benefits from guidance + hidden state, not cost alone. DA-V2 cannot replace the stereo matching feature extractor (Tab.13).

### 5. Losses
- Eq. 6: L = sum_{t=1}^T gamma^{T+2-t} ||D_d^t - D_G||_1 + gamma ||D~_m - D_G||_1 + ||D_f - D_G||_1; gamma = 0.9 (exponent "T+2-t" as printed). L1 on every iteration, registered mono, and final fusion.

### 6. Training recipe
- 4x NVIDIA A40, AdamW, one-cycle LR, SceneFlow only, DA-V2 frozen. Stage 1: without global fusion, max LR 2e-4, bs 8, 100k steps. Stage 2: train only mono registration part of global fusion (others frozen), LR 5e-4, bs 32, 100k. Stage 3: train full global fusion (confidence) others frozen, LR 5e-4, bs 32, 100k (p. 5). r=1, LBP windows 5,3. Crop size/augmentation: not stated. No extra stereo data (except the TranScene supp experiment, trained end to end with SceneFlow-pretrained weights, no multi-stage).

### 7. Results (all zero-shot from SceneFlow, no extra data; all pixels)
- Tab.1 (p. 6): KITTI-15 EPE 1.12 / bad3 5.60 (HVT-RAFT, extra aug: 1.12 / 5.20); KITTI-12 0.87 / 4.10 (NerfStereo with extra data 0.84/3.6); Middlebury-H All 1.15 / 8.39, NonOcc 0.85 / 5.67, Occ 2.89 / 26.50 (Mocha 24.16 better on Occ bad2); ETH3D EPE 0.25 / bad1 1.88. RAFT-Stereo official: 1.13/5.69, 0.9/4.35, 1.92/12.6, 0.36/3.3.
- Tab.2 Booster-Q: ALL EPE 2.26, bad2 11.02, bad3 8.59, bad5 6.6; Trans EPE 7.93 / bad2 59.83 / 50.36 / 38.44; NonTrans 1.52 / 6.98 (RAFT-Stereo+ME 1.45/6.96 slightly better). Prior Trans bad2: IGEV 68.96, Selective-IGEV 66.85, RAFT 67.69, NerfStereo (extra data) 62.67, RAFT-Stereo+ME 64.84.
- Tab.3 DrivingStereo EPE sunny/cloudy/rainy/foggy: 0.93/0.92/1.29/0.93 (NerfStereo 0.90/0.91/1.46/1.01).
- Runtime: 0.40 s vs RAFT-Stereo 0.32 s (p. 7; GPU, resolution and iteration count not stated). Memory (excl. feature encoder, Tab.12, supp): 750x2484 alt 3452 MB vs RAFT alt 1716 MB, reg 5031 vs 2268; scales like RAFT; IGEV/Selective-IGEV/Mocha OOM at large res.

### 8. Negative results & limitations
- Authors: failure on glass door/window (both surface and background matter; needs multi-depth-per-pixel representation) and very dark scenes/tunnel (registration hard); "information from video streams and segmentation becomes essential, like video stereo matching or simultaneously learning segmentation" (supp Sec. 13, p. 16; Fig. 22-23). Replacing stereo feature extractor by DA-V2 or MASt3R fails (Tab.13). Learnable/deeper LBP convs worse. ME alone is worse than RAFT on KITTI-15 (1.18/6.18 vs 1.13/5.69, Tab.1) -- the text claim "worse than RAFT-Stereo" (p. 7) holds on KITTI-15 only; on Middlebury/ETH3D/KITTI-12 +ME is better.
- Mine: heavy (0.4 s, ViT per frame, 32-iteration-class RAFT); not "real-time"; training from scratch per ablation with high std (±0.16 baseline); not best on every column despite "state-of-the-art" wording; baselines are official weights evaluated under their own metric; Booster-Q is quarter-res balanced set. Key per-region claim (stereo better at fine detail, mono better at overall shape) is visual only (Fig. 7-8).

### 9. Relevance to OUR model
- Not directly deployable (ViT DA-V2 + RAFT iterations), but several mechanisms are portable and cheap:
  1. Fixed-weight LBP-style neighborhood relation maps computed from a semantic/aux map and from the current disparity, feeding a gating head. For us: LBP of disparity candidates vs semantic-edge maps ("is neighbor in same class", class-boundary affinity from the 14-class logits), insertion in SemanticCostGate input at 1/8. Cost ~0. Benefit: explicit boundary-aware gating; evidence from D-FUSE that fixed weights beat learnable ones when data are small (relevant to our 800-pair training). Risk: ordering of classes is meaningless; use same-class equality maps, which is a different (untested) quantity. Novel.
  2. Iteration-ramped guidance (guide strength r*t/T) applies to any iterative refiner; our ClassResidual is one-shot so mostly N/A; could ramp across tile refinement stages of FusionStereoLite.
  3. Global registration + confidence blend (a,b,c) is the principled way to inject an affine-ambiguous prior; applicable only if we add a mono-depth source (we have none now; our semantic branch has no depth). A cheap version: per-class affine (a_c,b_c) of stereo disparity, which resembles ClassResidual.
  4. Staged freezing (train registration, then confidence) mirrors our frozen-predictor policy; supports training only the fusion head.
  5. Lesson: naive fusion of strong-prior features can hurt (RAFT context features hurt generalization: 1.83 vs 2.11) -- supports using an equal-capacity no-semantics control (our C4/D3 controls). Their authors name segmentation as the future fix for glass/dark cases: supports our direction but unevidenced.
- Already done by us: semantic gating (D2/E3) and no-semantics controls. Not done: LBP relation maps, confidence-blend output fusion, registration.

### 10. Key quotes/equations
- Eq.1: M_O(u,v)={sigma(D(u',v')-D(u,v))}, (u',v') in N (p. 4). Eq.2: Delta~_d = Delta_d(1+G r t/T). Eq.4-5 registration and confidence blend (p. 5).
- "The iterative direct fusion ... more robust ... [ILF]"; "fixed-weight convolutions are more robust than learnable convolutions ... limited data makes monocular-related learning unreliable for generalization" (p. 7).
- "information from video streams and segmentation becomes essential" (p. 16).

### Errors in existing summary (summaries/fusion/D-FUSE.md)
1. "SOTA zero-shot across all five real-world datasets": false per paper's own Tab. 1/2: KITTI-15 bad3 5.60 vs HVT-RAFT 5.20; KITTI-12 EPE/bad3 0.87/4.10 vs NerfStereo 0.84/3.6 (extra data); Middlebury Occ bad2 26.50 vs Mocha 24.16; Booster NonTrans EPE 1.52 vs 1.45.
2. Ordering-map direction "near 1 if neighbor is farther": paper does not state direction, and D is inverse-depth-like (p. 3), so sigma(D(u')-D(u)) is large when the neighbor is CLOSER. Summary's claim is probably reversed/unsupported. Also maps are soft (sigmoid), not binary.
3. "D-FUSE is the first paper to systematically analyze..." the paper does not claim "first".
4. "0.08 s overhead ... suggests it could work on edge devices": 0.32 -> 0.40 s is correct (p. 7) but hardware/resolution unstated, and total runtime is non-real-time with a frozen ViT; relevance claim is speculative. The summary also omits DA-V2's ViT cost and the staged training/ramped guidance, and that Baseline w/o mono feature beats baseline.
5. "10-point improvement for transparent bad2": true only vs weakest priors (~70 -> 59.83); vs best comparable without extra data (Selective-IGEV 66.85) it is ~7 points, vs NerfStereo 2.8.
Numbers 1.12/5.60, 1.15/8.39, 0.25/1.88, 2.26/11.02, a,b,c equations: verified. Training-stage and key hyperparameters are absent from summary.
