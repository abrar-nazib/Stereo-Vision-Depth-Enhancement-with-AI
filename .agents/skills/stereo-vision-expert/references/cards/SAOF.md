<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SAOF block card

Evidence grade: **C** (training-free classical pipeline; ablation and headline numbers are on the same 10 hand-picked US3D images; no non-semantic region-partition control; internal table inconsistencies). Remote-sensing transfer caveat in section 9.

### 0. Meta
- Title: SAOF: A Semantic-Aware Optical Flow Framework for Fine-Grained Disparity Estimation in High-Resolution Satellite Stereo Images. Dingkai Wang, Feng Wang, Jingyi Cao, Niangang Jiao, Yuming Xiang, Enze Zhu, Jingxing Zhu, Hongjian You (AIRCAS / UCAS / Tongji). Remote Sensing 2025, 17, 4017 (published 12 Dec 2025). 31 PDF pages incl. appendices and references, read fully. Page refs: "p. N of 31".
- PDF: paper/reference_papers/semantic_stereo/SAOF_Wang_RemoteSensing2025.pdf. Code: https://github.com/PacificRobot/SAOF (p.24).
- Domain: remote sensing (WorldView-3 pseudo-epipolar pairs; 8 km x 8 km SuperView-1 qualitative). Dataset: US3D Track-2, Jacksonville 2135 + Omaha 2152 pairs, 1024x1024, GSD about 0.3 m (Tab. 1, p.13).
- No existing summary under paper/reference_papers/summaries/ for this paper.

### 1. Problem & failure modes targeted
- Large disparities, textureless regions (rooftops, gobi, lakes, runways), structurally complex buildings, edge loss from cost aggregation; residual y-disparity from pushbroom pseudo-epipolar rectification (p.2-3). Top-pyramid error propagation in coarse-to-fine flow (p.8). Illumination change (Census cost).
- Optical-flow-based disparity gives sub-pixel output and tolerates small vertical residuals (p.3).

### 2. Pipeline by stage
This is NOT a learned network. No training occurs anywhere in SAOF; the only learned component is the off-the-shelf SAMgeo model.
- 2a Feature extraction: SAMgeo (SAM tuned for geospatial imagery) encoder, used only inside SAMgeo-Reg to get region prototypes (p.6). Matching itself uses raw color, gradient and Census on pixels (App. A). Frozen: SAMgeo untouched; resolution/channel counts of its features: not stated.
- 2b Semantic / prior branch (SAMgeo-Reg, Fig. 2, p.6-7): SAMgeo decoder produces candidate class-agnostic masks per image; per-mask prototype = mean of encoder features over the mask; cosine similarity between all left and right prototypes; pairs above threshold tau = 0.9 (Tab. 2, p.14) are matched, each matched pair gets a unique index converted to a gray value, yielding two semantic guidance maps S1, S2 (instance-level, no class names, label-free). Post-processing: morphological hole filling, Gaussian smoothing, Douglas-Peucker contour simplification (Fig. 3, p.7).
- 2c Cost volume: none. Block-matching cost D(p,q) over a window, no volume; 1D horizontal epipolar search by PatchMatch-style random search + propagation (App. B, Algorithm A1).
- 2d Aggregation: window cost with bilateral weights omega_bil (Eq. 3, A6), robustified per feature by G(d) = 1 - exp(-d^2/lambda^2) (Eq. 2). Cost = G(d_col)+G(d_grad)+G(d_cen) (Eq. 4); feature weights 0.2, 0.1, 0.1 (Census weighted highest) (Tab. 2). d_col = max over RGB channel difference (A1), gradient magnitude via central differences (A2-A4), Census XOR (A5).
- 2e Disparity computation: forward flow and backward flow, forward-backward consistency with level-dependent threshold eps_L = eps_3 - floor(log2(W3/WL)) (Eq. 6), weighted median per level, sub-pixel by 2D paraboloid fit around candidates (p.11). Flow converted to disparity.
- 2f Refinement / iterative: 4-level pyramid (P=4, L3..L0). Sub-top pyramid re-PatchMatch (STPR): after the top-level flow, a second self-similarity propagation + random search runs at level 2 (Fig. 4, p.9). Self-similarity propagation (large-displacement flow of Bao et al., ref [49]) with scale-adaptive window R(w) = R0*(1+floor(k*max(0,log2(w/theta)))) (Eq. 5), R = 17 (Tab. 2), n random candidates scanned forward and backward.
- 2g Upsampling: joint bilateral upsampling plus local matching, level by level (p.9).
- 2h FUSION POINTS: one. Stage: cost computation. Operator: multiplicative cost-window weight omega_seg(p,q) = exp(-E(p,q)/sigma_r^2) (Eq. 9) multiplying omega_bil (Eq. 11); E is the color cost of neighbors that fall in a different semantic region than the window center (zero if the neighbor shares the center's region in both views, Eq. 7-10). Direction: regions -> matching cost; no disparity -> segmentation feedback. Resolution: each pyramid level; guidance map resolution handling not stated.

### 3. Block -> problem -> evidence table
Ablations were run "on the test images presented in Section 4" (p.20) = the 7 complex + 3 textureless hand-picked US3D images; averages over those. Time is per image, hardware/units basis not stated (seconds).
| block | problem it solves | evidence (ablation delta with ref) | context | cost |
|---|---|---|---|---|
| SAMC (semantic constraint) alone | cross-region mismatch, flat regions | complex: EPE 1.566 -> 1.417 (-9.5%), D1-3 12.56 -> 10.32 (-17.8%); textureless: 1.371 -> 1.305 (-4.8%), D1-2 5.90 -> 5.78 (-2.0%) (Tab. 8/9, p.20-21) | 7 + 3 hand-picked test images | +0.091 s (0.779 -> 0.870) complex; +0.071 s textureless |
| STPR alone | large disparity, top-level error | complex: 1.453 (-7.2%), D1-3 10.27 (-18.2%); textureless 1.338 (-2.4%) (Tab. 8/9) | same | +0.137 s complex |
| SAMW alone | scale-adaptive window | complex 1.514 (-3.3%), D1 12.48; textureless 1.345 (-1.9%) | same | +0.007 s |
| STPR + SAMC | best pair | complex 1.335 / 9.31; textureless 1.296 / 5.35 (Tab. 8/9) | same | +0.252 s complex, +0.250 s textureless |
| SAMW + SAMC | pair | complex 1.397 / 9.77; textureless 1.318 / 5.66 | same | - |
| SAMW + STPR | pair | complex 1.434 / 10.14; textureless 1.322 / 5.51 | same | - |
| all three (full SAOF) | claimed | no all-three row in ablation tables; Tab. 4/5 full SAOF = 1.317 / 9.09 (complex), Tab. 6 = 1.258 (textureless); these do not equal any ablation row (best ablation 1.335 / 1.296), unexplained | - | - |
| Multi-feature cost (color+gradient+Census) | illumination/weak texture | not ablated | - | - |
| Bilateral weighting | edge preservation | not ablated | - | - |
| Semantic-region choice (SAMgeo vs. superpixels / random regions / non-SAM) | claim: semantic priors help | not ablated: no control that substitutes a non-semantic region partition | - | - |
| Tau (0.9), R (17), lambda weights, sigmas | thresholds | not ablated; selection procedure not stated | - | - |
| SAM mask errors | robustness | qualitative only (Fig. 12, p.22-23) | 3 regions | - |

### 4. Interactions & dependencies
- STPR and SAMC are complementary: STPR alone fixes large-disparity levels but blurs edges; SAMC alone keeps edges but is weak on large disparity; together best (Fig. 11, p.22).
- SAMC gain is larger in complex regions (-9.5% EPE) than textureless (-4.8%) (Tab. 8/9) but the paper says SAMC is the biggest gain in textureless regions (p.21), true only relative to STPR/SAMW there.
- Needs pseudo-epipolar rectified pairs; semantic maps need left/right region correspondence via SAMgeo features; a wrong match removes the penalty for that region.

### 5. Losses
- None: training-free. Only data-term formulas (Eq. 1-4, 9-11), the dynamic consistency threshold (Eq. 6) and the window-size rule (Eq. 5).

### 6. Training recipe
- None for SAOF. Baseline HMSMNet trained 100 epochs, LR 1e-3 halved every 10 epochs (p.14); its training set, split, and crop are not stated. SGM block 11, P1 = 8*C*block, P2 = 64*C*block; AD-Census 9x7 window, lambda_census 30, lambda_AD 10; Gefolki 4 levels, 5 iterations, median 5; MGM 8 directions (p.14). Implementation hardware: not stated.

### 7. Results
- Complex regions, 7 images (Tab. 4/5, p.19-20): average EPE SAOF 1.317, HMSMNet 1.360, MGM 1.552, SGM 1.656, Gefolki 1.861, ADC 1.945; D1-3 SAOF 9.09%, HMSMNet 10.14%, MGM 11.29%, SGM 12.01%, ADC 12.19%, Gefolki 14.15%. SAOF is not best on every image: HMSMNet wins EPE on images I (1.231 vs 1.261) and IV (0.866 vs 0.901), and D1 on IV and V.
- Textureless, 3 images (Tab. 6, p.19): average EPE SAOF 1.258, SGM 1.403, MGM 1.362, Gefolki 1.491, ADC 1.495, HMSMNet 1.448. D1-2 per image (Tab. 7): SAOF 2.240 / 12.94 / 0.460.
- Margin over the only learned baseline HMSMNet: 0.043 px EPE (3.2%) on complex regions. The abstract's "EPE of 1.317 and D1 9.09%" is a 7-image average, not the US3D test set.
- Runtime: about 0.78-1.03 s per image (Tab. 8); not real-time. No params/FLOPs (non-learned).

### 8. Negative results & limitations
- Authors: only WorldView-3 data tested; SAM errors affect results (Fig. 12); future work on semantic disentanglement and semantic-constrained search range (p.24).
- Mine: (i) 10 hand-picked images, no full-test-split evaluation; (ii) the learned baseline is trained by the authors on unspecified data; (iii) Tab. 7 "Average" row is a copy of Tab. 5's averages (12.01, 11.29, 14.15, 12.19, 10.14, 9.090), but averaging the per-image numbers in Tab. 7 gives about 6.81 / 6.90 / 7.96 / 6.30 / 6.14 / 5.21 for SGM / MGM / Gefolki / ADC / HMSMNet / SAOF (my arithmetic) - an editing error; (iv) full-model numbers (Tab. 4/6) do not match any ablation row; (v) no control separating "semantic" from "any region-consistency prior"; (vi) hyperparameters (tau 0.9, R 17) tuned on unknown data; (vii) no tests with a different satellite or sensor except qualitative SuperView-1.

### 9. Relevance to OUR model
- Idea: penalize cost contributions from neighbors in a different semantic region (edge-aware aggregation with region labels). It is a training-free hand-crafted analogue of our SemanticCostGate. In our pipeline the analogue would be a per-pixel "same-region" affinity from the frozen 14-class decoder (argmax or soft-probability dot product between a pixel and its neighbors) feeding the ClassResidual smoothing or tile-plane refinement of A09. Expected benefit: sharper class boundaries (more helpful on thin poles/cars at depth discontinuities); cost: small (a 3x3/5x5 affinity from logits); risk: our semantic decoder is 14 VKITTI classes at coarse resolution, boundary-quality uncertain; the paper gives only -9.5% EPE on a 7-image classical pipeline and no control, so it is hypothesis, not evidence.
- Not portable: SAMgeo instance prototypes + cosine left/right region registration (heavy, offline, not class-based), PatchMatch large-disparity flow, STPR (no pyramid/top-level error in our pipeline).
- Remote-sensing caveat: constant disparity inside a segment (flat rooftops, ground) is plausible for nadir satellite imagery; in driving views a road or facade is a slanted plane spanning the whole disparity range, so "intra-region smooth, inter-region discontinuous" is only a boundary cue (discontinuity at class change), not a within-region constancy prior. Our E3 already beats equal-capacity no-semantics controls (E4); SAOF does not add a causal control and does not change our direction.
- Novelty note: region-consistency penalty in cost is old (SegStereo-like, SGNet-style gating already inspired E3). Using class-boundary affinity for edge-preserving residual propagation is a possible small ablation (G-series style), low priority.

### 10. Key quotes/equations worth citing
- Eq. 9-11 (p.12-13) semantic penalty into bilateral-weighted cost; Tab. 8 (p.20) ablation; "SAMgeo-Reg enables semantic-aware disparity estimation in a label-free manner" (p.3).
- PDF: paper/reference_papers/semantic_stereo/SAOF_Wang_RemoteSensing2025.pdf
