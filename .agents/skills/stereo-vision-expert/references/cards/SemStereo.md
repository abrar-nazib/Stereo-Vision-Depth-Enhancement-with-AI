<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SemStereo card

PDF (9 pp, AAAI 2025 main text incl. refs; no supplement) read in full; Fig. 3 and Eq. 1-9 verified from page renders (pp.3-4).

### 0. Meta
- Title: SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing
- Authors: Chen Chen, Liangjin Zhao, Yuanchun He, Yingxuan Long, Kaiqiang Chen, Zhirui Wang, Yanfeng Hu, Xian Sun (AIRCAS)
- Venue: AAAI 2025 (arXiv 2412.12685)
- PDF: paper/reference_papers/semantic_stereo/SemStereo_Chen_AAAI2025.pdf
- Code: https://github.com/chenchen235/SemStereo (p.1)
- Domain: REMOTE SENSING (satellite US3D, aerial WHU) - not driving. The method's motivation (disparities of one class concentrated in a narrow range, Fig. 2) is explicitly stated NOT to hold for ground-level images (p.1).
- Datasets/splits (p.4): US3D Jacksonville 2,139 pairs -> 1,500 train / 139 val / 500 test (random selection); US3D Omaha 2,153 pairs used for cross-city generalization (zero-shot; 50/500-pair fine-tune with 1,500 val); WHU aerial 8,316 train / 2,618 test, NO semantic labels (only the "SemStereo*" variant is run).

### 1. Problem & failure modes targeted
- "Constraints between the two heterogeneous tasks are not explicitly modeled": parallel two-branch nets share only shallow features -> "weak and implicit" coupling (Abstract, p.1-2).
- Task-conflict statement: none explicit. The paper's thesis is the opposite (tasks are complementary), and it modifies sharing to be DEEPER (decoder-end features), with a ablation showing no seg loss from doing so (Tab. 1).
- Targets: boundaries in dense buildings, small objects, textureless regions, occlusion; cross-city generalization (Fig. 4-5, Tab. 3).

### 2. Pipeline by stage
- 2a Feature extraction: SHARED U-shaped extractor: MobileViTv2 encoder + decoder with skip connections & transposed convs, giving multi-scale features D^l_i, D^r_i in R^{C'_i x H/i x W/i}, i in {2,4,8,16,32} (p.3). Shared by both views and both tasks, trained end-to-end, nothing frozen. Pretraining of MobileViTv2 not stated. Channel counts not stated.
- 2b Semantic branch: attached to the DEEPEST (= highest-resolution, i=2, post-decoder) feature volume D_2: conv + upsample + softmax -> P^l, P^r in R^{N x H x W} (p.3). I.e. the semantic head sits at the end of the shared decoder, and its input features are the same decoder features that feed the cost volume (cascade). N classes = US3D 5 classes (Ground, Trees, Building Roof, Water, Bridge/Elevated Road; Tab. 4).
- 2c Cost volume: 1x1 "siamese" convs halve channels of D_i -> T_i in R^{C''_i x H/i x W/i}, C''_i = C'_i/2; then Fast-ACV attention-concatenation volume V in R^{C''' x Dmax/2 x H/4 x W/4} at 1/4 resolution (with disparity range [-Dmax, Dmax-1] adaptation for satellite negative disparity). US3D range [-64,64), WHU [0,128).
- 2d Aggregation: Fast-ACV regularization (attention weights from correlation volume filter the concatenation volume, top-k prior etc., inherited unchanged from Fast-ACV).
- 2e Disparity: Fast-ACV output d_init in R^{1 x H/4 x W/4}.
- 2f Refinement: SSR branch (below) - not iterative.
- 2g Upsampling: d_init bilinearly upsampled to d'_init in R^{1xHxW}.
- 2h FUSION POINTS:
  1. SGC (feature level, implicit, seg->stereo and stereo->seg): the same decoder features D_i serve both the semantic head (at D_2) and the cost volume (T_i) -> a "cascade" rather than parallel. Direction both (shared gradients).
  2. SSR (refinement, explicit, seg->disp, multiply-gate): feature volume F in R^{N x H x W} (built by progressively upsampling and concatenating the T_i; N channels = class count) carries both semantic and disparity info. Eq.1: F' = σ(F · P^l) · F, where "F · P^l" = inner product/elementwise coupling with the N-channel left softmax map and σ = 1x1 conv + BN + sigmoid (channel/spatial attention gate). Then a second weight map σ(F') filters d''_init (d'_init normalized, 3x3 conv expanded to N channels): R = Conv(σ(F') · d''_init) (Eq.2, 1x1 conv to 1 channel) and d_final = R + d'_init (Eq.3). So the gate is a per-class soft mask: class-wise confidence selects which class channel of the expanded disparity contributes to the residual, i.e. "intra-class disparity consistency". Uses left view only. Text before Eq.3 says "adding with d''_init" but Eq.3 and shapes (1 channel) require d'_init: a typo in the paper; the repo summary copied the typo (d''_init).
  3. LRSC (loss-only, seg<->disp): warp left labels (or left prediction when no GT) to the right view with d_final, CE against P^r (Eq.4-6).
  Semantic labels are used only through the seg head loss and LRSC; there is no pretrained frozen seg model.

### 3. Block -> problem -> evidence table
All US3D Jacksonville test (500 pairs), EPE px / D1 % / mIoU % / PA % (Tab. 1, p.5). "*" = no explicit semantic labels (semantic labels not used; how alpha/beta are set for these rows is not stated).
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Baseline (parallel two-branch, Fast-ACV stereo, shallow sharing; exact baseline spec not stated) | reference | 1.2087 / 7.28 / 75.84 / 93.65. Baseline*: 1.2260 / 7.47. So label-supervised seg in the parallel baseline only buys 0.017 px EPE / 0.19 D1 for stereo. Tab. 2 Fast-ACVNet alone reports 1.1706 / 7.06 (better than this "Baseline" 1.2087 -> baseline is not exactly Fast-ACV) | | |
| SGC (deep shared features into cost volume) | weak coupling | row 2 vs 1: EPE -17.3% (0.9995), D1 -31.6% (4.98); mIoU 75.74 vs 75.84 (-0.10, no seg gain), PA +0.05. Without labels: SGC-Net* 1.0499/5.61 vs Baseline* 1.2260/7.47 = EPE -14.4%, D1 -24.9%. CONFOUND: ~80% of the SGC EPE gain survives with NO semantic supervision, i.e. most of it is architectural (deeper U-shaped MobileViT decoder features into the cost volume), not semantics. Semantics-attributable part (SGC-Net vs SGC-Net*): EPE -4.8% (1.0499->0.9995), D1 -11.2% | single run | |
| SSR | refine with class-selective residual | row 3 vs 2: EPE 0.9995->0.9702 (-2.9%), D1 4.98->4.76 (-4.4%), mIoU +1.11 (75.74->76.85). Row 7 vs 6 (no labels): 1.0499->1.0164, 5.61->5.34 | replaces plain bilinear upsampling | |
| LRSC | cross-view semantic consistency | row 4 vs 3: EPE 0.9702->0.9582 (-1.2%), D1 4.76->4.58 (-3.8%), mIoU +0.17, PA +0.30. Without labels (self-supervised from P^l): 1.0164->0.9956, D1 5.34->5.00 | | free at inference |
| Semantic supervision overall (full vs *) | | row 4 vs 8: EPE 0.9582 vs 0.9956 (-3.7%), D1 4.58 vs 5.00 (-8.4%) | | |
| Stereo supervision for seg (Tab. 4) | | SemStereo* (seg w/o stereo loss) mIoU 67.57 / PA 92.99 vs SemStereo 77.02 / 94.13 (+9.45 mIoU). Baseline joint model already 75.84 -> most of this +9.45 is just "having any stereo loss" (a seg-only run of the same net would be the control; labelled * "without stereo matching supervision") | | |
| Shared MobileViTv2 U-Net vs separate encoders | | not ablated (the cascade variant is never compared against two separate encoders) | | |
| Intra-class disparity assumption | | only motivated by Fig. 2 histogram; no per-class ablation | | |
| Dice+CE seg loss; lambda_i multi-stage weights | | not ablated | | |
| Cross-city zero-shot (Tab. 3) | | EPE 1.4996 vs IGEV 1.5120, PSMNet 1.5163 (best), but D1 9.70 is WORSE than PSMNet 9.27, GwcNet 9.32, ACVNet 9.36, HMSMNet 9.41, IGEV 9.91 (better only vs StereoNet 11.76, Fast-ACV 11.13). After 500-pair fine-tune: 1.1002/4.54 (best) | Jacksonville->Omaha | |

### 4. Interactions & dependencies
- SSR needs the seg head's probability map P^l (class-confidence gate) AND the feature volume F; ablated only on top of SGC, never alone.
- LRSC with GT labels warps hard labels by the predicted disparity: label indexing is non-differentiable w.r.t. d_final, so in the labelled case LRSC mainly produces right-view pseudo-labels (noisy at occlusions) that supervise P^r, i.e. seg-side supervision; whether it really back-propagates to disparity is not stated (my inference). In the self-supervised case warp(P^l) can be differentiable. The reported EPE gain of LRSC (1.2%) is therefore likely seg-side regularization; unproven.
- Everything depends on the remote-sensing prior (class -> narrow disparity range). The authors explicitly say it does not apply to ground-level images.
- Fast-ACV top-k/priors inherit; disparity range must be negative-to-positive adaptation for satellites.

### 5. Losses (exact)
- L_Seg = L_CE(L^p, L^gt) + L_Dice(L^p, L^gt) (Eq.7).
- L_Disp = sum_i lambda_i SmoothL1(d_i - d_gt) (Eq.8) over multi-stage predictions d_i; lambda_0=1, lambda_1=0.6, lambda_2=0.5, lambda_3=0.3 (p.5). Which d_i (initial, final, intermediate Fast-ACV stages) not itemized.
- L_LRSC = L_CE(P^r, GT^r), GT^r = warp(GT^l, d_final) (labels available) else warp(P^l, d_final) (Eq.4-6; disparity = x_l - x_r).
- L = L_Disp + alpha L_Seg + beta L_LRSC, alpha = beta = 1 (Eq.9).
- SSR gate (Eq.1-2) is architecture, not loss.

### 6. Training recipe
2x NVIDIA A40; Adam (0.9, 0.999); batch 4; original resolution, NO augmentation; "train each stage for 48 epochs" (stages not defined in text); lr0 = 0.001 halved after epochs 12, 22, 30, 38, 44; US3D disparity [-64,64), WHU [0,128). Fine-tune Omaha 12 / 48 epochs for 50 / 500 pairs (p.5). Baselines "standardized": same optimizer, batch, resolution. Pretraining of MobileViTv2 / checkpoint selection not stated. No parameter count/FLOPs/latency anywhere.

### 7. Results
- US3D Jacksonville stereo (Tab. 2): SemStereo 0.9582 EPE / 4.58 D1; SemStereo* 0.9956 / 5.00; Fast-ACVNet 1.1706 / 7.06; PSMNet 1.1770 / 6.87; IGEV 1.2051 / 7.32; GwcNet 1.2120 / 6.99; ACVNet 1.2836 / 7.73; HMSMNet 1.2338 / 7.91; S3Net 1.403 / 9.58 (official numbers); StereoNet 1.6053 / 12.13. WHU: SemStereo* 0.2236 / 0.731 vs Fast-ACVNet 0.2257 / 0.740 (only 0.9% better; the cascade gives almost nothing on WHU where no labels exist), PSMNet 0.2432 / 0.814.
- US3D seg (Tab. 4): SemStereo 94.13 PA / 77.02 mIoU (Ground 90.84, Trees 74.63, Building 88.30, Water 68.94, Bridge 62.37) vs PSPNet 67.06, DeepLabV3 66.53, UNet 65.98, SegFormer 63.60, S2Net 69.10 (official), S3Net 67.39; SemStereo* 67.57.
- Cross-city (Tab. 3): above.
- DOES STEREO IMPROVE? Yes, large on US3D vs Fast-ACV (-18% EPE), but ~80% of it is present without any semantic label (SGC-Net* vs Baseline*), so the semantic share is ~5% EPE, ~11% D1. DOES SEG IMPROVE? +9.45 mIoU vs seg trained without stereo loss, but in the parallel baseline the joint seg is already 75.84, so cascade+SSR+LRSC adds only +1.18 mIoU over that baseline.

### 8. Negative results & limitations
- SGC does not improve seg (75.84 -> 75.74).
- Zero-shot D1 not best (above); WHU gain tiny.
- Authors: "semantic instances exhibit a closer relationship with disparities compared to semantic categories" (p.5) -> future work: class-level prior is only partially valid.
- My critique: confounded SGC ablation (architecture vs semantics); baseline spec vague (and not equal to Fast-ACVNet in Tab. 2: 1.2087 vs 1.1706); single seed/run; US3D test sampled randomly from one city by the authors (random pair selection, adjacency leak possible); S2Net/S3Net numbers are copied from other papers with different splits (marked †), unfair; semantic comparison models (FCN, UNet, DeepLabV3, PSPNet, SegFormer) are single-task, trained at unknown resolution; no latency/params; remote sensing prior does not transfer to driving (acknowledged in Fig. 2 discussion).
- Repo summary errors: (a) d_final = d''_init + R (typo copied from the text; shapes require d'_init); (b) summary says the SSR gate "per-class confidence" without noting P^l is the left view only; (c) it omits the key control that SGC-Net* gets ~80% of the stereo gain without labels; (d) summary table row "WHU, SemStereo without explicit semantic labels" fine.

### 9. Relevance to OUR model
- Closest-to-ours idea: SSR = class-probability map used as a multiplicative gate on a feature volume, then a residual added to the upsampled initial disparity. This is structurally almost identical to our D2/E3 (SemanticCostGate + ClassResidual): they use P^l softmax confidence as the gate (we use the 14-class decoder logits/features) and a residual head. So SSR is NOT a novel combination relative to us; it confirms the design, and also that SSR alone adds only ~3-4% (row 3 vs 2: -2.9% EPE, -4.4% D1, +1.1 mIoU) in remote sensing, on top of a stereo model whose decoder was already changed.
- Portable pieces: (1) Eq.1-style soft per-class gate on the feature volume before the residual (we already have an analogue; check whether ours uses class-confidence rather than logits, and whether a double-gate sigma(F') filter on the upsampled initial disparity helps). Cost: negligible convs at 1/4. Risk: low. (2) LRSC: cross-view semantic consistency loss. Not directly usable because our seg decoder is frozen and our disparity head is trained; but it can be a TRAIN-ONLY auxiliary on the fusion head: warp left semantic logits to the right using predicted disparity and compare to right-view logits from the frozen decoder (both views run through the frozen trunk once already). Gradient reaches the disparity only through a differentiable (bilinear) warp of the left logits (the paper's hard-label version likely does not). Expected benefit: small in-domain (1-4% by their ablation); possible occlusion/consistency regularizer; zero inference cost; moderately novel for a frozen-predictor head (needs a bilinear-sampling warp; we tried a stereo-warp correction in G3 which did not help, so expectation low). Risk: warping errors at occlusions, requires masking by LR check.
- Do NOT adopt the intra-class-disparity-concentration assumption; false in driving (their own Fig. 2).
- Task-conflict evidence: deeper sharing did not hurt seg here (-0.10 mIoU) but their win is confounded by architecture. For our frozen shared trunk the lesson is limited: our trunk is frozen, so no conflict arises by construction.

### 10. Key quotes/equations
- "the Semantic-Guided Cascade structure ... deep features enriched with semantic information are utilized for the computation of initial disparity" (Abstract).
- Eq.1 F' = σ(F·P^l)·F; Eq.2 R = Conv(σ(F')·d''_init); Eq.3 d_final = R + d'_init (p.4).
- Eq.5-6 LRSC; Eq.9 L = L_Disp + αL_Seg + βL_LRSC, α=β=1 (p.4).
- "disparities corresponding to the same category are concentrated within a distinct and narrow range in remote sensing images, while this characteristic does not apply to typical images taken from a ground-level perspective" (p.1).
- Table 1 (p.5): the SGC-Net* vs Baseline* rows are the key unpublicized evidence that the architecture, not semantics, drives most of the gain.
