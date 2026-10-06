<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SGNet card

Read: whole PDF (17 pp = 14 pp body + references; the PDF has NO supplement/appendix). All refs are to the PDF's printed page numbers (p. N) unless Fig./Tab./Eq.

### 0. Meta
- Title: SGNet: Semantics Guided Deep Stereo Matching
- Authors: Shuya Chen, Zhiyu Xiang (corresponding), Chengyu Qiao, Yiman Chen, Tingming Bai (Zhejiang Univ.)
- Venue: ACCV 2020 (CVF author version; p. 1)
- PDF: paper/reference_papers/semantic_stereo/SGNet_Chen_ACCV2020.pdf
- Code URL: not stated in the PDF.
- Domain: driving (urban street scenes).
- Datasets: Scene Flow (pretrain stereo branch only, no semantic labels; 35,454 train / 4,370 test), KITTI 2015 (200 train imgs WITH semantic labels; ablations use 160 train / 40 val; 200 test), KITTI 2012 (194 train, no semantic labels; 195 test), Virtual KITTI 2 (val = "15-deg-left" subsequence of sequence 2, 233 imgs; train = same subsequences of the remaining sequences, 1,893 imgs) (p. 9-10).

### 1. Problem & failure modes targeted
- Stated targets (p. 1): lack of reliable scene cues, illumination change, occlusion, low texture in stereo matching. Semantics supplies "high-level" cues.
- Concrete mechanisms claimed:
  1. Wrong cost-volume candidates: a candidate disparity whose left/right pixels have different semantic category is unlikely to be a true match; "initial probability distribution drifted from GT" gets re-weighted (Fig. 3b, p. 6-7).
  2. Piecewise-smooth category surfaces: road is flat/smooth, trees uneven, so refinement should be category-dependent (p. 5-6).
  3. Over-smooth / noisy disparity inside objects and at boundaries: semantic boundaries should coincide with disparity boundaries for foreground objects, semantic interiors should be smooth in disparity (p. 7-9).
- Context: PSMNet-style 3D cost-volume network, KITTI/vKITTI in-domain. No claim about thin structures, domain shift, or latency beyond "run time similar to baseline".

### 2. Pipeline by stage
Built on PSMNet [15] (disparity branch) + a semantic branch + three semantic-guided modules (Fig. 1, p. 4).
- 2a Feature extraction: PSMNet weight-sharing siamese encoder + spatial pyramid pooling -> features at H/4 x W/4 (the confidence module states "H/4 x W/4" inputs, p. 7). Channel counts, depth: not stated in this paper (inherits PSMNet). Trained (not frozen), end to end.
- 2b Semantic branch (p. 5, Sec. 3.2): shares the "shallow layers" of the disparity branch (exact split point not stated), then its own independent high-level layers = "two more residual blocks with 256 channels", a pyramid pooling module, and a classification layer. Left and right images both go through it (Fig. 1 shows two Semantic-branch boxes; features C, D are the left/right semantic features, A, B are the left/right disparity features after pyramid pooling). Trained jointly with L_sem (not frozen). Class count: not stated explicitly (KITTI-15 semantic labels; residual module uses C classes). Semantic mIoU 48.12 / mAcc 55.25 on the KITTI-15 val split (p. 12).
- 2c Cost volume: PSMNet 4D concat volume: left features concatenated with right features shifted along x for each disparity (p. 3). Max disparity 192 (p. 8). At 1/4 resolution so D/4 = 48 levels (the 1/4 factor is PSMNet's, not restated by SGNet).
- 2d Aggregation: PSMNet 3 stacked hourglass ("3D conv layers" boxes) giving three outputs; each output is bilinearly upsampled to full size then soft-argmin regression -> disp1, disp2, disp3 (Fig. 1). disp3 is PSMNet's final output.
- 2e Disparity computation: soft-argmin ("sum of disparity weighted by predicted probabilities", p. 3) on the cost volume upsampled by bilinear interpolation.
- 2f Refinement: the Residual module (see 2h) produces disp4 = disp3 + category-dependent residual. Single pass, not iterative. SGNet's final output is disp4.
- 2g Upsampling: bilinear upsample of the cost volume before regression (as in PSMNet). Residual module's depthwise stride-2 conv + transposed conv is a down/up pair at full-res.
- 2h FUSION POINTS (three):
  1. **Confidence module** (Fig. 3a, p. 6-7). Stage: cost volume / aggregation. Operator: multiply-gate. Direction: sem -> disp. Resolution: H/4 x W/4 x (disparity levels).
     - Inputs: disparity features A (left), B (right) and semantic features C (left), D (right), all at H/4 x W/4. Channel counts: not stated.
     - Step 1: two correlation layers, Corr(x,y,d) = (1/N_c) * <f1(x,y), f2(x-d,y)> (Eq. 1) - one on (A,B) -> "Disp-cor", one on (C,D) -> "Seg-cor". Each gives a (D_levels x H/4 x W/4) volume; normalized by channel count N_c. No learned weights in the correlation.
     - Step 2: the two correlation volumes are multiplied element-wise (Table 1: x is better than +).
     - Step 3 (Fig. 3a): product -> Conv3d K3 S1 -> Conv3d K3 S2 -> TransConv3d K3 S2, with a skip that ADDS the product back (residual) -> Sigmoid. Text says "three consecutive 3D convolution layers with a residual structure" (p. 7). K3S2 = kernel 3, stride 2. 3D conv channel widths, activation, norm: not stated. Output spatial/disparity size equals the cost volume (stride-2 then transposed stride-2).
     - Step 4: the sigmoid confidence (values in (0,1)) is multiplied into "disp1's cost volume" (p. 7; Fig. 1 places the multiply on the output of the FIRST hourglass, immediately before Bilinear+Regression 1). Whether the gated volume also feeds hourglass 2/3 is not stated; Fig. 1 shows the downstream hourglass chain unchanged, so probably it only changes disp1's regression and (via the supervised disp1 loss) the shared first-stage weights. Authors' rationale: "better to improve the disparity in the early stage".
     - Confidence output channel count (presumably 1, broadcast against the 1-channel cost volume): not stated.
  2. **Residual module** (Fig. 2, p. 5; text p. 5-6). Stage: refinement. Operator: category-wise multiplication then depthwise conv. Direction: sem -> disp. Resolution: full H x W.
     - Inputs: disp3 (H x W) and the semantic probability map (softmax output, H x W x C).
     - disp3 (broadcast) x semantic prob map -> category-wise raw disparity H x W x C ("each channel is the raw disparity under a certain category"; soft assignment, probabilities in [0,1]).
     - Depthwise Conv K3 S2 (one 3x3 filter per category, stride 2) -> pointwise Conv K1 S1 ("integrate all category channels") -> TransConv K3 S2 -> residual map; ADD to disp3 -> disp4 (Fig. 2). Channel counts after the pointwise conv, normalization, activation: not stated. Whether the added residual is clipped/scaled: not stated.
     - Which semantic probability resolution feeds in (full res, from upsampled logits?): not stated explicitly; the product with disp3 requires H x W.
  3. **Loss module**: semantic-boundary loss and semantic-smooth loss (Eq. 4-7) between the semantic GROUND TRUTH and predicted disp4 - loss-only coupling, sem(GT) -> disp. Also joint training with L_sem couples the shared shallow layers (feature-sharing, sem <-> disp).
  - Note: semantic probabilities used in confidence (features C,D) and residual (prob map) are the model's own predictions from the jointly trained semantic branch; at test time no GT semantics.

### 3. Block -> problem -> evidence table
All on KITTI 2015 40-image validation split, Scene Flow pretrain -> KITTI-15 finetune (160 train imgs), PSMNet baseline, unless noted. Metrics: 3px error (%) / EPE (px). Single run each, no variance reported. Runtime on one TITAN 1080Ti.

| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| Baseline PSMNet | n/a | 1.415 % / 0.6341 px (Tab. 3) | Scene Flow -> KITTI-15 | 0.671 s (Tab. 3) |
| +Confidence (C) alone | wrong/drifted candidate probabilities | 1.371 / 0.6275: -0.044 pt / -0.0066 px (Tab. 3) | same | not reported separately |
| +Residual (R) alone | category-dependent smoothness | 1.368 / 0.6253: -0.047 pt / -0.0088 px (Tab. 3) | same | not reported separately |
| +C+R | complementary | 1.328 / 0.6203: -0.087 pt / -0.0138 px (Tab. 3) | same | n/r |
| +C+R+Loss (full CRL) | boundary/inner smooth supervision | 1.299 / 0.6198: -0.116 pt / -0.0143 px vs baseline; the loss module adds only -0.029 pt / -0.0005 px over CR (Tab. 3) | same | 0.674 s vs 0.671 s (Tab. 3), input size not stated |
| Loss module alone, or R+L, C+L | not ablated (no L-only row) | not ablated | - | - |
| Full CRL on Virtual KITTI | in-domain synthetic check | baseline 4.108 / 0.6237 -> 3.874 / 0.5892: -0.234 pt / -0.0345 px (Tab. 3) | trained from scratch on vKITTI, crop 160x320, 233 val imgs | n/r |
| Confidence combination: Disp-cor + Seg-cor | (variant) | 1.362 / 0.6269 (Tab. 1) | same | - |
| Confidence combination: Disp-cor x Seg-cor (chosen) | | 1.299 / 0.6198 (Tab. 1) - best | | - |
| Disp-cor x Disp-cor x Seg-cor | (variant) | 1.319 / 0.6212 (Tab. 1) | | - |
| Disp-cor x Seg-cor x Seg-cor | (variant) | 1.309 / 0.6238 (Tab. 1) | | - |
| Loss weights/threshold sweep (w_bdry, w_sm, lambda) | tuning | (0.5,0.5,2) 1.330/0.6274; (0.5,0.5,3) 1.299/0.6198; (0.5,0.5,4) 1.326/0.6226; (0.7,0.5,3) 1.336/0.6252; (0.5,0.7,3) 1.336/0.6235 (Tab. 2) | same val set | - |
| Depthwise vs plain conv in residual module | claimed motivation: category-dependent conv | not ablated | - | - |
| Semantic branch itself (shared layers + L_sem) | provides semantics | not ablated separately (never run "semantic branch + joint train but no C/R/L") | - | - |
| Sem-vs-no-sem equal-capacity control | causal check | not ablated | - | - |
| Full model, KITTI-15 benchmark vs PSMNet | | D1-all (All px) 2.32 -> 1.99 (-0.33), Noc 2.14 -> 1.78 (-0.36); bg 1.86->1.63, fg 4.62->3.76 (Tab. 4) | trained on all 200 KITTI-15; PSMNet number is the published benchmark row, not re-run | n/r |
| Full model, KITTI-12 benchmark vs PSMNet | | Noc: 2px 2.44->2.22, 3px 1.49->1.38, 4px 1.12->1.05, 5px 0.90->0.86; All: 2px 3.01->2.89, 3px 1.89->1.85, 4px 1.42->1.40, 5px 1.15->1.15 (equal) (Tab. 5) | KITTI-12 has no semantic labels; semantic branch trained on KITTI-15 only | n/r |

### 4. Interactions & dependencies
- Gains are sub-additive: C -0.044, R -0.047, CR -0.087 pt; the loss module adds a further -0.029 pt but only -0.0005 px EPE (Tab. 3). The EPE gains from each module are 0.007-0.014 px on a 0.63 px baseline.
- Multiplicative combination of the two correlations beats additive and beats multiplying one correlation twice (Tab. 1); authors say extra multiplications "only result in worse performance" (p. 10). Consequence: the gate relies on AGREEMENT of both cues (high only when both correlations are high, p. 7).
- The confidence module needs semantic features from BOTH views (C and D); the residual and loss modules need the semantic probability map / GT of the left view.
- Semantic labels are needed in training: only KITTI-15 provides them; Scene Flow pretraining trains only the stereo branch (p. 9-10). The semantic branch is therefore trained on 160-200 images; mIoU is only 48.12 % (p. 12). KITTI-12 submission uses a semantic branch trained on KITTI-15 labels only (p. 10).
- Boundary loss assumes foreground semantic boundary = disparity boundary; the mask m_b (Eq. 5) drops background classes (road, sidewalk, vegetation, terrain) because their semantic boundaries often have no disparity boundary (Fig. 4, p. 8). The smooth loss is masked by m_s (Eq. 7) to skip true disparity discontinuities. The two terms are mutually dependent on lambda = 3 (Tab. 2).
- Loss weights w_disp1/2/4 = 0.5/0.7/1.0 are inherited from PSMNet's three-output weights.
- disp3 (the residual module input) is NOT directly supervised: the loss supervises disp1, disp2, disp4 only (p. 7). disp3 is trained only through disp4 = disp3 + residual, so the residual module and last hourglass are not independently supervised - a design choice the paper does not discuss.

### 5. Losses (exact)
Total (Eq. 8-9, p. 9):
- L = L_disp + w_sem * L_sem + w_bdry * L_bdry + w_sm * L_sm
- L_disp = w_disp1 * L_disp1 + w_disp2 * L_disp2 + w_disp4 * L_disp4, with L_disp = (1/N) sum smooth_L1(d_i - d*_i) over N valid pixels (Eq. 2). Weights w_disp1 = 0.5, w_disp2 = 0.7, w_disp4 = 1 ("according to [15]"). disp3 has no loss term; disp4 is the test output.
- L_sem = -(1/N) sum_i sum_c y(i,c) log p(i,c) (cross entropy, Eq. 3), w_sem = 1.
- L_bdry (Eq. 4) = (1/N) [ sum_{i,j} |phi2_x(sem_{i,j,m_b})| * exp(-|phi2_x(d_{i,j,m_b})|) + sum_{i,j} |phi2_y(sem_{i,j,m_b})| * exp(-|phi2_y(d_{i,j,m_b})|) ]. phi2_x, phi2_y are SECOND-order gradients along x and y; sem = semantic GT label map, d = predicted disparity (disp4); N = number of pixels with m_b = 1. Intent: punish pixels where GT semantic boundary exists but predicted disparity gradient is small (exp term large).
  - m_b (Eq. 5): 1 where the pixel's class is NOT in {road, sidewalk, vegetation, terrain} (KITTI-15 example), else 0. Whether "pixel's class" is the GT class at p(i,j) or the boundary-neighbour class: not stated.
- L_sm (Eq. 6) = (1/N) [ sum |phi2_x(d_{i,j,m_s})| * exp(-|phi2_x(sem_{i,j,m_s})|) + sum |phi2_y(d_{i,j,m_s})| * exp(-|phi2_y(sem_{i,j,m_s})|) ]. Punishes pixels smooth in the semantic GT but non-smooth in predicted disparity.
  - m_s (Eq. 7): 1 where "gradient of disparity map at p(i,j) less than lambda", else 0 (i.e. skip pixels with a real disparity discontinuity). Which gradient order (first or second), and whether computed on GT or predicted disparity: not stated.
  - lambda = 3, w_bdry = 0.5, w_sm = 0.5 (Tab. 2 sweep; chosen on the same 40-image val split used for reporting).
- Application: all losses applied at full resolution on disp4 (the boundary/smooth losses use the final prediction, "Loss module takes the final prediction disp4 and semantics as input", p. 5). Deep supervision on disp1, disp2 only through L_disp.

### 6. Training recipe (p. 8-10)
- PyTorch; Adam beta1 0.9, beta2 0.999; LR 0.001 then 0.0001 later; batch size 2 (GPU limited); random crops 160x320 (vKITTI) / 256x512 (others); max disparity 192.
- Ablation protocol: Scene Flow pretrain 15 epochs LR 0.001; fine-tune KITTI-15 train: 600 epochs at 0.001 then 100 epochs at 0.0001. vKITTI: from scratch, 200 epochs at 0.001 then 100 at 0.0001.
- Benchmark protocol: Scene Flow pretrain; train on mixed KITTI-15 + KITTI-12 for 500 epochs; finetune on KITTI-15 only or KITTI-12 only for 200 epochs. Semantic labels exist only for KITTI-15.
- Nothing frozen (end-to-end, semantic branch jointly trained from the start of the KITTI stage; during Scene Flow pretrain only the stereo branch is trained because no labels, p. 9).
- Augmentation beyond random crop: not stated. Hardware for training: not stated; inference timing on one TITAN 1080Ti (p. 11-12).

### 7. Results
- Ablation (KITTI-15 val, 3px % / EPE px): baseline 1.415 / 0.6341 -> full 1.299 / 0.6198 (Tab. 3). vKITTI val: 4.108 / 0.6237 -> 3.874 / 0.5892.
- Semantic quality (KITTI-15 val): mIoU 48.12 %, mAcc 55.25 % (p. 12).
- Runtime: baseline 0.671 s, full 0.674 s (Tab. 3; TITAN 1080Ti; input size not stated). Params/FLOPs/memory: not stated.
- KITTI-15 benchmark (Tab. 4), D1 bg/fg/all, All px: SGNet 1.63 / 3.76 / 1.99; PSMNet 1.86 / 4.62 / 2.32; SegStereo 1.88 / 4.07 / 2.25; SSPCV-Net 1.75 / 3.89 / 2.11; EdgeStereo-V2 1.84 / 3.30 / 2.08; AANet+ 1.65 / 3.96 / 2.03. Non-occluded: SGNet 1.46 / 3.40 / 1.78 vs PSMNet 1.71 / 4.31 / 2.14. EdgeStereo-V2 is best on fg.
- KITTI-12 benchmark (Tab. 5), Noc/All: SGNet 2px 2.22/2.89, 3px 1.38/1.85, 4px 1.05/1.40, 5px 0.86/1.15. EdgeStereo-V2 is better on 2px-All 2.88, 3px-All 1.83, 4px-All 1.34, 5px-All 1.04; PSMNet ties 5px-All.
- Qualitative: Fig. 5 error maps (KITTI-15), Fig. 6 (KITTI-12) "eliminate some holes inside objects".

### 8. Negative results & limitations
- Authors: multiplying a correlation twice hurts (Tab. 1); loss thresholds/weights sensitive (Tab. 2: lambda 2 and 4 are worse by 0.027-0.031 pt); KITTI-12 improvement is limited because no semantic GT is available (p. 13). SGNet is not better than PSMNet on 5px-All (equal).
- Weaknesses I see:
  - All ablation deltas are tiny (0.007-0.014 px EPE) from a single run on a 40-image val split that is also used for choosing the combination mode and loss hyperparameters (Tab. 1, 2). Not distinguishable from seed noise; no variance reported.
  - No equal-capacity no-semantics control: the confidence module adds 3D convs and a second set of correlations; the gain may come from extra capacity, not from semantics. The semantic branch's own contribution (shared layers + L_sem) is also not isolated.
  - Loss module's EPE gain is 0.0005 px; the 3px gain 0.029 pt. Loss-only ablations are missing.
  - Fig. 4 itself shows the loss assumption fails for road/sidewalk; the class list in m_b is KITTI-specific and hand-picked.
  - Semantic quality is low (mIoU 48.12 %, 160-200 training images), so the "semantic" signal is weak; GT semantics are used in the losses but predicted semantics at test time.
  - Boundary losses use second-order gradients on integer class labels (class-index difference is arbitrary, so |phi2(sem)| weights depend on label numbering); not discussed.
  - Benchmark gain over PSMNet is vs its published number, not a re-trained baseline with the same recipe (mixed KITTI-12/15 500 epochs).
  - Not tested out-of-domain; vKITTI trained from scratch. No latency on embedded; 0.67 s on 1080Ti.
- The existing summary paper/reference_papers/summaries/semantic_stereo/SGNet.md has errors: (1) it says the paper "does not report EPE ... latency" - it does (EPE in Tab. 1-3; runtime 0.671/0.674 s in Tab. 3); (2) it writes the residual as R_cat(F_sem, d_init), but the residual module takes the semantic PROBABILITY map multiplied with disp3, not semantic features; (3) it omits that disp3 is unsupervised and the loss is on disp1/2/4.

### 9. Relevance to OUR model
Our D2/E3 head is explicitly SGNet-inspired. I checked experiments/D/D_1_semantic_cost/model.py for the mapping (read-only); what we already do vs what SGNet does:
- Already done (ported, with changes):
  - Confidence/gate: ours = `SemanticCostGate`, a 2-layer Conv3d (hidden 8) over [mean of A09 volume, semantic agreement], output 1 + 0.5*tanh(...), identity-initialised, on A09's 1/16 volume. Differences: (a) agreement = dot product of 14-class PROBABILITIES left(x) vs right(x-d), not the correlation of high-dimensional semantic features; (b) fused by concat + learned 3D conv, not by product of two correlation volumes; (c) gate bounded to [0.5, 1.5] rather than sigmoid (0..1) multiply; (d) applied to a frozen predictor's hidden volume.
  - Residual: ours = `ClassResidual`: depthwise 3x3 (groups=14) on prob x normalized-disparity (same category-wise disparity trick as SGNet), then concat with probs, disparity, confidence -> 2-conv fuse -> 4*tanh residual added to the disparity, zero-initialised, no stride-2 / transposed conv. Equivalent to SGNet's Fig. 2 with a richer fuse.
- Not yet tried (portable pieces):
  1. **Semantic-feature (not probability) correlation** as the second correlation, e.g. a projected decoder feature from the frozen trunk at 1/8 or the pre-classifier decoder map; cost = one extra correlation (no params) and the existing Conv3d. Expected benefit: less saturated than 14-way probabilities (soft confusion between road/sidewalk, car/truck is preserved). Risk: small; if probabilities are near one-hot the two are almost the same.
  2. **Multiplicative combination** of the two correlations before the 3D conv (SGNet Tab. 1: x beats +, 1.299 vs 1.362 pt, -0.007 px EPE; measured on the full C+R+L model). Zero-parameter variant of our concat; cheap A/B with the same capacity. Only SGNet's tiny val set supports it.
  3. **Semantic-boundary / semantic-smooth losses** (Eq. 4-7) as training-only terms on the final disparity of the head. No inference cost. Expected benefit: our own edge screening (G/H series) found that RGB-guided residuals, convex upsampling and warp correction did NOT visibly sharpen edges; a loss that uses semantic GT boundaries is a different signal (VKITTI2 has dense GT semantics, so this is cheaper to run than SGNet's KITTI-only labelling). Risks: SGNet's own ablation shows ~0.0005 px EPE gain from the loss module; the m_b class mask is dataset-specific (we would need a VKITTI2 14-class list of "object" vs "background"); second-order gradient on label IDs is ill-defined (use a boundary indicator of the one-hot map instead); with the frozen A09 and a small head, the loss may only change the head's residual.
  4. **Depth-first supervision**: SGNet does not supervise disp3; we supervise the head output only, which is already stricter.
- Novel-combination angle: SGNet trains everything end to end on 160 labelled KITTI images; our "frozen shared trunk + frozen semantic decoder + tiny trained head + equal-capacity control" is a different and more controlled setting. SGNet lacks the equal-capacity control that is our central causal claim, so its reported gains are weaker evidence than ours (-0.24 px in-domain, 2.59 vs 2.71 control vs 3.08 on KITTI-15 zero-shot).
- Cost/risk: SGNet module overhead is small (0.671 -> 0.674 s), but that is mostly because PSMNet is already heavy; on a 3050/Jetson the 3D conv gate at 1/16 volume is the part to profile.
- Do NOT assume SGNet's 3px/EPE gains transfer; they are < 0.015 px EPE on a single 40-image split.

### 10. Key quotes/equations worth citing
- "The semantic features as well as the disparity features with size H/4 x W/4 are fed into the correlation layer ... The output of the correlation layers ... are then multiplied and fed into three consecutive 3D convolution layers with a residual structure ... a sigmoid function" (p. 7).
- "disp3 with size HxW is multiplied with the semantic probability map with size HxWxC ... Depthwise convolution is then performed for each channel. A pointwise convolution is followed to integrate all category channels. Finally a transposed convolution is used to compute the disparity residual." (p. 6).
- Eq. 1: Correlation(x,y,d) = (1/N_c) inner<f1(x,y), f2(x-d,y)> (p. 7).
- Eq. 4-7: boundary and smooth losses, m_b (Eq. 5), m_s (Eq. 7), lambda = 3 (p. 8-9, Tab. 2 p. 10).
- "we supervise disp1, disp2, disp4, and output disp4 during testing" (p. 7).
