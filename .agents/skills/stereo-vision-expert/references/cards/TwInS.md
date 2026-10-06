<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# TwInS card

Source read: full 13-page PDF (arXiv 2602.13588v1). The PDF contains NO supplement/appendix; the text repeatedly points to a "supplement" (alpha sweep, KITTI Semantics benchmark, extra qualitative results) that is not in our copy. Anything that would live there is marked "not stated". Text extracted with pdftotext and cross-checked against Fig. 2 render (p. 5). No repo summary existed for this paper, so no summary errors to report. Percent gains quoted in the paper are RELATIVE (e.g. "29.19%" = 82.13/63.57 - 1), not absolute points.

### 0. Meta
- Title: Two-Stream Interactive Joint Learning of Scene Parsing and Geometric Vision Tasks (TwInS = "Two Interactive Streams").
- Authors: Guanfeng Tang, Hongbo Zhao, Ziwei Long, Jiayao Li, Bohong Xiao, Wei Ye, Hanli Wang, Rui Fan (Tongji Univ.; same group as S3M-Net and TiCoSS).
- Venue/year: arXiv preprint, 14 Feb 2026 (v1). Journal style (IEEE Trans. format); not yet published.
- PDF: paper/reference_papers/semantic_stereo/TwInS_Tang_arXiv2026.pdf
- Code: "will be made publicly available upon publication" (Abstract). None yet.
- Domain: autonomous driving. Tasks: semantic seg, instance seg (scene parsing) + stereo matching or optical flow (geometric).
- Datasets: vKITTI2 (700 collections: 500 train / 200 val, 15 sem classes, 3 inst classes), Cityscapes (2,975 train / 500 val, 19 sem, 8 inst; no disparity GT), KITTI 2015 (140 train / 60 val collections; sparse LiDAR disparity, used for stereo/flow validation) (Sec. IV-A, p. 7). "Collection" = stereo pair + one temporally successive left image.

### 1. Problem & failure modes targeted
- Prior joint frameworks are either (a) shared encoder + independent decoders (DSNet, SGDepth, SG-RoadSeg), where cross-view geometry is never injected into semantics; or (b) cascaded (S3M-Net, TiCoSS): disparity output goes through an extra spatial encoder into a feature-fusion seg net (Fig. 1, p. 2). Stated defects of (b): semantic features never feed back to stereo (imbalanced optimisation); disparity treated as an extra input so decoded GRU features are ignored; extra duplex encoder is expensive; limited to stereo+semseg (no instance seg, no optical flow).
- Unreliable disparity in occlusion / texture-less / reflective regions misleads segmentation (Sec. III-C, p. 6).
- Joint training needs both disparity and seg GT, so Cityscapes (no disparity GT) cannot be used; unsupervised photometric stereo (SegStereo, SG-RoadSeg) is too weak (Sec. I).
- Limitations they admit (Sec. IV-F, p. 11-12): geometry stream dominates, so overexposed/poorly lit regions (bad correspondences) corrupt segmentation (Fig. 8 failure: vehicle in overexposed area misidentified); semi-supervised scheme still needs seg annotations.

### 2. Pipeline by stage
Overall (Fig. 2, p. 5): two streams. Scene-parsing stream = ConvNeXt encoder + Mask2Former decoder (query-based). Geometric stream = RAFT-Stereo-style all-pairs correlation pyramid + 3-level multi-GRU iterative refinement. Input: target image I^t and source image I^s in R^{HxWx3}.
- 2a Feature extraction: ConvNeXt (pre-trained; pre-training dataset not stated) producing F^t_1..F^t_4 and F^s_1..F^s_4, F_i in R^{(H/S_i) x (W/S_i) x C_i}, S_i = 2^{i+1} (strides 4, 8, 16, 32) (Sec. III-B, p. 4). Shared by both streams and BOTH views (the same encoder is the seg encoder AND the stereo feature/context encoder; this replaces RAFT-Stereo's separate shallow feature/context encoders). The encoder is trained end-to-end (not frozen; no freezing is mentioned). Backbone sizes Tiny/Small/Base/Large (Tab. VI): 67.77 / 89.40 / 137.97 / 275.89 M total params. Per stage keeps first and last conv block outputs of stages 1-3: early features F^e_i (shallow, local texture) and late features F^l_i (deep, global semantics).
- 2b Semantic / prior branch: it IS the encoder + Mask2Former decoder (semantic: 15 vKITTI2 / 19 Cityscapes classes; instance: 3/8 classes). Not frozen. Multi-level fused features F^f_i (after CTA) feed the Mask2Former decoder (Sec. III-C, p. 6).
- 2c Cost volume: all-pairs correlation V_{p,q} = sum_c F^t_1(p,c) F^s_1(q,c) (Eq. 1) from stage-1 (stride-4) features; "context-aware" because stage-1 features come from the shared contextual encoder. Pyramid P = {P^1, P^2, P^3} by repeated average pooling (RAFT-Stereo). Disparity range / search radius / number of levels beyond 3: not stated. Not group-wise, no 3D conv.
- 2d Cost aggregation: none (no 3D regularisation). Aggregation only through the GRU updates over lookups of P.
- 2e Disparity computation: hidden states at all iterations go through conv layers that DIRECTLY REGRESS correspondence (a sequence of progressively refined outputs); flow = 2D, disparity = 2D with 2nd component fixed to zero (footnote 2, p. 7). No soft-argmin.
- 2f Refinement: 3-level multi-GRU. H^{k+1}_j = GRU(H^k_j, F~^l_j, F^c_j) (Eq. 3): level j hidden state, contextual late features aligned by Eq. 2, correlation features F^c_j from P^j. Early features F~^e_j initialise the hidden states (replacing RAFT-Stereo's context-encoder-derived init). Number of iterations K: not stated in the main text.
- 2g Upsampling to full res: not described (inherits RAFT-Stereo; not stated).
- 2h FUSION POINTS (bidirectional, feature level; no cost-volume gating):
  1. SEM->GEO (a) correlation pyramid is built from the shared contextual encoder's stage-1 features (Tab. III col "Correlation Pyramid"). Stage: cost volume input. Operator: shared weights (no explicit fusion). Resolution 1/4.
  2. SEM->GEO (b) late features F^l_j, aligned with F~^{e,l}_i = ReLU(GroupNorm(Conv1x1(F^{e,l}_i))) (Eq. 2), enter every GRU update at every iteration as context features ("Context Features" col in Tab. III). Stage: refinement. Operator: input to GRU gates. Resolutions 1/4, 1/8, 1/16 (levels 1-3).
  3. SEM->GEO (c) early features F~^e_j initialise the GRU hidden states ("Hidden States" col in Tab. III). Stage: refinement init.
  4. GEO->SEM Cross-Task Adapter (CTA), per encoder stage i = 1..3 (Fig. 2 shows three-level inputs; text says multi-level): aligned GRU final hidden states H~_i (channel+resolution aligned to F^t_i) are the key/value; contextual features F^t_i are re-mapped as queries (Sec. III-C, Eq. 4, p. 6). Note the text is internally muddled (it says "projected geometric features G_i (remapped as query embeddings Q'_i) ... aggregate contextual F^t_i (as K', V')") but Fig. 2 and the sentence "contextual features are remapped as query embeddings to adaptively retrieve task-relevant geometric cues" agree: Q from contextual features, K,V from hidden states. Operator: ELU+1-shifted linear attention, G_i = MLP(RMSNorm( phi(Q_i)(phi(K_i)^T V_i) / (phi(Q_i) Reshape(phi(K_i)^T)) )). Output G_i is the geometry projected into contextual space; a second linear-attention step with Q'_i (from G_i) over K'_i,V'_i (from F^t_i) produces the fused F^f_i (Fig. 2: two linear-attention blocks, MLP, reshape). Resolutions 1/4-1/16 (and the stage-4 feature presumably passes unchanged; not stated). Cost: "negligible computational overhead" (Sec. III-C; no number given).
  5. Loss-only / training-only: teacher-student pseudo-label supervision of the stereo head (Sec. III-D).

### 3. Block -> problem -> evidence table
All TwInS ablations use the Base ConvNeXt unless stated (Tab. VI/IV caption). KITTI 2015 = 140 train / 60 val.
| block | problem it solves | evidence (ablation delta) | context/conditions | cost |
|---|---|---|---|---|
| Correlation pyramid from shared contextual encoder (vs separate encoders, RAFT-Stereo style baseline) | context-aware matching | EPE 1.13 -> 1.10, D1 4.08 -> 3.96 (Tab. III row 2) | KITTI15 val, semi-sup pipeline, encoder trained jointly | removes the separate feature encoders |
| + context features from scene-parsing stream into GRU | ambiguity in texture-less/reflective regions | EPE 1.10 -> 1.06, D1 3.96 -> 3.90 (Tab. III row 3) | same | alignment 1x1 conv+GN+ReLU |
| + hidden-state init from early features | same | EPE 1.06 -> 1.00, D1 3.90 -> 3.84 (Tab. III row 4); total -11.5% EPE vs baseline | same | negligible |
| Cross-Task Adapter (CTA) vs no geometry baseline | scene parsing in occlusion/boundary regions | Cityscapes mIoU 78.52 -> 80.05, mAcc 86.80 -> 87.98; KITTI15 mIoU 68.19 -> 72.03, mAcc 78.54 -> 80.92 (Tab. IV) | baseline = contextual features only | linear attention (O(N)) |
| CTA vs FFM (S3M-Net element-wise add) | | FFM 78.81 / 67.81 mIoU (Cityscapes/KITTI) (Tab. IV): essentially no gain, KITTI slightly worse than baseline | | |
| CTA vs HFSB (RoadFormer self-attn heterogeneous fusion) | | HFSB 76.42 / 66.06: WORSE than baseline by 2.1 / 2.1 pts; t-SNE (Fig. 7) shows worse clusters | | |
| Semi-supervised V->C->K vs unsupervised V->C->K | unreliable geometry from photometric loss | mIoU 68.49 -> 72.03; EPE 1.58 -> 1.00; D1 6.49 -> 3.84 (Tab. V) | KITTI15 val | training only |
| Semi-supervised V->C->K vs V->K only | exploit large unlabeled multi-view data (Cityscapes) | mIoU 54.35 -> 72.03; EPE 1.09 -> 1.00; D1 4.02 -> 3.84 (Tab. V) | | |
| Backbone Tiny/Small/Base/Large | accuracy-efficiency | KITTI mIoU 70.86/70.81/72.03/72.36; EPE 1.10/1.05/1.00/0.96; Cityscapes mIoU 79.49/79.89/80.05/82.13; params 67.77/89.40/137.97/275.89 M; FPS 22/21/20/18 at 640x320 on RTX 4090 (Tab. VI) | | |
| Plug-in to other nets (versatility) | | RTFNet KITTI mIoU 34.47 -> 39.10 (Cityscapes 57.89 -> 63.00), params 254.15 -> 199.32 M; SNE-RoadSeg KITTI mIoU 43.21 -> 45.88 (params 201.32 -> 168.50); S3M-Net 58.29/41.54 -> 63.91/45.42 (params 346.70 -> 242.68); TiCoSS 63.57/47.66 -> 67.42/52.19 (params 385.05 -> 243.02, -36.88%) (Tab. VII) | column order printed is Cityscapes mIoU, KITTI mIoU, Params (M). I inferred this from TiCoSS 63.57/47.66 matching Tab. I. | |
| Quantile threshold alpha | pseudo-label selection | "impact of alpha ... in the supplement"; value used: not stated | | |
| EMA teacher update, weak/strong augmentation teacher/student, iteration-fluctuation uncertainty head (Eq. 6) | pseudo-label quality | NOT ABLATED (no EMA-vs-fixed teacher, no iteration-uncertainty vs left-right check, no fixed-threshold vs quantile comparison in the main text) | | |
| Mask2Former decoder / ConvNeXt choice | | not ablated | | |
Note on Tab. VII RTFNet: params printed 254.15 -> 199.32 (columns Cityscapes mIoU 57.89 -> 63.00; KITTI 34.47 -> 39.10). I read the three numbers per row as (Cityscapes mIoU, KITTI mIoU, Params). Params drop because TwInS-style integration removes the duplex encoder.

### 4. Interactions & dependencies
- CTA works only because both streams share one encoder family and the GRU hidden states are first projected into the contextual space (linear-attention with contextual Q). Methods designed for duplex-encoder fusion (HFSB, FFM) do NOT transfer: HFSB degrades segmentation.
- The semi-supervised scheme is bound to the GRU iterative design: uncertainty is computed from fluctuations between iterates (Eq. 6), so it does not apply to a single-shot / cost-volume head.
- Pseudo labels only help when the teacher is already good: pure unsupervised photometric training on the new data degrades both tasks (Tab. V).
- V->C->K (Cityscapes in the loop) is needed for semantic quality: V->K only gives 54.35 mIoU on KITTI, i.e. in-domain Cityscapes segmentation labels, not stereo data, drive the real-domain seg gain.
- Dependency of seg on geometry: failure of the geometry stream (bad light) propagates into segmentation (Fig. 8). No gating/confidence on the CTA path (the TiCoSS gate was dropped in favour of attention).
- Joint training end-to-end (not frozen) is assumed everywhere; a frozen-encoder variant was not tried.

### 5. Losses (exact formulas, weights, where applied, deep-supervision schedule)
The main text gives NO supervised stereo loss formula, NO segmentation loss formula, NO loss weights, NO deep-supervision/iteration weighting (probably in the missing supplement). What is stated:
- Correspondence loss: MAE (L1) is assumed, "when a correspondence matching model is trained by minimizing MAE loss, residuals are Laplace" (Sec. III-D, citing Kendall&Gal). Likelihood Eq. 5: p(x_c | x^_c, sigma) = 1/(2 sigma) exp(-||x_c - x^_c||_1 / sigma).
- Uncertainty head (Eq. 6): U = MLP(Concat{ psi(X^i_c - X^j_c) | 1<=i<j<=K }), psi = elementwise square; one value sigma per pixel from pairwise differences among the K iterative predictions. Trained by minimising the negative log-likelihood of Eq. 5 AND a KL divergence aligning the uncertainty distribution with the Laplace distribution of residuals (following Chen et al. CVPR'23 [52]); exact KL/NLL expressions and weights: not stated.
- Pseudo-label threshold (Eq. 7): tau = mu + b * log(2(1 - alpha)), mu = median, b = mean absolute deviation of the image's uncertainty map; pixels with uncertainty < tau are kept as sparse pseudo-labels. alpha: not stated in the main text.
- Student trained on pseudo labels under strong augmentation; teacher (identical architecture) sees weakly augmented images; teacher params = EMA of student (EMA momentum not stated).
- Segmentation: Mask2Former-style mask classification is implied; losses not stated.

### 6. Training recipe
- Hardware: 2x RTX 4090, total batch size 2 (!) (Sec. IV-B, p. 8).
- Crops: 512x256 for vKITTI2 and KITTI 2015; 1024x512 for Cityscapes. AdamW, eps 1e-8, weight decay 1e-5, LR fixed 1e-4 (no schedule).
- Iterations: 50k on vKITTI2 (supervised teacher stage), 75k on Cityscapes, 5k on KITTI 2015. Augmentation: random crop, colour jitter, asymmetric occlusion (+ weak/strong for teacher/student).
- Curriculum: (1) supervised pre-training on vKITTI2 (both seg and correspondence labels) -> teacher; (2) semi-supervised on Cityscapes (seg GT + pseudo correspondences from teacher) -> (3) semi-supervised on KITTI 2015 (KITTI seg labels consistent with Cityscapes + pseudo correspondences) (Sec. IV-E.3). Student/teacher updated by EMA throughout.
- Optical flow variant: same architecture with consecutive left frames; vKITTI2 provides optical flow GT.
- Frozen parts: none stated.

### 7. Results
Semantic seg, mIoU / mFSc, Tab. I (p. 8). Rows we care about:
| method | vKITTI2 mIoU/mFSc | Cityscapes | KITTI15 |
| S3M-Net (TIV'24) | 84.00 / 88.27 | 58.29 / 65.45 | 41.54 / 48.02 |
| TiCoSS (TASE'25) | 87.94 / 90.52 | 63.57 / 74.06 | 47.66 / 54.69 |
| SemStereo (AAAI'25) | 83.69 / 87.01 | 59.86 / 68.42 | 45.89 / 52.04 |
| DSNet / SGDepth / SG-RoadSeg | 76.61 / 80.19 / 81.06 | 54.30 / 55.96 / 56.37 | 39.19 / 40.20 / 32.89 |
| Mask2Former | 86.83 / 92.65 | 78.41 / 86.29 | 67.48 / 78.50 |
| DFormer / RoadFormer+ | 85.99 / 88.44 | 77.76 / 80.73 | 70.13 / 70.90 |
| DINOv2 / Depth Anything / ViT-CoMer | 86.36 / 92.10 / 92.37 | 82.49 / 82.98 / 79.72 | 74.65 / 73.82 / 71.99 |
| TwInS (Large, headline) | 88.67 / 93.69 | 82.13 / 89.73 | 72.36 / 82.69 |
Relative gains claimed: +29.19% (Cityscapes) and +51.82% (KITTI) mIoU over TiCoSS (sic: 82.13/63.57, 72.36/47.66). Honest reading: TwInS is BELOW Depth Anything (92.10) and ViT-CoMer (92.37) on vKITTI2, below DINOv2 (74.65) on KITTI15, and below Depth Anything on Cityscapes (82.98 vs 82.13). Instance seg (Tab. I-b): mAP/AP50 83.20/95.00 (vKITTI2), 32.10/51.31 (Cityscapes), 25.90/38.91 (KITTI15) vs Mask2Former 80.79/93.01, 30.06/49.52, 22.79/36.94.
Stereo (Tab. II-a, KITTI15, 60 val collections): EPE / D1 = 0.96 / 3.34 (Large) vs SegStereo 1.59/7.44, CRD-Fusion 1.37/5.49, ES3Net 1.84/8.79, RoSe 1.04/3.62 (semi-supervised). Only un/self-supervised baselines are listed. Flow (Tab. II-b): 3.72 / 10.86 vs SemARFlow 3.88/11.40, UPFlow 3.90/12.89.
Runtime / params (Tab. VI, 640x320, RTX 4090): Tiny 67.77 M, 22 FPS; Base 137.97 M, 20 FPS; Large 275.89 M, 18 FPS. No ms, FLOPs, VRAM, or Jetson numbers; no runtime comparison against TiCoSS/S3M-Net other than param counts (S3M-Net 346.70 M, TiCoSS 385.05 M, Tab. VII). Note even the Tiny joint model is ~68 M params, an order of magnitude beyond our target.

### 8. Negative results & limitations
- Authors: geometry stream dominance and failure under poor illumination (Fig. 8); still needs seg annotations; open-vocab extension proposed as future work.
- Reported negatives: HFSB and FFM fusion do not help; unsupervised training degrades; V->K-only semi-sup is poor for seg.
- My concerns:
  1. Baseline fairness: TiCoSS/S3M-Net need disparity GT, so their Cityscapes/KITTI numbers presumably come from vKITTI2-only training (protocol "following previous works" but their training data for Tab. I is not stated). TwInS additionally trains on Cityscapes seg labels (2,975 images) and KITTI. The +29%/+52% over TiCoSS is therefore largely a data/label-access effect, and the authors say as much ("largely attributed to the semi-supervised training strategy").
  2. Stereo accuracy comparison is only against unsupervised/self-supervised methods. No supervised RAFT-Stereo/IGEV, no teacher-only (vKITTI2-only) baseline, no zero-shot numbers. EPE 0.96 is on a 60-pair held-out subset of the same KITTI 2015 training set that the student's semi-supervised stage ran on (whether the 60 val images were included in unlabeled data: not stated). It is not a KITTI online-benchmark number, and not comparable to our 200-pair zero-shot EPE 2.59-3.08.
  3. Tab. III baseline = "multiple independent encoders", so the 11.5% EPE gain measures replacing RAFT-Stereo's shallow encoders with a big jointly-trained ConvNeXt, not a clean semantic-vs-no-semantic control. No equal-capacity no-semantic control (the kind we run as C4/D3/D4/E4/F3).
  4. Seg-side gain from geometry (CTA) vs plain baseline is 1.5 pts Cityscapes and 3.8 pts KITTI (Tab. IV), measured with the same jointly-trained encoder; no seg-only network trained on the same data protocol is given (the Mask2Former rows use their own training).
  5. Batch size 2, single run, no seeds/variance, no loss weights, no K, no alpha, no supplement in our copy; code not released: not reproducible.
  6. Params 68-276 M and 18-22 FPS (RTX 4090, 640x320): not real-time on edge hardware. Efficiency claim is relative to TiCoSS/S3M-Net only.
  7. Inconsistencies: abstract-level text says mIoU gains over Mask2Former of "2.12% to 7.23%" (relative; matches 88.67/86.83=2.1%, 72.36/67.48=7.2%); the Table VII "+Ours" configuration backbone is not specified. "KITTI Semantics" benchmark comparison is only in the missing supplement.

### 9. Relevance to OUR model
Positioning: TwInS is the strongest 2026 competitor on the "joint stereo + semantic seg" axis but occupies a different design point. It trains a large shared ConvNeXt end-to-end, injects semantics into a RAFT-Stereo GRU (feature level, context+hidden init), feeds GRU hidden states back to segmentation with a cross-task adapter, and relies on semi-supervised pseudo-labels. We freeze a small YOLO26m trunk (layers 0-6), freeze the stereo predictor and the 14-class decoder, and train only a semantic cost-gating + class-residual head, with equal-capacity no-semantics controls and KITTI zero-shot evaluation.
Portable ideas (ordered by value/risk):
1. Reverse-direction (GEO->SEM) adapter. Insertion: between frozen A09 refinement features (tile hypotheses / aggregated cost features) and the frozen semantic decoder input at 1/8; linear-attention with semantic features as query and geometry as key/value (Eq. 4). Benefit: they get +1.5 / +3.8 mIoU (Tab. IV, in a trained-encoder setting); for us it would be a new capability (disparity-aware semantics), but would need either a trainable seg adapter or fine-tuning, which breaks the "frozen decoder" story. Cost: small (linear attention, few hundred K params at 1/8). Risk: medium; their failure case (bad geometry -> bad semantics) argues for a confidence gate, which we already have. Novel vs done: bidirectional exchange with frozen backbones is not done in TwInS.
2. Uncertainty-aware pseudo-label scheme (Eqs. 5-7) for training ONLY our fusion head on unlabeled real stereo (e.g. KITTI raw, Cityscapes): teacher = frozen A09/E3; confidence from iteration/candidate disagreement (A09 has tile/plane refinement steps, so iterate fluctuation is available) with a per-image Laplace-quantile cut. Benefit: addresses the thing our results cannot claim (real-domain gain beyond zero-shot) and is cheap; risk: confirmation bias with a frozen teacher; EMA unusable with frozen weights (only the head would update). Novel combination: pseudo-labelled domain adaptation of a gating head with a frozen shared encoder.
3. Use their Eq. 6 idea (pairwise iterate differences -> sigma) as the confidence input to SemanticCostGate rather than only semantics. Low cost; the gate currently has no explicit stereo-confidence cue. Risk: our A09 may expose too few iterates.
4. Early/late feature split: late (semantic-rich) features as GRU context, early (texture) features for init. In our model, YOLO layers 0-6 give mid-level features only; a late/early split is not available without un-freezing deeper layers. Not recommended.
What to avoid / not worth copying: HFSB-style heavy attention fusion (they themselves show it fails), end-to-end joint training (conflicts with our frozen constraint and 4 GB VRAM), Mask2Former decoder.
Must compare against: TwInS-Tiny (67.77 M, 22 FPS @640x320 on 4090; KITTI15 EPE 1.10, mIoU 70.86, Tab. VI) as the nearest reported small variant, TiCoSS (385 M, 47.66 KITTI mIoU) and S3M-Net (347 M) on the same semantic protocol; but note incompatible protocols (they train on 140 KITTI15 pairs plus pseudo-labels; ours is zero-shot on 200 pairs). A fair reproduction is impossible until code is released. In the paper, state the protocol difference explicitly and add param/FPS columns (ours: a frozen trunk, trainable head only).
Novelty implication: "joint stereo + semantic model with a shared encoder" and "semantics improves stereo in iterative models" are NOT novel after TwInS (and TiCoSS/S3M-Net/SGNet/SegStereo). What remains unclaimed by TwInS: frozen shared detector/segmenter trunk reused for both tasks, semantic cost-candidate gating, class-conditioned residual refinement, matched-capacity causal controls, real-time sub-4GB deployment.

### 10. Key quotes/equations worth citing
- "TwInS is the first joint learning framework capable of learning geometric vision tasks in a semi-supervised manner" (Sec. II-C, p. 4).
- Eq. 4 (p. 6): G_i = MLP(RMSNorm( phi(Q_i)(phi(K_i)^T V_i) / (phi(Q_i) Reshape(phi(K_i)^T) ))), phi = ELU+1-type shifted map.
- Eq. 6 (p. 7): U = MLP(Concat{psi(X^i_c - X^j_c) : 1<=i<j<=K}); Eq. 7: tau = mu + b log(2(1-alpha)).
- "the geometric vision task plays a dominant role ... under poor illumination ... degrades scene parsing" (Sec. IV-F, p. 11).
- "TwInS equipped with a tiny backbone reduces model parameters by over 75.43% and increases inference speed by 21.75% ... without compromising accuracy" (Sec. IV-E.4, p. 11).
