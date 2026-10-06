<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# TiCoSS card

PDF (13 pp) read in full; Fig. 1 (p.4) and Eq. 1-8 (p.4-6) verified from page renders. The PDF text repeatedly refers to a "supplement" (TGF theory, DSCC theory, weather robustness, KITTI Semantics benchmark submission; pp.5,6,11) that is NOT included in this PDF; those claims are unverifiable here.

### 0. Meta
- Title: TiCoSS: Tightening the Coupling between Semantic Segmentation and Stereo Matching within A Joint Learning Framework
- Authors: Guanfeng Tang, Zhiyuan Wu, Jiahang Li, Ping Zhong, Wei Ye, Xieyuanli Chen, Huimin Lu, Rui Fan
- Venue: IEEE Trans. Automation Science and Engineering 2025 (arXiv 2407.18038v4)
- PDF: paper/reference_papers/semantic_stereo/TiCoSS_Tang_TASE2025.pdf
- Code URL: none in PDF (repo summary's GitHub link not verifiable from the paper).
- Domain: driving. Datasets: vKITTI2, KITTI 2015, Cityscapes.
- Splits (Sec. IV-A, p.6): vKITTI2 = 700 pairs, 500 train / 200 "validation" (same as S3M-Net; grouping of the 10 weather variations not stated). KITTI 2015 = 400 pairs, 200 with GT, "split 7:3 train/test" (presumably 140/60). Cityscapes = 2,975 train / 500 val with "depth" from ViTAStereo pseudo-labels (no real GT depth; so Cityscapes only measures seg). Stereo evaluated only on vKITTI2 and KITTI 2015.

### 1. Problem & failure modes targeted
- Seg side: "simplistic and indiscriminate fusion of heterogeneous features, which often causes conflicting learning representations and erroneous segmentation results" (p.1, citing [23]) - the closest thing to a "conflict" claim: it is about RGB-vs-disparity-feature fusion in the seg encoder, NOT about shared-encoder gradient conflict between tasks.
- Geometric features lose proportion in deep decoder input (cites [24]); slow convergence/vanishing gradients; deep-supervision heads ignore each other; losses of two tasks computed independently "fails to leverage complementarity at output level" (p.2).
- Related-work statement on sharing: SSNet "employs a single encoder ... however, as demonstrated in [43], such shareable features may not be suitable for both dense prediction and geometric vision tasks" (p.3). This is a citation, not an experiment here. TiCoSS responds by NOT fully sharing: only the shallowest 3 contextual layers share weights with the stereo net's feature extractor (Sec. III-B, p.5); deeper encoders are separate. So TiCoSS side-steps the shared-encoder question.
- Stated focus: improve SEGMENTATION; stereo is a secondary beneficiary (Note to Practitioners; Sec. IV-D2 p.7: "primary focus ... improve semantic segmentation performance").

### 2. Pipeline by stage
- 2a Feature extraction: stereo branch = S3M-Net's (RAFT-Stereo style) feature+context encoders; contextual encoder F^C_i for left RGB; first three layers' contextual features SHARE weights with the stereo feature extractor, deeper ones separate (p.5). Fig. 1 shows F^C_1..F^C_5. Trained end-to-end.
- 2b Semantic branch: duplex "tightly-coupled" encoder: geometric stream F^G_i from the predicted disparity map D^L (E^G_i, residual/conv blocks), contextual stream F^C_i, fused stream F^F_i (Eq. 2-3). Decoder = SNE-RoadSeg dense-skip decoder with up to 4 outputs (s/2, s/4, s/8, s/16 per Fig. 1; number of deep-supervision branches n/L not given in text). Trained from scratch jointly; no pretrained seg net stated.
- 2c-2g Stereo path: "adopt the stereo matching approach used in S3M-Net" (p.3): correlation pyramid + multi-level GRU; max disparity 192. Not re-described. The right disparity map is also produced (Fig. 1) for the left-right consistency check; how (second pass with flipped inputs vs second head) is NOT stated.
- 2h FUSION POINTS:
  1. Disparity (D^L) -> geometric encoder E^G -> F^G_i -> fused into contextual stream with element-wise sum at every layer (disp->seg, feature level, resolutions Hi x Wi per layer).
  2. Selective Inheritance Gate (SIG) (Eq.1): gate maps G_i in [0,1]^{HixWi} (sigmoid of conv on the features) decide how much of the previous layer's remapped feature is inherited into the current layer: X~_i = (1+G_i) ⊙ X_i + (1−G_i) ⊙ [G_{i−1} ⊙ R(X_{i−1})] (verified from p.4 render; 1 = ones matrix, ⊙ channel-broadcast). Applied in BOTH the geometric stream and the fused stream.
  3. HDS: highest-resolution fused feature F^F_1 guides side-branch decoders through a Feature Dynamic Alignment (FDA) block = l stacked 3x3 stride-2 conv+BN+ReLU downsampling units, concatenated with deepest decoded features at each scale (Sec. III-C, p.5).
  4. Loss coupling: L_DIA (stereo L-R inconsistency re-weights seg CE), L_DSCC (KL between seg outputs), L_SCG (GT-semantic re-weights both) (Sec. III-D).
  Direction: stereo->seg (features+loss); seg->stereo only via the 3 shared shallow layers' gradients and SCG weights.

### 3. Block -> problem -> evidence table
All TiCoSS ablations are SEGMENTATION-ONLY on KITTI 2015 (the 60-ish pair test split; hyperparameters also tuned there). No stereo metric is given for any component ablation.
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| TGF (SIG gates) vs ASFF vs GFF (Tab. VI) | indiscriminate fusion | mIoU (disparity-branch fusion only / RGB-branch only / both): ASFF 54.72/54.16/55.92; GFF 57.13/57.66/58.03; TGF 55.80/58.44/59.06; Baseline (S3M-Net w/o SCG, plain add) 54.33. TGF(both) vs baseline +4.73; vs GFF +1.03; vs ASFF +3.14. fwIoU 83.44 base -> 84.72 TGF. (Paper's "up to +7.93" is not reproducible from these rows: best-case 59.06-54.16=4.90 abs.) | KITTI 2015 | not separately reported |
| HDS vs SDS vs FDS (Tab. VII) | deep supervision | baseline (S3M+TGF) 59.06; +SDS 60.01; +FDS 60.62; +SDS+FDS 60.86; +HDS 62.36 mIoU (fwIoU 84.72->86.33) | KITTI | "no significant" memory |
| HDS guidance layer/type (Tab. VIII) | which feature guides | layer1 fused FF 62.36; layer1 GF 61.74; layer1 CF 62.09; layer2 best FF 62.11; layer3 FF 60.26 (GF 56.67) | KITTI | |
| DIA / DSCC / SCG within CT loss (Tab. IX) | output-level coupling | on top of TGF+HDS (no extra loss = 62.36): +SCG alone 62.75; +DIA 63.04; +DSCC 62.88; DIA+DSCC 63.54; DIA+SCG 63.15; DSCC+SCG 62.99; all three 63.63 mIoU (fwIoU 86.33 -> 86.68). Each term adds 0.4-0.7 mIoU; total +1.27 | KITTI, single seed | free at inference |
| alpha (DIA weight) 0-2.0, beta (DSCC) | | Fig. 7: alpha=1.5, beta=1.0 chosen (plots only) | tuned on KITTI test split | |
| All three (Tab. X) | | TGF only 59.06 mIoU/91.26 mFSc; HDS only 58.49/92.10; TGF+HDS 62.36/92.74; HDS+CT 61.18/92.46; TGF+HDS+CT 63.63/92.90 | KITTI | |
| Shared-weights first 3 layers | efficiency | not ablated | | |
| Stereo contribution of any block | | NOT ablated; only Tab. V final: vs S3M-Net EPE KITTI 0.55->0.54, PEP3 1.62->1.60, PEP1 10.02->10.39 (WORSE); vKITTI2 EPE 0.38->0.34, PEP1 5.56->5.43, PEP3 2.55->2.58 (WORSE) | | |
| FDA block, SIG gate count, G_i design | | not ablated | | |

### 4. Interactions & dependencies
- Fusion must be done in BOTH encoder branches (Tab. VI: both-branch rows are best for ASFF/GFF/TGF).
- CT loss (DIA/DSCC) is "based on the HDS strategy" (needs multiple deep-supervision outputs); the paper does not run CT without HDS (p.10).
- Component gains are sub-additive: TGF alone +4.7 mIoU, TGF+HDS +8.0 vs baseline 54.33, +CT +9.3 total (Tab. X; baseline 54.33 from Tab. VI).
- L_DIA needs a right disparity map for the left-right check -> extra stereo pass (cost not reported).
- Best guidance for HDS is the shallowest fused feature (rich local detail), not deeper ones.
- Conflict: authors say deep geometric features carry "irrelevant semantic information" and dilute contextual features -> gates; but they never test whether gating or the 385M parameters (see 7) drive the gain.

### 5. Losses (exact)
- L_CT = alpha L_DIA + beta L_DSCC + L_SCG + L_SM (Eq.4), alpha=1.5, beta=1.0 (Fig.7 text p.10). L_SCG, L_SM are exactly S3M-Net's (see S3M-Net card: a=0.1, gamma=0.9). NOTE the repo summary writes L_CT = L_disp + a L_seg + b L_DIA + g L_DSCC: that is WRONG (no gamma, L_seg is inside L_SCG).
- Left-right disparity inconsistency (Eq.5): W(p) = D^L(p) − D^R(p − D^L(p;0)) (D^L(p;0) = left disparity at p warped horizontally; text notation unclear, the second argument is "0" offset). W^N(p) = 1/(1+e^{−|W(p)|}) (Eq.6). Paper says W^N in [0,1]; actually |W|>=0 so W^N in [0.5,1): perfectly consistent pixels get 0.5, inconsistent -> 1 (my observation). Higher weight = more inconsistent = more attention.
- L_DIA = sum_{i=1..n} { -(1/N) sum_{j=1..N} sum_{k=1..C} W^N(p) y_k(p) log yhat_k(p) } (Eq.7): weighted CE summed over the n deep-supervision outputs (index j in sum is a typo for pixels). So disparity inconsistency (occlusion indicator) re-weights SEGMENTATION loss. Weight range is 2x (0.5-1) times alpha=1.5, a much stronger re-weighting than SCG's ~7%.
- L_DSCC = sum_{r=1..L} sum_{s=1..L, s≠r} [ -(1/N) sum_i yhat^r_k(p) log( yhat^r_k(p) / yhat^s_k(p) ) ] (Eq.8 as printed, p.6; class sum implicit). Pairwise KL between every ordered pair of the L auxiliary classifier outputs, inspired by dynamic hierarchical mimicking [47]. The printed leading minus on a KL term looks like a sign slip; verify against code before reimplementing. L (number of auxiliary classifiers) not stated in text.
- Deep supervision: main + side branch outputs each get CE-type losses (schedule/weights per branch NOT stated).

### 6. Training recipe
RTX 3090, batch 1, crop 512x256 (smaller than S3M-Net's 1000x320), max disp 192, AdamW (eps 1e-8, wd 1e-5), lr 2e-4 (schedule not stated); 100,000 iters vKITTI2, 20,000 KITTI 2015, 50,000 Cityscapes; "standard augmentation". Same from-scratch joint training, no freezing. 385.05M trainable parameters, 308.86 GFLOPs at 512x256, 0.30 s/img, ~5.82 GB on RTX 3090 / i7-13700KF (p.11). Seg evaluated per-image then averaged (Sec. IV-C); a separate mmsegmentation-framework evaluation (Tab. IV) gives mIoU S3M-Net vs TiCoSS: vKITTI2 84.00 vs 87.94; KITTI 41.54 vs 47.66; Cityscapes 55.40 vs 62.16.

### 7. Results
- Seg KITTI 2015 (Tab. I) Acc/mAcc/mIoU/Pre/Rec/mFSc: TiCoSS 91.90/71.97/63.63/92.43/94.10/92.90 vs S3M-Net 90.66/65.90/57.80/90.85/93.55/91.80 (+5.83 mIoU abs); DFormer 90.59/69.01/58.18; RoadFormer+ 91.35/66.29/57.69.
- Seg vKITTI2 (Tab. II): TiCoSS 98.69/91.66/88.46/98.55/98.67/98.57; S3M-Net 98.32/88.24/84.18/98.37/98.28/98.31 (+4.28 abs); DFormer mIoU 85.54.
- Seg Cityscapes (Tab. III): TiCoSS 90.70/81.76/68.36 (Acc/mAcc/mIoU) vs S3M-Net 88.47/77.30/62.59 vs DFormer 91.37/80.16/65.59 (Acc lower than DFormer/MENet 93.39).
- Stereo (Tab. V) EPE/PEP>1/PEP>3: vKITTI2 TiCoSS 0.34/5.43/2.58 vs S3M-Net 0.38/5.56/2.55 vs RAFT-Stereo 0.40/5.88/2.67; KITTI2015 TiCoSS 0.54/10.39/1.60 vs S3M-Net 0.55/10.02/1.62 vs RAFT-Stereo 0.60/10.78/1.96. Text claims "3.64% EPE, 2.47% PEP3 (KITTI); 5.26% EPE, 1.26% PEP1 (vKITTI2)" - these percentages do not match the table values (0.55->0.54 is 1.8%; 0.38->0.34 is 10.5%; KITTI PEP3 1.62->1.60 is 1.2%) -> internal inconsistency; and PEP1 on KITTI and PEP3 on vKITTI2 got worse.
- DOES STEREO IMPROVE? Barely/ambiguously: EPE -0.01 (KITTI) / -0.04 (vKITTI2) px, mixed PEP. The big gains are in seg (+4 to +6 mIoU absolute). So the answer is: mostly seg improves; stereo is essentially unchanged by tighter coupling, and the paper says so ("slightly better").

### 8. Negative results & limitations
- Authors: still needs both annotations; plan to improve efficiency (385M params, 0.30 s/img on 3090).
- Stereo gain not significant (above); all ablations seg-only; no stereo ablation per block.
- Hyperparameters alpha, beta chosen on the KITTI test split; single seed; tiny test sets (200 vKITTI2, ~60 KITTI).
- Gains may come from capacity (385M params) and from changed deep-supervision, no parameter-matched baseline.
- No cross-domain/zero-shot depth evaluation. Cityscapes depth is pseudo-labelled by ViTAStereo, so the Cityscapes stereo branch is supervised by another model.
- Per-image-averaged mIoU inflates/changes numbers vs standard dataset-level mIoU (Tab. IV mmseg numbers differ by 6-14 points from Tab. I-III).
- Supplement referenced but absent in the provided PDF.
- Repo summary errors: wrong L_CT formula; describes L_DIA as "disparity-informed semantic alignment" (it is a LR-consistency-weighted CE) and L_DSCC as "disparity-semantic cross-consistency" (it is KL among the deep-supervision seg outputs); says "iterative stereo branch with shared encoder" while only the first 3 contextual layers share weights; "official checkpoint" claim is not in the PDF.

### 9. Relevance to OUR model
- Insertion-point-compatible ideas:
  1. L_DIA-style weighting: re-weight the semantic or disparity loss by left-right disparity inconsistency (occlusion/mismatch indicator). For our frozen A09 we can compute |D^L - warp(D^R)| at inference/train time if the stereo model emits both views, otherwise via a flipped-input pass (2x cost; for training only). Use: weight the D2/E3 residual loss (or the gate target) toward occluded/inconsistent pixels; and at inference the same inconsistency map can be an extra gate input to SemanticCostGate (cost: one extra stereo pass, or a warp check if right disparity is free). Expected benefit: modest on EPE in occlusions; low risk; moderately novel (SGNet-style semantic gate + LR-confidence is not in the E/F series).
  2. SIG gate formula (1+G)⊙X + (1−G)⊙G_prev⊙R(X_prev): multi-scale selective inheritance. Portable only if we gate semantic features across 1/8 and 1/16 scales into the cost gate; unverified benefit for stereo (their evidence is seg-only), param cost small (convs+sigmoid). Risk medium; skip as first priority.
  3. HDS/DSCC: deep supervision + KL consistency among multi-scale predictions - irrelevant for us since seg decoder is frozen; a disparity-side analogue (consistency across candidate-gate scales) is speculative.
- The paper's evidence cuts AGAINST needing heavy coupling for stereo: tighter coupling bought ~0 px EPE even with 385M params. This supports our finding that a small head on frozen features is the efficient regime; the semantic cue's value is likely limited to occlusion/context ambiguity, not matching accuracy.
- Not novel to us: gated fusion of two cues (done in D1/D2), boundary-weighted loss (G-series).
- Do not copy their eval: per-image mIoU; random 700-pair vKITTI2 split.

### 10. Key quotes/equations
- "contextual feature maps of the first three layers share weights with the feature maps extracted from the stereo matching network" (p.5).
- "such shareable features may not be suitable for both dense prediction and geometric vision tasks" (p.3, about SSNet citing [43]; not tested).
- Eq.1 SIG (p.4), Eq.5-6 W, W^N (p.5), Eq.7 L_DIA, Eq.8 L_DSCC (p.6), Eq.4 L_CT (p.5).
- "the stereo matching performance of TiCoSS is slightly better than that of S3M-Net" (p.7).
- "385.05 million trainable parameters ... 308.86 GFLOPs ... 512x256 ... 0.30 seconds per image ... 5.82 GB" (p.11).
