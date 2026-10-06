<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# DispSegNet card

### 0. Meta
- Title: DispSegNet: Leveraging Semantics for End-to-End Learning of Disparity Estimation from Stereo Imagery. Zhang (first), Skinner, Vasudevan, Johnson-Roberson. IEEE RA-L 2019 (arXiv 1809.04734v2).
- PDF: paper/reference_papers/semantic_stereo/DispSegNet_Zhang_RAL2019.pdf (9 pp; p. 9 is a reference tail). No supplement in the PDF. Code: not stated.
- Domain: driving. Datasets: Cityscapes (5,000 stereo pairs: 2,975 train/500 val/1,525 test; no GT disparity, SGM disparity provided; left-image semantic GT) and KITTI 2015 (200 train, split 160/40 for validation; only 2015 has semantic GT) (Sec. IV-A).
- Training is UNSUPERVISED for disparity (no GT disparity used for the loss).

### 1. Problem & failure modes targeted
- Ill-posed regions: textureless (centre of road), occluded, reflective/specular (car windshields) (Sec. I, Fig. 1).
- Scarcity of GT disparity: unsupervised photometric learning, with semantics to close the gap to supervised methods.
- Disparity smoothness over-smoothing small objects (poles, signs) under naive smoothness regulariser (Sec. IV-D, Tab. III).
- Differs from SegStereo: SegStereo uses a correlation layer (information loss); DispSegNet retains all features in a concat cost volume and uses a two-stage refinement; semantics also regularise the loss (Sec. II).

### 2. Pipeline by stage
- 2a Feature extraction (Sec. III-A, Fig. 2): shared siamese ResNet-50 (weights shared L/R), first conv 7x7, others 3x3, BN + leaky ReLU, 2D residual blocks 3 layers deep. Disparity branch features at 1/4 resolution; segmentation uses DEEPER layers of the same ResNet (more context). Trained, not frozen. Channel counts: not stated.
- 2b Semantic branch (Sec. III-D): PSP module (PSPNet style) on features at 1/8 res: avg-pool to 1/2, 1/4, 1/8 of input size, 1x1 conv to 1/4 of the feature dimension each, bilinear upsample, concat, 1x1 conv to mix -> class logits. Applied to both left and right features (seg for both views). 'Segment embedding' = learned seg features; resized to full image shape before concat with initial disparity (Sec. III-A). Class count not stated (KITTI 2015/Cityscapes labels). Right-view seg is supervised via warp: left disparity warps the right segmentation to left, which is supervised by left GT (Sec. III-D).
- 2c Cost volume (Sec. III-B): 5D, per-side: Batch x (max_disp+1) x H x W x Feature, formed by concatenating every left feature with all candidate right features (and symmetrically for the right view). Features at 1/4 res; max disparity 192 (Sec. IV-B; whether at 1/4 or full res not stated). No correlation, no group-wise.
- 2d Aggregation: 8-layer 3D conv encoder-decoder, 3D transposed convs in decoder, 3D residual blocks (2 layers deep) in the skip connections (Fig. 2). Memory-heavy; encoder-decoder used to keep footprint down.
- 2e Disparity: soft-argmin for the initial disparity (Sec. III-B).
- 2f Refinement (Sec. III-C): residual 2D conv net takes [initial disparity, semantic segment embedding] concatenated (embedding resized to image shape), outputs a residual added to the initial disparity. Focus on ill-posed regions; "same smoothness within semantic segment". Layer count/widths not stated.
- 2g Upsampling: not described (embedding is resized to image shape; initial disparity upsampling method not stated).
- 2h FUSION POINTS:
  1. Shared ResNet-50 trunk: feature level; seg uses deeper layers. Gradient coupling only.
  2. Refinement stage: concat of segment embedding with initial disparity (full-res), sem -> disp only, feature level at the refinement stage.
  3. Loss level: smoothness loss L_s uses shallow shared features f_L ("shallower layers ... biasing to smaller segments", Sec. III-E) in the edge-aware weights, so semantic-derived features modulate smoothness inside segments (sem -> disp).
  4. Seg supervision for the right view via disparity warping (disp -> sem), but the paper itself notes seg IoU falls after refinement (47.6% -> 46.9%).

### 3. Block -> problem -> evidence table
Evidence from Tab. IV (KITTI 2015 train set, no pretraining, no post-proc; units: % error pixels (3 px & 5%), NOC / ALL), 40-image val split in Tab. I.
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Initial stage losses only (L_p, L_c, L_r init) | baseline | NOC 7.18 / All 8.75 (Tab. IV row 1) | no pretrain, no PP | not stated |
| + refinement stage with photometric+consistency (no smoothness, no seg) | two-stage structure | 7.04 / 8.60 (row 2): -0.14 / -0.15 | | not stated |
| + seg supervision on initial only (no refinement) | semantic features | 6.70 / 8.14 (row 3): -0.48/-0.61 vs row 1 | | not stated |
| + refinement with smoothness loss L_s (feature-based) | over/under-smoothing within segments | 6.53 / 6.94 (row 4): -0.51 vs row 2 NOC, -1.66 All | | |
| ref losses (p,c,s) + seg, with NO init-stage losses | tests value of init-stage supervision | 5.99 / 6.42 (row 5; init losses add only -0.06/-0.10 when all else present, row 6) | | |
| all losses | | 5.93 / 6.32 (row 6) | | |
| Smoothness + seg: per-class error reduction | | Tab. III e.g. road 2.65->1.35 (-48.88 %), pole 11.26->7.62 (-32.37 %), car 12.94->8.83 (-31.72 %), traffic sign 6.33->4.67, bus 2.31->1.77, motorbike 0.34->0.36 (+6.38 % worse) | | |
| smoothness alone (no seg), per class | | Tab. III row 2 vs baseline: improves road 2.65->1.51, car 12.94->10.53, bus 2.31->1.85, but WORSENS pole 11.26->13.13, traffic light 2.34->2.63, traffic sign 6.33->6.35, rider 1.06->1.24 | | |
| Cityscapes pretrain, post-processing (Tab. I, 40 val) | domain/PP | K 5.93/6.32; CS 6.55/7.24; CS&K 5.84/6.29; K+pp 5.29/5.69; CS&K+pp 5.20/5.67 | | PP = LR check (thresh 1) + median + background interpolation |
| Segment embedding (architecture) | the headline claim | NOT ABLATED as architecture (no refinement net with disparity-only input vs with embedding). Fig. 1: single qualitative example 6.22% -> 1.42% error in a boxed region | | |
| PSP module, ResNet-50, concat cost volume, 8-layer 3D enc-dec, photometric weights | | not ablated | | |

### 4. Interactions & dependencies
- Smoothness loss helps large classes and HURTS small ones unless combined with seg supervision (Tab. III, p. 7): without semantic supervision it "blindly" smooths to neighbors; with seg supervision the shallow features become segment-aware.
- L_s is only valid in the refinement stage because it relies on a left-right consistency mask (|D_L - warp(D_R)| <= t, t=3) that needs a reasonably good initial disparity (Sec. III-E).
- The refinement stage presumes the initial disparity is "reasonable in most regions" (Sec. III-C).
- Photometric supervision needs matched regions; foreground (occlusions/reflections on vehicles) has no correspondence, so fg stays bad (Sec. IV-C).
- Seg and disparity compete: IoU 47.6% -> 46.9% after the disparity loss/refinement (Sec. IV-E).

### 5. Losses
- Eq. 1: Loss = a1 L_init + a2 L_ref + a3 L_seg; a = 0.3, 0.7, 0.1.
- Eq. 2: L_init = b1 L_p + b2 L_c + b3 L_r; b = 0.8, 0.01, 0.001.
- Eq. 3: L_ref = g1 L_p + g2 L_c + g3 L_s; g = 0.8, 0.05, 0.005.
- Eq. 4: L_p = l1 S(I_L, I'_L) + l2 |I_L - I'_L| + l3 |grad I_L - grad I'_L|; l = 0.85, 0.15, 0.15 (as printed; sums to 1.15). S = SSIM; I'_L = warp(I_R, D_L), computed for both L and R.
- Eq. 5: L_r = (1/N) sum |d2x D_L| exp(-|d2x I_L|) + |d2y D_L| exp(-|d2y I_L|) (second-derivative edge-aware smoothness, init stage).
- Eq. 6: L_c = |I_L - I''_L| + |I_R - I''_R| (reconstruct from the reconstructed other view).
- Eq. 7: Diff = ||D_L - D'_L|| if <= t else t, with D'_L = warp(D_R, D_L), t=3.
- Eq. 8: L_s = (1/N) sum |d2x D_L| (exp(-|d2x f_L|) + exp(Diff - t)) + |d2y D_L| (exp(-|d2y f_L|) + exp(Diff - t)) with f_L the shallow left feature vectors.
- L_seg: softmax cross-entropy; right-view seg warped by left disparity to left GT.
- Note the weights: the segmentation weight a3 is tiny (0.1) and the L_s weight g3 is 0.005; no GT disparity loss.

### 6. Training recipe
- TensorFlow, single NVIDIA Titan X, batch size 1 (memory limit), random 256x512 crops, max disparity 192, images normalised to [-1, 1] (Sec. IV-B).
- Adam beta (0.9, 0.999), eps 1e-8; lr 2e-4 for Cityscapes pre-training, 1e-4 for KITTI fine-tune, halved every 20,000 iterations. Cityscapes 100,000 iterations, KITTI 50,000 iterations (~1 day). No data augmentation.
- Single joint stage (no frozen modules described); all losses jointly.
- Post-processing at test: LR consistency check (threshold 1), median filter on mask, interpolate invalid regions from background.

### 7. Results
- KITTI 2015 val (40 imgs), unsupervised Tab. I (NOC/All %): USCNN 11.17/16.55; Zhou 8.61/9.91; Godard 9.19 (All); SegStereo 7.70/8.79; Luo 6.31/6.63; Ours CS 6.55/7.24; K 5.93/6.32; CS&K 5.84/6.29; K+pp 5.29/5.69; CS&K+pp 5.20/5.67.
- KITTI 2015 test Tab. II (D1-bg/fg/all, NOC | All, runtime): Ours 3.86/15.89/5.84 | 4.20/16.97/6.33, 0.9 s; DispNet 4.11/3.72/4.05 | 4.32/4.41/4.34, 0.06 s; PSMNet 1.71/4.31/2.14 | 1.86/4.62/2.32, 0.41 s; GC-Net 2.02/5.58/2.61 | 2.21/6.16/2.87, 0.9 s. Beats supervised DispNet on bg only.
- Seg baseline IoU 47.6% (40 val imgs); 46.9% after refinement (Sec. IV-E).
- EPE, params, FLOPs, memory, hardware for the 0.9 s runtime: not stated.
- Local summary errors: it quotes "KITTI all-pixel EPE 2.17 to 1.89 px and D1 10.53 to 10.03%" - these numbers do NOT appear in the PDF; the paper reports no EPE at all, and the corresponding ablation is Tab. IV in % error (e.g. 8.14 -> 6.32). It says "supervised disparity regression when ground truth exists" - the paper is fully unsupervised. It calls the semantic part an "encoder-decoder" - it is a PSP head over ResNet-50 features.

### 8. Negative results & limitations
- Authors: large foreground error (D1-fg 15.89/16.97 vs DispNet 3.72/4.41) because of occlusions/reflections on vehicles and reliance on photometric correspondence; smoothness loss alone degrades small classes (pole, traffic light/sign); seg IoU drops slightly after disparity learning (47.6 -> 46.9).
- Mine: (1) No architecture ablation isolating the segment embedding input to refinement; the ablations vary losses, and seg gains could be a feature-regularisation effect, not "semantic content". (2) Weak/inconsistent claim: its own supervised comparison is far behind (D1-all 6.33 vs 2.32 PSMNet). (3) Single dataset family, 40-img val. (4) Eq. 4 weights sum to 1.15. (5) Embedding source layer ambiguous (final seg vs shallow). (6) KITTI semantic-label images overlap the evaluation images used for stereo (same 200 scenes), a mild leakage risk for semantic supervision on the val frames is not discussed. (7) Not real-time (0.9 s).

### 9. Relevance to OUR model
- Idea 1: concat segment embedding + initial disparity -> residual (insertion at refinement) = essentially our ClassResidual, already done; adds nothing beyond full-res concat. Evidence base is weak (loss ablations only), so do not cite as evidence that embedding concat works.
- Idea 2 (potentially portable, novel combination with a frozen trunk): semantic-aware edge-aware smoothness, i.e. replace RGB gradient weights with feature-gradient (semantic embedding) weights and penalise disparity only within segments (Eq. 8). Needs a TRAINING loss only (no runtime cost on RTX 3050/Jetson). Relevant because our G series showed RGB-guided residuals did not sharpen edges: a semantics-gated smoothness could address edge blur differently. Expected benefit: modest; risk: smoothness hurts small classes unless features are segment-aware (Tab. III), which our frozen YOLO features/14-class logits may provide. Loss only; works with supervised VKITTI2 GT, so use as regulariser on the fusion head.
- Idea 3: left-right consistency mask for refinement-only regularisation - needs unsupervised data; low relevance with GT.
- Idea 4: warping seg from the other view using disparity to supervise the right view - irrelevant (our decoder frozen).
- Unsupervised photometric training on real driving pairs could adapt the fusion head to KITTI-like domains (our F-series zero-shot gap), but their unsupervised numbers (D1-all 6.33 test) are far worse than supervised so only attractive as adaptation.
- Cost: concat 5D volume + 3D conv is heavy (0.9 s, batch 1), not portable to our real-time target.

### 10. Key quotes/equations worth citing
- "disparity tends to be smooth within an object or segment" (p. 1); "Disparity in the ill-posed regions should have similar values as regions from the same semantic segment" (p. 3).
- Eq. 8 (p. 4) feature-aware smoothness; Tab. III (p. 7) per-class error with/without seg supervision; Tab. IV (p. 7) loss ablation.
- "the smoothness loss helps ... large semantic classes but not for small semantic classes" (p. 7).
