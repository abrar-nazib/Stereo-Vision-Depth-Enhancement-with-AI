<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# RTS2Net card

### 0. Meta
- Title: Real-Time Semantic Stereo Matching (RTS2Net). Authors: Dovesi (first), Poggi, Andraghetti, Marti, Kjellstrom, Pieropan, Mattoccia. ICRA 2020 (arXiv 1910.00541v2).
- PDF: paper/reference_papers/semantic_stereo/RTS2Net_Dovesi_ICRA2020.pdf (8 pages; pp. 7-8 are references only).
- Code: not stated. Supplement: only a KITTI video sequence at https://www.youtube.com/watch?v=wbtQcWAbgo0 (p. 8 footnote). There is NO written appendix/supplement in the PDF, so no extra tables exist beyond the 8 pages.
- Domain: driving. Datasets: Cityscapes (CS; ~25K stereo pairs, 5K dense + 20K coarse labels, SGM-derived disparity GT, Sec. IV-A), SceneFlow (only as a comparison pretrain), KITTI 2015 (160 train / 40 val split of the 200 train images, Sec. IV).
- Hardware: NVIDIA RTX 2080 Ti and Jetson TX2 (Tab. II-IV). Training hardware not stated.

### 1. Problem & failure modes targeted
- Latency/memory: prior semantic-stereo nets (SegStereo, DispSegNet) "barely break the 1 FPS barrier" (p. 2); goal is one compact net with one forward pass for both tasks, usable on <15 W embedded (Sec. I).
- Task redundancy: running separate seg + stereo nets = two passes; single large joint nets are complex (abstract).
- Mutual benefit claims: depth ambiguous in reflective regions improved by knowing "car"; seg ambiguity (vegetation vs terrain) helped by depth (Sec. I). Only the first is actually measured (Tab. III); seg gain from depth is NOT isolated (the seg drop after refinement, see Sec. 3).
- Deployment flexibility: a speed/accuracy knob at test time (anytime early exit, width c) (abstract, Sec. IV-D).

### 2. Pipeline by stage
- 2a Feature extraction (Sec. III-B, Fig. 2): ONE shared siamese encoder (weights shared across L/R; seg uses left only), jointly trained with both decoders (NOT frozen). Two 3x3 convs with c channels bringing res to 1/2, then 4 blocks each = 2x2 max-pool + two 3x3 convs, outputting 2c, 4c, 8c, 16c at 1/4, 1/8, 1/16, 1/32. BN+ReLU after every conv. c is the width hyperparameter (c=1 -> AnyNet encoder; c=8 is the "practical" setting; c=32 for KITTI online submission). Tiny vs VGG-style encoders. Disparity branch taps 8c@1/16, 4c@1/8, 2c@1/4 (Fig. 2 "L feat/R feat" labels); the 16c@1/32 features go ONLY to the semantic branch for context (Sec. III-D). The disparity branch deliberately does not use 1/32 (tested: no gain, Sec. III-C).
- 2b Semantic branch (Sec. III-D, Fig. 2 green): extra 2D convs on shared features (per-task embedding, as in the disparity branch), 3 coarse-to-fine stages at 1/16, 1/8, 1/4. Each stage emits per-pixel class probability scores (KITTI benchmark classes; class count N not stated in the paper); previous-stage probabilities are upsampled and summed via residual connections; argmax gives the map. Trained jointly, not frozen. Only left image.
- 2c Cost volume (Sec. III-C): distance-based (feature DIFFERENCE, not concat/corr): right features shifted up to d_max and subtracted from left. Stage 1 at 1/16: d_max=12 (-> 192 px at full res). Stages 2, 3: residual volumes after warping right features with the upsampled coarse disparity, d_max=+-2 (Sec. III-C says "+-16 at full resolution", which matches +-2 at 1/8; stage 3 is "identical to the previous", so by my inference +-2 at 1/4 = +-8 px full-res).
- 2d Aggregation: 3 3D conv blocks (BN+ReLU) channels 16, 16, 1 at stage 1; stages 2 and 3: 3 3D convs with 4, 4, 1 channels ("same amount of channels as AnyNet"). No hourglass/attention.
- 2e Disparity: soft-argmin per stage; later stages add residual to the upsampled previous estimate.
- 2f Refinement: the coarse-to-fine residual stages above (3 stages, no iterations/GRU). Plus the Synergy Disparity Refinement module (see 2h).
- 2g Upsampling: bilinear from 1/4 to full-res (Sec. III-C). Convex/learned upsampling absent.
- 2h FUSION POINTS (the only explicit one is the synergy module; the other coupling is shared-encoder multi-task):
  1. Shared encoder: feature level, bidirectional only through gradients (disp loss + seg loss shape one trunk). No explicit exchange. Gain on disparity from this alone: EPE 0.91->0.90, D1 3.98->3.90 (Tab. III row 1 vs 2), tiny.
  2. Synergy Disparity Refinement (Sec. III-E, Fig. 2 purple): stage = after the disparity and semantic decoders, per resolution (1/16, 1/8, 1/4). Operator: concat of COMPRESSED semantic class-probability features (compressed to dimensionality similar to the disparity cost volume, to "balance contributions") with the disparity cost volume reorganized so disparity is the channel dimension ("hybrid volume"); at stages 2 and 3 also concat the previously refined disparity. Three 2D convs produce RESIDUALS added to the (reorganized) cost volume, then soft-argmin gives refined disparity. Direction: semantic -> disparity only (disparity does not feed back to semantics). Residual cascade like the rest of the net. Conv widths/params of this module: not stated.
  3. Loss coupling: stage-weighted joint loss (Sec. 5).

### 3. Block -> problem -> evidence table
All on KITTI 2015 val (40 imgs), c=8 unless noted, models pretrained on CS then KITTI. FPS: TX2 / 2080Ti.
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Disp-only subnet at c=8 (baseline) | stereo | EPE 0.91, D1 3.98 (Tab. III row 1) | c=8 | 8.1 / 96.2 FPS |
| + semantic decoder, joint training (no synergy) | seg output + mild disp regularisation | EPE 0.90 (-0.01), D1 3.90 (-0.08); mIoU 64.21, pAcc 91.56 (Tab. III row 2) | joint training CS->KITTI | 6.6 / 76.9 FPS (derived: +~28 ms TX2, +~2.6 ms 2080Ti) |
| + Synergy refinement | disparity in ambiguous regions | EPE 0.91 -> 0.84 (-0.07), D1 3.91 -> 3.33 (-0.58) (Tab. III row 3, bracket values are refined; unbracketed = before refinement). Costs seg: mIoU 64.21->62.22 (-1.99), pAcc 91.56->90.64 (-0.92) (p. 5) | same | 6.3 / 60.4 FPS (derived: +~8 ms TX2, +~3.6 ms 2080Ti) |
| Width c sweep | seg needs capacity | RTS2Net EPE/D1/mIoU: c=1 1.12/5.57/58.86; c=4 0.90/3.80/60.93; c=8 0.84/3.33/62.22; c=16 0.78/2.90/67.41; c=32 0.74/2.62/69.62 (Tab. II). AnyNet same c: EPE 1.14/0.96/0.91/0.87/0.82, D1 5.75/4.22/3.98/3.52/3.12 | | TX2 FPS 8.3/7.4/6.3/4.5/2.3; 2080Ti 60.5/60.5/60.4/60.4/42.2 |
| Gap vs disparity-only AnyNet widens with c | claim: wider shared trunk benefits more from multitask | D1 margin 0.18/0.42/0.65/0.62/0.50 (derived from Tab. II); EPE margin 0.02/0.06/0.07/0.09/0.08 from the table, but TEXT says 0.12 for c=32 (p. 5), inconsistent with table (0.82-0.74=0.08) | | |
| Anytime / early stop (Tab. IV, TX2, D1-all) | latency knob | RTS2Net c=8: stage1 17.2 FPS 8.00%, stage2 10.9 FPS 4.70%, stage3 6.3 FPS 3.33%. AnyNet (this is the c=1 AnyNet: 10.4 FPS / 5.75% in Tab. II): 34.6 FPS 11.60, 20.5 FPS 8.40, 10.4 FPS 5.75 | | stage costs derived: ~58 ms, ~92 ms (+34), ~159 ms (+67) |
| Coarse-to-fine stages stopping at 1/16, 1/8, 1/4 | memory/runtime | choice of 1/32 decoder "did not improve results"; lower-res decoders add runtime with "negligible improvements" (Sec. III-C); no table | | not ablated |
| CS pretraining (coarse 60 ep -> fine 75 ep) vs SceneFlow | training data | c=1 AnyNet-like variant: SceneFlow 10 ep + KITTI 300 ep: EPE 1.24 / D1 6.47; SceneFlow 40 + KITTI 800: 1.18/6.28; CS 60->75 + KITTI 800: 1.14/5.75 (Tab. I) | SF has no semantic labels (instance only), so CS is required for seg | |
| Class-balanced seg loss weights W_j; coarse-label loss L_s* | class imbalance; coarse CS labels | not ablated | | |
| Hierarchical stage weights | stabilise early exits | not ablated | | |
| Smooth-L1 | | not ablated | | |

Remark: the (0.91 -> 0.84) refinement gain has NO no-semantics control (a refinement of the disparity volume alone with same capacity); attribution to semantics is unproven.

### 4. Interactions & dependencies
- Early exit works because every stage is supervised (deep supervision) and decoders are residual: "early losses stabilize and accelerate" training (Sec. III-A).
- Semantic quality depends on c: c=1 gives mIoU 58.86, c>=16 needed for 67+; the shared encoder is the bottleneck for seg, and for the joint model the encoder must be wide (c=1 "insufficient" for seg, Sec. III-B).
- Synergy refinement trades semantic accuracy (-1.99 mIoU) for disparity (-0.58 D1): the refinement path appears to push shared features/gradients towards disparity (seg loss weight W_s=2 vs W_dr=2 equal). The paper does not explain the seg drop.
- Disparity subnet dominates runtime (~120 ms of ~160 ms at c=8 on TX2, Sec. IV-B), so the semantic branch and encoder are cheap relative to stereo.
- SceneFlow pretraining is unusable for seg (no semantic labels), so CS is a joint-training prerequisite.

### 5. Losses
- Smooth L1 (Eq. 1): 0.5(d-d^)^2 if |d-d^|<1 else |d-d^|-0.5, averaged over pixels, for disparity at each stage (L_d), refined disparity (L_dr); multi-class cross-entropy for seg (L_s).
- Eq. 2: L = sum_{st=1..3} W_st (W_d L_d,st + W_s L_s,st + W_dr L_dr,st). W_st = 1/4, 1/2, 1 for the three stages; W_d, W_s, W_dr = 1, 2, 2 (Sec. III-F).
- Eq. 3: class-weighted CE with W_j = N / ( log(P_j + k) * sum_i 1/log(P_i + k) ) (ENet-style), k=1.12 for Cityscapes, 2 for KITTI 2015. N = number of classes, P = class probability.
- Eq. 4: coarse-annotation reweight L_s* = L_s (1 + gamma * A_unlab / (A_tot - A_unlab)), gamma=0.1.
- All losses are applied to coarse-resolution outputs at their own resolution (GT downscaled; method for downscaling not stated).

### 6. Training recipe (Sec. IV-A)
- Adam, beta (0.9, 0.999), lr 5e-4 constant on SF/CS, halved every 200 epochs on KITTI. Crops 256x512, batch 6.
- Schedule: CS coarse 60 epochs -> CS fine 75 epochs -> KITTI 800 epochs (Tab. I; best vs SF alternatives). Everything is trained jointly; nothing frozen; no curriculum between branches stated.
- Augmentation: not stated. Training hardware: not stated. Framework: not stated.
- KITTI val: 40 imgs, 160 train.

### 7. Results
- KITTI 2015 val, c=8: EPE 0.84, D1 3.33, mIoU 62.22, pAcc 90.64, 6.3 FPS on TX2 (~160 ms), 60.4 FPS on 2080Ti (Tab. II, Sec. IV-B).
- c=32 (online submission): KITTI 2015 online D1-bg/fg/all 3.09/5.91/3.56 %, runtime 0.02 s (Tab. V), vs StereoNet 4.30/7.45/4.83 0.02 s, MADNet 3.75/9.20/4.66 0.02 s, DispNetC 4.32/4.41/4.34 0.06 s, PSMNet 1.86/4.62/2.32 0.41 s, GANet 1.48/3.46/1.81 1.80 s, SegStereo 1.88/4.07/2.25 0.60 s.
- KITTI semantic benchmark (Tab. VI): class IoU/iIoU 57.67/27.42, category IoU/iIoU 82.85/60.72, 0.02 s (0.008 s semantic-only). SegStereo 59.10/28.00/81.31/60.26 0.60 s; SDNet 51.14/17.74/79.62/50.45 0.20 s.
- Anytime TX2 results: Tab. IV above. "RTS2Net stage 2 vs AnyNet stage 3": 10.9 vs 10.4 FPS and D1 4.70 vs 5.75, i.e. -1.05 %; at 10 FPS-compatible budget (KITTI camera 10 FPS, Sec. IV-D). Full model reduces D1 by 2.42 % vs AnyNet (5.75 -> 3.33) at 6.3 vs 10.4 FPS.
- Params/FLOPs/memory: not stated anywhere.

### 8. Negative results & limitations
- Authors: c=1 insufficient for semantics (Sec. III-B); 1/32 disparity decoder no gain (Sec. III-C); synergy refinement lowers seg accuracy (p. 5); TX2 6.3 FPS is slow for c>=16 (4.5, 2.3 FPS).
- Mine: (1) no same-capacity no-semantics refinement control, so the -0.58 D1 cannot be attributed to semantic content (our E/F series do this). (2) All gains in-domain on a 40-image KITTI val, single seed, no variance; D1 differences of 0.1-0.4 on 40 images are fragile. (3) AnyNet comparison is at c=1 in Tab. IV (unequal width) - the Tab. II rows at the same c are the fair ones; Tab. IV is "AnyNet c=1 vs RTS2Net c=8". (4) FPS protocol (batch, precision, image size, power mode) not stated; Tab. IV stage FPS does not say whether semantic/refinement are executed before exit. Stage 3 FPS equals the full-model FPS (6.3) while D1 3.33 is the refined value, so early stages' D1 are likely refined outputs "REF. stage 1/2" (Fig. 2) but this is not stated. (5) Fig. 2 and text disagree on nothing material, but the EPE margin "0.12" in the text for c=32 contradicts Tab. II (0.08). (6) CS disparity GT is SGM output (noisy); pretraining on it helps (Tab. I) which is a real/domain-style advantage unrelated to semantics. (7) The semantic gain on disparity from multitask alone is 0.01 EPE: essentially none.
- Errors in the local summary (summaries/semantic_stereo/RTS2Net.md): it says "19 classes" - the paper never states a class count; it omits that the synergy module is applied to the COST VOLUME (hybrid volume, pre-soft-argmin) and compresses semantic channels; omits the Tab. III ablation (semantic-only vs refinement gains), the loss weights, the Tab. II/IV baseline mix, and the 0.12-vs-0.08 inconsistency. Tab. II/IV numbers it lists are correct.

### 9. Relevance to OUR model
Closest real-time shared-encoder precedent, but their encoder is trained from scratch/jointly for both tasks (not a frozen YOLO trunk) and the disparity side is a tiny AnyNet (c=8 up to c=32 channels), so absolute results are not comparable.
- Shared encoder, run once: same principle as ours; RTS2Net proves the semantic branch is cheap relative to stereo (about +28 ms of 159 ms on TX2, derived from Tab. III FPS; stereo subnet ~120 ms). Supports the claim that our shared-trunk design is the right latency strategy. NOT novel; our novelty is the frozen pretrained trunk reuse (YOLO26m seg trunk) and no re-training of stereo.
- Synergy refinement = our SemanticCostGate/ClassResidual lineage, but additive concat-then-conv on the cost volume with COMPRESSED semantic probabilities, versus our multiplicative gating of candidates (SGNet-style). Portable idea: compress semantic probabilities to the cost-volume channel width before fusing, to avoid semantics dominating (their balance trick); cheap (3 2D convs). Insertion: our candidate/cost-volume stage at 1/4. Expected benefit: small, they got -0.07 EPE/-0.58 D1 in-domain with no control. Cost: small. Risk: our G-series shows extra residual stacks can fail to improve EPE; their gain also lacks a control, so low evidence of semantic specificity. Priority: low (we already have a gate that beats controls).
- Anytime exit: portable and genuinely useful for Jetson targets. Our frozen A09 produces tiles/refinement at one resolution; exposing a "skip fusion head" path is trivially available (fusion head is the last stage). The real lesson: RTS2Net spends +67 ms (stage 3) for the finest-resolution stage, and the final refinement is only +8 ms. In our design the equivalent is: the fusion head is the cheap part; the cost is in the frozen stereo model. Don't re-architect; expose head-on/off and measure per-stage latency on RTX 3050 4GB/Jetson.
- Deep supervision with stage weights 1/4,1/2,1 and early-stop outputs: only applicable if the frozen stereo predictor exposes multi-res outputs; novel-combination potential is low.
- Pretraining with real-domain semantic data (CS, SGM GT) beats SceneFlow (EPE 1.14 vs 1.18, D1 5.75 vs 6.28, Tab. I): relevant to our KITTI zero-shot result; our F-series gap (3.08 stereo-only vs 2.59 E3) suggests semantics helps transfer, and RTS2Net's Tab. I suggests data-domain matters at least as much. Idea: fine-tune the fusion head on pseudo-labelled or SGM-labelled real pairs (risk: label noise; separate decision).
- Class-balance weighting formula: not applicable (our semantic decoder is frozen).
- Seg drop after refinement (-1.99 mIoU) is a trap RTS2Net could not avoid because seg shares the trunk; our frozen decoder is immune by design, which is a stated advantage to claim in the manuscript.
- Novel combination vs done: frozen shared trunk + frozen stereo + semantic cost gate + matched no-semantics control + zero-shot KITTI is not in RTS2Net. What is done: shared encoder + semantic->disparity hybrid volume residual refinement + early exit.

### 10. Key quotes/equations worth citing
- "RTS2Net represents the first real-time solution for joint semantic segmentation and stereo matching running seamlessly on high-end GPUs and low-power devices." (p. 2)
- "By building the volume at low resolution, a small d_max is enough to look for the entire disparity range at the original resolution... d_max = 12, corresponding to 192 maximum disparity" (p. 3)
- Eq. 2 (p. 4): L = sum_st W_st (W_d L_d + W_s L_s + W_dr L_dr), W_st = 1/4,1/2,1; W = 1,2,2.
- "most of the time is taken by the disparity subnetwork (120ms)" (p. 5)
- "this fully residual setup provides consistent advantages both at training-time ... and at testing-time since we can dynamically adjust the speed/accuracy trade-off" (p. 2)
