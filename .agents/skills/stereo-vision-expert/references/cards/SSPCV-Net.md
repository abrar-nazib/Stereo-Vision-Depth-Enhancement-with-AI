<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SSPCV-Net card

Read: whole PDF (10 pp = 8 pp body + 2 pp references; no supplement). Page refs use the ICCV page numbers 7484-7493 as "p. N" where N = page 1..10 of the file (p. 1 = 7484).

### 0. Meta
- Title: Semantic Stereo Matching with Pyramid Cost Volumes
- Authors: Zhenyao Wu, Xinyi Wu, Xiaoping Zhang, Song Wang, Lili Ju (Univ. of South Carolina, Wuhan Univ., Farsee2)
- Venue: ICCV 2019
- PDF: paper/reference_papers/semantic_stereo/SSPCV-Net_Wu_ICCV2019.pdf
- Code URL: not stated.
- Domain: driving + synthetic.
- Datasets: Scene Flow (35,454 train / 4,370 test), KITTI 2015 (200 train, left images have semantic labels; 200 test), KITTI 2012 (194/195, no semantic labels), Cityscapes (1,525 test pairs, SGM-precomputed disparity, used only for generalisation) (p. 5).

### 1. Problem & failure modes targeted
- Single-scale cost volume cannot capture the spatial relationship between objects and neighbours (p. 1); disparity details and object boundaries are error-prone; semantics (object extent, boundaries) "can help rectify the disparity values along the object boundaries" (p. 1).
- Generalisation to a new domain (Cityscapes), qualitatively (p. 10).
- Not targeted: latency, memory (no numbers reported), occlusion, thin structures explicitly.

### 2. Pipeline by stage
- 2a Feature extraction: ResNet-50 with dilated convolution strategy [6,40] (p. 3), shared-weight siamese for left/right. Output feature maps at 1/4 resolution (largest cost volume level is alpha = 1/4). Then adaptive average pooling compresses the features into three scales (1/4, 1/8, 1/16 relative to input, Fig. 3) each followed by a 1x1 conv "to change the dimension" (channel counts not stated). Trained (frozen status of ResNet: not stated).
- 2b Semantic branch: a semantic segmentation subnetwork following PSPNet [44]; it takes the same pooled multilevel features, upsamples the low-dimension feature maps to the same size, concatenates them, followed by a conv layer to produce the segmentation map. The features BEFORE the classification layer form the semantic features (p. 4). Class count: not stated (KITTI-15 labels; Scene Flow segmentation labels are "transformed from object labels"). Stage-1 training: segmentation subnet alone supervised, then joint training (p. 5). Whether weights are frozen/lr-reduced in the joint stage: not stated (appears trained). KITTI-15 semantic sub-network quality: IoU 56.43 % per class, 82.21 % per category (p. 6).
- 2c Cost volume construction: GC-Net style concat of left unary with right unary shifted at each disparity, packed into a 4D volume (C x aW x aH x aD), alpha in {1/4, 1/8, 1/16}: three "spatial pyramid cost volumes" built from the three pooled feature levels, plus ONE semantic cost volume of size C x W/4 x H/4 x D/4 built the same way from the semantic features, "the same size as the largest spatial cost volume" (p. 3-4). Max disparity: 256 (Scene Flow), 192 (KITTI-15/12), 16 channels for the Cityscapes generalisation test (p. 5-6). Channel counts: not stated.
- 2d Aggregation/regularisation (3D multi-cost aggregation, Fig. 4, p. 4): hourglass on each of the four volumes (hourglass = 3D conv, stride-2 3D conv, 3D deconv with skip connections; layer counts not stated). Recursive fusion bottom-up: 1/16 volume -> upsample to the 1/8 volume's size -> **3D Feature Fusion Module (FFM)** -> hourglass -> upsample -> FFM with the 1/4 spatial volume -> hourglass -> FFM with the semantic volume (after its own hourglass) -> hourglass -> conv, conv, bilinear to 1 x W x H x D.
  - FFM (Fig. 4): the two cost volumes are summed (residual-block style), 3D adaptive average pooling -> FC -> ReLU -> FC -> sigmoid gives a weight vector (SE-style channel attention); the UPSAMPLED (lower-level) volume is multiplied by this weight vector and ADDED to the other volume.
- 2e Disparity computation: softmax over the final fused cost volume -> soft-argmin (Eq. 1: d^ = sum_{d=0}^{Dmax} d * P(d)).
- 2f Refinement: n/a.
- 2g Upsampling: bilinear to original size on the cost volume (1 x W x H x D) before regression.
- 2h FUSION POINTS:
  1. **Semantic cost volume** (Sec. 3.2.2): stage = cost volume construction. Operator = same concat-shift cost volume built from semantic features. Direction = sem -> disp. Resolution 1/4 (same as the largest spatial level).
  2. **Final FFM with the semantic volume** (Fig. 4): stage = aggregation. Operator = FFM (sum -> channel attention gate -> multiply the upsampled lower-level volume + add). Direction = sem + spatial -> disp. Resolution 1/4.
  3. **Boundary loss** (Eq. 4): loss coupling of semantic GT gradient and predicted disparity gradient; sem(GT) -> disp. No sem -> disp refinement afterwards.
  4. Joint training: shared early features (ResNet-50) are updated by both tasks.

### 3. Block -> problem -> evidence table
Tab. 1 (p. 6): Scene Flow validation (20 epochs; average EPE, px) and KITTI-15 validation (80/20 split, trained WITHOUT Scene Flow pretraining; "percentage of pixels with errors", threshold not stated, assume 3px). Single run each.

| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| Single spatial cost volume (baseline) | n/a | SF 2.12 / KITTI 2.63 (Tab. 1) | 20 epochs on SF; KITTI-15 from scratch | n/r |
| + Semantic branch (semantic cost volume, separately trained segmentation, no joint train) | object context | SF 1.76 / KITTI 2.42: -0.36 EPE (-17 %), -0.21 pt (Tab. 1) | row marks only "semantic branch" ticked | adds a second cost volume + PSP branch; not measured |
| + Semantic branch (joint-train) | | 1.78 / 2.37: SF +0.02, KITTI -0.05 vs separate (Tab. 1) | | |
| + Spatial pyramid cost volumes (no semantics) | multiscale spatial context | 1.21 / 2.11: SF -0.91 EPE (-43 %), KITTI -0.52 pt vs baseline (Tab. 1) | | 3 volumes + hourglasses, not measured |
| + 3D multiple cost volumes (semantic + pyramid, no dilated conv) | combination | 1.04 / 1.99 (Tab. 1): semantic adds -0.17 EPE / -0.12 pt on top of pyramid | | |
| Full w/o FFM (with dilated conv) | feature fusion | 1.07 / 2.10 (Tab. 1) vs full 0.98 / 1.85: FFM worth -0.09 EPE / -0.25 pt (the row is labelled "excluding FFM" but also ticks dilated conv, so the dilated conv's own effect cannot be read) | | |
| Full w/o boundary loss in joint training | boundary alignment | 1.01 / 1.93 vs full 0.98 / 1.85: -0.03 EPE / -0.08 pt from the loss (Tab. 1) | | |
| Dilated convolution | feature extraction | "feature extraction has been improved when the dilated convolution strategy was used" (p. 6); no isolated row | | not ablated in isolation |
| Full SSPCV-Net | | 0.98 / 1.85 (Tab. 1) | | |
| Cost-volume-level removal | | qualitative only: removing lowest-level / highest-level / semantic volume changes small-object accuracy, context/scene detection, edge/shape (Fig. 5, p. 6) | | not quantified |
| Scene Flow test vs PSMNet | | EPE 0.87 vs 1.09; D1-all 3.1 vs 4.2 (Tab. 2) | trained from scratch 40+40 epochs | n/r |
| KITTI-15 test vs SegStereo / PSMNet | | D1-all All 2.11 vs 2.25 / 2.32; Noc 1.91 vs 2.08 / 2.14; fg All 3.89 vs 4.07 / 4.62 (Tab. 3) | | |
| KITTI-12 test vs PSMNet | | 3px Noc 1.47 vs 1.49; All 1.90 vs 1.89 (PSMNet better); 2px Noc 2.47 vs 2.44 (PSMNet better) (Tab. 4) | boundary loss excluded (alpha = 1) | |
| Cityscapes generalisation | domain shift | qualitative (Fig. 9) vs PSMNet/GC-Net; no numbers | train Scene Flow + KITTI-15 only | |
| Equal-capacity non-semantic control for the semantic cost volume | causal check | not ablated | | |

### 4. Interactions & dependencies
- Semantic contribution is smaller than the pyramid contribution: semantic volume alone helps most when the baseline is weak (-0.36 EPE on single-volume SF) but only -0.17 EPE on top of the pyramid (Tab. 1).
- Joint training is not clearly better than separately trained semantic branch (SF 1.78 vs 1.76; KITTI 2.37 vs 2.42) (Tab. 1).
- Boundary loss only possible where semantic labels exist: KITTI-12 trained with alpha = 1 (p. 5); Scene Flow semantic labels derived from object labels.
- Two-step training is required: segmentation subnet first (40 epochs SF / 300 KITTI-15), then joint (40 / 400 epochs).
- FFM requires the volumes to be spatially/disparity aligned: lower levels are upsampled, the semantic volume is at the same size as the finest spatial volume so it can be fused last.
- Memory: four 4D volumes + four hourglasses; the paper gives no numbers but this is a heavy design (training 120 h on 2 x 1080 for Scene Flow).

### 5. Losses (exact)
- L = alpha * L_disp + (1 - alpha) * L_bdry (Eq. 2), alpha = 0.9 for Scene Flow and KITTI-15 (weight on the disparity term, 0 <= alpha <= 1); alpha = 1 for KITTI-12 (no boundary loss) (p. 5). [Existing summary writes L = L_smoothL1 + lambda_b L_boundary: different parameterisation.]
- L_disp = (1/N) sum smooth_L1(d* - d^) over N labeled pixels (Eq. 3).
- L_bdry (Eq. 4) = (1/N) sum_{i,j} ( |phi_x(sem_ij)| * exp(-|phi_x(d^_ij)|) + |phi_y(sem_ij)| * exp(-|phi_y(d^_ij)|) ), phi_x, phi_y = intensity gradients between neighbouring pixels in x and y (FIRST-order, per the text) of the semantic GT label map and the predicted disparity. No mask, no thresholds. Where the semantic GT has a boundary, penalises small predicted disparity gradient. (SGNet later makes this second-order and masked.)
- Semantic subnetwork loss: not stated explicitly (stage 1 trains it "supervised"); presumably cross-entropy, not stated. Whether it also contributes to L in the joint stage: the total L (Eq. 2) lists only disparity and boundary terms; joint loss on semantic segmentation is mentioned in the abstract ("supervision on both semantic segmentation and disparity estimation") but its formula/weight is not given.
- Deep supervision: none (single final regression).

### 6. Training recipe (p. 5)
- PyTorch; two Nvidia 1080 GPUs; Adam beta1 0.9, beta2 0.999; random crops 256x512 or 256x792 (which dataset gets which: not stated).
- Scene Flow: from scratch, constant LR 0.001, batch 2, alpha 0.9; segmentation subnet first 40 epochs (labels from object labels), then joint 40 epochs. ~120 h.
- KITTI-15 & 12: start from the Scene Flow model; LR 0.01, halved every 100 epochs; segmentation subnet 300 epochs on KITTI-15, then joint 400 epochs with alpha 0.9 (KITTI-15) or alpha 1 (KITTI-12). ~70 h per dataset.
- Cityscapes: used only as test; max disparity set to 16 channel (Cityscapes cost-volume channel "16", p. 6).
- Augmentation beyond crops: not stated. Batch size on KITTI: not stated.
- Ablations: Scene Flow 20 epochs; KITTI-15 80/20 split without Scene Flow pretraining (Tab. 1).

### 7. Results
- Scene Flow (Tab. 2): SSPCV-Net EPE 0.87, D1-all 3.1; PSMNet 1.09 / 4.2; EdgeStereo 1.11; SegStereo 1.45 / 3.5; CRL 1.32 / 6.7; iResNet-i2 1.40 / 5.0; GC-Net 1.84 / 9.7; MC-CNN 3.79.
- KITTI-15 test (Tab. 3): All: D1-est 2.11, bg 1.75, fg 3.89, all 2.11; Noc: est 1.91, bg 1.61, fg 3.40, all 1.91. Others All: SegStereo 2.25, PSMNet 2.32, EdgeStereo 2.59, CRL 2.67. CRL best fg (3.59 All, 3.12 Noc).
- KITTI-12 test (Tab. 4) Out-Noc / Out-All: 2px 2.47/3.09, 3px 1.47/1.90, 4px 1.08/1.41, 5px 0.87/1.14. Best in 5 of 8 metrics (text p. 7); PSMNet better on 2px-Noc (2.44) and 3px-All (1.89); EdgeStereo better on 2px-All (2.43).
- Semantic subnetwork on KITTI-15: IoU 56.43 % (per class) / 82.21 % (per category) (p. 6).
- Params, FLOPs, latency, memory: not stated.
- Correction to the existing summary: its "headline KITTI-2015 all-pixel 0.87 px EPE, 3.1 % D1-all" are Scene Flow numbers (Tab. 2); KITTI-15 D1-all is 2.11 %. The summary also says it reports no mIoU (it reports IoU 56.43 / 82.21 on KITTI-15) and says the volumes are "concatenated" (they are fused recursively with FFM and hourglasses).

### 8. Negative results & limitations
- Authors: do not discuss failures; FFM and loss gains small; KITTI-12 gains are narrower (no semantic labels).
- Weaknesses I see:
  - The Scene Flow ablation attributes -0.36 EPE to "semantic branch", but no equal-capacity control (a second cost volume from non-semantic features) is run. Because the semantic volume is just another 1/4 4D volume, capacity and a second set of features may explain much of the gain. Our own controls would show this.
  - On Scene Flow, semantic labels are "transformed from object labels" (p. 5), a different, coarser label set than KITTI's.
  - Ablation row "excluding FFM" also ticks "dilated convolution", so the effect of dilated conv and of FFM are not separated; the dilated-conv ablation is absent.
  - KITTI-15 ablations are from-scratch on 160 images with a 40-image val split, single run, threshold unstated.
  - Joint training is worse than separate training on Scene Flow (1.78 vs 1.76), so the "semantic information" is not demonstrably coupled.
  - Cityscapes generalisation is qualitative only, with GT from SGM.
  - No runtime/memory despite four 4D volumes and a four-hourglass cascade.
  - Table 2 comparison to PSMNet at SF EPE 1.09 (their number from [4]) vs 0.87: different training schedules; no re-run baseline.
  - Boundary loss uses first-order gradients of integer class IDs, the same flaw noted for SGNet; no mask for background (road/sidewalk) boundaries, which SGNet adds precisely because of these false boundaries (SGNet Fig. 4).

### 9. Relevance to OUR model
- Not a good fit overall: a semantic cost volume of concatenated 4D features and a four-hourglass 3D cascade is far too heavy for RTX 3050 4GB / Jetson live use, and our A09 stereo predictor is frozen with its own shallow head.
- Portable pieces:
  1. **FFM-style channel-attention gate** (sum -> global pool -> FC-ReLU-FC-sigmoid -> multiply + add) to fuse a semantic evidence volume with the stereo candidate volume. In our D2 gate we use a bounded 3D conv gate; an SE-style global gate is cheaper and might offer image-level reweighting of semantic evidence (e.g. down-weight semantics on scenes with unreliable segmentation, matching our KITTI domain gap). Insertion point: SemanticCostGate output. Expected benefit: small, plausibly useful in the KITTI zero-shot setting; Risk: global pooling over the disparity axis loses locality; SSPCV gives no isolated FFM-vs-sum ablation that separates gate effect from the dilated conv (Tab. 1 row mixes them).
  2. **Semantic evidence at a second scale**: A09's volume is at 1/16; adding a 1/8 semantic agreement volume would be the pyramid idea restricted to semantics. Their pyramid-only gain was the larger effect (SF -0.91 EPE), but that comes from spatial context in a heavy cascade; for a frozen predictor the stereo volume would not itself have a pyramid. Cost: extra memory; risk: little evidence for a benefit with frozen stereo.
  3. **Boundary loss** (first-order) - superseded by SGNet's masked version; if we adopt a boundary loss we should use SGNet's masking idea and one-hot boundaries.
  4. **Two-stage "semantic first, then joint"** is irrelevant for us: the semantic decoder is frozen.
- Already done in our design: semantic-conditioned cost-volume gating via class agreement (D1/D2), plus equal-capacity controls absent here. The SSPCV claim "semantic cost volume helps" is the claim our D1-vs-D3 control tests (-0.24 px) under a frozen-stereo setting.
- Novelty: SSPCV is evidence that semantic cost volumes are not new; our novelty must rest on the frozen shared trunk (YOLO encoder reused for stereo and semantics) and controlled ablations, not on the semantic cost-volume operator.

### 10. Key quotes/equations worth citing
- "we form a cost volume by concatenating the corresponding unaries from the left and right image features and then packing them into a 4D volume ... C x aW x aH x aD with a in {1/4, 1/8, 1/16}" (p. 3).
- "To form the single semantic cost volume, we use the features before the classification layer ... C x 1/4 W x 1/4 H x 1/4 D, the same size as the largest spatial cost volume" (p. 4).
- FFM: "first the two 3D cost volumes are summed ... the adaptive average pooling is used to transform ... through a fc-ReLU-fc-sigmoid structure ... the upsampled one of the two cost volumes is multiplied by the weight vector and added with the other" (p. 4).
- Eq. 2: L = alpha L_disp + (1 - alpha) L_bdry; Eq. 4: first-order-gradient boundary loss (p. 5).
- Tab. 1 ablation rows (p. 6): 2.12 -> 1.76 (semantic) -> 1.21 (pyramid) -> 1.04 (both) -> 0.98 (full), Scene Flow EPE.
