<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SegStereo card

Read: whole PDF (16 pp, arXiv v1: 14 pp body + references). The PDF does NOT include the supplementary material the paper refers to ("more setting details in supplementary material", p. 6, 9; ResNetCorr definition, KITTI-12 benchmark, segmentation results are there). Facts that live only in the supplement are therefore "not stated". Page refs are printed page numbers.

### 0. Meta
- Title: SegStereo: Exploiting Semantic Information for Disparity Estimation
- Authors: Guorun Yang, Hengshuang Zhao (equal), Jianping Shi, Zhidong Deng, Jiaya Jia
- Venue: ECCV 2018 (arXiv 1807.11699v1)
- PDF: paper/reference_papers/semantic_stereo/SegStereo_Yang_ECCV2018.pdf
- Code URL: not stated in the PDF (implemented in a customized Caffe, p. 9).
- Domain: driving (Cityscapes, KITTI) + synthetic (FlyingThings3D).
- Datasets: Cityscapes (5,000 fine-annotated: 2,975/500/1,525 train/val/test; 19,997 "extra" stereo images with SGM pseudo-disparity), KITTI 2015 (200 train, semantic labels from [1], 200 test), KITTI 2012 (194/195), FlyingThings3D (22,390 train / 4,370 test) (p. 9).

### 1. Problem & failure modes targeted
- Local ambiguity in flat / featureless regions, e.g. road centre and large vehicle interiors where "the matching clues ... are not enough to guide the model to seek correct direction for convergence" (p. 2, Fig. 1). Applies to both unsupervised (photometric) and supervised training.
- Human analogy: semantic consistency as an extra cue; large objects are easy to classify (p. 2).
- Also unsupervised learning without GT disparity (CityScapes, KITTI), a context where semantics acts as extra regularisation.

### 2. Pipeline by stage
- 2a Feature extraction: PSPNet-50 / ResNet-50 shallow part "conv1_1 to conv3_1" gives left/right features F^l, F^r at 1/8 input size (Sec. 4.1, p. 8). Shared (siamese). FROZEN ("weights in the shallow part and segmentation network are fixed when training SegStereo", p. 8). In the "corr13" variant the shallow part ends at conv1_3 (higher-res features, max displacement/padding 96, p. 12-13).
- 2b Semantic branch: PSPNet-50 [44], shares the shallow part; semantic features F_s^l, F_s^r = output of "conv5_4" (1/8 size, channel count not stated in main text); "classes": Cityscapes label set (class count not stated in this text). Pretrained on Cityscapes fine labels and FROZEN (p. 8, 12). A conv classifier ("Conv. Classifier in Segment Network", Fig. 2) on the warped feature produces the semantic prediction.
- 2c Cost volume: 1D correlation (DispNetC-style) between F^l and F^r along the epipolar line, max displacement 24 with padding 24 -> 25 channels F_c at 1/8 resolution (p. 8). With 1/8 stride this covers up to 24 x 8 = 192 px full res (my inference; stated only as "24" in feature units). corr13 variant: max displacement 96 at higher-resolution features. No explicit disparity-indexed 4D volume; 2D correlation channels.
- 2d Aggregation: not a cost-volume regulariser; a 2D encoder-decoder: hybrid feature F_h -> encoder of 12 residual blocks (some convs replaced by dilated convs, "dilation patterns [44]") -> decoder of 3 deconvolution blocks + 1 conv regression layer (p. 8). Exact channel widths: not stated in main text.
- 2e Disparity computation: direct regression of full-size disparity D (not soft-argmin). L1 regression loss for supervised, photometric for unsupervised.
- 2f Refinement: n/a (single-shot encoder-decoder).
- 2g Upsampling: learned deconv decoder to full size.
- 2h FUSION POINTS (two):
  1. **Semantic feature embedding** (Sec. 3.2, p. 6): stage = feature level, entering the aggregation network. Operator = concat. Direction = sem -> disp. Resolution 1/8. F_h = concat[F_t^l (1x1 conv, 256 ch, of left feature), F_c (25-ch correlation), F_s^l (left semantic feature, conv5_4)]; "All of F_c, F_t^l, F_s^l have the same spatial size" (p. 8).
  2. **Semantic loss regularisation / softmax loss** (Sec. 3.3, p. 6-7): stage = loss-only but with a differentiable bridge. Operator = warping + classification loss. Direction = disp <- sem labels: the right semantic feature F_s^r is upsampled to full size, warped with the predicted disparity D to left view, downsampled back to 1/8 -> F~_s^l, passed through a conv classifier, cross-entropy against the LEFT GT labels. The gradient flows back through the warp into the disparity network (the semantic network is frozen). It penalises disparities that bring semantically mismatched pixels together.
  - No sem -> disp refinement, no cost-volume gating.

### 3. Block -> problem -> evidence table
Metrics EPE (px) and D1 error (%) on KITTI-15 (Noc / All), see tables.

| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| Semantic feature embedding (F_s^l concat) vs ResNetCorr (no semantics), UNSUPERVISED, photometric loss only | local ambiguity | EPE Noc 2.46 -> 1.98, All 3.36 -> 2.72; D1 Noc 12.78 -> 10.76, All 14.08 -> 12.08 (Tab. 1) | Cityscapes pretrain, KITTI-15 eval, 200 train imgs | not reported |
| same, photometric + smooth loss | | EPE Noc 2.13 -> 1.87, All 2.43 -> 2.17; D1 Noc 11.05 -> 9.39, All 12.16 -> 10.53 (Tab. 1) | | |
| Softmax loss regularisation (unsup.) | semantic consistency under warp | EPE Noc 1.87 -> 1.61, All 2.17 -> 1.89; D1 Noc 9.39 -> 8.95, All 10.53 -> 10.03 (Tab. 1) | Cityscapes pretrain | |
| Smoothness loss (unsup.) | local incoherence | SegStereo photometric 2.72/12.08 -> photo+smooth 2.17/10.53 (All) (Tab. 1) | | |
| Fine-tune on KITTI-15, unsup. (all losses) | domain adaptation | EPE Noc 1.61 -> 1.46, All 1.89 -> 1.84; D1 Noc 8.95 -> 7.70, All 10.03 -> 8.79 (Tab. 1) | | |
| Embedding, SUPERVISED, pretrain Cityscapes (SGM labels) | | Test EPE Noc/All 1.43/1.46 -> 1.39/1.41; D1 7.33/7.64 -> 7.01/7.34 (Tab. 2, part 1) | 90K iters | |
| Embedding, SUP., pretrain Cityscapes extra + FlyingThings3D, no softmax loss (no labels) | | Test EPE 1.19/1.21 -> 1.15/1.17; D1 5.46/5.64 -> 5.20/5.38 (Tab. 2, part 2) | 500K iters; seg net fixed | |
| Embedding, SUP., fine-tune KITTI-12+15 (40-image val; Tab. 2 labels the columns "Test") | overfitting | Train EPE equal (0.40/0.41); val EPE Noc/All 0.73/0.76 -> 0.73/0.75; D1 2.13/2.40 -> 2.11/2.30 (Tab. 2, part 3) | 160 train / 40 val | |
| corr13 (higher-res features, max disp 96) | fine-scale matching | val EPE 0.66/0.70, D1 1.96/2.25 vs SegStereo 0.73/0.75, 2.11/2.30 (Tab. 2) | pretrained on extra+FT3D | more correlation channels (97 if 1D, my inference, not stated) |
| FlyingThings3D | cross-dataset | EPE 3.50 -> 1.45, D1 8.45 -> 3.50 vs ResNetCorr (Tab. 4); iResNet is better on EPE 1.27 | models of Tab. 2 part 2, no extra FT3D finetune | |
| Benchmark KITTI-15 test | | D1-all Noc 2.14 (PSMNet) vs 2.08; All 2.32 vs 2.25; fg All 4.62 vs 4.07; runtime 0.6 s (Tab. 3) | | 0.6 s, hardware not stated |
| Equal-capacity non-semantic control (a second PSPNet without semantic supervision) | causal check | not ablated | | |
| Frozen vs finetuned semantic network | | not ablated | | |
| Softmax loss in supervised setting | | not ablated (only implied by weights 1,1,0.1) | | |
| Semantic feature layer choice (conv5_4 vs other) | | not ablated | | |

### 4. Interactions & dependencies
- Softmax loss needs semantic LABELS for every training image: Cityscapes train and KITTI-15 only; for the Cityscapes "extra" set and FlyingThings3D the term is switched off (weight 0, p. 10, 12).
- The loss only changes the disparity net because the segmentation net is frozen; its effect is on warping-consistent matches ("big objects such as road and car", Fig. 3, p. 10).
- Semantic embedding is concatenated at 1/8 only; the gain shrinks as data grows: Tab. 2 part 1 -> part 3 differences go from 0.04 EPE to ~0 on EPE (D1 gains persist).
- ResNetCorr baseline lacks F_s^l and the softmax loss; the PSPNet-50 conv5_4 feature carries Cityscapes supervision, a pretrained network twice the depth of the shallow part - an unmatched-capacity comparison (see 8).
- In the unsupervised setting the semantic loss weight must be small relative to photometric/smoothness terms (weights below); an ambiguity in the paper prevents knowing which one is 10.

### 5. Losses (exact)
- L_p (Eq. 1) = (1/N) sum delta^p_{ij} * || I~^l_{ij} - I^l_{ij} ||_1 (photometric), I~^l = right image warped with D; delta^p = 0 where the photometric difference exceeds epsilon (= 10) or at borders/occlusions, else 1.
- L_s (Eq. 2) = (1/N) sum [rho_s(D_{i,j} - D_{i+1,j}) + rho_s(D_{i,j} - D_{i,j+1})] with rho_s the generalised Charbonnier penalty (alpha, beta, epsilon = 0.21, 5.0, 0.001, from [23]).
- L_r (Eq. 4) = (1/N_V) sum_{(i,j) in V} || D_{ij} - D^_{ij} ||_1 (supervised L1 over valid GT pixels).
- L_seg (Sec. 3.3): cross-entropy (softmax loss) between the classified warped right-semantic map F~_s^l and the left GT label map. No formula number is given.
- Unsupervised: L_unsup = lambda_p L_p + lambda_s L_s + lambda_seg L_seg (Eq. 3). Supervised: L_sup = lambda_r L_r + lambda_s L_s + lambda_seg L_seg (Eq. 5; note the existing summary drops the L_s term).
- Weights (p. 10): unsupervised "1.0, 10.0, 0.1 for photometric, softmax and smoothness". Supervised "1.0, 1.0, 0.1 for regression, softmax and smoothness". The symbolic order in Eq. 3 and 5 is (p, s, seg) = (photometric, SMOOTHNESS, seg), so reading the numbers in the order given in the text conflicts with the equation order: either smoothness = 10 and softmax = 0.1, or softmax = 10 and smoothness = 0.1 (unsup.). Unresolved from the PDF alone; flagging rather than choosing. The text also says the softmax weight is set to 0 when no semantic labels exist.
- Deep supervision: none (single output).

### 6. Training recipe (p. 9-13)
- Caffe; "poly" LR: lr = base * (1 - iter/max_iter)^power, base 0.01, power 0.9; momentum 0.9; weight decay 0.0001.
- Augmentation: random resize (factor 0.5-2.0), colour shift (max 10 per RGB), brightness shift (max 5), contrast multiplier 0.8-1.2; crop 513 x 513; batch 16.
- Unsupervised: pretrain on Cityscapes (iterations not stated), fine-tune on KITTI-15 200 imgs: max iteration 500, batch 16 (= 40 epochs).
- Supervised: (1) Cityscapes (SGM disparity labels), 90K iterations; (2) Cityscapes extra + FlyingThings3D, 500K iterations, segmentation net fixed (pretrained on Cityscapes train), softmax loss off; (3) fine-tune KITTI-12/15, 90K iterations, base LR 0.01, 160 train / 40 val images from KITTI-15.
- Frozen: shallow part + segmentation network throughout. Trainable: correlation-side 1x1 conv, 12-block encoder, decoder, regression layer, warp classifier (whether the warp-classifier conv in Fig. 2, labelled "in Segment Network", is trained or fixed: not stated).
- Hardware: not stated.

### 7. Results
- Unsupervised KITTI-15 (Tab. 1): final fine-tuned EPE 1.46 Noc / 1.84 All, D1 7.70 / 8.79 vs Zhou D1 8.61 / 9.91 and Godard 9.19 (All).
- Supervised (Tab. 2): see table above. Best: corr13 val EPE 0.66/0.70, D1 1.96/2.25.
- KITTI-15 test (Tab. 3, p. 13): D1-bg/fg/all Noc 1.76 / 3.70 / 2.08; All 1.88 / 4.07 / 2.25; runtime 0.6 s (hardware not stated). PSMNet 1.71/4.31/2.14 Noc, 1.86/4.62/2.32 All, 0.41 s. SegStereo was best D1-all at submission among the listed methods.
- FlyingThings3D (Tab. 4): EPE 1.45, D1 3.50 (iResNet 1.27 / 4.90; GC-Net 1.84 / 9.67; CRL 1.67 / 6.70).
- Params / FLOPs / mIoU: not stated.

### 8. Negative results & limitations
- Authors: none directly. They report that gains are smaller with more data (Tab. 2) and that the softmax-loss gain "mainly arises on big objects, such as road and car" (p. 10).
- Weaknesses:
  - The baseline ResNetCorr has no semantic features AND no extra pretrained network; the semantic embedding adds a 2nd network's frozen features from conv5_4 trained on Cityscapes labels; there is no equal-capacity control, and conv5_4 features are strong generic features, not only "semantics" (same class of weakness our controls were designed to remove).
  - The largest gains (EPE -20 %) are in the weaker unsupervised photometric setting, where any additional pretrained feature would likely help; in supervised KITTI fine-tuning the validation gain is 0.00-0.03 px EPE (Tab. 2 part 3, 40 images, single run).
  - The unsupervised-Cityscapes pretrain and KITTI-15 eval share the same label taxonomy; segmentation network was trained on Cityscapes which is not the stereo test domain.
  - Tab. 2's "Test" header for part 3 actually refers to a 40-image validation split (text p. 12).
  - Loss-weight ambiguity (Sec. 5) and a missing supplement.
  - D1 definition typo: "percentage of errors below a threshold" (p. 9); read as above-threshold.
  - Runtime 0.6 s, with no hardware named.
- The existing summary paper/reference_papers/summaries/semantic_stereo/SegStereo.md: (1) says semantic branch is "trained jointly" - it is frozen (p. 8, 12); (2) writes L_sup = L_reg + lambda_sem L_softmax, omitting the smoothness term in Eq. 5; (3) is otherwise consistent about reported metrics (no mIoU, params, FLOPs).

### 9. Relevance to OUR model
- Closest to our setting in ONE respect: SegStereo also uses a FROZEN segmentation net sharing early layers with the stereo net (PSPNet-50 conv1_1-conv3_1 shared, "fixed" throughout). So the frozen shared-trunk design is not new; what is new in ours is the semantic gate/residual on a frozen stereo predictor and the equal-capacity control.
- Portable idea, high value: the **warped semantic consistency loss** (Sec. 3.3). With the frozen YOLO trunk + frozen 14-class decoder we can warp the right-view class logits/probabilities with the head's output disparity and apply cross-entropy against the left VKITTI2 label (or the left prediction as a pseudo-label). Insertion: training loss on the fusion head's disparity (differentiable through grid_sample), zero inference cost. Expected benefit: penalises matches across semantic classes on flat regions and large objects, which is where our semantic arms already help; it directly supervises disparity from the sem decoder without changing capacity. It can also be run on KITTI (no GT disparity) with predicted semantics from the frozen decoder -> a self-supervised adaptation signal, relevant to our domain-gap results (F-series). Risks: warping at 1/8 requires upsampling disparity; interpolating probabilities near boundaries creates mixed classes; gradient is only informative for pixels where the class changes with disparity. Occluded pixels must be masked as in L_p.
- Not portable / already done: the concat embedding (F_t^l + F_c + F_s^l -> encoder-decoder) is a feature-level fusion we did not choose; our gate/residual works on the cost-volume and disparity. The B/C-series fused-baseline arms are closer to this idea (semantic features into a depth head) and our C4/C5 controls (zeroed / misaligned semantics) address the SegStereo capacity confound that this paper lacks.
- Cost: loss-only, so 0 ms at inference; ~14 more channels of warp and CE at train time. Risk of conflict with the existing boundary behaviour: none known; check that semantic warp loss doesn't regress EPE like G/H edge terms did.
- Honest caution: SegStereo's supervised-regime gains are ~0 px EPE on a 40-image val; the unsup. gains are the real signal and come from a regime we do not use.

### 10. Key quotes/equations worth citing
- "The weights in the shallow part and segmentation network are fixed when training SegStereo." (p. 8)
- F_h = concat[F_t^l, F_c, F_s^l]; "Both max displacement and padding size are set to 24 so that the channel number of correlated features F_c is 25." (p. 8)
- "We first upsample F_s^r to the full size. We afterwards downsample warped feature map to 1/8 size" (p. 8).
- Eq. 3, Eq. 5: L_unsup = lambda_p L_p + lambda_s L_s + lambda_seg L_seg; L_sup = lambda_r L_r + lambda_s L_s + lambda_seg L_seg (p. 7-8).
- Tab. 1: softmax loss 2.17 -> 1.89 All-px EPE (p. 10).
