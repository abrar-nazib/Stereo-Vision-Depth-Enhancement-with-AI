<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SSPGNet block card

Evidence grade: **B** (component ablation of FTM and SPM, backbone ablation and zero-shot comparison on held-out KITTI/Middlebury/ETH3D, single run, no capacity control; headline "peak" numbers are best-epoch on the target sets). Not a semantic-segmentation method: the "prompt" is a self-estimated geometric sparse disparity; "semantic awareness" is only frozen VFM features.

### 0. Meta
- Title: Sparse Self-Prompt-Guided Stereo Matching for Real-World Generalization. Hangbiao Li, Haojun Mo, Xing Li, Tao Fang, Sikun Liu, Shuzhen Yu, Zhibo Rao (Nanchang Hangkong Univ. et al.). Sensors 2026, 26, 3173 (published 17 May 2026). 27 PDF pages, read fully. Page refs "p. N of 27".
- PDF: paper/reference_papers/semantic_stereo/SSPGNet_Li_Sensors2026.pdf. Code + ZED 2 in-the-wild set: https://github.com/Archaic-Atom/SSPSNet (p.17).
- Domain: general (outdoor + indoor, driving included via KITTI; zero-shot). Datasets: train SceneFlow or CreStereo only; test KITTI 2012/2015, Middlebury, ETH3D, plus ZED 2 in-the-wild set and Flickr1024 qualitative (p.9).
- No existing summary under paper/reference_papers/summaries/ for this paper.

### 1. Problem & failure modes targeted
- Zero-shot / real-world generalization: benchmark-tuned models fail in the wild (p.1). VFM tokens are spatially coarse (patch 14) and lose pixel coordinate information needed for correspondence (p.2, p.6-7).
- Occlusion, repetitive texture, thin structures, boundaries, illumination (p.11-12).
- Admitted failures: disparity range limit (about 197), nonzero noisy output for two identical images (Fig. 13), transparent/reflective surfaces, large repetitive-texture regions (App. A.4, p.23).

### 2. Pipeline by stage
- 2a Feature extraction: frozen DINOv2 or Depth Anything V2 ViT-L/14 (final model DAv2); tokens T in R^{C x H/14 x W/14} from layers {4, 11, 17, 23} of the 24-layer ViT-L (p.6). 304.37M frozen parameters (p.15). Siamese, shared weights. Feature transform module (FTM): per-layer 1x1 conv for channel alignment then FeatUp (image-guided feature upsampler) to F in R^{C x 4H/14 x 4W/14} (about 0.29 of input resolution, i.e. between 1/3 and 1/4); concat of four layers + 1x1 conv (Eq. 1-2, p.6). Channel count C: not stated.
- 2b Semantic / prior branch: none separate. The VFM features are the only "semantic" content; no segmentation head, no class labels.
- 2c Cost volume: concatenation volume C(d,h,w) = [F_L(h,w), F_R(h,w-d)] over d = 1..4D/14, D = 196 (so about 56 levels) at 4H/14 x 4W/14 (Eq. 3, p.7, p.17).
- 2d Aggregation: 3D hourglass (PSMNet-style) producing three stage outputs i = 1,2,3; softmax over disparity (Eq. 4-6, p.7).
- 2e Disparity computation: soft-argmin per stage. Sparse prompt: threshold the final stage distribution P3 with confidence t and keep d3 where the indicator is 1 (d_s = f(P3,t) * d3, Eq. 8, p.7). What f(P3,t) tests (max probability? mass near the mode?) is not stated; the sparsity (fraction retained) is never reported in numbers. t in {0.08, 0.10, 0.12, 0.15}; train t = 0.10, test t = 0.15 (p.18).
- 2f Refinement (SPM, sparse prompt module, Eq. 9-12, p.8): warp right features by d_s hat, NMRF (neural Markov random field, cross-shift-window attention) on [F_L, warped F_R] gives latent z; one-layer MLP gives affinity A in R^{8 x H x W} (eight neighbors of a 3x3 CSPN-style propagation, p.18); sparse disparity embedded to H in R^{8 x H x W}; spatial propagation network (SPN) update d_r = SPN(H, A). Single step (not iterative GRU) (Tab. 1, p.4). Resolution of this module is written as H x W; the actual working resolution is not stated explicitly (likely the 4H/14 feature grid, then 3 output upsampling; not stated).
- 2g Upsampling to full res: not described beyond FeatUp on features; final output same spatial size as input (Tab. 2 caption, p.10). Method for the last step: not stated.
- 2h FUSION POINTS: (1) VFM features -> feature stage (replaces CNN encoder, frozen), direction VFM -> stereo, at 4H/14. (2) Sparse disparity prompt -> refinement stage via embedding into the SPN hidden state; direction stereo -> stereo (self-prompt), confidence gated by P3. (3) Left features + warped right features -> affinity (cross-attention) -> propagation weights. Disp -> sem: none. No segmentation involved.

### 3. Block -> problem -> evidence table
Tab. 2 columns: SF = SceneFlow bad-3 (in-domain), KT-12 = KITTI 2012 bad-3 (zero-shot). Training set for Tab. 2 is not stated in the caption; presumably SceneFlow (Tab. 4 header uses SF and KT-12 alike); runtime on RTX 3090 at 378x1246, mean of 10 passes after 3 warm-ups (p.10).
| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| FTM (feature transform / upsampled tokens vs. decoding small cost volume from tokens) | token resolution loses coordinates | FTM only vs neither: SF 5.30 -> 4.00, KT-12 5.32 -> 4.07 (Tab. 2, p.10) | frozen VFM, single run | +0.01 s (0.54 -> 0.55) |
| SPM (sparse prompt module) | dense propagation from anchors | SPM only vs neither: SF 5.30 -> 3.98, KT-12 5.32 -> 4.06 (Tab. 2) | same | +0.14 s (0.54 -> 0.68) |
| FTM + SPM | complementary | SF 3.20, KT-12 3.62 (Tab. 2): additional -0.80 SF, -0.45 KT-12 vs FTM only | same | 0.70 s |
| Backbone: CNN vs DINOv2 vs DAv2, S vs L (Tab. 4, p.11) | feature quality | SF/KT-12: CNN 3.90/4.98 (size column blank); DINOv2-S 3.65/4.84, -L 3.22/4.54; DAv2-S 3.61/3.79, -L 3.20/3.62 | same | L is 304M frozen |
| Confidence threshold t (sparse-map quality) | prompt reliability | sparse bad-3 falls with t: SF 2.1 -> 1.5, KT-12 4.6 -> 1.6, KT-15 4.0 -> 1.8 for t 0.08 -> 0.15 (Tab. 3, p.11); MB bad-2 8.7 -> 7.8, ET bad-1 2.7 -> 1.7 | sparsity not reported | - |
| Pretraining dataset: SF vs CreStereo | domain gap | KT-12 4.6 -> 3.6, KT-15 4.7 -> 4.2, MB 7.9 -> 7.6, ET 2.5 -> 2.1 (Tab. A1, p.18) | confound: CreStereo chosen for headline table | - |
| Sparse prompt itself vs. a dense/no-prompt SPN | prompt value | not ablated: Tab. 2 SPM row bundles NMRF attention + sparse prompt + SPN together | - | - |
| NMRF cross-shift-window attention vs simpler affinity | affinity quality | not ablated | - | - |
| Mixture-of-Laplacians CE loss | distribution calibration for prompt | not ablated | - | - |
| Test-time t = 0.15 vs training t = 0.10 | - | not ablated | - | - |
| Capacity control for SPM | extra params | not stated; SPM parameter count absent, only total 313.38M / 9.01M trainable (p.15) | - | - |

### 4. Interactions & dependencies
- FTM is required for VFM tokens: the authors' claim is tokens alone do not carry pixel coordinates (p.6-7, p.10); SPM gives a smaller gain without FTM (3.98 vs 3.20 combined).
- The prompt is empty in repetitive/flat regions because probability is flat (App. A.4, p.23), so propagation has no anchors there and extrapolates noise: the method depends on a confident cost volume.
- Higher t gives more accurate but (by the Tab. 3 trend) presumably sparser prompts; the text says t = 0.15 is "densest reliable", which conflicts with this reading (the direction of the confidence criterion is unstated).
- Trained only with SF or CreStereo; cross-domain numbers depend on training set (Tab. A1).

### 5. Losses
- Smooth-L1 on all four disparity outputs (three hourglass stages and SPM output), summed with equal weight (Eq. 13, p.8).
- Cross-entropy of each hourglass stage's disparity distribution against a ground-truth multi-modal mixture-of-Laplacians P_gt (Eq. 7, 14; K and parameters "default setting of [46]"), stages i = 1..3, normalized by pixel count; L_total = L_s + L_ce (Eq. 15). SPM output is not distribution-supervised.

### 6. Training recipe (App. A.1, p.17-18)
- Adam (0.9, 0.999); LR 1e-3 for 50 epochs then 1e-4 for 10 epochs (60 epochs); batch 3 per GPU on 6 GPUs (18); random crop to 518 x 266 only; D = 196; VFM frozen throughout; trained only on SceneFlow or CreStereo; GPU type for training not stated. Seeds: single.

### 7. Results
- Peak zero-shot bad-pixel (%) (Tab. 5, p.14), training on CreStereo: SSPGNet KT-12 3.6 (>3 px), KT-15 4.4, MB 7.6 (>2 px), ET 2.1 (>1 px). Next best per column: KT-12 DEFOM-Stereo 3.7 / HVT-RAFT 3.7; KT-15 SMFormer 4.7; MB NMRF 7.5 (SSPGNet second); ET SMFormer 2.9 / DEFOM 2.3.
- Volatility across training epochs (Tab. 6, p.15): SSPGNet 3.65 +/- 0.01, 4.45 +/- 0.03, 7.66 +/- 0.05, 2.29 +/- 0.20; Mask-CFNet 5.03 +/- 0.03 / 6.08 +/- 0.07 / 12.82 +/- 0.37 / 6.63 +/- 0.21 on KT-12 / KT-15 / MB / ET.
- Cost (p.15): 313.38M params (9.01M trainable), 2.92 TFLOPs, 0.609 s per pair on RTX 3090 at 378x1246; Tab. 2 gives 0.70 s for the same config (unreconciled). Not real-time.
- In-the-wild ZED 2: qualitative only, vs NMRF (Fig. 8/9).

### 8. Negative results & limitations
- Authors: not real-time; disparity range <= ~197; identical-image test yields noise (Fig. 13f), suggesting incomplete grasp of matching; transparent/reflective surfaces and repetitive texture fail (App. A.4).
- Mine: (i) "Peak" Tab. 5 numbers are best-epoch picked against the target benchmarks, which the authors acknowledge via Tab. 6; (ii) Tab. 6 std is over epochs of one run, not seeds, and the compared methods' rows come from other papers; the +/-0.01 std is striking; (iii) SPM ablation bundles attention affinity, sparse prompt and SPN, so the value of the "self-prompt" idea itself is untested; (iv) no param/capacity matching of SPM; (v) t chosen using the same target sets (Tab. 3 reports target-domain metrics) and train/test t differ; (vi) sparsity of the prompt never quantified; (vii) comparison mixes training sets/backbones (ViT-L 304M vs CNN baselines); (viii) in-the-wild evaluation is qualitative.

### 9. Relevance to OUR model
- Frozen shared backbone is the same philosophy as our frozen YOLO26m layers 0-6: SSPGNet's Tab. 4 supports that a frozen large pretrained encoder plus a small trained stereo head (9.01M trainable here) generalizes zero-shot, consistent with our F-series (E3 beating E4/A09 on KITTI 2015). Different: ours is a CNN (no token-resolution problem; FTM-style fix is unneeded) and about 100x smaller.
- Portable idea (speculative): confidence-gated sparse anchors + affinity-based propagation could replace or augment ClassResidual: take A09's top-1 candidate probability, keep high-confidence pixels as anchors, and propagate with class-boundary-aware affinities. Insert after A09 candidates / before E3's gate; cost: one small 3x3 propagation (cheap) but the NMRF cross-attention is not (they add 0.14 s on a 3090); risk: our tile/plane refinement already propagates, and G1/G2/G3/H1/H2 refinement-style add-ons did not help EPE, so expected benefit is low. Novel combination (class-aware affinity from a frozen decoder driving SPN) vs. already done: not found in this paper.
- Nothing here changes the semantic-fusion conclusion: it contains no segmentation or semantic ablation, so it neither supports nor weakens D2/E3.
- Remote-sensing caveat: not applicable (general domain); but 313M params / 0.6 s per pair on a 3090 rules it out for RTX 3050 4GB live use.

### 10. Key quotes/equations worth citing
- "tokens from vision foundation models alone do not preserve sufficient pixel-level coordinate information for stereo matching" (Conclusion, p.17).
- Eq. 8-12 (p.7-8): sparse prompt and SPN propagation; Tab. 2 (p.10): FTM/SPM ablation; Tab. 6 (p.15): epoch volatility.
- PDF: paper/reference_papers/semantic_stereo/SSPGNet_Li_Sensors2026.pdf
