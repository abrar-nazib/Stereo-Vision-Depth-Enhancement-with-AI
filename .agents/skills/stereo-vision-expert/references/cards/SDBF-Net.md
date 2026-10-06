<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SDBF-Net card

### 0. Meta
- Title: SDBF-Net: Semantic and Disparity Bidirectional Fusion Network for 3D Semantic Detection on Incidental Satellite Images. Rao (first), He, Zhu, Dai, He. APSIPA ASC 2019 (Lanzhou), pp. 438-444.
- PDF: paper/reference_papers/semantic_stereo/SDBF-Net_Rao_APSIPA2019.pdf (7 pp = journal pp. 438-444; refs on the last page). No supplement. Code: not stated.
- Domain: remote sensing (incidental satellite stereo). Dataset: US3D (Bosch et al. WACV 2019): 100 km2 over Jacksonville FL and Omaha NE; 4,292 epipolar-rectified training pairs 1024x1024 with semantic + disparity labels; 50 test pairs without GT (scored via CodaLab server). Classes (C=6): ground, high vegetation/trees, building roof, elevated road/bridge, water, unlabeled (Sec. III-A).

### 1. Problem & failure modes targeted
- Seasonal appearance differences and lighting/perspective changes between incidental satellite pairs make stereo matching hard; semantic info should help (Sec. I).
- Treating seg and stereo as isolated tasks; want mutual promotion via a fusion network (Sec. I).
- Fig. 1/6: stereo around bridges/overpasses (flat elevated structures), building boundaries.
- Not targeted: latency/embedded (runtime reported but unoptimised, 1.08 s), domain shift.

### 2. Pipeline by stage
Three-stage design; three SEPARATE networks (not shared encoder).
- 2a Feature extraction: two independent trunks.
  - Semantic (SSM, left image only): ResNet-101 (dilated), output (H/8)x(W/8)x1024 (Sec. II-A).
  - Stereo (SMM, shared L/R): four 32-ch 2D convs (stride 1 except the first layer, stride 2), three 32-ch residual blocks, one 32-ch conv stride 2, fifteen 64-ch residual blocks, then SPP (hierarchical context), concat with earlier features, one 128-ch conv + one 32-ch conv (last without BN/ReLU); output unary features (H/4)x(W/4)x32 (Sec. II-B(1), Fig. 4). Trained, with L/R weight sharing.
- 2b Semantic branch: ASPP (3x3, dilation 3/6/12/18) on ResNet-101 features, concat with deep features, 1024-ch + 512-ch conv, then 256-ch, 128-ch, C-ch 2D deconvolutions (stride 2 each) -> score map HxWxC; argmax -> initial segmentation (Sec. II-A, Fig. 3). Trained from scratch on US3D? (ResNet init not stated).
- 2c Cost volume: concatenation (left unary, right unary shifted by d): V(d,u,v,f) = stack{ f_L(u,v) || f_R(u-d,v) }, size (D/4)x(H/4)x(W/4)x64 (Eq. 1). D=128 (Sec. III-A; whether at full or 1/4 res: ambiguous, "D/4" suggests D is the full-res range).
- 2d Aggregation: 3D U-Net-like multi-scale 3D CNN, four levels: 3D convs (32, 64, 96, 128 channels, stride 2) encode; 3D deconvs (96, 64, 32, stride 2) decode; skip (residual) links at equal levels; then a 16-ch and a 1-ch 3D deconv (stride 2 each) back to D x H x W (Sec. II-B(3)).
- 2e Disparity: softmax over d then weighted sum (soft regression): d^_init = sum_d d * P(d), P = softmax(...). The paper prints softmax(d) (Eq. 2) which is notation sloppy: it should be softmax of the (negative) cost volume V; flagged. (Local summary writes softmax(-V_d); that is its guess, not the paper's text.)
- 2f Refinement: the fusion module (below), full resolution HxW 2D convs.
- 2g Upsampling: deconvs inside SMM (cost volume brought to full D x H x W); no separate upsampler; 2D FM runs at full res.
- 2h FUSION POINTS (only one, bidirectional, late):
  - Fusion module (FM) after both nets finish. Seg fusion: concat [initial seg map (C=6 channels) + left RGB image (3) + initial disparity (1)] = 10 channels -> 32-ch 2D conv -> three 32-ch residual blocks -> 6-ch 2D conv; output ADDED to initial seg map (residual; last layer has no BN/ReLU so negative residuals can be learned) (Sec. II-C(1), Fig. 5).
  - Disparity fusion: "very similar" network, last layer 1-ch conv; input is the initial disparity with the seg map (and left image per Fig. 5), residual to initial disparity. Exact input channel count for the disparity FM: not stated explicitly. 
  - Direction: bidirectional (seg -> disp and disp -> seg), operator concat + conv residual, at full resolution, on OUTPUT maps (not features/cost volume). Right image is not used by the FM (the local summary's FM(S, d, I^L, I^R) is wrong).

### 3. Block -> problem -> evidence table
US3D test (CodaLab-scored; ablations stated "on the US3D test set", Sec. III-B), time = seconds per image (hardware: 1080Ti presumably; not stated per-table).
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| ResNet-50 backbone, no ASPP | seg baseline | mIoU 0.646 (Tab. I) | | 0.151 s |
| ResNet-50 + ASPP | context | 0.723 (+0.077) | | 0.156 s (+0.005) |
| ResNet-101, no ASPP | deeper | 0.739 (+0.093 vs R50 noASPP) | | 0.234 s |
| ResNet-101 + ASPP | | 0.759 | | 0.239 s |
| + Fusion (seg) | seg refinement using disparity | 0.759 -> 0.767 (+0.008 mIoU) | | 0.245 s (+0.006) |
| SMM no SPP no 3D CNN | stereo (2D-only) | D1 40.32 %, EPE 8.77 (Tab. II) | | 0.143 s |
| SMM + SPP only | | 40.40 %, 8.63 | | 0.147 s |
| SMM + 3D CNN only | cost aggregation | 10.77 %, 1.55 | | 0.696 s |
| SMM SPP + 3D CNN | | 10.55 %, 1.50 | | 0.701 s |
| + Fusion (disparity) | stereo refinement using semantics | 10.55 -> 8.02 % D1 (-2.53 pts), EPE 1.50 -> 1.31 (-0.19) | no no-semantics control | 0.713 s (+0.012) |
| Overall vs others (Tab. III) | | mIoU 0.767, D1 8.02, EPE 1.31, 1.08 s; MRFCNet mIoU 0.790, D1 9.06, EPE 1.39; ICNet+iResNet-i2 D1 33 %, EPE 3.05; CU-PSM mIoU 0.772; iResNet-i2 etc. | | |
| Semantic-only vs disparity-only fusion, residual form, concat of left image | | not ablated individually | | not ablated |
Remark: Tab. III runtime 1.08 s vs sum of Tab. I (0.245) + Tab. II (0.713) = 0.958 s; the gap is not explained.

### 4. Interactions & dependencies
- The disparity fusion is conditioned on the left RGB image as well as seg; without a control (RGB-only 2D residual refinement of equal capacity) the -2.53 pt D1 gain cannot be attributed to semantics. Fig. 6 row 3 shows the initial disparity being globally wrong (nearly uniform dark) and the fusion recovering structure, which looks like a 2D image-guided correction of a failed cost volume rather than semantic reasoning (my reading).
- Staged training: SSM 1000 epochs, SMM 200, FM 200 (Sec. III-A); whether SSM/SMM are frozen during the FM stage: not stated.
- 3D CNN is essential for stereo (D1 40 % -> 10.8 %); SPP alone does nothing for stereo (D1 40.32 -> 40.40, EPE 8.77 -> 8.63) but helps on top of the 3D CNN (D1 10.77 -> 10.55, EPE 1.55 -> 1.50).
- Seg gains from fusion are small (+0.8 mIoU), disparity gains large: asymmetric.

### 5. Losses
- Seg (Eq. 3): cross-entropy sum over pixels and classes of -P_i(i,p) log Q(i,p), applied to the initial seg (Loss_i) and the fused seg (Loss_r); Q = one-hot GT occupancy volume.
- Stereo (Eq. 4): L1 over valid labeled pixels: Loss_i = (1/N) sum ||d(p) - d^_i(p)||_1, same for refined Loss_r; N = number of valid labels.
- No weighting between initial and refined losses is stated; no class balancing stated; deep supervision: none.

### 6. Training recipe (Sec. III-A)
- TensorFlow, Adam (0.9, 0.999), lr 1e-3, batch 2, D=128, C=6, inputs scaled to [0,1], random 448x448 crops, 4x GTX 1080 Ti.
- Three sequential stages: SSM 1000 epochs, SMM 200 epochs, FM 200 epochs. Whether lr is decayed / what is frozen: not stated. Augmentation beyond random crop: not stated. Pretrained init of ResNet-101: not stated.
- Metrics: mIoU, EPE, D1 (3-px) and mIoU-3 (needs both correct class and disparity error < 3 px) (Eq. 5).

### 7. Results
- Tab. III (US3D test): SDBF-Net mIoU 0.767, D1 8.02 %, EPE 1.31 px, 1.08 s. Best baselines: MRFCNet mIoU 0.790, D1 9.06, EPE 1.39; CU-PSM mIoU 0.772. Others: ICNet+SGM 0.700 / 43 % / 10.34; DeepLabv3+SGM 0.750 / 43 / 10.34; ICNet+iResNet-i2 0.700 / 33 / 3.05 (listed with their own numbers; baselines taken from [32] Bosch et al., not re-run).
- Claims first rank on CodaLab at time of submission (mIoU-3 0.7561, Fig. 7, "before August 27, 2019"). Note SDBF's mIoU 0.767 is BELOW MRFCNet 0.790 and CU-PSM 0.772; the win is on the combined mIoU-3/D1.
- Params/FLOPs/memory: not stated. Runtime hardware: not stated (training 4x 1080Ti).
- Local summary errors: states "no runtime" (the paper reports 0.245/0.713/1.08 s); writes FM(S, d, I^L, I^R) (FM uses left image only); writes softmax(-V_d) (paper prints softmax(d)).

### 8. Negative results & limitations
- Authors: none discussed. No failure analysis.
- Mine: (1) no no-semantics fusion control; (2) ablations on the test set (via server) with no validation split stated, risk of test-tuning; (3) two separate big networks (ResNet-101 + 3D CNN), 1.08 s, no real-time relevance; (4) late output-level fusion at full-res 2D, so semantics cannot gate matching candidates; (5) baselines copied from the dataset paper, not re-run under same protocol; (6) seg improvement from fusion tiny (+0.8 mIoU) and the 0.767 mIoU is lower than two listed baselines; (7) 6 classes, one of them 'unlabeled'. Weird entries: ICNet+iResNet-i2 D1 33 % but EPE 3.05 listed.
- Satellite transferability caveats (inference unless a ref is given): (a) US3D disparity = height; scenes are nadir-ish with huge flat regions (water, ground, roofs) and a semantic-to-height correlation that is far stronger than in street scenes; (b) disparity range small and mostly one sign vs driving 0-192+; (c) paper's explicit hard case is seasonal/lighting appearance changes between the two acquisitions, which does not exist in a synchronised stereo rig; (d) data is 1024x1024 epipolar-resampled orthographic-like imagery, no perspective foreshortening, sky, or foreground/background occlusion layers; (e) a 2D post-hoc refinement can fix global offsets (Fig. 6 row 3), a symptom of satellite-specific mismatch; (f) evidence is one dataset, one benchmark server.

### 9. Relevance to OUR model
- Bidirectional output-level fusion: sem <- disp is the only novel element versus our design; our semantic decoder is frozen and shared, so a disparity-conditioned seg refinement cannot be trained without unfreezing; and +0.8 mIoU is little. Skip.
- Disparity refinement by concat [seg probs, RGB, disparity] -> residual conv blocks: our E3 ClassResidual does the equivalent with class-conditioned residuals; our G1 (RGB-guided residual) failed to help EPE, which matches the missing control here: SDBF's gain probably comes largely from the extra 2D refinement net, not semantics. Use as caution: any semantic-refinement claim needs a no-semantics equal-capacity control (our D/E/F design already has one).
- Portable ablation insight: 3D-conv aggregation vs 2D (D1 40 -> 10.8 %) says nothing about our shallow head.
- mIoU-3-style joint metric (class correct AND disparity error < 3 px) is a reusable evaluation idea for our manuscript's semantic-3D claim. Cheap to compute; not a model block.
- Risk of citing: satellite domain, test-set ablation, copied baselines. Cite as related work only.

### 10. Key quotes/equations worth citing
- Eq. 1 (p. 3): V(d,u,v,f) = stack{ f_L(u,v) || f_R(u-d,v) } (concat cost volume).
- Eq. 5 (p. 5): mIoU_t = (1/C) sum_c TP_t / (TP_t + FP_t + FN) (mIoU-3 requires correct label and disparity error < t=3).
- Tab. II (p. 6) disparity fusion: 10.55 -> 8.02 % D1, 1.50 -> 1.31 EPE.
- "the fusion module has a significant performance improvement for semantic segmentation task and stereo matching task in mIoU and EPE respectively" (p. 6).
