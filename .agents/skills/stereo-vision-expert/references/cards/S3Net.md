<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# S3Net card

### 0. Meta
- Title: S3Net: Innovating Stereo Matching and Semantic Segmentation with a Single-Branch Semantic Stereo Network in Satellite Epipolar Imagery. Yang (first), Chen, Tan, Wang, Wang, Zhang. IGARSS 2024 (arXiv 2401.01643v3). Wuhan University (LIESMARS).
- PDF: paper/reference_papers/semantic_stereo/S3Net_Yang_IGARSS2024.pdf (4 pp, last page refs). No supplement. Code: https://github.com/CVEO/S3Net (p. 1).
- Domain: remote sensing (satellite epipolar). Dataset: US3D, 4,292 stereo pairs 1024x1024; they use 3,500 512x512 crops for train, 338 val, 454 test (Sec. 3.1). Five classes in results: ground, tree, building, water, bridge (Tab. 3). The paper never states the class count in the text.
- The paper is only ~2.5 pages of method+results: many implementation facts are not stated (listed below).

### 1. Problem & failure modes targeted
- Tie-point/appearance differences, training instability and confusion in disparity from varying data distribution between the two satellite acquisitions (Sec. 1).
- Blurred object disparity edges (semantic features per pixel to sharpen) and foreground/background distinction aiding segmentation (Sec. 1).
- Existing semantic-stereo methods use parallel branches or one task's output to improve the other (e.g. S2Net) rather than one joint representation.

### 2. Pipeline by stage
- 2a Feature extraction: DCSFEM (Disparity-Classification Spatial Feature Extraction Module), weight-shared between left and right. Two feature processes: a disparity extractor with multi-scale and sequence processing, and a semantic extractor; "both processes undergo four times downsampling" (Sec. 2; ambiguous: 4 downsample steps or x4; Fig. 1 DCSFEM shows several downsample levels with SFM units, a multi-scale sum/concat). SFM used on multi-scale disparity features; semantic features concatenated ("for synergy"). Channel counts, backbone type, resolutions: not stated.
- 2b Semantic branch: part of DCSFEM (SM), no separate decoder; semantic info rides in the cost volume itself. Frozen? No (end-to-end).
- 2c Cost volume (Sec. 2.2): "selective" stacking of multi-scale left/right features; 4D volume of shape H x W x D x C where D = disparity layers and C = feature channels (paper order: height, width, number of disparities, number of feature maps). The TOPMOST disparity layer of the 4D cost volume is reserved for semantic information; the remaining layers carry disparity features from multi-scale features. So semantic info is placed as one extra slice along the disparity axis (cost volume contains 'semantic cost volume' SCV and 'disparity cost volume' DCV, per Tab. 1 caption). Disparity range D: not stated.
- 2d Aggregation: MFM (Mutual-Fuse Module) of 3D convolutions, three rounds. Round 1 takes cost1 (initial); 3D SFM; "disparity dimension isolation" (separates the semantic slice from disparity slices, per Fig. 1 orange layer); downsample the fused feature to generate cost2 and cost3, which re-enter via skip connections as input to rounds 2 and 3; upsample and concatenate back with the semantic layer as cost1 of next round; after 3 rounds with different weights the final cost1 goes to the heads. Fig. 1 shows Output1-3 (possibly multi-scale outputs / deep supervision, but not described in the text).
- 2e Disparity: trilinear upsampling of the cost volume -> disparity map; the text does not state soft-argmin (most likely, but "not stated"). Classification map from bilinear upsampling of the semantic slice.
- 2f Refinement: n/a (MFM is the only iterative-like block; no GRU).
- 2g Upsampling: bilinear (classification) / trilinear (disparity) to the original size (Sec. 1, Fig. 1).
- 2h FUSION POINTS:
  1. DCSFEM: feature level, semantic features concatenated with multi-scale disparity features (single stream, shared weights).
  2. Cost volume construction: semantic slice stacked into the same 4D volume (disparity axis).
  3. SFM (Sec. 2.3): gating operator inside features and 3D cost volumes: two parallel conv branches (different weights) multiplied element-wise per channel, applied twice in series (Fig. 1); acts as learned multiplicative gate on information flow (2D version in DCSFEM, 3D version in MFM).
  4. MFM: bidirectional through shared cost volume; the semantic slice is re-concatenated each round.
  Direction: bidirectional by construction (shared tensor), one stage after another; no separate refinement net; fully trainable end to end.

### 3. Block -> problem -> evidence table
US3D (own split: 454 test), batch size 4. Tab. 1:
| config | evidence | notes |
|---|---|---|
| Full model (SFM + DM + SM + DCV + SCV) | mIoU 67.39, mIoU-3 66.27, D1 9.579, EPE 1.403 | |
| w/o SFM (DM, SM, DCV, SCV on) | 64.13 / 62.72 / 10.443 / 1.483 => SFM contributes +3.26 mIoU, +3.55 mIoU-3, -0.864 D1, -0.080 EPE | SFM is the biggest ablated block; the hypothesised gating effect |
| SFM on, disparity-only (DM + DCV; SM, SCV off) | mIoU not applicable, D1 11.391, EPE 1.567 => removing the semantic parts costs +1.812 D1, +0.164 EPE vs the full model | measures benefit of joint training but also removes capacity |
| SFM on, semantic-only (SM + SCV; DM, DCV off) | mIoU 52.42 (-14.97 vs full) | |
| SFM, MFM structure, 3 rounds, bilinear/trilinear heads, SFM in 2D vs 3D | not ablated separately | |
| Selective cost volume construction | not ablated (vs concat/PSMNet or S2Net stacking) | |
Comparisons: Tab. 2: PSMNet D1 11.872/EPE 1.695; GwcNet 11.387/1.618; GANet 10.876/1.526; CFNet 11.024/1.570; S2Net 10.051/1.439; S3Net 9.579/1.403. Tab. 3 mIoU: SDFCNv2 58.60; SegFormer 60.21; PSPNet 60.51; HRNetV2 61.38; Ours 67.39 (ground 81.94, tree 66.39, building 73.45, water 79.23, bridge 35.96).
Cost (params/ms/FLOPs): not stated anywhere.

### 4. Interactions & dependencies
- Disparity quality depends on the semantic slice: the disparity-only variant is worse (11.391 vs 9.579) but is not a fair capacity-matched control.
- SFM is claimed to make the net "more resistant to interference"; without SFM both tasks degrade (Tab. 1).
- Since semantics live in the cost volume, the semantic branch is not frozen/reusable; the method needs joint labels (US3D pairs both).
- 4D volumes at 512x512 crops, batch 4, V100 16GB: heavy 3D conv with all channels carried.

### 5. Losses
- Not stated (no equation, weights or deep-supervision schedule). Fig. 1's Output1/2/3 suggest multi-output supervision but it is not described.

### 6. Training recipe
- PyTorch 1.8.1, batch size 4, Tesla V100 16GB (Sec. 3.1); 3,500 train crops 512x512, 338 val, 454 test. Optimizer, LR, schedule, epochs, augmentation, init, disparity range, frozen parts: not stated. Baselines "trained and tested" on the same split (Sec. 3.1) - however S2Net's numbers equal published values? Not verifiable here.

### 7. Results
- Disparity (Tab. 2): D1-Error 9.579 %, EPE 1.403 (S2Net 10.051/1.439; PSMNet 11.872/1.695).
- Segmentation (Tab. 3): mIoU 67.39 vs HRNetV2 61.38 (abstract: 61.38 -> 67.39). mIoU-3 66.27 (Tab. 1).
- Runtime/FPS/params/memory: not stated.
- Local summary issue: it says DCSFEM extracts "four-times-downsampled" features; the paper says "four times down-sampling", ambiguous. It otherwise matches Tab. 2/3.

### 8. Negative results & limitations
- Authors: only conclude future work on multi-view/3D reconstruction. No failure cases reported.
- Mine: (1) mIoU baselines (SegFormer, PSPNet, HRNetV2) are single-image 2D seg nets; S3Net sees both images, so a +6 mIoU gain may come from the stereo/height cue, not from "fusion"; no 2D-seg + stereo-input control. (2) Ablation rows remove both the semantic and disparity components, not matched capacity. (3) No loss, training or runtime details: not reproducible from the paper (code link exists). (4) No error bars, one split, no cross-city/cross-sensor test, yet the abstract claims "robustness and generalizability". (5) Tiny comparison: S2Net D1 gap 0.47, EPE gap 0.036. (6) Bridge mIoU is 35.96 (low) and is the only class where big gains (+8.95 vs PSPNet) matter.
- Satellite transferability caveats (inference): same as SDBF: US3D semantic classes (water, ground, roof, bridge) are tied to height, disparity represents elevation with piecewise-constant planes, epipolar-resampled nadir imagery, wide acquisition-time variation; driving stereo has perspective depth gradient, occlusion layers, thin structures, sky. Putting a semantic slice into the cost volume exploits the class->height prior which is much stronger in this domain. Metric: D1 here is a 3 px error rate (their definition) not KITTI D1.

### 9. Relevance to OUR model
- Semantic info inside the cost volume as an extra "disparity slice": a cute trick but needs joint training of one trunk; incompatible with our frozen predictor/decoder. Our SemanticCostGate already modulates candidates multiplicatively, which is the more portable version of this idea.
- SFM = two-branch conv product (a learned multiplicative gate), applied to features and 3D volumes. Portable as the gating operator inside our gate: cheap (2D version). Expected benefit unknown: ablation only measures the whole block (+3.26 mIoU / -0.86 D1) and only on one satellite dataset, in a joint-training setting where the gate also changes trunk learning. Our fusion head already trains only a gate; test as an ablation arm "SFM-gate vs sigmoid-gate" at equal capacity. Risk: modest; evidence quality low.
- MFM-style multi-round shared-volume refinement: not applicable (3D conv, heavy for RTX 3050/Jetson).
- Not novel relative to us except SFM; their domain differs strongly (see caveats). Cite as 2024 evidence that semantic+disparity joint cost volumes are still an active direction in remote sensing, not as evidence for real-time driving.

### 10. Key quotes/equations worth citing
- "The topmost layer of disparity in this 4D cost volume is reserved for semantic information, whereas the successive layers encapsulate disparity information from multi-scale features." (p. 2)
- "bilinear ... classification map; trilinear ... disparity map" (Fig. 1, p. 2).
- Tab. 1 (p. 3): full 67.39 / 66.27 / 9.579 / 1.403; w/o SFM 64.13 / 62.72 / 10.443 / 1.483.
- No equations in the paper beyond the 4D cost-volume shape H x W x D x C.
