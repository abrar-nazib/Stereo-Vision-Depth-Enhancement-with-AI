<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SIGLNet block card

Evidence grade: **C** (single-run ablation computed on the test split; the "semantic" module has no non-semantic guided-upsampling control; measured gain over plain bilinear is 0.019 px EPE). Remote-sensing transfer caveat in section 9.

### 0. Meta
- Title: SIGLNet: a semantic information-guided lightweight stereo matching network for satellite imagery. Xinsheng Wang, Mi Wang, Yingdong Pi, Niming Fan, Xu Cheng (Wuhan Univ., LIESMARS). Geo-spatial Information Science 2026, vol. 29(3), pp. 1806-1822 (online 27 Nov 2025). DOI 10.1080/10095020.2025.2584938. 18 PDF pages (cover page + 17 article pages), read fully. Page refs below are journal page numbers.
- PDF: paper/reference_papers/semantic_stereo/SIGLNet_Wang_GSIS2026.pdf. Code URL: not stated.
- Domain: remote sensing (very-high-resolution optical satellite, nadir-ish, pseudo-epipolar). Datasets: US3D (DFC2019 Track 2; WorldView-3, 0.3 m GSD, 4292 pairs 1024x1024: Jacksonville 2139, Omaha 2153) and WHUStereo (GaoFen-7 panchromatic, 1757 pairs, <0.8 m GSD) (p.1812). Disparity GT from airborne LiDAR.
- No existing summary under paper/reference_papers/summaries/ for this paper, so nothing to cross-check.

### 1. Problem & failure modes targeted
- Accuracy-oriented satellite stereo nets (PSMNet, HMSMNet) are heavy; goal is on-board / large-scale processing with low params and FLOPs (p.1808).
- Textureless, repetitive texture, disparity discontinuities and occlusions (Fig. 7/8 panels a-c, p.1814/1816). Low-res (1/4) disparity must be restored to full resolution without mesh artifacts at building boundaries (p.1811, p.1817).
- Admitted failure: water bodies, weak texture, illumination, clouds (Fig. 10, p.1817).

### 2. Pipeline by stage
- 2a Feature extraction: weight-sharing pretrained MobileNetV2, outputs 16/24/32/96/160 channels at 1/2, 1/4, 1/8, 1/16, 1/32 (p.1809). Multiscale feature fusion (MFF, Fig. 3): bilinear-upsample the deepest map, two 2D convs to compress channels, concat with the next-shallower map, repeated 3 times down to 1/4 -> 48 channels at H/4 x W/4 (p.1809). Frozen: not stated (no freezing mentioned; trained end to end).
- 2b Semantic / prior branch: none as a network. No segmentation head, no class labels, no semantic loss anywhere (only Eq. 3 smooth-L1). "Semantic information" in SIGDU is the left RGB image plus its gradient maps (p.1809, p.1811), processed by convs. The paper's own Introduction contribution 2 says "semantic guidance from RGB images" (p.1808). This is a naming choice; it is a learned guided-upsampling kernel predictor.
- 2c Cost volume: feature correlation (inner product over 48 channels) at H/4 x W/4 x D/4 (Eq. 1, p.1809). Disparity range [-96, 96] on US3D, [-128, 64] on WHUStereo at full resolution (p.1813), so about 48 levels at 1/4 for US3D.
- 2d Aggregation (p.1810, Fig. 4): 3D conv+BN+act raising channels 1 -> Ccv (Ccv value not stated); two 3D inverted residual blocks (MobileNetV2-style pointwise/depthwise/pointwise 3D); lightweight hourglass of four multi-branch adjustable-bottleneck (MAB, from MABNet) branches: dual-branch 3D convs with different dilations halve the volume to H/8 x W/8 x D/8 x Ccv, then 3D MAB, then two more dilated 3D convs to H/16 x W/16 x D/16 x Ccv/2; four branch outputs concatenated, fused by a 3D MAB, two 3D transposed convs restore H/4 x W/4 x D/4 x Ccv.
- 2e Disparity computation: soft-argmin over D/4 (p.1810) -> initial H/4 x W/4 disparity.
- 2f Refinement / iterative: n/a (single feed-forward pass).
- 2g Upsampling (SIGDU, Fig. 5, p.1811): left RGB + gradient maps projected to Cup=8 channels by a 2D conv; four sequential dilated-conv groups (different rates); softmax gives a 9-channel weight map (3x3 neighborhood, Fig. 9 caption). The 1/4 disparity is bilinearly upsampled to full resolution, 3x3 neighborhoods gathered with unfold, multiplied by weights (Hadamard) and summed. This is the same family as RAFT-style learned convex upsampling, with guidance from RGB+gradient instead of the GRU hidden state.
- 2h FUSION POINTS: one. Stage: upsampling (post disparity). Operator: image-conditioned per-pixel 3x3 kernel weights (softmax, weighted sum of upsampled disparity neighbors). Direction: image -> disparity. Resolution: full resolution. No cue enters feature, cost-volume or aggregation stages.

### 3. Block -> problem -> evidence table
All Table 4 numbers are from the "All" US3D test set (EPE 1.521 for full model matches Tab. 1 "All"), so they are test-split ablations, single run.
| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| SIGDU vs trilinear upsampling of 1/4 cost volume then regress (A1) | full-res detail recovery | A1 EPE 1.618 / D1 10.36 vs full 1.521 / 9.53 (-0.097 px, -0.83 pp) (Tab. 4, p.1817) | US3D test, 100 epochs, batch 1 | A1 1.43M, 26.35G, 120.72 ms |
| SIGDU vs bilinear upsampling of the 1/4 disparity (A2) | same | A2 1.540 / 9.83 vs 1.521 / 9.53 (-0.019 px, -0.30 pp) (Tab. 4) | same | A2 1.43M, 26.35G, 112.96 ms; full 1.44M, 30.00G, 113.12 ms. SIGDU adds ~0.01M params, +3.65 GFLOPs, +0.16 ms |
| Serial vs parallel dilated convs (A3) | receptive field | parallel 1.538 / 9.75, +6.7 GFLOPs (Tab. 4) | same | 1.44M, 32.72G, 117.94 ms |
| Cup = 16 / 24 / 32 (A4-A6) | feature width | EPE 1.525 / 1.522 / 1.516; D1 9.66 / 9.57 / 9.49 (Tab. 4); non-monotone vs Cup=8 (1.521): differences within 0.01 px, same order as run noise | same | FLOPs 38.5 / 51.8 / 69.9 G |
| Cup = 8 choice | accuracy-complexity | chosen by looking at Tab. 4 test numbers | - | 1.44M |
| MobileNetV2 + MFF to 1/4 | efficiency | not ablated | - | - |
| 3D IRB + MAB hourglass aggregation | efficiency vs PSMNet hourglass | not ablated (only whole-network comparison, Tab. 3) | - | whole net 1.44M, 30.0 GFLOPs |
| Correlation vs concat cost volume | memory | argued, not ablated (p.1809) | - | - |
| Semantic (class) guidance | claimed | no segmentation input exists; no control swapping RGB+gradient for a non-semantic cue | - | - |
| Edge/boundary metric | claimed boundary gains | no edge-region metric reported; evidence is Fig. 7/9 visuals only | - | - |

### 4. Interactions & dependencies
- The 9 softmax weights are trained only through the final smooth-L1 (Eq. 3); the paper shows they drift toward building outlines as training proceeds and that the best WHUStereo epoch is epoch 78 (Fig. 9, p.1818), implying epoch selection that may have used the test set (selection rule not stated for WHU).
- A1 vs A2: bilinear on the disparity already beats trilinear on the cost volume (1.540 vs 1.618), so most of the gain over A1 comes from operating on the disparity, and only 0.019 px from the learned kernel.
- Authors list failures where "semantic guidance fails": water, clouds, shadowed building (Fig. 10, p.1817); the guide has no segmentation prior to fall back on.

### 5. Losses
- Smooth L1 between final full-res disparity and GT (Eq. 3-4, p.1812): 0.5x^2 if |x|<0.5 else |x|-0.5, averaged over pixels. Single term, no deep supervision on the 1/4 disparity, no edge loss, no semantic loss.

### 6. Training recipe (p.1813)
- RTX 4090, PyTorch 1.12.1, 100 epochs, batch size 1, Adam (beta 0.9/0.999), LR 1e-3 halved every 10 epochs, no data augmentation, images normalized to [-1,1].
- US3D: 1600 Jacksonville pairs train, 270 Jacksonville val, test = 2153 Omaha + 269 remaining Jacksonville. WHUStereo: official 1220/122/415. Baselines (PSMNet, HMSMNet, GMStereo, StereoNet, BGNet) trained with the same recipe (p.1813). Seeds: single run. Checkpoint selection rule: not stated.

### 7. Results
- US3D test, all (Tab. 1, p.1815): SIGLNet EPE 1.521 / D1 9.53%; PSMNet 1.611 / 10.05; BGNet 1.625 / 10.77; HMSMNet 1.649 / 10.73; GMStereo 1.815 / 13.99; StereoNet 1.764 / 13.12; SGM 2.851 / 23.08. D1 threshold t = 3 px (p.1812).
- WHUStereo (Tab. 2, p.1817): PSMNet 2.639 / 25.99 best; SIGLNet 2.641 / 27.65 (second EPE); BGNet 2.676 / 28.27. The abstract's "highest accuracy" holds on US3D only.
- Complexity (Tab. 3, p.1817): SIGLNet 1.44M params, 30.00 GFLOPs, 113.12 ms; StereoNet 0.624M / 209.28G / 83.18 ms; BGNet 2.98M / 128.04G / 89.64 ms; PSMNet 5.22M / 1400.97G / 570.88 ms. Input size and GPU for FLOPs/latency: not stated. SIGLNet is about 30 ms slower than StereoNet/BGNet despite fewer FLOPs.
- Qualitative large-scale DSM from GF-7 Tianjin (35,800 x 4000 px, 2100 tiles of 1024 with 25% overlap, 221 s on a GPU) (p.1818).

### 8. Negative results & limitations
- Authors: water bodies, weak texture, illumination, clouds (Fig. 10); slower than StereoNet/BGNet by about 30 ms (p.1817-1818). Future work: integrate "advanced semantic segmentation" (p.1819), confirming the current guide is not segmentation.
- Mine: (i) the only module-level ablation is on the test split and single run; (ii) no RGB-guided-but-not-semantic control exists, because the guide is already RGB; thus "semantic" cannot be separated from ordinary guided upsampling; (iii) the effect over plain bilinear is 0.019 px EPE (1.2%) and 0.30 pp D1, within plausible seed noise; (iv) no edge-specific metric despite boundary claims; (v) Cup choice is made on test numbers; (vi) latency hardware unspecified.

### 9. Relevance to OUR model
- Nothing semantic is portable: no segmentation labels or logits are used. The upsampler is an RGB+gradient-conditioned 3x3 softmax kernel on a bilinear-upsampled disparity; our G1/G2/G3 (RGB-guided residual, learned convex half-to-full reconstruction, stereo-warp correction) and H1/H2 are the same family and did not sharpen edges or lower EPE. SIGLNet's tiny gain (-0.019 px vs bilinear, test split, single run) is consistent with that, and independently corroborates "guided upsampling gives at most marginal EPE gain".
- The question "does semantic guidance beat non-semantic guided upsampling?" is unanswered by this paper: it never tests it. Our D2 vs D4 / E3 vs E4 equal-capacity pairs (about -0.24 px) are stronger evidence than anything here.
- Remote-sensing caveat: in nadir satellite imagery, class (roof / tree / ground / water) is a strong prior for height and hence disparity, and rooftop planes are near-constant disparity; in forward-facing driving scenes class gives no disparity prior (road and building facades are slanted, disparity varies with range inside one class), so any claimed class-to-disparity behaviour must be re-validated for VKITTI/KITTI.
- Efficiency pointers only: correlation at 1/4 with D/4 levels and 3D inverted-residual aggregation are standard lightweight choices already in LightStereo/CoEx lineage. Risk of adopting: none needed. Verdict: not a novel-combination source; at most cite as satellite lightweight baseline and as an example of "guided upsampling ~ negligible".

### 10. Key quotes/equations worth citing
- "...semantic information-guided upsampling module... according to semantic guidance from RGB images" (contribution 2, p.1808): evidence that "semantic" = RGB.
- Eq. 3-4: smooth-L1 only loss (p.1812). Tab. 4 A1/A2/full rows (p.1817). "the guiding role of semantic information is still limited" (Sec. 4.2, p.1817).
- PDF: paper/reference_papers/semantic_stereo/SIGLNet_Wang_GSIS2026.pdf
