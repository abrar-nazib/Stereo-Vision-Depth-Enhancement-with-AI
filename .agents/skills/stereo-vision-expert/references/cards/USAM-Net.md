<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# USAM-Net card (cautionary negative result)

PDF (15 pp, arXiv 2503.14950v1) read in full; no supplement. Figures are simple block diagrams / sample outputs (Fig. 1 architecture, Fig. 2 self-attention).
FILENAME ERROR: the file is named "Sankaranarayanan"; the PDF's authors are Joseph Emmanuel DL Dayo and Prospero C. Naval Jr. (Univ. of the Philippines Diliman). The repo summary also lists "Sankaranarayanan et al." - wrong.

### 0. Meta
- Title: USAM-Net: A U-Net-based Network for Improved Stereo Correspondence and Scene Depth Estimation using Features from a Pre-trained Image Segmentation network
- Authors: Dayo and Naval Jr. (see above). Venue: arXiv 2025 (no peer-reviewed venue).
- PDF: paper/reference_papers/semantic_stereo/USAM-Net_Sankaranarayanan_arXiv2025.pdf
- Code: none given for USAM-Net (only a fork of OpenStereo cited as the eval framework, ref [2]).
- Domain: driving. Datasets: DrivingStereo (174,437 train pairs, 7,751 test; sparse LiDAR GT; half-resolution 879x400 used, p.10); KITTI 2015 and Middlebury used for transfer/"unseen" claims.
- Splits: DrivingStereo official train/test; KITTI 2015 evaluated "before/after fine-tuning" (fine-tune protocol, split sizes, epochs NOT stated); Middlebury: no numbers reported at all.

### 1. Problem & failure modes targeted
- Textureless/featureless regions (roads, flat vehicles, shadows, sky) and occlusion (p.2). Hypothesis: segmentation masks as extra input give object-boundary and surface cues.
- Not a joint-learning paper: there is no segmentation task, no task conflict analysis. The "segmentation pathway" is a frozen off-the-shelf SAM ViT-B run offline.

### 2. Pipeline by stage
- 2a Feature extraction: none separate. Left, right and SAM mask are concatenated channel-wise at the input (9 channels = 3+3+3; Tab. 1, p.6) and passed through a plain U-Net encoder: 5x [Conv 3x3 s2 -> LeakyReLU(0.01) -> BN] with channels 64-128-256-512-1024 (spatial 1/2 ... 1/32). Not siamese; no weight-shared per-view feature extraction.
- 2b Semantic prior: SAM ViT-B (93.7M params, ~900 ms/image on RTX 3090) run on the LEFT image only, pre-computed for all train/test images (p.10). The 3-channel rendering of the SAM masks (instance colour map? class-agnostic) is NOT specified. SAM is class-agnostic (instances/segments), so no class semantics exist (my note).
- 2c Cost volume: NONE. No correlation, no concatenation volume, no disparity range search. Left/right are just stacked as channels; the net regresses disparity directly. This is not a stereo-matching architecture in the usual sense.
- 2d Aggregation: U-Net decoder: 5 ConvTranspose (1024->512->256->128->64->32; kernels 3x3/4x4/4x3 as printed) + BN/LeakyReLU; optional single self-attention layer (query/key/value, softmax, skip) at the 1024-ch bottleneck (1/32) (Fig. 2); skip connections drawn in text only ("containing skip connections", Fig. 1 does not show them explicitly).
- 2e Disparity: 3 final convs (32->64->128->1, 1x1 last) + Sigmoid x 255 -> disparity in [0,255] directly (so a half-res 879x400 prediction in "full disparity range units"; scaling to image resolution unspecified). Regression, no soft-argmin.
- 2f Refinement: none. 2g Upsampling: transposed convs only.
- 2h FUSION POINTS: only one: SAM mask concatenated at the INPUT (early fusion, seg->disp, concat, full res). Plus optional self-attention at the bottleneck (not semantic-conditioned). No semantic gating, no consistency loss, no disp->seg path.
- Size: base U-Net 15.1M params, 2.18 ms on RTX 3090 (p.4); full pipeline adds SAM's 93.7M / 900 ms.

### 3. Block -> problem -> evidence table
Single training run per variant (30 epochs each; 56 GPU hours on 3xA100 total, p.10), no seeds. DrivingStereo half-res test.
| block | problem | evidence (Tab. 2) EPE / D1 / GD / L1 | context | cost |
|---|---|---|---|---|
| U-Net baseline (no seg, no attn) | reference | 0.964 / 2.53% / 4.17% / 0.12 | | 15.1M, 2.18 ms |
| + self-attention (no seg) | global context | 0.889 / 1.94% / 3.8% / 0.10 (best D1 and L1 of all four, EPE -0.075, D1 -0.59) | | + attention layer |
| + SAM seg (no attn) | boundaries/textureless | 0.924 / 2.65% / 3.68% / 0.115: EPE -0.040, GD -0.49, but D1 WORSE (+0.12) and L1 worse than attn-only | | +93.7M, +900 ms |
| + SAM seg + attn | both | 0.88 / 2.26% / 3.61% / 0.116: vs attn-only EPE -0.009, GD -0.19, D1 WORSE (1.94 -> 2.26), L1 worse (0.10 -> 0.116) | | |
| KITTI 2015 transfer, before fine-tune (Tab. 3) d1_all / epe | | Baseline 8.57/1.62; Attn 7.03/1.30 (best); Seg 8.27/1.46; Seg+Attn 10.04/1.63 (WORSE than baseline on d1_all, equal EPE) | trained on DrivingStereo only | |
| KITTI 2015 after fine-tune | | Baseline 5.72/1.21; Attn 5.60/1.11 (best); Seg 6.10/1.27 (worse than baseline); Seg+Attn 6.00/1.21 | | |
| Sky masking via SAM (Sec. 2.5, Fig. 4) | artifacts in top (no-LiDAR) region | qualitative artifacts removed, but "significantly reduces the accuracy on the test dataset" (p.9) - numbers not given | | |
| Middlebury | "segmentation matters more on unseen data" (p.14) | NO numbers or figure, only a sentence | | |
| Stage-wise 'depth bucket' ARD (Fig. 5) | | curve only; text: attn and seg act on mid-to-far range | | |

### 4. Interactions & dependencies
- Segmentation and self-attention are not additive: attention alone gave the best KITTI D1/EPE both before and after fine-tuning; adding SAM made things worse (authors: "adding segmentation information has little effect", p.12).
- The semantic cue is available only at left-view input resolution as arbitrary instance/segment colours; the network must discover how to use it from scratch with no explicit gating -> the fusion mechanism is too weak to exploit it (my interpretation).
- SAM masks must be precomputed; end-to-end inference costs ~900 ms for SAM.

### 5. Losses
Smooth-L1 between predicted and ground-truth disparity ("L1-loss" column in Tab. 2; formula given only by name, ref [17] cited for smooth-L1) over valid (non-zero GT) pixels (p.10). No smoothness term, no semantic term, no multi-scale supervision. (The repo summary's "L = L_disp + lambda L_smooth" and "A = sigma(f_seg)" attention formulation is NOT in the paper.)

### 6. Training recipe
30 epochs, Adam lr 1e-3, decay 0.9 per epoch, colour-jitter 10% (saturation, hue, contrast, brightness), per-channel normalisation (mean [0.50625,0.52283,0.41453], std [0.21669,0.19807,0.18691]); GT-mask for empty LiDAR pixels; sky-mask augmentation (randomly replace top part with middle/bottom part) tried; half-res 879x400; 3x A100 (DGX), 56 GPU-hours; forked OpenStereo framework for eval.

### 7. Results (and why it is a cautionary negative result)
Main table (Tab. 2) vs published (EPE/D1/GD): USAM-Net(Seg+Attn) 0.88/2.26%/3.61%; CFNet 0.98/1.46%/-; SegStereo 1.32/5.89%/4.78%; EdgeStereo 1.19/3.47%/4.17%; iResNet 1.24/4.27%/4.23%; StereoBase 1.15/2.19%; IGEV 1.06/1.50%. Abstract: "GD 3.61%, EPE 0.88 outperforming CFNet, SegStereo, iResNet".
Exactly why it is negative / not credible:
1. The semantic cue barely helps and is inconsistent: seg raises D1 (worse) in two of two comparisons (2.53->2.65; 1.94->2.26), L1 worse in the combined model, and on KITTI the best model is attention-only; seg+attn is worst on KITTI zero-shot d1_all (10.04 vs 8.57 baseline). The authors concede "adding segmentation information has little effect" (p.12).
2. The headline "best EPE/GD" is mostly NOT due to segmentation: the plain U-Net baseline already has EPE 0.964 (< CFNet 0.98, IGEV 1.06) and GD 4.17 (= EdgeStereo 4.17), as the authors note (p.10). Delta from segmentation on EPE: 0.964 -> 0.924 (-0.04 px), i.e. within what one would expect from run noise (single seed).
3. The comparison is invalid: others' EPE/D1 are quoted from other papers (resolution/protocol of those numbers is not stated here; DrivingStereo's native size is 1762x800, my knowledge, not from this PDF), while USAM-Net is evaluated at HALF resolution 879x400 (p.10), where disparities and thus EPE in px are ~half as large, so the table is likely not like-for-like; GD/ARD definition mirrors DrivingStereo but sample/resolution differ; D1 formula text ("exceeds either 3 px or 5%") contradicts the displayed max() definition (which requires both). Evaluation on masked sparse LiDAR pixels only, so "GD" excludes sky/far. EPE "average Euclidean distance" over "all pixels" yet masked.
4. Architecture is not stereo matching (no cost volume, early-concatenated stack, direct 0-255 sigmoid regression): it can learn monocular cues, so even the "baseline" may be largely monocular depth from the left image; segmentation is then just another mono cue. The KITTI transfer shows poor generalization.
5. Transfer claims unsupported: the Middlebury statement "segmentation has more impact on unseen datasets" has no data; the KITTI numbers show the opposite.
6. Sky-mask training reduced test accuracy (unquantified).
7. SAM costs 93.7M params / 900 ms (400x the base net's latency 2.18 ms) for a tiny/inconsistent gain; no real-time path.
8. No ablation of how the mask is encoded (class vs instance colours, channels), no semantic-class information at all (SAM is class-agnostic), no baselines with a standard cost-volume stereo net trained the same way, no seed variance, no code.

### 8. Negative results & limitations
All authors-admitted items: segmentation "little effect" on DrivingStereo/KITTI; larger effect "on unseen datasets, especially Middlebury" (asserted without numbers); sky masking hurts accuracy; the attention layer, not segmentation, is the effective ingredient (and only when no segmentation is present); Middlebury worse; KITTI harder. See Sec. 7 for my list. The repo summary "USAM-Net.md" misdescribes the method (claims attention A=sigma(f_seg) modulates stereo features; in fact SAM masks are an input channel group and the attention is plain self-attention at the bottleneck), omits the negative KITTI results and the ablation, and names the wrong first author.

### 9. Relevance to OUR model
- Nothing to port. Value is as a warning: (a) early-fusing a class-agnostic mask as input channels into a net without cost volume gives ~no stereo benefit; (b) a semantic cue needs a mechanism that conditions the matching/gating (as in SemanticCostGate/ClassResidual) to be useful; (c) never claim SOTA from half-res EPE vs full-res published numbers (we compare only against our own same-protocol controls, as in D/E/F); (d) attention/context modules without semantics can account for the whole gain - supports having equal-capacity no-semantics controls (our C4/D3/D4/E-controls).
- Our design already addresses USAM's failures: semantic information passes through a dedicated gate and residual with an equal-capacity control, evaluated on a real-domain set (F series).

### 10. Key quotes/equations
- "...adding segmentation information has little effect" (p.12; Sec. 3 Tab. 3 discussion).
- "...while the quality impact of segmentation on the DrivingStereo and KITTI dataset is small, it can be seen that it has more of an impact on unseen datasets, especially on the Middleburry images" (p.14) - no supporting data.
- "while this improves the qualitative output of the disparity, it was found out that this significantly reduces the accuracy on the test dataset" (p.9, sky masking).
- "Even the baseline U-Net model is equivalent to the EdgeStereo model on the GD metric." (p.10)
- SAM-ViT-b 93.7M params, 900 ms vs base 15.1M, 2.18 ms (p.4).
