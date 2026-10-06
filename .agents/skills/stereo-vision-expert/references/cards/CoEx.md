<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# CoEx card

Page refs are PDF pages (7 pages total, incl. appendix and references). Fully read.

### 0. Meta
- Title: Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation
- Authors: Bangunharcana, Cho, Lee, Kweon, Kim, Kim (KAIST). IROS 2021 (arXiv 2108.05773).
- PDF: paper/reference_papers/lightweight/CoEx_Bangunharcana_IROS2021.pdf
- Code: https://github.com/antabangun/coex ("will be made available", p.1)
- Domain: driving / general. Datasets: SceneFlow finalpass, KITTI 2012, KITTI 2015.

### 1. Problem & failure modes targeted
- Latency/memory of volumetric stereo; spatially varying aggregation (GANet SGA/LGA, CSPN) is accurate but slow (1.8 s / 1.0 s) and complex (p.1).
- Correlation cost volume loses channel information vs concatenation (poor accuracy, but cheap) (p.1).
- Multi-modal / flat disparity distributions at edge boundaries and textureless regions make soft-argmin's expected value diverge from GT (p.2, Fig. 4). Authors name "edge boundaries" as a multi-peak case; no edge metric is reported.
- Real-time accuracy gap (StereoNet 15 ms but weak).

### 2. Pipeline by stage
- **2a Feature extraction.** Shared MobileNetV2 (ImageNet pretrained) + U-Net style upsampler with long skips; outputs features I(s) at scales 1/4, 1/8, 1/16, 1/32 (p.3, Fig. 2). Channel counts per scale: **not stated**. 10 ms on RTX 2080Ti (Tab. II). Whole net 2.7M params (p.5). Trained end to end (backbone fine-tuned).
- **2b Semantic/prior branch.** n/a (ImageNet-pretrained backbone is the only external prior).
- **2c Cost volume.** Correlation (DispNetC style) of 1/4-scale left/right features, D=192 so D/4=48 levels, output D/4 x H/4 x W/4 (single channel; normalisation not stated) (p.3). Then conv3d 3x3x3, 8 ch (Tab. VI [2-1]).
- **2d Cost aggregation.** GC-Net-style 3D hourglass with reduced channels/depth (Tab. VI, p.6): conv3d 8 ch @1/4; down conv3d (s=2) 16 ch @1/8; 32 ch @1/16; 48 ch @1/32 (each stage 2 convs); then deconv3d 4x4x4 s=2: 32 @1/16, 16 @1/8, final deconv3d to 1 ch @1/4. A GCE module follows each stage: at I(4), I(8), I(16), I(32) on the way down and I(16), I(8) on the way up = **6 GCE modules**. Skip connections inside the hourglass are not spelled out in Tab. VI (Fig. 2 shows them).
  - GCE (Eq. 1): alpha = sigmoid(F2D(I(s))), C_o = alpha * C_i, F2D = 2D pointwise (1x1) conv producing c weights per pixel (c = cost-volume channels at that stage), broadcast multiplicatively across the disparity dimension (shared over d). Cost: (cI*c*h*w) + (c*d*h*w) (Eq. 7), n x n cheaper than neighbourhood aggregation (Eq. 8, p.7).
- **2e Disparity computation.** Top-k soft-argmax at 1/4 resolution over 48 levels: take the k largest cost values per pixel, softmax over them (others masked), weighted average of their disparity indices (p.3). Best k=2. k=1 is argmax (non-trainable); k=D recovers soft-argmin. Cannot be swapped in at test time (Tab. III, 48->2 = 0.7782 vs trained-with-2 0.6854).
- **2f Refinement.** none ("does not use any post aggregation refinement", p.5). Compare AANet+ with refinement (8.4M params).
- **2g Upsampling.** 1/4 -> full by "superpixel" 3x3 learned weighted average following Yang et al. [34] (superpixel FCN): "another CNN branch predicts the weights for each superpixel" (p.4). Architecture of that branch, whether it takes image or cost features, and whether disparity is rescaled x4: **not stated**. No ablation of this upsampler.
- **2h Fusion points.** (1) GCE: image (reference-left) feature -> cost volume, multiplicative channel gate (sem-like cue could replace/augment I(s)), direction image->cost, at 1/4,1/8,1/16,1/32 resolutions, 6 sites, shared across disparity. (2) Upsampling weights branch uses guidance (details unstated). Where a semantic/edge cue could enter: concatenate class features to I(s) before F2D (cheap), or use semantics to select k per pixel in top-k (no existing mechanism).

### 3. Block -> problem -> evidence table
All EPE on SceneFlow finalpass (Tab. III-V, p.5). Times ms on RTX 2080Ti.

| block | problem | evidence | context | cost |
|---|---|---|---|---|
| GCE at every scale ("Full") | correlation loses info; image guidance for aggregation | CoEx k=48: none 0.8552, One 0.8242, Full 0.7426. Full + k=2: 0.6854 (vs corr-only k=2 0.7942) | SF only, 10 epochs | +~1 ms total with top-k (26 -> 27 ms) |
| GCE on PSMNet (single layer) | make correlation ~ concatenation | PSMNet corr k=192: 1.053 -> One GCE 0.8285; concat k=192 0.8291 | SF | 223 -> 225 ms |
| GCE on concat PSMNet | | concat k=192 0.8291 -> One GCE 0.8176; k=6: 0.7437 -> 0.7321 | SF | +5 ms |
| Excite vs Add (Tab. IV) | is multiplicative gating the reason? | none 0.7426 / 3px 4.308; add 0.7310 / 4.159; excite 0.6854 / 4.021 | CoEx Full-GCE k=48 base | same |
| GCE vs neighbourhood aggregation (Tab. V) | is the 3x3 neighbourhood needed? | One GCE 0.7684 / 4.409 / 26 ms vs graph-neighbourhood 0.7732 / 4.435 / 47 ms | CoEx, one layer, DGL graph conv with relative-position MLP edges | 26 vs 47 ms |
| Top-k regression (CoEx) | multi-modal / flat cost distributions | corr-only: k=48 0.8552, 6 0.8262, 3 0.7928, 2 0.7942; Full-GCE: 48 0.7426, 6 0.7185, 3 0.7115, 2 0.6854 | SF | no added time at 1/4 res |
| Top-k on PSMNet | same | concat k192 0.8291 -> k6 0.7437; corr k192 1.053 -> k6 0.8798; corr+One: k192 0.8285, k6 0.7653, **k2 1.108 (diverges/worse)** | SF | full-res sorting: PSMNet 292 -> 405 ms |
| Train-with-k requirement | | train k=48, test k=2: 0.7782 vs 0.7426 base (CoEx Full), PSMNet corr+One 192->6: 0.8088 vs 0.8285 (small gain, vs trained-with-6 0.7653) | | - |
| MobileNetV2 backbone | speed | "ImageNet pretraining allows faster convergence" (p.4); **not ablated** | | 10 ms feature stage |
| Superpixel (3x3 learned weights) upsampling | 1/4 -> full res | **not ablated** | | |
| Correlation vs concat | cost | PSMNet corr 1.053 vs concat 0.8291 (without GCE) | SF | 223 vs 292 ms |
| SWA, ImageNet pretrain, aug | | not ablated | | |

### 4. Interactions & dependencies
- GCE rescues correlation: its benefit is largest when the cost volume is a 1-channel correlation (info the correlation drops is "excited back" by image features) (p.5-6). On concat volumes it helps less (0.8291 -> 0.8176 One).
- Top-k must be trained with top-k (48->2 test-only swap does not help, p.6). Too-small k can hurt where gradient flow is limited: PSMNet corr + One GCE k=2 = 1.108 (worse than k=192). In CoEx k=2 works with Full GCE. The k=48->2 gain is similar with and without Full GCE (-0.061 corr-only: 0.8552 -> 0.7942; -0.057 Full GCE: 0.7426 -> 0.6854), so the two modules look roughly additive (single run).
- Top-k at full resolution is too slow (sorting) -> forces regression at 1/4 + learned superpixel upsampling.
- Excitation vs addition: addition acts like a U-Net skip and helps little; the multiplicative gate matters.

### 5. Losses
Smooth-L1 only, on the final full-res disparity: L = 1/N sum_i smoothL1(d_GT,i - d_hat,i), smoothL1(x) = 0.5 x^2 if |x|<1 else |x|-0.5 (Eq. 3-4). N = labeled pixels. Single output, no deep supervision, no intermediate losses.

### 6. Training recipe
Adam (b1=0.9, b2=0.999) with Stochastic Weight Averaging; random crop 576x288; SceneFlow: 10 epochs, lr 1e-3 for 7 then 1e-4 for 3, batch 8; KITTI: init from SceneFlow, 800 epochs, lr 1e-3 halved at epochs 30, 50, 300, 90/10 train/val split of the training set. SceneFlow: finalpass, only pixels with disparity < 192 (p.4). RTX 2080Ti. Data augmentation: not stated. Pretrained MobileNetV2 (ImageNet).

### 7. Results
- Tab. I (p.4): SceneFlow EPE 0.69 (the ablation best is 0.6854), KITTI12 3px out-all 1.93, KITTI15 D1-all 2.13, 27 ms. Others: StereoNet 1.101 / 6.02 / 4.83 / 15 ms; AANet 0.87 / 2.42 / 2.55 / 62 ms; AANet+ 0.72 / 2.04 / 2.03 / 60 ms; DeepPrunerFast 0.97 / - / 2.59 / 62 ms; LEAStereo 0.78 / 1.45 / 1.65 / 300 ms; PSMNet 1.09 / 1.89 / 2.32 / 410 ms; GANet-deep 0.84 / 1.60 / 1.81 / 1800 ms.
- Tab. II (same hardware): feature / cost-agg (incl. cost-volume + regression) / refine / total: CoEx 10 / 17 / - / 27; AANet 22/32/32/88; AANet+ 11/21/45/80; LEAStereo 12/463/-/475. Text: 3.3x faster than AANet, -0.18 EPE, -0.46% KITTI12, -0.42% KITTI15.
- Params: 2.7M (CoEx) vs 8.4M (AANet+).
- **Edge/boundary measurements:** none. No EPE-edge, no D1-fg (CoEx reports only D1-all on KITTI15), no thin-structure or Middlebury/ETH3D data. The only boundary evidence is Fig. 4 (3 example disparity distributions where top-2 beats soft-argmin, one bimodal "edge" example) and Fig. 3 (KITTI15 test error maps, qualitative). So CoEx edge gains are asserted by motivation, not measured. Conditions of evidence: SceneFlow finalpass overall EPE only, 1/4-res regression + learned 3x3 superpixel upsampling.

### 8. Negative results & limitations
- Authors: top-k with too small k hurts (PSMNet k=2: 1.108); top-k at full-res is too costly; swap-in at test time does not work (p.6); addition-guidance only slightly helps.
- Mine: ablation is on SceneFlow only, 10-epoch schedule, single run, no seeds; KITTI numbers are leaderboard-only. The neighbourhood aggregation baseline (Tab. V) uses a custom DGL graph implementation with a single layer ("One"), so "GCE beats neighbourhood aggregation" is a weak comparison (0.7684 vs 0.7732 is within noise-level for one run), the 21 ms speed gap is the real argument. Whether GCE benefits from semantic features untested. Number of GCE modules (six) vs the common reading of "four scales".
- Errors in repo summary (summaries/lightweight/CoEx.md): it states GCE at four scales {1/4..1/32}; paper Tab. VI has six GCE modules (4 down + 2 up, p.6). It writes the correlation as normalised by N_c; the paper gives no normalisation. It states GCE conv has "c output channels" (OK).

### 9. Relevance to OUR model
- **GCE as the insertion mechanism for semantics.** GCE is a 1x1-conv + sigmoid channel gate broadcast across disparity; our SemanticCostGate is already semantic-conditioned gating of candidates, so GCE is essentially the "already done" form of this idea but applied inside an aggregator rather than on output candidates. Portable variant: gate per-candidate / per-channel features of the frozen A09 cost aggregation using YOLO-trunk semantic features (1x1 conv + sigmoid). Expected benefit: per CoEx Tab. IV/III, excitation >> addition (0.6854 vs 0.7310 EPE); cost near zero. Risk: A09 is frozen; the gate can only act on exposed intermediate tensors; GCE evidence is on a trained-from-scratch network, not a frozen one.
- **Top-k regression** (k=2): portable only if the disparity regression stage is trainable; with a frozen A09, a test-time swap does not help (p.6). Possible in the fusion head's candidate-weighting if our head chooses among a small candidate set (it already does something like top-k). Edge relevance: authors' motivation is bimodal edge pixels; no measurement, so treat as unproven for our edge problem.
- **Superpixel 3x3 learned upsampling**: close to our convex upsampling in G2/H2 (we found no sharpening); CoEx gives no ablation of it, so it offers no counter-evidence.
- Realtime relevance: 27 ms on a 2080Ti with 2.7M params; MobileNetV2 + 1/4 correlation + small 3D hourglass is a plausible Jetson baseline but 3D convs are TensorRT-heavy.
- Novel combination: GCE with semantic features (instead of RGB features) as the gate -> not in this paper; potentially novel but SemStereo/SGNet-type papers may cover; check.

### 10. Key quotes/equations
- "simple channel excitation of cost volume guided by image can improve performance considerably" (abstract).
- Eq. 1: alpha = sigma(F2D(I(s))); C_o(s) = alpha x C_i(s) (p.3).
- "the models need to learn to use the top-k soft-argmin regression during training" (p.6).
- "the two proposed modules only add 1 ms ... but gives 0.17 lower test EPE" (p.6).
