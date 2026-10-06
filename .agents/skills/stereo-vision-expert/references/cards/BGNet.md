<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# BGNet card

Page refs are PDF pages 1-10 (printed 12497-12506). Fully read (no separate supplement exists).

### 0. Meta
- Title: Bilateral Grid Learning for Stereo Matching Networks
- Authors: Xu, Xu, Yang, Jia, Guo (Orbbec / Hefei Univ. Tech. / Sun Yat-sen Univ.). CVPR 2021.
- PDF: paper/reference_papers/lightweight/BGNet_Xu_CVPR2021.pdf
- Code: https://github.com/YuhuaXu/BGNet (abstract, p.1)
- Domain: driving + indoor generalization. Datasets: SceneFlow finalpass, KITTI 2012/2015, Middlebury 2014 (eval; fine-tune on 78 extra pairs for ablation), IRS (extra synthetic training in one experiment).

### 1. Problem & failure modes targeted
- Latency: top accuracy methods aggregate at 1/3-1/4 resolution (GANet 1.8 s, PSMNet 0.41 s); StereoNet (1/8, bilinear upsample + hierarchical refinement) is fast but inaccurate (p.1).
- Edge blur and thin-structure loss from bilinear/linear upsampling of low-res cost volumes: the module is explicitly an edge-preserving cost-volume upsampler (abstract, p.5).
- Generalization to Middlebury (synthetic -> real) is also reported.

### 2. Pipeline by stage
- **2a Feature extraction** (p.3-4): ResNet-like, shared; 3 convs 3x3 strides 2,1,1; four residual layers strides 1,2,2,1 -> unary features at 1/8; two hourglass nets for receptive field; all 1/8 features concatenated = 352 channels -> fl, fr. Trained from scratch. A separate high-res feature map (source layer/resolution **not stated in text**; Fig. 2 shows a branch from the early feature extraction of the left image) feeds the guidance map.
- **2b Semantic/prior branch.** n/a.
- **2c Cost volume.** Group-wise correlation (GwcNet) at 1/8, Ng=44 groups (p.4). D_max=192 for SceneFlow evaluation.
- **2d Aggregation.** One hourglass only: two 3D convs reduce channels 44 -> 16, then a U-Net-like 3D conv net with skip connections replaced by element-wise summation (p.4). Output C_L at 1/8 (aggregated channel count not stated).
- **2e Disparity.** Soft-argmin over the UPSAMPLED cost volume: D_pred = sum_d d softmax(C_H(x,y,d)) (Eq. 2), i.e. regression at high resolution, not at 1/8.
- **2f Refinement.** BGNet: none. BGNet+ adds an hourglass disparity-refinement module "as in AANet" [37] (+7.0 ms) (p.6, Tab. 5). Its input resolution is not stated.
- **2g Upsampling = CUBG (Cost volume Upsampling in the learned Bilateral Grid)** (Sec. 3.1, Fig. 1):
  1. A 3x3x3 3D conv converts the aggregated cost volume C_L (x,y,d,c) into the bilateral grid B (x,y,d,g); grid size W/8 x H/8 x Dmax/8 x 32, i.e. 32 is the guidance (g) dimension (p.3).
  2. Guidance map G: single-channel high-res map from the high-res feature maps through two 1x1 convs; "guidance information of each pixel depends on its own feature vector", hence sharp edges (p.3).
  3. Slice (parameter-free): C_H(x,y,d) = B(sx, sy, sd, sG G(x,y)) (Eq. 1), linear interpolation in the 4D grid; s in (0,1) spatial/disparity ratio, sG = ratio of grid g-levels to guidance levels. Output C_H in R^{W,H,D}, scalar cost per (x,y,d). Resolution of C_H for BGNet itself is not stated explicitly (for embedded nets: 1/2, 1/4, 1/3, 1/2).
  - Runtime: grid generation 4.3 ms, guidance 0.07 ms, slicing 0.74 ms; in-network ~0.2 ms slower than linear upsampling (LU 3.78 ms) (p.6).
- **2h Fusion points.** The only second cue is the high-res image feature -> guidance map G -> picks the grid cell (i.e. which low-res disparity/cost to read) per pixel. Direction image->cost, resolution full/high; operator = learned lookup index (not concat/add/gate). Semantic/edge cue could enter: (a) concatenate class features (or a class edge map) to the features feeding the two 1x1 convs of G (the luma ablation below shows richer guidance helps); (b) multi-channel guidance (class-aware) instead of a scalar G.

### 3. Block -> problem -> evidence table
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| CUBG vs linear cost-volume upsampling (LU) | edge blur / thin structures | SceneFlow EPE 1.17 vs 1.40; EPE-edge 5.95 vs 8.13 (-2.18); EPE-flat 0.68 vs 0.71 (-0.03) (Tab. 1) | BGNet, SF finalpass, 1/8 cost volume | 25.3 vs 25.1 ms |
| same, Middlebury 2014 | | Bad2.0 16.8 vs 18.7 %; near edges 45.3 vs 48.9 %; flat 13.0 vs 14.7 % (p.6) | models fine-tuned with 78 extra Middlebury pairs; resolution not stated in ablation text (Tab. 6 is at half-res) | |
| same, KITTI15 (160/40 split) | | D1-all 2.01 vs 2.14 % (p.6) | | |
| learned guidance vs luma guidance | what guides the slice | luma(input image) guidance: SF EPE 1.17 -> 1.28 (p.6); edge/flat split not reported | | |
| cost volume at 1/16 instead of 1/8 | speed/accuracy | SF EPE 1.17 -> 1.58, 25.3 -> 17.1 ms (p.6) | | |
| CUBG embedded in GCNet (cv 1/2 -> 1/8) | speed, accuracy | EPE 2.51 -> 1.07, 1673 -> 57.1 ms (Tab. 2) | SF | 29x faster |
| CUBG in PSMNet (1/4 -> 1/8) | | 1.09 -> 0.92, 439.6 -> 89.4 ms; KITTI15 D1-all 1.95 -> 2.07, 410 -> 79 ms; Middlebury Bad2.0 19.36 -> 20.77 (Tab. 2,3) | SF EPE improves but KITTI/Middlebury get slightly worse | |
| CUBG in GANet-deep (1/3 -> 1/6) | | 0.95 -> 0.63 EPE, 2240 -> 533 ms; KITTI15 1.58 -> 1.67; Middlebury Bad2.0 15.67 -> 15.30 | | |
| CUBG in DeepPrunerFast (PatchMatch bounds replaced by full 1/8 agg., upsample to 1/2) | | 0.97 -> 0.84, 64.7 -> 56.6 ms; KITTI15 2.06 -> 1.91; Middlebury 16.80 -> 15.14 (Tab. 2,3) | | |
| BGNet+ refinement hourglass | accuracy | KITTI15 D1-all 2.51 -> 2.19 (Tab. 4); +7 ms | | 25.4 -> 32.3 ms |
| IRS + FlyingThings3D training (data) | generalization | Middlebury Bad2.0 BGNet 17.5 -> 13.5, BGNet+ 17.2 -> 11.6 (p.7, Tab. 6) | indoor synthetic data | no runtime cost |
| Group-wise corr (Ng=44), single hourglass w/ summed skips, ResNet-like extractor | | **not ablated** | | |
| Slicing disparity-axis interpolation (sd) vs only spatial | | **not ablated** | | |

### 4. Interactions & dependencies
- Gains depend on what the baseline upsampler is: large against linear upsampling of a 1/8 cost volume (BGNet), small or negative when the original network already built its volume at 1/4 or 1/3 (PSMNet-BG D1-all worse 1.95 -> 2.07; GANet-BG 1.58 -> 1.67); for those nets the benefit is mainly 4-29x speed.
- The "learned grid" (3x3x3 conv) and guidance are trained jointly under only the final smooth-L1 loss; the guidance becomes edge-aware via end-to-end training (no explicit edge supervision).
- Guidance quality matters: luma < learned feature guidance (1.28 vs 1.17).
- Cost-volume resolution: 1/16 is too coarse even with slicing (1.58).

### 5. Losses
Smooth-L1 on the final disparity only: L = sum_p smoothL1(D_pred(p) - D_gt(p)), smoothL1(x)=0.5x^2 if |x|<1 else |x|-0.5 (Eq. 3, p.4). No deep supervision, no auxiliary edge loss. (Sum, not mean, as printed.)

### 6. Training recipe
SceneFlow (finalpass): augmentation = asymmetric chromatic aug, y-disparity aug [38], Gaussian blur, scale zoom, each with 50% probability; remove pairs with >25% disparity > 300 (as CRL); one-cycle LR schedule with max lr 0.001; batch 16; crop 512x256; 50 epochs; Adam (0.9, 0.999); max disparity 192 for eval, out-of-range pixels excluded. KITTI15: 20 val pairs, 180 + 194 KITTI12 train; fine-tune from SceneFlow at constant lr 0.001 for 300 epochs, repeated three times, best chosen (selection by evaluation metric; the selection set is the 20 val pairs). Middlebury ablation: 13 additional datasets with GT (78 pairs), KITTI15 split 160/40. Hardware: RTX 2080Ti; PyTorch.

### 7. Results
- Tab. 4 (KITTI test): BGNet: 2012 2-noc 3.13, 2-all 3.69, 3-noc 1.77, 3-all 2.15, EPE-noc 0.6, EPE-all 0.6; KITTI15 D1-bg 2.07, D1-fg 4.74, D1-all 2.51; 25.4 ms (39 fps). BGNet+: 2.78/3.35/1.62/2.03/0.5/0.6; D1 1.81/4.09/2.19; 32.3 ms. Compare: AANet 2.55 D1-all @ 62 ms (fg 5.39), DeepPruner-Fast 2.59 @ 61 ms, FADNet 2.82 @ 50 ms (fg 3.50), StereoNet 4.83 @ 15 ms, GANet 1.81 @ 1800 ms (fg 3.46), PSMNet 2.32 (fg 4.62) @ 410 ms, EdgeStereo-V2 2.08 (fg 3.30) @ 320 ms.
- Tab. 5 runtime (KITTI15, RTX 2080Ti): feature 8.8, cost build+agg 12.2, bilateral grid 4.3, refinement (BGNet+) 7.0, total 32.3 ms.
- Tab. 6 Middlebury 2014 (half-res) Bad2.0, trained on synthetic only: PSMNet 25.1, iResNet-i2 19.8, GANet 20.3, DSMNet 13.8, DeepPruner-Fast 17.8 -> 16.5 with BG, BGNet 17.5, BGNet+ 17.2, BGNet(IRS) 13.5, BGNet+(IRS) 11.6. DSMNet adds 15.4 ms by domain-invariant norm.
- Params: **not stated** anywhere in the paper.
- **Edge/boundary measurements (the only one of the four papers with a real edge metric).**
  - Definition (p.5): Canny edges on the GT disparity map, dilated with a 5x5 square structuring element; EPE-edge = mean EPE inside the dilated edge band, EPE-flat = EPE elsewhere. Evaluated on the SceneFlow finalpass test set, valid pixels per the usual rule (disp < 192).
  - Result: CUBG vs LU: EPE-edge 5.95 vs 8.13 (-27%, -2.18 px), EPE-flat 0.68 vs 0.71 (-0.03). So 0.23 of the overall EPE gain (1.40 -> 1.17) is edge-driven; "in flat regions the errors are comparable". Edge-band error remains ~8.7x the flat error (5.95 vs 0.68).
  - Middlebury: near-edge Bad2.0 45.3 vs 48.9 (-3.6 points), flat 13.0 vs 14.7 (-1.7 points): near-edge gain is larger in absolute but both are similar in relative terms (7% vs 12%); so on real data the "edge-specific" advantage is far smaller than on SceneFlow and the flat-region gain is also real.
  - Conditions: gains measured against **linear upsampling of a 1/8-resolution cost volume** (baseline cannot see the image at all), at resolution 1/8 -> full; guidance = learned feature map (not luma; luma costs 0.11 EPE overall); SceneFlow is synthetic with sharp, clean GT edges. When embedded into nets whose native volume was already at 1/4 or 1/3, KITTI D1-all got slightly worse (PSMNet-BG, GANet-BG), i.e. no edge benefit visible on real KITTI at those resolutions. No D1-fg ablation between LU and CUBG. Thin-structure claims are qualitative (Fig. 3 SceneFlow, Fig. 4 KITTI, Fig. 5 Middlebury).

### 8. Negative results & limitations
- Authors: 1/16 volumes too coarse (1.58); CUBG slightly hurts KITTI D1-all when added to PSMNet/GANet (Tab. 3), described as "comparable accuracy"; DSMNet generalizes better.
- Mine: (1) The edge ablation is single-run on synthetic SceneFlow against a deliberately naive LU baseline; no comparison against edge-aware alternatives (guided filter, learned convex upsampling, CARAFE, RAFT-style convex upsampler) so it does not show that bilateral slicing beats convex upsampling. (2) 5x5 dilation of Canny edges on GT disparity at 960x540 is a thin band; includes occlusion boundaries and texture edges of GT alike. (3) Guidance source resolution not stated; "luma guidance" result is only overall EPE. (4) Model selection for KITTI by best-of-3 runs on a 20-pair val set. (5) No parameter counts. (6) KITTI numbers are fine-tuned; no D1-fg gain claimed from CUBG itself (BGNet fg 4.74 worse than FADNet 3.50, AANet 5.39 better).
- Errors in repo summary (summaries/lightweight/BGNet.md): wrong code URL (3DCVdeveloper vs paper's YuhuaXu); says the cost volume is "reinterpreted directly, no conversion" and the grid is (x,y,g,c) with "intensity replaced by disparity", whereas the paper uses a 3x3x3 conv to produce a 4D grid (x,y,d,g) with g = 32 guidance levels and slices over disparity too (Eq. 1).

### 9. Relevance to OUR model
- Maps to our G/H finding. BGNet is the only paper here that quantifies edge-band EPE, and its gain is relative to linear cost-volume upsampling from 1/8. Our predictor already outputs near-full-resolution tile disparity (HITNet lineage), so the headroom BGNet exploited (a blurry 1/8 cost volume) is mostly absent. This explains, plausibly (hypothesis, not tested), why RGB-guided residuals and convex upsampling gave no sharpening in G/H: the baseline was not an edge-blind linear upsampler.
- Where it could still matter: if our frozen A09 produces candidates at coarse resolution (YOLO trunk features at 1/8-1/16), a slicing step on the candidate cost/probability volume under a learned guidance (RGB feature + semantic edges) is a parameter-free upsampler with a cheap 0.74 ms slice. Insertion point: between the SemanticCostGate output (candidate volume at coarse res) and the final regression. Expected benefit: edge-band EPE; cost: small; risk: negative-ish on real data (Tab. 3 shows no real KITTI gain at 1/4-1/3), and requires the candidate volume to exist as a (x,y,d,g) tensor (4D grid memory W/8 x H/8 x D/8 x 32).
- Semantic guidance idea: replace/augment scalar G with a class-informed guidance (concat class probabilities before the two 1x1 convs), so grid lookups cannot cross class boundaries: not in this paper, plausibly novel in combination, but our convex upsampling null result suggests testing against a strong baseline first.
- Luma ablation (1.28 vs 1.17) suggests learned features > raw luma, consistent with using YOLO features as guidance rather than RGB.

### 10. Key quotes/equations
- "Performing disparity regression in a high-resolution cost volume obtained via a slicing layer in the bilateral grid forces the predictions of our network to follow the edges in the guidance map G" (p.5).
- Eq. 1: C_H(x,y,d) = B(sx, sy, sd, sG G(x,y)) (p.3).
- "EPE-edge ... Canny ... dilated with a 5x5 square structuring element" (p.5); Tab. 1: 5.95 vs 8.13 edge, 0.68 vs 0.71 flat.
- "If the original guidance map is replaced with the luma-version of the input image, the EPE on Scene Flow increases from 1.17 to 1.28" (p.6).
