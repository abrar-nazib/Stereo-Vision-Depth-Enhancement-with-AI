<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# HITNet card

Page refs are PDF pages (1-11 main paper, 12-18 supplement). I read all 18 pages including figures.

### 0. Meta
- Title: HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching
- Authors: Tankovich, Hane, Zhang, Kowdle, Fanello, Bouaziz (Google). CVPR 2021 (arXiv 2007.12140v5).
- PDF: paper/reference_papers/lightweight/HITNet_Tankovich_CVPR2021.pdf
- Code: https://github.com/google-research/google-research/tree/master/hitnet (p.6, p.16) (trained models + eval scripts for benchmark submission).
- Domain: general passive stereo (driving, indoor, high-res Middlebury). Datasets: SceneFlow (FlyingThings3D only for train), KITTI 2012/2015, ETH3D two-view, Middlebury v3 (p.6, p.12).

### 1. Problem & failure modes targeted
- Latency: full 3D cost volumes + 3D conv are too slow; low-res cost volumes (StereoNet 1/8 etc.) lose thin structures and edges because the initialization is missing them (p.2-3).
- Edge/thin-structure recovery: "matches from the initialization module are provided at each resolution to facilitate recovery of thin structures that cannot be represented at low resolution" (p.2).
- Textureless regions: handled by starting at low resolution and hierarchically upsampling with propagation (p.2). Ablation shows real KITTI benefits more from multi-scale than synthetic SceneFlow (p.13).
- Slanted surfaces / subpixel accuracy: fronto-parallel tiles cause stair-stepping when upsampled; slant plus warping gives subpixel matching (p.2, p.14).
- Generalization from synthetic: argued via few parameters (p.18, Tab. 5).
- Occlusion: addressed by the random right-image patch replacement augmentation ("inpainting"), not by an architectural block (p.13).

### 2. Pipeline by stage
- **2a Feature extraction.** Shared (siamese) small U-Net run on left and right images. Down block = 3x3 conv + 2x2 stride-2 conv; up block = 2x2 stride-2 transpose conv, concat with skip, 1x1 conv, 3x3 conv; leaky ReLU (p.3-4). Outputs a feature map e_l at every decoder resolution l=0..M (2^l downsampled), all trained (no frozen parts). Default M=4 (5 levels); channels 16,16,24,24,32 for the SceneFlow HITNet (p.15); L: 32,40,48,56,64; XL: 32,40,48,56,64 extractor with 64-ch propagation (p.16). Middlebury uses M=5 (6 scales, p.12). The decoder outputs at high resolution still carry global context from the upsampling path (p.3).
- **2b Semantic/prior branch.** n/a. Pure stereo. Note: nothing in HITNet uses a pretrained or semantic encoder.
- **2c Cost volume construction.** No stored cost volume (p.4). At each level l, a 4x4 conv over feature map e_l extracts tile features: stride 4x4 on the LEFT (reference) so tiles tile the image (overlap-free), stride 4x1 on the RIGHT so disparity resolution is kept at full width; followed by leaky ReLU and an MLP (p.4). Matching cost  rho(l,x,y,d) = ||eL~_{l,x,y} - eR~_{l,4x-d,y}||_1 (Eq. 2), evaluated exhaustively for all d in [0,D] but only the argmin is kept ("a fused Op", never materialised, 0.25 ms for all resolutions, p.16). Init conv outputs 16 ch, then 2-layer MLP with 32 and 16 ch (p.15). Max disparity D: 320 SceneFlow training, 256 KITTI, 160 Middlebury runtime default (p.12, p.16). Whether d is rescaled per level (units of level-l pixels vs full-res) is **not stated**.
- **2d Cost aggregation.** None as such. Spatial aggregation happens only in the propagation CNN (2f). Local re-evaluation of costs in a +-1 disparity band replaces a cost volume ("sparse version of the learnable 3D cost volume", p.3).
- **2e Disparity computation.** WTA argmin per tile at init (Eq. 3, integer disparity; the tile hypothesis is h_init=[d_init,0,0,p_init]). Final disparity = plane equation of the winning full-res 1x1 tile (p.5). No soft-argmin.
- **2f Refinement / iterative update (the core).**
  - Tile hypothesis h=[d, dx, dy, p] (Eq. 1): slanted plane (disparity, x/y gradient) plus a learned descriptor p (13 ch default, so h has 16 ch, p.15).
  - Descriptor init: p_init = D(rho(d_init), eL~) with D a perceptron = 1x1 conv + leaky ReLU, input contains the best-match cost so the net knows match confidence (Eq. 4).
  - Warping: each tile is expanded to a 4x4 patch d'_ij = d + (i-1.5)dx + (j-1.5)dy (Eq. 5); right features eR_l are linearly interpolated along the scan line; cost vector phi(e,d') in R^16 = the 16 per-pixel L1 differences (Eq. 6).
  - Local cost volume: a = [h, phi(d'-1), phi(d'), phi(d'+1)] (Eq. 7) so 16 + 3x16 channels input to the update CNN U_l.
  - Update CNN U_l: 1x1 conv + leaky ReLU to cut channels, then residual blocks (no batch norm, 3x3 convs, dilations to enlarge receptive field, Fig. 11). Outputs deltas dh for each of n hypotheses plus a scalar confidence w per hypothesis (Eq. 8).
  - Hierarchy: at the coarsest level M there is n=1 hypothesis (the init). Apply update (sum), upsample x2 with nearest neighbour in position, using the plane equation to upsample d (dx,dy,p are NN-upsampled). At level M-1 there are n=2 hypotheses (init + upsampled coarse one); both are warped/updated (Fig. 12; output 2x(16+1) ch), highest w wins per location. Continue to level 0 at 4x4 tiles. Then 3 extra propagation steps on 4x4, 2x2, 1x1 tile sizes with n=1 (p.5). The 2x2/1x1 tiles are "in-painting" steps at full resolution (p.15).
  - Per-step blocks: KITTI/ETH3D "last 3 propagation steps" use 4,4,2 res blocks, 32,32,16 ch, dilations (1,3,1,1),(1,3,1,1),(1,1); SceneFlow HITNet 6,6,6 blocks with 32,32,16 ch and dilations 1,2,4,8,1,1 style; L/XL/Middlebury use 6,6,6 blocks with 32/32/32 (L, Middlebury) or 64/64/64 (XL) (p.15-16). Intermediate coarse steps use 2 res blocks, no dilation (p.15). Feature scale used for warping at the final three steps is described inconsistently (see section 8).
- **2g Upsampling.** Not a separate module: plane-equation upsampling x2 per level, NN for the rest, ending with 4x4 -> 2x2 -> 1x1 tile propagation at full-res feature resolution (p.5). The slant is explicitly what makes upsampling edge-aware: ablation "No slant" replaces plane upsampling with bilinear (p.14).
- **2h Fusion points (second cue).** None present. Candidate insertion points for a semantic/edge cue: (i) concatenate class features/probs into the descriptor p at init (cheap, no extra matching); (ii) append them to the update-CNN input a (Eq. 7), as a gate/bias on delta-h and w; (iii) use semantics to modulate the confidence w that selects between the init and upsampled-coarse hypothesis (this is exactly the multi-hypothesis fusion site, the closest analogue of our SemanticCostGate); (iv) per-class priors on (dx,dy) (planar classes should have smooth, small slants; thin classes should not).

### 3. Block -> problem -> evidence table
Ablation numbers are Tab. 6 (p.15): SceneFlow finalpass EPE / bad0.1 / bad1 / bad3 and KITTI 2012 EPE / bad2 / bad3. The full "HITNet" baseline row: SF 0.529 px, 24.0%, 5.52%, 3.00%; KITTI12 0.484 px, 2.91%, 2.00%. Caveat: Tab. 7 (p.17) lists "HITNet Single-scale" with EPE 0.53 / bad1 5.52 which equals this row, so the SF baseline is likely the single-scale model, and the SF "No multi-scale" row is "-" because the text says no substantial difference on synthetic (p.13). The multi-scale SceneFlow HITNet number is not reported.

| block | problem it solves | evidence | context | cost |
|---|---|---|---|---|
| Multi-scale prediction (coarse hierarchy) | textureless areas, global context | KITTI12: "No Multi-scale" 0.747 vs 0.484 EPE, bad3 3.62 vs 2.00; "4 Scales" 0.507 / 3.10 / 2.20 (Tab. 6). SF: not reported (no big difference, p.13) | real KITTI trained from scratch on 394 imgs; real scenes full of walls | multi-scale 0.66M params 36(61) GMac vs single 0.45M 52(92) at 1280x384 (Tab. 7; GMac in parens = both disparity maps with shared extractor). Note single-scale has MORE GMac in the table; as printed |
| High-res initialization (cost eval at 4x4 tiles of full-res feature map) | thin structures/edges lost by low-res init | "4x4x4 downsampled" init (cost volume 4x down in H, W, D): SF 0.561 vs 0.529, bad0.1 26.4 vs 24.0, bad3 3.15 vs 3.00; KITTI12 0.526 vs 0.484, bad2 3.16 vs 2.91 (Tab. 6). "16x16x8" SF 0.615, KITTI 0.536; "16x16x1" SF 0.651, bad3 3.62, KITTI 0.554 | both | init costs 0.25 ms (fused op) |
| Slant prediction (dx, dy; plane upsampling) | sub-tile surface variation, upsampling aliasing | No slant (dx=dy=0, bilinear upsampling): SF 0.548 vs 0.529, bad0.1 25.2 vs 24.0; KITTI12 0.513 vs 0.484, bad2 3.23 vs 2.91, bad3 2.18 vs 2.00 (Tab. 6). Text calls it "substantial drop" although EPE delta is only 0.019 / 0.029 px | both | 2 extra channels, trivial |
| Tile feature descriptor p | extra per-tile info (planarity, match quality) | No tile features: SF 0.538 vs 0.529; KITTI12 0.488 vs 0.484 (Tab. 6); text says "useful component" but KITTI delta is 0.004 px, within plausible noise | both | 13 of 16 channels of h |
| Warping (image warp cost in propagation) | sub-pixel precision | No warping: SF 0.588 vs 0.529, bad0.1 31.6 vs 24.0; KITTI12 0.602 vs 0.484, bad2 3.72 vs 2.91, bad3 2.54 vs 2.00 (Tab. 6) | both; largest single drop | warp is >100 elementwise ops per tile, needs a custom CUDA op: without it the runtime is x3 (p.16) |
| Model size (L/XL) | accuracy on large synthetic set | SF EPE 0.43 (L), 0.36 (XL) vs 0.529; KITTI12 0.490 / 0.492 (no gain, over-fitting small set) (Tab. 6) | L: 32 ch + 6 blocks last 3 steps: 54 ms; XL: 64 ch: 114 ms (p.15) | L 0.97M / 146 GMac; XL 2.07M / 396 GMac (Tab. 7) |
| Confidence w + multi-hypothesis selection | fuse init (high-res detail) with upsampled coarse hypothesis | **not ablated** | - | n=2 hypotheses at levels < M |
| Contrastive init loss | train features so matching is discriminative | **not ablated** | - | - |
| Data augmentation (asym. brightness/contrast, right-image patch replacement, y-offset, noise) | occlusion in-painting, mis-calibration, lighting | **not ablated** | - | - |
| Hierarchical GT max-pool | per-level supervision | **not ablated** | - | - |

### 4. Interactions & dependencies
- Warping needs the slant: tile -> 4x4 patch uses (dx,dy) (Eq. 5); slant is only useful together with warp-based cost refresh (the plane tells the warp how to deform the patch).
- High-res init and propagation are co-designed: propagation only refines +-1 px around the current plane (local cost volume), so it cannot recover a match missing from the init; that is why a low-res init is penalised in Tab. 6.
- The confidence w exists to arbitrate between hypotheses from init and from the coarser level; at the last single-hypothesis levels w is still supervised (A=inf there).
- Slant loss is gated by |d_diff|<B=1: only tiles already near the GT get slant supervision, so slant quality depends on prop-loss convergence.
- Descriptor p receives the best-match cost at init so it can encode confidence; it is never directly supervised, only through downstream losses.
- Takes custom CUDA ops (fused init, warping) to hit 20 ms; TF default is much slower (p.16). Portability to TensorRT is nontrivial.
- Training from scratch on tiny real sets works only with the heavy augmentation and small-lr/long schedule (p.12-13); Middlebury needs SceneFlow pretrain + fine-tune.

### 5. Losses (exact)
Total loss = sum over all levels l and pixels (x,y) of L_init + L_prop + L_slant + L_w, all weights 1 (Sec. 4, p.6). GT disparity is max-pooled to each level's resolution for init.
- Init (Eq. 9-11): psi(d) = (d - floor d) * rho(floor d + 1) + (floor d + 1 - d) * rho(floor d) (linear interpolation of integer costs to get subpixel GT cost). L_init(d_gt, d_nm) = psi(d_gt) + max(beta - psi(d_nm), 0), beta=1 (all experiments). d_nm = argmin of rho over d in [0,D] excluding the window [d_gt-1.5, d_gt+1.5] (the best non-match). l1 contrastive loss à la Hadsell et al. [18].
- Propagation (Eq. 12): L_prop(d,dx,dy) = rho_robust(min(|d_diff|, A), alpha, c), d_diff = d_gt - d_hat, d_hat obtained by expanding tiles to full-res with the plane equation. rho = Barron general robust loss (smooth-l1/Huber-like). SceneFlow alpha=0.9, c=0.1; all other experiments alpha=0.8, c=0.5 (p.12). Truncation A=1; for the final levels where only a single hypothesis exists the loss is applied to all pixels (A=inf).
- Slant (Eq. 13): L_slant = || [dx_gt - dx, dy_gt - dy] ||_1 * chi(|d_diff| < B), B=1. GT gradients are computed by robustly fitting a plane to d_gt in a 9x9 window centred on each pixel. The slant is upsampled to full res by nearest neighbour before the loss.
- Confidence (Eq. 14): L_w(w) = max(1-w,0) chi(|d_diff|<C1) + max(w,0) chi(|d_diff|>C2), C1=1, C2=1.5 (hinge: raise w when hypothesis within 1 px, lower w when beyond 1.5 px; in between no gradient).
- Deep supervision: every scale and every propagation iteration contributes (sum over l). No explicit edge/boundary, normal, or smoothness loss.

### 6. Training recipe
- SceneFlow: FlyingThings3D only (adding Driving/Monkaa hurt SF accuracy and Middlebury pretraining, p.12). Random crops 320x960, batch 8, max disp 320, Adam, 1.42M iterations, lr 4e-4 -> 1e-4 -> 4e-5 -> 1e-5 at 1M/1.3M/1.4M. Separate experiment (Fig. 5): lr 1e-4 for 200 epochs then 1e-5, no over-fit; lr 1e-3 gives lower error early (SF EPE 0.66 vs 0.85 after 10 epochs) but "smaller initial lr and longer is better". EPE protocol of PSMNet (exclude GT disparity > 192).
- KITTI: from scratch (no SceneFlow pretrain), batch 4, crops 311x1178, max disp 256; 400k its lr 4e-4, 8k at 1e-4, 2k at 4e-5. Ablation split 75/25; benchmark submission uses all 394 train images from KITTI 2012+2015 (p.12).
- ETH3D: KITTI 394 + half/quarter-res Middlebury V3 train + ETH3D train, same hyperparams, stopped at 115k its via 4-fold CV.
- Middlebury: HITNet L-like model, M=5, pretrained on FlyingThings3D 445k its, batch 8, crop 512x960, lr 4e-4, 1e-4/4e-5/1e-5 after 300k/400k/435k, then fine-tune 5k its at 1e-5 on 23 Middlebury14-perfectH images.
- Augmentation (p.13): symmetric brightness/contrast jitter [0.8,1.2], asymmetric [0.95,1.05]; random rectangle replacement in the right image with a crop from elsewhere in the right image (size [50,50]..[180,250]); Middlebury colour normalisation from AdaStereo; y-offset in [-2,2] px generated at H/64,W/64 and bilinearly upsampled; Gaussian noise var in [0,5] once per image.
- Hardware: Titan V for timing (p.16). Training hardware not stated.

### 7. Results
- SceneFlow finalpass EPE (Tab. 1): HITNet XL 0.36 @ 0.114 s; L 0.43 @ 0.054 s; EdgeStereo 0.74; LEAStereo 0.78; GA-Net 0.84; PSMNet 1.09; StereoNet 1.1 @ 0.015 s. Clean-pass XL: 0.31 EPE, bad0.1 15.6 (p.15).
- ETH3D (Tab. 2): EPE 0.20, bad1 2.79, bad2 0.80, 0.02 s (R-Stereo 0.18/2.44/0.44 @ 0.81 s; PSMNet 0.33/5.02/1.09 @ 0.54 s). 
- KITTI12 (Tab. 3): 2-noc 2.00, 2-all 2.65, 3-noc 1.41, 3-all 1.89, EPE-noc/all 0.4/0.5, 0.02 s. KITTI15: D1-bg 1.74, D1-fg 3.20, D1-all 1.98 (LEAStereo 1.40/2.91/1.65 @ 0.3 s; GANet-deep 1.48/3.46/1.81 @ 1.8 s; StereoNet 4.30/7.45/4.83).
- Middlebury v3 (Tab. 4): RMS 9.97, avg err 1.71, bad0.5 34.2, bad1 13.3, bad2 6.46, bad4 3.81, A50 0.40, 0.14 s. Text: rank first for bad0.5 and A50, second for bad1 and avgerr among end-to-end methods; LEAStereo 8.11 RMS, 1.43 avgerr, bad0.5 49.5, bad1 20.8, bad2 7.15, bad4 2.75, A50 0.53.
- Cross-domain (Tab. 5, SceneFlow-trained with aug, columns HITNet / CRL / iResNet / PSMNet / EdgeStereo): KITTI12 EPE 1.06 / 1.38 / 1.27 / 5.54 / 1.96, >3px 6.44 / 9.07 / 7.89 / 27.33 / 12.27; KITTI15 EPE 1.36 / 1.35 / 1.21 / 6.44 / 2.06, >3px 6.49 / 8.88 / 7.42 / 29.86 / 12.46 (iResNet/CRL beat HITNet on KITTI15 EPE).
- Params/compute (Tab. 7): single-scale 0.45M, multi-scale 0.66M, L 0.97M, Middlebury 1.62M, XL 2.07M; GMac at 1280x384: 52(92), 36(61), 146(235), 187 (450 at 1.57 Mpix), 396(735).
- Runtime: 19 ms per KITTI frame (0.5 Mpix) on Titan V; last 3 propagation steps 7.5 ms, extractor 6 ms, init 0.25 ms (p.16). ~107.5 ms/Mpix for Middlebury at D=160, about 109 ms/Mpix at D=1024 (small increase with disparity range). CPU TF default 3.3 s/Mpix (p.16).
- **Edge / boundary measurements (what is and is not measured).**
  - There is **no edge-region EPE, no foreground-boundary metric and no thin-structure metric** in the paper. Edge claims ("crisp edges", "recovers thin structures") are qualitative: Fig. 1, 3, 4 (KITTI vs GC-Net/RTSNet/GA-Net, "edge fattening artifacts" of competitors), Fig. 7 (Middlebury, per-image Bad 0.5), Fig. 8 (ablation, red boxes).
  - Indirect numeric proxies: (a) Middlebury Bad 0.5 and A50 (strict thresholds sensitive to boundary bleeding): Tab. 4 above; (b) SceneFlow bad0.1 in Tab. 6 (strict threshold: 24.0% baseline; 31.6% w/o warping; 26.4% with 4x4x4 down-sampled init); (c) KITTI15 D1-fg 3.20 (not best: LEAStereo 2.91) vs D1-bg 1.74.
  - Fig. 7 per-image Bad 0.5 vs best competitor: HITNet 17.3 vs EHCINet 15.9 (worse), 17.2 vs LocalExp 19.6, 39.7 vs LPU 43.3, 34.8 vs NOSS 35.8, 23.3 vs LE_PC 28.4. Mixed, not uniformly better.
  - Conditions under which the edge quality appears: full-resolution tile initialization (matching at 4x4 tiles of full-res feature map), plane-based upsampling, warp-based cost re-evaluation and a 1x1 full-res propagation step using full-res features; trained from scratch end-to-end with robust l1 on full-res d_hat; no separate RGB/luma guidance image: guidance is the learned U-Net feature itself. Ablation attribution to edges is only qualitative (Fig. 8), the EPE deltas are small (0.02-0.12 px).

### 8. Negative results & limitations
- Authors: needs GT depth; no self-supervision. Different datasets trained separately with slightly different architectures (p.8). Does not handle harsh lighting between pair views (DjembL, p.7). Larger models over-fit small KITTI (L/XL no KITTI gain, 0.490/0.492 vs 0.484, p.15). Single vs multi-scale: no SF gain (p.13).
- Adding Driving/Monkaa to SceneFlow training hurt accuracy (p.12).
- Mine: (1) Edge sharpness is claimed but never quantified; every "crisp edges" statement is a picture. (2) Ablation EPE deltas for slant (0.02 px), tile feature (0.009/0.004 px) are tiny and there are no seeds/variance; "substantial drop" language is not supported by the numbers for slant. (3) The SceneFlow "HITNet" baseline row in Tab. 6 appears to be the single-scale model (matches Tab. 7), so multi-scale SF is unreported. (4) Feature scale for warping at the last 3 steps is described inconsistently: "full-resolution feature maps for 4x4 tiles" vs later "4x4 tiles use 4X downsampled features, 2x2 use 2X, 1x1 use full-res" (p.15-16); I could not reconcile, treat the exact scale pairing as unspecified. (5) Per-level disparity scaling not stated. (6) The GMac ordering in Tab. 7 (single-scale 52 > multi-scale 36) is as printed; likely means the multi-scale uses fewer high-res steps. (7) KITTI results trained from scratch on 394 images with heavy augmentation: leaderboard-only evidence.
- Errors in repo summary (summaries/lightweight/HITNet_Tankovich_CVPR2021.md): says the init "evaluates a small set of disparity candidates"; paper evaluates ALL d in [0,D] per tile (exhaustively, never stored). Says "20 ms on a desktop GPU" correct. Summary's Pip-Stereo "93% D1 on DrivingStereo weather" cannot be verified from this PDF (not in the paper). "ETH3D bad 1.0 / 2.0 = 2.79 / 0.80" correct.

### 9. Relevance to OUR model
Our A09 (FusionStereoLite) borrows the tile/plane refinement lineage; D2/E3 is a frozen-predictor fusion head on top. Concretely:
1. **Confidence w with hinge supervision (Eq. 14)**: portable as the training signal for our SemanticCostGate: instead of only supervising final disparity, give each candidate a confidence supervised with C1=1/C2=1.5 and let the semantic gate modulate it. Insert at the candidate-selection step. Expected benefit: better calibrated gating (and a usable confidence output for downstream fusion); cost negligible (1 channel); risk: our head is trained only on 1,000 VKITTI pairs, hinge has dead zone [1,1.5] px.
2. **Init contrastive loss with best-non-match margin (Eq. 10-11)**: would sharpen the correlation volume itself, but A09 is frozen, so only usable if retraining A09 (a new series, not a fusion-head change). Expected: lower ambiguity at depth edges and thin objects; moderate.
3. **GT plane fit for slant labels (9x9 robust fit) and slant loss gated by |d_diff|<1**: portable to ClassResidual as a per-class slant/plane residual target (e.g. road/building/sky classes planar). Our G experiments did not test slant outputs; slant ablation in HITNet is only 0.02-0.03 px EPE, so do not expect EPE gains, expect smoother planar regions. Risk: none for cost, small for gain.
4. **Descriptor p / update-CNN input (Eq. 7) as the semantic insertion point**: concatenating class probabilities/YOLO features to a and to p is the lowest-friction way to get sem->disp with warp costs; but we have no tile features in a frozen A09 unless its refinement module exposes them. Check A09's tile/descriptor tensors before planning.
5. **Max-pooled GT per level**: our evaluation uses native-pixel; if training any coarse-level heads, HITNet's max-pool GT biases toward foreground (larger disparity); this is my inference, the paper does not discuss it, and it plausibly helps thin foreground retention.
6. **Edge lesson (critical)**: HITNet's edge quality comes from (i) full-res initialization, (ii) warp-based re-evaluation of matching cost with learned full-res features inside the loop, (iii) end-to-end training from scratch. A post-hoc RGB-guided residual or convex upsampler on a frozen predictor (our G/H) lacks (i)-(iii), so the negative G/H result is not contradicted by HITNet; HITNet also never shows that guidance or upsampling alone sharpens edges (no such ablation). Our G3 (stereo-warp correction) is the nearest analogue to HITNet's warp, but frozen and without the full-res learned features it did not help.
7. Our shared YOLO trunk provides only ~1/8-1/16 features, so a HITNet-style full-res (1x1 tile) propagation step would require adding a shallow full-res feature stem; cost modest (HITNet's last 3 steps are 7.5 ms of 19 ms on a Titan V; on RTX 3050/Jetson likely proportionally more) and needs the custom warp op for speed.
Novel combination status: slant/tile hypotheses + semantic gating of candidate hypotheses by confidence is already close to what D2/E3 does; the not-yet-done pieces are slant-aware class residual and w-hinge supervision.

### 10. Key quotes/equations worth citing
- "tile hypothesis ... h = [d, dx, dy, p]" (Eq. 1, p.3).
- "matches from the initialization module are provided ... to facilitate recovery of thin structures that cannot be represented at low resolution" (p.2).
- Matching cost Eq. 2; propagation augmented tile a=[h, phi(d'-1), phi(d'), phi(d'+1)] Eq. 7 (p.5).
- "Using ... tile resolution for disparity (cost volume is 4X downsampled in H, W and D) the accuracy substantially drops. This demonstrates the importance of our proposed fast high resolution initialization." (p.14)
- Confidence loss Eq. 14: L_w = max(1-w,0)chi(|d_diff|<C1) + max(w,0)chi(|d_diff|>C2), C1=1, C2=1.5 (p.6).
