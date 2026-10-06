<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# ViTAS card

Page refs are PDF pages (13 pages, no supplement; main 1-11, refs 12-13). No existing summary of this paper in paper/reference_papers/summaries/.

### 0. Meta
- Title: Playing to Vision Foundation Model's Strengths in Stereo Matching (method names: ViTAS = ViT Adapter for Stereo; ViTAStereo = ViTAS + cost-volume back-end)
- Authors: Chuang-Wei Liu, Qijun Chen, Rui Fan (Tongji University)
- Venue/year: NOT stated. PDF is "arXiv:2404.06261v1 [cs.CV] 9 Apr 2024", IEEE-style layout (Index Terms, "Senior Member, IEEE") but no journal named. Treat as arXiv preprint; do not cite a venue.
- PDF: paper/reference_papers/fusion/ViTAS_Liu_arXiv2024.pdf
- Code: not stated (leaderboard links to cvlibs only, p. 7 footnote).
- Domain: driving (KITTI) + indoor (Middlebury, ETH3D); intelligent vehicles.
- Datasets: SceneFlow, Virtual KITTI (cited as [43] = SceneFlow paper), KITTI 2012/2015, Middlebury, ETH3D (Sec. IV-A).

### 1. Problem & failure modes targeted
VFM features (SAM/DINOv2/DepthAnything) are not distinct enough for similarity measurement; ViT gives single-resolution tokens (1/16) while stereo wants 1/32..1/4 pyramids (Sec. I, III-A). Targets texture-less regions, small objects/boundaries, occlusion (Sec. IV-D), scale ambiguity of cost-volume-free VFM networks like CroCo-Stereo (Sec. IV-E).

### 2. Pipeline by stage
- 2a Feature extraction: DINOv2 (N = 24 transformer blocks, so ViT-L; size inferred from N=24, not named) ViT, shared across left/right ("twin architecture, weight-sharing"). Input resized by 14/16 so patch-14 tokens land at 1/16 resolution. Tokens taken from the output of the last block of each of four groups: T_6, T_12, T_18, T_24 (Sec. III-A). Frozen except the last 5 or 6 blocks (inconsistent: Sec. IV-A says "excluding the last six ViT encoder blocks are frozen"; Sec. IV-C says "unfreeze the last five VFM blocks", p. 5-6).
- Adapter (ViTAS) outputs pyramid F_k, k=0..3 at 1/2^{5-k} i.e. 1/32,1/16,1/8,1/4. Channel widths "reduced" but not stated.
  - SDM (spatial differentiation module): per token group, 1x1 conv, 3x3 stride-2 conv, transposed conv chains (Fig. 1a) to build initial pyramid D_0..D_3 (T_24 -> coarsest 1/32, T_6 -> finest 1/4). Deep tokens -> low-res, shallow tokens -> high-res (Sec. IV-B text p. 4).
  - Then hierarchically: F_0 = CAM(D_0^L, D_0^R); for i=1..3: M_i = PAFM(F_{i-1}, D_i); F_i = CAM(M_i^L, M_i^R) (Alg. 1). Per sub-network: 1 SDM, 4 CAMs, 3 PAFMs.
  - PAFM (patch attention fusion module, Fig. 2, Eq. 1-4): local patch attention (pixel-to-2x2-patch similarity, Q from higher-res D, K/V_F from lower-res F, softmax over the 4 patch positions, w_L) + quasi-global attention (squeeze-and-excitation-style: spatial weights w_S via MLP of pooled Q over the patch dim, channel weights w_C from pooled K, w_G = sigmoid(w_C + w_S)); W = w_G * V_F + 4(w_L - w_L*w_G) * V_D; M = D + Reshape(LN(MLP(W))). Complexity O(N) per pixel patch instead of O(N^2).
  - CAM (cross-attention module): 2 attention blocks, each self-attention then cross-attention with the other view (Fig. 1c); 2 blocks chosen from Fig. 3a.
- 2b Prior branch: the VFM is the feature extractor itself (DINOv2, partially fine-tuned). No separate semantic branch.
- 2c Cost volume: back-end of IGEV-Stereo unchanged (group-wise corr + geometry encoding volume); Sec. IV-A.
- 2d Aggregation: IGEV 3D hourglass (not restated).
- 2e Disparity: IGEV (soft-argmin init) - not restated in this paper.
- 2f Refinement: IGEV multi-level GRU, not modified.
- 2g Upsampling: IGEV convex upsampling (not stated).
- 2h FUSION POINTS: the VFM cue enters at the FEATURE stage only (replaces the CNN feature extractor, feeds cost volume build-up). Operators: multi-scale re-assembly of intermediate ViT tokens (SDM), attention-weighted additive fusion (PAFM), left-right cross-attention (CAM). Bidirectional only in the sense of left<->right. Resolutions 1/32 .. 1/4. No cost-volume-stage, refinement-stage, or loss-level injection.

### 3. Block -> problem -> evidence table
Tab. I: pre-trained SceneFlow + full Virtual KITTI, 50 epochs, evaluated on Flying3D test set (in-domain, NOT zero-shot): EPE(px) / D1-all(%) / runtime(s).
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| none (VFM tokens only; row 8) | baseline | 0.435 / 1.394 / 0.261 | partially unfrozen DINOv2 (last blocks), IGEV back-end; what the "none" row feeds the volume with is not described | |
| SDM only | 1/16 single-scale tokens -> pyramid | 0.427 / 1.389 / 0.262 (-1.8% EPE) | | +0.001 s |
| PAFM only | fine-detail multiscale fusion | 0.383 / 1.266 / 0.265 (-11.9%) | | +0.004 s |
| CAM only | stereo context | 0.417 / 1.348 / 0.272 (-4.1%) | | +0.011 s |
| SDM+PAFM | | 0.362 / 1.281 / 0.266 | | |
| SDM+CAM | | 0.388 / 1.349 / 0.273 | | |
| PAFM+CAM | | 0.351 / 1.206 / 0.275 | | |
| SDM+PAFM+CAM (full) | | 0.334 / 1.109 / 0.278 | | |
| PAFM vs SDFA [21] vs VAF [22] (rows 9-10 + Tab. II) | alternative fusion | SDM+SDFA+CAM 0.351/1.155/0.274; SDM+VAF+CAM 0.335/1.122/0.287; PAFM 0.334/1.109/0.278. VAF is within 0.001 px of PAFM in EPE | | params 0.22 M vs 1.38 M vs 1.31 M; memory 118 vs 98.3 vs 360 MB (Tab. II) |
| # CAM attention blocks 0-4 (Fig. 3a) | | EPE falls with more blocks (approx. 0.362, 0.341, 0.334, 0.332, 0.331, read from plot, not tabulated); learnable params rise ~linearly; 2 chosen | | |
| # unfrozen VFM blocks 0-7 (Fig. 3b) | frozen vs finetune | EPE falls monotonically from ~0.365 (0 unfrozen) to ~0.332 (7); params grow to ~80 M (plot); 5 (or 6) chosen | | |
| w/ vs w/o ViTAS across 3 back-ends (Tab. V) | generalisation | see 7 | fine-tuned on KITTI-train or Midd-train, tested on held-out KITTI-eval, Midd-eval, ETH3D | |
| Cost-volume-free (CroCo-Stereo) with this VFM (Tab. VI) | is the cost volume expendable | see 4 | | |
| Fully frozen VFM | | not ablated beyond "0 unfrozen blocks" point of Fig. 3b | | |
| DINOv2 vs DepthAnything/SAM as VFM | | not ablated | | |
| Token-layer choice (blocks 6/12/18/24) | | not ablated | | |

### 4. Interactions & dependencies
- Modules are "modular independent": removing any one degrades EPE roughly as much as using it alone (Sec. IV-C, p. 6).
- PAFM is the most influential (-11.9% alone).
- ViTAS must be paired with a cost-volume back-end for generalisation: CroCo-Stereo (cost-volume-free ViT decoder), even with this VFM backbone, shows scale ambiguity and fails zero-shot (Tab. VI: after KITTI fine-tune, ETH3D EPE 51.5 -> 50.7 modified; Midd-train fine-tune ETH3D 50.7 -> 62.9 EPE; D1 85-97%). Cross-attention alone does not solve scale ambiguity (Sec. V).
- Gains are not universal: CREStereo + ViTAS is worse on Midd Eval D1 after KITTI fine-tune (16.1 -> 16.6) and on ETH3D EPE after Midd fine-tune (28.6 -> 29.1) (Tab. V).
- Unfreezing more VFM blocks monotonically improves in-domain EPE (Fig. 3b) - the opposite direction from FoundationStereo (unfreeze hurts), with different data scale and architectures; not a controlled comparison.

### 5. Losses
Not stated: "loss function, learning rate, optimizer identical to the settings in their publications" (IGEV, GMStereo, CREStereo, CroCo-Stereo) (Sec. IV-A, p. 6). No adapter-specific loss.

### 6. Training recipe
4x RTX 4090 (24 GB). Random crop 320x720, colour/rescale/erase aug. Ablation: SceneFlow train + full Virtual KITTI, 50 epochs, select by Flying3D test. Final KITTI: pre-train on combined five datasets 100 epochs, then KITTI train 200 epochs (Sec. IV-A). Generalisation: fine-tune on KITTI Train (half of KITTI train) or Midd Train (MiddEval3 held out), evaluate on KITTI Eval, Midd Eval, ETH3D (all of it). VFM frozen except last 5-6 ViT blocks. Seeds not stated.

### 7. Results
- KITTI 2012 (Tab. III): 2-noc 1.46, 2-all 1.80, 3-noc 0.93, 3-all 1.16, 4-noc 0.71, 4-all 0.87, 5-noc 0.58, 5-all 0.71; beats StereoBase (1.54/1.95/1.00/1.26/0.76/0.97/0.62/0.80) on all columns. IGEV-Stereo row reduced by up to 24.47% (text).
- KITTI 2015 (Tab. IV): D1-bg/fg/all all pixels 1.21/2.99/1.50; non-occluded 1.12/2.90/1.41. StereoBase (1.28/2.26/1.44; 1.17/2.23/1.35) is better on D1-fg and D1-all - ViTAStereo is second on D1-all, as the abstract admits ("second-best on KITTI 2015").
- Generalisation w/o -> w/ ViTAS (Tab. V; EPE px / D1-all %; KITTI-ft: KITTI eval, Midd eval, ETH3D):
  - IGEV: 0.55/1.71 -> 0.49/1.36; 5.27/16.8 -> 3.05/10.9; 1.15/5.23 -> 1.01/5.08. Midd-ft: 1.07/5.27 -> 0.87/3.45; 2.14/10.8 -> 1.34/6.00; 4.20/5.49 -> 2.65/3.68.
  - GMStereo: KITTI-ft 0.59/1.82 -> 0.56/1.62; 2.96/14.1 -> 2.65/12.4; 1.08/8.99 -> 0.82/2.30. Midd-ft 1.14/6.14 -> 1.02/4.41; 2.65/13.3 -> 1.99/10.2; 1.72/10.1 -> 0.67/4.34.
  - CREStereo: KITTI-ft 0.70/2.32 -> 0.66/2.02; 5.20/16.1 -> 4.99/16.6; 3.45/18.8 -> 1.75/14.8. Midd-ft 1.18/6.24 -> 1.07/4.91; 3.98/14.4 -> 2.59/12.7; 28.6/35.0 -> 29.1/25.9.
- Runtime ~0.278 s (Tab. I, per frame; image size/GPU for runtime not stated). Total parameters not stated; PAFM 0.22 M.
- Caveat: "generalisation" is fine-tune on one small real dataset then test on another, not zero-shot from synthetic.

### 8. Negative results & limitations
- Cross-view-attention-only (cost-volume-free) fails to generalise (Tab. VI) - authors' main negative finding.
- CAM "remains unchanged and still requires substantial computational and memory resources" (Sec. V); no param/FLOP total for the whole network, no latency on edge hardware.
- My critique: ablations are in-domain (Flying3D test), single-run, differences of 0.001-0.017 px EPE between PAFM and VAF are within noise; the "none" row is under-specified; five vs six unfrozen blocks inconsistency; "32.8% lower memory than VAF" text is wrong arithmetic (118 MB vs 360 MB = 67.2% lower, 32.8% of); SDFA actually uses less memory (98.3 MB). Pre-train on SceneFlow+KITTI sets in the final benchmark mixing makes Tab. III/IV not zero-shot. DINOv2 token layers 6/12/18/24 and image 14/16 resize not ablated. Generalisation baselines CREStereo mixed results show adapter benefit depends on back-end.
- Fine-tuning 5-6 VFM blocks means it is NOT a frozen-encoder adapter strictly; it is a partially-unfrozen one with 0 unfrozen ~0.365 EPE (Fig. 3b), i.e. frozen+adapter still beats no-adapter 0.435 baseline only if row 8 is the frozen setting (not stated).

### 9. Relevance to OUR model
- Closest to our structure (frozen CNN trunk -> adapter -> cost volume), but ViTAS adapter is ViT-specific (reassembling 1/16 tokens into a pyramid). Our YOLO trunk is a CNN with native 1/8, 1/16 features, so SDM is unnecessary.
- Portable: PAFM-style cheap local patch attention + SE-weighted merging (0.22 M params, 118 MB) as a coarse-to-fine feature fuser when combining semantic features at 1/16 with stereo features at 1/8; it was the biggest single gain (-11.9% EPE alone) with +0.004 s. Real risk: evidence is in-domain on Flying3D with a ViT; no reason to expect gain from a CNN trunk already at multiple scales.
- CAM (self+cross attention left-right) is memory-heavy: not suitable for 3050/Jetson.
- Strong support for a design choice we already follow: keep a cost volume (Tab. VI shows dropping it destroys cross-domain scale), do not use the foundation model as a direct disparity regressor.
- Fusion stage evidence: only feature stage tested, so no support for choosing cost-gate vs feature fusion.

### 10. Key quotes/equations
- "stereo matching networks relying solely on cross-attention mechanism have limited generalizability, primarily due to the absence of cost volumes" (p. 2).
- Eq. 3: W = RepPad(w_G * V_F) + 4(RepPad(w_L) - RepPad(w_L) * RepPad(w_G)) * V_D; Eq. 4 M = D + Reshape(LN(MLP(W))) (p. 5).
- Tab. II: PAFM 0.22 M params / 118 MB vs SDFA 1.38 M / 98.3 MB vs VAF 1.31 M / 360 MB (p. 6).
