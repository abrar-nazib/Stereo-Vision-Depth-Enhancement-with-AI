<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# Lite Any Stereo V2 (LAS2) card

Source: paper/reference_papers/lightweight/LiteAnyStereoV2_Jing_arXiv2026.pdf (17 pp: 14 body+refs, 3 author bios; no appendix/supplement in PDF). All pages read. V1 card: cards/LiteAnyStereo.md (V1 numbers below are quoted from it and from V2's own tables). No V2 summary in summaries/.

### 0. Meta
- Title: Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching. Jing, Zuo, Shen, Zhou, Potamias, Zafeiriou, Mikolajczyk, Deng (Imperial College London).
- Venue: arXiv 2606.24457v1, 23 Jun 2026. V1 [3] is CVPR 2026.
- Code: https://tomtomtommi.github.io/LiteAnyStereoV2/ (project page, abstract; repo URL not separately stated).
- Domain: general zero-shot (indoor, driving, in-the-wild). Datasets: train SceneFlow 35K, FallingThings 30K, FSD 1.1M, CREStereo 0.2M, VKITTI2 21K, TartanAir 0.31M, Dynamic Replica 0.14M (~1.8M synthetic); real unlabeled 0.5M (Tab. I: Flickr1024 1K, InStereo2k 2K, Holopix50K 49K, DrivingStereo 174K, SouthKenSV 113K, UASOL 156K). Eval: KITTI 2012/2015, ETH3D, Middlebury (half-res non-occ), DrivingStereo Weather.

### 1. Problem & failure modes targeted
- "Efficient stereo cannot generalize zero-shot" belief; most efficient models need dataset fine-tuning (p.1-2).
- MACs do not predict real latency on GPUs and edge devices; V1's hybrid 3D+2D aggregation is slow on edge (Sec. III-A, "Differences").
- Synthetic-to-real gap; noisy pseudo labels (occlusion, depth-discontinuity, sky) destabilize real-world adaptation (Sec. III-C).
- Failure modes admitted: strong reflections, transparent surfaces, severe illumination change, ambiguous geometry (Fig. 9, Sec. V). No explicit thin-structure/edge claim, only qualitative "sharper boundaries" (Figs. 7-8).

### 2. Pipeline by stage
- 2a Feature extraction: weight-shared ImageNet-pretrained FasterNet [59] (V1: MobileNetV2) producing pyramids at 1/4, 1/8, 1/16, 1/32, all upsampled to 1/4 by upsampling blocks following LightStereo [19]. Trainable (not frozen). Channel counts not stated. FasterNet chosen over MobileNetV2 because measured latency lower (101 vs 107 ms on Orin 8G, Tab. V) despite more MACs (47.6 vs 33.9 G).
- 2b Semantic/prior branch: n/a at inference (DepthAnything/other priors explicitly rejected as too heavy, Sec. III-A). SAM3 segmentation model [73] used ONLY offline to produce a sky mask for pseudo-label filtering (Sec. III-C).
- 2c Cost volume: correlation at 1/4, C(d,h,w) = 1/Nc <F_L(h,w), F_R(h,w-d)> (Eq. 1), d in [0, Dmax/4], Dmax=192 -> 48 levels. LAS2-H also builds group-wise volume C_g (Eq. 3, Ng groups, number not stated) and an all-pairs correlation volume C_a for lookup; averaging C_g over groups recovers Eq. 1 (Eq. 4).
- 2d Aggregation: 2D-ONLY U-Net (V1: 3D-then-2D). Cost volume (disparity as channels) goes through strided residual layers over 3 resolutions, then two transposed-conv upsampling layers with skip connections; residual layers are FasterNet blocks; keeps the attention from LightStereo [19]. Depth per size: S = 1,2,4 layers at 1/4,1/8,1/16; M = 4,8,16; L = 8,16,32 (encoder/decoder configs "4,8,16" and "8,16,32" for M/L; Sec. III-A).
- 2e Disparity: soft-argmax over d in [0, Dmax/4] at 1/4 res (Eq. 2).
- 2f Refinement (LAS2-H only): initialised from frozen-then-finetuned LAS2-M. d_0 = d_init. n=4 iterations. Per iter: G_k = Lookup(C_g, C_a, d_{k-1}) (Eq. 5); x_k = [Enc_g(G_k), Enc_d(d_{k-1}), d_{k-1}, c] (Eq. 6, c = context feature of left image); two ConvGRUs 1x1 and 3x3 (Eqs. 7-8) fused by spatial attention map A (selective ConvGRU from Selective-IGEV [8]); h_k = A*h^s + (1-A)*h^l (Eq. 9); d_k = d_{k-1} + Head_d(h_k) (Eq. 10). Hidden and context 64 channels; Enc_g, Enc_d two conv layers each. Replaces IGEV-style 3D regularization.
- 2g Upsampling: convex upsampling 1/4 -> full for all variants, applied to every iteration output in H (Fig. 3, Sec. III-A/B). NOT ablated in V2.
- 2h Fusion points: (i) LAS2-H: context features and cost-volume lookups enter the GRU (feature/refinement stage, concat, left-image -> disparity). (ii) Training-time only: feature-alignment L_feat (Stage 2; teacher->student), pseudo-label distillation from FoundationStereo (Stage 3; loss-only), SAM3 sky mask and edge/LR masks gating the loss. No semantic cue at inference.

### 3. Block -> problem -> evidence table
Metric order for ablation tables: K.12 D1 / K.15 D1 / ETH3D Bad1.0 / Middlebury Bad2.0 (lower is better). Tabs. V-X trained on LAS2-M backbone, 150K iterations, 1.4M synthetic subset, no aug (Tab. V caption) unless noted. Latency: Orin NX 8G.
| block | problem it solves | evidence (ablation delta with ref) | context/conditions | cost |
|---|---|---|---|---|
| 2D-only aggregation (vs V1 3D+2D) | edge-device latency | LAS2-M vs LAS (V1): Orin 101 vs 193 ms, H200 8.1 vs 12.7 ms (Tab. II/XI); K.12 D1 2.88 vs 3.04, K.15 3.61 vs 3.87, ETH 2.59 vs 3.53, Mid 5.47 vs 7.51 (Tab. II; includes new training) | architecture + training changed together; not separated | pure 2D |
| Cost agg block FasterNet | accuracy-latency | vs ConvNeXt 4.92/4.86/7.99/9.76 (162 ms, 31.2G), MobileNetV2 4.81/4.85/6.25/10.73 (118 ms), MobileNetV3 4.54/4.85/5.59/10.41 (145), EfficientNetV2 5.62/4.52/8.76/10.28 (123), FasterNet 4.49/4.57/5.62/9.49 (107 ms, 33.9G), GhostNet 4.82/ (K.15 not legible)/5.64/11.53 (185) (Tab. V) | feature extractor fixed MobileNetV2 | FasterNet latency lowest |
| Feature extractor FasterNet | latency | MobileNetV2 4.49/4.57/5.62/9.49 (107 ms, 33.9G); FasterNet+1x1 conv 5.18/4.91/5.16/10.98 (95 ms, 35.6G); FasterNet native channels 4.84/4.77/5.37/10.49 (101 ms, 47.6G) (Tab. V) | | accuracy slightly worse than MNv2 on K.12/Mid; ~6% faster. Authors choose FasterNet. |
| Iterative LAS2-H (4 iters, from LAS2-M) | accuracy ceiling | H vs M: K.12 2.64 vs 2.88, K.15 3.31 vs 3.61, ETH Bad1 1.83 vs 2.59, Mid 3.71 vs 5.47 (Tab. II) | | Orin 344 vs 101 ms (3.4x), H200 15.1 vs 8.1 ms |
| Stage 1 only | synthetic baseline | 4.21/4.66/4.25/7.95 (Tab. IX, full train set) | | |
| Stage 2 self-distillation | domain-invariant feats | Tab. VII (1.4M): none 4.84/4.77/5.37/10.49; data aug 4.15/4.97/5.94/9.06; self-dist 3.85/4.78/4.89/8.83. Tab. IX full: 3.59/4.65/4.67/6.91 | ETH Bad1 worsens in full run (4.25->4.67) | training only |
| Teacher choice in Stage 2 | | Tab. VIII: EMA 4.46/5.25/6.84/9.64; hard copy 4.01/4.85/6.66/8.83; fixed 3.85/4.78/4.89/8.83 | fixed best | |
| Stage 3 pseudo-label training | sim-to-real | Tab. IX: 2.88/3.61/2.59/5.47 (largest single gain) | 0.5M real, teacher FoundationStereo | training only |
| Valid mask: none | | 2.91/3.66/3.01/5.88 (Tab. VI) | | |
| + M_LR | geometric inconsistency | 2.91/3.73/2.76/5.29 | | |
| + M_LR+M_sky | sky ambiguity | 2.86/3.62/2.62/5.68 | mixed, kept for deployment | |
| + M_LR+M_sky+M_edge (final) | disparity discontinuities not backed by image edges | 2.88/3.61/2.59/5.47 (underlined default, matches Tab. IX Stage 3; Tab. VI) | | |
| + M_rgb | | 2.89/3.63/2.85/5.41 (K.15, ETH worse) | not used | |
| Error clamp tau | high-error pixels dominate grads | w/o 3.17/3.92/3.22/6.21; tau=5 2.89/3.59/2.54/5.57; tau=10 2.88/3.61/2.59/5.47 (default); tau=20 2.98/3.70/2.88/5.98 (Tab. VI) | | |
| Feature alignment in Stage 3 | | w. 2.93/3.65/2.66/5.99 vs w/o 2.88/3.61/2.59/5.47 | removed | extra fwd passes |
| Teacher: FS vs FS+MS+S2M2 | | FS 2.88/3.61/2.59/5.47; multi-teacher 2.94/3.35/2.10/7.53 (K.15, ETH better; K.12, Mid worse) | | higher pseudo-label cost |
| Extra data Stereo4D (1.4M) / Xperience (3.6M) | | base 2.88/3.61/2.59/5.47; +Stereo4D 2.93/3.74/2.94/6.52; +Xperience 4.37/5.08/2.77/9.17 (Tab. VI) | quality > quantity | |
| Generality of 3-stage recipe | | LightStereo-M: S1 4.34/5.27/6.68/10.29 -> S2 3.80/4.62/5.44/8.96 -> S3 2.74/3.74/2.73/7.08 (V1 recipe 3.35/4.14/4.22/9.85); BANet-2D: 4.34/4.78/7.71/10.54 -> 3.87/4.80/4.86/9.54 -> 2.69/3.66/2.61/7.81 (V1 3.28/4.08/4.05/10.30) (Tab. X) | | |
| Convex upsampling, correlation volume, soft-argmax, group-wise volume (H), GRU details, n=4 | | not ablated | | |
| Variant sizes S/M/L | capacity | Orin: 81/101/166 ms; H200 6.6/8.1/11.4 (Tab. II) | | |

Note: Tab. VI cell values are read from the page image; 0.01-level read errors possible.

### 4. Interactions & dependencies
- LAS2-H depends on LAS2-M weights (initialised from LAS2-M, M frozen for first 100K steps of Stage 1, Sec. IV-B); the 2D aggregation head is what makes reuse possible (V1 3D design could not feed an IGEV-style loop cheaply).
- Stage 3 gains depend on filtering AND clamping: clamp alone removes the worst effect (3.17->2.88 K.12) and LR mask helps Mid/ETH.
- Teacher must stay fixed in Stage 2; feature alignment in Stage 3 gives no gain and costs compute.
- More data is not better unless it is clean/high-res (Stereo4D 18M low-res, HRWSI rectification artifacts, SCOD narrow).
- Sky mask hurts Mid (small sky), kept for real-world deployment: a deployment-motivated component that is not supported by benchmark numbers.
- Multi-teacher ensemble helps KITTI 2015 / ETH3D, hurts KITTI 2012 / Middlebury.
- MACs vs latency: ConvNeXt has lowest MACs (31.2G) but highest latency (162 ms) on Orin (Tab. V).

### 5. Losses
- L_disp = smoothL1(D - D_gt) (Eq. 11).
- LAS2-H: L_iter = L_disp + sum_{k=1..n} gamma^{n-k} ||D_k - D_gt||_1, gamma = 0.9, n=4 (Eq. 12).
- Stage 2 total: L_disp + L_feat; L_feat = 1 - 1/HW sum_i cos(F_i, F'_i) (Eq. 13); weight of L_feat not stated.
- Stage 3: L_clamp = sum M_valid * min(L_disp, tau_clamp) (Eq. 16), tau = 10 (default, Tab. VI); M_valid = M_LR * M_edge * M_sky.
  - M_LR = 1(|D_L - W(D_R, D_L)| < tau_LR), tau_LR = 1 px (Eq. 14).
  - M_edge = 1 - 1(||grad D_L||_1 > q_d) * (1 - 1(||grad I_L||_1 > q_I)) (Eq. 15), both quantile thresholds 90th percentile per image, 3x3 erosion of the resulting mask.
  - M_sky from SAM3 [73] segmentation. Sky pixels set to 0.
- Pseudo-labels from FoundationStereo [15] (dense; used even where sparse GT exists).

### 6. Training recipe
- PyTorch, H200 GPUs, total batch 128, AdamW, one-cycle LR peak 2e-4, random crop 384x768, Dmax = 192 (Sec. IV-B).
- Steps: Stage 1 200K, Stage 2 50K, Stage 3 200K (V1: 150K/50K/100K, batch 176, A100).
- Stage 1: from scratch on 1.8M synthetic, NO augmentation. LAS2-H: init feature extraction + aggregation from LAS2-M, frozen first 100K steps, then all fine-tuned.
- Stage 2: teacher (clean) and student (perturbed) both init from Stage 1 weights; teacher fixed. Perturbations: large color jitter (brightness, contrast, saturation, hue), optional 5x5 Gaussian blur sigma in [0.1, 2.0], gamma in [0.7, 1.5], applied symmetrically or asymmetrically with small probability. (This was in V1's supplement, now in main text.)
- Stage 3: 0.5M real pairs, FoundationStereo pseudo labels with filtering and clamp; no self-distillation in this stage.
- Baselines retrained with official code on the same synthetic data (Sec. IV-A). Fast-FoundationStereo uses the "20-30-48" 4-iteration variant.
- Ablation protocol (Tab. V): 150K iters, 1.4M subset, no aug; Tab. VI uses the stage-3 setting; Tab. VII/VIII 1.4M synthetic.

### 7. Results
Zero-shot (Tab. II; latency at 384x1248, torch.compile disabled; K12 D1/EPE, K15 D1/EPE, ETH Bad1/EPE, Mid Bad2/EPE, H200 ms, Orin ms):
- LAS (V1): 3.04/0.79, 3.87/0.99, 3.53/0.32, 7.51/0.94, 12.7, 193.
- LAS2-S: 2.97/0.78, 3.83/0.99, 3.34/0.32, 8.87/1.17, 6.6, 81.
- LAS2-M: 2.88/0.74, 3.61/0.95, 2.59/0.27, 5.47/0.77, 8.1, 101.
- LAS2-L: 2.57/0.71, 3.38/0.94, 1.83/0.23, 5.28/0.77, 11.4, 166.
- Fast-FoundationStereo (iter.): 2.90/0.75, 3.66/0.94, 1.83/0.22, 3.73/0.63, 27.3, 918.
- LAS2-H: 2.64/0.69, 3.31/0.90, 1.83/0.22, 3.71/0.61, 15.1, 344.
- Lite-CREStereo++: 23.1/482 ms; RT-MonSter++: 36.3/763 ms; LightStereo-S 7.1/89 ms but K.12 D1 4.54.
- FoundationStereo (accurate): 2.51/0.67, 2.83/0.86, 0.49/0.14, 1.12/0.37, 292 ms H200, OOM on Orin.
DrivingStereo weather (Tab. III) overall D1/EPE: LAS2-S 7.23/1.63, M 7.67/1.70, L 7.98/1.70, H 7.44/1.67; LAS 8.74/1.80; FoundationStereo 10.71/2.22; MonSter++ 5.21/1.77. (Note: S is best feed-forward and better than M/L here.)
KITTI leaderboard (Tab. IV, fine-tuned): LAS2-M K15 D1-bg/fg/all 1.33/3.00/1.61, K12 3-noc/3-all 1.13/1.51; latency 8.1 H200 / 101 Orin.
Latency (Tab. XI, ms, 384x1248): RTX 4090 / A5000 / A100 / H200 | Orin NX 8G 10W / 20W / MAXN
- LAS2-S 11.5/16.3/22.9/6.6 | 181/144/81
- LAS2-M 16.8/21.4/29.2/8.1 | 225/179/101
- LAS2-L 23.2/29.2/41.8/11.4 | 372/283/166
- LAS2-H 26.2/37.2/65.9/15.1 | 689/594/344
- LAS (V1) 21.4/26.7/46.6/12.7 | 469/345/193
- LightStereo-S 16.2/19.9/23.6/7.1 | 229/155/89
- Fast-ACVNet+ 23.3/31.5/56.6/14.3 | 469/380/221
Real-device: only Orin NX 8G (three power modes), framework not stated (appears PyTorch eager; torch.compile disabled; no TensorRT/FP16 statement). Params and peak memory: not stated. Largest Fast-FoundationStereo variant OOM on Orin.

### 8. Negative results & limitations
- Authors: gap to FoundationStereo/MonSter-class models; limited high-quality real data; failures on reflection/transparent/illumination (Fig. 9).
- Tried and dropped: multi-teacher, extra data (Stereo4D/Xperience, hurt), RGB consistency mask (worse), Stage-3 feature alignment (no gain), self-distillation EMA/hard copy (worse), FasterNet+1x1 (accuracy loss), tau=20.
- Mine: (1) architecture and training recipe changed together in V1->V2, so the 2D-only change is not isolated on accuracy; only latency is cleanly attributable (Tab. XI). (2) Latency in PyTorch eager at 384x1248 only; no TensorRT/FP16 numbers, so deployed Orin latency is unknown (could be far lower). (3) Stage-3 teacher is FoundationStereo, so zero-shot rank depends on a heavy-prior teacher (same caveat as V1); LAS2 beating FoundationStereo on DrivingStereo weather partly reflects in-domain real data (DrivingStereo non-weather is in Stage 3). (4) Eval protocol is "re-evaluated by us", baselines retrained; fair but not independently replicated. (5) Tab. VI intermediate rows: no ablation of V2's own M_edge versus alternatives under a common seed; deltas of 0.03-0.1 are within plausible noise. (6) Abstract claims 13.7% error reduction; I did not verify the aggregate (Sec. I text only).
- V1 -> V2 inconsistencies noted: V1 said ConvNeXt 2D layer + small 3D; V2 uses FasterNet and no 3D. Hardware changed A100 -> H200, steps and batch changed.

### 9. Relevance to OUR model
Our A09 (correlation at ~1/4 or 1/8, shallow head, tile refinement) is already in the LAS family: shared-weight encoder -> correlation -> 2D head -> upsample.
- Directly portable, zero inference cost: the pseudo-label filtering recipe (M_LR + M_edge + M_sky, clamped loss tau = 10) for training the fusion head (SemanticCostGate/ClassResidual) on UNLABELED real stereo (KITTI raw, DrivingStereo) with an offline FoundationStereo teacher. Our weak spot is VKITTI-only training (F-series KITTI EPE 2.59). Expected benefit: largest in their ablations was Stage 3 (K.12 4.21 -> 2.88 D1, Tab. IX). Risk: our head is small and the trunk and A09 frozen, so the gain ceiling is capped by frozen features; pseudo-labels from a teacher bias the head toward teacher errors; the semantic decoder is trained on VKITTI only so real-domain class errors could poison the gate. A matched no-semantics control is needed to attribute gain to semantics.
- Already tried by us: convex upsampling (LAS2 never ablates it; our G2/H2 found no benefit). LAS2's own claims of sharper boundaries are qualitative only (Figs. 7-8).
- Latency lessons: MACs do not predict latency (ConvNeXt 31G MACs = 162 ms vs FasterNet 34G = 107 ms on Orin 8G, Tab. V); measure on device. Iterative refinement costs 3.4x on Orin (344 vs 101 ms): avoid a GRU loop in the live fusion head; our gate + single residual is the right shape.
- Orin NX 8G power modes: LAS2-M goes 101 ms (MAXN) -> 225 ms (10W). Budget for the lowest power mode, not MAXN. The M variant at 384x1248: 8.1 ms on H200 vs RTX 4090 16.8 ms and A5000 21.4 ms; a 3050 is slower than A5000, so expect >25 ms for LAS2-M-class cost, no TensorRT.
- Self-distillation with perturbed student (Stage 2): modest, trunk is frozen so gains would be head-only. Low priority.
- Novel combination: pseudo-label distillation with a SEMANTIC-conditioned gate head on a frozen shared-detector trunk; LAS2 has no semantic cue. Not done in this paper.

### 10. Key quotes/equations worth citing
- "MACs alone do not reliably reflect real inference speed on modern GPUs and edge devices" (Differences, p.3).
- "FasterNet ... slightly higher MACs but ... faster inference in practice" (Sec. III-A, p.3-4).
- L_clamp = sum M_valid min(L_disp, tau_clamp) (Eq. 16, p.8). tau = 10.
- Eq. 15 edge mask: disparity gradient strong without corresponding image gradient is marked unreliable (p.8).
- "data quality and diversity are more important than its raw scale" (p.8-9).
- Tab. XI: LAS2-M Orin NX 8G 225/179/101 ms at 10W/20W/MAXN (p.13).
