<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# MobileStereoNet card

Page refs are PDF pages 1-10 (printed 2417-2426). The PDF ends at the references: the appendix the paper cites (detailed architecture, extra ablation tables, p.5-6 "tables available in the appendix") is **not in this file**, so channel-level architecture details and the training recipe beyond the main text are "not stated" here.

### 0. Meta
- Title: MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching
- Authors: Shamsafar*, Woerz*, Rahim, Zell (Univ. Tuebingen). WACV 2022.
- PDF: paper/reference_papers/lightweight/MobileStereoNet_Shamsafar_WACV2022.pdf
- Code: https://github.com/cogsys-tuebingen/mobilestereonet
- Domain: driving + synthetic; embedded/mobile targeting (parameters/MACs/model-size, no runtime). Datasets: SceneFlow finalpass, KITTI 2015 (159/40 split; benchmark submission).

### 1. Problem & failure modes targeted
- Memory/compute of 3D stereo networks (OOM on moderate GPUs, embedded platforms) (p.1).
- Weak accuracy of 2D (3D-cost-volume) models vs 3D models, addressed by a learnable cost-volume construction.
- No edge/thin-structure/occlusion specific mechanism. Only a qualitative statement that 3D convs preserve fine details better (p.7).

### 2. Pipeline by stage
- **2a Feature extraction.** Shared ResNet-like backbone (GwcNet) with MobileNet-V1 blocks (v1) replacing the 3x3 convs (and the first three convs replaced by v2, t=3) (p.5). 2D baseline: 320 x H/4 x W/4 features then channel reduction by four 1x1 convs 320->256->128->64->32 (Fig. 4). Pretrained/frozen: no.
- **2b Semantic/prior branch.** n/a.
- **2c Cost volume.**
  - 2D model: **Interlacing cost volume**, 3D data of size (dmax/4) x H/4 x W/4 (dmax=192) (Fig. 4). For each disparity, left features f_L(.,x,y) and shifted right features f_R(.,x-d,y), each C=32 channels, are interleaved along channels to 2C x H x W, unsqueezed to 1 x 2C x H x W, and passed through 3D convs: first kernel 2i x 3 x 3 stride (2i,1,1) so each kernel sees i channels from each view (non-overlapping groups); Fig. 5 (i=4): 16 filters, 8x3x3, stride (8,1,1) -> 8 slices; 32 filters, 4x3x3, stride (4,1,1) -> 2; 16 filters, 2x3x3, stride (2,1,1) -> 1; then one 2D conv -> H x W (filter counts read from Fig. 5; text says the two later layers have "double and same number of kernels" so ambiguous). Output spatial resolution unchanged (p.5). Because the channel axis is collapsed per disparity, the volume is 3D.
  - 3D model: Gwc40 group-wise correlation, 40 x (dmax/4) x (H/4) x (W/4) (GwcNet-g with one hourglass) (Fig. 4).
- **2d Aggregation.** Pre-hourglass + stack of 3 hourglasses. 2D model: 2D conv v2 blocks (t=2 in hourglass, t=3 in pre-hourglass), hourglass width 48. 3D model: 3D-v2 blocks, width 32 (as GwcNet). MobileNet blocks raised to 3D (depthwise 3x3x3 + pointwise, Fig. 2).
- **2e Disparity.** Soft-argmin on the downsampled (1/4) volume ("outputting a downsampled disparity map", p.4); regression operator stated in the summary only; main text says output "after upsampling is compared against the GT with smooth-L1". Upsampling method **not stated**.
- **2f Refinement.** none.
- **2g Upsampling.** Not described (bilinear implied for the baseline GwcNet; the repo summary says bilinear but the paper doesn't say).
- **2h Fusion points.** None. A semantic/edge cue could enter: (a) the interlacing sub-network could take class features as extra interlaced channels; (b) pre-hourglass v2 blocks; no existing mechanism.

### 3. Block -> problem -> evidence table
SceneFlow EPE, 20-epoch ablations; MACs at 256x512 (p.6).

| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Interlaced_i cost volume vs concat/corr (2D model) | 2D model weak with correlation | Tab. 2 EPE/D1/px3: concat 1.86/7.46/8.48; corr 1.71/6.80/7.84; Interlaced1 1.70/6.20/7.06; I2 1.61/6.39/7.31; **I4 1.55/6.15/7.06**; I8 1.64/6.41/7.35; I16 1.73/6.65/7.58 | 2D baseline, SF | learnable (3D convs on 2C ch) |
| v1 in feature extraction (2D model) | MACs | EPE 1.55 -> 1.66 alone, MACs 74.42 -> 30.43, params 4.07 -> 1.52M (Tab. 3a) | | |
| v2 in hourglass (2D) | | conv FE + v2 HG 1.63 (74.32 MACs); v2/v2 1.53 (35.44); **v1/v2 1.50 (30.33 G, 1.21M)** best EPE; v2/v1 1.60 | | |
| v1/v2 in 3D model | | baseline conv/conv 0.97 EPE 155.2 GMac 4.21M; v2/conv 0.96 (116.3, 1.94M); v2/v2 0.97 (110.1, 1.27M); v1/v2 0.99 (105.0, 0.98M); v1/v1 1.03 (99.7, 0.87M) (Tab. 3b) | single hourglass | |
| 3 stacked hourglasses + t | accuracy | final 2D EPE 1.14, 3D 0.80 (Tab. 4); higher t -> worse and heavier (p.6, no table here) | | |
| Channel reduction / cost-volume convs replaced by v1/v2 | | worse accuracy -> kept standard (p.5-6, tables in appendix, not available) | | |
| Initial convs / pre-hourglass with v2 (t=3) | | "both complexity and error reduced" for the 2D baseline (appendix table not available) | | |
| 2D vs 3D convs in encoder-decoder | detail preservation | qualitative only: 3D "obtains crisp edges", 2D "visually similar" (p.7, Fig. 7). Numerically: KITTI15 test All D1-bg 2.49 vs 1.75, D1-fg 4.53 vs 3.87 (Tab. 6) | | 2D: 32.2 GMac, 3D: 153.14 GMac |

### 4. Interactions & dependencies
- 2D model works only because the interlacing volume is learnable and the hourglass is widened to 48 channels (2D model has MORE params than 3D: 2.23-2.32M vs 1.77M) (p.6).
- MobileNet v2 expansion factor t: reduction factor drops with t, 2D blocks become heavier than a standard conv beyond t>5 (Fig. 3); chosen t=2 (hourglass) / 3 (first convs, pre-hourglass).
- Interlaced group size i has an optimum at 4 (non-monotonic).
- Replacing the channel-reduction 1x1 convs or the cost-volume convs by MobileNet blocks hurts (learning capacity).
- Over-parameterized pretrained models fine-tune poorly on small KITTI (DeepPruner-Fast cited, p.7).

### 5. Losses
Smooth-L1 between the upsampled predicted disparity and GT (p.4); weight/formula: not given beyond "smooth-L1". Single output, no deep supervision stated (3-hourglass stack's intermediate supervision: not stated; GwcNet uses it, can't confirm here).

### 6. Training recipe
Mostly **not stated** in this PDF (appendix missing). Main text: ablation models chosen by least EPE after 20 epochs; SceneFlow finalpass 35,454/4,370 at 540x960, dmax=192; KITTI15 159/40 split fine-tune from SceneFlow; for the benchmark submission "finetuned the epoch with the best cross-domain generalizability from SceneFlow to KITTI 2015" (p.7). Optimizer, LR, batch, crop, augmentation, hardware: not stated.

### 7. Results
- SceneFlow (Tab. 4): 2D-MobileStereoNet EPE 1.14 px, 2.23M params (vs DispNet-C 1.67/38M, CRL 1.60/78.77M, AutoDispNet-C 1.51/37M, iResNet 1.40/43.11M); 3D-MobileStereoNet 0.80 px, 153.14 GMac, 1.77M (GwcNet-gc 0.76 / 260.49 / 6.82; GwcNet-g 0.79 / 246.27 / 6.43; GA-Net-deep 0.84 / 670.25 / 6.58; PSMNet 0.88 / 256.66 / 5.22; DeepPruner-Fast 0.97 / 51.83 / 7.47).
- KITTI15 val (Tab. 5): 2D EPE 0.79, D1 2.53, px3 2.67, 32.2 GMac, 2.32M; 3D 0.66 / 1.59 / 1.69 / 153.14 GMac / 1.77M; PSMNet 0.88 / 2.00 / 2.10; GwcNet-g 0.62 / 1.49 / 1.53.
- KITTI15 test (Tab. 6, All): 2D D1-bg/fg/all 2.49/4.53/2.83; 3D 1.75/3.87/2.10 (GwcNet-g 1.74/3.93/2.11; PSMNet 1.86/4.62/2.32; DeepPruner-Fast 2.32/3.91/2.59).
- Model size (Tab. 7): 2D 10.03 MB, 3D 7.99 MB (PSMNet 21.1, GwcNet-g 26.3). **No runtime/latency measured**; MACs only.
- **Edge/boundary measurements:** none quantitative. Only KITTI15 D1-fg vs D1-bg as a coarse foreground proxy (3D 3.87 vs 1.75; 2D 4.53 vs 2.49) and the qualitative claim in Fig. 7 that 3D convs give crisper edges than 2D convs in the encoder-decoder. No upsampling ablation. The MobileNet-block substitution is not tied to edges.

### 8. Negative results & limitations
- Authors: MobileNet replacement of channel-reduction and cost-volume convs deteriorates accuracy; higher t makes the net less accurate and heavier; Interlaced>4 worse; 2D model's crisp-edge deficit vs 3D.
- Mine: no latency data; MAC reduction (7-19x on a single conv) does not translate to GPU/TensorRT speed since depthwise 3D convs are memory-bound (not tested here). Edge claim is a visual statement from the KITTI benchmark figure. The 2D/3D comparison's "fewer parameters" statements compare against over-parameterized baselines. Appendix missing from the provided PDF, so exact channel lists and training hyperparameters can't be verified here. Ablations are single 20-epoch runs.
- Errors in repo summary (summaries/lightweight/MobileStereoNet_Shamsafar_WACV2022.md): "soft-argmin followed by bilinear upsampling" (bilinear not stated in the paper); describes expansion factor t "chosen per-layer" (paper: t=3 for first convs and pre-hourglass, t=2 in hourglass); says it paved the way for BGNet (BGNet is CVPR 2021, earlier than this WACV 2022 paper and not cited by it). Numbers quoted (1.55/1.86/1.71 EPE, 1.14/2.23M, 0.80/1.77M/153 GMac) are correct.

### 9. Relevance to OUR model
- Weakly relevant. Our stereo is frozen A09; MobileNet blocks matter only for the fusion head cost, which is already small. Possible use: depthwise-separable convs in the fusion head/ClassResidual if we need to cut latency on Jetson (no latency evidence in the paper, must benchmark).
- Interlacing cost volume is a learnable correlation replacement; only relevant if retraining A09's cost volume (new series). The Interlaced4 gain (1.71 -> 1.55 EPE over correlation) suggests learnable similarity helps a 2D-regularized net; our A09 uses a correlation volume with shallow head, so a learned-grouped similarity is a candidate A-series ablation, expected moderate gain, TensorRT-friendly (2D head) but the 3D-conv interlacing step is itself 3D.
- Edge relevance: only the observation that 2D-conv encoder-decoders blur fine detail relative to 3D ones; our shallow 2D head may share this; no quantitative evidence here, so risk of over-reading.
- Novelty: none for semantics.

### 10. Key quotes/equations
- Eq. 2: C3D(d,x,y) = Interlace{ f_L(.,x,y), f_R(.,x-d,y) } (p.5).
- "Interlaced4 ... 1.55 EPE vs concatenation 1.86 and correlation 1.71" (Tab. 2).
- "3D-MobileStereoNet obtains crisp edges due to deploying 3D convolutions in the encoder-decoder" (p.7), purely qualitative.
- Tab. 1: block MAC formulas, reduction factors 7.9x/2.7x (2D v1/v2), 18.9x/7.0x (3D v1/v2) at k=3, Cin=32, Cout=64, t=2.
