<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# FoundationStereo card

All page refs are PDF pages: main text 1-8, references 9-12, supplement 13-16. Tab./Fig. numbers as printed. No existing summary for this paper exists in paper/reference_papers/summaries/ (only Fast-FoundationStereo.md), so no summary errors to report.

### 0. Meta
- Title: FoundationStereo: Zero-Shot Stereo Matching
- Authors: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield (NVIDIA)
- Venue/year: CVPR 2025 per the paper's own reference lists in PromptStereo (cites it as CVPR 2025); the PDF itself only shows arXiv:2501.09898v4, 4 Apr 2025 (p. 1). Venue not stated in this PDF.
- PDF: paper/reference_papers/fusion/FoundationStereo_Wen_CVPR2025.pdf (16 pages)
- Code/project: https://nvlabs.github.io/FoundationStereo/ (p. 1). No repo URL printed.
- Domain: general zero-shot (indoor/outdoor/driving/robotics).
- Datasets: train on FSD (new, 1M synthetic pairs) + Scene Flow, Sintel, CREStereo, FallingThings, InStereo2K, Virtual KITTI 2 (Sec. 4.1). Eval: Scene Flow, Middlebury, ETH3D, KITTI12/15, Booster (supp).

### 1. Problem & failure modes targeted
- Zero-shot generalization without per-domain fine-tuning; prior networks trained on small Scene Flow (40K pairs) (Sec. 1).
- Sim-to-real gap (fixed by side-tuning a monocular foundation model), textureless/reflective/translucent/thin/repetitive structures (Fig. 1, Fig. 3 right), large disparity range and long-range context in cost filtering (Sec. 3.2), ambiguous synthetic samples (self-curation, Sec. 3.5).
- Limits: speed/memory (0.7 s on 375x1242 A100, Sec. 5), few transparent objects in FSD.

### 2. Pipeline by stage
- 2a Feature extraction: Side-Tuning Adapter (STA). Frozen DepthAnythingV2 (ViT, L variant chosen by Tab. 5 rows 2-4) + trainable EdgeNeXt-S CNN, siamese (shared weights across left/right, p. 3). Input image resized to be divisible by 14 for the ViT. CNN gives pyramid f^(i), i in {4,8,16,32} i.e. 1/4..1/32 (Sec. 3.1). At the 1/4 level only: the DAv2 feature taken from the DPT head's last feature before the final output head is downsampled by a 4x4 conv, stride 4, then concatenated with the same-level CNN feature -> hybrid 1/4 feature (Fig. 3 (c), Fig. 2). Channel counts C_i not stated. STA also used for the context branch (CNN replaced by residual blocks + downsampling), context features at 1/4, 1/8, 1/16, used to init GRU hidden states and as GRU input (Sec. 3.1, Eq. 5-8). EdgeNeXt-S chosen "for memory efficiency; larger CNN backbones did not yield additional benefits" (p. 3; not tabulated).
- 2b Semantic/prior branch: the frozen DAv2 latent feature IS the prior (not the depth output): "Instead of using the raw monocular depth ... which has scale ambiguity, we use its latent feature" (p. 3). Frozen throughout.
- 2c Cost volume: hybrid. V_C in R^{C x D/4 x H/4 x W/4}, V_C = [group-wise corr (G=8 groups, L2-normalised features), concat volume of 1x1-conv-reduced features (reduced to 14 channels, weights shared L/R)] (Eq. 1). So channels = 8 + 2x14 = 36 (derived, not stated). Max disparity 416 at inference (Sec. 4.1). Separate all-pairs correlation V_corr at 1/4 used only for GRU lookup (Eq. 3).
- 2d Aggregation: Attentive Hybrid Cost Filtering (AHCF) = hourglass (3 down, 3 up 3D conv blocks with residuals) using Axial-Planar Convolution (APC) everywhere except the down/up layers, + Disparity Transformer (DT) in parallel; DT output trilinear-upsampled and summed with hourglass output (Fig. 2, p. 4). APC: 3x3x1 spatial conv + 1x1x17 disparity conv, each with BN+ReLU (best kernel, Tab. 6 rows 10-15). DT: 3D conv k4x4x4 stride 4 downsample, reshape so each (H/16,W/16) location is a token sequence along the disparity axis (length D/16), + positional encoding (cosine chosen), 4 transformer encoder blocks, 4 heads, FlashAttention, FFN (p. 5).
- 2e Disparity: soft-argmin over filtered volume -> d0 at 1/4 res (Eq. 2).
- 2f Refinement: ConvGRU, 3 levels (1/4,1/8,1/16) coarse-to-fine, attention-based level selection [67]; lookup of V'_C (filtered hybrid volume) and V_corr at current disparity (Eq. 4-10). Context feature c = ReLU(f_c) enters x_k (Eq. 5). 22 iters train, 32 iters inference.
- 2g Upsampling: convex upsampling (RAFT) from 1/4 to full res (p. 5).
- 2h FUSION POINTS (the second cue = frozen DAv2 latent feature):
  1. Feature stage, 1/4 res, left AND right images: concat of 4x4/s4-conv'd DAv2 feature with CNN feature (side-tuning adapter (c)). Enters BOTH the group-corr matching features and the concat volume, hence "rich monocular priors embedded into the 4D cost volume" (p. 2).
  2. Context branch: DAv2-adapted context features (same STA design) initialise GRU hidden state and feed each iteration (Sec. 3.1, 3.3). Direction: mono -> disparity only. No loss coupling, no depth supervision from DAv2.

### 3. Block -> problem -> evidence table
All Tab. 5-7 numbers: Middlebury training-set BP-2 (%) (lower better), zero-shot, trained on a randomly subsampled 100K FSD subset (Sec. 4.5, p. 8). Everything below is one seed, no variance stated.
| block | problem it solves | evidence | context | cost |
|---|---|---|---|---|
| STA vs no STA (Tab. 7 row 1 vs 2) | sim-to-real gap, ambiguous regions | 2.48 -> 2.21 BP-2 (-0.27); "W/o STA" = CNN-only features (Fig. 3 caption) | 100K FSD, frozen DAv2 | DAv2 (ViT) forward per image; "STA module" is peak-memory bottleneck at half/quarter res (supp Tab. 10 text) |
| Foundation model choice (Tab. 5 rows 1-4) | which VFM to adapt | DINOv2-L 2.46; DAv2-S 2.22; DAv2-B 2.11; DAv2-L 1.97 | same | bigger = better monotone |
| STA design (a) DPT-head pyramid of frozen DAv2 only, no CNN (Tab. 5 row 5) | - | 6.48 (LOST by a wide margin; far worse than the CNN-only 2.48 of Tab. 7 row 1) | | |
| STA design (b) ViT-Adapter-style feature exchange CNN<->ViT (Tab. 5 row 6) | - | 2.22 (lost) | | |
| STA design (c) 4x4/s4 conv of DPT-head final feature, concat with CNN at 1/4 (row 7) | adopted | 1.97 (best) | | |
| Freeze vs unfreeze ViT (Tab. 5 rows 8-9) | preserve priors | Unfreeze 3.94 vs Freeze 1.97 (unfreezing corrupts priors) | | |
| DT position encoding (Tab. 6 rows 1-2) | | RoPE 2.19; Cosine 1.97 | | |
| DT feature scale (rows 3-4) | | 1/32 2.06; 1/16 1.97 | | |
| DT attention axis (rows 5-6) | | Full volume 2.25; Disparity-only 1.97 | | |
| DT placement vs hourglass (rows 7-9) | | pre 2.06; post 2.20; parallel 1.97 | | |
| APC kernels (rows 10-15), (spatial),(disparity) | | (3,3,1),(1,1,5) 2.10; (1,1,9) 2.06; (1,1,13) 2.01; (1,1,17) 1.97; (1,1,21) 1.98; (7,7,1),(1,1,17) 1.99. Saturates ~17 | | 5x5x5 3D conv OOM on 80 GB (p. 4) |
| APC only added (Tab. 7 row 3) | large-disparity receptive field | 2.21 -> 2.16 (with STA) | | |
| DT only added (row 4) | long-range context | 2.21 -> 2.05 | | |
| STA+APC+DT (row 5) | | 1.97 | | |
| No-STA with AHCF (not run) | | not ablated: no row has AHCF without STA | | |
| FSD dataset (Tab. 7 right) | data scale/diversity | 2.34 w/o FSD -> 1.15 with | full training set | |
| FSD on other methods (supp Tab. 9) | | IGEV Scene Flow->FSD: Midd BP-2 8.8->7.8, ETH3D BP-1 4.0->3.5, KITTI12 D1 5.2->3.2, KITTI15 5.7->4.7. Selective-IGEV: 9.2->7.9, 5.7->3.5, 4.5->3.0, 5.6->4.4 | | |
| Self-curation (supp Tab. 8) | ambiguous synthetic pairs | Middlebury BP-2 1.27 w/o -> 1.15 with | | |
| Hybrid cost volume (corr + concat), G=8, 14-ch | | not ablated | | |
| Context from STA vs CNN-only context | | not ablated separately | | |
| Convex upsampling, 22/32 iters | | not ablated | | |

### 4. Interactions & dependencies
- STA only works as (c): the frozen ViT feature must be fused with a CNN stream; using the DPT pyramid alone is catastrophic (6.48), and the ViT must stay frozen (3.94 if unfrozen).
- DAv2 features beat DINOv2 features for stereo despite DINOv2's good correspondence (p. 8): attributed to task relevance and resolution of pixel-level correspondence (hypothesis, not tested).
- DT only helps when attending over disparity dimension of a downsampled (1/16) cost volume; full-volume attention is worse (2.25) - authors hypothesise the huge token space (untested).
- APC kernel in disparity saturates ~17; enlarging gains come only up to there.
- AHCF gains are additive on top of STA (2.21 -> 1.97) but the reverse ordering (AHCF alone) is not shown.
- Large-scale synthetic data (FSD) is the biggest single lever (2.34 -> 1.15 BP-2) - larger than any module. Architecture gain is ~0.5 BP-2 total (2.48 -> 1.97).

### 5. Losses
L = |d0 - d*|_smooth(L1) + sum_{k=1..K} gamma^{K-k} ||d_k - d*||_1, gamma = 0.9, exponentially increasing weights (Eq. 11). No weight on d0 other than 1. No auxiliary loss on the monocular branch.

### 6. Training recipe
AdamW (decoupled WD), 200K steps, total batch 128 over 32 A100, lr 1e-4 decayed by 0.1 at 80% of training, random crop 320x736, RAFT-Stereo-style augmentation [36], 22 GRU iters training; inference 32 iters, max disp 416 (Sec. 4.1). DAv2 frozen; CNN, cost volume, AHCF, GRU trained from scratch (presumably; init of CNN not stated). Ablations use a 100K FSD subset (smaller schedule not stated). FSD generation: NVIDIA Omniverse, RTX path tracing 32-128 spp, 48 A40 GPUs for 10 days, >5K assets, 12 large scene models, 16 skyboxes, >150 materials, 400 textures, 1280x720, camera focal/baseline randomised (Tab. 1, supp Sec. 11, pp. 14-16). Self-curation: train an initial model on FSD, evaluate on FSD, samples with BP-2 > 60% treated as ambiguous and regenerated; two rounds (Sec. 3.5).

### 7. Results
- Zero-shot, Scene Flow-only training (Tab. 2): Midd BP-2 5.5, ETH3D BP-1 1.8, KITTI12 D1 3.2, KITTI15 D1 4.9 (best over CRE++, IGEV, NMRF, ...).
- Any-data training (excluding target): 1.1 / 0.5 / 2.3 / 2.8 (Tab. 2).
- Scene Flow test EPE 0.34 vs 0.41 best prior (Tab. 3).
- ETH3D leaderboard: fine-tuned BP-0.5 1.26, BP-1 0.26, EPE 0.09; zero-shot 2.31 / 1.52 / 0.13 (Tab. 4). Ranked 1st on ETH3D and Middlebury leaderboards at submission (supp Figs. 6, 8; Middlebury avg bad-2 1.84).
- Runtime: 0.7 s at 375x1242 on A100 (Sec. 5). Supp Tab. 10 (RTX 3090, Middlebury): full res 18.5 GB / 8.14 s, half 10.5 GB / 2.97 s, quarter 2.3 GB / 0.55 s; slower than IGEV-family baselines (e.g. Selective-IGEV half 1.7 GB / 0.72 s - read from table).
- Booster half res zero-shot: BP-2 ~9.6, BP-1 19.0 (supp p. 13).

### 8. Negative results & limitations
- Tried and lost: STA variants (a) 6.48, (b) 2.22; unfreezing ViT 3.94; DINOv2 2.46; RoPE, 1/32, full-volume attention, pre/post-hourglass DT (all Tab. 6); larger CNN backbones gave no benefit (unreported numbers).
- Authors admit: not efficient; FSD has few transparent objects.
- My critique: (1) every ablation is a single run on one benchmark (Middlebury training set, 100K FSD) with 0.05-0.25 BP-2 gaps and no variance - small gaps (1.97 vs 1.98 vs 1.99, DT rows) are within plausible noise. (2) No row isolates the injection STAGE: only feature-stage injection variants were compared; no cost-volume-level or refinement-level injection tested. (3) The adapter ablation (Tab. 5) never compares against the same CNN without DAv2 under the same 100K protocol except via Tab. 7 row 1 (2.48), so the adapter's net gain is -0.27 (STA) - tiny compared to data effect (2.34 -> 1.15). (4) PromptStereo (Sec. 6.2 supp) notes training code unreleased and that FS needs batch 128/32 A100, so reproduction is infeasible.

### 9. Relevance to OUR model
Our trunk is a frozen CNN (YOLO26m layers 0-6) shared by stereo and segmentation; our fusion head adds semantics at the disparity-candidate level.
- Directly portable principle: frozen prior + trainable CNN stream, fused by concat at 1/4 res and fed into the cost volume. In our design the "second cue" (semantics) is currently fused after the stereo predictor; FS Tab. 5 suggests feature-stage fusion before the cost volume could yield more than output-stage gating, but FS provides no stage comparison, so this is a hypothesis. Insertion: concat a conv'd 1/8 semantic feature (strided/1x1 reduced) into A09's matching features. But A09 is frozen, so this requires retraining the stereo head = out of scope for the frozen design; risk high.
- Strong takeaway for frozen-trunk design: never unfreeze (3.94 vs 1.97), do not use the prior alone (6.48), keep a task-specific trainable CNN stream next to it. Our YOLO trunk is already such a CNN; the missing piece is the trainable side stream.
- Portable cheap block: APC (3x3x1 + 1x1xK_d separable 3D conv) as a cost-filter replacement; gain small (2.21 -> 2.16) and our model is shallow/real-time; DT (+0.16) is too heavy for a 3050. Low priority.
- Context-feature init of refinement from the adapted prior: matches our ClassResidual idea of class-conditioned refinement; conceptually same (prior -> refinement hidden state).
- Not portable: FSD scale, 22-iter GRU, ViT-L (0.7 s on A100).
- Novel-combination angle: FS uses a monocular geometry prior; we use semantic class prior from a frozen CNN trunk, so "semantic side-tuning" feeding the cost volume is not done here.

### 10. Key quotes/equations
- "we use its latent feature as geometric priors extracted from both stereo images and compared through cost filtering" (p. 3).
- "Surprisingly, while being simple, we found (c) significantly surpasses the alternatives" and "unfreezing ViT corrupts the pretrained monocular priors" (p. 8).
- Eq. 1 hybrid volume; Eq. 11 loss; APC = "decouples 3x3x3 conv into 3x3x1 and 1x1xK_d" (p. 4-5).
