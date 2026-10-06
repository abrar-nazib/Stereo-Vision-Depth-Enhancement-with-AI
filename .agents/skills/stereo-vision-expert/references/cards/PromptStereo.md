<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# PromptStereo card

Page refs are PDF pages: main 1-8, references 9-11, supplement 12-13. No existing summary of this paper in paper/reference_papers/summaries/.

### 0. Meta
- Title: PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts
- Authors: Xianqi Wang, Hao Yang, Hangtian Wang, Junda Cheng, Gangwei Xu, Min Lin, Xin Yang (HUST, Optics Valley Lab)
- Venue/year: PDF shows only "arXiv:2603.01650v2 [cs.CV] 3 Mar 2026" (p. 1); CVPR 2026 is from the filename and is NOT stated in the PDF text.
- PDF: paper/reference_papers/fusion/PromptStereo_Wang_CVPR2026.pdf (13 pages)
- Code: https://github.com/Windsrain/PromptStereo (p. 1)
- Domain: general zero-shot (KITTI, Middlebury, ETH3D, DrivingStereo weather, Booster).
- Datasets: train Scene Flow, or "unlimited" mix = FoundationStereo dataset (FSD), CREStereo, FallingThings, Scene Flow, Virtual KITTI 2 (Sec. 4.1). Supp Tab. 7: without FSD, TartanAir added instead (following MonSter).

### 1. Problem & failure modes targeted
GRU refinement is the bottleneck for exploiting monocular depth priors: (1) GRU is trained from scratch so cannot inherit foundation priors, (2) hidden states live in a narrow range (tanh) -> trouble with extreme disparity/complex geometry, (3) GRU fuses inputs and hidden state by direct convolution, distorting/compressing both (Sec. 1, p. 1-2). Targets zero-shot robustness incl. reflective/transparent surfaces and imperfect rectification (Midd-2021, Booster).

### 2. Pipeline by stage (built on MonSter, Sec. 3, Fig. 2)
- 2a Features: MonSter's. Mono branch: frozen Depth Anything V2 gives relative depth d_M (1xHxW) and last-layer depth feature F_M (C x H/4 x W/4). Stereo branch: pretrained DAv2 (DINOv2 frozen) + feature-transfer network, multi-level features F_L^i, F_R^i at H/2^{i+2} (i=0..3) = 1/4..1/32 (Sec. 3.1). Channels not stated.
- 2b Prior branch: frozen DAv2, ViT size not stated in the PDF. Not retrained.
- 2c Cost volume: IGEV/MonSter: group-wise corr V_G (G x D x H/4 x W/4, with G/C0 scaling) -> lightweight 3D aggregation -> geometry encoding volume V_E; plus all-pairs corr V_A (1 x W/4 x H/4 x W/4); pooled into pyramid V_C (Eq. 1, Sec. 3.2). No mono injection here (identical to MonSter).
- 2d Aggregation: lightweight 3D hourglass [1] (as IGEV).
- 2e Disparity: initial d0 regressed from V_E (soft-argmin implied; not stated explicitly).
- 2f Refinement: Prompt Recurrent Unit (PRU) replaces the ConvGRU; 16 iters train / 32 test. See 2h.
- 2g Upsampling: not stated (inherits MonSter/IGEV).
- 2h FUSION POINTS:
  1. Initialization, Affine-Invariant Fusion (AIF) (Sec. 3.3, Eq. 2-4): normalise d_M and d0 by median t(d) and mean absolute deviation s(d); project d'_M = s(d0) d^_M + t(d0); confidence c = sigmoid(Conv([F_L, warp(F_R by d0)])); d_F = c*d0 + (1-c)*d'_M. Fused disparity d_F starts refinement. Mono->disp, full-res.
  2. Hidden state init: h_0^i = ConvBlocks([F_L^i, warp(F_R^i; d0)]) (stereo-aware, vs GRU's left-only context init) (Sec. 3.4).
  3. Structure Prompt P_S = Encoder(F_M, D), D = |d^_k - d^_M| (affine-normalised current disparity vs normalised mono depth); h += ConvBlock(P_S), residual addition, only at highest resolution (i=0, 1/4), iteration-wise (Eq. 5-6, 8).
  4. Motion Prompt P_M = Encoder(V_k, d_k) (local cost lookup + disparity, RAFT/IGEV motion encoder); h += ConvBlock(P_M), highest res only (Eq. 7-8).
  5. PRU architecture itself: the DPT decoder of Depth Anything V2 (multi-resolution refinement layers) initialised from DAv2 weights, used as the recurrent unit -> prior inherited via weights not just features.
  Direction: mono -> disparity for AIF/SP; stereo cost -> hidden state for MP.

### 3. Block -> problem -> evidence table
Ablations use reduced augmentation/crop/iterations vs main results (Sec. 4.4, p. 6) so absolute values differ from Tab. 1. Training: Scene Flow. Metrics Bad-thresholds (%).
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| PRU + MP replacing GRU (Tab. 3) | GRU capacity/prior inheritance | Midd-T(H) EPE/Bad2: 0.94/7.27 -> 0.66/4.18; ETH3D EPE/Bad1: 0.26/2.86 -> 0.22/1.38; time 0.64 -> 0.35 s | MonSter baseline re-trained | time -45% |
| + Structure Prompt (Tab. 3) | structure prior in refinement | 0.66/4.18 -> 0.62/3.90 Midd; 0.22/1.38 -> 0.21/1.35 ETH3D; 0.35 -> 0.36 s | | +0.01 s |
| + AIF (full) (Tab. 3) | init consistency, convergence | 0.62/3.90 -> 0.60/3.76; 0.21/1.35 -> 0.20/1.30; 0.36 s | | negligible |
| PRU universality (Tab. 4) | | RAFT-Stereo 5.68/8.41/2.29 (KITTI15 B3 / Midd B2 / ETH3D B1), 0.36 s -> Prompt-RAFT 4.78/6.39/1.49, 0.38 s; IGEV 6.03/7.04/3.61, 0.37 -> Prompt-IGEV 4.84/6.50/2.21, 0.38; MonSter 5.52/7.27/2.86, 0.64 -> PromptStereo 4.59/3.76/1.30, 0.36 | | +0.01-0.02 s |
| PRU w/o pretrained DAv2 decoder weights (Tab. 5) | prior inheritance | 5.03/4.39/1.39 vs full 4.59/3.76/1.30 (still beats GRU baselines) | | |
| Hidden state + prompt merged into one conv (Tab. 5) | information separation | 4.86/4.37/1.34 | | |
| Zero-init last conv in prompt block (Tab. 5) | | 4.83/4.04/1.42 (hinders convergence) | | |
| Iterations 4/8/12/16/32 (Tab. 6) | convergence | Midd-2021 bad2 PromptStereo 4.35/3.28/2.93/2.79/2.78 vs MonSter 10.75/10.64/10.16/9.61/8.46; Booster(Q) 5.25/4.39/4.24/4.09/3.67 vs 14.09/13.62/12.82/12.62/11.55 | | |
| FSD in training (supp Tab. 7) | | w/o FSD vs with: KITTI15 B3 3.50/3.40; Midd-T B2 2.41/2.21; Midd21 3.97/2.78; ETH3D B1 0.71/0.79; KITTI12 3.34/3.33 | | |
| Stereo Anywhere recipe (supp Tab. 8) | | PromptStereo 3.77/4.59/3.76/4.84/1.30 vs with Stereo Anywhere's augmentation+init ("w/ STA") 3.33/4.12/4.03/5.06/1.26 (KITTI12/15, Midd-T, Midd21, ETH3D); Stereo Anywhere 3.90/3.93/4.49/5.18/1.43; 0.36 s vs 0.65 s | | |
| MP only vs SP only, AIF only, no-prompt PRU | | NOT ablated (Tab. 3 is cumulative PRU+MP -> +SP -> +AIF only) | | |
| Update strategy (no reset gate, z from higher-res h) and hidden-state warped-feature init | | NOT ablated individually | | |
| PRU vs injecting mono at the cost volume | | NOT ablated (see 4) | | |

### 4. Interactions & dependencies
- The question "is prompting the refinement better than injecting at the cost volume?" is NOT answered by any ablation. PromptStereo shares MonSter's cost volume construction exactly (Sec. 3.2) and only changes initialisation (AIF) + recurrent unit; no variant puts F_M/d_M into the cost volume. Only indirect evidence: Tab. 3 vs MonSter (which already injects mono at init and into GRU guidance), so the full gain is attributable to PRU/prompts over GRU in the same cost-volume context. Do not cite this paper as evidence that refinement-stage injection beats cost-volume injection.
- Benefits are largest on hard domains (Midd-2021, Booster, ETH3D): pretrained decoder weights matter (Tab. 5) but even random-init PRU beats GRU, so architecture also helps.
- Prompts as residual additions (not merged conv) are required for information separation (Tab. 5).
- AIF depends on affine-invariant normalisation (median/MAD) and on the iterative loop to fix residual misalignment (p. 4).
- PRU is architecture-agnostic across GRU-based stereo (Prompt-RAFT, Prompt-IGEV) but each still has the cost-volume-based pipeline.

### 5. Losses
L = ||d0 - d_gt||_smooth + sum_{k=1..K} gamma^{K-k} ||d_k - d_gt||_1 (IGEV loss), gamma = 0.9 (Eq. 9). No loss on mono branch.

### 6. Training recipe
PyTorch, 4x RTX 4090, AdamW, one-cycle LR 2e-4; Scene Flow: batch 8, 200K steps from scratch; crop 384x768; RAFT-Stereo augmentation; 16 iters train, 32 test (Sec. 4.1). DINOv2/DAv2 frozen; "we do not use MonSter's checkpoint initialization". PRU initialised from DAv2 DPT decoder. MonSter is retrained with official code because public SF checkpoint was reportedly trained on wrong data (supp Sec. 6.1). Seeds not stated.

### 7. Results
Scene Flow training (Tab. 1; EPE All/Noc, Bad All/Noc): KITTI12 0.79/0.73, B3 3.77/3.32; KITTI15 1.09/1.06, 4.59/4.39; Midd-T(H) 0.96/0.60, B2 6.03/3.76; Midd-2021 0.95/0.75, 8.26/4.84; ETH3D 0.23/0.20, B1 1.56/1.30. Retrained MonSter*: 0.93/0.88, 4.62/4.12; 1.17/1.17, 5.52/5.32; 1.04/0.94, 8.97/7.27; 1.70/1.21, 15.55/10.58; 0.35/0.35, 3.20/2.86. Unlimited training: KITTI12 0.70/0.65, 3.33/3.05; KITTI15 0.88/0.87, 3.40/3.27; Midd-T 0.59/0.44, 3.90/2.21; Midd21 0.78/0.61, 5.97/2.78; ETH3D 0.16/0.15, 0.97/0.79. Booster(Q) unlimited: EPE 0.67, Bad2/4/6/8 = 3.67/2.35/1.95/1.70 (FoundationStereo 1.23, 5.15/3.87/3.39/3.03; Tab. 2). Runtime on Scene Flow 0.36 s vs MonSter 0.64 s (Tab. 3; GPU not stated). Parameters not stated. FoundationStereo is excluded from the main comparison as its training code is unreleased and needs batch 128/32 A100 (supp Sec. 6.2).

### 8. Negative results & limitations
- Authors: weak under extreme adverse weather (Sec. 5); DrivingStereo Rainy PromptStereo is worse than several baselines under SF training (1.59 / 14.30 vs BridgeDepth 1.19 / 6.40, Tab. 2).
- Not shown: any variant putting mono info into the cost volume; AIF-alone; SP-alone; params/FLOPs. Ablation protocol reduced (smaller crop/aug) so not same as main numbers.
- Baseline MonSter was re-trained by the authors after the official checkpoint issue; fairness depends on their retraining. FSD-free ablation shows ETH3D slightly worse with FSD (0.71 -> 0.79) - noise-level.
- Comparison with Stereo Anywhere needs different augmentation/init, "special training strategy" shown in separate table.
- Venue not stated in PDF.

### 9. Relevance to OUR model
- The honest lesson for a frozen encoder: injecting a frozen prior into the iterative refinement via residual "prompt" addition on the hidden state at 1/4 res, with a decoder that inherits pretrained weights, gave large gains over GRU (Midd bad-2 7.27 -> 3.76). Our ClassResidual refinement is the closest analogue: our semantic features should be added residually (h += ConvBlock(P_S)) rather than concatenated and mixed with the hidden state; Tab. 5 shows merging prompt+state in one conv costs ~0.3-0.6 Bad2.
- AIF-like confidence-gated fusion (c*d0 + (1-c)*aligned prior) is cheap (a conv + sigmoid), and parallels our SemanticCostGate, but gates on correspondence reliability (warped-feature similarity) rather than class. Portable idea: add a warp-confidence map as gate input to the semantic gate; expected benefit unknown, low cost, moderate risk since our baseline is a tile/plane refinement not a GRU.
- Not portable directly: DAv2 decoder-as-GRU requires a ViT decoder; we have none. Pretrained-decoder benefit (Tab. 5: 5.03 vs 4.59 / 4.39 vs 3.76) suggests initialising our semantic-conditioned refiner from the frozen semantic decoder's layers; this is a concrete, novel-looking combination but speculative.
- Evidence limitation for us: no stage comparison, so it cannot justify moving our fusion from cost-gate to refinement or vice versa. Our D/E ablations are the only evidence on that.

### 10. Key quotes/equations
- "PRU ... directly inherits monocular depth priors" (p. 2); update: z_k from higher-res hidden state, no reset gate (Eq. 8, p. 5).
- Eq. 4 d_F = c*d0 + (1-c)*d'_M; Eq. 5 D = |d^_k - d^_M|; Eq. 6 h = h + ConvBlock(P_S).
- "prompts are only injected at the highest resolution" (p. 5).
- Tab. 5 caption text: merging hidden state and prompt in one conv "disturbs the separation of information" (p. 8).
