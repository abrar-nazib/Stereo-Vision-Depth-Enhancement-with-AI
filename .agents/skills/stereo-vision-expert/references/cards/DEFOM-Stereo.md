<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# DEFOM-Stereo block card

### 0. Meta
- Title: DEFOM-Stereo: Depth Foundation Model Based Stereo Matching. Hualie Jiang, Zhiqiang Lou, Laiyan Ding, Rui Xu et al. (Insta360 Research, CUHK-Shenzhen). CVPR 2025 (arXiv 2501.09466v3, 23 Apr 2025). 20 pages incl. supplement (read fully).
- PDF: paper/reference_papers/fusion/DEFOM-Stereo_Jiang_CVPR2025.pdf. Code: project page https://insta360-research-team.github.io/DEFOM-Stereo (p.1; no GitHub URL in the PDF).
- Domain: general (driving, indoor, high-res). Datasets: Scene Flow (pretrain, 200k steps), KITTI 2012/2015 (+ virtual KITTI 2), Middlebury, ETH3D, Booster (qualitative), Flickr1024 (qualitative); RVC mixes (TartanAir, CREStereo, FallingThings, CARLA HR-VS, Sintel, IRS, 3D Ken Burns, InStereo2k...).

### 1. Problem & failure modes targeted
- Recurrent stereo (RAFT-Stereo) zero-shot weakness: occlusion, non-texture, blur, high-res, reflective/ill-exposed regions (p.1).
- Monocular DAv2 gives relative inverse depth with unknown scale AND shift; within-image scale is inconsistent, especially on synthetic data (Fig. 2; supp. Tab. 6: least-squares affine-aligned DAv2 EPE 8.04 STD 3.31 on Scene Flow, 5.10 on CREStereo, 2.08 on KITTI15, 5.00 on Middlebury-half (high-res), 0.65 on ETH3D). Authors conjecture training with affine-invariant loss over varying FoV causes it (p.2).
- Large disparities beyond RAFT-Stereo's pyramid-lookup range (4-level pyramid, r=4 -> max 2^2*2^3*4=128 px, p.5).

### 2. Pipeline by stage (built on RAFT-Stereo, simplified to 2 pyramid levels)
- 2a Feature extraction: two encoders. Matching feature encoder on both views, output 1/4 res (h=H/4,w=W/4); context encoder on left only at 1/4,1/8,1/16. Both are "combined": plain CNN (RAFT-Stereo style, trainable) + features from a pre-trained DAv2 backbone (ViT-S or ViT-L, FROZEN, "we fixed the DEFOM", p.7) passed through a NEW, trainable DPT head (initialized fresh because the original DPT must stay fixed to predict depth). Combined Feature Encoder (CFE): final fused DPT feature -> conv block aligning channels -> element-wise ADD to CNN feature map. Combined Context Encoder (CCE): taps Reassemble_4, Reassemble_8, Reassemble_16 of the new DPT, three conv blocks for channel alignment, ADD to CNN context maps at 1/4,1/8,1/16 (p.4). Channel counts, ViT patch size/input resolution: not stated.
- 2b Semantic/prior branch: frozen Depth Anything V2 (ViT-S or ViT-L + its frozen DPT depth head) gives relative depth z. Used for (i) features (above) and (ii) initialization (below). Not semantic.
- 2c Cost volume: RAFT all-pairs correlation C1 in R^{h x w x w} = sum_h f^l f^r (Eq. 1), pyramid by 1D avg pooling on the last dim. Pyramid level for delta update reduced to 2 (as IGEV-Stereo); scale lookup uses only the finest level C1. Disparity levels: all-pairs (full width), no fixed range.
- 2d Aggregation: none (no 3D conv); context features enter the GRU.
- 2e Disparity computation: no argmin; recurrent regression starting from monocular initialization d0 (Eq. 4).
- 2f Iterative update: total iterations N=18 train / 32 eval; the first 8 are Scale Update (SU, fixed in train and eval, p.6), the remainder Delta Update (DU). Hidden state update: standard multi-resolution ConvGRU (1/16, 1/8, 1/4) where the finest GRU gets x_n=[Encoder_c(c), Encoder_d(d_{n-1}), Up2(h^{1/8})] and context features c_z,c_r,c_h are added to the gate pre-activations (Eq. 3). SU: a ConvGRU head predicts a dense scale map s (2 convs), d_n = s * d_{n-1} (Eq. 5), input is the Scale Lookup. DU: d_n = d_{n-1} + Δd (Eq. 2) with pyramid lookup. Whether SU has separate GRU weights: implied by the extra trainable params (+4.4M by Tab. 2 table rows), not explicitly stated.
- 2g Upsampling: convex combination weights predicted from the hidden state (RAFT) (p.5).
- 2h FUSION POINTS (all mono->stereo, unidirectional, DAv2 frozen):
  1. Feature level (matching): CFE additive fusion of DPT features into f^l, f^r at 1/4 (feeds correlation).
  2. Context level (GRU conditioning): CCE additive fusion at 1/4,1/8,1/16; enters every GRU iteration via c_z,c_r,c_h and initial hidden state (this is where mono "monocular cue controls the update" p.2).
  3. Disparity init: d0 = eta*w*z/max(z) + eps (Eq. 4), eta=1/2, eps=0.05, w = image width (resolution-proportional disparity range). No scale/shift is estimated; the unknown scale and shift are handled by SU.
  4. Scale update: dense multiplicative per-pixel scale on the current disparity, driven by Scale Lookup: correlations sampled at C1(s_m d_n) with scale factors s_m in {1,2,4,6,8,10,12,16}/8 (8 scales, plus neighbors +/-1 px => 24 correlation values) (p.6). Global search range since multiplication reaches any disparity, which fixes the PL range limit.
- Confidence mechanism: NONE. No confidence map, no gate, no uncertainty output (the summary agrees). Arbitration between mono and stereo is implicit: the GRU sees the correlation lookups at scaled disparities and learns the scale.
- Scale-shift: only SCALE is modeled (multiplicative, dense); shift is not modeled explicitly (eps bias only; DU additive residuals absorb remaining shift). Normalization by max(z) and width w.
- Real-time variant: none. Latency (960x540, 4090): baseline RAFT-Stereo(2-level) 0.222 s, ViT-S 0.255 s, ViT-L 0.316 s (Tab. 2). Not real-time.

### 3. Block -> problem -> evidence table
Tab. 2 (ViT-S unless noted; Scene Flow 200k steps; KITTI/Middlebury/ETH3D zero-shot bad px: K12 bad3, K15 bad3, Midd-half bad2, ETH3D bad1). Baseline = RAFT-Stereo with 2-level pyramid. Delta vs baseline in brackets.
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Baseline | - | SF EPE 0.56 / bad1 6.66; K12 4.65; K15 5.57; Midd 10.67; ETH 3.45 (Tab. 2) | SF in-domain + zero-shot | 11.11M, 0.222 s |
| +CCE | scale-weak context | SF 0.49 (-0.07); K12 4.40 (-0.25); K15 5.84 (+0.27 worse); Midd 8.42 (-2.25); ETH 2.82 (-0.63) | mixed zero-shot, in-domain good | 12.10M, 0.242 s |
| +CFE | matching features | SF 0.50; K12 4.13 (-0.52); K15 5.53 (-0.04); Midd 10.45 (-0.22); ETH 2.83 (-0.62) | in-domain ~ -10% | 13.89M, 0.243 s |
| +CCE+CFE | both | SF 0.49; K12 4.02 (-0.63); K15 5.75 (+0.18 worse); Midd 8.31 (-2.36); ETH 2.53 (-0.92) | "not much additional gain" over each alone (p.7) | 14.11M, 0.246 s |
| +DI (depth init only) | init | SF 0.57 (+0.01 worse); K12 4.57 (-0.08); K15 5.63 (+0.06); Midd 12.40 (+1.73 WORSE); ETH 2.77 (-0.68) | helps zero-shot only on K12, ETH; hurts Midd | 11.11M, 0.242 s |
| +DI+SU | scale inconsistency | SF 0.50 (-0.06); K12 4.15 (-0.50); K15 5.12 (-0.45); Midd 8.15 (-2.52); ETH 2.67 (-0.78); text (p.7) claims "Bad 2.0 of Middlebury is reduced by over 50%", but Tab. 2 gives -34% vs +DI (12.40->8.15) and -24% vs baseline (10.67->8.15): claim not reproducible from the table | in-domain and zero-shot | 15.51M, 0.244 s |
| Full ViT-S (CCE+CFE+DI+SU) | all | SF 0.46 (-0.10), bad1 5.57; K12 4.29 (-0.36); K15 5.29 (-0.28); Midd 6.76 (-3.91); ETH 2.61 (-0.84) | note slight regressions vs single components on some sets (p.7: K15 vs DI+SU) | 18.51M, 0.255 s |
| Full ViT-L | backbone size | SF 0.42, bad1 5.10; K12 3.76; K15 4.99; Midd 5.91; ETH 2.35 | zero-shot | 47.30M, 0.316 s |
| CNN removal (DPT-only encoders, supp. Tab. 7) | is CNN needed | SF EPE 0.619 (+35% vs 0.458), K15 5.998, Midd 9.379 (+39%), ETH 3.872 vs full 0.458 / 5.289 / 6.760 / 2.614 | CNN+ViT complementary | - |
| Fixed (original) DPT instead of new trainable DPT (Tab. 7) | adaptability | SF 0.473 (+3%), K15 5.450 (+3%), Midd 7.959 (+18%), ETH3D 2.278 (BETTER by 13%) vs full | new DPT better overall; fixed DPT better for small-disparity (<64) data | - |
| SU iteration count (Tab. 8; 50k steps, batch 4, SF only) | how many scale iters | EPE 0.752 (0 it) -> 0.668 (1) -> 0.651 (3) -> 0.650 (5) -> 0.640 (7) -> 0.636 (8) -> 0.637 (9) -> 0.660 (10); Bad1.0 9.018 -> 7.697 (8) | in-domain only | - |
| Ill-posed regions (Tab. 9, Midd-half Bad2.0 zero-shot, SF-trained) | occlusion/textureless | All 13.44 (own baseline) / 11.49 Mocha / 6.76 ViT-S / 5.91 ViT-L; Noc 10.64/9.11/4.29/3.26; Occ 30.33/25.79/20.83/20.64; Textureless 13.01/12.25/7.05/6.04 | zero-shot | - |
| Scale lookup vs pyramid lookup for scale recovery | search range | argued only (p.5-6); no direct SL-vs-PL ablation | - | not ablated |
| eta, eps, scale set {1..16}/8, DU pyramid level 2 | design | not ablated (eta=1/2 chosen so SL reaches whole image) | - | not ablated |
| ViT-S vs ViT-L Tab. 1 (SF-only zero-shot, bad px: K12/K15/Midd-full/half/quarter/ETH3D) | generalization | RAFT-Stereo 4.35/5.74/18.33/12.59/9.36/3.28; ViT-S 4.29/5.29/14.70/6.76/6.38/2.61; ViT-L 3.76/4.99/11.95/5.91/5.65/2.35 (Tab. 1). ViT-L vs RAFT: K -13%, ETH -28%, Midd half -53% | ZERO-SHOT, SF-only | - |

### 4. Interactions & dependencies
- DI alone is useless or harmful (SF +0.01, Midd-half +1.73 worse); it only pays with SU. SU needs eps>0 (multiplicative update cannot recover from exact zero) and the global scale lookup.
- Zero-shot gains of CCE/CFE are dataset-dependent: both worsen KITTI15 slightly alone (K15 5.84, 5.75 vs 5.57) and the full model on K15 (5.29) is worse than DI+SU alone (5.12).
- Frozen DAv2 + new DPT: DPT-only (no CNN) fails; fixed DPT hurts large-disparity data.
- SU iteration number is sensitive beyond 9 (EPE rises 0.636 -> 0.660 at 10) because DU iterations are crowded out in a fixed 18-iteration budget.
- Scene Flow's scale inconsistency "helps train the SU module as it poses more challenges" (supp. A).

### 5. Losses
- L = sum_{n} gamma^{N-n} ||d_gt - d_n||_1, gamma=0.9 (Eq. 6, as printed the sum index is mixed i/n), applied to ALL N iterates (SU and DU) with RAFT exponential weights. No auxiliary, confidence, or depth-prior loss.

### 6. Training recipe
- PyTorch, RTX 4090s. AdamW, one-cycle LR 2e-4, batch 8. Scene Flow pretrain 200k steps, crop 320x736 (p.6). Iterations 18 train / 32 eval, SU fixed 8. eps=0.05, eta=1/2.
- Frozen: DAv2 backbone (+ its depth head); trainable: CNN encoders, new DPT, conv aligners, GRUs (incl. SU), heads.
- Finetune: KITTI = 50k steps on KITTI12+15+virtual KITTI2 (KITTI 50% of mix). Middlebury: 200k steps crop 384x512 on a 7-dataset mix, then 100k at 512x768. ETH3D: 300k at 384x512 then 90k. RVC: ViT-S model, 200k synthetic mix + 100k adding real sets, crop 384x768, batch 8, final 20k with KITTI augmented to half the mix. Augmentation specifics: not stated.

### 7. Results
- Zero-shot (SF only) Tab. 1 above. In-domain SF: EPE 0.42 (ViT-L), 0.46 (ViT-S) vs 0.56 baseline (Tab. 2).
- KITTI (fine-tuned): K15 all D1-all 1.41 (rank 1), D1-fg 2.23 (1st), non-occ D1-all 1.33; K12 Bad2 noc 1.43 / Bad3 noc 0.94 (Tab. 3). Run-time 0.30 s.
- Middlebury (fine-tuned) Bad2 all 5.02, noc 2.39; ETH3D Bad1 noc 0.70, all 0.78 (Tab. 4). RVC (ViT-S): K15 D1-all all 1.63, Midd Bad2 noc 3.28 / all 6.90, ETH3D Bad1 noc 0.98 / all 1.09 (Tab. 5).
- Params/time: ViT-S 18.51M trainable, 0.255 s; ViT-L 47.30M, 0.316 s at 960x540 (Tab. 2); baseline trick note: the baseline got a speed-up (0.329 -> 0.222 s) by pre-defining neighbor indices (footnote Tab. 2).

### 8. Negative results & limitations
- Authors: mirrors/large transparent surfaces fail (supp. F: "When there is a large mirror, our model cannot work"; refers readers to Stereo Anywhere). Scale inconsistency "not eliminated".
- Slight regressions on K15 for CCE/CFE/full vs components (Tab. 2). DI alone worsens Midd-half by 1.73.
- Params text vs table mismatch: text says CCE+CFE +2M (+18%) and SU +5.4M (+49%) while table rows give 14.11-11.11=3.0M and 15.51-14.11 or 15.51-11.11=4.40M; both decompositions sum to 7.4M for the full ViT-S. Treat per-component param numbers as approximate.
- Not real-time; no latency/FLOPs breakdown except total time; hardware only "RTX 4090".
- Zero-shot comparison (Tab. 1) rates other models with released Scene Flow weights; their own baseline is a custom 2-level RAFT-Stereo, whose Midd-half 10.67 differs from the stock RAFT-Stereo's 12.59 in Tab. 1, so the "baseline" in Tab. 2 is itself stronger than stock RAFT.
- Which numbers are in-domain vs zero-shot: Scene Flow EPE/bad1 are in-domain; K12/K15/Midd/ETH3D columns of Tabs. 1,2,7,9 are ZERO-SHOT from Scene Flow only; Tabs. 3-5 are fine-tuned leaderboards (not zero-shot).
- Errors found in summaries/fusion/DEFOM-Stereo.md: (1) the ablation table lists "+DI+SU" with 11.11M params and the full ViT-S as 15.5M; Tab. 2 gives DI+SU 15.51M and full ViT-S 18.51M (the "ViT-S 15.5M" model table is wrong); (2) "DI only hurts on some datasets" is right (SF, K15, Midd) but it helps K12 (-0.08) and ETH3D (-0.68);
  (3) "18 SU+DU iterations ... 32 eval" omits that SU is fixed at 8 iterations in both; (4) summary says the sum in the loss starts at i=n - this is the paper's own notational typo, not a method detail; (5) the summary's "Stage 1/2 ... Encoder_c ... " is fine; (6) "no explicit confidence output" is verified; (7) the claim "CCE more impactful than CFE" is only true on SF EPE by 0.01 and Midd-half; CFE is better on K12.

### 9. Relevance to OUR model
- Closest analogue to our fusion point is the **Combined Context Encoder**: prior features projected by conv and ADDED to the GRU context features, entering gates at every iteration. For us: inject frozen-YOLO/semantic-decoder features (1/8, 1/16) into the context/conditioning of the tile/plane refiner of A09 by conv-align + add. This is a cheap (conv + add), already close to what ClassResidual does implicitly if it uses decoder features; making it explicit as gate-conditioning (c_z,c_r,c_h style) is the portable part. Expected benefit: DEFOM saw ~-10% SF EPE from CCE and -2.25 bad2 on Midd-half zero-shot, but with a dense geometric prior; for a categorical prior expect far less. Risk: K15 got worse with CCE/CFE alone, so zero-shot effects are not guaranteed. Novelty: low (we do this); a combined feature-add into the matching feature (CFE) would break our frozen A09 and is not applicable.
- Scale Update / Scale Lookup / depth init: NOT transferable. They exist to resolve scale of a relative metric prior; a semantic map has no depth. The only transferable lesson: multiplicative-scale refinement with a global lookup helps when initialization is far off; our A09 init is already stereo-derived, so no.
- DI-alone-hurts finding is a caution: injecting a prior as initialization without a mechanism to correct it can regress (Midd +1.73); the analogue for us is that a semantic gate lacking reliability inputs may over-trust classes. DEFOM provides no confidence mechanism to borrow.
- Frozen-prior + new adapter head (new DPT) with the CNN branch kept (CNN removal = +35% EPE) matches our design: frozen trunk, small trainable head, stereo branch retained. Supports keeping A09 correlation features alongside semantic features rather than replacing them.
- Latency: DEFOM ViT-S/L (0.255/0.316 s at 960x540 on a 4090, 18.5M/47.3M trainable) is two orders too heavy for RTX 3050/Jetson; only the ideas, not the architecture, are relevant.

### 10. Key quotes/equations worth citing
- "we use the image width to normalize the depth estimate in initialization" (p.2); d0 = eta*w*z/max(z) + eps (Eq. 4, p.5-6); d_n = s * d_{n-1} (Eq. 5, p.6).
- Scale factors {1,2,4,6,8,10,12,16}/8, "In total, 24 correlation values are retrieved in the scale lookup" (p.6).
- "DI achieves better generalization on KITTI 2012, Middlebury, and ETH3D" while "does not improve in-domain" (p.7).
- "Can we simply abandon CNNs? The answer is No." (supp. B.1, p.15).
