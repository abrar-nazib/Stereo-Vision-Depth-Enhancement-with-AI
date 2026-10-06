<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# AIO-Stereo card

### 0. Meta
- Title: All-in-One: Transferring Vision Foundation Models into Stereo Matching. Authors: Zhou, Zhang, Yuan et al. (Fudan / Xiaomi / Shanghai AI Lab). AAAI 2025 (arXiv 2412.09912). 9 pages, NO supplement (p. 1-9; p. 8-9 refs).
- pdf: paper/reference_papers/fusion/AIO-Stereo_Zhou_AAAI2025.pdf. Code URL: not stated.
- Domain: general (indoor Middlebury, outdoor ETH3D, driving KITTI). Datasets: Scene Flow, Middlebury 2014, KITTI-15, ETH3D; finetune mixtures Tartan Air, CREStereo, Falling Things, Sintel, InStereo2k, CARLA HR-VS, KITTI-12 (Sec. Datasets / Experiments, p. 5-6).

### 1. Problem & failure modes targeted
- Weak encoder features of iterative stereo (Selective-IGEV) in dark areas and low-texture areas (Fig. 1b, p. 1); small, mostly synthetic stereo training data -> poor general features (p. 1).
- Two obstacles to using VFMs: (1) architecture heterogeneity (ViT VFM vs CNN stereo), naive feature merge/distill hurts: "-0.71 in terms of EPE" (p. 3; NOT in a table, baseline/setting not stated); (2) VFMs disagree: DINO focuses global/foreground, SAM small objects+edges, Depth Anything dark/low-texture; indiscriminate acceptance causes conflicts (p. 2-3).

### 2. Pipeline by stage
- 2a Feature extraction: Selective-IGEV/RAFT-style. Feature network -> correlation volumes (pixel-wise inner product Eq.1, 4 resolutions via average pooling, RAFT-Stereo style, p. 4). Separate Context network: one 7x7 conv then 3 residual blocks (residual + downsampling layers). Feature maps f_i in R^{C_i x H/2^{i-1} x W/2^{i-1}}, i=1..3 (p. 4; as printed f_1 would be full-res, likely typo, true scales not stated). Channel counts not stated. Trained end-to-end (student); VFMs frozen (lock icons Fig. 2).
- 2b Semantic/prior branch: three frozen VFMs, DINOv2 (self-supervised ViT), SAM (ViT encoder, segmentation FM, only image encoder features used as inferred from "ith stage of SAM"), Depth Anything v2. Their stage-i features d_i, s_i, a_i are the distillation targets; sizes/ViT variants not stated. Not class labels; feature-level only.
- 2c Cost volume: correlation, 4-level pyramid; Selective-IGEV baseline's geometry-encoding volume not described in this paper. Disparity levels: not stated.
- 2d Aggregation: n/a in paper (inherits baseline).
- 2e Disparity: not stated beyond GRU update.
- 2f Refinement: GRU update operator (iterations not stated) consuming correlation lookups + context features; Selective-GRU/contextual attention from Selective-IGEV (p. 1).
- 2g Upsampling: not stated.
- 2h FUSION POINTS (all in the Context network, 3 sites after residual blocks 1,2,3): DSKT module. Operator: (i) distillation: per VFM x in {d,s,a}, lightweight expert E_i^x(f_i) -> e_i^x; heavier Feature Alignment net A_i^x maps e_i^x to VFM space, loss MSE vs interpolated VFM feature F(s_i) (Eq.3-4); (ii) forward fusion: f_{i+1}=B_i(f_i)+sum_x e_i^x * g_i(x) (Eq.8; Eq.5 for single-expert case); (iii) gating: g_i=KeepTopK(Softmax(G_i(f_i|psi_i), dim=0), k) per PIXEL over the 3 experts (Eq.7). Direction: VFM->stereo context only (training-time distillation + inference-time expert additions). Resolution: per-block context resolution. Gate input is the CNN feature itself (no confidence/uncertainty). k value: not stated.

### 3. Block -> problem -> evidence table
Ablation Tab. 1 (p. 5): Middlebury v3 TRAINING set, full-res, all pixels; baseline = Selective-IGEV. Training regime (SceneFlow-only vs finetuned) for this table: NOT stated. Metrics EPE px / >2px %.
| block | problem | evidence | context | cost |
|---|---|---|---|---|
| Baseline Selective-IGEV | - | 0.74 / 4.68 (Tab.1) | Midd v3 train | n/a |
| Full AIO (3 VFM + distill + forward + select) | weak encoder | 0.66 / 3.48 (-0.08 EPE, -1.20 pts >2px vs baseline) | same | params/ms not stated anywhere |
| w/o Selection (non-selective sum of experts) | VFM conflict | 0.68 / 3.57 (+0.02 EPE, +0.09 pts) | same | gate net, not reported |
| w/o Distillation (L_KD removed, experts kept) | knowledge transfer | 0.72 / 3.87 (+0.06, +0.39) -- largest single drop; isolates VFM knowledge from extra params | same | - |
| w/o Forward Fusion (distill only) | knowledge attrition via backprop only | 0.67 / 3.52 (+0.01, +0.04) | same | - |
| Only DINO | single VFM | 0.66 / 3.64 (EPE equal to full!, >2px +0.16) | same | - |
| DINO+SAM | two VFMs | 0.68 / 3.61 (EPE WORSE than DINO-only by 0.02; >2px -0.03) | same | - |
| DINO+SAM+DepthAny (full) | | 0.66 / 3.48 | same | - |
| Per-VFM region assignment (DINO fg, SAM edges, DA dark/textureless) | which VFM for which region | ONLY qualitative: Fig. 4 selection-weight maps on a street (KITTI-like) image; text p. 6: DINO weights high on foreground, SAM weights mainly on edges, DA weights on dark/low-texture. No quantitative per-region EPE, no SAM-only, no DA-only, no leave-one-out of SAM or DA. | one image | - |
| Expert/FA network depth, higher FA LR, gamma_KD, k, number of experts | design | not ablated | - | - |
| Zero-shot (SceneFlow -> Middlebury, Tab. 3, p. 6) | generalization | AIO F EPE 4.16 / D1 11.67; H EPE 0.89 / D1 6.48 vs Selective-IGEV 5.28/12.07 and 1.35/7.31; IGEV 5.87/11.85, 1.36/7.21; RAFT 3.84/15.64 (RAFT has best F EPE), GMStereo 4.10/29.15 | zero-shot, F/H not defined in caption (presumably full/half) | - |
Honest reading: the multi-VFM benefit is mainly in >2px outliers (4.68 -> 3.48); EPE gain is small (0.74 -> 0.66) and does not grow monotonically with VFM count. Single-DINO already gets EPE 0.66.

### 4. Interactions & dependencies
- Distillation is required (without it 3.87 vs 3.48): extra expert parameters alone give most of nothing. Selection helps only modestly (0.09 pts) but is claimed essential for avoiding conflicts. Forward fusion (add expert output) gives small gain, motivated as counter to knowledge attrition. FA net must be heavier than expert and gets a higher LR with larger LR decay (p. 4) to avoid misalignment/over-retention. Direct heterogeneous merge hurts (-0.71 EPE claim). Gating reads CNN feature only, so its behavior depends on what the stage-i context features encode.

### 5. Losses
- L_P = sum_{i=1}^N gamma_P^{N-i} ||p_i - p_GT||_1 (Eq.2, RAFT sequence loss). L_KD,i = sum_{x in {d,s,a}} MSE(A_i^x(e_i^x|theta), F(s_i^x)) (Eq.4,6). L_AIO = L_P + sum_{j=1}^3 gamma_KD^{4-j} L_KD,j (Eq.9; later stage weighted more if gamma_KD<1). gamma_P, gamma_KD values: not stated.

### 6. Training recipe
- PyTorch, NVIDIA A100s, AdamW. Pretrain: Scene Flow clean+final, 200k steps, batch 8, crop 320x720, one-cycle LR max 2e-4, warm-up 1% (p. 5). Finetune LR linear 3e-4 -> 0. Middlebury: stage1 mix of Tartan Air, CREStereo, Scene Flow, Falling Things, Sintel(?), InStereo2k, CARLA HR-VS, Middlebury, 200k steps, crop 384x512, bs 8; stage2 mix CREStereo, Falling Things, InStereo2k, CARLA HR-VS, Middlebury, crop 384x768, bs 8, 100k. ETH3D: 300k + 90k steps (mix incl. ETH3D). KITTI: KITTI-12+15, bs 8, 50k. VFMs always frozen. Augmentation: not stated (only random crop). GPU count not stated.

### 7. Results
- Tab. 2 (p. 5), benchmark/FINETUNED (not zero-shot): Middlebury bad1.0/bad2.0/avgerr/A90 = 6.08 / 2.36 / 0.85 / 0.76 (Selective-IGEV 6.53/2.51/0.91/0.79; claims rank 1 on Middlebury leaderboard, bad2 -5.98% vs Selective-IGEV, -26.25% vs DLNR (verified arithmetic)). ETH3D bad0.5/1.0/2.0/avgerr = 2.91/0.94/0.21/0.13 (Selective-IGEV 3.06/1.23/0.22/0.12 -> avgerr is NOT best). KITTI-15 D1-bg/fg/all 1.35/2.46/1.54 (Selective-IGEV 1.33/2.61/1.55; so only D1-fg -5.75% and all by 0.01).
- Params, FLOPs, runtime, memory: NOT reported.

### 8. Negative results & limitations
- Authors: naive VFM merge -0.71 EPE. No limitations section. Mine: no efficiency numbers though framed as "efficient" and teacher cost is training-only (inference VFM-free is implied by frozen-teacher distillation, not stated explicitly); backbone is the heavy Selective-IGEV (not a lightweight CNN); gains vs Selective-IGEV on KITTI/ETH3D tiny; selection-region claims rest on one qualitative figure; SAM contribution never isolated; Tab.1 EPE non-monotonic; ablation training regime missing; zero-shot only on Middlebury; Fig. 4 shown only on one driving image. No semantic-class or boundary-specific metric, so "SAM helps edges" is unmeasured.

### 9. Relevance to OUR model
- Strongest precedent that a segmentation-model's features help stereo, but evidence is weak and implicit (feature distillation, not class labels, no edge metric). Portable idea 1: per-pixel top-k MoE gate choosing among expert residuals (e.g. semantic-residual expert, edge/identity expert) added to the stereo feature/disparity -- compatible with SemanticCostGate; insertion: on the shared 1/8 trunk features before fusion head, k=2 of 3. Cost small (1x1/3x3 conv gate). Idea 2: distilling a big VFM into encoder is NOT applicable when encoder is frozen (YOLO trunk), only possible if we unfreeze or add a small adapter distilled to SAM/DINO/DepthAnything (training-only cost). Risk: their gain (EPE 0.74->0.66) is small and mostly outlier-rate; our D2/E3 already gets -0.24 px in-domain and zero-shot gain on KITTI, which exceeds AIO's visible effect size. Novel combination: class-conditioned gate (explicit semantic classes) plus MoE top-k of heterogeneous cues; AIO has no class-conditioning and no real-time claim. Heterogeneity argument (ViT vs CNN mismatch) does not apply to us: trunk is CNN and shared.

### 10. Key quotes/equations
- "simply merging or distilling features between vision foundation models and stereo matching models is unsuitable ... negative impact (i.e. -0.71 in terms of EPE)" (p. 3).
- Eq. 7: g_i = KeepTopK(Softmax(G_i(f_i|psi_i), k, dim=0)); Eq. 8: f_{i+1}=B_i(f_i|zeta_i)+sum_x e_i^x ⊙ g_i(x) (p. 4).
- "preference for foreground regions ... DINO; SAM mainly selected on the edges; Depth Anything ... dark and low-texture areas" (Fig. 4 text, p. 6).

### Errors in existing summary (summaries/fusion/AIO-Stereo.md)
1. "Deployed model remains a lightweight CNN" / "lightweight CNN stereo backbone": paper's student is Selective-IGEV (heavy IGEV-class); no params/ms reported. Over-claim.
2. "No VFM inference cost" stated as fact; paper never states inference-time cost or that FA nets/VFMs are dropped. Inference-time VFM-free is plausible inference, not a stated finding.
3. Omits that the ablation shows non-monotonic EPE (DINO-only 0.66 equals full; DINO+SAM 0.68 worse) and that region claims are only qualitative (Fig. 4).
4. Omits that Tab. 2 KITTI/ETH3D numbers are finetuned benchmark results, and ETH3D avgerr 0.13 is not best (Selective-IGEV 0.12).
5. "Priority/Relevance: adapt to a single compact VFM" is speculation; the paper never tests a single-SAM-only or DA-only variant.
Numbers quoted in summary (2.36, 0.94, 1.54, Eq. 7, Eq. 9 form) verified correct.
