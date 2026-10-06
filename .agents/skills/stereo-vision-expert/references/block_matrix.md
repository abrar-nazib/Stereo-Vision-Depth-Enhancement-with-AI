# Block matrix: which block solves which failure, where, and how well

Synthesised from the 38 verified cards in `cards/` (2026-10-06). Every number has a
source ref; open the card before quoting it in a manuscript. "ID" means in-domain,
"ZS" means zero-shot (trained elsewhere). "No ctrl" means the paper ran no
capacity-matched control without the cue, so the gain cannot be attributed to the cue.

## 1. Pipeline stages and fusion-point taxonomy

A stereo network has seven stages:
1. feature extraction
2. cost-volume construction
3. aggregation
4. disparity computation (argmin / soft-argmin / top-k / regression)
5. refinement (one-shot or iterative)
6. upsampling
7. loss

A second cue (semantics, mono depth, VFM features, edges) can enter at any of them.
The operator and the stage both matter.

| Stage the cue enters | Operator | Papers | Best evidence | Verdict |
|---|---|---|---|---|
| Input (extra channels) | concat | USAM-Net (frozen SAM mask) | Seg adds no D1 gain; seg+attn transfers worse to KITTI than attn-only (USAM Tab. 2-3) | **Avoid.** Without a matching mechanism the cue is ignored or hurts |
| Shared encoder (implicit) | multitask gradients | RTS2Net, JointTree, SemStereo, TwInS | RTS2Net joint seg changes EPE 0.91→0.90 only (Tab. III). TwInS shared encoder gives EPE 1.13→1.10 (Tab. III), but against a separate-encoder baseline, not a no-semantics one | Sharing alone buys ~nothing for disparity; it buys compute |
| Feature / matching stream | concat at 1/4 with a trainable CNN side stream | FoundationStereo STA, DEFOM CFE, GGEV SCF | FoundationStereo: frozen prior only 6.48, ViT-Adapter exchange 2.22, concat side-tune 1.97, unfrozen 3.94 (BP-2, Tab. 5) | Strongest evidence for **frozen prior + trainable CNN stream**. Unfreezing hurts |
| Feature stream | replace the stereo encoder by the prior | D-FUSE supp. (DA-V2 / MASt3R as feature extractor) | EPE 3.26 / 4.41 vs 1.15 (D-FUSE supp. Tab. 13) | **Never** replace the matching features with a prior's features |
| Cost volume: separate cue volume | concat-volume + SE-gate fusion | SSPCV-Net | −0.36 EPE on a weak baseline, but only −0.17 on top of a pyramid (Tab. 1), no ctrl | Heavy, weak attribution |
| Cost volume: candidate gating | multiply-gate (agreement of two correlations) | SGNet confidence module, **our D1/D2 SemanticCostGate** | SGNet −0.0066 px alone (Tab. 3, 40-image val, no ctrl). **Ours**: D1−D3 −0.2424 px and D2−D4 −0.2479 px vs equal-capacity no-sem controls | Our controlled result is stronger than any published one |
| Cost volume: cue as a reserved slice | 4D stacking + gates | S3Net (satellite) | SFM gate −0.86 D1 (Tab. 1), joint training, no ctrl | Domain-specific (class↔height) |
| Cost volume: depth-bin / class masking | mask matches to the same bin | StereoAnywhere (8 depth-quantile sub-volumes) | Bundled in row (D) of Tab. 1, not isolated | Categorical analogue (L-R same-class mask) is **untested** |
| Aggregation: image-conditioned excitation | sigmoid(1x1(feat)) × cost | CoEx GCE, LightStereo MSCA, GGEV DDCA | CoEx excite 0.6854 vs add 0.7310 vs none 0.7426 (Tab. IV). LightStereo MSCA −0.034 EPE for +0.2 ms. SE+MSCA saturates (0.6810 vs 0.6809, Tab. IV). GGEV texture-only dynamic kernels: ID gain, no ZS gain (K12 6.80); prior-keyed kernels: K12 4.11 (Tab. 4) | **Multiply > add.** A gate only generalises when keyed on a domain-robust prior |
| Disparity computation | top-k soft-argmin | CoEx | k=2: −0.057 EPE on top of GCE. Must be trained with top-k; a test-only swap fails (p. 6) | Only for a trainable regression stage |
| Refinement: context / hidden-state injection | add prior features to GRU context / init | DEFOM CCE, TwInS, PromptStereo, GGEV | TwInS context −0.04, hidden init −0.06 EPE (Tab. III). PromptStereo residual prompt beats merged conv (4.59 vs 4.86 KITTI15 B3, Tab. 5). DEFOM CCE hurts K15 ZS (+0.27, Tab. 2) | Inject **residually**; keep the hidden state separate |
| Refinement: class-conditioned residual | depthwise per-class conv on prob×disp | SGNet residual, SemStereo SSR, **our ClassResidual** | SGNet −0.0088 px (Tab. 3). SemStereo SSR −2.9% EPE (Tab. 1). Both without a ctrl | Ours already implements it |
| Refinement: output-level residual with RGB + seg | concat [seg, RGB, disp] → 2D residual | SDBF-Net, DispSegNet, RTS2Net synergy | SDBF −2.53 D1 with no RGB-only ctrl. Our **G1 RGB residual = null** | Gains are mostly from the extra 2D refiner, not semantics |
| Refinement: confidence-blended prior | c·d_stereo + (1−c)·prior | D-FUSE GF, PromptStereo AIF, MonSter++ | D-FUSE hybrid confidence 1.15 vs cost-only 1.18 vs none 1.23 EPE (Tab. 7). MonSter++ SGA/MGR beat an equal-param conv fusion (−6.5% EPE, Tab. IX) | Needs a metric/continuous prior. The categorical analogue is a gate |
| Upsampling | bilateral slice / convex / superpixel / ShuffleMixer / plane | BGNet, RAFT, CoEx, ESMStereo, HITNet | Only BGNet measures edges: EPE-edge 5.95 vs 8.13 vs **linear** upsampling of a 1/8 volume (Tab. 1). It hurts KITTI when the baseline is already 1/4 (PSMNet-BG 1.95→2.07). Our **G2/H2 convex = null** | Gains exist only over a naive coarse baseline |
| Loss only (train-time) | semantic warp CE, boundary/smooth, SCG weighting, LR-inconsistency weighting, distillation | SegStereo, SSPCV, SGNet, S3M, TiCoSS, DispSegNet, AIO, LiteAnyStereo, Pip | SegStereo warp CE: big gains only in the *unsupervised* regime (2.17→1.89, Tab. 1), ~0 in supervised KITTI. SGNet loss module −0.0005 px. AIO distillation is the largest of its blocks (+0.06 EPE if removed) | Free at inference. Small supervised gains. Best use is **real-domain adaptation without GT** |

### Operator rules distilled
1. **Multiplicative gating beats additive injection** when the cue modulates matching
   (CoEx Tab. IV, SGNet Tab. 1: × beats +, LightStereo MSCA).
2. **Pointwise beats spatial mixing for generalisation.** GGEV SCF 1×1 gives K12 4.11 vs
   3×3 4.59 (Tab. 7). Our SemanticCostGate uses 3×3×3 Conv3d, so a 1×1×1 variant is an
   untested, cheap ablation.
3. **Key the gate on the robust cue, query with the cost.** GGEV Tab. 8: Q=cost plane,
   K=prior gives 4.11. The reverse gives 4.33. With no prior, 4.58.
4. **Residual, separated injection into refinement.** PromptStereo Tab. 5: merging the
   prompt and hidden state costs 0.3–0.6 bad-2. Zero-init of the last prompt conv hurt
   convergence.
5. **Plain addition can beat heavy fusion at low data.** S3M-Net Tab. V: add 54.33 vs concat
   48.40 vs SA-Gate 52.10 mIoU. TwInS Tab. IV: RoadFormer HFSB fusion *degrades* seg.
   TiCoSS says gates beat add (+4.7 mIoU). The disagreement is unresolved, and all of it is
   KITTI-140-scale.

## 2. Failure mode → mechanism (with conditions)

| Failure mode | Mechanisms that measurably help | Conditions / caveats | Do not expect |
|---|---|---|---|
| **Textureless / flat regions** (road, car bodies) | Semantic warp-CE loss (SegStereo, unsupervised regime). Prior-keyed dynamic aggregation (GGEV DDCA). Mono context (StereoAnywhere C row: Midd 11.15→9.62). Multi-scale coarse-to-fine (HITNet KITTI 0.747→0.484 EPE without/with multi-scale, Tab. 6) | Semantic gains concentrate on big classes: "road and car" (SegStereo p. 10). DispSegNet road −48.9%, car −31.7% (Tab. III) | Gains on thin structures |
| **Occlusion** | softLRC + entropy confidence (StereoAnywhere: Occ 29.06→20.34 Midd, bundled). LR-inconsistency loss weighting (TiCoSS DIA, seg-side). Right-patch inpainting aug (HITNet, unablated) | Needs a right-view disparity or a second pass | Semantic class alone resolving occlusion |
| **Reflective / transparent / mirror** | Truncate the stereo volume at mono depth (StereoAnywhere, qualitative). Prior-keyed aggregation (GGEV KITTI12 reflective 3-all 5.34 vs RT-IGEV 7.26, Tab. 5). Reflective data (D-FUSE TranScene) | All need a *continuous* depth prior. VKITTI's 14 classes have no glass/mirror class | A categorical prior fixing mirrors |
| **Edges / boundary blur / thin structures** | Full-res init + warp-based cost re-evaluation (HITNet: low-res init +0.03 SF, no warping +0.06 SF / +0.12 KITTI EPE, Tab. 6). Learned bilateral slicing **vs linear** (BGNet). Learned > luma guidance (BGNet 1.17 vs 1.28). 3D > 2D decoders (MobileStereoNet D1-fg 3.87 vs 4.53) | Edge gains appear only against an edge-blind coarse upsampler. Post-hoc guidance on a near-full-res predictor showed nothing in **our G/H**, and no paper contradicts that (see cards HITNet §9, BGNet §9) | RGB-guided residuals, convex or superpixel upsamplers sharpening an already full-res predictor |
| **Small objects / poles / signs** | Segment-aware smoothness (DispSegNet: smoothness alone *worsens* pole 11.26→13.13, with seg 7.62, Tab. III). SGNet masked boundary loss (tiny effect) | Plain image-weighted smoothness over-smooths small classes | — |
| **Bimodal / multimodal cost curves** | Top-k regression (CoEx, trained with k). Entropy confidence as a gate input (StereoAnywhere). Multi-hypothesis tiles + confidence w (HITNet, unablated) | Top-k needs a trainable regression stage. A frozen A09 can't use it | — |
| **Domain shift (synthetic → real)** | Frozen depth/VFM prior features (GGEV DFE: K12 6.63→4.40). Pseudo-labelled real data (Fast-FS Tab. 4: LightStereo-L K15 12.08→7.63. LiteAnyStereo stage 3. LAS2 filters). Real semantic pretraining data (RTS2Net Cityscapes > SceneFlow: D1 5.75 vs 6.28, Tab. I). **Our semantics: KITTI15 ZS EPE 2.59 (E3) vs 2.71 (ctrl) vs 3.08 (stereo-only)** | Prior *quality* drives indoor ZS (GGEV ViT-L Midd 6.53→4.41), not KITTI (4.11→4.20). Data beats architecture (FoundationStereo FSD 2.34→1.15 vs adapter 2.48→1.97) | RGB-feature distillation alone (D-FUSE: RAFT context features *hurt* ZS 1.83 vs 2.11) |
| **Scale / shift ambiguity of a prior** | Global LSQ + per-pixel shift GRU (MonSter++ SGA). Scale lookup (DEFOM SU). Ordering/LBP maps (D-FUSE). Registration + confidence blend (D-FUSE GF) | Only for continuous mono priors | Applicability to semantic priors (no scale) |
| **Latency** | Shared trunk (RTS2Net: semantic branch ≈ +28 ms of 159 ms on TX2). 2D-only aggregation with channel boost (LightStereo). Iteration pruning (Pip: 1 iteration 0.45 EPE, needs regularised init). Logits KD + pruning (DTPnet 16 ms Xavier). Fused GWC (Fast-FS ~6× faster) | MACs ≠ latency (LightStereo V1 blocks 2× slower at equal FLOPs; LAS2 ConvNeXt slowest on Orin). See `deployment.md` | GRU loops on Jetson (LAS2: 3.4× cost) |
| **Task conflict when sharing an encoder** | Frozen shared trunk (conflict impossible by construction). Gated fusion (TiCoSS, seg-only evidence). Cross-task adapter with linear attention (TwInS CTA +1.5/+3.8 mIoU) | **No paper measures disparity-side conflict** (TiCoSS card §1, S3M card §1). RTS2Net synergy refinement cost −1.99 mIoU (shared trunk) | Clean evidence either way |

## 3. Interaction / dependency rules

- **A gate ignores the cue unless training forces it.** StereoAnywhere's volume
  augmentations (roll a wrong peak, noise, zeroing, oracle-prior substitution) give
  −2.32 Booster bad-2 on top of the full fusion (row E vs D, Tab. 1).
- **A prior injected as initialisation without a correction mechanism can regress.**
  DEFOM depth-init alone gives Midd +1.73 bad-2 (Tab. 2). A semantic gate without
  reliability inputs may over-trust classes.
- **Dense refinement over a converged disparity adds little.** Pip-Stereo: <1% of pixels
  change by iteration 32, hit ratio >0.99 after ~10 iterations (Fig. 1). This is
  consistent with our G/H null.
- **Iteration pruning needs a regularised init.** It fails for RAFT zero-init: EPE stays >2
  at 1 iteration (Pip §4).
- **Top-k must be trained with top-k.** Too small a k can diverge without enough
  gradient paths (CoEx: PSMNet corr k=2 gives 1.108).
- **A self-distillation teacher must be fixed, not EMA** (LiteAnyStereo Tab. 2f). TwInS uses
  EMA (unablated).
- **Joint-training gains shrink as data grows** (SegStereo Tab. 2: 0.04 → ~0 EPE).
  Semantic gains concentrate where matching is ambiguous.
- **Seg ↔ disparity trade-off in a shared trunk:** RTS2Net refinement −0.58 D1 but −1.99 mIoU.
  DispSegNet IoU 47.6→46.9. A frozen decoder (ours) avoids this by construction, which
  is worth stating in the manuscript.
- **Fusion-module transfer across architectures fails.** TwInS: modules designed for
  duplex-encoder fusion (HFSB, FFM) do not help a shared-encoder model (Tab. IV).

## 4. Edge-sharpening decision tree (use before proposing any edge block)

1. Is the predictor's output already ≥ 1/4 resolution, with tile, plane or warp refinement?
   → Post-hoc RGB guidance, convex upsampling and stereo-warp correction are expected
   to be null. **Our G1/G2/G3 and H1/H2 confirmed this.**
2. Is there a coarse (≥ 1/8) cost volume upsampled linearly? → Learned bilateral slicing
   (BGNet) is the only block with a measured edge-band gain.
3. Want real edge gains on a frozen predictor? → They must come from *matching*
   information, not appearance. Options:
   - full-res init or a warp re-evaluation with learned full-res features (HITNet)
   - or candidate re-weighting keyed on a robust prior (GGEV DDCA)
4. Always report an edge-band metric. BGNet's definition: mean EPE in a 5×5-dilated
   Canny band of GT disparity. Report boundary-F1 / bg-IoU for seg (JointTree).
