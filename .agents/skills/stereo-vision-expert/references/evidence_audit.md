# Evidence audit: how much to trust semantic-stereo claims

Reading all 38 papers end to end (2026-10-06) showed that the joint
stereo + segmentation literature rests on weaker evidence than its abstracts suggest.
Use this file twice: when judging a paper before borrowing an idea, and when writing
our own manuscript so we don't repeat these confounds.

## 1. Recurring confounds (check every claim against this list)

| # | Confound | Where it occurs | What it does to the claim |
|---|---|---|---|
| 1 | **No capacity-matched no-cue control.** The cue arm adds layers, volumes or a second network vs a baseline without them | SGNet, SSPCV, SegStereo, RTS2Net, SDBF, S3Net, DispSegNet, S3M, TiCoSS, TwInS, AIO, GGEV (DFE rows) | The gain may be capacity or extra features, not semantics. SemStereo's own no-label variant keeps ~80% of the stereo gain (Tab. 1). Only GGEV's "+DCA texture-only" row and MonSter++'s "Mono+Conv" row approximate a control |
| 2 | **Hyperparameters or loss weights tuned on the reported split** | SGNet (40-image val used for ablation *and* tuning), S3M-Net (α on the KITTI test split), TiCoSS (α, β on the test split), SDBF (ablations via test server) | Optimistic, unreplicable deltas |
| 3 | **Tiny val/test sets, single seed, sub-0.02-px deltas** | SGNet (0.007–0.014 px on 40 images), HITNet slant / tile-feature (0.004–0.02 px), CoEx neighbourhood vs GCE, S3M SCG (0.01 px) | Within seed noise. Only D-FUSE reports mean ± std over checkpoints |
| 4 | **vKITTI2 random-pair splits** with the 10 weather/lighting variations of one (scene, frame) possibly spread across train and test | S3M-Net, TiCoSS (700 random pairs), likely TwInS | Near-duplicate leakage inflates mIoU (~84–88%). **Our grouped 80/10/10 split avoids this. Say so explicitly** |
| 5 | **Per-image-averaged mIoU** vs dataset-level mIoU | S3M-Net, TiCoSS (6–14 pt gaps vs mmseg numbers, TiCoSS Tab. IV) | Not comparable to standard mIoU or to ours. Always state the definition. Our B used union-present classes, C onward GT-present ones |
| 6 | **Published-number baselines** (not re-run under the same recipe) | SGNet, SSPCV vs PSMNet, SemStereo (S2Net/S3Net copied), SDBF, TwInS stereo (compared only to un/self-supervised methods) | Recipe differences are confounded with method |
| 7 | **Resolution or protocol mismatch** in SOTA tables | USAM-Net (half-res EPE vs full-res published), LiteAnyStereo "4K at 21 ms" vs 33 GMACs at 1242×375 | Invalid comparisons |
| 8 | **Trainable-only parameter counts** that hide a frozen prior | GGEV (3.68M "trainable" hides DA-V2-S), AIO (no params or runtime at all), DTPnet (inconsistent 0.64M / 0.26M) | Understated cost. **Report frozen + trainable params and full-pipeline latency** |
| 9 | **Gain mostly from data or labels, not the module** | FoundationStereo (FSD 2.34→1.15 ≫ adapter 2.48→1.97), TwInS (Cityscapes labels and pseudo-labels; "largely attributed to semi-supervised training"), RTS2Net (Cityscapes SGM pretraining) | Separate the data effect from the architecture effect |
| 10 | **Regime mismatch.** Gains in unsupervised / photometric training presented as general | SegStereo (big unsup gains, ~0 supervised), DispSegNet (unsup only) | Does not transfer to supervised fine-tuning |
| 11 | **Domain-specific priors.** Remote-sensing class→height correlation | SemStereo (Fig. 2 itself says the prior fails at ground level), S3Net, SDBF | Do not port intra-class-disparity assumptions to driving |
| 12 | **Edge claims without an edge metric** | HITNet, CoEx, MobileStereoNet, LightStereo MSCA, ESMStereo, AIO's SAM-edges role | Qualitative only. BGNet is the sole edge-band metric |
| 13 | **Bundled ablations** (several sub-blocks enter in one row) | StereoAnywhere rows D/E, SSPCV "w/o FFM" also toggles dilated conv, PromptStereo cumulative-only | Cannot attribute to a sub-block |
| 14 | **Text vs table inconsistencies** | DEFOM ("Midd −50%" vs Tab. −24/−34%), TiCoSS ("+7.93" not reproducible), RTS2Net (0.12 vs 0.08), DTPnet (params/FLOPs), LightStereo (RepViT 0.6823 vs 0.6876), SENSE (attention claim vs Tab. 3), LiteAnyStereo (100K vs 200K iterations) | Always quote the **table**, not the prose |
| 15 | **File ≠ cited version.** `MonSter_Cheng_CVPR2025.pdf` is MonSter++ (arXiv 2501.08643v2) | — | Cite MonSter++ numbers as MonSter++ |

## 2. Strength of the core claims in this field (as of 2026-10)

| Claim | Status | Best evidence |
|---|---|---|
| "Semantics improves stereo accuracy" | **Weak in the literature.** Supervised in-domain deltas are 0.01–0.07 px without controls (SGNet, RTS2Net, S3M, TiCoSS). SegStereo's large gains are unsupervised-only | **Our D/E series is the strongest controlled evidence**: D1−D3 −0.2424 px, D2−D4 −0.2479 px vs equal-capacity no-semantics controls (in-domain VKITTI2) |
| "Semantics improves zero-shot transfer" | Almost untested in the literature. USAM-Net is negative, SemStereo cross-city is mixed | **Ours**: KITTI15 all-valid EPE F2 2.5876 vs F3 control 2.7139 vs F1 stereo-only 3.0822 (200 pairs). One run, one teacher split, so it needs seeds |
| "Stereo improves segmentation" | Moderate. TwInS CTA +1.5/+3.8 mIoU (Tab. IV), SDBF +0.8, SemStereo seg+stereo loss vs none +9.45 (but confounded) | — |
| "Shared encoders cause task conflict" | **Unmeasured** for disparity. Seg-side only (TiCoSS), and JointTree finds rank ρ=0.89 (weak) | Our frozen trunk sidesteps it. Claim that, don't claim to have measured it |
| "Frozen priors beat fine-tuned ones" | Good for depth/VFM priors | FoundationStereo freeze 1.97 vs unfreeze 3.94 (Tab. 5). ViTAS is the opposite in-domain (partial unfreezing helps), so it's not a controlled contradiction |
| "Edge-aware upsampling sharpens edges" | Only vs naive linear 1/8 upsampling (BGNet) | **Ours: null** for RGB residual, convex upsampling and warp on the frozen E3 (G) and on the scratch A09 head (H) |
| "Multiplicative gating > addition" | Moderate, consistent across papers | CoEx Tab. IV, SGNet Tab. 1, LightStereo MSCA |

## 3. Protocol we must keep (manuscript-grade)

1. **Equal-architecture, equal-capacity no-semantics control** for every semantic arm.
   Also keep a zeroed-semantics control (C4) and a spatially misaligned one (C5). This is
   the single biggest advantage over every paper in `cards/`.
2. **Grouped vKITTI2 split** (all 10 variations of a `(scene, frame)` in one split). State it,
   and contrast it with S3M/TiCoSS random pairs.
3. **Report mIoU with its definition.** Never compare against per-image mIoU tables
   without noting the difference.
4. **Seeds.** Our current D/E/F results are single-seed. Repeat with ≥3 seeds and report
   mean ± std before claiming 0.1-px-scale effects. D-FUSE's checkpoint mean ± std
   is the minimum bar.
5. **Teacher-split overlap.** The semantic teacher's random full-VKITTI split may overlap
   the depth test frames. Audit or eliminate this before a generalisation claim (AGENTS.md).
6. **Edge-band metric** (BGNet definition) and **boundary-F1 / background-IoU** for seg
   (JointTree collapse detectors). Report them alongside EPE / D1 / bad-x.
7. **Cost columns.** Params (frozen and trainable separately), latency on the target device
   at a stated power mode, resolution, precision and batch, and peak VRAM.
8. **Competitor table caveats.** TwInS / TiCoSS / S3M-Net train on KITTI15 140 pairs
   (+ pseudo-labels for TwInS); we are zero-shot on 200 pairs. Put the protocol column in
   the table. TwInS has no released code, so a reproduction is impossible.

## 4. Summary errata
See `summary_errata.md` for factual errors in `paper/reference_papers/summaries/`. The
old `lightweight.md` / `fusion.md` / `semantic.md` skill references were built partly on
those summaries and have been replaced by the cards.
