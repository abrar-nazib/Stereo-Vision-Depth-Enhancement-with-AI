# Design playbook for our model (frozen shared trunk + semantic fusion head)

Written 2026-10-06 from the 38 verified cards and the repo state. Re-check `AGENTS.md`
and the series `INSIGHTS.md` files for newer results before acting. Each move lists its
evidence, insertion point, cost, risk and the **control it needs**. Without that control
the result is not publishable (see `evidence_audit.md`).

## 0. What exists (do not re-propose)

- **Trunk:** frozen YOLO26m layers 0–6, run once per stereo pair and shared by stereo and
  semantics (`experiments/B/B_0_fused_baseline/model.py`, `FusedStereoSemantic`).
- **Stereo:** frozen A09 `FusionStereoLite` (`experiments/a06_shallow/runners/model_a07v2.py`),
  HITNet/LightStereo lineage, tile/plane refinement, 1/16 group cost volume.
- **Semantics:** frozen 14-class VKITTI2 decoder
  (`models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt`).
- **Fusion head** (`experiments/D/D_1_semantic_cost/model.py`, width-scaled in `experiments/E/`):
  - **`SemanticCostGate`**
    - inputs: the mean of A09's group volume + the **left(x)·right(x−d) class-probability
      agreement**. That agreement *is* an L-R class-consistency mask, the categorical
      analogue of StereoAnywhere's depth-bin masking. Already done.
    - network: 2-layer **3×3×3** Conv3d, identity-initialised, output `1 + 0.5·tanh` in [0.5, 1.5]
    - no-semantics control: zero the agreement
  - **`ClassResidual`**: depthwise per-class conv on prob × disparity (SGNet-style) +
    fuse → `4·tanh` residual. The same idea as SGNet's residual module and SemStereo's SSR.
- **Results:**
  - in-domain: D1−D3 −0.2424 px, D2−D4 −0.2479 px
  - KITTI15 zero-shot EPE: E3 2.5876 vs E4 control 2.7139 vs A09-only 3.0822
  - edge refiners on E3 (G1 RGB residual, G2 convex, G3 stereo-warp): null
  - H1/H2 on a stereo-only scratch head: tiny edge/outlier gains, no EPE gain, no
    visual gain
- **Done in spirit by the literature** (cite, don't claim):
  - shared encoder for seg+stereo: RTS2Net 2020, TwInS 2026
  - semantic cost gating: SGNet 2020
  - class-conditioned residual: SGNet, SemStereo
  - frozen segmentation features for stereo: SegStereo 2018

## 1. Tier 1: cheap, strengthens the existing claim (do first)

| # | Move | Evidence | Insertion / cost | Control | Risk |
|---|---|---|---|---|---|
| 1.1 | **Seeds + teacher-split audit**: ≥3 seeds for E3/E4 (and D2/D4); audit the semantic-teacher split overlap with the depth test frames | Field-wide single-seed weakness (evidence_audit §1 #3) | Compute only | n/a | None. Without it, 0.1-px-scale claims are fragile |
| 1.2 | **Stereo-reliability channels into the gate**: (a) normalised entropy of A09's candidate distribution `1 + Σp log2 p / log2 K` (StereoAnywhere); (b) feature-warp residual `|F_L − warp(F_R, d)|` on trunk features (MonSter++ SGA condition, PromptStereo AIF) | StereoAnywhere confidence is bundled (row D), MonSter++ SGA vs conv −8.1% bad-1 (Tab. IX), D-FUSE hybrid confidence 1.15 vs 1.18 (Tab. 7). The 3×3×3 gate sees only 5 disparity levels, so it cannot compute a global-over-d statistic | Extra input channels to `SemanticCostGate`. ~0 params, ~0 ms | **Give the same channels to the no-semantics control**, else the gain is confounded | Low. Tests the hypothesis "semantics helps where stereo is uncertain" |
| 1.3 | **Gate-training volume augmentation**: roll a wrong peak, add noise, zero candidates; occasionally substitute the GT label map for the predicted probabilities (oracle prior) | StereoAnywhere row E vs D: Booster bad-2 −2.32, Midd −0.71 (Tab. 1) | Training only | Identical augmentation for the control arm | Low–medium. Synthetic corruption may not match real failures |
| 1.4 | **Pointwise gate**: 1×1×1 Conv3d instead of 3×3×3 (equal or lower params) | GGEV SCF 1×1 K12 4.11 vs 3×3 4.59 zero-shot (Tab. 7) | Swap kernel | Same change in the control | Low. Targets the zero-shot gap |
| 1.5 | **Semantic-feature correlation** instead of (or with) probability agreement: correlate projected pre-classifier decoder features L vs R; combine multiplicatively with the stereo correlation (SGNet's actual design) | SGNet × beats + (1.299 vs 1.362, Tab. 1; tiny, 40 images) | One more correlation, no params | Control with a non-semantic feature correlation of equal width | Medium–low. Probabilities may be near one-hot, so the two could be equivalent |
| 1.6 | **Edge-band + boundary-F1 metrics** in every eval (BGNet / JointTree definitions) | evidence_audit §3 | Eval only | n/a | None |
| 1.7 | **Device latency** of the full pipeline at the lowest Jetson power mode + RTX 3050, head on/off (anytime exit) | RTS2Net anytime design; LAS2 power-mode 2.2× spread | Measurement | n/a | None. Needed for the "real-time" claim |

## 2. Tier 2: the real-domain story (highest paper value)

| # | Move | Evidence | Cost | Control | Risk |
|---|---|---|---|---|---|
| 2.1 | **Pseudo-label adaptation of the fusion head only** on unlabeled real driving stereo:<br>• data: KITTI raw **excluding all KITTI15 scenes**, or DrivingStereo<br>• teacher: offline FoundationStereo<br>• filters: LR + edge + sky masks (sky from *our* decoder), clamped loss (LAS2), normal-consistency (Fast-FS), or per-image quantile on candidate fluctuation (TwInS) | Fast-FS Tab. 4: LightStereo-L K15 D1 12.08→7.63. LiteAnyStereo stage 3. LAS2 filters. RTS2Net: real-domain data > SceneFlow | Offline teacher compute (a Modal decision, not local). Head training is cheap | The no-semantics control is adapted identically. Report pre- and post-adaptation | Teacher bias. **Scene leakage into KITTI15 eval must be excluded.** The protocol changes from "zero-shot" to "self-adapted", so label it |
| 2.2 | **Self-supervised semantic warp loss** on unlabeled real pairs: warp right-view class probabilities (from the frozen decoder) to the left with the head's disparity via bilinear `grid_sample`; CE / KL against left probabilities, occlusion-masked | SegStereo: large gains exactly in the unsupervised regime (2.17→1.89 EPE, Tab. 1); SemStereo LRSC works without labels | Train-only, zero inference cost | Same photometric/adaptation loss without the semantic term | Our G3 stereo-warp correction was null, but that was an inference-time correction, not a training signal. Gradient only informative at class changes |

Running 2.1 and 2.2 against the same control gives a clean ablation: "semantic gate
+ semantic self-supervision adapts better than an equal-capacity non-semantic head".

## 3. Tier 3: architectural novelty (one at a time, after Tier 1)

| # | Move | Evidence | Cost | Control | Risk |
|---|---|---|---|---|---|
| 3.1 | **Semantic-keyed dynamic candidate aggregation** (GGEV DDCA analogue): query = A09 cost plane per disparity, key = semantic decoder features pooled to S×S centres → per-plane dynamic K×K kernels before regression | GGEV: prior-keyed kernels K12 4.11 vs texture-keyed 6.80 (Tab. 4); Q=cost/K=prior best (Tab. 8). Edges via matching, not appearance (block_matrix §4) | +~0.03M params; GGEV +9 ms on desktop | Same module keyed on non-semantic trunk features of equal width | Medium: memory-bound reshapes, TensorRT friendliness unknown. **Novel**: no paper keys dynamic aggregation on semantics |
| 3.2 | **Residual prompt injection into A09's tile refiner** (PromptStereo / DEFOM CCE style): `h += Conv(sem_feat)`, kept separate from the state, not zero-initialised in the last layer | PromptStereo Tab. 5 (merged 4.86 vs residual 4.59; zero-init worse). TwInS hidden-init −0.06 EPE | Requires un-freezing or wrapping A09's refinement. Breaks "frozen stereo" | A non-semantic prompt of equal width | High (changes the frozen-predictor story). Do only if 3.1 fails |
| 3.3 | **Geo→sem reverse adapter** for a bidirectional claim: disparity (or A09 hidden features) → small trainable adapter on the frozen decoder at 1/8 (SENSE SDAF sigmoid-gate form, ~1M params; or TwInS CTA linear attention) | TwInS CTA +1.5 / +3.8 mIoU (Tab. IV); SENSE SDAF +1.5 mIoU | ~1M params | An adapter fed with zeroed disparity | Medium. Adds an mIoU result; bad geometry → bad semantics (TwInS Fig. 8), so gate it |
| 3.4 | **Loss-only depth-prior distillation** into the head (Pip-Stereo MPT style): align the head's internal features to frozen DA-V2 relative depth during training only | Pip MPT 32-iter 0.44→0.40; GGEV shows depth priors are the strongest generalisers (K12 6.63→4.40) | Training only | Separate arm. It must not be credited to semantics | Medium. Could overshadow the semantic contribution. Keep it as a separate arm, not merged |

## 4. Do NOT spend compute on (evidence says null or wrong regime)

- More post-hoc edge refiners: RGB residuals, convex/superpixel/ShuffleMixer upsamplers,
  stereo-warp corrections on the frozen near-full-res predictor (our G/H; block_matrix §4).
- Iterative GRU heads on Jetson: LAS2 3.4× cost on Orin, GGEV ~2 ms/iteration on desktop (35→47 ms
  for 2→8 iterations, Tab. 6).
  Pip shows <1% of pixels change late.
- Unfreezing the trunk: FoundationStereo freeze 1.97 vs unfreeze 3.94. It also breaks the shared
  seg decoder (RTS2Net −1.99 mIoU when disparity pulls the trunk).
- Heavy attention fusion: TwInS HFSB degrades seg; ViTAS CAM is too memory-heavy for 4 GB.
- Input-level mask concat (USAM-Net), semantic slices inside the cost volume (S3Net, needs
  joint training), SAM/VFM at inference (USAM +900 ms; AIO no runtime reported).
- Intra-class narrow-disparity priors (SemStereo): false at ground level (its own Fig. 2).
- Top-k regression or test-time swaps on the frozen A09 (CoEx: must train with top-k).

## 5. Novelty statement (defensible as of 2026-10)

Not novel alone:
- a shared encoder for seg + stereo
- semantics injected into stereo
- semantic cost gating
- class-conditioned residuals
- frozen segmentation features

Defensible combination:
1. one **frozen, pretrained detector/segmenter trunk** (YOLO26m 0–6) computed once per
   pair and reused by a **frozen** stereo predictor and a **frozen** semantic decoder
2. a tiny trained fusion head doing **semantic cost-candidate gating + class-conditioned
   residuals**
3. **equal-capacity, zeroed and misaligned semantic controls** showing that the gain is semantic, on
   a **leakage-free grouped vKITTI2 split**
4. **zero-shot real-domain transfer** (KITTI15) attributed to semantics via the same
   controls
5. **real-time, sub-4-GB** cost with no seg degradation (decoder frozen)

The closest competitor is TwInS (2026):
- end-to-end ConvNeXt, 68–276M params
- semantics in GRU context and init, no cost gating, no no-semantics control
- 18–22 FPS on an RTX 4090
- KITTI15 trained with pseudo-labels

RTS2Net is the closest real-time precedent: jointly trained, with no control.
See `trends_and_novelty.md` for the full competitor table.
