---
name: stereo-vision-expert
description: Evidence-graded architectural knowledge for stereo disparity estimation fused with semantic segmentation (38 papers read end-to-end, 2018–2026), centred on this repo's frozen-YOLO-trunk + semantic cost-gate model. Use when choosing or changing a stereo head, fusion point, gate, residual, loss, or training recipe; when diagnosing edge blur, textureless/occlusion/reflective errors, EPE spikes or zero-shot drops; when judging whether a paper's gain is real; when planning the next ablation, its controls, or a Modal vs local run; when positioning novelty against TwInS/TiCoSS/S3M-Net/SGNet/RTS2Net; when writing related-work or a manuscript claim; or when planning real-time Jetson/RTX-3050 deployment.
---

# Stereo Vision Expert

Knowledge base for designing a **real-time model that fuses stereo disparity with semantic
segmentation**. Every paper in `paper/reference_papers/` was read in full on 2026-10-06 and
turned into a verified block-card with table and page refs. Paths are relative to the repo root.
Files under `references/` are relative to this skill.

## How to use this skill

1. Find the question type in the routing table below and read that file **before**
   answering. The SKILL.md body is the summary. The files hold the numbers.
2. Before quoting any number in a manuscript or a decision, open the paper's card in
   `references/cards/<Paper>.md` and quote the **table** value with its ref. Paper prose
   often contradicts its own tables (`evidence_audit.md` §1 #14).
3. Before proposing any experiment, check `design_playbook.md` §0 ("what exists")
   and §4 ("do not spend compute on"). Name the **control arm** it needs.
4. Never cite `paper/reference_papers/summaries/*.md` as fact: they contain verified errors
   (`summary_errata.md`).

| Question | Read |
|---|---|
| Which block fixes failure X? Where can a cue enter, and with which operator? | `references/block_matrix.md` |
| Is this paper's gain real? How do we make our claims hold up? | `references/evidence_audit.md` |
| What should our next experiment be? What is already done? What is a dead end? | `references/design_playbook.md` |
| Is our idea novel? Who are the competitors? What is the field doing? | `references/trends_and_novelty.md` |
| Losses, optimiser, augmentation, curricula, pseudo-labels, debugging signatures | `references/training.md` |
| Latency, TensorRT/Jetson operator support, multi-head runtime, compression | `references/deployment.md` |
| One specific paper: architecture, shapes, exact losses, every ablation | `references/cards/<Paper>.md` (index: `references/paper_index.md`) |

## Our model in one paragraph (verify in code before relying on it)

- **Trunk:** frozen YOLO26m layers 0–6, run once per stereo pair.
- **Frozen predictors on the trunk:**
  - A09 stereo predictor `FusionStereoLite`: HITNet/LightStereo lineage, 1/16 group
    cost volume, tile/plane refinement
  - 14-class VKITTI2 semantic decoder
- **Trained:** only a fusion head (`experiments/D/D_1_semantic_cost/model.py`, width-scaled in E).
  - `SemanticCostGate`: a 3×3×3 Conv3d over [mean A09 volume, left(x)·right(x−d)
    class-probability agreement], output a multiplicative [0.5, 1.5]
  - `ClassResidual`: a per-class depthwise residual
  - no-semantics control: zero the agreement at equal capacity
- **Results** (single seed, `experiments/*/INSIGHTS.md`):
  - in-domain: semantic arms beat equal-capacity controls by ≈0.24 px EPE
  - KITTI15 zero-shot EPE: E3 2.59 vs E4 control 2.71 vs A09-only 3.08
  - edge refiners: RGB-guided residual, convex upsampling and stereo-warp correction were all null (G/H)

## Ten rules that the evidence supports

1. **Keep priors frozen and add a small trainable path. Never replace the matching
   features.**
   - FoundationStereo: freeze 1.97 vs unfreeze 3.94 vs prior-only 6.48 BP-2 (Tab. 5).
   - D-FUSE: DA-V2 as the feature extractor gives EPE 3.26 vs 1.15.
   - DEFOM: removing the CNN branch costs +35% EPE.
2. **Multiply, don't add, when a cue modulates matching.**
   - CoEx excite 0.685 vs add 0.731 (Tab. IV).
   - SGNet: × beats + (Tab. 1).
   - LightStereo MSCA: −0.034 EPE for +0.2 ms.
3. **A gate generalises only when keyed on a domain-robust cue, and pointwise fusion
   generalises better.**
   - GGEV texture-keyed kernels give an in-domain gain only (K12 6.80). Prior-keyed give 4.11.
   - GGEV 1×1 fusion 4.11 vs 3×3 4.59 (Tabs. 4, 7, 8).
4. **Gates learn to ignore a cue unless training forces them to use it.**
   - Use volume corruption plus oracle-prior substitution: StereoAnywhere −2.32 Booster
     bad-2 (row E vs D).
5. **Inject into refinement residually, with the state kept separate.**
   - PromptStereo: merged-conv 4.86 vs residual 4.59 (Tab. 5).
   - DEFOM: depth-init without correction regresses (Midd +1.73).
6. **Edge gains come from matching information, not appearance guidance.**
   - BGNet's edge-band gain exists only against linear upsampling of a 1/8 volume.
   - On a near-full-res predictor, post-hoc RGB guidance and convex upsampling are null:
     our G/H, and nothing in the literature contradicts it.
   - Decision tree: `block_matrix.md` §4.
7. **Dense refinement over a converged disparity adds little.**
   - Pip-Stereo: <1% of pixels change by iteration 32.
   - Iteration pruning needs a regularised init.
   - Keep our head single-pass. GRU loops cost 3.4× on Orin (LAS2).
8. **Data and labels move results more than modules do.**
   - FoundationStereo: FSD data 2.34→1.15 vs the whole adapter 2.48→1.97.
   - Fast-FS pseudo-labels: LightStereo-L K15 12.08→7.63.
   - RTS2Net: Cityscapes > SceneFlow pretraining.
   - The cheapest real-domain lever for us is pseudo-label or self-supervised adaptation of the head alone (`design_playbook.md` Tier 2).
9. **Semantic losses help most without GT.**
   - SegStereo warp-CE: 2.17→1.89 EPE unsupervised, ~0 supervised.
   - SGNet loss module: −0.0005 px.
   - Use them for real-domain adaptation, not in-domain polish.
10. **Every semantic claim needs an equal-capacity no-semantics control, seeds, and a
    leakage-free split.**
    - Almost no published semantic-stereo paper has the control (`evidence_audit.md` §1 #1).
    - Ours is the main methodological edge. Keep it in every new arm.

## Fast diagnosis

| Symptom | First check | Likely fix (evidence) |
|---|---|---|
| Edges look soft but EPE is fine | Is the output already ≥ 1/4-res with tile/plane refinement? | Don't add guided upsamplers. Try matching-side changes: semantic-keyed dynamic aggregation (playbook 3.1) or reliability-aware gating (1.2) |
| Semantic arm ≈ control | Does the gate see stereo reliability? Was it trained with corrupted candidates? | Playbook 1.2 + 1.3 (give the control the same inputs and augmentation) |
| In-domain gain but no zero-shot gain | Is the gate keyed on texture/RGB? Is the fusion spatial (3×3)? | Pointwise fusion (1.4). Key on semantics or depth priors (GGEV) |
| Train curve spiky, val smooth | Score every pair with the latest checkpoint | `training.md` debugging signatures (dark/unmatchable frames, A04b) |
| Small classes (poles/signs) worse | Is an image-weighted smoothness term on? | Segment-aware smoothness (DispSegNet Tab. III) or drop the term |
| Mirrors / glass punch through | Is there a continuous depth prior? | Truncation needs mono depth (StereoAnywhere). A 14-class prior cannot fix it |
| Latency over budget on Jetson | Is it measured on the device at the lowest power mode? | `deployment.md`: 2D ops, no loops, 1×1 fusion, shared trunk, TensorRT |

## Novelty position (short; full table in `trends_and_novelty.md`)

These are not novel on their own:
- a shared encoder for seg + stereo (RTS2Net 2020, TwInS 2026)
- semantic cost gating (SGNet 2020)
- class residuals (SGNet, SemStereo)
- frozen segmentation features for stereo (SegStereo 2018)

Defensible: the **combination**, which no 2018–2026 paper found has all of:
- a frozen pretrained detector/segmenter trunk shared by a frozen stereo predictor and a
  frozen decoder
- a tiny semantic cost-gate + class-residual head
- equal-capacity, zeroed and misaligned semantic controls on a grouped split
- semantic-attributable zero-shot transfer
- a real-time, sub-4-GB budget

The closest competitor is **TwInS (2026)**:
- end-to-end ConvNeXt, 68–276M params, 18–22 FPS on an RTX 4090
- semantics enter GRU context and init; no cost gating, no no-semantics control
- KITTI15 trained with pseudo-labels, so protocols differ from our zero-shot

## Maintenance

- **New paper:**
  1. Write a card with the same section template as `references/cards/` (sections 0–10;
     numbers carry refs; "not stated" rather than guesses).
  2. Add it to `paper_index.md` with an evidence grade.
  3. Update `block_matrix.md` / `trends_and_novelty.md` if it changes a rule.
- **New project result:** update `design_playbook.md` §0 and any rule it confirms or breaks.
  Section 9 ("Relevance to OUR model") of the cards is dated 2026-10-06.
- **Edit location:** this skill lives in `.agents/skills/` and is symlinked into
  `.claude/skills/`. Edit it in `.agents/skills/`.
