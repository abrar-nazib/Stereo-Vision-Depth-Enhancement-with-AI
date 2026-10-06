# Trends (2018–2026) and our novelty position

Built from the verified cards plus two web/citation searches (2026-10-06). Recall is
limited: no paywalled IEEE-only search, and the citation graph may be incomplete (see §5).
Re-run a search before submission.

## 1. Four eras of semantic + stereo

| Era | Representative | How semantics enter | Training | What it showed |
|---|---|---|---|---|
| **2018–2020: semantic features into 3D-cost-volume nets** | SegStereo, SSPCV-Net, DispSegNet, SGNet, SDBF-Net | Concat at 1/8 (SegStereo); separate semantic cost volume + SE fusion (SSPCV); multiply-gate on candidates + per-class residual (SGNet); output-level residual (SDBF, DispSegNet) | End to end, except SegStereo (frozen PSPNet). PSMNet-scale, 0.6–1 s | Small supervised gains (SGNet ≈0.01 px). Large *unsupervised* gains (SegStereo). No controls |
| **2020: first real-time joint model** | RTS2Net | Shared tiny encoder + synergy residual on the cost volume | Joint, Cityscapes → KITTI | 6.3 FPS on TX2, but refinement costs −1.99 mIoU. Joint training alone ≈0 EPE |
| **2024–2026: RAFT-era joint models (one group: Rui Fan et al.)** | S3M-Net (TIV'24) → TiCoSS (TASE'25) → TwInS (arXiv'26) | Feature-level: add (S3M), gated duplex encoders (TiCoSS), shared ConvNeXt into GRU context/init + geometry→seg cross-task adapter (TwInS) | End to end, 68–385M params. vKITTI2 random splits, KITTI 140/60 | Seg improves a lot. Stereo barely moves (S3M 0.40→0.38, TiCoSS 0.38→0.34 on vKITTI2). No no-semantics controls. TwInS adds semi-supervised pseudo-labels |
| **2024–2026: remote-sensing semantic stereo** | S3Net, SemStereo, SIGLNet, SAOF (+ S2Net, closed) | Semantic slice in the cost volume; class-gated residual (SSR) + warp CE (LRSC); semantic-guided upsampling; SAM prototypes | Joint | Exploits class→height priors that **do not hold at ground level** (SemStereo Fig. 2) |

In parallel, **foundation-prior stereo** (2024–2026) became the main route to
generalisation, through **depth** priors, not semantic ones:
- **DEFOM:** add prior features to context + scale update
- **MonSter++:** bidirectional mono↔stereo GRUs
- **StereoAnywhere:** normals volume + entropy confidence + volume augmentation
- **D-FUSE:** ordering maps + registration
- **FoundationStereo:** frozen DA-V2 side-tuned into a CNN stream + 1M synthetic pairs
- **PromptStereo:** prior as residual prompts in refinement
- **GGEV:** frozen DA-V2-S keys dynamic aggregation, real-time
- **AIO-Stereo:** distils DINO/SAM/DA. The only VFM paper that includes a *segmentation* model,
  and its SAM contribution is not cleanly measured

**Efficiency** converged on:
- 2D-only aggregation with channel boost (LightStereo, LAS2)
- iteration pruning and sparse GRUs (Pip-Stereo)
- distil→NAS→prune with pseudo-labels (Fast-FoundationStereo, LiteAnyStereo)

None of these efficient models has a semantic branch.

## 2. Observable trends

1. **Fusion moved from "concat features" to "condition the matcher".** Gates, dynamic
   kernels and confidence-blended priors (CoEx → LightStereo MSCA → GGEV DDCA → StereoAnywhere
   confidence → PromptStereo AIF) replaced concatenation. Multiplicative or conditioned
   designs win the ablations (`block_matrix.md` §1).
2. **Frozen priors + small trainable adapters are standard** for depth/VFM priors
   (FoundationStereo freeze 1.97 vs unfreeze 3.94). Joint stereo+seg work has *not*
   adopted this: S3M/TiCoSS/TwInS all train end to end. A frozen shared segmentation trunk
   is the gap our design sits in.
3. **Generalisation is now the headline metric** (zero-shot KITTI/Middlebury/ETH3D/Booster),
   and it is driven by data, pseudo-labels and depth priors more than by modules. Joint
   stereo+seg papers still report mostly in-domain numbers.
4. **Real-time joint stereo+segmentation is nearly empty after RTS2Net (2020).** TwInS-Tiny
   (68M params, 22 FPS on an RTX 4090) is the fastest recent joint model and is not
   edge-real-time. VPEngine shows the multi-head-on-shared-backbone runtime pattern on Jetson
   but has no stereo head.
5. **The reverse direction (geometry → semantics) is growing**: TwInS CTA, SENSE SDAF (open-vocab
   seg conditioned on disparity), SDBF. It makes bidirectional claims attractive.
6. **Segmentation foundation models (SAM) in stereo stay niche** and weakly evidenced:
   - USAM-Net: negative result
   - AIO: SAM's role is qualitative only
   - SAOF, SMFormer, DEFOM+SAM3 pipelines: remote sensing, self-supervised or post-processing
7. **Evidence quality is low across the sub-field** (`evidence_audit.md`): no capacity-matched
   controls, test-split tuning, vKITTI2 random-pair leakage, per-image mIoU.

## 3. Competitor table for the manuscript (fill in our numbers on the same protocol)

| Method | Year | Shared encoder | Frozen? | Semantics → stereo mechanism | No-sem control? | Params | Speed (hardware, res) | Stereo eval protocol | Stereo numbers (as reported) |
|---|---|---|---|---|---|---|---|---|---|
| SegStereo | 2018 | ResNet-50 shallow | seg net frozen | concat at 1/8 + warp CE | no | n.s. | 0.6 s | KITTI15 fine-tuned | D1-all 2.25 (test) |
| SSPCV-Net | 2019 | ResNet-50 | no | semantic cost volume + FFM + boundary loss | no | n.s. | n.s. | KITTI15 fine-tuned | D1-all 2.11 (test) |
| SGNet | 2020 | PSMNet shallow | no | seg×disp correlation gate + class residual + losses | no | n.s. | 0.674 s (1080Ti) | KITTI15 fine-tuned | D1-all 1.99 (test) |
| RTS2Net c=8 / c=32 | 2020 | tiny CNN | no | synergy residual on cost volume | no | n.s. | 6.3 FPS TX2 / 0.02 s desktop | KITTI15 val / test | EPE 0.84 (val), D1-all 3.56 (test, c=32) |
| S3M-Net | 2024 | joint ~256-ch | no | add-fusion (FFA) + SCG loss | no | 347M (per TwInS Tab. VII) | 0.66 FPS | vKITTI2 random / KITTI 140-60 | EPE 0.38 / 0.55 |
| TiCoSS | 2025 | 3 shared layers | no | gated duplex + losses (seg-side) | no | 385M | 0.30 s (3090) | same | EPE 0.34 / 0.54 |
| TwInS-Tiny / Large | 2026 | ConvNeXt | no | GRU context + hidden init; geo→sem CTA | no | 68M / 276M | 22 / 18 FPS (4090, 640×320) | KITTI15 140 + pseudo-labels, 60 val | EPE 1.10 / 0.96 (val) |
| **Ours (E3)** | 2026 | YOLO26m 0–6 | **trunk, stereo, decoder frozen** | candidate gate on L-R class agreement + class residual | **yes (E4, C4 zeroed, C5 misaligned)** | fill: frozen + trainable | fill: RTX 3050 / Jetson | VKITTI2 grouped split; **KITTI15 zero-shot, 200 pairs** | KITTI15 all-valid EPE 2.5876 (ctrl 2.7139, stereo-only 3.0822) |

Protocols are not comparable. Put the protocol column in the paper and do not rank
across rows. TwInS has no released code.

## 4. Novelty claims: what holds and what doesn't

| Claim | Status | Why |
|---|---|---|
| "First joint stereo + semantic model with a shared encoder" | ✗ | RTS2Net 2020, S3M-Net, TwInS |
| "First real-time joint stereo + semantic model" | ✗ | RTS2Net 2020 (TX2) |
| "Semantic cost gating" / "class-conditioned residual" | ✗ alone | SGNet 2020. SemStereo SSR 2025 |
| "Frozen segmentation features help stereo" | ✗ alone | SegStereo 2018 |
| "One frozen pretrained detector/segmenter trunk shared by a frozen stereo predictor and a frozen seg decoder, with only a tiny fusion head trained" | ✓ (none found) | Joint works train end to end. Frozen-prior works use depth/VFM priors, not a shared seg trunk |
| "Semantic gain established against equal-capacity, zeroed and misaligned controls" | ✓ (strong) | No surveyed semantic-stereo paper has such controls |
| "Semantics improves zero-shot real-domain stereo, causally attributed" | ✓ if seeds hold | USAM-Net is negative. Others don't test it. Ours: 2.59 vs 2.71 ctrl, single seed |
| "Real-time on sub-4-GB hardware with no seg degradation" | ✓ if measured | Requires device latency (`deployment.md`). The frozen decoder guarantees unchanged mIoU, unlike RTS2Net's −1.99 |
| "Geometry improves semantics" | only if we add playbook 3.3 | TwInS/SENSE already show it; ours would be a frozen-decoder adapter variant |

## 5. Citation-graph search results (PARTIAL, unverified)

The Semantic Scholar citation pass was stopped early by rate limits: only the **SegStereo** (381
citers, 129 from 2023–2026) and **SSPCV-Net** (129 / 64) seeds were processed. SGNet,
RTS2Net, DispSegNet, S3M-Net, TiCoSS and SemStereo were **not** processed. The rows below
were filtered from titles and abstract snippets only. Venue, frozen status and real-time
status are unchecked. Raw JSON was in the session scratchpad and is not kept.

**Must check before any novelty claim** (joint or semantic-fused stereo, likely relevant):

| Title | Year | Venue | Note |
|---|---|---|---|
| SSNet: joint learning for semantic segmentation and disparity estimation | 2024 | The Visual Computer | Joint model. TiCoSS cites an "SSNet" with a single shared encoder. Check if it's the same |
| Efficient multi-task progressive learning for semantic segmentation and disparity estimation | 2024 | Pattern Recognition | "Efficient" multitask. Possible real-time competitor |
| Fast-Sesnet: Applying Semantic Information for Fast Deep Stereo Matching | 2025 | conference | Semantic + fast stereo. Possible real-time competitor |
| Semantic-Guided Stereo Matching Network Based on Parallax Attention and SegFormer | 2025 | CMC | SegFormer semantics guide stereo |
| LSEPNet: Joint Prediction of Disparity and Semantics Based on Binocular Vision | 2023 | ISCER | Joint, robots/driving |
| Trustworthy Perception for AD: Stereo-Semantic Fusion with XAI | 2026 | workshop | ResNet-18 U-Net depth + road seg, claims real time |
| Joint Depth Prediction and Semantic Segmentation with Multi-View SAM | 2023 | WACV-W | Multi-view, SAM features (arXiv 2311.00134) |
| Learning Representations from Foundation Models for Domain Generalized Stereo Matching | 2024 | ECCV | VFM prior in stereo (not joint seg) |

Off-domain (low priority): S2Net (TGRS 2024, satellite), MCF-SMSIS (surgical), CSC-MVS (remote-sensing MVS).
Metadata note: Semantic Scholar lists USAM-Net at ICONIP 2025 (the PDF on disk is arXiv). Unverified.

**To finish the search:** re-run the citation pull for the 6 remaining seeds with a longer
backoff or an API key, then read the full text of the "must check" rows. Until then, the
novelty statement in §4 is "none found in a partial search", not "none exist".

## 6. Candidates found but not downloaded

| Paper | Link | Note |
|---|---|---|
| Progressive Per-Branch Depth Optimization (DEFOM-Stereo + SAM3), Lin & Xue 2026 | https://arxiv.org/abs/2602.20539 | SAM3 masks as post-processing of a stereo foundation model. Not learned fusion |
| SMFormer, Wang et al. 2026 | https://arxiv.org/abs/2604.10218 | Self-supervised stereo with a SAM prior |
| MatchAttention, Yan et al. 2025 | https://arxiv.org/abs/2510.14260 | Efficient matching-constrained attention. General efficiency |
| Rethinking Monocular Depth Embedding for Generalized Stereo, Lin et al. 2026 | https://arxiv.org/abs/2607.09284 | Soft constraints / reduced coupling for mono priors |
| StereoDiffuer 2026 | https://arxiv.org/abs/2608.21710 | Saliency/boundary attention + diffusion refinement. Not real-time |
| H-Net / TransForSeg, Fekri et al. 2025 | https://arxiv.org/abs/2501.00514, https://arxiv.org/abs/2509.01605 | Joint stereo segmentation for catheters (medical) |
| S2Net, Liao et al., TGRS 2024 | closed access | Remote-sensing semantic stereo. Abstract-only summary in `paper/reference_papers/summaries/semantic_stereo/S2Net.md` |
