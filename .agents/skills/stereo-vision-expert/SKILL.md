---
name: stereo-vision-expert
description: Architectural and training knowledge for stereo disparity estimation — which head, loss, and fusion design solves which failure mode. Use when choosing a stereo head, debugging EPE spikes or edge blur, designing a shared semantic+disparity encoder, porting a loss recipe, quantizing or TensorRT-converting a stereo head, improving zero-shot generalization, cleaning training pairs, handling mirrors or occlusions, picking pretraining or augmentation, or deciding what to read next.
---

# Stereo Vision Expert

26 papers live in `paper/reference_papers/` (9 lightweight, 6 fusion, 11 semantic-stereo).
Per-paper detail lives in `references/` — read the matching file before citing specifics.
Paths below are relative to the repo root.

## When to load a reference file

- Picking or swapping a disparity head → `references/lightweight.md`
- Fusing mono/semantic priors, fixing dark/textureless/occluded regions → `references/fusion.md`
- Sharing one encoder between segmentation and disparity, joint losses → `references/semantic.md`
- Writing the training script (optimizer, schedule, aug, curriculum) → `references/training.md`
- Need a number for the paper (EPE/D1, latency, params) → the relevant file above

## Decision rules

**Head selection (defaults below are scoped to our rig: frozen semantic encoder, Driving-200 pairs, 4 GB GPU — re-verify elsewhere):**
1. Default: iterative GRU head (RAFT-family). On our Driving-200/A04 budget it won (5.87 vs 6.20/6.52 EPE) at the fewest trainable params (0.84M). Refinement beats capacity at this data scale.
2. If the GRU loop is unshippable — banned ops (RNN control flow, slicing, trilinear interp, deformable convs) for your TensorRT/edge SDK, or latency budget blown (rule of thumb: >50 ms on Orin-NX-class hardware at 320×640): use 2D channel-boost aggregation (LightStereo) + MSCA gating. Accept ~0.3–0.6 EPE penalty. If even 2D hourglasses are too heavy, DTPnet's all-2D channel-to-disparity + single hourglass is the floor.
3. If edges melt — diagnose by edge-region EPE (mask pixels within ~5 px of a GT disparity jump >4 px; suspect when edge-EPE exceeds flat-region EPE by >2×) or by a disparity-gradient error map: add bilateral-grid slicing (CUBG, ~4 ms, edge-EPE 5.95 vs 8.13 linear) or plane-tile upsampling instead of bilinear. Top-k regression (CoEx, k=2 — train with it, test-only swap fails) fixes bimodal boundary pixels.
4. Freeze the encoder first; train adapters + fusion only. All six fusion papers do this. Exception: joint-from-scratch works on tiny real datasets given an SCG-style boundary-weighted loss (S3M-Net, no SceneFlow pretrain). Encoder co-training is otherwise the last 5%, not the first 80%.
5. If disparity punches through mirrors/transparent surfaces: truncate the cost volume at the mono disparity (StereoAnywhere, threshold 0.98) so the mirror match cannot win.
6. If occlusions dominate error: SoftLRC entropy masking + LR-consistency check (StereoAnywhere; DispSegNet's double-warp `|I''−I|` + median-fill post-process).
7. If zero-shot transfer collapses (e.g. SceneFlow→KITTI/Driving drop): 3-stage curriculum — supervised synthetic → feature-alignment self-distill → real pseudo-labels (LiteAnyStereo); pseudo-label via teacher disparity vs mono-depth normal-consistency mask (Fast-FoundationStereo). Never distill RGB features alone — align in depth/normal/order space.
8. If mono/depth priors are affine-misaligned or early-iteration noise corrupts fusion: contribute structure as binary local-ordering maps with iteration-ramped Beta guidance (D-FUSE), not raw depth values.

**Loss recipe (port in this order, cumulative; weights are Driving-200/`experiments/a05_clean/losses.py:modal_loss(pred, gt, left)` tuning — re-tune per dataset):**
1. Smooth-L1 full-res — baseline, converges but plateaus ~6 px on our Driving-200.
2. Multi-scale L1 at 1/2 + 1/4 (weights 0.5/0.3) — teaches coarse structure, biggest single jump.
3. Gradient consistency (0.5) — sharpens edges without a refinement net.
4. Threshold hinge (0.2) + D1 hinge (0.2) — pulls down bad-1/bad-3/D1 directly.
5. Edge-aware smoothness (0.02) — small polish, keep weight tiny or it oversmooths poles/signs.
6. If semantics available: boundary loss coupling semantic edges to disparity jumps (SSPCV/SGNet form), masked to ignore fake boundaries (road/sidewalk/vegetation).
7. Iterative heads: L1 over iterates with γ=0.9 decay (RAFT-family standard). Later iterates weigh more.
8. Effect sizes are dataset-dependent — ablate each step (ΔEPE/Δbad1) before keeping it; the ordering above is by typical impact, not guaranteed impact.

**Killing EPE spikes (train curve spiky, val curve smooth):**
1. Score every pair with the latest checkpoint at full res — rank by EPE. Command pattern: load `runs/<run>/checkpoints/latest.pth` into the run's model, inference all pairs in `manifest_train.json + manifest_validation.json` at native res with replicate-pad stripped, compute `metrics()` per pair, sort descending. (A04b anecdote, not a universal law: our spikes came from 2/200 near-black tunnel frames, mean brightness ~30. Recalibrate thresholds per dataset.)
2. Pre-filter suspects by brightness (our cutoff mean < 45) and valid fraction (< 0.5), but confirm by inference — most dark frames are fine; only unmatchable ones spike.
3. Replace from the same sequence (sequence id = `record["sequence"]` in the split manifests; preserves stratification), verified bright + left/right/PFM all present. Do not delete — replace, or sequences unbalance.
4. If spikes persist after cleaning: suspect the loss, not the data. Single-scale L1 averages large-disparity regions into flat penalties; multi-scale terms fix this.

**Shared-encoder design (one YOLO backbone → seg + disparity):**
1. Keep semantic and disparity volumes separate, then attention-fuse (SSPCV's FFM). Naive concat/summation causes task conflict (TiCoSS Table V: gating recovers ~7.9% mIoU).
2. Gate per region (AIO's KeepTopK over priors; TGF's selective inheritance). Foreground/edges/dark areas want different cues.
3. Feed fusion from depth/normal/order space, not RGB (StereoAnywhere, D-FUSE). Relative geometry transfers; RGB doesn't.
4. Distinguish two frozen-fusion cases: frozen-VFM (SAM) early concat without a warp/aux loss hurts transfer (USAM-Net — avoid). Frozen-semantic concat PLUS a semantic-warp supervision loss fixes textureless regions (SegStereo — the warp loss is what makes it work, not the concat alone).
5. Adapter pattern: 1×1 per scale, kept wide (≥48 ch at 1/4; verify by ablation). Our A02B run had a 128→24 chokepoint at the correlation scale — general rule, local example.

**Training script defaults (distilled from all 26 papers; batch/VRAM/clip values after "our rig" are local):**
- Optimizer AdamW, LR 1–2e-4, weight decay 1e-4–1e-5, grad clip norm 1.0 (or clip values [−1,1]).
- Schedule: OneCycle (short runs) or cosine/step decay (long runs). Flat LR is fine past 20k only for BGNet-style KITTI finetune (constant LR, multi-seed, pick best); otherwise decay.
- Our rig: batch 1–8 with AMP fp16, 0.5–1.5 GB, clip norm 1.0. Scale up on bigger GPUs (batch 8–176 on A100s). Gradient accumulation if needed.
- Augmentation that preserves epipolar geometry: asymmetric brightness/contrast, color jitter, right-image eraser patches, y-offset ±2px, scale 0.9–1.15 with disparity rescaling. Photometric asymmetry is fine; geometric vertical flip and independent left/right warps are banned (they break epipolar lines).
- Pretrain on SceneFlow (FlyingThings; Driving/Monkaa can hurt per HITNet), finetune on target domain. Cityscapes-coarse→fine beats synthetic for KITTI transfer (RTS2Net).
- Disparity init matters for iterative heads: regularized/Gev-init beats zero-init; width-normalized init absorbs per-image amplitude variation (DEFOM).

## Comparison tables (EPE on SceneFlow unless noted; latency at KITTI res)

**Cost volumes** (detail: `references/lightweight.md`):
| type | exemplar | EPE | params | latency | when to pick |
|---|---|---:|---:|---:|---|
| none (tile match + propagate) | HITNet-XL | 0.36 | 2.07M | 20 ms | real-time + thin structures, no 3D ops |
| correlation + 2D boost | LightStereo-S | 0.73 | 3.44M | 17 ms | cheapest deployable, edge SDK-safe |
| correlation + tiny-3D + 2D | LiteAnyStereo | ~0.7 | ~2M | 21 ms | 2D budget but disparity continuity matters |
| group-wise corr + 3D hourglass | BGNet+ | ~0.9 | ~5M | 32 ms | accuracy headroom, 3D allowed |
| learned interlaced | MobileStereoNet-2D | 0.79 | 2.32M | — | 2D-only constraint, near-3D accuracy |
| all-pairs + GRU | RAFT-family | 0.44–0.72 | 1–11M | 50+ ms | best accuracy, iteration budget available |

**Upsampling to full res:**
| method | edge quality | cost | source |
|---|---|---:|---|
| bilinear | melts boundaries | ~0 | baseline |
| convex (RAFT) | crisp, learned weights | small conv | RAFT-Stereo |
| bilateral-grid slice (CUBG) | edge-EPE 5.95 vs 8.13 linear | ~4 ms | BGNet |
| plane-equation tile | preserves slanted surfaces | ~0 | HITNet |
| learned superpixel 3×3 | boundary-aware | 1 conv | CoEx |

**Iteration budget (accuracy vs latency):**
| model | iters | EPE | latency | source |
|---|---|---:|---:|---|
| Selective-IGEV | 12 | 0.44 | high | baseline |
| Pip-Stereo | 1 | 0.45 | 19 ms 4090 / 75 ms Orin | pruning works |
| MonSter++ | 4 (2+2) | 0.37 | — | beats 32-iter baseline |
| GGEV | 8 / 4 / 2 | 0.46 / 0.49 / 0.54 | 47 ms at 8 | graceful degradation |

## Paper index

Lightweight heads — detail in `references/lightweight.md`:
- HITNet (CVPR21) `lightweight/HITNet_Tankovich_CVPR2021.pdf` — tile hypotheses + propagation, no stored volume, 20 ms.
- LightStereo (ICRA25) `lightweight/LightStereo_Guo_ICRA2025.pdf` — 2D channel-boost aggregation + MSCA gating.
- CoEx (IROS21) `lightweight/CoEx_Bangunharcana_IROS2021.pdf` — guided excitation + top-k regression.
- BGNet (CVPR21) `lightweight/BGNet_Xu_CVPR2021.pdf` — bilateral-grid edge-aware cost upsampling.
- MobileStereoNet (WACV22) `lightweight/MobileStereoNet_Shamsafar_WACV2022.pdf` — 2D-vs-3D cost analysis, interlaced volume.
- LiteAnyStereo (arXiv25) `lightweight/LiteAnyStereo_Jing_arXiv2025.pdf` — tiny-3D + 3-stage distillation curriculum.
- Pip-Stereo (CVPR26) `lightweight/Pip-Stereo_Zheng_CVPR2026.pdf` — iteration pruning, 1-step GRU.
- GGEV (AAAI26) `lightweight/GGEV_Liu_AAAI2026.pdf` — frozen depth prior as guidance, dynamic per-plane kernels.
- DTPnet (ICRA24) `lightweight/Distill-then-Prune_Pan_ICRA2024.pdf` — hardware-first all-2D + logits KD + pruning.

Fusion priors — detail in `references/fusion.md`:
- MonSter++ (CVPR25) `fusion/MonSter_Cheng_CVPR2025.pdf` — bidirectional stereo↔mono refinement, shared frozen ViT.
- StereoAnywhere (CVPR25) `fusion/StereoAnywhere_Bartolomei_CVPR2025.pdf` — normals volume + depth-fed context encoder.
- DEFOM (CVPR25) `fusion/DEFOM-Stereo_Jiang_CVPR2025.pdf` — multiplicative scale update + trainable DPT adapter.
- D-FUSE (ICCV25) `fusion/D-FUSE_Yao_ICCV2025.pdf` — ordering-map fusion, iteration-ramped guidance.
- AIO-Stereo (AAAI25) `fusion/AIO-Stereo_Zhou_AAAI2025.pdf` — multi-VFM distill + MoE selection (DINO/SAM/DA).
- Fast-FoundationStereo (CVPR26) `fusion/Fast-FoundationStereo_Wen_CVPR2026.pdf` — distill→NAS→prune→pseudo-label compression playbook.

Semantic-stereo sharing — detail in `references/semantic.md`:
- SegStereo (ECCV18) `semantic_stereo/SegStereo_Yang_ECCV2018.pdf` — frozen PSPNet + early concat + warp loss.
- SSPCV-Net (ICCV19) `semantic_stereo/SSPCV-Net_Wu_ICCV2019.pdf` — separate volumes + attention fusion + boundary loss.
- DispSegNet (RAL19) `semantic_stereo/DispSegNet_Zhang_RAL2019.pdf` — semantic residual refinement + segment smoothness.
- SGNet (ACCV20) `semantic_stereo/SGNet_Chen_ACCV2020.pdf` — confidence gating + per-class depthwise residual.
- RTS2Net (ICRA20) `semantic_stereo/RTS2Net_Dovesi_ICRA2020.pdf` — real-time shared encoder + anytime exits.
- S3M-Net (TIV24) `semantic_stereo/S3M-Net_Wu_TIV2024.pdf` — FFA bridging + SCG boundary-weighted loss.
- SemStereo (AAAI25) `semantic_stereo/SemStereo_Chen_AAAI2025.pdf` — deep cascade + semantic-gated residual + warp CE.
- TiCoSS (TASE25) `semantic_stereo/TiCoSS_Tang_TASE2025.pdf` — gated fusion + inconsistency-weighted supervision (cautionary).
- S3Net (IGARSS24) `semantic_stereo/S3Net_Yang_IGARSS2024.pdf` — single-branch single-volume multitask.
- USAM-Net (arXiv25) `semantic_stereo/USAM-Net_Sankaranarayanan_arXiv2025.pdf` — frozen-SAM early fusion (cautionary).
- SDBF-Net (APSIPA19) `semantic_stereo/SDBF-Net_Rao_APSIPA2019.pdf` — bidirectional late residual fusion.
