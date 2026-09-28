# Fusion: monocular/semantic priors into stereo

Papers in `paper/reference_papers/fusion/`. All six freeze the foundation/VFM encoder and train adapters + fusion only — that is the house rule.

## MonSter++ (CVPR 2025)
`fusion/MonSter_Cheng_CVPR2025.pdf` — file contains the MonSter++ extension manuscript.
- **Problem:** stereo/MVS fail ill-posed regions; mono priors have scale/shift ambiguity. Unifies rectified stereo + MVS as scale/shift recovery from relative depth.
- **Architecture:** frozen DepthAnythingV2 (ViT-L 335M + DPT). Stereo branch: shared frozen DINOv2 ViT → 2D-conv transfer net (pyramid 1/4–1/32) → GEV + all-pairs correlation → ConvGRU. Mutual refinement: least-squares global scale+shift fit (20–90th pct pixels) → Stereo-Guided Alignment (per-pixel shift GRU) → Mono-Guided Refinement (residual disparity GRU). RAFT convex upsample. RT variant: coarse-to-fine 1/16→1/4, single-layer GRU, 47 ms / 2 GB at 1K.
- **Loss:** L1 both branches, exponential weights γ=0.9 over iterates.
- **Training:** AdamW one-cycle 2e-4, batch 8, 200k SceneFlow; grad clip [−1,1]; ViT frozen. Two-stage curriculum (Basic → Full sets, >2M pairs). KITTI 50k. 4 iters already beats 32-iter baseline.
- **Tricks:** (1) bidirectional SGA↔MGR beats one-way fusion (~15% combined); (2) confidence-gated conditioning via warp-residual avoids noisy stereo injection; (3) frozen shared ViT + transfer net = free foundation features (+5% EPE).
- **Numbers:** SceneFlow 0.37. KITTI15 D1-all 1.37; ETH3D bad1 0.25; Middlebury bad4 1.18. #1 on all four leaderboards.
- **Relevance:** direct proof a frozen encoder can be shared between semantic/mono and stereo heads with only a light conv transfer net, with ablations quantifying the gain.

## StereoAnywhere (CVPR 2025)
`fusion/StereoAnywhere_Bartolomei_CVPR2025.pdf`
- **Problem:** stereo fails textureless/occluded/mirror surfaces; mono VFMs fail perspective illusions. Synthetic-only training.
- **Architecture:** dual-branch RAFT skeleton. Stereo: frozen RAFT CNN encoder → all-pairs volume + pyramid. Mono: frozen DA-V2 on L+R → surface-normal maps → normals correlation volume, 8 depth-quantile masked sub-volumes → 3D-CNN regularizer (CoEx-style excitation) → disparity + confidence heads. Differentiable scaler: softargmax → entropy confidence → SoftLRC masking → weighted least-squares scale+shift. Fusion: dual-lookup Multi-GRU init at scaled mono; convex upsample. Context encoder retrained on mono depth, not RGB.
- **Loss:** RAFT-style L1 over iterates + L1 on coarse/mono + normal loss (weight 10) + BCE on confidences vs SoftLRC targets. Unweighted sum of seven terms.
- **Training:** from SceneFlow RAFT checkpoint; single A100, 3 epochs, AdamW 1e-4, batch 2, crop 320×640; VFM + image encoder frozen; GRU 12 train / 32 infer. Volume Rolling/Noising/Zeroing augments + mirror truncation.
- **Tricks:** (1) normals (not depth) correlation = texture/view-invariant; (2) entropy + SoftLRC confidence makes scale/shift outlier-proof; (3) volume augments teach the GRU when to trust mono vs stereo.
- **Numbers:** zero-shot Middlebury-H bad2 6.96; ETH3D bad1 1.66; KITTI15 bad3 3.93; Booster-Q bad2 9.01.
- **Relevance:** context features best drawn from mono/depth representation, not RGB; frozen dual encoder + learned fusion GRU generalizes — keep the shared encoder frozen, train fusion.

## DEFOM-Stereo (CVPR 2025)
`fusion/DEFOM-Stereo_Jiang_CVPR2025.pdf`
- **Problem:** recurrent stereo breaks on occlusion/textureless/blur/high-res; DEFOM depth robust but scale-inconsistent.
- **Architecture:** simplified RAFT base (2-level pyramid, radius 4). Frozen DEFOM ViT + new trainable DPT head (original fixed) → fused map added to CNN map (feature encoder); reassembled DPT maps added to context encoder. All-pairs volume + 2-level pyramid. Scale Update (multiplicative scale map via scale lookup: 8 factors × ±1 neighbors = 24 samples, full-image range) then Delta Update (standard GRU + Δd head + convex upsample). 8 SU + 10 DU train, 8+24 eval. +7.4M params, +15% time.
- **Loss:** pure RAFT-style L1 over all iterates, γ=0.9. No auxiliary terms.
- **Training:** AdamW one-cycle 2e-4, batch 8, SceneFlow 200k crop 320×736; DEFOM frozen. KITTI 50k; Middlebury 200k+100k; ETH3D 300k+90k.
- **Tricks:** (1) multiplicative scale update converts affine mono depth to metric disparity globally; (2) trainable duplicate DPT head reshapes frozen ViT features without corrupting depth; (3) width-normalized init absorbs per-image amplitude variation.
- **Numbers:** SceneFlow 0.42 (ViT-L). KITTI15 D1-all 1.33; ETH3D bad1 0.70; many #1 spots + best RVC joint model.
- **Relevance:** minimal recipe (frozen ViT + small trainable adapter + conv add-fusion) transferable to grafting YOLO semantic features into stereo.

## D-FUSE (ICCV 2025)
`fusion/D-FUSE_Yao_ICCV2025.pdf`
- **Problem:** affine-misaligned mono depth; mono features cause overconfident local optima; early-iteration noise misleads fusion.
- **Architecture:** RAFT-Stereo backbone + frozen DA-V2 (mono features → hidden-state init + context). Iterative Local Fusion: LBP-like binary local-ordering maps from mono depth and current disparity → convs predict Beta(α,β) guidance; multi-level GRU predicts Δd; update ramps guidance with iteration (dodges early noise). Global Fusion: conv net regresses (a,b) registering mono to disparity; confidence blending with cost-volume confidence.
- **Loss:** L1 γ=0.9 over iterates + γ on registered mono + final fused.
- **Training:** 4×A40, AdamW one-cycle, frozen DA-V2; 3 frozen-staged phases on SceneFlow (fusion → registration → global, 100k each).
- **Tricks:** (1) ordering maps unify relative+absolute depth, immune to affine shifts; (2) iteration-ramped Beta guidance escapes local optima; (3) final noisy-linear registration preserves fine mono shape.
- **Numbers:** zero-shot KITTI15 EPE 1.12; Middlebury-H 1.15; ETH3D 0.25; Booster-Q 2.26; DrivingStereo-rainy best.
- **Relevance:** ordering maps let a shared encoder contribute relative structure without metric calibration — ideal when its depth output is affine-invariant; staged freezing protocol for adding fusion.

## AIO-Stereo (AAAI 2025)
`fusion/AIO-Stereo_Zhou_AAAI2025.pdf`
- **Problem:** weak CNN encoders trained on small synthetic data; fail dark/low-texture areas.
- **Architecture:** Selective-IGEV base (all-pairs volume + pyramid + GRU) + 3-block ResNet context net. Teachers: DINOv2 (foreground), SAM (edges/small objects), DA-V2 (dark depth). Per-block per-VFM expert net + heavier alignment net (bridges CNN↔ViT); MSE distillation; experts added back into stream. Gating net → softmax → KeepTopK per-pixel weights (MoE-style). Inference = student CNN only.
- **Loss:** RAFT-style L1 + Σ γ-weighted MSE distillation per block. γ values not stated.
- **Training:** A100, AdamW. SceneFlow 200k batch 8 crop 320×720, one-cycle 1% warmup to 2e-4. Finetune linear decay 3e-4→0. Asymmetric LR (alignment net higher + faster decay).
- **Tricks:** (1) dual use (distill AND forward-fuse) prevents forgetting; (2) per-pixel KeepTopK resolves VFM conflicts; (3) asymmetric LR bridges heterogeneity. Roles confirmed: DINO→foreground, SAM→edges, DA→dark.
- **Numbers:** Middlebury bad2 2.36 (#1); ETH3D bad1 0.94; KITTI15 D1-all 1.54.
- **Relevance:** strongest precedent for one CNN encoder absorbing heterogeneous priors (incl. segmentation model SAM) via adapters + gating, zero inference overhead — blueprint for YOLO-semantic distillation.

## Fast-FoundationStereo (CVPR 2026)
`fusion/Fast-FoundationStereo_Wen_CVPR2026.pdf`
- **Problem:** foundation stereo generalizes zero-shot but costs ~500 ms; real-time nets need per-domain finetuning.
- **Architecture:** FoundationStereo skeleton (GW-correlation + concat volume, dual-branch 3D filter with axial-planar convs + disparity transformer, ConvGRU). Acceleration: backbone distillation (frozen DA-V2 + side-tuning teacher → efficient student via MSE) → blockwise NAS (filter split at channel transitions, per-block MSE, ILP selection under latency budget) → structured ConvGRU pruning (Taylor importance) + retrain. 1.4M-pair pseudo-label pipeline (Stereo4D + normal-consistency mask, sky excluded, sky disp 0).
- **Loss:** distillation MSE; smooth-L1 vs GT at final block; prune-retrain loss (γ=0.9, λ=0.1).
- **Training:** mixed synthetic + 1.4M pseudo-labeled real; per-budget family members. Optimizer/LR not stated.
- **Tricks:** (1) blockwise distill-then-ILP makes 10^18 NAS tractable; (2) normal-space teacher↔mono consistency robust to wild depth ranges; (3) output-space pseudo-label distillation complements feature distillation.
- **Numbers:** zero-shot Middlebury-H BP-2 2.20 (teacher 1.10); ETH3D 0.62; Booster-Q 6.61 (best real-time). 49 ms (21 ms TensorRT) vs 496 ms teacher.
- **Relevance:** full compress-a-foundation-model playbook (distill→NAS→prune→pseudo-label) for getting a shared YOLO+disparity encoder to real-time.
