# Lightweight stereo heads

Papers in `paper/reference_papers/lightweight/`. Read the PDF for anything beyond this summary.

## HITNet — Hierarchical Iterative Tile Refinement (CVPR 2021, Google)
`lightweight/HITNet_Tankovich_CVPR2021.pdf`
- **Problem:** real-time stereo without 3D cost volumes, which are accurate but 100 ms–1.8 s and lose thin structures.
- **Architecture:** tiny U-Net encoder (5 scales, 16/16/24/24/32 ch; L/XL 32–64 ch). No stored volume: fused single-op exhaustive match, L1 cost, argmin init. Tile hypothesis `h=[d,dx,dy,p]` (slanted plane + learned descriptor). Propagation module (1×1 reduce + residual blocks, dilated convs), warp + local volume at d−1/d/d+1, delta + confidence heads. Hierarchical M→0 with winner-take-all on confidence; plane-equation upsampling. Params: base 0.66M, XL 2.07M.
- **Loss:** `L_init+L_prop+L_slant+L_w`. Init: L1-contrastive (margin 1). Prop: Barron robust loss on clamped error. Slant: gradients from 9×9 robust plane fit. Confidence hinge: boost err<1, suppress err>1.5.
- **Training:** Adam. SceneFlow FlyingThings only (Driving/Monkaa hurt): crop 320×960, batch 8, Dmax 320, 1.42M iters, LR 4e-4→1e-4→4e-5→1e-5 at 1M/1.3M/1.4M. KITTI from scratch (no pretrain). Aug: symmetric bright/contrast, asymmetric ±5%, right-patch transplant, y-offset ±2px, Gaussian noise.
- **Tricks:** (1) exhaustive match without storing volume; (2) slanted tiles + descriptors = sparse learnable 3D volume; (3) high-res init injected at every scale recovers thin structures.
- **Numbers:** SceneFlow XL 0.36 EPE; KITTI15 D1-all 1.98, 20 ms; ETH3D EPE 0.20.
- **Shared-encoder relevance:** strongest no-3D-volume template — U-Net features + tile/warp/propagate head could hang off a YOLO backbone; plane upsampling preserves seg-aligned edges.

## LightStereo — Channel Boost Is All You Need (ICRA 2025)
`lightweight/LightStereo_Guo_ICRA2025.pdf`
- **Problem:** 2D aggregation is fast but inaccurate; can a 2D encoder-decoder suffice?
- **Architecture:** MobileNetV2 (H: EfficientNetV2) ImageNet backbone; 1/4–1/32 pyramid to 1/4. Correlation volume at 1/4. Aggregation: 2D inverted-residual V2 blocks at 1/4,1/8,1/16 (S blocks 1/2/4 exp 4; M 4/8/14; L 8/16/32 exp 8). MSCA: strip convs (7/11/21) → mixer → multiply with cost; stops propagation at discontinuities. Soft-argmax + upsample.
- **Loss:** single smooth-L1 on final disparity. No deep supervision.
- **Training:** AdamW + OneCycle, maxLR 1e-4×batch. SceneFlow 90 epochs (S batch 24 / M 12 / L 8), crop 320×736 only. KITTI 500 epochs batch 2 from SceneFlow weights.
- **Tricks:** (1) expanding disparity channels beats larger spatial kernels (0.867→0.714 EPE, fewer FLOPs); (2) MSCA strip attention as cheap semantic gate (+0.03 EPE, +0.5G FLOPs); (3) full V2-vs-V1-vs-ViT expansion study.
- **Numbers:** SceneFlow S 0.73 / M 0.62 / L 0.59 / H 0.51. KITTI15 D1-all S 2.30 / H 1.82. S: 3.44M, 17 ms.
- **Relevance:** tests the shared-encoder hypothesis directly — frozen efficient backbone + 2D head + semantic attention; MSCA could take YOLO features.

## CoEx — Correlate-and-Excite (IROS 2021)
`lightweight/CoEx_Bangunharcana_IROS2021.pdf`
- **Problem:** volumetric aggregation heavy; spatially-varying complements add too much compute.
- **Architecture:** MobileNetV2 + U-Net skips; correlation at 1/4 (48 levels). Slim GC-Net hourglass 3D (8→16→32→48 ch). Guided excitation: image feat → 1×1 → sigmoid → broadcast-multiply over disparity per pixel. Top-k soft-argmax (k=2) + learned 3×3 superpixel upsample.
- **Loss:** single smooth-L1 full-res.
- **Training:** Adam + SWA. SceneFlow crop 576×288 batch 8, 10 epochs, 1e-3→1e-4. KITTI 800 epochs from SceneFlow weights. 2.7M params, 27 ms.
- **Tricks:** (1) excitation ≫ additive skip (0.685 vs 0.731); (2) top-k regression handles bimodal distributions (must train with it, test-only swap fails); (3) correlation+GCE matches concatenation accuracy at far lower cost.
- **Numbers:** SceneFlow 0.69. KITTI15 D1-all 2.13. 3.3× faster than AANet at −0.18 EPE.
- **Relevance:** GCE is the minimal shared-encoder interface — excite disparity channels with YOLO features at near-zero cost; top-k is a drop-in boundary fix.

## BGNet — Bilateral Grid Learning (CVPR 2021)
`lightweight/BGNet_Xu_CVPR2021.pdf`
- **Problem:** 3D at high res is slow; low-res + bilinear upsample blurs edges.
- **Architecture:** ResNet-like to 1/8 (352 ch) + 2 hourglasses. Group-wise correlation (44 groups). Two 3D convs + single 3D hourglass. CUBG: 3×3×3 to bilateral grid W/8×H/8×D/8×32; guidance map from high-res features via 2× 1×1; parameter-free trilinear slicing.
- **Loss:** single smooth-L1 final.
- **Training:** Adam, one-cycle max 1e-3, batch 16, crop 512×256, 50 epochs. Aug: asymmetric chromatic, y-disparity, blur, zoom; drop pairs >25% disp>300. KITTI finetune const LR 300 epochs ×3, pick best. 25 ms (BG step 4.3 ms).
- **Tricks:** (1) learned grid + slicing = edge-aware upsample (edge EPE 5.95 vs 8.13 linear); (2) plug-in accelerator — GCNet 29×, PSMNet 4.9× with equal/better accuracy; (3) learned guidance ≫ luma (1.17→1.28 if swapped).
- **Numbers:** SceneFlow 1.17 (linear 1.40). KITTI15 D1-all 2.51 (BGNet+ 2.19). Best <50 ms at the time.
- **Relevance:** aggregate cheap at 1/8, slice to 1/2 guided by YOLO edge map — crisp seg-aligned boundaries for ~4 ms.

## MobileStereoNet (WACV 2022)
`lightweight/MobileStereoNet_Shamsafar_WACV2022.pdf`
- **Problem:** 3D nets OOM on embedded; need light 2D and 3D variants + cost analysis.
- **Architecture:** shared ResNet-like backbone. 2D: learnable Interlaced_i volume (interlace L/R → 3D convs, best i=4) + 3 stacked 2D hourglasses with v2 blocks. 3D: Gwc40 volume + 3 hourglasses with 3D-raised v2. Keep channel-reduction convs standard (lightening hurts); FE→v1 + hourglass→v2 is the sweet spot.
- **Loss:** smooth-L1; hourglass weighting not stated.
- **Training:** Dmax 192, SceneFlow 960×540; KITTI15 159/40 finetune. Optimizer/LR/batch not stated.
- **Tricks:** (1) raising v1/v2 to 3D saves more than 2D; (2) interlaced cost lets 2D approach 3D; (3) selective replacement analysis.
- **Numbers:** SceneFlow 2D 0.79 (2.32M) / 3D 0.66 (1.77M) vs GwcNet 0.62 (6.43M). KITTI15 2D 2.83 / 3D 2.10.
- **Relevance:** quantifies the 2D-vs-3D head trade (2D fewest ops, 3D fewest params/crisper); reusable blocks for a YOLO-shared head.

## LiteAnyStereo (arXiv 2025)
`lightweight/LiteAnyStereo_Jing_arXiv2025.pdf`
- **Problem:** efficient models lack zero-shot generalization.
- **Architecture:** MobileNetV2 backbone, 1/4–1/32 pyramid to 1/4. Correlation at 1/4. Hybrid 3D→2D aggregation (serial best; 3D only 4.8% of budget; ConvNeXt 2D layers). Soft-argmax + convex upsample. No DepthAnything (overhead).
- **Loss:** stage1 smooth-L1; stage2 + feature-alignment cosine loss vs frozen teacher; stage3 L1 vs FoundationStereo pseudo-labels.
- **Training:** AdamW, one-cycle peak 2e-4, batch 176, 150K+50K+100K steps. 1.8M synthetic stage1; 0.5M unlabeled real stage3. Fixed teacher beats EMA.
- **Tricks:** (1) tiny-3D preserves disparity continuity pure-2D loses, at negligible cost; (2) 3-stage curriculum (synthetic → self-distill → real KD), transferable to LightStereo-M; (3) data quality > scale curation.
- **Numbers:** zero-shot KITTI15 D1 3.87 EPE 0.99; Middlebury-H bad2 7.51; 33G vs Selective-IGEV 3619G. 21 ms.
- **Relevance:** frozen/shared semantic encoder + small head suffices for zero-shot given feature-alignment + pseudo-label training.

## Pip-Stereo (CVPR 2026)
`lightweight/Pip-Stereo_Zheng_CVPR2026.pdf`
- **Problem:** iterative GRU refinement is accurate but RNN control-flow/quantization/memory stalls edge deployment.
- **Architecture:** RepViT student backbone (supernet + genetic search emphasizing 1/4 features). Teacher DepthAnythingV2-L in training only. Selective-IGEV context + importance map. Successive halving 32→1 iters; only GRU finetuned during pruning. FlashGRU fallback (top-k 70% sparsity + fused kernels).
- **Loss:** stage1 MSE on context/cost embeddings; stage2 skip-step equivalence + final + hidden alignment (equal weights).
- **Training:** 8×4090. Stage1 200K LR 2.5e-4 batch 24; stage2 update-block-only 50K LR 2e-4 batch 64.
- **Tricks:** (1) hit-ratio >0.99 after ~10 iters — <1% pixels update by iter 32, motivates pruning; (2) monocular priors transfer without keeping the ViT; (3) FlashGRU 7.3× RNN speedup at 2K.
- **Numbers:** SceneFlow 0.45 at 1 iter (vs Selective-IGEV 0.44 at 12). 75 ms Orin-NX, 19 ms 4090.
- **Relevance:** iterative bias aids generalization even distilled to 1 step; single-pass GRU template for an edge refinement head on YOLO features.

## GGEV (AAAI 2026)
`lightweight/GGEV_Liu_AAAI2026.pdf`
- **Problem:** real-time nets fail unseen occlusions/textureless; MFM-augmented models are slow and scale-shifted.
- **Architecture:** texture MobileNetV2 (1/4,1/8,1/16) + frozen DepthAnythingV2-Small (1/2–1/16). Selective Channel Fusion 1×1 → depth-aware prior. Group-wise correlation (8 groups). Depth-aware dynamic cost aggregation: per-plane Q/K affinity → dynamic K×K kernels. Single-layer ConvGRU (train 11, infer 8 iters). Depth-guided upsampling.
- **Loss:** smooth-L1 on d0 + Σ γ^(N−i)·L1 over iters (γ=0.9).
- **Training:** AdamW + grad clip + one-cycle. SceneFlow crop 320×768 batch 12 + asymmetric aug. KITTI 50K. 3.68M trainable, 47 ms.
- **Tricks:** (1) frozen MFM features only as guidance (sidesteps scale-shift, +2% params); (2) disparity-wise dynamic kernels; (3) wins at fewer iters than RT-IGEV.
- **Numbers:** SceneFlow 0.46. Zero-shot KITTI15 5.56, ETH3D bad1 2.84 (−51% vs RT-IGEV). KITTI15 test D1-all 1.70 (SOTA real-time).
- **Relevance:** prototype for injecting priors without a second heavy encoder — SCF fusion + dynamic kernels + depth-init GRU replicable with YOLO features.

## DTPnet — Distill-then-Prune (ICRA 2024)
`lightweight/Distill-then-Prune_Pan_ICRA2024.pdf`
- **Problem:** 3D convs/slicing/iterative convs unsupported by TensorRT/edge SDKs.
- **Architecture:** siamese 2-block pyramid → 1/4 feats. Channel-to-disparity (concat → 3-layer 2D conv). Single stacked hourglass. Conv D/4→D + bilinear (no trilinear). Soft-argmax. Post-prune 0.26M.
- **Loss:** logits-only KD L1 on temperature-softened distributions (t annealed 0.5→1.0). Teacher-only beats GT-only and mixed. L1 beats KL. Feature-KD untrainable for stereo.
- **Training:** student AdamW LR 1e-3 wd 1e-2. SceneFlow 20 epochs, KITTI 300+300. DepGraph structured pruning 50% (rate 0.1×5) with per-step distillation finetune.
- **Tricks:** (1) hardware-first: enumerate unsupported ops, then replace; (2) logits-only teacher-only distillation; (3) distill→prune loop with coupling-aware pruning.
- **Numbers:** SceneFlow 1.56, 8 ms Titan-XP (fastest). KITTI15 D1-all 3.28.
- **Relevance:** all-2D + single hourglass + bilinear fits TensorRT/NPU on a shared YOLO encoder; teacher-only KD + 50% pruning recipe for compressing the disparity head.
