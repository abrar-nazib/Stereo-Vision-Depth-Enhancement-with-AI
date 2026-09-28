# Semantic-stereo sharing: one encoder, two tasks

Papers in `paper/reference_papers/semantic_stereo/`. These are the most relevant to shared-encoder work — the fusion pattern that works (and the two cautionary tales) are marked.

## Patterns that work

### SegStereo (ECCV 2018) — frozen semantics + early concat + warp loss
`semantic_stereo/SegStereo_Yang_ECCV2018.pdf`
- Encoder: ResNet-50 to conv3_1 shared (1/8); PSPNet-50 conv5_4 semantic, frozen during disparity training.
- Volume: 1D correlation (DispNetC-style), displacement 24 (corr13 variant 96). 1×1 transform on left feats; concat [left, corr, semantic] → 12 dilated residual blocks → 3 deconv + regression to full-res.
- Loss: unsupervised `1.0·photo + 10.0·semantic-warp + 0.1·smooth`; supervised `1.0·L1 + 0.1·smooth + 1.0·CE`. Warp loss backprops through warping + classifier into disparity.
- Training: Caffe SGD poly LR base 0.01, crop 513, batch 16. Curriculum: Cityscapes (+19,997 SGM disparities) 90K → +FlyingThings3D 500K (seg frozen) → KITTI 90K. Aug: resize 0.5–2.0, RGB ±10, bright ±5, contrast 0.8–1.2.
- Tricks: (1) two complementary injections (early concat + late warp supervision); (2) frozen features help even without seg labels at finetune; (3) same plumbing works unsupervised and supervised.
- Numbers: KITTI15 test D1-all 2.25%; FlyingThings 1.45 EPE (vs 3.50 baseline).
- Relevance: most direct precedent — frozen semantic backbone + concat at 1/8 + warp loss fixes textureless road errors; template for YOLO shallow-layer sharing.

### SSPCV-Net (ICCV 2019) — separate volumes, attention-fused
`semantic_stereo/SSPCV-Net_Wu_ICCV2019.pdf`
- Encoder: ResNet-50 dilated + 3-scale pooling; PSPNet seg head.
- Volumes: 3-level spatial pyramid (1/4,1/8,1/16, GC-Net style) + semantic volume from pre-classifier features at largest size.
- Aggregation: 3D hourglass + SENet-style 3D Feature Fusion Module (pool → FC-ReLU-FC-sigmoid → weighted sum); recursive low→high, then fuse semantic, upsample. Soft-argmin, Dmax 256/192.
- Loss: `0.9·smooth-L1 + 0.1·boundary` where boundary = `|∇sem|·exp(−|∇d|)` — disparity must jump where semantics jump.
- Training: PyTorch 2×1080, Adam, batch 2, crops 256×512/792. Two-stage: seg 40 epochs → joint 40. SceneFlow ~120h.
- Tricks: (1) separate volumes + learnable fusion beats single concat; (2) 3D attention fusion over naive concat; (3) boundary loss couples edges to disparity jumps.
- Numbers: SceneFlow 0.87 (vs PSMNet 1.09); KITTI15 test D1-all 2.11%.
- Relevance: why to keep volumes separate then fuse; joint training helps even with coarse semantics; informs YOLO dual-volume + FFM design.

### SGNet (ACCV 2020) — confidence gating + per-class residual
`semantic_stereo/SGNet_Chen_ACCV2020.pdf`
- Base PSMNet (shared shallow layers + 2-block semantic branch). Confidence module: multiply disparity-correlation × semantic-correlation → 3D convs → sigmoid → correct disp1 volume. Residual module: category-wise disparity → depthwise conv per channel + pointwise → residual → final. Masks ignore fake boundaries (road/sidewalk/vegetation); thresholds ignore disparity-semantic mismatches.
- Loss: `Ldisp (0.5/0.7/1.0 staging) + CE + 0.5·boundary + 0.5·smooth`, λ=3.
- Training: Adam batch 2 crop 256×512. SceneFlow 15 epochs disparity-only → KITTI 600+100. Negligible overhead (0.674 vs 0.671 s).
- Tricks: (1) multiplicative correlation vetoes cross-class matches; (2) per-class depthwise residual (road flat vs tree uneven); (3) masked losses handle layout mismatches.
- Numbers: KITTI15 test D1-all 1.99%.
- Relevance: most modular upgrade for a YOLO-shared encoder — cheap gating + per-class residual + masked losses, no runtime cost.

### RTS2Net (ICRA 2020) — real-time shared encoder, anytime exits
`semantic_stereo/RTS2Net_Dovesi_ICRA2020.pdf`
- Shared encoder (2×3×3 + 4 pool blocks, width hyperparam c) → 1/4–1/32. Disparity decoder 3 stages (distance volume dmax 12, then warp-residual dmax 2) + soft-argmin + bilinear. Symmetric semantic decoder. Synergy refinement: compress semantic to volume dim, concat, 2D convs → residual.
- Loss: hierarchical `Σ Wst·(1·L1 + 2·wCE + 2·refine)`, Wst = 1/4,1/2,1. Class-balanced CE + coarse-label reweight.
- Training: crop 256×512 batch 6, Adam 5e-4. Curriculum: Cityscapes-coarse 60 → fine 75 → KITTI 800. Cityscapes pretrain beats SceneFlow for KITTI transfer.
- Tricks: (1) fully residual coarse-to-fine enables anytime exit (stage2 hits 10 FPS); (2) subtraction volume + tiny dmax = minimal memory; (3) Cityscapes beats synthetic pretraining.
- Numbers: KITTI15 test D1-all 3.56%, 0.02s (7–90× faster than GANet/PSMNet). TX2 6.3 FPS full, 10.9 stage2.
- Relevance: real-time template — shared light encoder + symmetric heads + synergy concat; joint training helps even at tiny width; YOLO-scale backbones can hit TX2 rates.

### S3M-Net (TIV 2024) — FFA bridging + boundary-weighted loss
`semantic_stereo/S3M-Net_Wu_TIV2024.pdf`
- Joint encoder (~256 ch) → all-pairs correlation + pyramid + multi-level GRU from D0=0 (RAFT-inspired). FFA: remap shared to 64/256/512; disparity encoder ResNet-152; semantic encoder to 1024/2048; fuse by addition (beats concat/gates). SNE-RoadSeg decoder.
- Loss: SCG — boundary weight `W` from pooled one-hot labels emphasizes both CE and disparity L1 at boundaries (α=0.1, γ=0.9).
- Training: RTX3090 batch 1, crop 1000×320, AdamW 2e-4, 100K vKITTI2 + 20K KITTI. Fully supervised end-to-end, no SceneFlow pretrain. 0.66 FPS (needs optimization).
- Tricks: (1) FFA bridges 256-vs-2048 channel gap; (2) SCG tames LiDAR sparsity; (3) joint regularization works with tiny real data.
- Numbers: vKITTI2 EPE 0.38 (vs RAFT 0.40); KITTI 0.55 (vs RAFT 0.60). Seg mIoU KITTI 57.80 (+4.84).
- Relevance: newest shared-encoder proof — disparity feeds back via FFA-addition to boost seg; SCG loss is plug-and-play for YOLO multitask boundaries.

### SemStereo (AAAI 2025) — deep cascade + gated residual + warp CE
`semantic_stereo/SemStereo_Chen_AAAI2025.pdf` (remote sensing)
- Shared MobileViTv2 U-shape (1/2–1/32). Semantic head on deepest volume. Fast-ACV attention-concatenation volume, range [−Dmax,Dmax). SSR: semantic-gated channel-attention residual on disparity. LRSC: disparity-warped cross-view semantic CE (works self-supervised).
- Loss: `CE+Dice + Σ λ·smoothL1 (1/0.6/0.5/0.3) + 1·seg + 1·LRSC`.
- Training: 2×A40, Adam, batch 4, full resolution, no augmentation, 48 epochs/stage, 0.001 halved at 12/22/30/38/44.
- Tricks: (1) feed semantic-enriched deep (not shallow) features to stereo; (2) per-class gating exploits narrow intra-class disparity range; (3) LRSC works without GT.
- Numbers: US3D EPE 0.958 (vs Fast-ACV 1.171); no-label variant 0.996.
- Relevance: SSR + LRSC heads are lightweight add-ons portable to a YOLO encoder with disparity branch.

### S3Net (IGARSS 2024) — single-branch single-volume multitask
`semantic_stereo/S3Net_Yang_IGARSS2024.pdf` (satellite)
- Siamese encoder, 4× downsample, Self-Fuse + concat. Selective 4D stacking (top slice reserved for semantics). MFM: 3 unshared rounds of 3D fusion with cross-scale skips. SFM per-channel gating. Trilinear (disp) + bilinear (class) heads.
- Loss/optimizer/LR: not stated.
- Tricks: (1) one volume carries both tasks; (2) SFM gating denoises; (3) MFM mutual fusion.
- Numbers: US3D D1 9.58% EPE 1.40; mIoU 67.39%.
- Relevance: closest single-branch prototype — semantic-reserved slice + fuse pattern for attaching YOLO seg + disparity heads to one volume.

### SDBF-Net (APSIPA 2019) — bidirectional late residuals
`semantic_stereo/SDBF-Net_Rao_APSIPA2019.pdf` (satellite)
- Separate heavy encoders (ResNet-101 seg + siamese stereo with SPP) → 3D U-Net → soft-argmin → two parallel residual heads on concat [init-seg + image + init-disp].
- Loss: CE (seg) + L1 (disp), init + refined. 3-stage curriculum: seg 1000 → stereo 200 → fusion 200 epochs.
- Tricks: (1) each task's output corrects the other via tiny late heads; (2) dilated ASPP + SPP for photometric change.
- Numbers: US3D mIoU 0.767, D1 8.02%, EPE 1.31.
- Relevance: earliest late-fusion proof, but two heavy encoders — motivates collapsing to one YOLO encoder with cheap fusion heads.

### DispSegNet (RAL 2019) — semantic residual refinement
`semantic_stereo/DispSegNet_Zhang_RAL2019.pdf`
- ResNet-50 siamese; 5D concat volume + learned 3D metric (not dot); 8-layer 3D encoder-decoder + soft-argmin. Refinement: seg embedding + initial disparity → residual CNN → final. PSP head; right seg warped to left for supervision.
- Loss: `0.3·init + 0.7·refine + 1.0·seg`; init/refine mix photometric (SSIM+L1+grad) + double-warp consistency + image/segment-weighted smoothness.
- Training: TensorFlow Titan-X batch 1 crop 256×512. Adam 2e-4 Cityscapes 100K → 1e-4 KITTI 50K. No augmentation.
- Tricks: (1) smoothness conditioned on good initial + LR-inconsistency mask; (2) per-class analysis: smoothness hurts poles/signs, seg fixes them (road −49%, pole −32%).
- Numbers: KITTI15 test D1-all 6.33% (bg 4.20% beats supervised DispNetC).
- Relevance: blueprint for YOLO-seg late refinement + segment-aware smoothness (critical for poles/signs) + LR-mask occlusions.

## Cautionary tales

### TiCoSS (TASE 2025) — naive sharing causes task conflict
`semantic_stereo/TiCoSS_Tang_TASE2025.pdf`
- Duplex encoder (context + disparity branches, first 3 layers shared) + selective inheritance gates + hierarchical deep supervision + inconsistency-weighted losses.
- Loss: `1.5·inconsistency-CE + 1.0·cross-scale-KL + SCG + stereo`, batch 1, AdamW 2e-4, 100K/20K/50K iters.
- Finding: indiscriminate RGB+disparity summation dilutes geometry; deep supervision without cross-branch interaction ignores complementarity. Gating recovers ~7.9% mIoU vs naive fusion.
- Numbers: 385M params, 0.30 s/img. Seg KITTI mIoU 63.63 (+10.57); stereo barely moves (0.34 vs 0.38).
- Lesson: YOLO sharing needs gated selection + inconsistency-weighted loss, or disparity corrupts context.

### USAM-Net (arXiv 2025) — frozen-SAM early fusion disappoints
`semantic_stereo/USAM-Net_Sankaranarayanan_arXiv2025.pdf`
- Vanilla U-Net, no cost volume (direct regression); 9-ch input (L+RGB + R+RGB + SAM mask); bottleneck self-attention.
- Loss: smooth-L1 on valid LiDAR pixels. Sky-mask forcing hurt accuracy (negative result).
- Training: Adam 1e-3 ×0.9/epoch, 30 epochs, DrivingStereo half-res, 3×A100 56 GPU-h. SAM-ViT-B adds 93.7M + 900 ms vs 2.18 ms base.
- Finding: frozen-SAM early fusion gives marginal in-domain gain, hurts zero-shot transfer, huge cost.
- Numbers: DrivingStereo EPE 0.88 vs baseline 0.96; KITTI zero-shot worse than attention-only.
- Lesson: jointly-trained lightweight YOLO semantics over frozen foundation-model concatenation; attention alone is the cheaper alternative.
