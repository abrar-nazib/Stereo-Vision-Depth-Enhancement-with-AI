# Training recipes distilled from 26 papers

## Optimizer and schedule

- Default: **AdamW, LR 1–2e-4**, weight decay 1e-4–1e-5, betas (0.9, 0.999). Plain Adam appears in older papers (HITNet, CoEx, BGNet); SGD+momentum only in SegStereo (Caffe era).
- Schedule: **OneCycle** (LightStereo, LiteAnyStereo, AIO, DEFOM, D-FUSE) for short runs; step decay (HITNet 4e-4→1e-5; CoEx halving; SemStereo halving at fixed epochs) or cosine for long runs. Flat LR past ~20k only for BGNet-style KITTI finetune (constant LR, multi-seed, pick best).
- Gradient clipping is near-universal for iterative heads: clip norm 1.0 (our runners, modal) or clip values [−1,1] (GGEV, MonSter).
- Batch: 1–8 with AMP fp16 on small GPUs (our A04/A05: batch 1, 0.5–1.5 GB); 8–176 on multi-GPU/A100 (LiteAnyStereo batch 176; HITNet batch 8; GGEV batch 12).
- AMP fp16 throughout our runs; bf16 in the modal A100 script. No paper reports fp32-only training except the A02 control eval.

## Loss ladder (cumulative, in port order; weights = our Driving-200 `experiments/a05_clean/losses.py:modal_loss` tuning — re-tune per dataset; ablate each step ΔEPE/Δbad1)

1. `smooth-L1(full)` — every paper starts here. Plateaus ~6 px on Driving-200.
2. `+ 0.5·L1(1/2) + 0.3·L1(1/4)` — modal multi-scale. Biggest single jump; teaches coarse structure. (Our `losses.py:modal_loss`.)
3. `+ 0.5·grad_consistency` — disparity gradients match GT gradients; sharpens edges without a refinement net.
4. `+ 0.2·threshold_hinge + 0.2·d1_hinge` — quadratic penalties above 0.5/1/2/3 px and the D1 region; pulls bad-1/bad-3/D1 directly.
5. `+ 0.02·edge_smooth` — image-weighted smoothness; keep tiny or it melts poles/signs (DispSegNet per-class proof).
6. With semantics: boundary loss `|∇sem|·exp(−|∇d|)` (SSPCV weight 0.1; SGNet 0.5 with masks ignoring road/vegetation fake boundaries).
7. Iterative heads: L1 over iterates with γ=0.9 decay (RAFT-family standard: GGEV, DEFOM, D-FUSE, MonSter, AIO). Later iterates weigh more.
8. Deep supervision staging (SGNet: 0.5/0.7/1.0 over outputs; SemStereo: 1/0.6/0.5/0.3 over scales).

## Augmentation that preserves epipolar geometry

Safe (used across papers): asymmetric brightness/contrast, color jitter, right-image eraser patches, y-offset ±2 px, scale 0.9–1.15 with disparity rescaling, random crop, Gaussian noise/blur.
Forbidden: vertical flip, independent left/right warps, anything breaking horizontal epipolar lines.
Notable specifics: HITNet right-patch transplant (50–250 px) + Middlebury color-transfer; LiteAnyStereo strong perturbation on student only; StereoAnywhere volume Rolling/Noising/Zeroing + mirror truncation; CoEx must train (not just test) with top-k.

## Curriculum and pretraining

- Standard: SceneFlow (FlyingThings; HITNet notes Driving/Monkaa hurt) → KITTI/Middlebury/ETH3D finetune.
- Cityscapes-coarse→fine beats synthetic SceneFlow for KITTI transfer (RTS2Net).
- Multi-stage: SDBF seg→stereo→fusion; D-FUSE fusion→registration→global (frozen stages); LiteAnyStereo synthetic→self-distill→real-KD; Pip-Stereo supernet→search→prune.
- KITTI finetune tip (BGNet): constant LR, 300 epochs ×3 seeds, pick best.
- Joint training from scratch works with tiny real data given the right loss (S3M-Net, no SceneFlow pretrain).

## Debugging signatures (from our runs + papers)

- Train spiky / val smooth → a few unmatchable pairs dominate random crops. Score all pairs, rank by EPE, inspect brightness + valid fraction. (Our A04b: 2/200 black tunnel frames.)
- Crop EPE collapses / held-out stalls → overfit on small budget; iterative heads resist it better than wide aggregation heads (A04: RAFT 5.87 vs M 6.52).
- Edges melted → bilinear upsampling; fix with bilateral slicing, plane-tile upsampling, or grad-consistency loss.
- Textureless road errors → semantic injection (SegStereo warp loss) or mono prior guidance (GGEV SCF).
- Dark/low-texture failure → DA/DINO priors as guidance only (AIO roles: DINO foreground, SAM edges, DA dark).
- Mirrors piercing the volume → truncation at mono disparity (StereoAnywhere).
- Poles/signs worse with smoothness → segment-aware weighting (DispSegNet) or masked boundary loss (SGNet).
