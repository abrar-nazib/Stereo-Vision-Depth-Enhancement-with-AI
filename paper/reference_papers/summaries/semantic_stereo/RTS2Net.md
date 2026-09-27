# RTS²Net: Real-Time Semantic Stereo Matching

**Authors:** Pier Luigi Dovesi et al.  
**Venue:** ICRA 2020  
**Paper:** `semantic_stereo/RTS2Net_Dovesi_ICRA2020.pdf`  
**Code / weights:** no maintained official PyTorch release or downloadable checkpoint was found during this review.  
**Domain:** Cityscapes pretraining and KITTI 2015 evaluation (19 classes).  
**Priority:** 8/10 as the best lightweight *architecture* reference; unavailable as a turnkey model.

---

## Core idea

RTS²Net is the clearest example of a lightweight real semantic–stereo fusion network: a compact shared encoder branches into coarse-to-fine disparity and semantic decoders, then a final **synergy refinement** uses both outputs to improve disparity. It can be stopped at an intermediate disparity stage to trade accuracy for speed.

![Figure 2, PDF p. 3: RTS²Net's shared blue encoder, yellow disparity stages, green semantic stages, and purple joint refinement.](../../../figures/RTS2Net_fig2_architecture.png)

### Architecture

- The shared encoder begins with two $3\times3$ convolutions, reduces resolution to one half, then has four max-pool + two-convolution blocks, yielding $2c,4c,8c,16c$ channels at $1/4,1/8,1/16,1/32$ scale.
- The **disparity decoder** estimates a coarse map at $1/16$, then residual disparity stages at $1/8$ and $1/4$; each has a small cost volume and 3-D convolutions. The final disparity is bilinearly upsampled to full resolution.
- The **semantic decoder** mirrors the three-stage coarse-to-fine progression using shared encoder features, producing class scores at each stage.
- The **synergy refinement** concatenates/upscales semantic and disparity evidence and emits a refined disparity map.
- $c$ is the width multiplier. $c=1$ recovers disparity-only AnyNet; larger $c$ makes semantic representation possible but lowers FPS.

## Loss equations and parameters

The paper supervises the three output scales with Smooth-$L_1$ stereo and refinement losses plus multiclass semantic cross-entropy:

$$L=\sum_{st=1}^{3}W_{st}\left(W_dL_{dst}^{(st)}+W_sL_{sst}^{(st)}+W_{dr}L_{drst}^{(st)}\right).$$

| Symbol | Meaning |
| --- | --- |
| $st$ | coarse-to-fine stage: $1/16$, $1/8$, or $1/4$ resolution. |
| $L_{dst}$ | Smooth-$L_1$ loss on disparity prediction. |
| $L_{sst}$ | multi-class semantic cross-entropy. |
| $L_{drst}$ | Smooth-$L_1$ loss on the synergy-refined disparity. |
| $W_{st}$ | stage weight; it emphasises later/finer predictions. |
| $W_d,W_s,W_{dr}$ | relative stereo, semantic, and refinement loss weights. |
| $c$ | channel-width hyperparameter controlling the accuracy/FPS trade-off. |

## Reported results

All values below are paper-reported on its **KITTI 2015 validation split**. Metrics available are EPE, D1-all, mIoU, pixel accuracy, and FPS. It does **not** report bad-0.5/1/2/3, parameters, FLOPs, or memory.

| RTS²Net width $c$ | EPE (px) | D1-all (%) | mIoU (%) | Pixel accuracy (%) | Jetson TX2 FPS | RTX 2080 Ti FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.12 | 5.57 | 58.86 | 80.86 | 8.3 | 60.5 |
| 4 | 0.90 | 3.80 | 60.93 | 89.77 | 7.4 | 60.5 |
| 8 | 0.84 | 3.33 | 62.22 | 90.64 | **6.3** | 60.4 |
| 16 | 0.78 | 2.90 | 67.41 | 92.92 | 4.5 | 60.4 |
| 32 | **0.74** | **2.62** | **69.62** | **93.57** | 2.3 | 42.2 |

For $c=8$, early stopping at stages 1/2/3 gives **17.2 / 10.9 / 6.3 FPS** and **8.00 / 4.70 / 3.33% D1-all** on Jetson TX2. Its KITTI 2015 online submission reports D1-bg/D1-fg/D1-all = **3.09 / 5.91 / 3.56%** at **0.02 s** on an RTX 2080 Ti. On the KITTI segmentation benchmark it reports class IoU/iIoU = **57.67/27.42%**, category IoU/iIoU = **82.85/60.72%**, at 0.02 s for joint output (0.008 s semantic-only).

## Relevance to this repository

RTS²Net is the correct conceptual answer to “one compact model for semantic classes plus stereo.” Its main practical limitation is reproducibility: no usable official checkpoint was found. Reimplementing it would be a research project; it should not replace the validated FastFS + YOLO live path without training and evaluation on matching road-scene data.
