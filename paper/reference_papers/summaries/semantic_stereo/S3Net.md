# S3Net: Single-Branch Semantic Stereo Network in Satellite Epipolar Imagery

**Authors:** Qingyuan Yang et al.  
**Venue:** IGARSS 2024  
**Paper:** `semantic_stereo/S3Net_Yang_IGARSS2024.pdf`  
**Code / pretrained weights:** https://github.com/CVEO/S3Net (the public US3D checkpoint is already stored under `models/stereo/semantic_stereo/`).  
**Domain:** US3D satellite epipolar imagery, five semantic categories.  
**Priority:** 5/10 for the live camera, 9/10 as a reproducible example of a genuine single-branch fusion model.

---

## Core idea

S3Net shares one 4-D stereo cost volume between pixel classification and disparity regression. It is a genuine joint model: a weight-shared **Disparity–Classification Spatial Feature Extraction Module (DCSFEM)** produces semantic and multi-scale disparity features; the **Self-Fuse Module (SFM)** filters feature streams; a three-round 3-D **Mutual-Fuse Module (MFM)** exchanges semantic and disparity evidence; separate bilinear/trilinear upsampling produces the class and disparity maps.

![Figure 1, PDF p. 2: S3Net's weight-shared DCSFEM, single cost volume, self/mutual fusion modules, and two output heads.](../../../figures/S3Net_fig1_architecture.png)

## Architecture and notation

| Item | Description |
| --- | --- |
| DCSFEM | shared-weight left/right extractor. Semantic features and four-times-downsampled multi-scale disparity features are concatenated. |
| $H\times W\times D\times C$ | 4-D cost-volume shape: spatial height, width, disparity hypotheses, and feature channels. The top disparity layer is reserved for semantic information; remaining layers encode disparity features. |
| SFM | self-fusion block. Two learned branches are multiplied channel-wise to gate/filter the input; both 2-D feature and 3-D cost-volume variants are used. |
| MFM | three rounds of 3-D SFM, disparity-dimension isolation, downsampling, skip-connected cost volumes (`cost1`, `cost2`, `cost3`), and upsampling. |
| Bilinear / trilinear heads | bilinear upsampling supplies the classification map; trilinear upsampling supplies the disparity map. |

The paper does not specify a standalone symbolic training loss equation. Its explicit mathematical model description is the cost-volume representation above; implementation is PyTorch 1.8.1, batch size 4, with 512×512 US3D crops and Tesla V100 16 GB training/evaluation.

## Reported results

**US3D test set:** 4,292 satellite stereo pairs in total; 3,500 train / 338 validation / 454 test after the paper's split. Stereo metrics are EPE and D1-Error; semantic metrics are class IoU, mIoU, and joint mIoU-3. The paper does **not** report bad thresholds, runtime/FPS, parameters, FLOPs, or memory.

| Model | D1-Error (%) | EPE (px) |
| --- | ---: | ---: |
| PSMNet | 11.872 | 1.695 |
| GwcNet | 11.387 | 1.618 |
| GANet | 10.876 | 1.526 |
| CFNet | 11.024 | 1.570 |
| S2Net | 10.051 | 1.439 |
| **S3Net** | **9.579** | **1.403** |

| Semantic method | Ground IoU | Tree IoU | Building IoU | Water IoU | Bridge IoU | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S3Net** | 81.94 | 66.39 | 73.45 | 79.23 | 35.96 | **67.39** |

Its complete joint ablation (SFM + both DCSFEM modules + both MFM volumes) reports **mIoU 67.39%, mIoU-3 66.27%, D1-Error 9.579%, and EPE 1.403 px**.

## Practical limitation

The released checkpoint is real and useful for verifying a joint-output pipeline, but it is trained for bird's-eye satellite image geometry and its five US3D classes. It should not be applied to an uncalibrated terrestrial UVC stereo camera as if its class labels or disparity statistics transferred. It is a research baseline, not the recommended live model.
