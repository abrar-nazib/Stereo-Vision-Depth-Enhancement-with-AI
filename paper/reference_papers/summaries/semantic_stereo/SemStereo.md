# SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing

**Authors:** Chen Chen et al.  
**Venue:** AAAI 2025  
**Paper:** `semantic_stereo/SemStereo_Chen_AAAI2025.pdf`  
**Code:** https://github.com/chenchen235/SemStereo  
**Weights:** official PyTorch training code is available, but the README does not publish a pretrained inference checkpoint.  
**Domain:** US3D and WHU aerial/remote-sensing stereo.  
**Priority:** 6/10 as a rigorous semantic-guided stereo reference; not appropriate for direct street-camera inference.

---

## Core idea

SemStereo explicitly distinguishes three forms of task coupling: deep shared features in a **Semantic-Guided Cascade (SGC)**, class-selective disparity residuals in **Semantic Selective Refinement (SSR)**, and **Left–Right Semantic Consistency (LRSC)** supervision. It outputs both segmentation maps and disparity, but its intra-class disparity assumption is motivated by aerial imagery, where a class tends to have a single disparity mode—unlike perspective-heavy street scenes.

![Figure 3, PDF p. 3: SemStereo's Semantic-Guided Cascade, Semantic Selective Refinement branch, and Left–Right Semantic Consistency supervision.](../../../figures/SemStereo_fig3_architecture.png)

### Architecture

1. A MobileViTv2 U-shaped shared encoder takes $I^L,I^R\in\mathbb R^{3\times H\times W}$ and emits multi-scale features at $1/2,1/4,1/8,1/16,1/32$.
2. A semantic head attached to the deepest feature produces $P^L,P^R\in\mathbb R^{N\times H\times W}$.
3. SGC injects deep semantic-enriched features into Fast-ACV cost-volume computation, yielding an initial disparity $d_{init}$.
4. SSR uses per-class confidence to gate the feature volume and predict a residual $R$ that refines $d_{init}$.
5. LRSC warps the left prediction/label with final disparity to synthesize right-view semantic supervision.

## Equations and parameter descriptions

### Semantic-selective refinement

$$F'=\sigma(F\cdot P^L)\cdot F,$$

$$R=\operatorname{Conv}\bigl(\sigma(F')\cdot d''_{init}\bigr),\qquad d_{final}=d''_{init}+R.$$

| Symbol | Meaning |
| --- | --- |
| $F$ | feature volume. |
| $P^L\in\mathbb R^{N\times H\times W}$ | left semantic class-probability maps; $N$ is the class count. |
| $\sigma$ | $1\times1$ Conv + batch norm + sigmoid gate (first equation); later a learned feature gate. |
| $F'$ | class-selectively filtered feature volume. |
| $d''_{init}$ | bilinearly upsampled and class-channel-expanded initial disparity. |
| $R$ | one-channel learned disparity residual. |

### Left-right semantic consistency and joint objective

$$R_{gt}=\begin{cases}\operatorname{warp}(GT^L,d_{final}),&GT^L\text{ available}\\
\operatorname{warp}(P^L,d_{final}),&GT^L\text{ unavailable},\end{cases}\qquad L_{LRSC}=L_{CE}(P^R,R_{gt}),$$

$$L_{Seg}=L_{CE}(P,L_{gt})+L_{Dice}(P,L_{gt}),$$

$$L_{Disp}=\sum_i\lambda_i\operatorname{SmoothL1}(d_i,d_{gt}),\qquad L=L_{Disp}+\alpha L_{Seg}+\beta L_{LRSC}.$$

| Setting | Paper value / role |
| --- | --- |
| $d_i,d_{gt}$ | staged predicted and ground-truth disparity. |
| $\lambda_0,\lambda_1,\lambda_2,\lambda_3$ | multi-stage disparity weights: 1, 0.6, 0.5, 0.3. |
| $\alpha,\beta$ | semantic and LRSC loss weights, both 1. |
| US3D disparity range | $[-64,64)$ pixels. |
| WHU disparity range | $[0,128)$ pixels. |

## Reported results

Available stereo metrics are EPE and D1; semantic metrics are pixel accuracy (PA), mIoU, and per-class IoU. The paper does **not** report bad-0.5/1/2/3, FPS/latency, parameters, FLOPs, or memory.

| US3D Jacksonville test model | EPE (px) | D1 (%) | mIoU (%) | PA (%) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.2087 | 7.28 | 75.84 | 93.65 |
| SGC-Net | 0.9995 | 4.98 | 75.74 | 93.70 |
| SGC-SSR-Net | 0.9702 | 4.76 | 76.85 | 93.83 |
| **SemStereo** | **0.9582** | **4.58** | **77.02** | **94.13** |

| Dataset / setting | EPE (px) | D1 (%) |
| --- | ---: | ---: |
| US3D, SemStereo without explicit semantic labels | 0.9956 | 5.00 |
| WHU, SemStereo without explicit semantic labels | **0.2236** | **0.731** |
| US3D cross-city zero-shot (Jacksonville → Omaha) | 1.4996 | 9.70 |
| Omaha, 50-pair fine-tuning | 1.3206 | 6.79 |
| Omaha, 500-pair fine-tuning | 1.1002 | 4.54 |

On US3D, the full model's five class-IoU entries in Table 4 are **90.84, 74.63, 88.30, 68.94, and 62.37%** (with PA **94.13%** and mIoU **77.02%**); use the original table for the paper's rendered class-column labels and full comparison.

## Relevance to this repository

SemStereo provides the most explicit equations for using semantic confidence to refine stereo disparity. The assumptions, labels, and aerial geometry differ materially from the connected stereo camera, so it should guide research design rather than be run as a pretrained live model.
