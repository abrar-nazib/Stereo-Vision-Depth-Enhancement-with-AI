# S³M-Net: Joint Learning of Semantic Segmentation and Stereo Matching for Autonomous Driving

**Authors:** Zhiyuan Wu et al.  
**Venue:** IEEE Transactions on Intelligent Vehicles, 2024  
**Paper:** `semantic_stereo/S3M-Net_Wu_TIV2024.pdf`  
**Code:** https://github.com/Crist-123/S3M-Net  
**Deployment status:** PyTorch code is public; the repository's demo requires a user-supplied checkpoint and does not link a released pretrained checkpoint.  
**Domain:** road scenes; vKITTI2 (15 classes) and KITTI 2015 (19 classes).  
**Priority:** 9/10 for the intended camera domain; not lightweight (reported 0.66 FPS).

---

## What it actually fuses

S³M-Net is a true **joint semantic–stereo** network. A shared encoder processes the rectified left/right images; its left features feed both a RAFT-Stereo-style correlation/GRU disparity path and a semantic decoder. The Feature Fusion Adaptation (FFA) module transforms the relatively narrow stereo representation into a semantic representation, injects encoded disparity features, and uses a dense skip decoder to emit class logits. The semantic-consistency-guided (SCG) loss then weights both heads near class boundaries.

![Figure 1, PDF p. 3: S³M-Net's shared encoder, correlation pyramid/GRU stereo head, FFA semantic head, and two SCG-supervised outputs.](../../../figures/S3M-Net_fig1_architecture.png)

### Architecture in reading order

1. **Joint encoder:** rectified $I^L,I^R\in\mathbb{R}^{H\times W\times3}$ produce multi-scale shared features $\mathcal F^L=\{F_1^L,\ldots,F_n^L\}$ and $\mathcal F^R$.
2. **Stereo branch:** an all-pairs 3-D correlation pyramid is sampled by a multi-level GRU, iteratively updating $D_0,\ldots,D_n$.
3. **FFA semantic branch:** shared left features are remapped to semantic channel widths and fused with an encoding of the final disparity estimate.
4. **Dense semantic decoder:** dense skips and upsampling predict the $C$-class semantic map.
5. **SCG supervision:** the ground-truth semantic map produces a spatial weight map. Both segmentation CE and every iterative disparity loss receive that weight.

## Key equations and parameters

### Stereo correlation pyramid

$$C_1(i,j,k)=F_n^L(i,j,:)\cdot F_n^R(i,k,:),$$

$$\mathcal C=\{C_1,\ldots,C_m\},\qquad C_m\in\mathbb{R}^{H\times W\times W/2^{m-1}}.$$

| Symbol | Meaning |
| --- | --- |
| $i,j,k$ | row, left-image column, and candidate right-image column. |
| $F_n^L,F_n^R$ | final-scale shared left/right feature vectors. |
| $C_1$ | initial all-pairs correlation volume (dot-product similarity). |
| $m$ | pyramid level; each later level uses 1-D average pooling with kernel/stride 2 along correspondence candidates. |
| $D_i$ | disparity estimate at GRU iteration $i$; $D_0$ is initialized to zero. |

### Feature Fusion Adaptation (FFA)

$$F_i^F=A_i(F^L)\oplus E^D(D_n),$$

$$A_i(F^L)=\begin{cases}
R(F_{2i-1}^L),&i\leq(n+1)/2\\
E\bigl(F_{i-1}^F\oplus E^D(D_n)\bigr),&i>(n+1)/2.
\end{cases}$$

| Symbol | Meaning |
| --- | --- |
| $F_i^F$ | $i$-th fused feature passed to the semantic decoder. |
| $R$ | $3\times3$, stride-2 Conv–BN–ReLU remapping from shared/stereo channels to semantic channels (64, 256, 512 in the paper). |
| $E^D$ | ResNet-152 encoding of the final disparity map. |
| $E$ | residual encoding of a previously fused feature. |
| $\oplus$ | feature fusion operation. |

### Semantic-consistency-guided loss

The one-hot semantic volume is $V_c^{3D}(p)=\delta(M^G(p),c)$. After channel-wise average pooling $P(\cdot)$ and normalization,

$$V^N(p)=e^{-(2V^I(p)-1)^2},\qquad W(p)=\max_cV_c^N(p).$$

$$L_{ss}=-\frac1N\sum_{p=1}^{N}\sum_{c=1}^{C}[(1-\alpha)+\alpha W(p)]y_c(p)\log\hat y_c(p),$$

$$L_{sm}=\sum_{i=1}^{n}[(1-\alpha)+\alpha W(p)]\gamma^{N-i}\lVert D^G-D_i\rVert_1,\qquad L_{SCG}=L_{ss}+L_{sm}.$$

| Symbol / setting | Meaning |
| --- | --- |
| $M^G(p),y_c(p)$ | ground-truth class and one-hot indicator at pixel $p$. |
| $\hat y_c(p)$ | predicted probability of class $c$. |
| $C,N$ | semantic class count and number of pixels. |
| $W(p)$ | high near/within semantic structures after the paper's class-volume transform; weights the two tasks consistently. |
| $D^G,D_i$ | ground-truth disparity and $i$-th GRU prediction. |
| $\alpha=0.1$ | SCG loss-weight setting selected by ablation. |
| $\gamma=0.9$ | decay applied to earlier iterative disparity predictions. |

## Reported results

The paper reports **EPE** and **PEP** (percentage of error pixels above 1 px / 3 px) for stereo; and Acc, mAcc, mIoU, fwIoU, precision, recall, and F-score for semantics. It does **not** report bad-0.5/2, D1, parameter count, FLOPs, or memory.

| Dataset / model | EPE (px) | PEP-1 (%) | PEP-3 (%) | Acc / mAcc / mIoU / fwIoU (%) | Precision / recall / F-score (%) |
| --- | ---: | ---: | ---: | --- | --- |
| vKITTI2, S³M-Net + SCG | **0.38** | **5.56** | **2.55** | 98.32 / 88.24 / 84.18 / 96.98 | 98.37 / 98.28 / 98.31 |
| KITTI 2015, S³M-Net + SCG | **0.55** | **10.02** | **1.62** | 90.66 / 65.90 / 57.80 / 84.53 | 90.85 / 93.55 / 91.80 |

The reported end-to-end speed is **0.66 FPS** (RTX 3090). This is research evidence of useful joint supervision, not a viable direct live-camera replacement for FastFS.

## Relevance to this repository

This is the strongest road-domain architecture reference for a future single-model semantic+stereo branch. It is unsuitable for immediate live deployment without a released compatible checkpoint, substantial optimisation, and calibration/rectification validation. A practical current system remains FastFS disparity plus a separate Cityscapes semantic model.
