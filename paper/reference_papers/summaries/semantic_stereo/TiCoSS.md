# TiCoSS: Tightening the Coupling Between Semantic Segmentation and Stereo Matching

**Tang et al., IEEE TASE 2025** · `semantic_stereo/TiCoSS_Tang_TASE2025.pdf` · [official code and inference checkpoint](https://github.com/Crist-123/TiCoSS) · **road domain: KITTI 2015 / vKITTI2 / Cityscapes**.

## Architecture

TiCoSS is the current practical joint-model candidate: a shared encoder feeds an iterative stereo branch and a semantic decoder. Its tightly gated feature fusion (TGF) lets contextual/semantic and geometric features gate each other across layers; hierarchical deep supervision (HDS) supervises both heads at multiple scales. It writes both a segmentation image and a disparity image in the official demo.

## Objective

The coupling-tightening objective combines disparity and semantic losses with two cross-task terms:

$$L_{CT}=L_{disp}+\alpha L_{seg}+\beta L_{DIA}+\gamma L_{DSCC}.$$

| Term | Role |
|---|---|
| $L_{disp}$ | multi-stage disparity regression loss. |
| $L_{seg}$ | semantic pixel-classification loss. |
| $L_{DIA}$ | disparity-informed semantic alignment: geometry improves semantic boundaries. |
| $L_{DSCC}$ | disparity–semantic cross-consistency constraint. |
| $\alpha,\beta,\gamma$ | paper-selected relative weights. |

## Evidence and use

The paper reports EPE and PEP at 1/3 pixels for stereo, plus Acc, mAcc, mIoU, fwIoU, precision, recall, and F-score for semantics on vKITTI2 and KITTI 2015. It reports no bad-0.5/2, D1, FPS, parameter count, FLOPs, or memory. Unlike S3M-Net, this repository publishes an inference checkpoint, making it the next joint model worth adapting to the live tool. Validate its KITTI camera geometry and 19-class taxonomy before trusting it on the UVC camera.
