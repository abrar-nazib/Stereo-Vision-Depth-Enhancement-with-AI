# SSPCV-Net: Semantic Stereo Matching With Pyramid Cost Volumes

**Wu et al., ICCV 2019** · `semantic_stereo/SSPCV-Net_Wu_ICCV2019.pdf` · **SceneFlow, KITTI, Cityscapes**.

## Architecture

SSPCV-Net extracts semantic features with a segmentation subnetwork and spatial features with hierarchical pooling. It builds multi-level semantic/spatial pyramid cost volumes, concatenates them, and applies 3-D multi-cost aggregation followed by soft-argmin disparity regression. The semantic path guides disparity; the paper's endpoint is disparity rather than a deployment-ready two-output interface.

$$\hat d=\sum_{d=0}^{D-1}d\,\operatorname{softmax}(-C_d),\qquad L=L_{smoothL1}+\lambda_bL_{boundary}.$$

$C_d$ is aggregated cost for disparity hypothesis $d$, $D$ the maximum disparity, and $L_{boundary}$ penalises disagreement near semantic boundaries.

## Results

Reported metrics include SceneFlow EPE and KITTI D1-est/bg/fg/all for all and non-occluded pixels; no bad-0.5/1/2, mIoU, FPS, parameter, FLOP, or memory result is reported. Its headline KITTI-2015 all-pixel result is **0.87 px EPE** and **3.1% D1-all**. No maintained official PyTorch weights were found.
