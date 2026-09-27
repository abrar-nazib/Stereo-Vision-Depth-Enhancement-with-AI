# SDBF-Net: Semantic and Disparity Bidirectional Fusion Network

**Rao et al., APSIPA 2019** · `semantic_stereo/SDBF-Net_Rao_APSIPA2019.pdf` · **US3D incidental satellite imagery**.

## Architecture and equation

SDBF-Net has a semantic segmentation module (SSM), 3-D-convolution stereo matching module (SMM), and bidirectional fusion module (FM). SSM gives initial class scores; SMM gives initial disparity; FM concatenates each with the other task's prediction and learns residual refinements for **both** maps.

$$\hat d_{init}=\sum_{d=0}^{D}d\,\operatorname{softmax}(-V_d),\qquad (S_{final},d_{final})=FM(S_{init},d_{init},I^L,I^R).$$

$V_d$ is regularised matching cost for disparity $d$; $S$ denotes semantic score maps; $D$ is the disparity range.

## Results and limits

It reports US3D semantic mIoU and stereo EPE/D1-style errors, but no bad thresholds, runtime, parameter count, FLOPs, or memory. It is a genuine bidirectional two-output fusion design, but is satellite-only and no official runnable code/checkpoint was found.
