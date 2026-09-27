# USAM-Net: Segmentation-Attention Stereo Correspondence Network

**Sankaranarayanan et al., arXiv 2025** · `semantic_stereo/USAM-Net_Sankaranarayanan_arXiv2025.pdf` · **DrivingStereo**.

## What it is

USAM-Net is semantic-guided stereo, not a joint semantic-output model. It supplies a pretrained segmentation map/features to a U-Net stereo matcher and uses attention to prioritise semantically informative regions.

$$A=\sigma(f_{seg}),\qquad F_{stereo}'=A\odot F_{stereo},\qquad L=L_{disp}+\lambda L_{smooth}.$$

$f_{seg}$ is the pretrained semantic representation, $A$ its learned attention map, $F_{stereo}$ stereo features, and $\odot$ elementwise modulation. The output is depth/disparity, not a predicted class map.

## Results

On DrivingStereo the paper reports **0.88 EPE** and **3.61% global difference**, comparing against CFNet, SegStereo, and iResNet. It does not report semantic mIoU, bad-0.5/1/2/3, D1, FPS, parameters, FLOPs, or memory. No public inference code/checkpoint was found.
