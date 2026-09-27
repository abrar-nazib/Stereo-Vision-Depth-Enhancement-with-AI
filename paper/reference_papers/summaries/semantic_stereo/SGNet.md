# SGNet: Semantics Guided Deep Stereo Matching

**Chen et al., ACCV 2020** · `semantic_stereo/SGNet_Chen_ACCV2020.pdf` · **PSMNet-based, KITTI / vKITTI**.

## Architecture

SGNet augments PSMNet with a semantic branch, then applies (1) a semantic/disparity joint-confidence module to cost-volume regression, (2) category-dependent residual disparity refinement, and (3) semantic-boundary/region-aware loss terms. Its primary published output is disparity; semantic logits are intermediate supervision.

$$d_{final}=d_{init}+R_{cat}(F_{sem},d_{init}),$$

where $R_{cat}$ is a category-dependent residual; confidence from left/right semantic and disparity consistency reweights the cost volume before regression.

## Results

The paper reports KITTI 2012/2015 D1-bg, D1-fg, D1-all for all/non-occluded pixels and semantic validation mIoU **48.12%**, mAcc **55.25%**. It does not report EPE, bad-0.5/1/2, latency, parameters, FLOPs, or memory. No official weights were found.
