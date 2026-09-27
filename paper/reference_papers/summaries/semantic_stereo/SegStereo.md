# SegStereo: Exploiting Semantic Information for Disparity Estimation

**Yang et al., ECCV 2018** · `semantic_stereo/SegStereo_Yang_ECCV2018.pdf` · [project/code](https://yangguorun.github.io/projects/segstereo/) · **Caffe, KITTI / Cityscapes**.

## Architecture and losses

SegStereo correlates left/right stereo features, aggregates the left semantic feature embedding into the disparity branch, and warps the right semantic logits to the left view for semantic-softmax regularisation. It predicts disparity; the semantic branch is trained jointly but is principally a stereo guide.

$$L_{unsup}=\lambda_pL_{photo}+\lambda_sL_{smooth}+\lambda_{sem}L_{softmax},\qquad L_{sup}=L_{reg}+\lambda_{sem}L_{softmax}.$$

$L_{photo}$ compares a warped image, $L_{smooth}$ preserves locally smooth disparity, $L_{softmax}$ enforces warped semantic consistency, and $L_{reg}$ is disparity supervision.

## Results and limits

The paper reports KITTI EPE, D1 (3 px / 5%), D1-bg/fg/all, and runtime, plus Cityscapes/FlyingThings evaluations; it does not report bad-0.5/1/2, mIoU, parameters, FLOPs, or memory. It is historically important and has weights, but they are Caffe checkpoints, so it is unsuitable for the project-wide PyTorch live CLI.
