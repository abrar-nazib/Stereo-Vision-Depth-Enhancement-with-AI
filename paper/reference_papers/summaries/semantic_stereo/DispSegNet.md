# DispSegNet: Leveraging Semantics for End-to-End Disparity Estimation

**Zhang et al., RA-L 2019** · `semantic_stereo/DispSegNet_Zhang_RAL2019.pdf` · **KITTI / Cityscapes road scenes**.

## Architecture

A Siamese stereo encoder builds a 3-D cost volume and regresses initial disparity. A semantic encoder–decoder emits class logits; its learned segment embedding is concatenated with the initial disparity to predict a residual disparity. The model therefore emits both labels and disparity, though its central goal is disparity quality.

$$d_{final}=d_{init}+R([d_{init},E_{seg}]),$$

where $d_{init}$ is the cost-volume disparity, $E_{seg}$ is the semantic embedding, and $R$ is the refinement network. Training combines photometric reconstruction, left–right consistency, edge-aware smoothness, semantic cross-entropy, and supervised disparity regression when ground truth exists.

## Results and limits

The paper reports KITTI 2015 D1-bg/D1-fg/D1-all, EPE, semantic-class regional error, and runtime; it does not report bad-0.5/1/2, parameters, FLOPs, or memory. Its ablation reports semantic softmax supervision reducing KITTI all-pixel EPE from **2.17 to 1.89 px** and D1 from **10.53 to 10.03%**. No maintained official code/checkpoint was located, so it is a design reference rather than a deployment candidate.
