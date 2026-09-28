# A04 — Shared-encoder prototypes (frozen ADE20K + lightweight stereo heads)

## Scope

Two heads, one shared encoder. Both are inference-visible when you return.

- **A04a · LightStereo-M** — EfficientNetV2-backed 2D aggregation head (OpenStereo, ~7.6M params). Clean adapter pattern: YOLO semantic pyramid → head with no chokepoint.
- **A04b · Selective-RAFT** — Iterative GRU head with CSA (CVPR'24, 0.47 EPE). Cleanest seam (rank 9/10) but deeper plumbing.

Protocol is identical for both so the heads are comparable:

- 200 fixed SceneFlow Driving pairs from `/media/abrar/AbrarSSD/Datasets/sceneflow_driving`
- Stratified 160 train / 40 held-out (seed 42), same file identity
- Native 384×640 crops, same window on L/R/disparity; validation at full 540×960, no resizing
- 15,000 steps, eval every 100, checkpoint every 500, logging to `run.log`

Lane chosen for both prototypes: **frozen ADE20K YOLO26s** (`yolo26s-sem-ade20k.pt:layers[0:7]`) only. Ghost control removed per your last instruction.
