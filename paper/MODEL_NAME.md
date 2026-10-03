# Model name and candidate titles

Selected model name: **SemTileStereo**.

Selected working paper title:

1. **SemTileStereo: Shared-Encoder Tile Stereo with Semantic-Guided Disparity Refinement and Segmentation**

Other titles retained for reconsideration:

2. **SemTileStereo: Joint Semantic Segmentation and Tile-Based Stereo Matching with a Shared Encoder**
3. **SemTileStereo: Semantic Cost Gating and Tile Propagation for Joint Disparity and Segmentation**

The title is provisional. The architecture figure and manuscript should distinguish
the frozen A09 tile-stereo predictor and frozen YOLO26m semantic predictor from the
trainable E3 semantic cost gate and class-conditioned disparity residual. "Joint"
describes the two inference outputs, not end-to-end joint fine-tuning of both heads.
