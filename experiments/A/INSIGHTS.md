# A-series insights and archive index

The A experiment folders remain at their original `experiments/` paths.
They were **not moved** into this folder: historical scripts and run manifests
refer to those paths. This file is the A-series navigation point, not a
relocated run. Early LightStereo run checkpoints were subsequently pruned;
see [the pruning record](CHECKPOINT_PRUNING.md).

## Key results to carry forward

- [A09 local YOLO26m-encoder V arm](../a09_encoder_m/runs/A09_V_yolo26m_ade20k/results.json):
  held-out Driving-200 EPE **3.3265 px** after 35,000 local steps;
  **5,486,370** total and **522,722** trainable parameters. This is a
  Driving-subset result, not full SceneFlow.
- [A09 full SceneFlow plateau run](../final_pass/RUN_A09M_A10.md): the model
  trained on all **35,454** official SceneFlow-family train pairs with the
  frozen YOLO26m ADE20K encoder, native crops and no resize. Its locally
  downloaded 4,370-pair full-test artifact reports **1.0405 px EPE**,
  **6.6188% bad-3**, and **5.7252% D1**.
- [A09 OneCycle comparator](../final_pass/RUN_A09M_A10_ONECYCLE.md): same
  architecture/protocol and an independent fresh run, differing in learning
  rate schedule. Its full-test artifact reports **1.1623 px EPE**,
  **7.5057% bad-3**, and **6.5418% D1**. On these runs the validation-aware
  plateau schedule won; do not generalize that scheduler ranking beyond this
  experiment.

The full-test numeric source files are on the external SSD under
`/media/abrar/AbrarSSD/ResearchArtifacts/SVDE/final_pass/` in the
`a09m_fullsf_a10_v1_20260929/full_test.json` and
`a09m_fullsf_a10_onecycle_v1_20260929/full_test.json` run directories.

- [A08 architecture figure and renderer](../a06_shallow/): a visual guide
  to the earlier shallow-head design. Check the actual A06/A09 implementation
  before using a diagram as the model specification. The historical
  "semantic veto" in `a06_shallow/runners/model_a07v2.py` correlates
  shared stereo features, **not predicted semantic classes**. Its effect
  cannot establish semantic-to-disparity causality.

The local A02/A03/A04/A05/A06 ablations remain in
[`lightstereo_s_a02`](../lightstereo_s_a02/),
[`hitnet_a03`](../hitnet_a03/),
[`lightstereo_m_a04`](../lightstereo_m_a04/),
[`selective_raft_a04`](../selective_raft_a04/),
[`a05_clean`](../a05_clean/), and
[`a06_shallow`](../a06_shallow/), with their versioned manifests, logs and
reports; some early LightStereo checkpoints have been pruned. Compare metrics
only with their split, image count,
preprocessing, and checkpoint-selection rules aligned.

The full SceneFlow test contains the 400 FlyingThings3D frames used for
checkpoint selection, so its 4,370-pair score is not a wholly independent
held-out test. It is also not a real-domain generalization measurement.
