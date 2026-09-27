# StereoLite SceneFlow baseline

`stereolite_yolo26s_gev4_sceneflow_best.pth` is the exact base checkpoint
used as the fallback by the `supervisor-demo` branch's
`model/scripts/demo_supervisor.py`. It is **not** an InStereo2K or real-camera
fine-tune.

- Architecture configuration: `gev4_opt_narrow_plane` (YOLO26s feature
  encoder), 2.9623 M parameters.
- Source run: `20260704_fullsf_gev4onp_nc`; checkpoint step 53,000.
- SceneFlow held-out validation (recorded in checkpoint): EPE 0.7896 px,
  bad-1 9.115%, bad-3 4.081%, D1 3.466%.
- SHA-256:
  `0f8db3551af32cdf3a3a9f4a4e6403022f7be4dd6c82ced05fb69bb7a94d7df2`

It must be instantiated through the source repository's
`overfit_efficiency_ablation.build_model("gev4_opt_narrow_plane")`; it is
not compatible with the LAS2 or LightStereo loaders.
