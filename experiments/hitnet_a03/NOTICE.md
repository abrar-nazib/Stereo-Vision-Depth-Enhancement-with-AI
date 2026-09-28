# A03 provenance

## `hitnet_head/` — ported from the author's own repository

These modules are the author's own StereoLite `v2_hitnet` design, copied from
`stero_research_claude/model/designs/`:

| File | Origin |
| --- | --- |
| `model.py` | `designs/StereoLite_v2_hitnet/model.py` |
| `tile_propagate.py` | `designs/StereoLite_v2_hitnet/tile_propagate.py` |
| `hitnet_propagate.py` | `designs/StereoLite_v2_hitnet/hitnet_propagate.py` |
| `cost_volume.py` | `designs/StereoLite_v2_hitnet/cost_volume.py` |
| `_blocks.py` | `designs/_blocks.py` |

This is the author's own work, so reuse carries no third-party licensing issue.
The HITNet propagation block follows Tankovich et al., CVPR 2021, Sec. 3.4:
residual blocks with dilated convolutions and no batch normalization, a 1x1
channel reduction, and a local cost volume built by warping at d-1, d, d+1.

### Modifications made for A03

1. `from _blocks import ...` became `from ._blocks import ...` and the
   `sys.path.insert` bootstrap was removed, because `_blocks.py` now lives in
   this package.
2. `StereoLite.__init__` previously asserted `backbone == "ghost"` ("fixed to
   ghost encoder for fair architecture A/B"). That assertion is replaced with a
   branch that can build a `FrozenYoloEncoder`, so the same head can host the
   YOLO26 backbone. The head sizes its own convolutions from
   `fnet.out_channels`, so no adapter layers are needed.
3. `StereoLiteConfig` gained `freeze_encoder: bool = True`.

`encoders.py` (`FrozenYoloEncoder`) is new for A03, not ported.

## Not used: `model.py` (HITNet-XL from the paper)

`experiments/hitnet_a03/model.py` is an independent HITNet implementation
written from the paper. It loads the Google-derived XL tensor set 164/164
strict, but its dataflow was not resolved — propagation and refinement produce
oversized deltas, so it is **parked and unused**. It is retained only as a
record of the strict-load result and the architectural analysis.
