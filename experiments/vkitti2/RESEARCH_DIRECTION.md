# Research direction: semantic guidance for stereo disparity

Status: working hypothesis and proposed experiments, **not** a demonstrated
result or manuscript claim. Written 2026-09-30. Revisit this note when planning
training or preparing the manuscript.

## Question and motivation

Can a useful, lightweight, shared-encoder model produce dense semantic labels
and stereo disparity together, and can *better* semantic predictions improve
disparity accuracy or cross-domain robustness without changing the pretrained
stereo predictor? High-quality metric depth ground truth is costly to collect;
LiDAR is an important source, but not the only source. Semantic labels can be
obtained from annotation, synthetic scenes, or suitable pseudo-labels, although
pixel-level annotation and domain shift remain substantial costs. These are
motivations, not claims established by our experiments.

The intended contribution is **semantic-to-disparity guidance**, with reliable
semantics also available to downstream robotics or point-cloud processing. We
do not need to claim that a small joint head improves both tasks. An in-domain
VKITTI gain alone would not establish real-world generalization.

## What the current evidence actually says

On the existing 1,000-pair VKITTI2 feasibility subset, 800 pairs train on
Scene01/02/06/18 and 200 held-out pairs come from Scene20. The frozen
ADE20K-semantic/stereo baseline has EPE 2.4166 px and **mapped nine-class**
mIoU 0.4303; the v1 joint residual at its reported checkpoint has EPE 2.3519
px and mapped mIoU 0.5558. These are not full VKITTI-class mIoU values. Several
classes remain weak, including traffic light at zero IoU in that split; class
support and label mapping must be reported. Source:
[`runs/ablation_v2_1000/summary.json`](runs/ablation_v2_1000/summary.json)
and [`data.py`](data.py).

The present stereo checkpoint consumes only YOLO26m semantic model layers
0–6, at strides 1/2, 1/4, 1/8, and 1/16, and holds them frozen during stereo
training. The rest of the semantic model is not the stereo encoder. Source:
[`../hitnet_a03/hitnet_head/encoders.py`](../hitnet_a03/hitnet_head/encoders.py).

## Which segmentation task fits

- **Semantic segmentation:** one class ID at every pixel. Two cars share the
  same `car` ID; road, sky, building, and other background regions are labeled.
  This is the primary task for dense `[u, v, disparity, class]` output and for
  region/boundary guidance of stereo.
- **Instance segmentation:** a separate mask for each detectable object. Two
  cars receive different instances, usually with boxes and object confidence.
  It is valuable for tracking and object-level point clouds, but does not by
  itself provide a complete labeled road/sky/building map. It may be added
  later if object identities are a separate robotics requirement.

Use the local `yolo26m-sem-ade20k.pt` (Ultralytics `task=semantic`) as the
starting teacher/checkpoint, **not** a `-seg` instance model. Its ADE20K labels
are not VKITTI labels. Plan a verified VKITTI color-to-contiguous-class-ID
conversion using lossless PNG masks; treat `Undefined` as ignore ID 255 unless
the dataset audit gives a reason otherwise. Audit every class, pixel frequency,
split, and mask/image alignment before training. Do not reuse the nine-class
ADE20K mapping as the target taxonomy without an explicit research reason.

## Freezing trade-off: a measured question

Ultralytics defaults to `freeze=None`, so ordinary checkpoint fine-tuning
updates the backbone. `freeze=N` holds layers 0 through N-1; the exact frozen
module names and BatchNorm state should be verified in the installed version.
For this architecture, **freezing layers 0–6 (`freeze=7`)** preserves the exact
shared features used by the trained stereo head while allowing later YOLO
backbone, neck, and semantic head layers to adapt. It is *not* the same as
freezing the complete YOLO backbone. A new class taxonomy may reinitialize
some final classifier weights even though transferred layers remain intact.

Freezing 0–6 may help preserve stereo behavior, reduce training cost, and
limit overfitting, but it may cap semantic accuracy, especially for small or
thin classes and synthetic-to-real appearance shifts. It is therefore a
starting constraint, not a belief that freezing is always good. Compare:

1. Semantic head/neck and later layers trained with layers 0–6 frozen (shared
   checkpoint remains compatible).
2. A controlled partial-unfreeze or full-fine-tune semantic model, *as a
   separate version*, to measure the semantic quality ceiling. If any of
   layers 0–6 change, the existing stereo head no longer shares the exact
   trained encoder; its EPE must be re-evaluated and may require adaptation.

Before and after each run, verify the frozen parameter names and exact
tensor/buffer equality or hashes for layers 0–6; log trainable parameters,
preprocessing, resolution, checkpoint provenance, mIoU, per-class IoU, pixel
accuracy, boundary quality, latency, and VRAM. Do not infer freezing solely
from a CLI flag.

## Proposed causal experiment

1. Establish a full-14-class VKITTI semantic teacher with a fixed-seed
   80/10/10 frame-grouped random split. This replaces the earlier proposed
   scene-held-out semantic split: GuardRail exists only in Scene20, and the
   immediate purpose is to obtain useful semantics for the fusion experiment,
   not to benchmark scene transfer of YOLO itself. Keep all variations of a
   `(scene, frame)` group in one split and publish the split manifest. Choose
   checkpoints by validation mIoU and weak-class performance, with per-class
   support; do not call its random-split test an out-of-domain result.
2. Hold the stereo checkpoint, splits, and training budget fixed. Compare
   depth without semantic guidance; guidance from the existing ADE20K model;
   guidance from VKITTI-trained semantics; and ground-truth semantic guidance
   as an **oracle**. The oracle is for analysis, never deployment. If oracle
   guidance does not help, the fusion interface is the likely bottleneck.
3. If semantics do help, train only a small **depth-side** guidance/adapter
   module and measure EPE, RMSE, bad-0.5/1/2/3, D1, class-/boundary-specific
   depth error, runtime, parameters, and peak memory against the frozen stereo
   baseline. Keep the semantic output unmodified in this experiment.
4. Test cross-domain generalization on a separate real-domain benchmark or
   calibrated rig data, with no depth labels from that test domain used for
   training or selection. Report synthetic and real-domain results separately.
   If no trustworthy real depth ground truth exists, show qualitative outputs
   but do not claim a measured real-domain EPE gain.

For segmentation-only training, resizing images and masks together does not
require disparity-label rescaling, but the original no-resize stereo protocol
must remain intact. The shared trunk must see compatible input normalization
and geometry at fusion time; document any aspect-ratio scaling, crop, padding,
and inference unpadding. Avoid interpreting high VKITTI mIoU as proof of real
camera semantic quality.

## Manuscript guardrails

- Separate *hypothesis*, *implementation*, and *measured result*.
- Call the present semantic metric **mapped nine-class mIoU**; never compare it
  directly to full-taxonomy mIoU from another paper.
- Do not claim that semantic guidance improves transfer until the fixed-depth
  cross-domain control and oracle ablation support it.
- State whether sharing means the exact same frozen layers 0–6 or merely two
  models initialized from one checkpoint; these are different architectures.
- Include class-wise failure cases, the exact random semantic split and a
  separate independent-domain depth evaluation, inference cost, and the effect
  of freezing on both semantic mIoU and stereo EPE.

## Sources to revisit

- [Ultralytics semantic segmentation task](https://docs.ultralytics.com/tasks/semantic)
- [Ultralytics semantic dataset format](https://docs.ultralytics.com/datasets/semantic/)
- [Ultralytics training options (`freeze`)](https://docs.ultralytics.com/modes/train/)
- [Ultralytics fine-tuning guidance](https://docs.ultralytics.com/guides/finetuning-guide/)
- Local literature précis: [`../../.agents/skills/stereo-vision-expert/references/semantic.md`](../../.agents/skills/stereo-vision-expert/references/semantic.md)
