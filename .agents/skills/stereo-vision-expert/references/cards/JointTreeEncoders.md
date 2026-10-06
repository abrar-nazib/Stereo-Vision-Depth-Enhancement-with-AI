<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# JointTreeEncoders card

Source read: full 8-page PDF (arXiv 2609.13232v1, 2 Sep 2026), text + pages 1-2 rendered. No appendix. I found no literal placeholder text (no "TODO"/"XX"/"??" strings) in the extracted text; the "draft" impression comes from the issues in Sec. 8 below (single seed, "will be released", unfinished runs, no losses weights, missing group B). No repo summary existed for this paper.

### 0. Meta
- Title: What Does the Encoder Actually Decide? A Controlled Comparison of 64 Vision Backbones on Joint Tree Segmentation and Stereo Depth.
- Authors: Yida Lin, Bing Xue, Mengjie Zhang (Victoria Univ. of Wellington), Sam Schofield, Richard Green (Univ. of Canterbury). arXiv preprint, Sep 2026.
- PDF: paper/reference_papers/semantic_stereo/JointTreeSegStereoEncoders_Lin_arXiv2026.pdf
- Code: "Code, the tree50 renderer and the trained checkpoints will be released with the paper" (Reproducibility, p. 7). Not available.
- Domain: forestry robotics, tree pruning; synthetic rendered forest scenes (thin, self-occluding vegetation). NOT driving.
- Dataset: "tree50", 50 rendered scenes x 48 viewpoints = 2,400 rectified stereo pairs at 1920x1080 with per-pixel depth and tree masks. Scene-exclusive split: scenes 1-40 train, 41-45 val, 46-50 test (test = 240 frames but only 5 independent scenes) (Sec. III-A). Trees occupy 75-91% of pixels. Dataset not public.

### 1. Problem & failure modes targeted
- Question: with dataset, decoders, losses, schedule and evaluation fixed, how much does the shared encoder choice change joint tree-segmentation + stereo-depth quality on thin structures? Encoders are usually chosen "by reputation".
- Failure modes studied: encoder choice itself; "label-everything-tree" degenerate collapse (hidden by region IoU since all-tree gives ~0.93 tree IoU); data-hungry transformers from scratch; far-range depth tail errors (right-skewed). No fusion/cross-task mechanism is targeted. By design there is NO cross-task pathway ("any cross-task pathway would blur the attribution", Sec. II-C).

### 2. Pipeline by stage
- 2a Feature extraction: one encoder with shared weights applied to left and right images, returning five maps at strides 2, 4, 8, 16, 32 with spatial sizes ceil(H/2^{i+1}); the "five-scale contract": any backbone implementing forward_features -> 5 maps can be swapped in a one-string change (Sec. III-C). Inputs reflection-padded to a multiple of 32. Channel counts differ per encoder. Trained from scratch; never frozen; never pretrained.
- 2b Semantic branch: U-Net decoder over all five scales, channel widths (128, 96, 64, 48), per-pixel tree/background labelling (2 classes). Footnote 1: the decoder also carries centre and offset heads for an instance-grouping extension (supervised in all runs, never reported). No separate semantic branch from the stereo branch.
- 2c Cost volume: group-wise correlation built from left and right stride-4 features, disparity range covers the native maximum (value not stated), at stride 4 (Sec. III-B, V). Number of groups: not stated.
- 2d Aggregation: 3D convolutions (details/layer counts not stated), as in PSMNet/GwcNet lineage.
- 2e Disparity: soft-argmin; depth by z = f_x B / d (disparity predicted once and converted; verified to 0.0 px mean against rendered output).
- 2f Refinement: "refined residually" (a coarse and a refined stage; no iterations, architecture not stated).
- 2g Upsampling: not stated.
- 2h FUSION POINTS: none between tasks. Only the shared encoder couples them (hard parameter sharing). Seg->depth and depth->seg interactions: n/a. Depth is evaluated on predicted tree mask as one secondary "end-to-end usability" view, which is a post-hoc evaluation, not a fusion.

### 3. Block -> problem -> evidence table
The paper's ablation unit is the encoder (64 runs); there is NO ablation of the heads, of sharing vs not sharing, of freezing, or of the cost-volume branch. Components claimed but not ablated are marked.
| block | problem it solves | evidence | context/conditions | cost |
|---|---|---|---|---|
| Encoder family: CNN / hybrid (G median score 0.600; D 0.590; A 0.554) | accuracy on thin-structure seg+depth | group medians Tab. II: G mIoU 0.599 / delta1 0.599 / EPE 19.74; D 0.585 / 0.597 / 19.32; A 0.596 / 0.564 / 21.26; F hier. transformer 0.569 / 0.594 / 21.30 | from scratch, single seed, 512x288, 100 epochs | see Tab. III |
| Plain ViT (group E, n=4) | | median 0.329; 3 of 4 collapse (beit_base 99.7 M, clip_vit_base 100.3 M, cait_xxs24 12.6 M); deit_tiny (6.4 M) survives rank 23 | scratch only | |
| SSM/Mamba (I, n=2) | | both collapse: mambavision_t mIoU 0.464, IoUbg 0.000 | | |
| MLP/attn-free (H, n=9) | | median 0.496; resmlp_12 last (delta1 0.032, AbsRel 122) | | |
| Top encoder internimage_t (D) | | mIoU 0.650, IoUbg 0.357, BF1 0.419, AbsRel 0.786, delta1 0.725, EPE 10.85 px; composite 0.688 (Tab. I) | | 28.8 M enc, 2213 GFLOPs, 3.7 FPS, 3.5 GB at 1920x1080 batch 1 (Tab. III) |
| Best seg densenet121 | | mIoU 0.654, BF1 0.489 | | 7.0 M, 2022 GFLOPs, 7.9 FPS |
| Best AbsRel swin_tiny_w4 | | AbsRel 0.358, delta1 0.701 | | 27.5 M, 4.0 FPS |
| Small compact encoders on Pareto front | efficiency | edgenext_xxs (G, 1.2 M enc): mIoU 0.626, delta1 0.686, composite rank 6, 9.2 FPS, 2.9 GB; convnext_atto (D, 3.4 M): 0.606 / 0.691, 9.3 FPS; wrn_40_4 (A, 16.1 M): 0.649 / 0.699, 9.4 FPS; hrnet_w18_small (3.9 M): 0.619 / 0.651 (Tab. I, III) | | |
| Collapse detector metrics (IoUbg, BF1) | all-tree degenerate solutions | 25 of 64 encoders collapse (tree IoU ~0.93, bg IoU ~0, mIoU ~0.46) (Sec. VI-A) | | |
| Hard parameter sharing (one trunk for both tasks) | amortise compute | Spearman rho = 0.89 between seg-rank and depth-rank over 64 encoders (Fig. 4) -> "no genuine task conflict"; NOT ablated against single-task or against separate encoders | | |
| Seg U-Net decoder, stereo cost-volume branch, losses | | not ablated | | |
| Capacity | | params do not predict quality; the two ~100 M ViTs are the slowest and worst (Sec. VI-C); only 22 of 64 encoders are within +/-10 M of the ~25 M target, median 11.7 M, range 0.2-100.3 M | | |
| Test-split noise | | cluster bootstrap over 5 test scenes, 2000 resamples: Spearman 0.994 (2.5th pct 0.962) vs reported order; internimage_t first in 100% (Sec. VI-D) | scene noise only; no seed variance | |

### 4. Interactions & dependencies
- Plain ViTs only fail because of the from-scratch regime ("data-hungry"); the paper itself says the ordering "need not match one obtained with pretrained initialisation" and that pretraining may reorder it.
- Depth metrics are mean-based and dominated by the far-range tail (median abs error 0.20 m vs mean 1.39 m for the best encoder, 2.45 m at the 90th percentile).
- Depth is scored on GT tree pixels, so a segmentation collapse does not by itself ruin depth (collapsed maxvit_tiny has delta1 0.615), yet the composite score and rank-correlation treat the two as co-varying.
- Memory-heavy encoders (SSM, global attention) needed micro-batching with gradient accumulation to keep the effective batch of 12 (Sec. V).

### 5. Losses
- Segmentation: cross-entropy. Stereo: smooth-L1 on disparity at both the coarse and refined stages (Sec. V). Weights between the two tasks, coarse/refined weights, and any instance-head loss weights: not stated. Checkpoint selection by lowest total validation loss (not by any single task metric).

### 6. Training recipe
AdamW lr 3e-4, weight decay 1e-2 (norm layers and biases excluded), cosine decay after linear warmup (warmup length not stated), mixed precision, gradient clipping at norm 1.0, 100 epochs, input 512x288 (image downscaled from 1920x1080; all resolution-dependent constants derived by one function), batch 12 (micro-batching where needed). All 64 encoders from scratch, single seed, isolated subprocess per encoder, GPU: a single 16 GB GPU (model not stated; Tab. III caption). Augmentation: not stated.
- Frozen encoders: NEVER tested. Pretrained: NEVER tested (deliberate).

### 7. Results
- Evaluation: seg = tree-vs-bg mIoU, IoUbg, BF1 (2 px), pixel accuracy; depth = AbsRel, RMSE, SILog, delta<1.25^k, EPE, bad-tau, on GT tree pixels; efficiency measured end-to-end at 1920x1080 batch 1 (Tab. III).
- Composite = mean(mIoU, delta1). Top 6 (Tab. I): 1 internimage_t 0.688; 2 wrn_40_4 (A) mIoU 0.649 / delta1 0.699; 3 swin_tiny_w4 (F) 0.633 / 0.701; 4 xception_mobile_order 0.634 / 0.687; 5 res2next50 0.635 / 0.680; 6 edgenext_xxs 0.626 / 0.686.
- Group medians (Tab. II): A Classic CNN n=15 composite 0.554; C Lightweight CNN n=5 0.477; D Modern CNN n=8 0.590; E Plain Transformer n=4 0.329; F Hier. Transformer n=9 0.552; G CNN-Transformer hybrid n=12 0.600; H MLP/attn-free n=9 0.496; I SSM n=2 0.446. (Group B is absent: labels run A, C-I; the paper says "8 groups, labelled A-I".)
- Lightweight CNNs (MobileNetV4-conv-S, FasterNet-T0, MobileNetV1-0.25, MicroNet-M0, GhostNet-0.5 [ghostnet_0_5 does fine: 0.567/0.560]) mostly collapse: group C median BF1 0.004. Within-group spread is as large as between-group spread (classical CNNs span rank 2 to near-worst).
- No stereo metrics on any standard benchmark; all absolute depth numbers are huge (best encoder EPE 10.85 px, group median EPE 19-33 px at the evaluation resolution; resolution of the EPE not stated).

### 8. Negative results & limitations
Authors (Sec. VIII): single seed; synthetic data and only 5 test scenes; no pretraining; one representative per family with unequal budgets (4 families only at ~100 M); runs that exceeded memory or did not finish are dropped ("64 encoders completed; the remainder exceeded the memory budget or had not finished").
My credibility notes:
1. Reads like an unfinished benchmark draft: no code released, dataset private, many specifics missing (disparity range, number of groups/3D-conv layers, loss weights, warmup, augmentation, GPU model, resolution of the reported EPE). Group "B" is missing from a scheme described as 8 groups A-I.
2. 25/64 encoders collapse to all-tree segmentation. Training failures in single from-scratch runs are more likely optimisation/LR failures than architecture properties (identical lr 3e-4 for every family, including ViTs and MLP-Mixers). The headline "CNN and hybrids beat transformers" is therefore a statement about a fixed recipe, and the authors concede it.
3. The rho = 0.89 "no task conflict" claim is rank correlation over 64 encoders, 25 of which are tied near-zero collapse on seg; large collapse clusters drive correlation. No single-task baselines, no separate-encoder baseline, no cross-task-gradient analysis, so "no genuine task conflict" is not measured; it only shows that good backbones are good for both under a shared recipe.
4. Composite score gives equal weight to mIoU (2-class, saturated) and delta1; score is acknowledged as "a convenience".
5. Depth numbers are poor in absolute terms (EPE >10 px for the best encoder) so the regime is far from usable stereo; conclusions might not transfer to a trained-to-convergence cost-volume net on standard data.
6. No stereo-specialised encoders; the stereo branch is a basic GwcNet-style net; semantic segmentation is binary. No YOLO/CSPDarknet encoder was tested.
7. Internal consistency looks fine: Tab. I is correctly sorted by the composite (checked rows 1-9), Tab. II medians are consistent with Tab. I.

### 9. Relevance to OUR model
- What it supports: (a) a single shared encoder for segmentation + stereo is not obviously harmful (rho 0.89), a weak external justification for our shared trunk; (b) small convolutional/hybrid trunks are competitive (edgenext_xxs 1.2 M, convnext_atto 3.4 M, hrnet_w18_small 3.9 M); consistent with a YOLO26m CNN trunk (layers 0-6) being adequate; (c) region IoU hides collapse: adopt boundary-F1 / bg-IoU style diagnostics when judging our semantic-gated outputs, and edge metrics (already in G/H series).
- What it does NOT support: our key design choice is a FROZEN trunk pretrained on detection/segmentation; this paper tests only from-scratch, end-to-end trained encoders, never frozen ones or pretrained ones. It says nothing on freezing and says pretraining might reorder the ranking. It does not test any semantics->stereo fusion (it deliberately has none), so it cannot be cited as evidence for or against semantic cost gating.
- Use in the paper: cite as the only controlled encoder-sweep for a shared seg+stereo trunk and as an honest caveat ("hard sharing without cross-task pathway; from scratch; synthetic forest; single seed"). Do not cite for "frozen encoders work". Possible low-cost extension for us: a small encoder swap on our frozen-trunk pipeline is NOT needed; the paper shows backbone swaps are high-variance and single-seed.
- Novelty implication: this paper is a benchmark, not a competitor; it does not claim a model. It reduces nothing about our novelty but supplies a "shared encoder without interaction" reference point (hard-sharing baseline). Our A09 + frozen decoder without fusion head (F1) is exactly that baseline.

### 10. Key quotes/equations worth citing
- "the segmentation and depth rankings agree strongly (Spearman rho = 0.89), so the shared encoder faces no genuine task conflict" (Abstract).
- "25 of 64 encoders collapse to a degenerate all-tree segmentation that region IoU hides but boundary F1 exposes" (Abstract).
- "We deliberately use the plain hard-sharing form: any cross-task pathway would give the encoder a second route to influence the result" (Sec. II-C, p. 2).
- "This lowers absolute accuracy and is a deliberate trade; it also means the ranking here need not match one obtained with pretrained initialisation" (Sec. V).
- d = f_x B / z (Sec. III-A).
