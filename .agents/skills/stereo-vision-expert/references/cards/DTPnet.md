<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# DTPnet (Distill-then-Prune) card

Source: paper/reference_papers/lightweight/Distill-then-Prune_Pan_ICRA2024.pdf (8 pp incl. refs; no supplement). Page refs: p.1-6 body.

### 0. Meta
- Title: Distill-then-prune: An Efficient Compression Framework for Real-time Stereo Matching Network on Edge Devices. Pan, Jiao, Pang, Cheng (UBTech Robotics; BUPT; SIAT-CAS).
- Venue: ICRA 2024 (arXiv 2405.11809v1, 20 May 2024).
- Code: not stated.
- Domain: driving (KITTI2015), synthetic (SceneFlow); edge devices Jetson AGX / TX2.
- Datasets: SceneFlow (35k train / 4.3k test, 960x540); KITTI 2015 (200 train / 200 test, 1242x375) -- "train and test" on KITTI2015, Fig. 1 uses a validation set (split not stated).

### 1. Problem & failure modes targeted
- Edge-device feasibility: three obstacles named (Sec. I): (1) inference SDK operator support (TensorRT has limited 3D conv support; slicing, iterative convs, trilinear interpolation unsupported), (2) heavy compute, (3) low accuracy of real-time nets.
- PSMNet runs 1 fps on AGX (Sec. I).
- Low-texture mismatches (car windows) claimed fixed by context learned by hourglass (Sec. V-D; qualitative only).

### 2. Pipeline by stage
- 2a Features: siamese shared-weight, "feature pyramid with two residual blocks" (two levels, SPP removed), outputs B x C x H/4 x W/4 (Sec. III-A). Feature1 0.01M/0.72G; Feature2 0.09M/0.50G (Tab. II setting 3: 4 residual blocks each in settings 1; fewer in setting 3). Whole extractor 0.10M/0.72G (Tab. I). Trained.
- 2b Semantic/prior branch: n/a. Only prior = teacher PSMNet (training-time distillation target).
- 2c Cost volume: "channel-to-disparity" module: concat(f_l, f_r) -> 3 conv layers (conv+BN+act) that map channels to D/4 disparity channels (Eq. 1); no shift/correlation loop, no slicing, no iteration. Result is a 4D->3D-style tensor handled by 2D conv only. 0.03M, 0.11 GFLOPs vs 0.13M / 7.16 G for MobileStereoNet iterative compression (Tab. II/IV). Max disparity dmax = 192, i.e. D/4 = 48 channels at 1/4.
- 2d Aggregation: ONE stacked-hourglass (2D conv + transposed conv, skip connections, Fig. 2) block instead of 3 (0.51M, 2.67 GFLOPs). 5.42 of 6.25 GFLOPs total (Tab. I).
- 2e Disparity: softmax + soft-argmax (Eq. 2) over dmax=192 levels after upsampling.
- 2f Refinement: none.
- 2g Upsampling: "two-step equivalent of trilinear": 1x1-style conv maps channels D/4 -> D, then bilinear to H x W (hardware friendly; Sec. III-B). Trilinear replaced "in the inference stage" (text p.3).
- 2h Fusion points: none at inference. Teacher (hand-trained PSMNet, EPE 0.69) -> student at logit/probability-volume level (training only).

### 3. Block -> problem -> evidence table
EPE: SceneFlow unless stated; Tab. II does not name the dataset (assumed SceneFlow).
| block | problem | evidence | conditions | cost |
|---|---|---|---|---|
| Setting 1 (MobileStereoNet-like: 4 feature stages, iterative cost, 3 hourglass) | reference | EPE 1.27 (Tab. II) | | 2.04M, 23.4 G, 104 ms on Jetson AGX |
| Setting 2 (fewer feature blocks, 1 hourglass? "/" marks Hourglass2/3 absent, Feature3/4 kept) | | EPE 1.48 (+0.21) | | 0.95M, 15.4 G, 44 ms AGX |
| Setting 3 / DTP arch (Feature3/4 dropped, new cost module, 1 hourglass) | | EPE 1.73 (+0.46 vs S1) | | 0.64M, 4.00 G, 16 ms AGX |
| SPP module | context | EPE 1.76 -> 1.75 (-0.01) for +0.01M +0.40 G (Tab. IV): redundant | | |
| MobileStereoNet iterative cost vs channel-to-disparity | operator support, FLOPs | 1.57 vs 1.73 (+0.16) (Tab. IV) | | -0.1M, -7.05 G |
| Hourglass removal | | EPE 1.73 -> 5.04 (+3.31) (Tab. IV) | | -0.5M, -2.67 G | most important accuracy block |
| Distillation supervision: GT only / GT+teacher / teacher only | | 1.73 / 1.61 / 1.56 (Tab. V) | student of Setting 3 | training-only |
| Distillation loss KL vs L1 on probabilities | | 1.86 vs 1.78 (Tab. VI) | | |
| Feature-based distillation | | claimed infeasible ("only logits-based approach is feasible", Sec. II); no numbers | | not ablated numerically |
| Pruning (DepGraph, L2-norm, r=0.1, E=5 steps -> 50% params) | | NO ablation of pruned vs unpruned EPE or latency in the paper | | not ablated |
| Temperature schedule t: 0.5 -> 1.0 | | not ablated |

### 4. Interactions & dependencies
- Hourglass is indispensable; everything else is cheap to cut (Tab. IV).
- Teacher-only supervision works because the dense GT is already similar to a softmax volume (authors' argument; not tested further).
- Pruning is applied after distillation; each prune step is followed by distillation finetune (Algorithm 1). Unknown whether the final pruned model keeps teacher supervision at KITTI finetune.
- Deployment: TensorRT limitations drive the architecture (2D only, no slicing).

### 5. Losses
- L(p,q) = sum_i | softmax(p_i/t) - softmax(q_i/t) |_1 over dmax=192 disparity bins (Eq. 4), p = student logits, q = teacher logits (roles per text: student learns from teacher). t grows 0.5 -> 1.0 with epoch. No GT term in final recipe ("Knowledge only" best, Tab. V). Per-pixel averaging/normalization not stated. The conventional smooth-L1 disparity loss is dropped.
- Algorithm 1: distill for M steps; then loop E times {prune r%; finetune by distillation M steps}.

### 6. Training recipe
- Teacher: own PSMNet, EPE 0.69 (standard protocol; "hand-trained").
- Student: AdamW (b1 0.9, b2 0.999), LR 1e-3, weight decay 1e-2; SceneFlow 20 epochs, then KITTI finetune 300 epochs at 1e-3 then 300 epochs at 1e-4.
- Pruning: DepGraph structural pruning, kernel importance I(w)=||w||_2, group importance I(g) = sum ||w_i||_2 (Eq. 5); prune rate r=0.1, E=5 steps -> 50% of parameters pruned (note 0.1 x 5 = 50%). After each step finetune 5 epochs SceneFlow and 100 epochs KITTI.
- Hardware for training not stated. Crop/batch/augmentation not stated.

### 7. Results
- KITTI2015 (Tab. III): DTPnet D1-bg/fg/all 2.64/6.47/3.28 %; noc 2.46/5.61/2.98 %; latency 16.3 ms on Jetson AGX. Competitors: MSN2d 2.83 all, 269 ms (AGX); StereoVAE 5.23, 29.8 ms (AGX); AnyNet 8.51, 38.4 ms (AGX); MADnet 4.66, 250 ms (TX2). Note: "same method's inference time on TX2 and AGX is roughly 4x" (Sec. V-C) so cross-platform latency comparisons are rough.
- SceneFlow (Tab. VII): DTPnet EPE 1.56, 0.26M params, 3.677 GFLOPs, 8 ms on Titan XP; PSMNet 1.12 (1083 G, 450 ms), MSN2d 1.12 (128.8 G, 107 ms), MADnet 1.66 (15.6 G, 65 ms), AAFnet 3.90 (11 ms).
- No Orin numbers, no TensorRT/FP16/INT8 stated (quantization is future work). Input resolution for latencies not stated.

### 8. Negative results & limitations
- Errors/inconsistencies I found: (1) Text says "our DPTnet achieves the lowest EPE among all the methods" (Sec. V-C) but Tab. VII shows MSN3d 0.80, DeepPruner 0.97, PSMNet 1.12 and MSN2d 1.12 lower than 1.56; and the table is not sorted descending as claimed. (2) Params/FLOPs are inconsistent across tables: 0.64M/6.25G (Tab. I), 0.64M/4.00G (Tab. II), 0.63M/6.30G (Tab. IV), 0.26M/3.677G (Tab. VII; maybe post-pruning, not stated). (3) Tab. III column "FLOPs(G)" values (0.04, 0.02, 0.36...) look like params in M, DTPnet 0.63. (4) Latency 16 ms (AGX, Tab. II) and 16.3 ms (Tab. III) vs 8 ms (Titan XP, Tab. VII) -- the final latency is not clearly of the pruned model. (5) EPE 1.73 (unpruned, GT) -> 1.56 (teacher-only) but pruned-model EPE is never isolated, so the "prune" half of the title has no ablation. (6) EPE 1.56 after distillation is not beaten by non-distilled lighter methods fairly: teacher is hand-trained PSMNet on same data. (7) No zero-shot/generalization evaluation. (8) Claims "learns semantic information from context" for car windows without evidence.
- Authors admit: feature-based distillation hard; quantization future work.

### 9. Relevance to OUR model
- Our trainable part (fusion head) is already tiny and the stereo/semantic parts are frozen, so compression of the head is not the bottleneck; the shared YOLO trunk is. Pruning ideas apply to the YOLO26m layers 0-6 only if we ever unfreeze/shrink (e.g. a 'trunk-S' student distilled from the 26m trunk with feature + logit losses) -- that would break the exact frozen-weight sharing and the frozen decoder, so we would need to re-distill the decoder too. Not novel in the field, high cost for us, uncertain benefit.
- Reusable cheap idea: logit-level (probability volume) distillation with L1 on softmax-over-disparity, teacher-only supervision. For our head: distill D2/E3 (heavier) into a smaller gate (E-series width sweep shows 1x vs 2x/4x heads), supervising on the candidate-probability distribution rather than only disparity L1. Expected benefit small (head capacity gave limited gains); risk: low; cost: training only.
- Deployment lessons for Jetson: avoid 3D conv, slicing, trilinear interpolation, iterative loops; use D/4 -> D 1x1 conv + bilinear. Our correlation volume + gate must be checked for TensorRT-friendliness (gather/shift ops). This is a concrete checklist item.
- Real-time evidence: 16.3 ms on Jetson AGX Xavier (not Orin) for a 0.64M-param model at KITTI-ish size; helps size budget: 4-6 GFLOPs fits ~16 ms on AGX.
- Not portable: PSMNet hourglass design, SceneFlow EPE claims.

### 10. Key quotes/equations worth citing
- "only the logits-based approach [31] is feasible in the task of stereo matching" (Sec. II).
- Eq. 4 L(p,q) = sum_i | softmax(p_i/t) - softmax(q_i/t) |_1; Eq. 5 I(g) = sum ||w_i||_2.
- Tab. II: 104 ms -> 44 ms -> 16 ms on Jetson AGX with EPE 1.27 -> 1.48 -> 1.73.

### Errata vs existing summary (summaries/lightweight/Distill-then-Prune.md)
- Summary says latency is measured "with TensorRT" on Jetson AGX; the paper never states TensorRT was used for measurement (only that TensorRT motivates operator choices).
- Summary writes "M = 5 pruning steps"; paper uses E = 5 steps (M is max training steps) in Algorithm 1 / Sec. V-A.
- Summary's "36x fewer params than PSMNet" uses 0.26M (Tab. VII) while Tab. I/II/IV say 0.64M; the paper never reconciles this (pre-prune vs post-prune is the likely, but unstated, explanation).
- Summary repeats the paper's claim of lowest EPE without noting Tab. VII contradicts it (MSN3d 0.80, DeepPruner 0.97 lower).
- Summary omits that no pruned-vs-unpruned ablation exists.
