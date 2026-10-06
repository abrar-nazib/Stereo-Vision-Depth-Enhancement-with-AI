<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# Fast-FoundationStereo card

### 0. Meta
- Title: Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching. Authors: Wen, Dewan, Birchfield (NVIDIA). CVPR 2026 (arXiv 2512.11130v2). 18 pages: main p. 1-8, refs p. 9-12, supplement p. 13-18 (cost-filtering search details, iteration study, model efficiency, GWC trick, limitations, qualitative figs).
- pdf: paper/reference_papers/fusion/Fast-FoundationStereo_Wen_CVPR2026.pdf. Code: https://github.com/NVlabs/Fast-FoundationStereo (p. 14); project page nvlabs.github.io/Fast-FoundationStereo.
- Domain: general zero-shot (indoor, outdoor, robotics, in-the-wild). Eval: Middlebury-H/Q, ETH3D, KITTI-12/15, Booster-Q. Training: same mix as FoundationStereo + 1.4M pseudo-labeled Stereo4D pairs.

### 1. Problem & failure modes targeted
- Latency/memory of stereo foundation models (FoundationStereo 496 ms, 374.5M params) vs poor generalization of real-time models (trained per-domain, mostly SceneFlow) (p. 1-2). Also robustness on translucent/specular/textureless (Booster) (Tab. 2).

### 2. Pipeline by stage (student; teacher = FoundationStereo)
- 2a Feature extraction: teacher = DepthAnythingV2 (ViT, frozen in distillation) + side-tuning CNN. Student = single timm-style CNN (EdgeNeXt / MobileNetV2 cited as variants; exact student per speed point not stated). Multi-level pyramid f^(i) in R^{C_i x H/i x W/i}, i in {4,8,16,32} (Sec. 3.1). Student trained to match teacher pyramid by MSE; linear projection if channels differ; both stereo images included in each batch (single-image encoder).
- 2b Semantic/prior branch: the DA-V2 mono prior is distilled INTO the single backbone (no separate branch at inference). Segmentation models only used offline for sky masks (Sec. 3.4).
- 2c Cost volume: V_C in R^{C x D/4 x H/4 x W/4} = group-wise correlation + concatenation volume; max disparity 192 default (416 for Midd-H cost-filtering variants, Sec. 4.3). Efficient GWC: left-pad then unfold (zero-copy sliding windows), single fused multiply-sum: ~6x runtime and 3x memory reduction on Middlebury-Q (supp Sec. 10, p. 15).
- 2d Aggregation: teacher: 3D hourglass with Axial-Planar Convolution (APC) layers + parallel Disparity Transformer (MHSA on tokenized V_C). Student: divided into N=8 blocks (transformer = one block), each replaced by searched candidate. Candidate layer types (supp Sec. 7): 3D conv (0.5x/1x/2x channels, k=3), 3D deconv, APC (0.5x/1x ch, axial kernel 3/9/17, planar 3), residual 3D conv (Basic Block, 0.5x/1x), feature-guided volume excitation (multi-level left features), transformer layers repeat 1-6, FFN 2x/4x, heads 2 or 4; layers per block <= teacher's; in/out channels fixed; block must be faster than teacher counterpart.
- 2e Disparity: initial disparity d_0 predicted by the filtered cost volume (soft-argmin implied; not stated), supervised with smooth L1 for the last block.
- 2f Refinement: ConvGRU (teacher) pruned; default 8 iterations (Sec. 4.1); hidden state initialized from context network; convex upsampling mask predicted by GRU final layers (Sec. 3.3).
- 2g Upsampling: convex upsampling mask (GRU output channels kept fixed, Sec. 3.3).
- 2h FUSION POINTS (mono prior): only inside backbone via feature distillation (feature level, mono->stereo features, pyramid 1/4-1/32); plus output-space distillation through pseudo-labels (loss-side, normal-consistency of stereo teacher vs UniDepthV2 mono depth). No semantic fusion at inference.

### Compression pipeline (stage by stage)
1. Backbone distillation (Sec. 3.1): ViT+CNN -> one CNN; MSE; loss ablation Tab.3.
2. Cost-filtering blockwise NAS (Sec. 3.2): train each candidate block B_i^c independently to mimic teacher block output ||B_i(f_{i-1}) - Bbar_i(f_{i-1})||_2^2 given TEACHER's previous-block features (last block: smooth L1 to GT disparity). Evaluate by swapping into teacher, measure relative error change Delta m and runtime change Delta t on a validation set. ILP (PuLP) solves min sum Delta m^T e s.t. sum Delta t^T e <= Delta tau (Eq.1). Search space 5.5e24 designs (5.8e19 faster than teacher); only 2584 blocks trained; complexity O(n^N) -> O(n); distillation takes 14 days on 128 A100 (supp p. 13); ILP <1 s per budget. Table-1 model uses Delta tau = -0.04 s.
3. Refinement structured pruning (Sec. 3.3): recurrent dependency graph; constraints: final disparity/mask layers keep output dims; input channel of layer consuming h_{k-1} and output channel of layer producing h_k jointly pruned; motion encoder input (indexed volume features) fixed. Importance by first-order Taylor expansion accumulated over multiple iterations of teacher end-to-end; prune ratio alpha; isomorphic pruning slightly worse. Retrain refinement end-to-end with rest of teacher frozen: L = sum_k gamma^{K-k}||d_k - dbar||_1 + lambda sum_l ||x_l - xbar_l||_2^2, gamma=0.9, lambda=0.1 (Eq.2; initial disparity excluded). Table-1 model pruning ratio 0.6.
4. Pseudo-labeling (Sec. 3.4, supp 13): Stereo4D rectified pairs (video, temporal stride 10) -> FoundationStereo disparity + UniDepthV2 mono depth; both converted to normal maps via camera params + Sobel; per-pixel cosine similarity thresholded -> consistency mask; sky (open-vocab segmentation [SAM2/ODISE refs 52,77], p. 6) excluded from the comparison and set to zero disparity; keep sample if >60% positive consistency pixels (excl. sky) -> 1.4M pairs. Mask can optionally gate supervision pixels.
5. Final: assembled student trained end-to-end on same mixed data + pseudo-labels (feature/arch pieces already distilled).

### 3. Block -> problem -> evidence table
| block | problem | evidence (ref) | context | cost |
|---|---|---|---|---|
| Backbone distillation (MSE) | ViT hybrid backbone latency; keep mono/stereo priors | Tab.3: no distill (ImageNet-pretrained) Midd-H BP-2 2.87 / ETH3D BP-1 2.11 / KITTI-12 D1 2.67 / KITTI-15 D1 4.32; cosine 2.29/1.19/2.39/3.31; MSE 2.20/1.22/2.35/3.25. Fig.4: distilled features capture edges+relative depth; translucent door example. | zero-shot | feature extraction ~0.1 s in FS (Fig.10) -> small in student |
| Cost filtering blockwise NAS | 3D hourglass + transformer latency | Fig.8: searched candidate beats 10 random candidates at every latency budget Delta tau in {-0.06..0}; random ones collapse at tight budgets (BP-2 up to ~5%) vs ~1.5-1.9% searched. Fig.11 (supp): direct structured pruning of cost filtering: BP-2 goes 2% -> ~55% at 0.2 and 100% at >=0.4 while runtime only 70 -> 60 ms: ineffective (small channel dim <100). | Midd-Q | search 14 d / 128 A100; one-time |
| Refinement pruning + retrain | GRU redundancy | Fig.9: prune-only BP-2 rises from ~2% (0) to ~11% at ratio 0.8; prune+retrain stays ~2% up to 0.8; refinement runtime 1 iter ~7 ms -> ~3.5 ms. Fig.12: at ratio 0.8 accuracy barely improves with more iterations; at 0.6 saturates ~8 iters; runtime at 2/4/8/16 iters ~33/38/46/62 ms (ratio 0.6) | Midd-Q, single refinement iter in Fig.9 | hardware: 3090 per Sec.4.3 (Fig.12 hw not stated) |
| Pseudo-labels (1.4M) | synthetic-only training gap | Tab.4 (in parentheses = with pseudo-labels): RT-IGEV Midd-H BP-2 11.52 (8.69), ETH3D BP-1 5.66 (5.12), KITTI-12 D1 4.54 (3.55), KITTI-15 6.00 (4.40); LightStereo-L 23.76 (18.41), 45.46 (21.12), 13.98 (5.27), 12.08 (7.63); Ours 2.53 (2.20), 1.31 (1.22), 2.44 (2.35), 3.48 (3.25). Help is much larger for SceneFlow-only baselines than for ours. | zero-shot | training data only |
| Efficient GWC | kernel-launch overhead | ~6x faster, 3x less memory at Midd-Q (supp p. 15); no ablation table | Midd-Q | free |
| TensorRT | deploy | 49 ms -> 21 ms (Tab.1 parentheses, Fig.2) | 3090 | - |
| Student backbone variants | speed/accuracy | family in Fig.2 (FPS ~20-75 on 3090; per-variant numbers not tabulated) | | |
| Normal-consistency filter vs depth/disparity-space compare | claimed more robust | not ablated | | |
| Quantization | | not done (future work, p. 8) | | |
| Specific student architecture/params | | Tab.5: 14.6M params, 309.9 GMACs; Tab.6: 17.65M (inconsistent; Tab.6 likely the slowest/largest model) | | |

### 4. Interactions & dependencies
- Blockwise search needs a frozen teacher (each block trained on teacher's previous-block features) and the ILP surrogate assumes block effects add (validated only by Fig.8 vs random candidates, not vs full evolutionary search). Pruning works for refinement only with retraining incl. feature distillation; direct pruning fails for cost filtering. Aggressively pruned GRU cannot exploit more iterations. Pseudo-labels help all models but relatively more those trained on SceneFlow alone. Backbone distillation depends on stereo-pair batches to keep statistics.

### 5. Losses
- Backbone: MSE between student and teacher pyramids (plus linear projection). Block: ||B_i(f_{i-1}) - Bbar_i(f_{i-1})||_2^2; final block smooth L1 vs GT. Refinement retrain Eq.2 above. Final end-to-end training loss: "same as FoundationStereo" (not restated; pseudo-label supervision details beyond mask not stated).

### 6. Training recipe
- Same mixed datasets as FoundationStereo + pseudo-labels; final model trained end-to-end. Optimizer/LR/batch/crop/steps for final stage: not stated. Block distillation: 14 days on 128 A100 (supp). 8 refinement iterations, 192 max disparity default. Latency hardware: RTX 3090 (main), plus 4090, A100 (Tab.5).

### 7. Results (all ZERO-SHOT; in-domain not reported)
- Tab.1 (Midd-H BP-1/2/3; Midd-Q; ETH3D; KITTI-12 BP-1/2/3/D1; KITTI-15 BP-1/2/3/D1; runtime ms), 3090, Midd-Q resolution for timing:
  - FoundationStereo: 2.49/1.10/0.88; 2.64/1.30/0.96; 0.50/0.30/0.24; 8.16/3.50/2.47/2.30; 18.65/5.20/2.95/2.80; 496 ms.
  - Ours: 4.80/2.20/1.60; 4.51/2.12/1.57; 1.22/0.62/0.50; 8.52/3.61/2.50/2.35; 19.62/5.78/3.43/3.25; 49 ms (21 ms TRT).
  - MonSter 9.33/4.24/2.69; ...; ETH3D 0.99/0.46/0.28; KITTI-15 D1 3.41; 336 ms. DEFOM-Stereo 8.84/3.76/2.46; ETH3D 2.16/1.03/0.78; KITTI-15 D1 4.58; 371 ms. StereoAnywhere 9.67/4.75/2.45; KITTI-15 D1 3.52; 427 ms. Zero-RAFT-Stereo 8.48/4.68/3.32; D1 4.48; 164 ms.
  - Real-time group (SceneFlow-only marked *): RT-IGEV* 16.95/11.52/9.40, D1(15) 6.00, 45 ms; RT-IGEV (combined data) 12.75/7.82/5.73, ETH3D 5.05/2.78/1.63, KITTI-15 D1 4.00; LightStereo-L 22.64/12.55/9.07, D1(15) 4.51, 30 ms; IINet* 25.88/16.69/13.03; BANet-2D*/3D* 43-45/28-30; 26-29 ms.
- Tab.2 Booster-Q (BP-2/4/6/8/EPE): Ours 6.61/4.62/3.91/3.49/1.54; FoundationStereo 5.18/4.07/2.91/2.59/1.13; RT-IGEV 23.09/16.86/14.10/12.47/5.03; RT-IGEV† (retrained with pseudo-labels) 18.19/13.39/11.37/10.16/4.20; StereoAnywhere 9.01/5.40/4.12/3.34/1.21.
- Tab.5 efficiency: FoundationStereo 496/295/308 ms (3090/4090/A100), 374.5M params, 5413.9 GMACs; Ours 49/30/41 ms, 14.6M, 309.9 GMACs. Tab.6 params: BANet-3D 3.63, BANet-2D 5.46, RT-IGEV 4.17, IINet 19.56, LightStereo-L 24.29, Ours 17.65. Peak memory 0.63 GB (Midd-Q), "fits Jetson Orin / Thor" (claim; no Jetson measured).
- Latency per stage (Fig.10, bar read-off, NOT tabulated): FoundationStereo total ~0.50 s: feature ~0.10, cost filtering ~0.10, refinement ~0.29 s; Ours total ~0.05 s with each of the three >10x faster (text p. 8). Pipeline progression numbers per individual stage (e.g. ms after distill only): not stated. Step ablation of accuracy vs speed per stage only in Figs. 8-9, 11-12.

### 8. Negative results & limitations
- Authors: translucent surfaces remain hard (inherited from teacher, Tab.2 / supp Sec. 14); direct pruning of cost filtering fails; quantization left to future work; accuracy loss vs teacher (BP-2 Midd-H 1.10 -> 2.20, ETH3D BP-1 0.50 -> 1.22, KITTI-15 D1 2.80 -> 3.25), i.e. roughly 2x error, not "slight".
- Mine: all latency on desktop GPUs (3090/4090/A100), none on Jetson; Table 1 compares to real-time baselines that are mostly SceneFlow-only (starred) -- they provide retrained (†) versions only for LightStereo/RT-IGEV and only in Tab.2/4; extraordinary compute for search (14 d x 128 A100) and reliance on a closed teacher; pseudo-label pipeline untested on semantic/boundary quality; "real-time" 49 ms on 3090 = ~20 FPS, TRT 21 ms; Table 5 vs 6 parameter inconsistency (14.6M vs 17.65M); no seed variance; one student config per table.

### 9. Relevance to OUR model
- Our network is already tiny (frozen YOLO trunk layers 0-6 + shallow head); the NAS/prune pipeline is not directly useful for a head this small, and our fusion head is not a 3D hourglass. Portable, in order of value:
  1. Pseudo-label pipeline for real-domain adaptation of the FUSION HEAD ONLY: run a stereo foundation teacher + mono depth on unlabeled real stereo, keep pixels with normal consistency, mask sky with our own segmentation (sky class from the YOLO semantic decoder, no extra model). Directly addresses our in-domain/VKITTI-only caveat; fine-tune D2/E3 gate on real pseudo-GT. Benefit: potentially large real-domain gain (Tab.4: huge for SceneFlow-only models); risk: pseudo-label bias toward teacher, test on KITTI2015 must be kept separate; cost: offline only.
  2. Fused GWC/unfold volume construction trick for the A09 correlation volume (6x time / 3x mem on 3090, supp p. 15), plus TRT compile: helps RTX 3050 4GB/Jetson. Zero accuracy risk.
  3. Feature-pyramid MSE distillation to the shared trunk: only if we later unfreeze/adapt the trunk to also mimic a stereo/mono teacher; conflicts with the frozen-trunk constraint. Lower priority.
  4. Blockwise-ILP search could size the fusion head (E-series 1x/2x/4x widths) per latency budget: modest, since our head sweeps already exist.
  5. Refinement pruning with retrain + feature distill: applicable if we add a recurrent refiner; we do not.
- Novel combination vs done: semantic gate trained on pseudo-labels with normal-consistency masks is novel for us; compression recipe is done by them.

### 10. Key quotes/equations
- "divide-and-conquer acceleration strategy" (p. 2); Eq.1 ILP, Eq.2 retraining loss; "reduces training complexity from O(n^N) to O(n)" (p. 5); "pseudo-labeling ... normal consistency is more robust to extremely diverse depth ranges" (p. 6); "performance on translucent surfaces remains a challenge" (p. 16).

### Errors in existing summary (summaries/fusion/Fast-FoundationStereo.md)
1. Zero-shot table mislabels columns: header "Midd-H BP-2 / ETH3D BP-1 / KITTI-15 D1" but values are wrong columns. Correct (Tab.1): FoundationStereo Midd-H BP-2 1.10 (summary 2.49 = BP-1), ETH3D BP-1 0.50 (summary 0.30 = BP-2), KITTI-15 D1 2.80 (summary 2.95 = BP-3). Ours Midd-H BP-2 2.20 (summary 4.80 = BP-1), ETH3D BP-1 1.22 (summary 0.62 = BP-2). MonSter Midd-H BP-2 4.24 (summary 9.33), DEFOM Midd-H BP-2 3.76 (8.84) and ETH3D BP-1 2.16 (1.01 appears nowhere), RT-IGEV Midd-H BP-2 7.82 (12.75) and ETH3D BP-1 5.05 (1.63 is BP-3). D1 values 4.58/3.41/4.00/3.25 and runtimes are right. Under the corrected numbers Ours doubles FoundationStereo's error (not "modest").
2. "timm CNNs (~17M)" under backbone: 17.65M is the WHOLE model (Tab.6); the backbone size is not stated. Also Tab.5 says whole model 14.6M; summary quotes both without noting the inconsistency.
3. "Student: EdgeNeXt, MobileNetV2": paper only cites them as feature-extractor variants (p. 4); exact student not stated.
4. "Peak memory 0.63 GB ... fits edge GPUs": measured on a desktop 3090 at Midd-Q; paper's Jetson claim is untested.
5. Omits: 14 days on 128 A100 search cost, Delta tau = -0.04 s, pseudo-label thresholds (>60%, stride 10, sky zeroed), and that segmentation models (SAM2/ODISE) are used for sky masks.
Verified correct: 374.5M/14.6M, 5413.9/309.9 GMACs, 496/49/21 ms, Tab.3 numbers, 0.6 pruning ratio, 8 iterations, 6x GWC speedup, 2584 blocks, 1.4M pairs, Eq.1-2.
