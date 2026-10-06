<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# S3M-Net card

PDF read in full via pdftotext + page render of Fig. 1 (p.3). The PDF has NO supplement/appendix (11 pages + refs). Page refs are PDF pages.

### 0. Meta
- Title: S3M-Net: Joint Learning of Semantic Segmentation and Stereo Matching for Autonomous Driving
- Authors: Zhiyuan Wu, Yi Feng, Chuang-Wei Liu, Fisher Yu, Qijun Chen, Rui Fan (Tongji)
- Venue: IEEE Trans. Intelligent Vehicles 2024 (arXiv 2401.11414v2)
- PDF: paper/reference_papers/semantic_stereo/S3M-Net_Wu_TIV2024.pdf
- Code: only "project webpage mias.group/S3M-Net" stated (p.1). A GitHub URL appears only in the repo's own summary, not in the PDF.
- Domain: driving. Datasets: vKITTI2 (synthetic, 15 classes stated; legend Fig. 3 lists 14 names) and KITTI 2015 (19 Cityscapes classes).
- Splits (Sec. IV-A, p.5): vKITTI2 = 700 randomly selected stereo pairs, 500 train / 200 "validation" (the 200 are the only held-out set; they serve as test; no separate val, no checkpoint-selection rule stated; whether the 10 weather/lighting variations of one (scene, frame) are grouped is NOT stated -> possible frame leakage). KITTI 2015 = 400 pairs, 200 with GT; "70% of the dataset training, rest test" (p.5) -> presumably 140/60 of the 200 GT pairs (my inference; not spelled out). Source of KITTI-2015 semantic labels not stated. Sparse LiDAR disparity GT for KITTI.

### 1. Problem & failure modes targeted
- Separate seg + stereo nets cost compute and miss shared info (p.1). Stereo ambiguity in texture-less / occluded regions; semantic consistency claimed to disambiguate (p.1, p.7).
- Prior joint nets need big pre-training (SegStereo/DispSegNet on Cityscapes; SceneFlow pretrain for [9][11][12]) or alternate-freeze training (DSNet) (Sec. II-C, p.3). S3M-Net is end-to-end, claims to work with limited data.
- Mismatched channel widths: stereo needs ~256 ch (RAFT-Stereo), seg wants up to 2048 ch (SNE-RoadSeg) (Sec. III-C, p.4). This is the ONLY place a task-demand mismatch is mentioned; it is solved by an adapter (FFA), not analysed as gradient conflict.
- Claims "joint learning acts as regularization that reduces over-fitting" (p.1, p.6, p.10) but gives no joint-vs-separate ablation (see Sec. 3 and 8).
- Explicit "task conflict" discussion: none. No gradient-conflict, loss-weight-sweep-between-tasks, or freeze-encoder ablation.

### 2. Pipeline by stage
- 2a Feature extraction: "joint encoder" = series of residual blocks + downsampling producing multi-scale F^L={F_1..F_n}, F^R (Sec. III-A). Built on RAFT-Stereo's encoder. Scales/channels not stated (RAFT-Stereo-typical 256 ch mentioned, p.4). Trained end-to-end; only F^L is shared with segmentation. Context features (separate RAFT-style context encoder, Fig. 1 yellow) feed the GRU; not shared with seg.
- 2b Semantic branch: NOT a light head. FFA module (Fig. 2, p.4) remaps left shared features via R (3x3 stride-2 conv-BN-ReLU, channels 64/256/512) and fuses with disparity features from E^D = ResNet-152 run on the predicted disparity map D_n (final 1024/2048-ch maps); residual encoders E on the fused features; then SNE-RoadSeg dense-skip decoder (3x3 convs, upsample, N-class output). Trained from scratch jointly (no pretrained seg net stated; ResNet-152 init not stated). So the seg branch is a big RGB-D (RGB-X) segmenter whose "X" is the predicted disparity.
- 2c Cost volume: all-pairs 1D correlation C1 in R^{HxWxW}, C1(i,j,k)=F^L_n(i,j,:)·F^R_n(i,k,:) (Eq.1), pyramid by 1D avg-pool k=2,s=2: C^m in R^{HxWxW/2^{m-1}} (RAFT-Stereo style). m (levels) not stated; max disparity 192 (p.5).
- 2d Aggregation: none (no 3D conv); multi-level GRU only.
- 2e Disparity: regression via GRU updates starting D0=0; D_i in R^{HxW} (stated as HxW; internal 1/4 resolution + upsampling not described).
- 2f Refinement: multi-level GRU update operator, n iterations (n not stated), outputs sequence D={D_1..D_n}, loss on all.
- 2g Upsampling: not described.
- 2h FUSION POINTS:
  1. Feature sharing: left encoder features F^L -> seg branch via R remap (feature level, disp->seg direction is "stereo features help seg"; element-wise addition, Eq.2).
  2. Disparity map D_n -> ResNet-152 E^D -> added to seg features at every stage (disp->seg, addition) (Eq. 2-3). Whether gradient flows from seg back through D_n into the stereo path is NOT stated (no stop-grad mentioned).
  3. Seg -> disparity: NO direct feature path. Seg influences stereo only via (a) gradients into the shared encoder through R, and (b) the SCG weight map W built from GT labels that re-weights the disparity loss (loss-only coupling).
  So the architecture is mostly stereo->seg; the stereo side gets semantics only through shared-encoder gradients + loss weighting.

### 3. Block -> problem -> evidence table
| block | problem it solves | evidence (ablation delta) | context | cost |
|---|---|---|---|---|
| Joint (shared) encoder + FFA vs separate nets | save compute; regularize; share info | NO controlled ablation (no "stereo-only RAFT-Stereo same recipe" row, no "seg-only" row). Indirect: S3M-Net vs RAFT-Stereo in Tab. III/IV: EPE 0.40->0.39 (no SCG) / 0.38 (SCG) on vKITTI2; 0.60->0.56 / 0.55 on KITTI. | RAFT-Stereo baseline training recipe not stated (trained by authors? unspecified) | 0.66 fps @1248x384 (p.11); params not stated |
| FFA fusion op: addition | stable fusion of heterogeneous feats | Tab. V (KITTI seg only): Addition mIoU 54.33, mAcc 62.48 vs Concat 48.40 (-5.93), CFM 48.77, DDPM 49.51, SA-Gate 52.10, SWS 49.64. Stereo metrics NOT reported for this ablation. | KITTI 2015 test split (also where hyperparameters tuned) | n/a |
| SCG loss | boundary/structure consistency in both tasks | w/ vs w/o SCG, vKITTI2 (Tab. I,III): seg mIoU 84.18 vs 84.25 (-0.07, i.e. no gain), EPE 0.38 vs 0.39, PEP1 5.56 vs 5.59, PEP3 2.55 vs 2.55 (tie). KITTI (Tab. II,IV): mIoU 57.80 vs 54.33 (+3.47), mAcc 65.90 vs 62.48, EPE 0.55 vs 0.56, PEP1 10.02 vs 10.33, PEP3 1.62 vs 1.72. | single run, no seeds | free at inference |
| alpha in SCG (0.0-0.4) | weight strength | Fig. 7 (values only in plot, not tabulated): alpha=0.1 best. Tuned on KITTI 2015 (the test split; no val) | | |
| gamma = 0.9 iteration decay (Eq.11) | RAFT sequence-loss | not ablated | | |
| Disparity-feature branch E^D (ResNet-152 on D_n) | give seg geometric cue | not ablated (no "seg w/o disparity input" row) | | ResNet-152 is heavy |
| Remap R channels 64/256/512 | channel alignment | not ablated | | |
| GRU correlation-pyramid stereo head | accuracy | not ablated (inherited from RAFT-Stereo) | | |
| Joint training vs two single-task nets | the headline claim | not ablated | | |

### 4. Interactions & dependencies
- SCG gain is dataset dependent: effectively zero on 500-pair vKITTI2, +3.5 mIoU on 140-pair KITTI. Authors: weight tuning "should be approached cautiously ... limited data to avoid over-fitting" (p.10).
- Seg branch requires the predicted disparity (D_n) -> seg quality depends on stereo convergence; no stop-gradient discussion.
- Addition fusion only wins on KITTI seg; the ablation does not check whether alternatives hurt stereo.
- Concat/gated fusions (SA-Gate, CFM) did WORSE than plain addition: more expressive fusion did not help in this low-data regime (Tab. V).
- TiCoSS (next card) argues addition is "indiscriminate" and gains +5 mIoU with gates -> the two papers disagree on whether plain addition is adequate; both are on KITTI-2015 140-pair-scale data.

### 5. Losses (exact)
- GT volume: V3D_c(p)=delta(M^G(p),c) (Eq.5) in R^{HxWxC}; V^I = P(V3D) per-channel average pooling (kernel size NOT stated) (Eq.6); V^N(p)=exp(-(2V^I(p)-1)^2) (Eq.7); W(p)=max_c V^N_c(p) (Eq.8).
- L_scg = L_ss + L_sm (Eq.9).
- L_ss = -(1/N) sum_p sum_c [(1-a)+a W(p)] y_c(p) log yhat_c(p) (Eq.10), a=0.1.
- L_sm = sum_{i=1..n} [(1-a)+a W(p)] gamma^{N-i} || D^G - D_i ||_1 (Eq.11), a=0.1, gamma=0.9 (the exponent's "N" is presumably n, the iteration count; the per-pixel weight sits inside the sum over pixels, written loosely in the paper).
- MY DERIVATION (not in paper): for a pixel in a class-pure window V^I=1 for the true class, 0 for others, so every channel gives e^{-1}=0.368 -> W=0.368; at a boundary where a class fills half the window W->1. With a=0.1 the pixel weight ranges only from 0.9+0.1*0.368=0.937 to 1.0, i.e. <=6.7% relative re-weighting. This explains why SCG barely moves the stereo numbers (EPE 0.39->0.38), and means "structural consistency" is an extremely mild boundary re-weighting; its effect on seg KITTI (+3.5 mIoU) is therefore more likely variance/over-fit regularization than real boundary focus (my reading; no seed study).
- Loss uses GT semantic labels to weight the DISPARITY loss, so it needs dense GT seg at train time; zero inference cost.
- No explicit balancing weight between L_ss and L_sm (both weight 1, sum).

### 6. Training recipe
AdamW (eps 1e-8, wd 1e-5), lr 2e-4 initial (schedule/decay not stated), batch 1, crop 1000x320 (vKITTI2 and KITTI), max disp 192, 100K iters on vKITTI2 and 20K iters on KITTI 2015 (from scratch each? fine-tune from vKITTI2 not stated), "traditional data augmentation" (unspecified), 1x RTX 3090. No pre-training, no freezing, no curriculum stated. n GRU iters not stated.

### 7. Results
- Seg vKITTI2 (Tab. I): S3M-Net Acc/mAcc/mIoU/fwIoU 98.27/88.28/84.25/96.92; +SCG 98.32/88.24/84.18/96.98. Best prior (RoadFormer) 97.54/86.58/80.83/95.34; SegFormer 94.75/70.56/64.98. mIoU is per-image-averaged? (TiCoSS states per-image averaging for the shared protocol; S3M paper does not say.)
- Seg KITTI (Tab. II): S3M-Net 90.01/62.48/54.33/83.44; +SCG 90.66/65.90/57.80/84.53; RoadFormer 90.05/62.34/55.13/83.40 (S3M-Net w/o SCG is actually BELOW RoadFormer on mIoU 54.33 vs 55.13 and Acc, text claim "best on all except Pre" holds only for the SCG version).
- Stereo vKITTI2 (Tab. III) EPE/PEP>1/PEP>3: S3M-Net 0.39/5.59/2.55; +SCG 0.38/5.56/2.55; RAFT-Stereo 0.40/5.88/2.67; IGEV 0.47/7.15/3.09; PSMNet 0.68/10.31/3.77.
- Stereo KITTI 2015 (Tab. IV): S3M-Net 0.56/10.33/1.72; +SCG 0.55/10.02/1.62; RAFT-Stereo 0.60/10.78/1.96; GwcNet 0.68/14.21/2.01; IGEV 0.62/12.15/1.99.
- Speed: 0.66 fps at 1248x384 (p.11); params/FLOPs/VRAM not stated.
- DOES STEREO ACTUALLY IMPROVE? Marginally: 0.40->0.38 EPE vs the RAFT-Stereo entry (-5%), 0.60->0.55 on KITTI (-8%). Table III/IV baselines all trained by the authors under a 500/140-pair regime (RAFT-Stereo is a big gain over other baselines already), single seed, no separate-training same-recipe control. The large seg gains (vs 14 off-the-shelf nets trained on 500 pairs) are not a joint-vs-single comparison for the same architecture either (no "S3M-Net seg branch with no stereo" row). Claimed "2.50%-71.32% EPE improvement" is the range across all baselines; the 2.50% end is 0.40->0.39.

### 8. Negative results & limitations
- Authors: needs both seg and disparity annotations; 0.66 fps, too slow for vehicles (p.11).
- Fusion alternatives (concat, CFM, DDPM, SA-Gate, SWS) worse than addition (Tab. V).
- SCG does not help vKITTI2 seg (84.25 -> 84.18).
- My critique: (i) no joint-vs-separate or seg-off/stereo-off ablation, so the regularization / "tasks help each other" claim is unsupported; (ii) hyperparameters tuned on the KITTI test split (no val); (iii) tiny data (500/140 pairs), single seed, claimed deltas of 0.01 px EPE are within noise; (iv) vKITTI2 sampling of 700 pairs from 5 scenes x 10 variations likely leaks near-duplicate frames (unstated), inflating mIoU 84%; (v) Table II/IV numbers from per-image averaging, hard to compare with dataset-level mIoU; (vi) KITTI mIoU 54-58 on 19 classes indicates weak seg in real domain; (vii) no cross-domain test.

### 9. Relevance to OUR model
- NOT portable as architecture (RAFT-Stereo + ResNet-152 + 0.66 fps vs our frozen YOLO26m trunk, RTX 3050).
- SCG weight idea: a loss-only, zero-inference-cost per-pixel re-weighting from GT semantic boundaries. Cheap to try on our D2/E3 head (we have 14-class GT in VKITTI2): weight the EPE loss by a boundary/pooled-label map. But by my derivation the weighting range is only ~7%; to matter it would need larger alpha (S3M tuned alpha<=0.4 and saw degrade beyond 0.1-ish in Fig.7). Expected benefit small; our own G/H series already found edge-targeted changes do not move overall EPE. Risk: low; novelty: low (boundary-weighted loss is well known).
- Feeding predicted disparity back into seg: irrelevant (our seg decoder is frozen).
- Evidence for us: the only task-conflict-adjacent datum is channel-width mismatch and "plain addition beats gated/concat fusion" at low data. Our fusion head is also low-data (800 pairs), supporting keeping the fusion op simple (add/multiply gate) and small.
- The S3M splits (500/200 vKITTI2 random pairs) match the scale of ours (800/100/100 grouped); we should not compare absolute numbers (they mIoU ~84% likely leak; our grouping avoids it).

### 10. Key quotes/equations
- "joint learning ... introduces a form of regularization that has shown its superiority over uniform complexity penalization in reducing over-fitting" (p.6) - asserted, not tested.
- L_ss, L_sm: Eq.10-11 (p.5). W(p)=max_c exp(-(2V^I_c(p)-1)^2): Eq.7-8 (p.4-5).
- "S3M-Net achieves a processing speed of 0.66 fps ... 1248x384" (p.11).
- Errors in repo summary summaries/semantic_stereo/S3M-Net.md: (a) it adds a GitHub code URL not in the PDF; (b) "W high near/within semantic structures" is vague - it is flat 0.368 in class interiors and peaks at mixed windows; (c) summary does not mention that the seg branch needs a ResNet-152 over the predicted disparity or the lack of any joint-vs-separate ablation. Formulas otherwise match.
