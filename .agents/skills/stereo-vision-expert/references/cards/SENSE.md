<!-- Verified block-card, read from the full PDF on 2026-10-06. Numbers carry (Tab./Fig./p.) refs. Section 9 reflects the project state as of that date (A09 + D2/E3 fusion head). -->
<!-- PDF: see the "PDF:" or "pdf path" line in section 0. -->

# SENSE card

Source read: full 17-page PDF (arXiv 2604.15946v1, 17 Apr 2026) including the appendix (Sec. 10), text extracted and Fig. 2/Tab. 3 page rendered. No repo summary existed for this paper. Companion paper PhraseStereo skimmed in the last section.

### 0. Meta
- Title: SENSE: Stereo OpEN Vocabulary SEmantic Segmentation.
- Authors: Thomas Campagnolo, Ezio Malis, Philippe Martinet (Inria Univ. Cote d'Azur), Gaetan Bahl (NXP). arXiv preprint Apr 2026.
- PDF: paper/reference_papers/semantic_stereo/SENSE_Campagnolo_arXiv2026.pdf
- Code: not stated.
- Domain: open-vocabulary (text-prompted) segmentation for ITS / driving, using stereo.
- Datasets: trained on PhraseStereo (synthetic right views, see last section); evaluated on PhraseStereo test (referring expression: mIoU, IoU_FG, AP), and zero-shot on Cityscapes and KITTI 2015 (mIoU over classes), with class names or Cityscapes descriptions as prompts.

### 1. Problem & failure modes targeted
Open-vocabulary segmentation is monocular and spatially imprecise, "especially under occlusions and near object boundaries" (Abstract). Stereo supplies geometry for boundaries/occlusions. CLIP's fixed input size (224 -> 352/512) forces a sliding window on Cityscapes/KITTI. Direction of interaction is DISPARITY -> SEGMENTATION only; no stereo-matching improvement is attempted.

### 2. Pipeline by stage
- 2a Feature extraction: frozen CLIP ViT-B/16 image encoder, shared weights, applied separately to left and right image (86.2 M params total reported for the image encoder, Sec. 10.5). Input crop 352x352 (SENSE-352) or 512x512 (SENSE-512). Activations taken from transformer blocks E = {3, 7, 9} including CLS token (Sec. 3.1; Fig. 2 also shows E1, E12 labels). ViT patch 16: 352 -> 22x22 tokens. Frozen.
- 2b Semantic / prior branch: CLIP text transformer (frozen, 37.8 M params) -> text embedding of the prompt, projected to P = 64 channels and applied to the decoder via FiLM. Binary mask per prompt (one prompt at a time). Classes: open vocabulary.
- 2c Cost volume: n/a inside SENSE. The disparity comes from a frozen external stereo network: Selective-IGEV with SceneFlow weights (max disparity 192) by default, plug-and-play (HITNet, MobileStereoNet also tested; Tab. 4).
- 2d Aggregation: n/a.
- 2e Disparity computation: external; D in R^{352x352}, normalised D_norm = D/192 (Sec. 3.4).
- 2f Refinement: the decoder: 3 transformer blocks D1-D3 (token width P = 64), each FiLM-conditioned on the text; progressive refinement; segmentation = linear projection of tokens. Then SDAF at the end.
- 2g Upsampling: inside SDAF: 16x16 transposed conv from 22x22 to full 352x352 (or 512), then 1x1 conv to 1 channel.
- 2h FUSION POINTS:
  1. SIEF (Stereo Intermediate-level Embedding Fusion), x3 (SIEF 1-3 on blocks 3, 7, 9): left and right ViT activations concatenated on channels (Eq. 1), then 1x1 conv (C -> C/sf, sf = 16) -> ReLU -> 1x1 conv (-> C x sf as written) -> sigmoid = W_LR (Eq. 2); split into W_L, W_R, softmax across the two (Eq. 3); F_LR = W_L*F_left + W_R*F_right (Eq. 4), then linear projection to P = 64. Direction: right->left view (stereo feature fusion); stage: feature; per-location view weighting "improves robustness to occlusions". Resolution: ViT token grid (22x22 for 352 input). Note: this is a learned left/right mix, not a correlation: no explicit matching happens, so geometry is only implicit.
  2. SDAF (Semantic Disparity Attention Fusion), last stage of the decoder: D_norm bicubic-downsampled to the decoder grid (22x22), passed through conv layers + nonlinearity to attention-like weights, multiplied (Hadamard) with the decoder feature map [h',w',P]; then 16x16 ConvTranspose2D to full res and 1x1 conv to 1 channel (Fig. 4). The final "two-layer" default has branches at 22x22 and 352x352 (disparity map used at both scales; the 3-scale variant adds 88x88). Direction: disparity -> semantics (sem features modulated by geometry), multiply-gate; stage: final decoder. The 3x3 conv layer widths are given as C in Fig. 4, C not specified.
  3. No loss-coupling. No stereo supervision.

### 3. Block -> problem -> evidence table
All on PhraseStereo test, referring-expression segmentation, SENSE-352 (Tab. 3, p. 10). mIoU / IoU_FG / AP.
| block | problem it solves | evidence | context/conditions | cost |
|---|---|---|---|---|
| CLIPSeg (PC+) baseline (mono, one CLIP stream, no SIEF/SDAF) | reference | 43.7 / 55.1 / 76.7 (reproduced) | | |
| SIEF + SDAF default (2-layer) | occlusion/boundary precision | 47.2 / 57.0 / 78.8 (+3.5 mIoU, +1.9 IoU_FG, +2.1 AP absolute) | ViT-B/16 frozen; PhraseStereo test | decoder+SIEF 2.01 M params; + SDAF 3.12 M total; 2nd CLIP pass doubles image-encoder FLOPs (~92 GFLOPs per pass) |
| SIEF only (SDAF removed, replaced by simple upsampling) | | 45.7 / 56.3 / 77.6 (so SDAF adds +1.5 mIoU, +0.7 IoU_FG, +1.2 AP) | | SDAF ~1.1 M params |
| SDAF 3-layer | | 46.7 / 56.6 / 78.0 ("propagates noise from disparity across scales") | | |
| SDAF 1-layer at 22x22 | | 46.7 / 56.4 / 77.4 | | |
| SDAF 1-layer at 352x352 | | 45.5 / 55.9 / 77.2 (worse than no SDAF: 45.7) | | |
| SIEF scale factor sf = 2 (with SDAF removed) | | 45.7 / 56.2 / 77.8 (no change from sf = 16, "robust") | | |
| SIEF replaced by plain concat + conv (SDAF removed) | | 45.8 / 56.1 / 77.6 vs SIEF-only 45.7 / 56.3 / 77.6: NO measurable difference; the text claims "attention-based fusion provides incremental but consistent gains" which Tab. 3 does not show | | |
| Stereo-matching weights: Middlebury instead of SceneFlow | | 47.3 / 56.8 / 78.1 vs 47.2 / 57.0 / 78.8 (robust) | | |
| Different stereo model at inference without retraining (Tab. 4) | runtime | SENSE-352: Selective-IGEV 194.74 ms, 47.2/57.0/78.8; HITNet 104.09 ms, 47.1/56.8/78.3; MobileStereoNet 76.91 ms, 47.1/57.1/78.9. SENSE-512: 284.54 ms 46.5/56.3/77.7; HITNet 136.01 ms 46.4/56.4/77.8; MobileStereoNet 91.68 ms 46.3/56.3/77.8 | RTX 3090 Ti, one text query, one stereo pair | |
| Backbone CLIP ResNet-50 instead of ViT-B/16 | | CLIPSeg-R50 40.5 / 53.1 / 75.0; SENSE R50 full 44.1 / 54.8 / 76.7 (+3.6 mIoU); SIEF only 42.6 / 54.0 / 75.9 | | |
| Sliding window + CRF + cosine blending (Cityscapes/KITTI) | CLIP resolution limit | not ablated; applied only to SENSE, not to baselines (Sec. 10.3) | | |
| Negative-sample rate (20%), FiLM conditioning, decoder depth (3), P = 64, layer selection E = {3,7,9} | | not ablated | | |
| Frozen vs fine-tuned CLIP/stereo | | not tested ("deliberate", future work, Sec. 10.1) | | |

### 4. Interactions & dependencies
- The disparity cue is only useful through the second (full-resolution) SDAF branch together with a low-res branch; one scale alone is no better than none (1 layer at 352x352: 45.5 < 45.7).
- SDAF is used at inference with ANY stereo model without retraining, which says the module reads coarse geometry (depth discontinuities), not model-specific disparity quirks, or that the signal is weak (all three stereo models give within 0.1-0.5 mIoU).
- Training disparity comes from Selective-IGEV run on PhraseStereo's generated right images, whose geometry is synthesised from monocular Depth Anything V2 depth (see PhraseStereo): the "stereo" cue may thus simply be a monocular-depth prior recycled through GenStereo+IGEV.
- SENSE is binary per prompt: multi-class use needs one forward pass per class (runtime scales with #classes; Tab. 2 is for a single query) plus CRF.

### 5. Losses
Not stated anywhere in the paper (not in main text or appendix): no loss function, weights, or negative-sample loss formulation beyond "20% negative samples". Presumably per-pixel BCE as in CLIPSeg, but this is not stated.

### 6. Training recipe
AdamW, lr 0.001, cosine schedule, batch 64 stereo pairs; epochs/iterations: not stated; hardware: Ryzen 9 5950X, RTX 3090 Ti, 64 GB RAM. CLIP image+text and Selective-IGEV frozen; trained: SIEF x3, decoder (3 blocks), FiLM, projections, SDAF (2.01 M decoder+SIEF; 3.12 M with SDAF). 20% negative samples; crops 352x352 or 512x512 in training; sliding-window + CRF at inference for Cityscapes/KITTI. Augmentation: not stated.

### 7. Results
Tab. 1a (PhraseStereo, referring expression), mIoU / IoU_FG / AP: CLIPSeg (PC+) 43.7 / 55.1 / 76.7; CLIPSeg (PC) 46.1 / 56.2 / 78.2 (numbers from CLIPSeg paper, trained on PhraseCut mono); MDETR 53.7 / - / -; HulaNet 41.3 / 50.8 / -; SENSE-352 47.2 / 57.0 / 78.8; SENSE-512 46.5 / 56.3 / 77.7. So SENSE loses to MDETR on mIoU by 6.5 points and gains 0.6 AP over CLIPSeg (PC).
Tab. 1b (zero-shot semantic seg, mIoU), Cityscapes / KITTI15: SAM+CLIP 38.6 / 35.2; OpenWorldSAM (their reproduction) 23.1 / 21.1; CLIPSeg (PC+) 34.9 / 32.1; OpenSeg 40.2 / 37.8; MaskCLIP 25.0; SCLIP 32.2; SENSE-352 40.7 / 37.9; SENSE-512 41.6 / (not evaluated, image aspect). Claims: abstract "+3.5% mIoU Cityscapes and +18% on KITTI compared to the baseline work": these correspond to SENSE-512 vs OpenSeg (41.6/40.2 = +3.5%) and SENSE-352 vs CLIPSeg (37.9/32.1 = +18%): two different baselines mixed in one sentence. Versus OpenSeg on KITTI the gain is 0.1 mIoU, i.e. a tie. Abstract's "+2.9% AP over the baseline" does not match Tab. 1a (+2.1 absolute, +2.7% relative vs CLIPSeg PC+).
Runtime (Tab. 2, RTX 3090 Ti, one text query): OpenSeg 123.35 ms; OpenWorldSAM 343.25; CLIPSeg 218.96; MDETR 256.87; SENSE-352 194.74 ms (17.12 ms w/o stereo matching); SENSE-512 284.54 (43.56). The stereo matcher (Selective-IGEV) is >90% of runtime. FLOPs: SENSE-512 ~190 G (2 x 92 CLIP + 4 text + 3 decoder) vs OpenSeg ~170 G. Model size: 86.2 M (CLIP image) + 37.8 M (text) + 3.12 M trainable. Qualitative: sharper boundaries, occluded person recovered (Fig. 9).

### 8. Negative results & limitations
- Authors: prompt phrasing sensitivity (CLIP text), negative instructions fail, user-defined prompts only, runtime (disparity module dominates), specialised closed-set models still better ("generalization penalty").
- My concerns: (1) Training stereo is synthetic: right views are GenStereo diffusion outputs conditioned on monocular Depth Anything V2 disparity, so "stereo" training pairs carry no independent geometry; the real-stereo evaluation on Cityscapes/KITTI works only through a frozen stereo network. (2) The headline gain is small and single-run: +3.5 mIoU over a CLIPSeg that has only one CLIP stream; the right-stream/SDAF contribution on PhraseStereo is +3.5 mIoU total, ~+1.5 from SDAF and +2.0 from SIEF with the SIEF attention design showing zero gain over concat. (3) Zero-shot comparisons against OpenSeg/SAM+CLIP use numbers copied from a survey and other papers with different protocols; SENSE uses sliding window + CRF + cosine blending, baselines do not. (4) KITTI "SENSE-512 not evaluated". (5) No loss, epochs or seeds. (6) Reference [2] for KITTI 2015 is the "Augmented reality meets computer vision" (Alhaija) paper, not the Menze&Geiger KITTI 2015 paper. (7) Pairing frozen CLIP with frozen IGEV shows that frozen foundation components can be used, but nothing here shows the geometry actually carries semantic information beyond what Depth Anything gives.

### 9. Relevance to OUR model
- Direction: DISPARITY->SEMANTICS only, via a frozen external stereo model and a tiny (1.1 M param) multiplicative attention gate at the end of a frozen-CLIP decoder. This is the mirror image of our SEM->DISPARITY path (SemanticCostGate, ClassResidual). It does not threaten the claim "semantic cost gating improves frozen stereo"; it is the closest published precedent for "everything frozen, train only a small fusion head" in stereo+semantics, so cite it as that precedent.
- Portable: (i) the SDAF recipe (D/maxdisp normalisation, bicubic to decoder grid, conv -> sigmoid -> Hadamard on the decoder features, 2 scales: coarse + full-res) could be added as an OPTIONAL disparity->semantic head on our frozen 14-class decoder at 1/8 and full-res; cost ~1 M params; expected benefit: boundary/occlusion sharpening of semantics (their +1.5 mIoU on an open-vocab task, which is weak evidence). Risk: our semantic decoder is frozen; this requires a trainable adapter and re-validating mIoU. Priority: low-medium (not core to the semantic-stereo novelty).
  (ii) Plug-and-play stereo substitution test (swap MobileStereoNet/HITNet/our A09 at inference without retraining and report metrics) is a good robustness experiment for our fusion head; SENSE shows <=0.5 mIoU drift when swapping.
  (iii) Learned per-location left/right weighting (SIEF) is an occlusion-aware alternative to concatenation; but their own ablation shows no gain over concat, so skip.
  (iv) The sigmoid-gate form W_LR: relevant only as a design alternative to our gate; their gate is soft sigmoid + softmax, ours is semantic-conditioned over disparity candidates.
- Do not copy: CLIP/ViT trunk (86 M x 2 views, ~92 GFLOPs per pass; not real-time on RTX 3050), one-prompt-at-a-time inference.
- Novelty implication: SENSE is "first stereo open-vocabulary segmentation" and says nothing about improving stereo. It is not a competitor for our metrics (no disparity EPE reported anywhere). Comparing against it is impossible on our metrics; mention in related work as the semantic-output-side use of frozen stereo.

### 10. Key quotes/equations worth citing
- "The choice to keep components such as CLIP and Selective-IGEV frozen was deliberate, allowing us to isolate the contribution of the stereo formulation. Jointly finetuning or replacing these components represents promising future work." (Sec. 10.1, p. 9).
- Eqs. 1-4 (SIEF): F_concat = Concat(F_left-E, F_right-E); W_LR = sigma(Conv1x1(ReLU(Conv1x1(F_concat)))); [W_L, W_R] = Softmax(W_LR); F_LR = W_L * F_left + W_R * F_right.
- "the disparity module dominates runtime and ... without it, SENSE becomes substantially faster than all compared methods" (Sec. 6, p. 7; 17.12 ms of 194.74 ms).
- Params: decoder + SIEF ~2.01 M; with SDAF ~3.12 M (Sec. 10.5).

---

## PhraseStereo (dataset companion)
Source: PhraseStereo_Campagnolo_ICCVW2025.pdf (arXiv 2510.00818, Oct 2025; 5 pages, skimmed in full). Dataset "will be released online upon acceptance"; availability unverified.
- Content: 77,262 stereo pairs and 345,486 phrase-region annotations; splits follow PhraseCut: 71,746 train pairs (310,816 phrases), 2,971 val (20,316), 2,545 test (14,354) (Sec. 3). Left images and masks are PhraseCut (Visual Genome images, indoor + outdoor, general scenes, NOT driving).
- Right views are GENERATED, not captured: images resized to 512x512, disparity from Depth Anything V2 monocular depth, then GenStereo (diffusion with disparity-aware coordinate embeddings and adaptive fusion with the warped image) synthesises I_R, resized back to the original resolution. Masks are transferred from the left view unchanged. No real stereo GT, no disparity GT.
- Quality: scale factor beta (baseline proxy) in {0.05, 0.15, 0.25}: SSIM 0.679 / 0.601 / 0.571, LPIPS 0.227 / 0.352 / 0.412 (Tab. 2); beta = 0.15 chosen as compromise. Larger baseline -> hallucinations, distorted edges, invented structures; authors cannot measure hallucination directly (no real-stereo GT) and call it "limited". Proposed future fix: run a stereo matcher on generated pairs and add a disparity consistency loss.
- Credibility consequence for SENSE: the mask transferred from the left image is only valid for the left view; stereo benefit is partially an artifact of a monocular-depth-derived pair; evaluation of "stereo helps" on this dataset cannot separate real stereo geometry from monocular-depth priors. For us: not useful as a stereo benchmark (no disparity GT, synthetic right view, non-driving), not a dataset to compare against.
