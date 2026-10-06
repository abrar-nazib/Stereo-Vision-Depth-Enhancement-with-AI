# Errata in paper/reference_papers/summaries/

The per-paper summaries in `paper/reference_papers/summaries/` predate the verified block-cards and contain the factual errors below, found by reading each full PDF on 2026-10-06. When a summary and a card disagree, trust the card (`references/cards/`), which cites table/page refs.

## AIO-Stereo

Target: `paper/reference_papers/summaries/fusion/AIO-Stereo.md`

### Errors in existing summary (summaries/fusion/AIO-Stereo.md)
1. "Deployed model remains a lightweight CNN" / "lightweight CNN stereo backbone": paper's student is Selective-IGEV (heavy IGEV-class); no params/ms reported. Over-claim.
2. "No VFM inference cost" stated as fact; paper never states inference-time cost or that FA nets/VFMs are dropped. Inference-time VFM-free is plausible inference, not a stated finding.
3. Omits that the ablation shows non-monotonic EPE (DINO-only 0.66 equals full; DINO+SAM 0.68 worse) and that region claims are only qualitative (Fig. 4).
4. Omits that Tab. 2 KITTI/ETH3D numbers are finetuned benchmark results, and ETH3D avgerr 0.13 is not best (Selective-IGEV 0.12).
5. "Priority/Relevance: adapt to a single compact VFM" is speculation; the paper never tests a single-SAM-only or DA-only variant.

## BGNet

Target: `paper/reference_papers/summaries/lightweight/BGNet.md`

- Errors in repo summary (summaries/lightweight/BGNet.md): wrong code URL (3DCVdeveloper vs paper's YuhuaXu); says the cost volume is "reinterpreted directly, no conversion" and the grid is (x,y,g,c) with "intensity replaced by disparity", whereas the paper uses a 3x3x3 conv to produce a 4D grid (x,y,d,g) with g = 32 guidance levels and slices over disparity too (Eq. 1).

## CoEx

Target: `paper/reference_papers/summaries/lightweight/CoEx.md`

- Errors in repo summary (summaries/lightweight/CoEx.md): it states GCE at four scales {1/4..1/32}; paper Tab. VI has six GCE modules (4 down + 2 up, p.6). It writes the correlation as normalised by N_c; the paper gives no normalisation. It states GCE conv has "c output channels" (OK).

## DEFOM-Stereo

Target: `paper/reference_papers/summaries/fusion/DEFOM-Stereo.md`

- Errors found in summaries/fusion/DEFOM-Stereo.md: (1) the ablation table lists "+DI+SU" with 11.11M params and the full ViT-S as 15.5M; Tab. 2 gives DI+SU 15.51M and full ViT-S 18.51M (the "ViT-S 15.5M" model table is wrong); (2) "DI only hurts on some datasets" is right (SF, K15, Midd) but it helps K12 (-0.08) and ETH3D (-0.68);
  (3) "18 SU+DU iterations ... 32 eval" omits that SU is fixed at 8 iterations in both; (4) summary says the sum in the loss starts at i=n - this is the paper's own notational typo, not a method detail; (5) the summary's "Stage 1/2 ... Encoder_c ... " is fine; (6) "no explicit confidence output" is verified; (7) the claim "CCE more impactful than CFE" is only true on SF EPE by 0.01 and Midd-half; CFE is better on K12.

### 9. Relevance to OUR model
- Closest analogue to our fusion point is the **Combined Context Encoder**: prior features projected by conv and ADDED to the GRU context features, entering gates at every iteration. For us: inject frozen-YOLO/semantic-decoder features (1/8, 1/16) into the context/conditioning of the tile/plane refiner of A09 by conv-align + add. This is a cheap (conv + add), already close to what ClassResidual does implicitly if it uses decoder features; making it explicit as gate-conditioning (c_z,c_r,c_h style) is the portable part. Expected benefit: DEFOM saw ~-10% SF EPE from CCE and -2.25 bad2 on Midd-half zero-shot, but with a dense geometric prior; for a categorical prior expect far less. Risk: K15 got worse with CCE/CFE alone, so zero-shot effects are not guaranteed. Novelty: low (we do this); a combined feature-add into the matching feature (CFE) would break our frozen A09 and is not applicable.
- Scale Update / Scale Lookup / depth init: NOT transferable. They exist to resolve scale of a relative metric prior; a semantic map has no depth. The only transferable lesson: multiplicative-scale refinement with a global lookup helps when initialization is far off; our A09 init is already stereo-derived, so no.
- DI-alone-hurts finding is a caution: injecting a prior as initialization without a mechanism to correct it can regress (Midd +1.73); the analogue for us is that a semantic gate lacking reliability inputs may over-trust classes. DEFOM provides no confidence mechanism to borrow.
- Frozen-prior + new adapter head (new DPT) with the CNN branch kept (CNN removal = +35% EPE) matches our design: frozen trunk, small trainable head, stereo branch retained. Supports keeping A09 correlation features alongside semantic features rather than replacing them.
- Latency: DEFOM ViT-S/L (0.255/0.316 s at 960x540 on a 4090, 18.5M/47.3M trainable) is two orders too heavy for RTX 3050/Jetson; only the ideas, not the architecture, are relevant.

### 10. Key quotes/equations worth citing

## D-FUSE

Target: `paper/reference_papers/summaries/fusion/D-FUSE.md`

### Errors in existing summary (summaries/fusion/D-FUSE.md)
1. "SOTA zero-shot across all five real-world datasets": false per paper's own Tab. 1/2: KITTI-15 bad3 5.60 vs HVT-RAFT 5.20; KITTI-12 EPE/bad3 0.87/4.10 vs NerfStereo 0.84/3.6 (extra data); Middlebury Occ bad2 26.50 vs Mocha 24.16; Booster NonTrans EPE 1.52 vs 1.45.
2. Ordering-map direction "near 1 if neighbor is farther": paper does not state direction, and D is inverse-depth-like (p. 3), so sigma(D(u')-D(u)) is large when the neighbor is CLOSER. Summary's claim is probably reversed/unsupported. Also maps are soft (sigmoid), not binary.
3. "D-FUSE is the first paper to systematically analyze..." the paper does not claim "first".
4. "0.08 s overhead ... suggests it could work on edge devices": 0.32 -> 0.40 s is correct (p. 7) but hardware/resolution unstated, and total runtime is non-real-time with a frozen ViT; relevance claim is speculative. The summary also omits DA-V2's ViT cost and the staged training/ramped guidance, and that Baseline w/o mono feature beats baseline.
5. "10-point improvement for transparent bad2": true only vs weakest priors (~70 -> 59.83); vs best comparable without extra data (Selective-IGEV 66.85) it is ~7 points, vs NerfStereo 2.8.

## DTPnet

Target: `paper/reference_papers/summaries/lightweight/Distill-then-Prune.md`

### Errata vs existing summary (summaries/lightweight/Distill-then-Prune.md)
- Summary says latency is measured "with TensorRT" on Jetson AGX; the paper never states TensorRT was used for measurement (only that TensorRT motivates operator choices).
- Summary writes "M = 5 pruning steps"; paper uses E = 5 steps (M is max training steps) in Algorithm 1 / Sec. V-A.
- Summary's "36x fewer params than PSMNet" uses 0.26M (Tab. VII) while Tab. I/II/IV say 0.64M; the paper never reconciles this (pre-prune vs post-prune is the likely, but unstated, explanation).
- Summary repeats the paper's claim of lowest EPE without noting Tab. VII contradicts it (MSN3d 0.80, DeepPruner 0.97 lower).

## Fast-FoundationStereo

Target: `paper/reference_papers/summaries/fusion/Fast-FoundationStereo.md`

### Errors in existing summary (summaries/fusion/Fast-FoundationStereo.md)
1. Zero-shot table mislabels columns: header "Midd-H BP-2 / ETH3D BP-1 / KITTI-15 D1" but values are wrong columns. Correct (Tab.1): FoundationStereo Midd-H BP-2 1.10 (summary 2.49 = BP-1), ETH3D BP-1 0.50 (summary 0.30 = BP-2), KITTI-15 D1 2.80 (summary 2.95 = BP-3). Ours Midd-H BP-2 2.20 (summary 4.80 = BP-1), ETH3D BP-1 1.22 (summary 0.62 = BP-2). MonSter Midd-H BP-2 4.24 (summary 9.33), DEFOM Midd-H BP-2 3.76 (8.84) and ETH3D BP-1 2.16 (1.01 appears nowhere), RT-IGEV Midd-H BP-2 7.82 (12.75) and ETH3D BP-1 5.05 (1.63 is BP-3). D1 values 4.58/3.41/4.00/3.25 and runtimes are right. Under the corrected numbers Ours doubles FoundationStereo's error (not "modest").
2. "timm CNNs (~17M)" under backbone: 17.65M is the WHOLE model (Tab.6); the backbone size is not stated. Also Tab.5 says whole model 14.6M; summary quotes both without noting the inconsistency.
3. "Student: EdgeNeXt, MobileNetV2": paper only cites them as feature-extractor variants (p. 4); exact student not stated.
4. "Peak memory 0.63 GB ... fits edge GPUs": measured on a desktop 3090 at Midd-Q; paper's Jetson claim is untested.
5. Omits: 14 days on 128 A100 search cost, Delta tau = -0.04 s, pseudo-label thresholds (>60%, stride 10, sky zeroed), and that segmentation models (SAM2/ODISE) are used for sky masks.

## GGEV

Target: `paper/reference_papers/summaries/lightweight/GGEV.md`

### Errata vs existing summary (summaries/lightweight/GGEV.md)
- Summary says "Speed comparable to DEFOM-Stereo (ViT-S) while reducing ETH3D error by 81%": the paper says accuracy comparable to DEFOM ViT-S (255 ms) with 81% LESS INFERENCE TIME; the 81% is not an ETH3D error number.
- Summary says "~48 ms range on edge GPUs" and "inference time on 3090": the paper reports 47 ms at 1248x384 but does not state the GPU or any edge device.

## HITNet

Target: `paper/reference_papers/summaries/lightweight/HITNet_Tankovich_CVPR2021.md`

- Errors in repo summary (summaries/lightweight/HITNet_Tankovich_CVPR2021.md): says the init "evaluates a small set of disparity candidates"; paper evaluates ALL d in [0,D] per tile (exhaustively, never stored). Says "20 ms on a desktop GPU" correct. Summary's Pip-Stereo "93% D1 on DrivingStereo weather" cannot be verified from this PDF (not in the paper). "ETH3D bad 1.0 / 2.0 = 2.79 / 0.80" correct.

## LightStereo

Target: `paper/reference_papers/summaries/lightweight/LightStereo.md`

### Errata vs existing summary (summaries/lightweight/LightStereo.md)
- Summary lists expansion t=8 EPE 0.6779; paper Tab. IV(c) gives 0.6853.
- Summary says LightStereo-S 10.39 ms feature extraction = "58%" (correct: 10.39/17.83). Summary labels all times RTX 3090: correct only for the "*" starred rows; other competitor times in Tab. V are leaderboard-reported on unknown GPUs.

## LiteAnyStereo

Target: `paper/reference_papers/summaries/lightweight/LiteAnyStereo.md`

### Errata / notes vs existing summary (summaries/lightweight/LiteAnyStereo.md)
- Summary reproduces "21 ms at 4K on GTX 1080" (Fig. 1 caption); I flag it as inconsistent with 33 GMACs at 1242x375 and an unspecified Tab. 7 resolution.
- Summary contains a local RTX 3050 fp16 measurement (70 ms at 384x640, 57 ms at 480x768, tested 2026-05-02), which is NOT from the paper; useful for our budget but unverified here.

## MobileStereoNet

Target: `paper/reference_papers/summaries/lightweight/MobileStereoNet_Shamsafar_WACV2022.md`

- Errors in repo summary (summaries/lightweight/MobileStereoNet_Shamsafar_WACV2022.md): "soft-argmin followed by bilinear upsampling" (bilinear not stated in the paper); describes expansion factor t "chosen per-layer" (paper: t=3 for first convs and pre-hourglass, t=2 in hourglass); says it paved the way for BGNet (BGNet is CVPR 2021, earlier than this WACV 2022 paper and not cited by it). Numbers quoted (1.55/1.86/1.71 EPE, 1.14/2.23M, 0.80/1.77M/153 GMac) are correct.

## MonSter

Target: `paper/reference_papers/summaries/fusion/MonSter.md`

- Errors found in summaries/fusion/MonSter.md: (1) labelled CVPR MonSter but all numbers are MonSter++ v2; (2) "RT-MonSter ~20M params" is not stated anywhere in the PDF; (3) KITTI15 "D1-all 1.37" is All-pixels MonSter++ (Tab. II); (4) the paper does not give a confidence map: "confidence" is an input feature (F_S), the summary's phrase "confidence-based flow residual: low = high confidence" is the authors' intent, not an explicitly computed/normalized confidence; (5) (verified, not an error) "mutual refinement reduces iteration count ... 4 iterations beats 32" is verified (Tab. X: 0.42 vs 0.47).

### 9. Relevance to OUR model
Our D2/E3 = frozen YOLO trunk + frozen A09 stereo + frozen semantic decoder; trained: SemanticCostGate + ClassResidual.
- Most portable idea: use *feature-warp residual* F_S (L1 between left 1/4 feature and right feature warped by the current disparity, Eq. 2) as a free, label-free stereo-reliability channel and feed it (with the cost lookup and disparity) to our class-conditioned gate/residual. Insertion: inside SemanticCostGate and ClassResidual inputs, at the A09 resolution. Cost: one gather+abs per candidate/iteration, negligible. Benefit: lets the semantic gate learn "trust semantics when stereo is inconsistent" rather than gating by class alone; analogous to the SGA condition. Risk: the shared YOLO features are optimized for segmentation, so their warp residual may be less discriminative than IGEV-type matching features; must be measured. Novelty: semantic class gate + feature-residual reliability has not been done in this series (E/F/G used semantics or RGB edges, not residual-conditioned arbitration).
- Bidirectional (mono<->stereo) refinement and per-pixel shift refinement do NOT transfer: they need a metric-free but continuous depth prior to align; a categorical prior has no scale or shift to recover. The nearest categorical analogue of "per-pixel shift refinement" is our ClassResidual (class-conditioned additive residual), which already exists; SGA-style hidden-state-of-the-prior GRU would be a new, heavier design (2x GRUs, +8.2M params in MonSter++ scale) and is not justified for a 4GB real-time target.
- Equal-parameter conv-concat control: MonSter++'s Mono+Conv vs MGR (-6.5% EPE) and Conv vs SGA (-8.1% bad-1) mirrors our no-semantics equal-capacity control logic; it supports using a GRU/gate with explicit reliability inputs rather than plain concat if we ever try a heavier head. Do not expect their magnitude (they have a dense metric-ish prior, we have 14 classes).
- RT-MonSter++ tricks portable to Jetson latency: cascaded local cost volumes centered on previous-stage disparity (Eqs. 10-12) and ~4 total GRU updates; our tile/plane lineage (HITNet) already searches locally, so little new. Note RT mono fusion is only at the finest scale and was never ablated.
- Frozen shared encoder with a conv "transfer" adapter (feature sharing, -5.13% EPE) is the exact structure we already use (frozen YOLO layers 0-6 shared by stereo and semantics); the paper is mild evidence that sharing a prior-pretrained trunk helps in-domain; zero-shot attribution is not ablated.
- Caution on claims: gains are mono-prior-specific and in-domain Scene Flow for the ablation; do not borrow the zero-shot headline as support for a semantic prior.

## Pip-Stereo

Target: `paper/reference_papers/summaries/lightweight/Pip-Stereo.md`

### Errata vs existing summary (summaries/lightweight/Pip-Stereo.md)
- Summary's "11.5x faster than DEFOM-Stereo" matches the Tab. 1 latencies (5.05/0.44) but not the paper's text claim of 14x (and 22x vs MonSter, 41x vs FoundationStereo-L, vs 17x and 32x from the table). The paper is internally inconsistent; use the table.
- Summary says teacher is "frozen"; only the DA-V2-L mono model is frozen (Fig. 2); the teacher stereo branch carries gradient-updated parts and the text says teacher and student "are co-updated in tandem".

## RTS2Net

Target: `paper/reference_papers/summaries/semantic_stereo/RTS2Net.md`

- Errors in the local summary (summaries/semantic_stereo/RTS2Net.md): it says "19 classes" - the paper never states a class count; it omits that the synergy module is applied to the COST VOLUME (hybrid volume, pre-soft-argmin) and compresses semantic channels; omits the Tab. III ablation (semantic-only vs refinement gains), the loss weights, the Tab. II/IV baseline mix, and the 0.12-vs-0.08 inconsistency. Tab. II/IV numbers it lists are correct.

## SegStereo

Target: `paper/reference_papers/summaries/semantic_stereo/SegStereo.md`

- The existing summary paper/reference_papers/summaries/semantic_stereo/SegStereo.md: (1) says semantic branch is "trained jointly" - it is frozen (p. 8, 12); (2) writes L_sup = L_reg + lambda_sem L_softmax, omitting the smoothness term in Eq. 5; (3) is otherwise consistent about reported metrics (no mIoU, params, FLOPs).

## SGNet

Target: `paper/reference_papers/summaries/semantic_stereo/SGNet.md`

- The existing summary paper/reference_papers/summaries/semantic_stereo/SGNet.md has errors: (1) it says the paper "does not report EPE ... latency" - it does (EPE in Tab. 1-3; runtime 0.671/0.674 s in Tab. 3); (2) it writes the residual as R_cat(F_sem, d_init), but the residual module takes the semantic PROBABILITY map multiplied with disp3, not semantic features; (3) it omits that disp3 is unsupervised and the loss is on disp1/2/4.

## StereoAnywhere

Target: `paper/reference_papers/summaries/fusion/StereoAnywhere.md`

- Errors found in summaries/fusion/StereoAnywhere.md: (1) states normals are used because they are "scale-invariant ... naturally compatible with absolute disparity" - paper's reasons are non-ambiguity/consistency between L and R mono maps and scale-invariance via lambda, OK, but the summary omits the 3D hourglass aggregation phi_A + CoEx excitation, which is more than "spatial gradients + dot products, no heavy processing" (efficiency claim is wrong: the mono branch includes a 3D conv regularizer on an H/4 x W/4 x W/4 volume); (2) omits the volume augmentations, the paper's key training mechanism (and the perfect-mono augmentation); (3) says "global scale/shift (single solve) is cheap ... reducing iterations" - the paper uses 12 train / 32 test iterations like RAFT; no reduced-iteration claim; (4) "frozen RAFT-Stereo feature encoder" correct, but context encoder is trained; (5) "MonoTrap ... painted checkerboards, murals" is not in the paper text (it says planar patterns creating illusions such as holes in walls/floors and simulated transparent surfaces); (6) says "Ranks 1st on Booster leaderboard (fine-tuned)" - paper says first at quarter resolution only; (7) says confidence C^ from "entropy of the mono correlation volume" - right, but omits that it comes from a separate learned head phi_C and softLRC occlusion masking; (8) "volume truncation virtually free" - it is a handful of elementwise ops on H/4xW/4xW/4, but the fuzzy mask needs both D^ and M^ (3D conv outputs), so it is not free.

### 9. Relevance to OUR model
This is the paper with the most explicit confidence-gating machinery, so it is the closest analogue of SemanticCostGate. Our fusion head is trained on a frozen A09 with VKITTI in-domain data; their key lesson is that a gate learns to ignore the extra branch unless trained with corrupted stereo hypotheses.
- Portable 1: entropy-based confidence of the stereo cost curve: C = 1 + sum p log2 p / log2(K) over our K disparity candidates/tile hypotheses (p = softmax of candidate scores). Free (prior-agnostic), differentiable, in [0,1], computed on the A09 candidate costs. Insertion: as an input channel (and/or multiplicative modulator) to SemanticCostGate, so semantics override only where the stereo curve is multimodal. Expected benefit: sharper where-to-trust-semantics than class alone; likely helps in occlusion/textureless (where SA gains concentrate: Occ columns). Cost: negligible. Risk: A09 cost curves from tiles/planes may already be peaky; calibrate against actual error (SA trains it with BCE vs soft correctness C_gt, which we can copy: BCE(C, softLRC-style |d-d_gt|) as an auxiliary loss on the gate input, a new loss for our D/E series).
- Portable 2: softLRC occlusion mask (needs right-view disparity; we have stereo pair, but A09 may not output right disparity; cost = a second pass or symmetric head). Moderate cost; benefit mostly for occluded regions where semantics matter.
- Portable 3 (analogue of depth-bin masking): SA restricts matching to same-depth-bin pixels via M_L^n(i,j) M_R^n(i,k). Categorical equivalent: class-consistency mask on the cost volume/candidates: down-weight candidates whose right-view pixel has a different semantic class than the left pixel (A09's shared trunk gives class logits for both views for free). Directly implementable as a soft multiplicative gate on candidates, restricted to non-ambiguous (large) classes. This is not "already done" in our D/E series (D1 gates by semantics but not by L-R class consistency, as far as the AGENTS.md describes). Risk: semantic errors near boundaries kill the true match; use soft gating and learned temperature.
- Portable 4: volume augmentations (rolling a peak to a wrong bin, noise, zeroing) applied to the stereo candidate volume during gate training so that the semantic branch must supply the right answer; plus an "oracle prior" substitution (their perfect monocular = GT normalized; ours = GT label map / teacher labels with some probability). This directly attacks the "gate ignores semantics" failure and could amplify our existing -0.24 px in-domain and KITTI semantic gains; cheap (training only). Risk: synthetic corruption may not match KITTI failure modes; keep the equal-capacity no-semantics control with identical augmentation to attribute gains.
- NOT transferable: normals-from-depth volume, differentiable scale/shift (needs a metric-free continuous prior), truncation by mono-depth ordering (a categorical "mirror/glass" class could drive a truncation analogue, but our 14 VKITTI classes have no mirrors; irrelevant for driving), dual full cost volumes with 3D hourglass (4 GB budget). Runtime of the paper (0.24 s at 512x512 on A100) is not real-time.
- Evidence caveat: all SA gains are with a dense continuous VFM prior; none of the blocks was ablated individually, so the attribution of gains to confidence vs volume vs context vs augmentation is unknown. Do not cite as evidence for categorical priors.


## DispSegNet

Target: `paper/reference_papers/summaries/semantic_stereo/DispSegNet.md`

- Local summary errors: it quotes "KITTI all-pixel EPE 2.17 to 1.89 px and D1 10.53 to 10.03%" - these numbers do NOT appear in the PDF; the paper reports no EPE at all, and the corresponding ablation is Tab. IV in % error (e.g. 8.14 -> 6.32). It says "supervised disparity regression when ground truth exists" - the paper is fully unsupervised. It calls the semantic part an "encoder-decoder" - it is a PSP head over ResNet-50 features.

## S3M-Net

Target: `paper/reference_papers/summaries/semantic_stereo/S3M-Net.md`

- Errors in repo summary summaries/semantic_stereo/S3M-Net.md: (a) it adds a GitHub code URL not in the PDF; (b) "W high near/within semantic structures" is vague - it is flat 0.368 in class interiors and peaks at mixed windows; (c) summary does not mention that the seg branch needs a ResNet-152 over the predicted disparity or the lack of any joint-vs-separate ablation. Formulas otherwise match.

## SemStereo

Target: `paper/reference_papers/summaries/semantic_stereo/SemStereo.md`

- Repo summary errors: (a) d_final = d''_init + R (typo copied from the text; shapes require d'_init); (b) summary says the SSR gate "per-class confidence" without noting P^l is the left view only; (c) it omits the key control that SGC-Net* gets ~80% of the stereo gain without labels; (d) summary table row "WHU, SemStereo without explicit semantic labels" fine.

## SSPCV-Net

Target: `paper/reference_papers/summaries/semantic_stereo/SSPCV-Net.md`

- Correction to the existing summary: its "headline KITTI-2015 all-pixel 0.87 px EPE, 3.1 % D1-all" are Scene Flow numbers (Tab. 2); KITTI-15 D1-all is 2.11 %. The summary also says it reports no mIoU (it reports IoU 56.43 / 82.21 on KITTI-15) and says the volumes are "concatenated" (they are fused recursively with FFM and hourglasses).

## TiCoSS

Target: `paper/reference_papers/summaries/semantic_stereo/TiCoSS.md`

- Repo summary errors: wrong L_CT formula; describes L_DIA as "disparity-informed semantic alignment" (it is a LR-consistency-weighted CE) and L_DSCC as "disparity-semantic cross-consistency" (it is KL among the deep-supervision seg outputs); says "iterative stereo branch with shared encoder" while only the first 3 contextual layers share weights; "official checkpoint" claim is not in the PDF.

## USAM-Net

Target: `paper/reference_papers/summaries/semantic_stereo/USAM-Net.md`

- The repo summary "USAM-Net.md" misdescribes the method (claims attention A=sigma(f_seg) modulates stereo features; in fact SAM masks are an input channel group and the attention is plain self-attention at the bottleneck), omits the negative KITTI results and the ablation, and names the wrong first author.

## No summary errors recorded

- ESMStereo
- FoundationStereo
- JointTreeEncoders
- LiteAnyStereoV2
- PromptStereo
- S3Net
- SDBF-Net
- SENSE
- TwInS
- ViTAS
- VPEngine
