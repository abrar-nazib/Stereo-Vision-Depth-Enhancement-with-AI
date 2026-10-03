---
name: ml-arch-diagram
description: Draw ML model architecture diagrams in the visual grammar of published paper figures and the StereoLite lab conventions — tensors as bar-pyramids and cuboids, ops as rounded rects, role colors with a bottom legend, red-dot supervision, real image thumbnails. Use when asked to create, redraw, or review an architecture diagram (excalidraw, matplotlib, or PNG) for a stereo/depth or any ML model, or when a diagram is criticized as "not understandable" or "not like a paper figure".
---

# ML Architecture Diagrams

Produce figures that read like published paper architecture figures
(RAFT-Stereo, HITNet, SGNet, IGEV) and the lab's own StereoLite figures.

**Read [references/grammar.md](references/grammar.md) before drawing anything** —
it holds the five laws, tensor/op vocabulary, color palette, and anti-pattern list.
Study the reference images in `references/` and `references/paper_figs/` with a
vision-capable read before the first draft; do not draw from memory of this text alone.

## Hard rules (violations caused every rejected diagram so far)

0. Before finishing, walk the **Common mistakes and remedies** table at the end of
   grammar.md — every entry records a real rejection.

1. Canvas holds tensors and operations only. No why/result/scoreboard/meta boxes.
   Use a plain paper-facing model title; put benchmark metrics and implementation
   provenance in the caption or accompanying text, not the architecture flow.
2. Tensors are pictures: bar-pyramids, cuboids, slot-glyph state boxes, rendered
   image thumbnails. A named rectangle is not a tensor.
3. Ops are quiet shapes: compact rounded stages, circled `⊗ ⊕ σ L Δ`, and
   trapezoids for upsampling. Show repeat counts or parameters only when they
   explain a visible architectural distinction.
4. Color = role per the palette in grammar.md; bottom legend decodes every color.
5. When the figure includes training, supervision = crimson dot `ℒ̂ᵢ` per
   supervised tensor; omit loss markers from inference-only overviews.
6. ≤ 12 blocks; split overview + module detail instead of cramming.
7. Every arrow lands on a shape; every label has 15% width headroom (app fonts run
   wider than the headless renderer — always verify in the excalidraw app too).
   For this lab's paper figures, prefer exact horizontal/vertical arrow segments
   and 90° bends; recheck coordinates after app edits that can introduce tiny
   diagonal segments. Route long branches in dedicated lanes.
8. Translate implementation names and ablation IDs into operations a new reader
   can recognize. For example, `A09 agg + veto` becomes stereo cost aggregation;
   `E3 gate` becomes semantic candidate weighting. State frozen/trainable status
   once in a legend or caption, unless the distinction is the figure's subject.
9. Draw the *model-specific transformation*, not a row of equally sized operation
   boxes: show feature/cost/class tensors, where semantic evidence enters matching,
   the tile state and scale changes, and the two distinct outputs. A label on a
   rectangle does not replace a tensor or explain how one branch affects another.

## Workflow

1. Inventory the model from code: tensors (name, shape, scale), ops, params,
   supervision points, frozen/trainable.
2. Read grammar.md + view 2–3 reference images closest to the target architecture.
3. Build in excalidraw (pair with the `excalidraw-diagram` skill for the
   render-inspect-fix loop). Real thumbnails: `type: "image"` elements + `files`
   map with dataURLs — the renderer supports them natively.
4. Render, inspect against grammar.md's anti-pattern list, fix, repeat. Only stop
   when a render passes and the app view passes.
5. For matplotlib output instead, port the palette/glyphs from grammar.md; the
   stereolite repo's `diagram-drawer/helpers/diag_helpers.py` is the reference impl.

## Reference gallery (all under references/)

| File | Takeaway |
|---|---|
| `stereolite_arch_mpl.png` | The lab's canonical overview: header badge, bar encoder, yellow CV, mint refine ×N, trapezoid upsamplers, stage brackets, legend |
| `stereolite_architecture.png` | Same model in excalidraw: real thumbnails, ⊕/⊗ circled ops, red supervised dots |
| `stereolite_gev_module.png` | Module-detail figure: zoom into one overview box |
| `sgnet_fig1.png` | Published semantic-gated stereo: lettered taps A–F, green confidence module, ⊗ into disparity path |
| `hitnet_fig2.png` | Two-stage figure split (Initialization / Propagation) with real disparity maps at every scale |
| `paper_figs/RAFTStereo_fig1_architecture.png` | Iterative grammar: correlation pyramid zoom callout, circled L lookups, ⊕ accumulators, `···` repetition |
| `paper_figs/` (13 more) | IGEV, PSMNet, CoEx, GGEV, FoundationStereo, DEFOM, Selective, StereoAnywhere, BANet, GANet, NMRF — survey before inventing notation |
