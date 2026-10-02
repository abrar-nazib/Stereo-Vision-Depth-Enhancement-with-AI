# ML Architecture Diagram Grammar

Distilled from 20+ published paper figures, the StereoLite project figures, and the
stereolite `diagram-drawer` conventions. The reference images live beside this file.

## The five laws

1. **The canvas holds data and operations — nothing else.** Every shape is a tensor or
   an op. No "why" notes, no result tables, no scoreboards, no "NOT in this graph"
   boxes. The one permitted meta element is a single header badge line (model name,
   params, latency, headline metric) — see `stereolite_arch_mpl.png`.
2. **Tensors are pictures, not boxes.** Feature pyramids are groups of vertical bars;
   cost volumes are cuboids; intermediate disparities are rendered image thumbnails.
   A rectangle with a tensor's *name* in it is not a tensor.
3. **Operations are quiet shapes.** Learnable stages = rounded rect with the op name,
   `×N` repeat count, and param count underneath. Pointwise ops = circled symbols
   (`⊗ ⊕ σ L Δ`). Geometric warps = parallelogram or trapezoid.
4. **Color means role, and the legend decodes it.** 3–6 role colors, identical across
   every figure of the same model. Legend strip at the bottom: chip → role name.
5. **Supervision is a red dot.** A crimson dot on every supervised tensor, labeled
   `ℒ̂ᵢ` in math italic, dashed line (or bare dot, stereolite style) to a loss note
   in the legend. Weights (`×1.0 … ×0.1`) sit in tiny text under the dot.

## Tensor vocabulary

| Tensor | Draw as | Reference |
|---|---|---|
| Feature map (one scale) | one vertical bar, height ∝ resolution | `stereolite_arch_mpl.png` encoder |
| Feature pyramid | 3–5 bars tallest→shortest; channel count on top (`96c`), resolution under (`1/16`) | same |
| 3D cost volume | isometric cuboid (or wedge), label carries resolution + `D=` | RAFTStereo pyramid, stereolite "local CV" |
| Correlation pyramid | row of cuboids with shrinking widths, often in a dotted zoom callout | `paper_figs/RAFTStereo_fig1_architecture.png` |
| Disparity / depth / seg map | rendered image thumbnail (jet colormap for depth) | RAFTStereo output, stereolite output |
| State / latent vector | white box of slot glyphs (`d sx sy f c`) + tiny per-letter legend below | `stereolite_arch_mpl.png` tile-state |
| Input images | real photo thumbnails, labeled `I_l` / `I_r` | every reference |

## Operation vocabulary

| Op | Draw as |
|---|---|
| Learnable stage (conv block, GRU, propagate) | rounded rect, name + `×N` + params |
| Iterative refinement | rounded rect with `↻` glyph inside |
| Pointwise multiply / add / concat | circled `⊗ ⊕` |
| Correlation lookup | circled `L` in role color |
| Plane / convex upsample | trapezoid (narrowing) or parallelogram, arrow label `2× upsample` |
| Recurrence / feedback | arc arrow or curved back-edge |
| Skip / auxiliary flow | dashed arrow |
| Softmax over disparity | small bar-group (one bar per hypothesis) |

## Color palette (role → fill / edge)

| Role | Fill | Edge |
|---|---|---|
| Encoder / frozen backbone | `#a8c8e4` | `#5a7fa0` |
| Cost volume / matching | `#ffc66d` | `#c8861c` |
| Refinement / iteration | `#95d5b2` | `#4a8a5f` |
| Generic block / IO | `#e9ecef` | `#6c757d` |
| Supervision | `#c1121f` | `#8a0000` |
| Semantic / veto / fusion branch | `#cdb4f0` | `#7c5cb0` (extension — pick one and keep it) |
| Text | `#1a1a2e` | |

Frozen modules: grey-out or `❄` badge on the block, stated once in the legend
(`❄ frozen — never trained`). Trainable blocks stay fully colored.

## Annotation rules

- Tensor shapes live **under or beside the wire**, never inside a block:
  `[B, 24, H/16, W/16]` or just `1/16 · D=24`.
- Arrow labels name the tensor (`fL16`, `conf`, `Δd`, `w·Δd`).
- Stage brackets: thin dotted brackets **under the flow** naming stages
  ("encoder 1/2→1/16", "iterative refinement (coarse → fine)").
- Source line under the title in small italics: `LiteAnyStereo / Jing et al., 2025`.
- Repetition: `···` ellipsis or `×N`; never draw 8 identical blocks.

## Anti-patterns (all observed in rejected diagrams)

- Text boxes that narrate ("WHY:", "RESULT:", "THE FIX:") — move to the paper text.
- Uniform boxes-in-rows with no tensor imagery — that is a slide, not a figure.
- Every conv drawn individually — show blocks; details live in code.
- > 12 boxes on one canvas — split into overview + module drill-downs.
- Arrows that start/end in whitespace or cross band borders without landing.
- Text overflowing its shape (check in the *app*, not just the renderer — fonts differ).
- Redundant color decoration — each color must appear in the legend.

## Common mistakes and remedies

Every one of these was made in real rejected diagrams. Check the list before
declaring a figure done.

| # | Mistake | Remedy |
|---|---|---|
| 1 | Meta boxes on the canvas: "WHY:", "RESULT:", "NOT in this graph", scoreboards | Meta lives in ONE header badge line or in the paper text — never on the figure |
| 2 | Tensors drawn as named rectangles | Bar-pyramids, cuboids, slot-glyph state boxes, rendered thumbnails (see Tensor vocabulary) |
| 3 | Uniform box rows with no visual argument | View 2–3 gallery references *before* drawing; copy their skeleton, then specialize |
| 4 | Every layer drawn (62-cell diagrams) | Blocks, not layers; ≤ 12 boxes; split into overview + module detail |
| 5 | Trusting the `width` field on text elements | Excalidraw recomputes natural text width from the string in the app; the headless renderer's font is *narrower*. Shorten the string itself and keep 15% headroom; verify in the app, not just the PNG |
| 6 | Labels colliding with neighboring labels (wire label vs bar channel label, caption vs side box) | ≥ 12px clearance between label boxes; give each label a dedicated row/band; re-render after every move |
| 7 | Redundant decorative arrows (a "taps" arrow duplicating dots already on the wires) — and then removing a required chain arrow in the same cleanup | An arrow must carry flow meaning; before deleting any arrow, re-verify the main input→output chain is still unbroken end-to-end |
| 8 | Arrow endpoints inside a shape or in mid-air (wire starting inside a bar; section arrow landing on a band border instead of a block) | Compute endpoints from shape edges; arrowheads must land on the target shape's border |
| 9 | Label rows at a fixed y under variable-height bars (resolutions collided with the tallest bar's bottom) | Anchor shared rows to the *group* bounding box max-y + offset, not to one bar |
| 10 | Stage brackets that stop short of the elements they name | Brackets span the full stage bounding box, including the last element |
| 11 | Branch annotations placed on top of the per-bar channel labels | Branch annotations go above the group with their own offset, clear of per-bar labels |
| 12 | Params/suffixes crammed into section titles ("2 · THE VETO — … (7.8K params)") until they overflow | Titles stay short; params go on their own line under the block name |
| 13 | A branch wire crosses or visually traverses an unrelated block (for example, a correlation branch routed through the aggregation prefix to reach a gate) | Give each branch a dedicated upper or lower routing lane. Its horizontal segment must clear all unrelated block bounds, and its arrowhead must terminate on the intended module border. |
| 14 | An upsample or intermediate operator looks bypassed because one continuous arrow passes through it | Draw one input arrow ending on the operator's entry face and a separate output arrow beginning at its exit face. Never run an arrow through a trapezoid, cuboid, or rounded block. |
| 15 | An isometric tensor has an open or missing face, especially the lower side edge | Draw closed top and side polylines: top `front-TL → back-TL → back-TR → front-TR → front-TL`; side `front-TR → back-TR → back-BR → front-BR → front-TR`. Inspect at 100% before accepting it. |
| 16 | A label technically fits in a headless render but touches the next arrow, block, or its own border | Treat labels as bounded annotations: shorten a compound operator label or move a secondary term to its output wire. Keep visible whitespace around all four sides rather than relying on natural text width. |

## Two figure kinds

1. **Overview** — end-to-end dataflow, inputs left, output right
   (`stereolite_arch_mpl.png`, `paper_figs/RAFTStereo_fig1_architecture.png`).
2. **Module detail** — zoom into one overview box: its inputs become tensors on the
   left, internal ops drawn out, outputs leave right
   (`stereolite_gev_module.png`, `sgnet_fig1.png` confidence path).
   Tag the overview box with a circled letter and tag the detail figure to match.

## Build workflow

1. Inventory from the model code: every tensor (name, shape, scale), op, param count,
   supervision point, frozen/trainable split.
2. Sketch the main axis on paper first: inputs → encoder → matching → refinement →
   output. Branches (context, semantic, supervision) hang off it.
3. Draw in excalidraw using the `excalidraw-diagram` skill's render loop. Embed real
   image thumbnails as `type: "image"` elements with a `files` map of dataURLs —
   the renderer supports them natively.
4. Run the render-inspect-fix loop until: every arrow lands on a shape, every tensor
   has a resolution, the legend covers every color, and no text overflows **in the
   app's fonts** (keep 15% width headroom on every label).
