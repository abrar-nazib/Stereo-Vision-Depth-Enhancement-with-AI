# Color Palette & Brand Style — Uttaran

**This is the single source of truth for all colors and brand-specific styles.**
Derived from the Uttaran logo mark (`#f36f21`). Tuned for **print on white A4**:
light fills, dark strokes, no transparency, no dark page backgrounds.

---

## Brand Core

| Role | Hex | Notes |
|------|-----|-------|
| Uttaran Orange | `#f36f21` | Sampled from the logo mark. Primary brand ink. |
| Orange Deep | `#b4460b` | Stroke / text weight on orange fills |
| Orange Tint | `#fde3ce` | Light fill for orange-family shapes |
| Orange Wash | `#fff4ec` | Very light band / section background |

---

## Shape Colors (Semantic)

| Semantic Purpose | Fill | Stroke |
|------------------|------|--------|
| Primary/Neutral (Uttaran brand) | `#fde3ce` | `#b4460b` |
| Secondary (training fee) | `#f36f21` | `#b4460b` |
| Tertiary (soft band) | `#fff4ec` | `#f0b78e` |
| Start/Trigger | `#ffe8d1` | `#c2410c` |
| End/Success (incentive, retention) | `#c8ece0` | `#0f766e` |
| Warning/Reset (drop-off) | `#fde8e8` | `#c23c3c` |
| Decision | `#fdf1cf` | `#a16207` |
| Verification (Uttaran gate) | `#e4edf7` | `#1f3a5f` |
| Inactive/Disabled | `#f1f5f9` | `#94a3b8` (use dashed stroke) |
| Error | `#fbd5d5` | `#b91c1c` |

**Rule**: Always pair a darker stroke with a lighter fill for contrast.
**Print rule**: never fill a large area with a saturated colour — use the tint,
and carry the saturated colour in the stroke and in small solid accents only.

---

## Text Colors (Hierarchy)

| Level | Color | Use For |
|-------|-------|---------|
| Title | `#7c2d12` | Section headings, major labels |
| Subtitle | `#b4460b` | Subheadings, secondary labels |
| Body/Detail | `#475569` | Descriptions, annotations, metadata |
| Ink (max contrast) | `#0f172a` | Numbers, formulas, anything that must read at arm's length |
| On light fills | `#334155` | Text inside light-coloured shapes |
| On dark/saturated fills | `#ffffff` | Text inside orange or teal solid fills |

---

## Money Encoding (project-specific)

Two payment pools must never be confused with each other:

| Pool | Fill | Stroke | Meaning |
|------|------|--------|---------|
| Training fee (`$X` per trainee) | `#f36f21` | `#b4460b` | 40 / 30 / 5 / 15 / 10 |
| Incentive (`$Y` per trainee) | `#14a08a` | `#0f766e` | 50 / 50, retention only |
| Unearned / forfeited | `#f1f5f9` | `#cbd5e1` (dashed) | The part attrition takes away |

---

## Evidence Artifact Colors

| Artifact | Background | Text Color |
|----------|-----------|------------|
| Worked-example / formula block | `#fff4ec` | `#0f172a` |
| Code snippet | `#1e293b` | Syntax-coloured |
| JSON/data example | `#1e293b` | `#22c55e` |

---

## Default Stroke & Line Colors

| Element | Color |
|---------|-------|
| Arrows | Stroke colour of the source element's semantic purpose |
| Structural lines (spines, dividers, timelines) | `#f0b78e` (brand rule) or `#94a3b8` (neutral rule) |
| Marker dots (fill + stroke) | `#f36f21` |

---

## Background

| Property | Value |
|----------|-------|
| Canvas background | `#ffffff` (always — output is printed on A4) |
