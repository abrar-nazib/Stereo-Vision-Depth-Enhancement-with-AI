"""Render the evidence-backed A08 V-chassis architecture figure.

The embedded input pair is validation example 0 (0022.png) from the latest
V-arm run.  Its predicted-disparity panel is cropped from the run's durable
step-035000 visualization, which was produced by inferencing that exact
checkpoint on the same pair.  This keeps the architecture figure reproducible
without silently substituting a segmentation visualization for disparity.

Run from the repository root:
    uv run python experiments/a06_shallow/render_a08_architecture.py
    cd .agents/skills/excalidraw-diagram/references
    uv run python render_excalidraw.py ../../../experiments/a06_shallow/a08_architecture.excalidraw
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "runs" / "A07v2V_20260928T154934Z"
STEP = 35_000
OUT = HERE / "a08_architecture.excalidraw"

# Paper-figure role palette, from ml-arch-diagram/references/grammar.md.
TEXT = "#1a1a2e"
NEUTRAL_FILL, NEUTRAL_EDGE = "#e9ecef", "#6c757d"
ENC_FILL, ENC_EDGE = "#a8c8e4", "#5a7fa0"
COST_FILL, COST_EDGE = "#ffc66d", "#c8861c"
REFINE_FILL, REFINE_EDGE = "#95d5b2", "#4a8a5f"
SEM_FILL, SEM_EDGE = "#cdb4f0", "#7c5cb0"
RED, DARK_RED = "#c1121f", "#8a0000"


class Scene:
    """Small, deterministic Excalidraw element builder."""

    def __init__(self) -> None:
        self.elements: list[dict] = []
        self.files: dict[str, dict] = {}
        self.n = 0

    def _id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}-{self.n:03d}"

    def _base(self, kind: str, x: float, y: float, width: float, height: float,
              *, stroke: str = TEXT, fill: str = "transparent", sw: int = 1,
              style: str = "solid") -> dict:
        return {
            "id": self._id(kind), "type": kind, "x": x, "y": y,
            "width": width, "height": height, "angle": 0,
            "strokeColor": stroke, "backgroundColor": fill,
            "fillStyle": "solid", "strokeWidth": sw, "strokeStyle": style,
            "roughness": 0, "opacity": 100, "seed": self.n * 101,
            "version": 1, "versionNonce": self.n * 1009, "isDeleted": False,
            "groupIds": [], "frameId": None, "roundness": None,
            "boundElements": None, "updated": 1, "link": None, "locked": False,
        }

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str,
             stroke: str, rounded: bool = True) -> str:
        el = self._base("rectangle", x, y, w, h, stroke=stroke, fill=fill, sw=2)
        if rounded:
            el["roundness"] = {"type": 3}
        self.elements.append(el)
        return el["id"]

    def text(self, x: float, y: float, text: str, *, size: int = 16,
             color: str = TEXT, align: str = "left", width: float | None = None) -> str:
        # The width is deliberately generous; Excalidraw recomputes it in-app.
        w = width if width is not None else max(20, len(max(text.split("\n"), key=len)) * size * 0.68)
        h = max(1, text.count("\n") + 1) * size * 1.28
        el = self._base("text", x, y, w, h, stroke=color)
        el.update({"text": text, "originalText": text, "fontSize": size,
                   "fontFamily": 3, "textAlign": align, "verticalAlign": "top",
                   "containerId": None, "autoResize": True, "lineHeight": 1.25})
        self.elements.append(el)
        return el["id"]

    def arrow(self, x: float, y: float, points: list[list[float]], *, color: str = TEXT,
              dashed: bool = False) -> str:
        max_x = max(p[0] for p in points)
        min_x = min(p[0] for p in points)
        max_y = max(p[1] for p in points)
        min_y = min(p[1] for p in points)
        el = self._base("arrow", x, y, max_x - min_x, max_y - min_y, stroke=color, sw=2,
                        style="dashed" if dashed else "solid")
        el.update({"points": points, "startBinding": None, "endBinding": None,
                   "lastCommittedPoint": None, "startArrowhead": None,
                   "endArrowhead": "triangle", "elbowed": False})
        self.elements.append(el)
        return el["id"]

    def line(self, x: float, y: float, points: list[list[float]], *, color: str = TEXT,
             dashed: bool = False, sw: int = 1) -> str:
        max_x, min_x = max(p[0] for p in points), min(p[0] for p in points)
        max_y, min_y = max(p[1] for p in points), min(p[1] for p in points)
        el = self._base("line", x, y, max_x - min_x, max_y - min_y, stroke=color,
                        sw=sw, style="dashed" if dashed else "solid")
        el.update({"points": points, "lastCommittedPoint": None,
                   "startBinding": None, "endBinding": None})
        self.elements.append(el)
        return el["id"]

    def dot(self, x: float, y: float) -> str:
        el = self._base("ellipse", x, y, 14, 14, stroke=DARK_RED, fill=RED, sw=2)
        self.elements.append(el)
        return el["id"]

    def image(self, x: float, y: float, w: float, h: float, path: Path, label: str) -> str:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        self.files[label] = {
            "id": label, "mimeType": "image/png", "created": 1,
            "lastRetrieved": 1, "dataURL": f"data:image/png;base64,{data}",
        }
        el = self._base("image", x, y, w, h, stroke="transparent", fill="transparent", sw=0)
        el.update({"fileId": label, "status": "saved", "scale": [1, 1], "crop": None})
        self.elements.append(el)
        return el["id"]

    def cuboid(self, x: float, y: float, w: float, h: float, *, fill: str, stroke: str) -> None:
        """A cost/semantic tensor rather than a named rectangle."""
        depth = 20
        self.rect(x, y + depth, w, h, fill=fill, stroke=stroke, rounded=False)
        # Closed top and side faces: every visible cuboid edge is intentional.
        self.line(x, y + depth,
                  [[0, 0], [depth, -depth], [w + depth, -depth], [w, 0], [0, 0]],
                  color=stroke, sw=2)
        self.line(x + w, y + depth,
                  [[0, 0], [depth, -depth], [depth, h - depth], [0, h], [0, 0]],
                  color=stroke, sw=2)

    def trapezoid(self, x: float, y: float, w: float, h: float, *, fill: str, stroke: str) -> None:
        # A filled neutral block plus slanted boundaries reads as a learned-free upsample.
        self.rect(x + 10, y, w - 20, h, fill=fill, stroke=stroke, rounded=False)
        self.line(x, y + h, [[0, 0], [10, -h], [w - 10, -h], [w, 0]], color=stroke, sw=2)


def png_copy(source: Path, destination: Path, *, crop: tuple[int, int, int, int] | None = None) -> None:
    image = Image.open(source).convert("RGB")
    if crop:
        image = image.crop(crop)
    image.save(destination, "PNG", optimize=True)


def build_assets() -> tuple[Path, Path, Path]:
    """Make source-identifiable thumbnails for the embedded Excalidraw files map."""
    manifest = json.loads((RUN / "manifest_validation.json").read_text())
    record = manifest[0]
    assets = HERE / ".a08_figure_assets"
    assets.mkdir(exist_ok=True)
    left, right, pred = assets / "left_0022.png", assets / "right_0022.png", assets / "pred_0022_step035000.png"
    png_copy(Path(record["left"]), left)
    png_copy(Path(record["right"]), right)
    collage = RUN / "visualizations" / "step_035000" / "validation_00.png"
    if not collage.exists():
        raise FileNotFoundError(f"Missing durable inference visualization: {collage}")
    # Matplotlib panel geometry in visualize_v2.py at 1650×935: predicted map only.
    png_copy(collage, pred, crop=(37, 518, 714, 897))
    return left, right, pred


def main() -> None:
    left, right, prediction = build_assets()
    result = json.loads((RUN / "results.json").read_text())
    train_n = len(json.loads((RUN / "manifest_train.json").read_text()))
    val_n = len(json.loads((RUN / "manifest_validation.json").read_text()))
    headline = (
        f"A08 V-chassis  ·  0.421 M trainable  ·  frozen YOLO26s encoder (ADE20K)  ·  "
        f"held-out EPE {result['final']['epe']:.3f} px @ {STEP // 1000}k ({train_n}/{val_n} pairs)"
    )

    s = Scene()
    # Header badge — the sole result/meta element on canvas.
    s.rect(530, 30, 1440, 52, fill="#ffffff", stroke=NEUTRAL_EDGE, rounded=False)
    s.text(556, 47, headline, size=20, width=1380)

    # Actual paired input sample. The two arrows meet at the shared encoder.
    s.image(50, 340, 190, 107, left, "left-0022")
    s.image(50, 550, 190, 107, right, "right-0022")
    s.text(98, 458, "Iₗ  left", size=16)
    s.text(94, 668, "Iᵣ  right", size=16)
    s.arrow(246, 395, [[0, 0], [88, 83]], color=TEXT)
    s.arrow(246, 603, [[0, 0], [88, -93]], color=TEXT)

    # Frozen shared feature pyramid f2/f4/f8/f16.
    bars = [(340, 430, 22, 120, "32c", "1/2"), (374, 448, 22, 84, "128c", "1/4"),
            (408, 466, 22, 48, "256c", "1/8"), (442, 478, 22, 24, "256c", "1/16")]
    for x, y, w, h, ch, scale in bars:
        s.rect(x, y, w, h, fill=ENC_FILL, stroke=ENC_EDGE, rounded=False)
        s.text(x - 10, y - 22, ch, size=13, color=ENC_EDGE)
        s.text(x - 8, 557, scale, size=13, color=NEUTRAL_EDGE)
    s.text(305, 588, "frozen YOLO26s encoder  ❄", size=16, color=ENC_EDGE)
    s.text(287, 612, "shared L/R weights · layers 0–6", size=13, color=NEUTRAL_EDGE)

    # f16 matching volume and the exact V-arm confidence insertion point.
    s.arrow(464, 490, [[0, 0], [94, 0]], color=ENC_EDGE)
    s.text(490, 465, "fL16  ·  fR16", size=13, color=ENC_EDGE)
    s.cuboid(558, 445, 132, 82, fill=COST_FILL, stroke=COST_EDGE)
    s.text(573, 472, "group CV", size=16, color=COST_EDGE)
    s.text(570, 493, "G=8 · D=24 · 1/16", size=13, color=COST_EDGE)
    s.text(560, 546, "dispCorr = meanG(CV)", size=12, color=COST_EDGE)

    s.arrow(712, 486, [[0, 0], [74, 0]], color=COST_EDGE)
    s.rect(786, 452, 116, 68, fill=COST_FILL, stroke=COST_EDGE)
    s.text(809, 466, "3-D agg", size=16, color=COST_EDGE)
    # agg[:3] is exactly one Conv3d → GroupNorm → SiLU block; VetoConf gates it.
    s.text(801, 488, "prefix  conv×1", size=12, color=COST_EDGE)
    s.text(807, 532, "h  [16,D,H/16,W/16]", size=11, color=NEUTRAL_EDGE)

    # Semantic correlation is a tensor, and VetoConf is a learned 3-D operation.
    s.arrow(464, 478, [[0, 0], [0, -146], [280, -146]], color=SEM_EDGE, dashed=True)
    s.text(500, 304, "fL16 · fR16", size=12, color=SEM_EDGE)
    s.cuboid(744, 280, 108, 54, fill=SEM_FILL, stroke=SEM_EDGE)
    s.text(763, 303, "semCorr", size=15, color=SEM_EDGE)
    # dispCorr uses its own upper lane: it must never pass through the agg-prefix box.
    s.arrow(690, 445, [[0, 0], [0, -64], [212, -64], [212, -135]], color=COST_EDGE)
    s.text(808, 367, "dispCorr", size=11, color=COST_EDGE)
    s.arrow(852, 317, [[0, 0], [50, 0]], color=SEM_EDGE)
    s.rect(902, 270, 158, 82, fill=SEM_FILL, stroke=SEM_EDGE)
    s.text(927, 283, "VetoConf", size=17, color=SEM_EDGE)
    s.text(918, 306, "dispCorr ⊗ semCorr", size=11, color=SEM_EDGE)
    s.text(920, 324, "3-D: 1→16→16→1", size=11, color=SEM_EDGE)
    s.text(947, 340, "residual + σ", size=11, color=SEM_EDGE)
    s.text(935, 362, "7.8k params", size=11, color=NEUTRAL_EDGE)

    # This direct downward wire fixes the original floating/ambiguous gate arrow.
    s.arrow(981, 354, [[0, 0], [0, 125]], color=SEM_EDGE)
    s.text(991, 397, "conf [1,D,…]", size=11, color=SEM_EDGE)
    s.arrow(904, 486, [[0, 0], [78, 0]], color=COST_EDGE)
    s.dot(981, 479)
    s.text(973, 477, "⊗", size=18, color=TEXT)
    s.arrow(995, 486, [[0, 0], [55, 0]], color=TEXT)
    s.rect(1050, 452, 108, 68, fill=COST_FILL, stroke=COST_EDGE)
    s.text(1065, 466, "3-D agg", size=16, color=COST_EDGE)
    s.text(1065, 488, "tail  conv×2", size=12, color=COST_EDGE)

    # Tile state is a slot glyph, not a generic named tensor box.
    s.arrow(1158, 486, [[0, 0], [46, 0]], color=COST_EDGE)
    s.text(1064, 534, "softargmin", size=11, color=COST_EDGE)
    s.rect(1204, 456, 114, 58, fill="#ffffff", stroke=NEUTRAL_EDGE, rounded=False)
    for x, label in zip((1218, 1240, 1268, 1290), ("d", "sₓ", "sᵧ", "h,c")):
        s.text(x, 476, label, size=15, color=TEXT)
    s.text(1206, 525, "tile state  1/16", size=12, color=NEUTRAL_EDGE)

    # Coarse-to-fine single-pass HITNet propagation. Trapezoids are plane upsamples.
    # Compact second-row staircase: state drops into the first scale, then all
    # propagation stages flow left→right through explicit plane-upsample ops.
    stage_y = 620
    stages = [
        (1204, "HITNet\npropagate", "1/16", "L16 ×0.1"),
        (1404, "HITNet\npropagate", "1/8", "L8 ×0.2"),
        (1604, "HITNet\npropagate", "1/4", "L4 ×0.3"),
        (1804, "HITNet\npropagate", "1/2", "L2 ×0.5"),
    ]
    previous = None
    for i, (x, name, scale, loss) in enumerate(stages):
        if i == 0:
            # Tile state enters the first propagation block through its top centre.
            s.arrow(1260, 514, [[0, 0], [0, stage_y - 514]], color=REFINE_EDGE)
        else:
            # Each operation has one arrow into its left face and one out of its right face.
            s.arrow(previous, stage_y + 42, [[0, 0], [x - previous, 0]], color=REFINE_EDGE)
        s.rect(x, stage_y, 112, 84, fill=REFINE_FILL, stroke=REFINE_EDGE)
        s.text(x + 11, stage_y + 16, name, size=13, color=REFINE_EDGE)
        s.text(x + 35, stage_y + 54, f"×1  ·  {scale}", size=12, color=REFINE_EDGE)
        # Offset only L16's loss arrow; it shares this top border with the state wire.
        sup_x = x + (85 if i == 0 else 49)
        s.dot(sup_x, stage_y - 48)
        s.text(x + 18, stage_y - 70, loss, size=12, color=DARK_RED)
        s.arrow(sup_x + 7, stage_y - 34, [[0, 0], [0, 32]], color=DARK_RED, dashed=True)
        if i < len(stages) - 1:
            up_x = x + 128
            s.trapezoid(up_x, stage_y + 10, 56, 64, fill=NEUTRAL_FILL, stroke=NEUTRAL_EDGE)
            s.text(up_x - 2, stage_y + 91, "plane up ×2", size=10, color=NEUTRAL_EDGE)
            s.arrow(x + 112, stage_y + 42, [[0, 0], [up_x - (x + 112), 0]], color=REFINE_EDGE)
            previous = up_x + 56
        else:
            previous = x + 112

    # The last non-learned plane upsample and actual predicted disparity from this pair.
    last_up_x = 1940
    s.arrow(previous, stage_y + 42, [[0, 0], [last_up_x - previous, 0]], color=NEUTRAL_EDGE)
    s.trapezoid(last_up_x, stage_y + 10, 72, 64, fill=NEUTRAL_FILL, stroke=NEUTRAL_EDGE)
    s.text(last_up_x + 3, stage_y + 91, "plane up ×2", size=10, color=NEUTRAL_EDGE)
    s.arrow(last_up_x + 72, stage_y + 42, [[0, 0], [34, 0]], color=NEUTRAL_EDGE)
    s.image(2046, 610, 190, 106, prediction, "prediction-0022-step035000")
    s.dot(2134, 562)
    s.text(2104, 540, "L1 ×1.0", size=12, color=DARK_RED)
    s.arrow(2141, 576, [[0, 0], [0, 32]], color=DARK_RED, dashed=True)
    s.text(2077, 728, "predicted disparity  d̂  (px)", size=15)
    s.text(2065, 750, "V checkpoint · val 0022 · step 35k", size=11, color=NEUTRAL_EDGE)

    # Stage brackets stay below all elements they name.
    bracket_y = 800
    for x0, x1, label in ((45, 520, "shared frozen encoder  1/2 → 1/16"),
                          (545, 1168, "V-arm matching + confidence gating @ 1/16"),
                          (1180, 2020, "plane-tile propagation  (coarse → fine)")):
        s.line(x0, bracket_y, [[0, 0], [0, 14]], color=NEUTRAL_EDGE, dashed=True)
        s.line(x0, bracket_y + 7, [[0, 0], [x1 - x0, 0]], color=NEUTRAL_EDGE, dashed=True)
        s.line(x1, bracket_y, [[0, 0], [0, 14]], color=NEUTRAL_EDGE, dashed=True)
        s.text((x0 + x1) / 2 - 125, bracket_y + 24, label, size=13, color=NEUTRAL_EDGE)

    # Legend: every role color and the loss dot are decoded.
    legend_y = 900
    entries = [(ENC_FILL, ENC_EDGE, "frozen YOLO26s encoder ❄"),
               (COST_FILL, COST_EDGE, "correlation / 3-D aggregation"),
               (SEM_FILL, SEM_EDGE, "semantic correlation + VetoConf"),
               (REFINE_FILL, REFINE_EDGE, "HITNet propagation"),
               (NEUTRAL_FILL, NEUTRAL_EDGE, "non-learned plane upsample")]
    x = 70
    for fill, edge, label in entries:
        s.rect(x, legend_y, 26, 17, fill=fill, stroke=edge, rounded=False)
        s.text(x + 36, legend_y + 1, label, size=13)
        x += 280 if label != "semantic correlation + VetoConf" else 315
    s.dot(1600, legend_y + 2)
    s.text(1625, legend_y + 1, "supervision: multi-scale L1 (1/.5/.3/.2/.1) + ∇ consistency .5 + τ1 hinge .2", size=13)

    data = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
            "elements": s.elements, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": 20},
            "files": s.files}
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT} with validation 0022 and step-{STEP:06d} prediction")


if __name__ == "__main__":
    main()
