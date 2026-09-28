"""A05 — cleaned 200-pair manifest: A04's 200 minus 4 dark-tunnel frames.

Nuked (near-black, EPE 16-34 at A04b latest ckpt):
  train  35mm_focallength/scene_backwards/fast/0300.png  (mean 30.4)
  val    35mm_focallength/scene_backwards/slow/0800.png  (mean 30.4)
  train  15mm_focallength/scene_backwards/fast/0300.png  (mean 34.5)
  val    15mm_focallength/scene_backwards/slow/0800.png  (mean 34.5)

Replacements (same sequence, verified bright, valid PFM + right present):
  train  35mm_focallength/scene_backwards/fast/0270.png  (mean 94.3)
  val    35mm_focallength/scene_backwards/slow/0582.png  (mean 104.9)
  train  15mm_focallength/scene_backwards/fast/0270.png  (mean 117.1)
  val    15mm_focallength/scene_backwards/slow/0710.png  (mean 125.4)

Split identity (which pair is train vs val) is preserved; only the 4 file
identities change. 160 train / 40 val, stratified by sequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving")

def _record_for(frame_rel: str) -> dict[str, str]:
    left = DATA / "frames_finalpass" / frame_rel
    parts = Path(frame_rel).parts  # <fl>/<dir>/<speed>/left/<file>
    right = DATA / "frames_finalpass" / parts[0] / parts[1] / parts[2] / "right" / parts[4]
    disp = DATA / "disparity" / parts[0] / parts[1] / parts[2] / "left" / Path(parts[4]).with_suffix(".pfm").name
    seq = str(Path(parts[0]) / parts[1] / parts[2] / "left")
    for p in (left, right, disp):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    return {"left": str(left), "right": str(right), "disparity": str(disp), "sequence": seq}


def build(a04_run: Path, out_dir: Path) -> tuple[list, list]:
    train = json.loads((a04_run / "manifest_train.json").read_text())
    val = json.loads((a04_run / "manifest_validation.json").read_text())

    swaps = {
        ("train", "35mm_focallength/scene_backwards/fast/left/0300.png"): "35mm_focallength/scene_backwards/fast/left/0270.png",
        ("val", "35mm_focallength/scene_backwards/slow/left/0800.png"): "35mm_focallength/scene_backwards/slow/left/0582.png",
        ("train", "15mm_focallength/scene_backwards/fast/left/0300.png"): "15mm_focallength/scene_backwards/fast/left/0270.png",
        ("val", "15mm_focallength/scene_backwards/slow/left/0800.png"): "15mm_focallength/scene_backwards/slow/left/0710.png",
    }
    by_split = {"train": train, "val": val}
    nuked = []
    for (split, src_rel), dst_rel in swaps.items():
        recs = by_split[split]
        src_abs = str(DATA / "frames_finalpass" / src_rel)
        hits = [r for r in recs if r["left"] == src_abs]
        if len(hits) != 1:
            raise RuntimeError(f"expected 1 hit for {src_abs}, found {len(hits)}")
        recs.remove(hits[0])
        nuked.append(hits[0]["left"])
        recs.append(_record_for(dst_rel))

    key = lambda r: (r["sequence"], r["left"])
    train_sorted = sorted(by_split["train"], key=key)
    val_sorted = sorted(by_split["val"], key=key)
    assert len(train_sorted) == 160 and len(val_sorted) == 40
    assert len({r["left"] for r in train_sorted + val_sorted}) == 200
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_train.json").write_text(json.dumps(train_sorted, indent=2) + "\n")
    (out_dir / "manifest_validation.json").write_text(json.dumps(val_sorted, indent=2) + "\n")
    (out_dir / "nuked.json").write_text(json.dumps(nuked, indent=2) + "\n")
    return train_sorted, val_sorted


if __name__ == "__main__":
    a04 = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "manifests")
    tr, va = build(a04, out)
    print(f"wrote {len(tr)} train / {len(va)} val to {out}")
