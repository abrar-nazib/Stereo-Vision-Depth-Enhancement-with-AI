"""Final pass — FULL SceneFlow Driving manifest with automatic dark-frame scan.

Protocol identical to manifest_1000.py (build_manifest + stratified 80/20 split
by sequence, seed 42), but count = every frame in the dataset (4400). Before
splitting, ALL left frames are luminance-scanned (grayscale mean < 45, the same
criterion as the A05 scratchpad scan) and excluded — the 4 historical nuked
frames plus any newly found dark frames (the A08 held-out set still contained
one: 0800.png, EPE 9.57).

Output paths are rewritten to the Modal container layout
(/data/sceneflow_driving/...) so the same JSON works on Modal without edits;
local verification can pass --prefix to keep local paths instead.

Writes:
    manifests/manifest_train.json
    manifests/manifest_validation.json
    manifests/dark_frames.json   (scan provenance: every excluded frame + stats)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving")
CONTAINER_PREFIX = "/data/sceneflow_driving"
DARK_MEAN = 45.0

sys.path.insert(0, str(ROOT / "experiments/lightstereo_s_a02"))
sys.path.insert(0, str(ROOT / "experiments/a06_shallow"))

HISTORICAL_NUKED = {
    "frames_finalpass/35mm_focallength/scene_backwards/fast/left/0300.png",
    "frames_finalpass/35mm_focallength/scene_backwards/slow/left/0800.png",
    "frames_finalpass/15mm_focallength/scene_backwards/fast/left/0300.png",
    "frames_finalpass/15mm_focallength/scene_backwards/slow/left/0800.png",
}


def full_records(data_root: Path) -> list[dict[str, str]]:
    """Every left frame in the dataset, in the same record format as
    lightstereo_s_a02.run.build_manifest but without the count constraint
    (Driving sequences have uneven lengths: 300..700 frames)."""
    records: list[dict[str, str]] = []
    for group in sorted((data_root / "frames_finalpass").glob("*/*/*/left")):
        for left in sorted(group.glob("*.png")):
            relative = left.relative_to(data_root / "frames_finalpass")
            records.append({
                "left": str(left),
                "right": str(left.parent.parent / "right" / left.name),
                "disparity": str((data_root / "disparity" / relative).with_suffix(".pfm")),
                "sequence": str(group.relative_to(data_root)),
            })
    return records


def split(records, seed: int = 42):
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(r["sequence"], []).append(r)
    rnd = random.Random(seed)
    train, val = [], []
    for seq in sorted(groups):
        g = sorted(groups[seq], key=lambda x: x["left"])
        rnd.shuffle(g)
        k = round(len(g) * 0.2)
        val.extend(g[:k])
        train.extend(g[k:])
    return train, val




def scan_dark(records) -> list[dict]:
    """Grayscale-mean luminance scan over every selected left frame."""
    dark = []
    for i, r in enumerate(records):
        img = cv2.imread(r["left"], cv2.IMREAD_GRAYSCALE)
        mean = float(img.mean())
        if mean < DARK_MEAN:
            dark.append({"left_rel": str(Path(r["left"]).relative_to(DATA / "frames_finalpass")),
                         "mean": round(mean, 2), "std": round(float(img.std()), 2)})
        if (i + 1) % 1000 == 0:
            print(f"scan {i + 1}/{len(records)}", flush=True)
    return dark


def rewrite(records) -> list[dict]:
    out = []
    for r in records:
        out.append({"left": r["left"].replace(str(DATA), CONTAINER_PREFIX),
                    "right": r["right"].replace(str(DATA), CONTAINER_PREFIX),
                    "disparity": r["disparity"].replace(str(DATA), CONTAINER_PREFIX),
                    "sequence": r["sequence"]})
    return out


def main() -> None:
    all_records = full_records(DATA)
    print(f"dataset frames: {len(all_records)}")

    dark = scan_dark(all_records)
    dark_rels = {d["left_rel"] for d in dark}
    excluded = dark_rels | HISTORICAL_NUKED
    records = [r for r in all_records
               if str(Path(r["left"]).relative_to(DATA / "frames_finalpass")) not in excluded]
    print(f"excluded {len(all_records) - len(records)} dark frames "
          f"({len(dark_rels)} scanned + {len(HISTORICAL_NUKED & dark_rels or HISTORICAL_NUKED)} historical)")

    train, val = split(records)
    print(f"split: {len(train)} train / {len(val)} held-out")

    out = HERE / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_train.json").write_text(json.dumps(rewrite(train), indent=2) + "\n")
    (out / "manifest_validation.json").write_text(json.dumps(rewrite(val), indent=2) + "\n")
    (out / "dark_frames.json").write_text(json.dumps(
        {"threshold_mean": DARK_MEAN, "historical": sorted(HISTORICAL_NUKED),
         "scanned_dark": sorted(dark, key=lambda d: d["left_rel"])}, indent=2) + "\n")
    print(f"wrote {out}/manifest_train.json, manifest_validation.json, dark_frames.json")


if __name__ == "__main__":
    main()
