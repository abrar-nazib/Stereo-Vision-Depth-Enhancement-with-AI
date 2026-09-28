"""A08 — 1000-pair SceneFlow Driving manifest, stratified, seed 42.

Same protocol as A05/A06/A07 manifests (build_manifest + stratified split) at
count=1000. Held-out fraction stays 20% (200 val / 800 train). Frame selection
is evenly spaced per sequence; the 200-pair A05 subset is a strict subset by
construction (same linspace indices at n=200 divide the n=1000 selection
roughly — NOT guaranteed; independence noted in README instead).

Nuke list from A05 (4 dark frames) excluded if encountered.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving")

sys.path.insert(0, str(ROOT / "experiments/lightstereo_s_a02"))
from run import build_manifest as _build_manifest  # noqa: E402

NUKED = {
    "frames_finalpass/35mm_focallength/scene_backwards/fast/left/0300.png",
    "frames_finalpass/35mm_focallength/scene_backwards/slow/left/0800.png",
    "frames_finalpass/15mm_focallength/scene_backwards/fast/left/0300.png",
    "frames_finalpass/15mm_focallength/scene_backwards/slow/left/0800.png",
}


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


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    records = _build_manifest(DATA, count)
    before = len(records)
    records = [r for r in records
               if str(Path(r["left"]).relative_to(DATA / "frames_finalpass")) not in NUKED]
    print(f"selected {before}, {before - len(records)} dark frames excluded")
    if len(records) < count:
        raise RuntimeError(f"nuke filter dropped below {count}: {len(records)}")
    train, val = split(records)
    out = HERE / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_train.json").write_text(json.dumps(train, indent=2) + "\n")
    (out / "manifest_validation.json").write_text(json.dumps(val, indent=2) + "\n")
    print(f"wrote {len(train)} train / {len(val)} val to {out}")


if __name__ == "__main__":
    main()
