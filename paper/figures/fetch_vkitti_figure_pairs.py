"""Fetch a few audited VKITTI2 *training* stereo pairs for figure selection.

Only paired RGB JPEGs are returned from a CPU-only Modal call. The large raw
archives remain on the existing volume; no dataset extraction or GPU runs.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import modal


app = modal.App("semtilestereo-figure-pairs")
volume = modal.Volume.from_name("svde-vkitti2")
SEMANTIC = Path("/vkitti/semantic/v3_random14_seed42/dataset.tar")
RIGHT = Path("/vkitti/full_stereo/teacher42_right_depth_v1.tar")
MANIFEST = Path("/vkitti/full_stereo/teacher42_manifest_v1.json")


@app.function(volumes={"/vkitti": volume}, timeout=1800, cpu=2, memory=4096)
def fetch() -> list[tuple[str, bytes, bytes]]:
    payload = json.loads(MANIFEST.read_text())
    clone = [row for row in payload["rows"]["train"] if row["variation"] == "clone"]
    selected = []
    for scene in ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20"):
        choices = [row for row in clone if row["scene"] == scene]
        selected.append(choices[len(choices) // 2])
    wanted_left = {f"dataset/images/train/{row['stem']}.jpg": row["stem"] for row in selected}
    wanted_right = {row["files"]["rgb_right"]: row["stem"] for row in selected}
    images: dict[str, dict[str, bytes]] = {row["stem"]: {} for row in selected}
    for archive_path, wanted, side in ((SEMANTIC, wanted_left, "left"),
                                       (RIGHT, wanted_right, "right")):
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                stem = wanted.get(member.name)
                if stem is None:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"unreadable member: {member.name}")
                images[stem][side] = stream.read()
                if all(side in images[key] for key in images):
                    break
    if not all(set(pair) == {"left", "right"} for pair in images.values()):
        raise FileNotFoundError("a selected VKITTI training stereo pair is missing")
    return [(stem, pair["left"], pair["right"]) for stem, pair in images.items()]


@app.local_entrypoint()
def main(output: str = "paper/figures/semtilestereo/candidates") -> None:
    destination = Path.cwd().resolve() / output
    destination.mkdir(parents=True, exist_ok=True)
    for stem, left, right in fetch.remote():
        (destination / f"{stem}_left.jpg").write_bytes(left)
        (destination / f"{stem}_right.jpg").write_bytes(right)
        print(stem, len(left), len(right))
