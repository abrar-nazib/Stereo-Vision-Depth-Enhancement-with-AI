"""Build a small verified VKITTI2 subset on Modal's CPU, never on a GPU.

When Modal access is available:
    uv run modal run -d experiments/vkitti2/modal_subset.py::stage_500
    uv run modal volume get svde-vkitti2 /subsets/ablation500.tar <SSD directory>
    uv run modal volume get svde-vkitti2 /subsets/ablation500.sha256 <SSD directory>
Verify SHA256 locally before extracting. No full modality archive is needed on SSD.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import modal

from experiments.vkitti2.subset import extract_selected, manifest_from_metadata


app = modal.App("svde-vkitti2-subset")
volume = modal.Volume.from_name("svde-vkitti2")
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "experiments.vkitti2.subset"
)


def _stage(frames_per_variation: int) -> None:
    pair_count = 50 * frames_per_variation
    name = f"ablation{pair_count}"
    archive_dir = Path("/vkitti/archives")
    staging = Path("/vkitti/subsets") / name
    output_tar = Path("/vkitti/subsets") / f"{name}.tar"
    output_hash = Path("/vkitti/subsets") / f"{name}.sha256"
    metadata = archive_dir / "vkitti_2.0.3_textgt.tar.gz"
    manifest = manifest_from_metadata(metadata, frames_per_variation)

    by_modality = {
        "rgb": {path for row in manifest for key, path in row["files"].items() if key.startswith("rgb_")},
        "depth": {row["files"]["depth"] for row in manifest},
        "classSegmentation": {row["files"]["class_seg"] for row in manifest},
    }
    for modality, paths in by_modality.items():
        source = archive_dir / f"vkitti_2.0.3_{modality}.tar"
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"[{modality}] extracting {len(paths)} exact members", flush=True)
        extract_selected(source, paths, staging)

    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary = output_tar.with_suffix(".tar.part")
    with tarfile.open(temporary, "w") as package:
        package.add(staging, arcname=name, recursive=True)
    temporary.replace(output_tar)
    digest = hashlib.sha256()
    with output_tar.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    output_hash.write_text(f"{digest.hexdigest()}  {output_tar.name}\n")
    volume.commit()
    print(f"[complete] {pair_count} pairs, {output_tar.stat().st_size} bytes, sha256 {digest.hexdigest()}", flush=True)


@app.function(image=image, volumes={"/vkitti": volume}, cpu=2, memory=8192, timeout=3 * 3600)
def stage_500() -> None:
    _stage(10)


@app.function(image=image, volumes={"/vkitti": volume}, cpu=2, memory=8192, timeout=3 * 3600)
def stage_1000() -> None:
    _stage(20)
