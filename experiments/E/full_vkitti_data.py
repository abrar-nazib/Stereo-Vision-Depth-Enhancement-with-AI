"""Build a stereo manifest from the frozen semantic teacher's exact split."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

from experiments.vkitti2.subset import VARIATIONS, required_members


def load_intrinsics(archive_path: Path) -> dict[tuple[str, str, int], float]:
    result = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/intrinsic.txt"):
                continue
            scene, variation, *_ = member.name.split("/")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable intrinsic: {member.name}")
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter=" ")
            for record in reader:
                if record["cameraID"] == "0":
                    result[(scene, variation, int(record["frame"]))] = float(record["K[0,0]"])
    if not result:
        raise ValueError("no left-camera intrinsic metadata")
    return result


def teacher_split(archive_path: Path) -> dict:
    with tarfile.open(archive_path, "r:") as archive:
        stream = archive.extractfile("dataset/split_manifest.json")
        if stream is None:
            raise ValueError("semantic archive lacks split_manifest.json")
        return json.load(stream)


def rows_from_teacher_split(split: dict, fx: dict[tuple[str, str, int], float],
                            variations: tuple[str, ...] = VARIATIONS) -> dict[str, list[dict]]:
    if split.get("seed") != 42 or split.get("grouping") != "scene+frame_across_variations":
        raise ValueError("semantic teacher split is not the agreed seed-42 frame grouping")
    groups = split.get("groups", {})
    if set(groups) != {"train", "val", "test"}:
        raise ValueError("teacher split must have train, val and test groups")
    if len(set(sum((values for values in groups.values()), []))) != sum(map(len, groups.values())):
        raise ValueError("source frame appears in multiple partitions")
    rows = {name: [] for name in ("train", "val", "test")}
    for name, keys in groups.items():
        for key in keys:
            scene, frame_text = key.split("__")
            frame = int(frame_text)
            for variation in variations:
                focal = fx.get((scene, variation, frame))
                if focal is None:
                    raise ValueError(f"missing intrinsic for {scene}/{variation}/{frame}")
                rows[name].append({"scene": scene, "variation": variation,
                                   "frame": frame, "stem": f"{scene}__{variation}__{frame:05d}",
                                   "split": name, "fx": focal, "baseline_m": 0.532725,
                                   "files": required_members(scene, variation, frame)})
    return rows


def copy_selected_members(source_path: Path, wanted: set[str],
                          output: tarfile.TarFile) -> None:
    """Copy an audited subset from a raw VKITTI tar without extracting it."""
    missing = set(wanted)
    with tarfile.open(source_path, "r:") as source:
        for member in source:
            if member.name not in missing:
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            output.addfile(member, stream)
            missing.remove(member.name)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} members from {source_path}: {sorted(missing)[:3]}")
