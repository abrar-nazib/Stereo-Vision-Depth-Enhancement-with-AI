"""Deterministic, scene-held-out Virtual KITTI 2 500-pair selection."""

from __future__ import annotations

import csv
import io
import shutil
import tarfile
from pathlib import Path


TRAIN_SCENES = ("Scene01", "Scene02", "Scene06", "Scene18")
VAL_SCENE = "Scene20"
VARIATIONS = (
    "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
    "clone", "fog", "morning", "overcast", "rain", "sunset",
)


def evenly_spaced_frames(frame_count: int, count: int = 10) -> list[int]:
    if count < 2 or frame_count < count:
        raise ValueError("sequence must contain at least the requested number of frames")
    return [(i * (frame_count - 1) + (count - 1) // 2) // (count - 1) for i in range(count)]


def required_members(scene: str, variation: str, frame: int) -> dict[str, str]:
    base = f"{scene}/{variation}/frames"
    index = f"{frame:05d}"
    return {
        "rgb_left": f"{base}/rgb/Camera_0/rgb_{index}.jpg",
        "rgb_right": f"{base}/rgb/Camera_1/rgb_{index}.jpg",
        "depth": f"{base}/depth/Camera_0/depth_{index}.png",
        "class_seg": f"{base}/classSegmentation/Camera_0/classgt_{index}.png",
    }


def manifest_from_metadata(metadata_archive: Path, frames_per_variation: int = 10) -> list[dict]:
    """Choose evenly spaced frames per variation, holding out Scene20 entirely."""
    rows = []
    with tarfile.open(metadata_archive, "r:gz") as archive:
        for scene in (*TRAIN_SCENES, VAL_SCENE):
            for variation in VARIATIONS:
                member = archive.extractfile(f"{scene}/{variation}/intrinsic.txt")
                if member is None:
                    raise FileNotFoundError(f"{scene}/{variation}/intrinsic.txt")
                intrinsic = list(csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"), delimiter=" "))
                cam0 = [r for r in intrinsic if r["cameraID"] == "0"]
                if not cam0:
                    raise ValueError(f"missing left intrinsics for {scene}/{variation}")
                fx_by_frame = {int(r["frame"]): float(r["K[0,0]"]) for r in cam0}
                frames = sorted(fx_by_frame)
                for index in evenly_spaced_frames(len(frames), frames_per_variation):
                    frame = frames[index]
                    rows.append({
                        "scene": scene, "variation": variation, "frame": frame,
                        "split": "val" if scene == VAL_SCENE else "train",
                        "fx": fx_by_frame[frame], "baseline_m": 0.532725,
                        "files": required_members(scene, variation, frame),
                    })
    if len(rows) != 50 * frames_per_variation or sum(r["split"] == "val" for r in rows) != 10 * frames_per_variation:
        raise AssertionError("unexpected train/held-out pair count")
    return rows


def extract_selected(archive_path: Path, members: set[str], output: Path) -> None:
    """Stream one modality tar and copy only exact, expected regular files."""
    missing = set(members)
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if member.name not in missing:
                continue
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            destination = output / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            missing.remove(member.name)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} members in {archive_path}: {sorted(missing)[:3]}")
