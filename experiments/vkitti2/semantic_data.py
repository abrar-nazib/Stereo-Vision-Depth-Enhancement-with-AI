"""Prepare full Virtual KITTI 2 left-camera semantic labels for Ultralytics.

The RGB and class-mask archives stay on Modal. Preparation uses container-local
scratch space; the caller can package the result into a small number of shards.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


CLASS_NAMES = (
    "Terrain", "Sky", "Tree", "Vegetation", "Building", "Road",
    "GuardRail", "TrafficSign", "TrafficLight", "Pole", "Misc",
    "Truck", "Car", "Van",
)
CLASS_COLORS = (
    (210, 0, 200), (90, 200, 255), (0, 199, 0), (90, 240, 0),
    (140, 140, 140), (100, 60, 100), (250, 100, 255), (255, 255, 0),
    (200, 200, 0), (255, 130, 0), (80, 80, 80), (160, 60, 60),
    (255, 127, 80), (0, 139, 139),
)
CLASS_NAMES_WITHOUT_GUARDRAIL = tuple(
    name for name in CLASS_NAMES if name != "GuardRail"
)
SPLIT_SCENES = {
    "train": frozenset(("Scene01", "Scene02", "Scene06")),
    "val": frozenset(("Scene18",)),
    "test": frozenset(("Scene20",)),
}
_SCENE_TO_SPLIT = {scene: split for split, scenes in SPLIT_SCENES.items() for scene in scenes}
_MEMBER = re.compile(
    r"^(Scene\d{2})/([^/]+)/frames/(rgb|classSegmentation)/Camera_0/"
    r"(?:rgb|classgt)_(\d{5})\.(?:jpg|png)$"
)


def decode_class_mask(rgb: np.ndarray, source: str) -> np.ndarray:
    """Map official RGB colors to 0..13 and Undefined black to ignore 255."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"{source}: expected uint8 RGB mask, got {rgb.shape} {rgb.dtype}")
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    result = np.full(rgb.shape[:2], 254, dtype=np.uint8)
    for class_id, (red, green, blue) in enumerate(CLASS_COLORS):
        result[packed == (red << 16 | green << 8 | blue)] = class_id
    result[packed == 0] = 255
    if np.any(result == 254):
        bad = rgb[result == 254][0].tolist()
        raise ValueError(f"{source}: unknown color {bad[0]}, {bad[1]}, {bad[2]}")
    return result


def build_split_index(members: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    """Pair Camera_0 RGB/masks by scene, variation and frame; hold out scenes."""
    found: dict[tuple[str, str, str], dict[str, str]] = {}
    for name in members:
        match = _MEMBER.fullmatch(name)
        if match is None:
            continue
        scene, variation, modality, frame = match.groups()
        if scene not in _SCENE_TO_SPLIT:
            raise ValueError(f"unexpected scene in {name}")
        key = (scene, variation, frame)
        bucket = found.setdefault(key, {})
        if modality in bucket:
            raise ValueError(f"duplicate {modality} for {key}")
        bucket[modality] = name
    if not found:
        raise ValueError("no Camera_0 RGB/class-mask members found")
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for (scene, variation, frame), pair in sorted(found.items()):
        if "rgb" not in pair:
            raise ValueError(f"missing RGB for {scene}/{variation}/{frame}")
        if "classSegmentation" not in pair:
            raise ValueError(f"missing class mask for {scene}/{variation}/{frame}")
        splits[_SCENE_TO_SPLIT[scene]].append((pair["rgb"], pair["classSegmentation"]))
    return splits


def prepare_dataset(rgb_tar: Path, mask_tar: Path, destination: Path) -> dict:
    """Convert all paired left-camera frames to YOLO semantic directory layout."""
    destination.mkdir(parents=True, exist_ok=True)
    counts = {split: Counter() for split in SPLIT_SCENES}
    split_counts = {split: 0 for split in SPLIT_SCENES}
    with tarfile.open(rgb_tar, "r:") as rgb_archive, tarfile.open(mask_tar, "r:") as mask_archive:
        rgb_members = {m.name for m in rgb_archive if m.isfile()}
        mask_members = {m.name for m in mask_archive if m.isfile()}
        pairs = build_split_index(rgb_members | mask_members)
        for split, records in pairs.items():
            (destination / "images" / split).mkdir(parents=True, exist_ok=True)
            (destination / "masks" / split).mkdir(parents=True, exist_ok=True)
            for rgb_name, mask_name in records:
                scene, variation = rgb_name.split("/", 2)[:2]
                frame = Path(rgb_name).stem.removeprefix("rgb_")
                stem = f"{scene}__{variation}__{frame}"
                rgb_source = rgb_archive.extractfile(rgb_name)
                mask_source = mask_archive.extractfile(mask_name)
                if rgb_source is None or mask_source is None:
                    raise ValueError(f"unreadable pair {rgb_name}, {mask_name}")
                image_path = destination / "images" / split / f"{stem}.jpg"
                with image_path.open("wb") as output:
                    shutil.copyfileobj(rgb_source, output, length=1024 * 1024)
                bgr = cv2.imdecode(np.frombuffer(mask_source.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"unreadable PNG {mask_name}")
                mask = decode_class_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask_name)
                if cv2.imread(str(image_path)).shape[:2] != mask.shape:
                    raise ValueError(f"image/mask shape mismatch: {rgb_name}, {mask_name}")
                mask_path = destination / "masks" / split / f"{stem}.png"
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"could not write {mask_path}")
                ids, amounts = np.unique(mask, return_counts=True)
                for class_id, amount in zip(ids.tolist(), amounts.tolist()):
                    name = "Undefined" if class_id == 255 else CLASS_NAMES[class_id]
                    counts[split][name] += amount
                split_counts[split] += 1
    audit = {
        "split_scenes": {split: sorted(scenes) for split, scenes in SPLIT_SCENES.items()},
        "split_counts": split_counts,
        "pixel_counts": {split: dict(counts[split]) for split in SPLIT_SCENES},
    }
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    dataset = {
        "path": str(destination), "train": "images/train", "val": "images/val",
        "test": "images/test", "masks_dir": "masks",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (destination / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    return audit


def reassign_validation_scene(root: Path, new_val: str = "Scene06",
                              old_val: str = "Scene18") -> None:
    """Correct the staging split when class audit shows old validation lacks classes.

    The initial archive used Scene18 validation. Reassign on ephemeral local disk
    without re-downloading or rewriting the 4 GB source archive.
    """
    for kind, extension in (("images", "jpg"), ("masks", "png")):
        train = root / kind / "train"
        val = root / kind / "val"
        incoming = sorted(train.glob(f"{new_val}__*.{extension}"))
        outgoing = sorted(val.glob(f"{old_val}__*.{extension}"))
        if not incoming or not outgoing:
            raise ValueError(f"cannot reassign {kind}: {new_val} or {old_val} absent")
        for source in incoming:
            destination = val / source.name
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
        for source in outgoing:
            destination = train / source.name
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)


def audit_converted_masks(root: Path, class_names: tuple[str, ...] = CLASS_NAMES) -> dict:
    """Count actual per-split class support after an optional scene reassignment."""
    pixel_counts: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}
    split_scenes: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        histogram = np.zeros(256, dtype=np.int64)
        scenes: set[str] = set()
        files = sorted((root / "masks" / split).glob("*.png"))
        for mask_path in files:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
                raise ValueError(f"invalid indexed mask: {mask_path}")
            histogram += np.bincount(mask.ravel(), minlength=256)
            scenes.add(mask_path.name.split("__", 1)[0])
        if np.any(histogram[len(class_names):255]):
            raise ValueError(f"out-of-range class ID in {split} masks")
        pixel_counts[split] = {
            **{name: int(histogram[i]) for i, name in enumerate(class_names)},
            "Undefined": int(histogram[255]),
        }
        split_counts[split] = len(files)
        split_scenes[split] = sorted(scenes)
    return {"split_scenes": split_scenes, "split_counts": split_counts,
            "pixel_counts": pixel_counts}


def remap_guardrail_to_ignore(mask: np.ndarray) -> np.ndarray:
    """Turn untrainable GuardRail (old ID 6) into void; compact old IDs 7–13."""
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("expected uint8 class-ID mask")
    if np.any((mask > 13) & (mask != 255)):
        raise ValueError("out-of-range source class ID")
    result = mask.copy()
    result[mask == 6] = 255
    result[(mask > 6) & (mask < 255)] -= 1
    return result


def convert_prepared_taxonomy(root: Path) -> None:
    """Apply 13-class taxonomy on prepared local data; leave RGB untouched."""
    for split in ("train", "val", "test"):
        for mask_path in (root / "masks" / split).glob("*.png"):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"unreadable mask {mask_path}")
            converted = remap_guardrail_to_ignore(mask)
            if not cv2.imwrite(str(mask_path), converted):
                raise OSError(f"failed to write {mask_path}")
    yaml_path = root / "dataset.yaml"
    config = yaml.safe_load(yaml_path.read_text())
    config["names"] = {i: name for i, name in enumerate(CLASS_NAMES_WITHOUT_GUARDRAIL)}
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))


def reshuffle_prepared_dataset(root: Path, seed: int = 42) -> dict:
    """Make a seeded 80/10/10 split, grouping scene+frame across variations.

    Operates only on an extracted scratch copy of the 14-class preparation.
    Every RGB and mask is paired before any file is moved.
    """
    split_names = ("train", "val", "test")
    inputs: dict[str, tuple[Path, Path]] = {}
    groups: dict[str, list[str]] = {}
    for split in split_names:
        images = {p.stem: p for p in (root / "images" / split).glob("*.jpg")}
        masks = {p.stem: p for p in (root / "masks" / split).glob("*.png")}
        if images.keys() != masks.keys():
            raise ValueError(f"unpaired image/mask stems in {split}")
        for stem in images:
            if stem in inputs:
                raise ValueError(f"duplicate prepared stem: {stem}")
            parts = stem.split("__")
            if len(parts) != 3 or not re.fullmatch(r"Scene\d{2}", parts[0]) or not parts[2].isdigit():
                raise ValueError(f"invalid prepared stem: {stem}")
            inputs[stem] = (images[stem], masks[stem])
            key = f"{parts[0]}__{parts[2]}"
            groups.setdefault(key, []).append(stem)
    if len(groups) < 3:
        raise ValueError("need at least three source-frame groups for train/val/test")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * 0.8)
    n_val = int(len(keys) * 0.1)
    if min(n_train, n_val, len(keys) - n_train - n_val) == 0:
        raise ValueError("too few groups for nonempty 80/10/10 split")
    assignments = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    for split, assigned in assignments.items():
        for key in assigned:
            for stem in groups[key]:
                image, mask = inputs[stem]
                image.rename(root / "images" / split / image.name)
                mask.rename(root / "masks" / split / mask.name)
    manifest = {
        "seed": seed,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "grouping": "scene+frame_across_variations",
        "groups": assignments,
    }
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"split_counts": {
        split: sum(len(groups[key]) for key in assigned)
        for split, assigned in assignments.items()
    }, "group_counts": {split: len(assigned) for split, assigned in assignments.items()}}
