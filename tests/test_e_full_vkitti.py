"""Contracts for the full-data E3/E4/D2 comparison."""

import io
import tarfile

import cv2
import numpy as np
import pytest

from experiments.E.full_vkitti_data import copy_selected_members, rows_from_teacher_split
from experiments.E.full_vkitti_train import ARM_CONFIGS, read_full_pair


def test_rows_follow_teacher_groups_without_cross_split_leakage():
    split = {"seed": 42, "grouping": "scene+frame_across_variations",
             "groups": {"train": ["Scene01__00001"],
                        "val": ["Scene01__00002"],
                        "test": ["Scene20__00003"]}}
    fx = {(scene, variation, frame): 700.0
          for scene, frame in (("Scene01", 1), ("Scene01", 2), ("Scene20", 3))
          for variation in ("clone", "fog")}
    rows = rows_from_teacher_split(split, fx, ("clone", "fog"))
    assert {key: len(value) for key, value in rows.items()} == {
        "train": 2, "val": 2, "test": 2}
    assert rows["test"][1]["stem"] == "Scene20__fog__00003"
    assert rows["test"][1]["files"]["rgb_right"] == (
        "Scene20/fog/frames/rgb/Camera_1/rgb_00003.jpg")
    assert rows["test"][1]["fx"] == 700.0


def test_missing_stereo_modality_fails_audit():
    split = {"seed": 42, "grouping": "scene+frame_across_variations",
             "groups": {"train": ["Scene01__00001"], "val": ["Scene01__00002"],
                        "test": ["Scene20__00003"]}}
    with pytest.raises(ValueError, match="missing intrinsic"):
        rows_from_teacher_split(split, {}, ("clone",))


def test_staging_rejects_missing_members(tmp_path):
    source = tmp_path / "source.tar"
    target = tmp_path / "target.tar"
    with tarfile.open(source, "w") as archive:
        payload = b"frame"
        member = tarfile.TarInfo("Scene01/clone/right.jpg")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(target, "w") as output:
        with pytest.raises(FileNotFoundError, match="missing 1"):
            copy_selected_members(source, {"Scene01/clone/right.jpg", "missing.png"}, output)


def test_modal_comparison_has_semantic_compact_reference():
    assert set(ARM_CONFIGS) == {"E3", "E4", "D2"}
    assert ARM_CONFIGS["E3"] == (32, 128, True)
    assert ARM_CONFIGS["E4"] == (32, 128, False)
    assert ARM_CONFIGS["D2"] == (8, 32, True)


def test_full_pair_reads_indexed_mask_without_resizing(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images/train"
    masks = dataset / "masks/train"
    right = tmp_path / "Scene01/clone/frames/rgb/Camera_1"
    depth = tmp_path / "Scene01/clone/frames/depth/Camera_0"
    for folder in (images, masks, right, depth):
        folder.mkdir(parents=True)
    left_image = np.full((8, 12, 3), 30, dtype=np.uint8)
    right_image = np.full((8, 12, 3), 40, dtype=np.uint8)
    labels = np.full((8, 12), 6, dtype=np.uint8)
    depth_mm = np.full((8, 12), 10000, dtype=np.uint16)
    assert cv2.imwrite(str(images / "Scene01__clone__00001.jpg"), left_image)
    assert cv2.imwrite(str(right / "rgb_00001.jpg"), right_image)
    assert cv2.imwrite(str(masks / "Scene01__clone__00001.png"), labels)
    assert cv2.imwrite(str(depth / "depth_00001.png"), depth_mm)
    row = {"split": "train", "stem": "Scene01__clone__00001", "fx": 700.0,
           "baseline_m": 0.532725,
           "files": {"rgb_right": "Scene01/clone/frames/rgb/Camera_1/rgb_00001.jpg",
                     "depth": "Scene01/clone/frames/depth/Camera_0/depth_00001.png"}}
    left, right_rgb, disparity, mask = read_full_pair(tmp_path, row)
    assert left.shape == right_rgb.shape == (8, 12, 3)
    assert disparity.shape == mask.shape == (8, 12)
    assert np.all(mask == 6)
    assert left[0, 0, 0] == 30 and right_rgb[0, 0, 0] == 40
