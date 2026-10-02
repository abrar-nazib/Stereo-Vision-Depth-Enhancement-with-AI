import io
import tarfile

import cv2
import numpy as np
import pytest

from experiments.vkitti2.semantic_data import (
    CLASS_NAMES,
    build_split_index,
    audit_converted_masks,
    decode_class_mask,
    prepare_dataset,
    reassign_validation_scene,
    remap_guardrail_to_ignore,
    convert_prepared_taxonomy,
    reshuffle_prepared_dataset,
)


def test_decode_all_classes_and_undefined():
    colors = np.array(
        [[
            (210, 0, 200), (90, 200, 255), (0, 199, 0), (90, 240, 0),
            (140, 140, 140), (100, 60, 100), (250, 100, 255),
            (255, 255, 0), (200, 200, 0), (255, 130, 0), (80, 80, 80),
            (160, 60, 60), (255, 127, 80), (0, 139, 139), (0, 0, 0),
        ]],
        dtype=np.uint8,
    )
    ids = decode_class_mask(colors, "fixture.png")
    assert len(CLASS_NAMES) == 14
    assert ids.dtype == np.uint8
    assert ids.tolist() == [list(range(14)) + [255]]


def test_unknown_color_fails_with_source():
    with pytest.raises(ValueError, match="fixture.png.*1, 2, 3"):
        decode_class_mask(np.array([[[1, 2, 3]]], dtype=np.uint8), "fixture.png")


def test_split_keeps_variations_together_and_pairs_modalities():
    members = [
        "Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg",
        "Scene01/clone/frames/classSegmentation/Camera_0/classgt_00000.png",
        "Scene18/rain/frames/rgb/Camera_0/rgb_00001.jpg",
        "Scene18/rain/frames/classSegmentation/Camera_0/classgt_00001.png",
        "Scene20/fog/frames/rgb/Camera_0/rgb_00002.jpg",
        "Scene20/fog/frames/classSegmentation/Camera_0/classgt_00002.png",
        "Scene20/fog/frames/rgb/Camera_1/rgb_00002.jpg",
    ]
    split = build_split_index(members)
    assert [len(split[k]) for k in ("train", "val", "test")] == [1, 1, 1]
    assert split["test"][0][0].startswith("Scene20/")


def test_missing_pair_fails():
    members = ["Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg"]
    with pytest.raises(ValueError, match="missing class mask"):
        build_split_index(members)


def test_prepare_writes_indexed_masks_and_support(tmp_path):
    rgb_tar = tmp_path / "rgb.tar"
    mask_tar = tmp_path / "class.tar"
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :] = (20, 30, 40)
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    mask[0, :] = (210, 0, 200)
    mask[1, 0] = (0, 0, 0)
    mask[1, 1:] = (90, 200, 255)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    ok, png = cv2.imencode(".png", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    assert ok
    entries = [
        (rgb_tar, "Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg", jpg.tobytes()),
        (mask_tar, "Scene01/clone/frames/classSegmentation/Camera_0/classgt_00000.png", png.tobytes()),
    ]
    for archive_path, name, payload in entries:
        with tarfile.open(archive_path, "w") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    audit = prepare_dataset(rgb_tar, mask_tar, tmp_path / "out")
    converted = cv2.imread(str(tmp_path / "out/masks/train/Scene01__clone__00000.png"), cv2.IMREAD_UNCHANGED)
    assert converted.tolist() == [[0, 0, 0], [255, 1, 1]]
    assert audit["split_counts"]["train"] == 1
    assert audit["pixel_counts"]["train"]["Terrain"] == 3
    assert audit["pixel_counts"]["train"]["Sky"] == 2
    assert audit["pixel_counts"]["train"]["Undefined"] == 1
    rescanned = audit_converted_masks(tmp_path / "out")
    assert rescanned["pixel_counts"]["train"]["Terrain"] == 3


def test_reassign_validation_scene_moves_both_image_and_mask(tmp_path):
    root = tmp_path
    for kind in ("images", "masks"):
        for split in ("train", "val"):
            (root / kind / split).mkdir(parents=True)
        (root / kind / "train" / f"Scene06__clone__00000.{ 'jpg' if kind == 'images' else 'png' }").write_bytes(b"x")
        (root / kind / "val" / f"Scene18__clone__00000.{ 'jpg' if kind == 'images' else 'png' }").write_bytes(b"x")
    reassign_validation_scene(root, "Scene06", "Scene18")
    assert (root / "images/val/Scene06__clone__00000.jpg").exists()
    assert (root / "masks/val/Scene06__clone__00000.png").exists()
    assert (root / "images/train/Scene18__clone__00000.jpg").exists()
    assert (root / "masks/train/Scene18__clone__00000.png").exists()


def test_guardrail_is_ignored_and_later_ids_become_contiguous():
    mask = np.array([[5, 6, 7, 8, 13, 255]], dtype=np.uint8)
    result = remap_guardrail_to_ignore(mask)
    assert result.tolist() == [[5, 255, 6, 7, 12, 255]]


def test_convert_prepared_taxonomy_rewrites_masks_and_yaml(tmp_path):
    import yaml

    root = tmp_path
    for split in ("train", "val", "test"):
        (root / "masks" / split).mkdir(parents=True)
        assert cv2.imwrite(str(root / "masks" / split / "x.png"),
                           np.array([[5, 6, 7, 255]], dtype=np.uint8))
    (root / "dataset.yaml").write_text(yaml.safe_dump({"names": {i: n for i, n in enumerate(CLASS_NAMES)}}))
    convert_prepared_taxonomy(root)
    assert cv2.imread(str(root / "masks/train/x.png"), cv2.IMREAD_UNCHANGED).tolist() == [[5, 255, 6, 255]]
    assert len(yaml.safe_load((root / "dataset.yaml").read_text())["names"]) == 13


def test_reshuffle_keeps_variations_of_source_frame_together(tmp_path):
    import json

    root = tmp_path
    stems = [f"Scene20__{variation}__{frame:05d}"
             for frame in range(10) for variation in ("clone", "rain")]
    for kind, suffix in (("images", "jpg"), ("masks", "png")):
        for split in ("train", "val", "test"):
            (root / kind / split).mkdir(parents=True)
        for stem in stems:
            (root / kind / "train" / f"{stem}.{suffix}").write_bytes(b"fixture")
    audit = reshuffle_prepared_dataset(root, seed=42)
    assert audit["split_counts"] == {"train": 16, "val": 2, "test": 2}
    for frame in range(10):
        locations = {split for split in ("train", "val", "test")
                     if (root / "images" / split / f"Scene20__clone__{frame:05d}.jpg").exists()}
        assert len(locations) == 1
        split = locations.pop()
        assert (root / "images" / split / f"Scene20__rain__{frame:05d}.jpg").exists()
        for variation in ("clone", "rain"):
            assert (root / "masks" / split / f"Scene20__{variation}__{frame:05d}.png").exists()
    manifest = json.loads((root / "split_manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["grouping"] == "scene+frame_across_variations"
    assert len(manifest["groups"]["train"]) == 8


def test_reshuffle_rejects_unpaired_input(tmp_path):
    root = tmp_path
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "train" / "Scene01__clone__00000.jpg").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unpaired"):
        reshuffle_prepared_dataset(root)
