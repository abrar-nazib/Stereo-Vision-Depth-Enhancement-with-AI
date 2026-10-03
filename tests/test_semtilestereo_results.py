import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from semtilestereo.core import InferenceResult
from semtilestereo.results import CameraParameters, load_bundle, save_result
from semtilestereo.visuals import colorize_disparity, segmentation_overlay


def sample():
    disparity = np.array([[0, 4, 8], [12, 16, 20]], np.float32)
    return InferenceResult(disparity, np.ones((14, 1, 2), np.float32),
                           np.array([[0, 1, 2], [3, 4, 5]], np.uint8),
                           np.full((2, 3), 0.8, np.float32),
                           np.full((2, 3, 3), 100, np.uint8))


def test_bundle_and_artifacts_roundtrip(tmp_path):
    result = sample()
    camera = CameraParameters(725, 725, 1, 1, 0.53)
    path = save_result(result, tmp_path, camera=camera,
                       metadata={"checkpoint": "test", "rectification": "none"}, display_max=32)
    assert path == tmp_path / "result.npz"
    loaded, metadata = load_bundle(path)
    for key in ("disparity_px", "semantic_logits", "class_id", "class_confidence", "left_rgb"):
        np.testing.assert_array_equal(getattr(loaded, key), getattr(result, key))
    assert metadata["schema_version"] == 1
    assert metadata["camera"]["baseline_m"] == 0.53
    assert metadata["checkpoint"] == "test"
    assert metadata["image_size"] == [3, 2]
    assert len(metadata["class_names"]) == 14
    np.testing.assert_array_equal(np.load(tmp_path / "disparity.npy"), result.disparity_px)
    np.testing.assert_array_equal(np.load(tmp_path / "semantic_logits.npy"), result.semantic_logits)
    assert cv2.imread(str(tmp_path / "disparity_color.png")).shape == (2, 3, 3)
    assert cv2.imread(str(tmp_path / "segmentation_overlay.png")).shape == (2, 3, 3)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"]))["schema_version"] == 1


def test_invalid_disparity_has_distinct_color():
    image = colorize_disparity(np.array([[0, 10]], np.float32), display_max=20)
    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[0, 1].tolist() != [0, 0, 0]
    np.testing.assert_array_equal(image, colorize_disparity(np.array([[0, 10]], np.float32), 20))


@pytest.mark.parametrize("camera", [
    CameraParameters(0, 1, 0, 0, 1), CameraParameters(float("nan"), 1, 0, 0, 1),
    CameraParameters(1, 1, 0, 0, -1),
])
def test_invalid_camera_rejected(tmp_path, camera):
    with pytest.raises(ValueError):
        save_result(sample(), tmp_path, camera=camera, metadata={}, display_max=20)


def test_invalid_range_and_shapes_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_result(sample(), tmp_path, camera=None, metadata={}, display_max=0)
    broken = replace(sample(), class_id=sample().class_id.astype(np.int32))
    with pytest.raises(ValueError):
        save_result(broken, tmp_path, camera=None, metadata={}, display_max=20)


def test_load_bundle_rejects_bad_schema_and_missing_keys(tmp_path):
    result = sample()
    path = save_result(result, tmp_path, camera=None, metadata={}, display_max=20)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["metadata_json"] = json.dumps({"schema_version": 99})
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_bundle(path)
    arrays.pop("class_id")
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="missing"):
        load_bundle(path)


def test_overlay_preserves_dimensions():
    result = sample()
    assert segmentation_overlay(result.left_rgb, result.class_id).shape == result.left_rgb.shape
