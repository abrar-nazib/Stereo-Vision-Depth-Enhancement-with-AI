from pathlib import Path

import numpy as np
import pytest
import torch

from semtilestereo.core import ModelPaths, infer_pair, load_model


class FakeModel:
    def __init__(self):
        self.inputs = None

    def __call__(self, left, right):
        self.inputs = (left, right)
        _, _, height, width = left.shape
        disparity = torch.full((1, 1, height, width), 7.0)
        logits = torch.zeros((1, 14, height // 8, width // 8))
        logits[:, 3] = 8
        return disparity, logits


def test_infer_pair_preserves_native_pixels_and_logits():
    model = FakeModel()
    left = np.zeros((37, 65, 3), dtype=np.uint8)
    left[0, 0] = (11, 22, 33)
    right = np.ones_like(left)
    result = infer_pair(model, left, right, "cpu")
    assert result.disparity_px.shape == (37, 65)
    assert result.class_id.shape == (37, 65)
    assert result.class_confidence.shape == (37, 65)
    assert result.semantic_logits.shape == (14, 8, 12)
    assert result.left_rgb.shape == (37, 65, 3)
    np.testing.assert_array_equal(result.left_rgb[0, 0], [33, 22, 11])
    assert model.inputs[0].shape == (1, 3, 64, 96)
    assert model.inputs[1].shape == (1, 3, 64, 96)
    assert model.inputs[0][0, :, 0, 0].tolist() == [33, 22, 11]
    assert np.all(result.disparity_px == 7)
    assert np.all(result.class_id == 3)
    assert np.all(result.class_confidence > 0.9)


@pytest.mark.parametrize("left,right", [
    (np.zeros((4, 5, 3), np.uint8), np.zeros((5, 5, 3), np.uint8)),
    (np.zeros((4, 5), np.uint8), np.zeros((4, 5, 3), np.uint8)),
    (np.zeros((4, 5, 4), np.uint8), np.zeros((4, 5, 4), np.uint8)),
])
def test_infer_pair_rejects_invalid_pairs(left, right):
    with pytest.raises(ValueError):
        infer_pair(FakeModel(), left, right, "cpu")


def test_load_model_reports_missing_checkpoint(tmp_path: Path):
    paths = ModelPaths(*(tmp_path / name for name in ("a.pth", "b.pth", "c.pt", "d.pt")))
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_model(paths, "cpu")
