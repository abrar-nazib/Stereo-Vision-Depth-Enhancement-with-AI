from dataclasses import replace

import numpy as np
import pytest

from semtilestereo.core import InferenceResult
from semtilestereo.geometry import VKITTI_CAMERA, points_from_result
from semtilestereo.results import CameraParameters


def sample():
    return InferenceResult(
        disparity_px=np.array([[2, 4], [0, np.nan]], np.float32),
        semantic_logits=np.zeros((14, 1, 1), np.float32),
        class_id=np.array([[1, 2], [1, 2]], np.uint8),
        class_confidence=np.ones((2, 2), np.float32),
        left_rgb=np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 0], [0, 0, 0]]], np.uint8),
    )


def test_camera_geometry_and_filtering():
    camera = CameraParameters(4, 4, 0, 0, 1)
    points, colors, counts = points_from_result(sample(), camera, visible_classes={1, 2},
                                                  min_depth_m=0.1, max_depth_m=10,
                                                  stride=1, semantic_colors=False)
    np.testing.assert_allclose(points, [[0, 0, 2], [0.25, 0, 1]])
    np.testing.assert_allclose(colors, [[1, 0, 0], [0, 1, 0]])
    assert counts == {1: 1, 2: 1}
    points, colors, counts = points_from_result(sample(), camera, visible_classes={2},
                                                  min_depth_m=0.1, max_depth_m=10,
                                                  stride=1, semantic_colors=True)
    assert len(points) == 1 and counts == {1: 1, 2: 1}
    assert not np.allclose(colors[0], [0, 1, 0])


def test_depth_and_stride():
    camera = CameraParameters(4, 4, 0, 0, 1)
    result = replace(sample(), disparity_px=np.full((2, 2), 2, np.float32))
    points, _, counts = points_from_result(result, camera, visible_classes={1, 2},
                                            min_depth_m=1, max_depth_m=3, stride=2,
                                            semantic_colors=False)
    assert len(points) == 1 and counts == {1: 2, 2: 2}


@pytest.mark.parametrize("minimum,maximum,stride", [(0, 10, 0), (-1, 10, 1), (5, 3, 1)])
def test_bad_range_rejected(minimum, maximum, stride):
    with pytest.raises(ValueError):
        points_from_result(sample(), CameraParameters(4, 4, 0, 0, 1),
                           visible_classes={1}, min_depth_m=minimum,
                           max_depth_m=maximum, stride=stride, semantic_colors=False)


def test_vkitti_preset():
    assert VKITTI_CAMERA.fx == 725.0087 and VKITTI_CAMERA.baseline_m == 0.532725
