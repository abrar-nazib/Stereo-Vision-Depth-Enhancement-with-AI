import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget

from semtilestereo.core import InferenceResult
from semtilestereo.results import CameraParameters, save_result
from semtilestereo.gui import ReviewWindow


class FakeViewport(QWidget):
    def __init__(self):
        super().__init__()
        self.points = np.empty((0, 3))
    def set_cloud(self, points, colors):
        self.points = points
    def reset_view(self): pass


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_load_filter_and_camera_rebuild(app, tmp_path):
    result = InferenceResult(np.full((2, 2), 2, np.float32),
                             np.zeros((14, 1, 1), np.float32),
                             np.array([[1, 2], [1, 2]], np.uint8),
                             np.ones((2, 2), np.float32),
                             np.full((2, 2, 3), 128, np.uint8))
    path = save_result(result, tmp_path, camera=CameraParameters(4, 4, 0, 0, 1), metadata={}, display_max=20)
    window = ReviewWindow(viewport=FakeViewport())
    window.stride_field.setText("1")
    window.open_result(path)
    assert set(window.class_checks) == {1, 2}
    assert len(window.viewport.points) == 4
    window.class_checks[1].setChecked(False)
    assert len(window.viewport.points) == 2
    previous = window.viewport.points.copy()
    window.fx_field.setText("500")
    window.apply_filters()
    assert not np.array_equal(previous, window.viewport.points)
    with pytest.raises(ValueError, match="1242"):
        window.use_vkitti_preset()
    window.close()


def test_worker_error_is_visible(app):
    window = ReviewWindow(viewport=FakeViewport())
    window.show_error("checkpoint missing")
    assert "checkpoint missing" in window.status.text()
    window.close()
