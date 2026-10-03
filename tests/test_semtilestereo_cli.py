from pathlib import Path

import cv2
import numpy as np
import pytest

from semtilestereo.camera import load_rectification, split_stereo
from semtilestereo.infer import parse_args, run_live, should_stop


def test_parse_modes():
    live = parse_args(["live"])
    assert live.camera == "/dev/video2" and live.width == 1280 and live.height == 480
    assert live.left_view == "left" and live.rectification is None
    pair = parse_args(["pair", "left.png", "right.png", "out"])
    assert pair.left == Path("left.png") and pair.output == Path("out")


def test_split_and_swap():
    frame = np.zeros((3, 8, 3), np.uint8)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    left, right = split_stereo(frame, "left")
    assert left.shape == (3, 4, 3) and left[0, 0, 0] == 1 and right[0, 0, 0] == 2
    left, right = split_stereo(frame, "right")
    assert left[0, 0, 0] == 2 and right[0, 0, 0] == 1
    with pytest.raises(ValueError, match="even"):
        split_stereo(np.zeros((3, 7, 3), np.uint8), "left")


def test_rectification_rejects_wrong_resolution(tmp_path):
    path = tmp_path / "maps.xml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for key in ("stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y"):
        fs.write(key, np.zeros((3, 4), np.float32))
    fs.release()
    with pytest.raises(ValueError, match="shape"):
        load_rectification(path, (5, 4))


@pytest.mark.parametrize("key,disp,seg,expected", [
    (ord("q"), True, True, True), (27, True, True, True),
    (-1, False, True, True), (-1, True, False, True),
    (-1, True, True, False),
])
def test_stop_conditions(key, disp, seg, expected):
    assert should_stop(key, disp, seg) is expected


def test_live_releases_capture_on_failure(monkeypatch):
    class Capture:
        released = False
        def isOpened(self): return True
        def set(self, *_): return True
        def get(self, key): return 1280 if key == cv2.CAP_PROP_FRAME_WIDTH else 480
        def read(self): return False, None
        def release(self): self.released = True
    capture = Capture()
    destroyed = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))
    monkeypatch.setattr("semtilestereo.infer.load_model", lambda *_: object())
    with pytest.raises(RuntimeError, match="read"):
        run_live(parse_args(["live"]))
    assert capture.released and destroyed
