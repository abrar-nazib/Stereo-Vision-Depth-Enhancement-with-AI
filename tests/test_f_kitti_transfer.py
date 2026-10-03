"""KITTI F-series input and metric contracts."""

from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
import torch

from experiments.F.kitti_data import list_pair_ids, read_pair
from experiments.F.evaluate_kitti import pair_metrics
from experiments.F.modal_kitti_t4 import worker_command


def _png(array: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", array)
    assert ok
    return encoded.tobytes()


def _archive(*, missing: str | None = None, bad_geometry: bool = False) -> ZipFile:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for number in (0, 1):
            stem = f"{number:06d}_10.png"
            rgb = np.array([[[4, 5, 6], [7, 8, 9]]], dtype=np.uint8)
            disparity = np.array([[0, 256, 512]] if bad_geometry else [[0, 256]],
                                 dtype=np.uint16)
            paths = {
                f"training/image_2/{stem}": _png(rgb),
                f"training/image_3/{stem}": _png(rgb),
                f"training/disp_occ_0/{stem}": _png(disparity),
                f"training/disp_noc_0/{stem}": _png(disparity),
            }
            for path, payload in paths.items():
                if path != missing:
                    archive.writestr(path, payload)
    buffer.seek(0)
    return ZipFile(buffer)


def test_kitti_index_requires_both_views_and_both_gt_masks():
    with _archive() as archive:
        assert list_pair_ids(archive) == ["000000_10", "000001_10"]
    with _archive(missing="training/image_3/000001_10.png") as archive:
        with pytest.raises(ValueError, match="missing"):
            list_pair_ids(archive)


def test_kitti_disparity_decodes_png_units_and_left_rgb():
    with _archive() as archive:
        left, right, occ, noc = read_pair(archive, "000000_10")
    assert left.shape == right.shape == (1, 2, 3)
    assert left[0, 0].tolist() == [6, 5, 4]
    assert occ.dtype == noc.dtype == np.float32
    assert occ[0].tolist() == [0.0, 1.0]


def test_kitti_pair_rejects_geometry_mismatch():
    with _archive(bad_geometry=True) as archive:
        with pytest.raises(ValueError, match="shape"):
            read_pair(archive, "000000_10")


def test_metric_mask_excludes_zero_and_out_of_model_range():
    prediction = torch.tensor([[[[0.0, 1.5, 190.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 1.0, 192.0, 5.0]]]])
    result = pair_metrics(prediction, target)
    assert result["gt_labeled_pixels"] == 3
    assert result["evaluated_pixels"] == 2
    assert result["out_of_range_pixels"] == 1
    assert result["epe"] == pytest.approx(2.75)


def test_f1_command_has_no_semantic_head_and_f2_f3_use_selected_heads(tmp_path):
    common = dict(archive=tmp_path / "kitti.zip", stereo=tmp_path / "stereo.pth",
                  semantic=tmp_path / "semantic.pt", encoder=tmp_path / "ade.pt",
                  head=tmp_path / "e3.pth", output=tmp_path / "out")
    f1 = worker_command("F1", **common)
    f2 = worker_command("F2", **common)
    f3 = worker_command("F3", **{**common, "head": tmp_path / "e4.pth"})
    assert f1[f1.index("--arm") + 1] == "F1"
    assert "--head" not in f1
    assert f2[f2.index("--arm") + 1] == "F2"
    assert f2[f2.index("--head") + 1] == str(tmp_path / "e3.pth")
    assert f3[f3.index("--arm") + 1] == "F3"
    assert f3[f3.index("--head") + 1] == str(tmp_path / "e4.pth")
