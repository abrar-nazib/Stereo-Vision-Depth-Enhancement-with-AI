"""B-series split and VKITTI-14 protocol tests."""

import json
from pathlib import Path

import numpy as np

from experiments.B.B_0_fused_baseline.run import split_rows, label_mask
from experiments.vkitti2.semantic_data import CLASS_COLORS


MANIFEST = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation1000/manifest.json")


def test_split_is_800_100_100_without_source_frame_leakage():
    rows = json.loads(MANIFEST.read_text())
    train, val, test = split_rows(rows, seed=42)
    assert [len(x) for x in (train, val, test)] == [800, 100, 100]
    keys = [{(r["scene"], r["frame"]) for r in group}
            for group in (train, val, test)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert {r["scene"] for r in val + test} == {"Scene20"}


def test_label_mask_decodes_all_fourteen_classes_and_undefined():
    rgb = np.array([list(CLASS_COLORS) + [(0, 0, 0)]], dtype=np.uint8)
    actual = label_mask(rgb)
    np.testing.assert_array_equal(actual, np.array([list(range(14)) + [255]], dtype=np.uint8))
