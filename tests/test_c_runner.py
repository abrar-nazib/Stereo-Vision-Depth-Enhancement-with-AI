"""C-series run configuration and metric contracts."""

import numpy as np

from experiments.C.C_1_large_control.run import semantic_metrics, select_head


def test_semantic_miou_excludes_gt_absent_predicted_class():
    pred = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    result = semantic_metrics([(pred, gt)], classes=14)
    assert result["present_classes"] == [0]
    assert result["class_iou"]["0"] == 0.25
    assert result["miou"] == 0.25
    assert result["pixel_accuracy"] == 0.25


def test_all_three_arms_have_distinct_heads():
    assert [select_head(name).__class__.__name__ for name in
            ("C_1_large_control", "C_2_feature_fusion", "C_3_semantic_match")] == [
                "LargeControlHead", "FeatureFusionHead", "SemanticMatchHead"]
