"""B-series training metric and crop-validity checks."""

import torch

from experiments.B.B_0_fused_baseline.run import SemanticMeter14, crop_valid_mask


def test_crop_validity_excludes_correspondences_outside_right_crop():
    disparity = torch.tensor([[[[2.0, 2.0, 2.0, 2.0]]]])
    assert crop_valid_mask(disparity).tolist() == [[[[False, False, True, True]]]]


def test_semantic_meter_ignores_undefined_and_counts_fourteen_classes():
    meter = SemanticMeter14()
    logits = torch.zeros(1, 14, 1, 3)
    logits[:, 0, 0, 0] = 2
    logits[:, 1, 0, 1] = 2
    labels = torch.tensor([[[[0, 2, 255]]]])
    meter.add(logits, labels)
    result = meter.result()
    assert result["pixel_accuracy"] == 0.5
    assert result["class_iou"]["0"] == 1.0
    assert result["class_iou"]["1"] == 0.0
    assert result["class_iou"]["2"] == 0.0
