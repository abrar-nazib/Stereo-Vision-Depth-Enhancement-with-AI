"""Contracts for the C-series depth-only ablations."""

import torch

from experiments.C.C_1_large_control.head import LargeControlHead
from experiments.C.C_2_feature_fusion.head import FeatureFusionHead
from experiments.C.C_3_semantic_match.head import SemanticMatchHead


def _inputs():
    return {
        "disparity": torch.full((1, 1, 64, 96), 12.0),
        "logits": torch.randn(1, 14, 8, 12),
        "left_f4": torch.randn(1, 256, 16, 24),
        "left_f8": torch.randn(1, 512, 8, 12),
        "right_f8": torch.randn(1, 512, 8, 12),
        "tile_f4": torch.randn(1, 16, 16, 24),
        "semantic_f8": torch.randn(1, 256, 8, 12),
        "tile_conf4": torch.rand(1, 1, 16, 24),
    }


def test_new_heads_start_as_identity_and_preserve_shape():
    inputs = _inputs()
    for head in (LargeControlHead(), FeatureFusionHead(), SemanticMatchHead()):
        actual = head(**inputs)
        assert actual.shape == inputs["disparity"].shape
        torch.testing.assert_close(actual, inputs["disparity"])
        assert 100_000 <= sum(p.numel() for p in head.parameters()) <= 1_500_000


def test_semantic_match_uses_right_feature_after_training_signal():
    inputs = _inputs()
    head = SemanticMatchHead()
    with torch.no_grad():
        head.delta.weight.normal_(0, 0.1)
    a = head(**inputs)
    b = head(**{**inputs, "right_f8": torch.zeros_like(inputs["right_f8"])})
    assert not torch.allclose(a, b)
