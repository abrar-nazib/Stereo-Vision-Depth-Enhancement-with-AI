"""B-series depth-only head invariants."""

import torch

from experiments.B.B_1_residual.head import ResidualHead
from experiments.B.B_2_confidence.head import ConfidenceHead
from experiments.B.B_3_classaware.head import ClassAwareHead


def test_all_heads_start_at_frozen_disparity_and_keep_semantics_unchanged():
    disparity = torch.full((1, 1, 64, 96), 12.0)
    logits = torch.randn(1, 14, 8, 12)
    left = torch.rand(1, 3, 64, 96)
    right = torch.rand_like(left)
    for head in (ResidualHead(), ConfidenceHead(), ClassAwareHead()):
        result = head(disparity, logits, left, right)
        assert result.shape == disparity.shape
        torch.testing.assert_close(result, disparity)


def test_heads_respond_to_semantics_after_residual_is_enabled():
    disparity = torch.full((1, 1, 64, 96), 12.0)
    left = torch.rand(1, 3, 64, 96)
    right = torch.rand_like(left)
    logits_a = torch.zeros(1, 14, 8, 12)
    logits_b = logits_a.clone()
    logits_a[:, 0] = 5
    logits_b[:, 1] = 5
    for head in (ResidualHead(), ConfidenceHead(), ClassAwareHead()):
        for parameter in head.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=0.02)
            else:
                torch.nn.init.constant_(parameter, 0.1)
        first = head(disparity, logits_a, left, right)
        second = head(disparity, logits_b, left, right)
        assert (first - second).abs().max().item() > 1e-6
