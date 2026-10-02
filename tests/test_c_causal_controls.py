"""C2 causal controls must alter guidance during both training and evaluation."""

import random

import torch

from experiments.C.C_1_large_control.run import apply_guidance, select_head
from experiments.C.C_1_large_control.queue import queue_selection


def _signals():
    logits = torch.zeros(1, 14, 8, 12)
    logits[:, 0, 2, 3] = 8
    features = {"semantic_f8": torch.zeros(1, 256, 8, 12),
                "left_f8": torch.ones(1, 512, 8, 12)}
    features["semantic_f8"][:, 0, 2, 3] = 8
    return logits, features


def test_controls_keep_the_exact_c2_head_architecture():
    reference = select_head("C_2_feature_fusion")
    for arm in ("C_4_no_semantics", "C_5_misaligned_semantics"):
        head = select_head(arm)
        assert type(head) is type(reference)
        assert sum(p.numel() for p in head.parameters()) == sum(
            p.numel() for p in reference.parameters())


def test_no_semantics_zeros_only_semantic_inputs():
    logits, features = _signals()
    guided, transformed, offsets = apply_guidance(logits, features, "none", random.Random(7))
    assert torch.count_nonzero(guided) == 0
    assert torch.count_nonzero(transformed["semantic_f8"]) == 0
    torch.testing.assert_close(transformed["left_f8"], features["left_f8"])
    assert offsets == (0, 0)
    assert features["semantic_f8"].sum() == 8  # original frozen output is untouched


def test_misaligned_semantics_preserve_values_but_break_pixel_alignment():
    logits, features = _signals()
    guided, transformed, offsets = apply_guidance(
        logits, features, "misaligned", random.Random(7))
    assert offsets[0] != 0 and offsets[1] != 0
    assert torch.equal(guided, logits.roll(offsets, dims=(-2, -1)))
    assert torch.equal(transformed["semantic_f8"],
                       features["semantic_f8"].roll(offsets, dims=(-2, -1)))
    assert torch.equal(transformed["left_f8"], features["left_f8"])
    assert not torch.equal(guided, logits)
    again = apply_guidance(logits, features, "misaligned", random.Random(7))
    assert again[2] == offsets


def test_control_queue_is_two_sequential_arms_with_distinct_log_root():
    arms, folder = queue_selection(True)
    assert arms == ["C_4_no_semantics", "C_5_misaligned_semantics"]
    assert folder == "C_4_no_semantics"
