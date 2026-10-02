import torch

from experiments.D.D_1_semantic_cost.model import SemanticCostGate, semantic_agreement


def test_semantic_agreement_respects_left_minus_disparity_geometry():
    left = torch.zeros(1, 2, 1, 4)
    right = torch.zeros_like(left)
    left[0, 1, 0, 2] = 1
    right[0, 1, 0, 1] = 1
    agreement = semantic_agreement(left, right, 3)
    assert agreement.shape == (1, 1, 3, 1, 4)
    assert agreement[0, 0, 1, 0, 2].item() == 1
    assert agreement[0, 0, 0, 0, 2].item() == 0
    assert agreement[0, 0, 2, 0, 1].item() == 0


def test_zero_initialized_gate_is_identity_and_trainable():
    gate = SemanticCostGate()
    volume = torch.randn(1, 16, 4, 2, 5)
    left = torch.softmax(torch.randn(1, 14, 2, 5), 1)
    right = torch.softmax(torch.randn(1, 14, 2, 5), 1)
    weight = gate(volume, left, right)
    assert torch.allclose(weight, torch.ones_like(weight))
    loss = (weight * volume.mean(1, keepdim=True)).sum()
    loss.backward()
    assert gate.net[-1].weight.grad is not None
    assert gate.net[-1].weight.grad.abs().sum() > 0


def test_null_guidance_keeps_same_module_shapes():
    gate = SemanticCostGate()
    volume = torch.randn(1, 16, 3, 2, 5)
    probabilities = torch.softmax(torch.randn(1, 14, 2, 5), 1)
    assert gate(volume, probabilities, probabilities, use_semantics=False).shape == (1, 1, 3, 2, 5)
