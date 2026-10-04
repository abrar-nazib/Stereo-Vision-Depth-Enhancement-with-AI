import torch

from experiments.G.model import EdgeHead


def test_all_arms_start_from_frozen_baseline_and_train_independently():
    torch.manual_seed(7)
    base = torch.rand(2, 1, 32, 48) * 50
    half = torch.rand(2, 1, 16, 24) * 25
    left = torch.rand(2, 3, 32, 48)
    right = torch.rand(2, 3, 32, 48)
    for arm in ("G_1_rgb_residual", "G_2_convex", "G_3_stereo_correct"):
        head = EdgeHead(arm)
        prediction = head(base, half, left, right)
        assert prediction.shape == base.shape
        torch.testing.assert_close(prediction, base, atol=1e-6, rtol=0)
        prediction.mean().backward()
        assert any(p.grad is not None for p in head.parameters() if p.requires_grad)


def test_half_resolution_is_required_for_convex_arms():
    base = torch.ones(1, 1, 32, 48)
    image = torch.zeros(1, 3, 32, 48)
    head = EdgeHead("G_2_convex")
    try:
        head(base, torch.ones(1, 1, 12, 24), image, image)
    except ValueError as exc:
        assert "half-resolution" in str(exc)
    else:
        raise AssertionError("wrong half-resolution disparity was accepted")
