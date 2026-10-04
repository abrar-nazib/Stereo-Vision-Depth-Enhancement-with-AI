import torch

from experiments.H.model import StereoUpsampleHead, ARMS


def test_heads_preserve_full_resolution_and_baseline_identity():
    base = torch.ones(1, 1, 16, 24) * 8
    half = torch.ones(1, 1, 8, 12) * 4
    left = torch.rand(1, 3, 16, 24)
    for arm in ARMS:
        head = StereoUpsampleHead(arm)
        out = head(base, half, left)
        assert out.shape == base.shape
        assert torch.isfinite(out).all()
        if arm == "H0_a09_scratch":
            torch.testing.assert_close(out, base)


def test_convex_upsampler_cannot_extrapolate_beyond_half_scale_neighbors():
    half = torch.arange(8 * 12, dtype=torch.float32).reshape(1, 1, 8, 12)
    base = torch.zeros(1, 1, 16, 24)
    left = torch.rand(1, 3, 16, 24)
    head = StereoUpsampleHead("H2_strict_convex_scratch")
    out = head(base, half, left)
    neighbors = torch.nn.functional.unfold(half, kernel_size=3, padding=1).reshape(1, 9, 8, 12)
    neighbors = torch.nn.functional.interpolate(neighbors, scale_factor=2, mode="nearest") * 2
    assert torch.all(out >= neighbors.min(1, keepdim=True).values - 1e-5)
    assert torch.all(out <= neighbors.max(1, keepdim=True).values + 1e-5)
