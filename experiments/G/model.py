"""Small, independently trainable Foundation-inspired edge heads for frozen E3."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("G_1_rgb_residual", "G_2_convex", "G_3_stereo_correct")


def warp_right(right: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    b, _, h, w = right.shape
    y, x = torch.meshgrid(torch.arange(h, device=right.device),
                          torch.arange(w, device=right.device), indexing="ij")
    gx = ((x[None].float() - disparity[:, 0]) / max(w - 1, 1)) * 2 - 1
    gy = (y.float() / max(h - 1, 1) * 2 - 1)[None].expand(b, -1, -1)
    grid = torch.stack((gx, gy), -1)
    return F.grid_sample(right, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


class EdgeHead(nn.Module):
    """G1 RGB residual; G2 learned 3x3 half-scale convex; G3 adds stereo correction."""

    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.rgb = nn.Sequential(nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(24, 1, 3, padding=1))
        nn.init.zeros_(self.rgb[-1].weight)
        nn.init.zeros_(self.rgb[-1].bias)
        if arm != "G_1_rgb_residual":
            self.convex = nn.Sequential(nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(24, 9, 3, padding=1))
            self.blend = nn.Parameter(torch.tensor(0.0))
        if arm == "G_3_stereo_correct":
            self.stereo = nn.Sequential(nn.Conv2d(11, 24, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(24, 1, 3, padding=1))
            nn.init.zeros_(self.stereo[-1].weight)
            nn.init.zeros_(self.stereo[-1].bias)

    def forward(self, base: torch.Tensor, half: torch.Tensor,
                left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1] != 1 or left.shape != right.shape or left.shape[1] != 3:
            raise ValueError("expected B1HW disparity and matching B3HW images")
        if left.shape[-2:] != base.shape[-2:] or half.shape != (
                base.shape[0], 1, base.shape[-2] // 2, base.shape[-1] // 2):
            raise ValueError("expected matching images and exact half-resolution disparity")
        base = base.float()
        left = left.float()
        right = right.float()
        value = base + 2 * torch.tanh(self.rgb(torch.cat((base / 192, left), 1)))
        if self.arm != "G_1_rgb_residual":
            b, _, hh, ww = half.shape
            neighbors = F.unfold(half.float(), kernel_size=3, padding=1).view(b, 9, hh, ww)
            neighbors = F.interpolate(neighbors, size=base.shape[-2:], mode="nearest") * 2
            weights = self.convex(torch.cat((base / 192, left), 1)).softmax(1)
            candidate = (neighbors * weights).sum(1, keepdim=True)
            value = value + torch.tanh(self.blend) * (candidate - value)
        if self.arm == "G_3_stereo_correct":
            warped = warp_right(right, value.detach())
            signal = torch.cat((value / 192, left, warped, (left - warped).abs(), base / 192), 1)
            value = value + 4 * torch.tanh(self.stereo(signal))
        return value
