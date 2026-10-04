"""Stereo-only A09 output heads; the YOLO encoder never receives gradients."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("H0_a09_scratch", "H1_rgb_guided_scratch", "H2_strict_convex_scratch")


class StereoUpsampleHead(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        if arm == "H1_rgb_guided_scratch":
            self.rgb = nn.Sequential(nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(24, 1, 3, padding=1))
            nn.init.zeros_(self.rgb[-1].weight)
            nn.init.zeros_(self.rgb[-1].bias)
        elif arm == "H2_strict_convex_scratch":
            self.mask = nn.Sequential(nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(24, 9, 3, padding=1))
            nn.init.zeros_(self.mask[-1].weight)
            nn.init.zeros_(self.mask[-1].bias)
            with torch.no_grad():
                self.mask[-1].bias[4] = 4.0

    def forward(self, base: torch.Tensor, half: torch.Tensor,
                left_normalized: torch.Tensor) -> torch.Tensor:
        if base.ndim != 4 or base.shape[1] != 1 or left_normalized.shape != (
                base.shape[0], 3, *base.shape[-2:]):
            raise ValueError("expected full-resolution disparity and RGB")
        if half.shape != (base.shape[0], 1, base.shape[-2] // 2, base.shape[-1] // 2):
            raise ValueError("expected exactly half-resolution tile disparity")
        if self.arm == "H0_a09_scratch":
            return base
        guide = torch.cat((base.float() / 192, left_normalized.float()), dim=1)
        if self.arm == "H1_rgb_guided_scratch":
            return base.float() + 4 * torch.tanh(self.rgb(guide))
        b, _, hh, ww = half.shape
        neighbors = F.unfold(F.pad(half.float(), (1, 1, 1, 1), mode="replicate"),
                             kernel_size=3).view(b, 9, hh, ww)
        neighbors = F.interpolate(neighbors, size=base.shape[-2:], mode="nearest") * 2
        weights = self.mask(guide).softmax(dim=1)
        return (neighbors * weights).sum(dim=1, keepdim=True)
