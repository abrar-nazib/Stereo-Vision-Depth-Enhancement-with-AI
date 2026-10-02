"""Capacity control: a larger output-only disparity residual head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class LargeControlHead(nn.Module):
    """No intermediate features: isolates capacity from new information."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(15, 64, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(3)))
        self.delta = nn.Conv2d(64, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor, **_features) -> torch.Tensor:
        size = logits.shape[-2:]
        low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        probs = logits.float().softmax(1)
        x = self.blocks(self.stem(torch.cat((low / 192.0, probs), 1)))
        delta = 4.0 * torch.tanh(self.delta(x))
        return disparity + F.interpolate(delta, size=disparity.shape[-2:],
                                         mode="bilinear", align_corners=False)
