"""Small zero-initialized semantic disparity residual (DispSegNet style)."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(15, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.SiLU(),
            nn.Conv2d(32, 16, 1), nn.SiLU(),
        )
        self.delta = nn.Conv2d(16, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor,
                left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        size = logits.shape[-2:]
        d_low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        probabilities = logits.float().softmax(dim=1).to(d_low.dtype)
        features = self.features(torch.cat((d_low / 192.0, probabilities), dim=1))
        delta = 4.0 * torch.tanh(self.delta(features))
        return disparity + F.interpolate(delta, size=disparity.shape[-2:],
                                         mode="bilinear", align_corners=False)
