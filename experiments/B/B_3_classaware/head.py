"""Class-dependent disparity correction inspired by SGNet's residual module."""

import torch
from torch import nn
from torch.nn import functional as F


class ClassAwareHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classwise = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, groups=14), nn.SiLU(),
            nn.Conv2d(14, 32, 1), nn.SiLU(),
            nn.Conv2d(32, 14, 3, padding=1),
        )
        nn.init.zeros_(self.classwise[-1].weight)
        nn.init.zeros_(self.classwise[-1].bias)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor,
                left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        size = logits.shape[-2:]
        d_low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        probabilities = logits.float().softmax(dim=1).to(d_low.dtype)
        raw = self.classwise(probabilities * (d_low / 192.0))
        delta = 4.0 * torch.tanh((probabilities * raw).sum(dim=1, keepdim=True))
        return disparity + F.interpolate(delta, size=disparity.shape[-2:],
                                         mode="bilinear", align_corners=False)
