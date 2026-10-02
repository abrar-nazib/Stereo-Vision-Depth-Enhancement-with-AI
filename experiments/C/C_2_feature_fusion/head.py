"""Frozen multi-scale stereo/semantic features with a learned semantic gate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.C.C_1_large_control.head import ResidualBlock


class FeatureFusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.left8 = nn.Sequential(nn.Conv2d(512, 32, 1), nn.SiLU())
        self.sem8 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.SiLU())
        self.left4 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.SiLU())
        self.tile4 = nn.Sequential(nn.Conv2d(16, 16, 1), nn.SiLU())
        self.low = nn.Sequential(nn.Conv2d(79, 64, 3, padding=1), nn.SiLU(),
                                 ResidualBlock())
        self.gate = nn.Conv2d(79, 32, 1)
        self.high = nn.Sequential(nn.Conv2d(114, 64, 3, padding=1), nn.SiLU(),
                                  ResidualBlock(), ResidualBlock())
        self.delta = nn.Conv2d(64, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def _semantic_feature(self, semantic_f8: torch.Tensor, **_features) -> torch.Tensor:
        return self.sem8(semantic_f8)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor,
                left_f4: torch.Tensor, left_f8: torch.Tensor,
                tile_f4: torch.Tensor, semantic_f8: torch.Tensor,
                tile_conf4: torch.Tensor, right_f8: torch.Tensor | None = None) -> torch.Tensor:
        low_size = logits.shape[-2:]
        d8 = F.interpolate(disparity, size=low_size, mode="bilinear", align_corners=False)
        probs = logits.float().softmax(1)
        sem = self._semantic_feature(semantic_f8=semantic_f8, left_f8=left_f8,
                                     right_f8=right_f8, disparity=disparity, logits=logits)
        left8 = self.left8(left_f8)
        low_input = torch.cat((d8 / 192.0, probs, left8, sem), 1)
        low = self.low(low_input)
        gated_sem = sem * torch.sigmoid(self.gate(low_input))
        high_size = left_f4.shape[-2:]
        high = torch.cat((F.interpolate(low, size=high_size, mode="bilinear", align_corners=False),
                          self.left4(left_f4), self.tile4(tile_f4), tile_conf4,
                          F.interpolate(disparity, size=high_size, mode="bilinear",
                                        align_corners=False) / 192.0), 1)
        # The semantic gate enters the low-level representation before refinement.
        high = self.high(high + torch.cat((F.interpolate(gated_sem, size=high_size,
                                                        mode="bilinear", align_corners=False),
                                          torch.zeros_like(high[:, 32:])), 1))
        delta = 4.0 * torch.tanh(self.delta(high))
        return disparity + F.interpolate(delta, size=disparity.shape[-2:],
                                         mode="bilinear", align_corners=False)
