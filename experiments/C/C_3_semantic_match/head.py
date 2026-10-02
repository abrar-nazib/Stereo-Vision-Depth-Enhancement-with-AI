"""Multi-scale head plus right-view feature consistency at the current disparity."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.C.C_2_feature_fusion.head import FeatureFusionHead


class SemanticMatchHead(FeatureFusionHead):
    def __init__(self):
        super().__init__()
        self.match = nn.Sequential(nn.Conv2d(512, 32, 1), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.semantic_gate = nn.Conv2d(14, 32, 1)

    def _semantic_feature(self, semantic_f8: torch.Tensor,
                          left_f8: torch.Tensor, right_f8: torch.Tensor,
                          disparity: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if right_f8 is None:
            raise ValueError("right_f8 is required for semantic matching")
        b, _, h, w = left_f8.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=left_f8.device),
                                torch.arange(w, device=left_f8.device), indexing="ij")
        d = F.interpolate(disparity, size=(h, w), mode="bilinear", align_corners=False)
        shift = d * (w / disparity.shape[-1])
        gx = 2 * (xx[None].to(d.dtype) - shift[:, 0]) / max(w - 1, 1) - 1
        gy = 2 * yy[None].expand(b, -1, -1) / max(h - 1, 1) - 1
        warped = F.grid_sample(right_f8, torch.stack((gx, gy), -1),
                               align_corners=True, padding_mode="zeros")
        mismatch = (left_f8 - warped).abs()
        gate = torch.sigmoid(self.semantic_gate(logits.float().softmax(1)))
        return self.sem8(semantic_f8) + gate * self.match(mismatch)
