"""Small residual heads over frozen stereo and semantic predictions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MAPPED_ADE_IDS = (1, 2, 4, 6, 20, 83, 93, 102, 136)


def apply_semantic_residual(logits: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Modify only the mapped ADE channels, keeping other priors untouched."""
    if delta.shape[1] != len(MAPPED_ADE_IDS):
        raise ValueError("delta must have nine mapped channels")
    output = logits.clone()
    output[:, MAPPED_ADE_IDS] = output[:, MAPPED_ADE_IDS] + delta.to(logits.dtype)
    return output


class JointResidual(nn.Module):
    """Refine full-resolution disparity and stride-8 ADE logits jointly."""

    def __init__(self, channels: int = 32, semantic: bool = True):
        super().__init__()
        self.semantic = semantic
        self.body = nn.Sequential(
            nn.Conv2d(154, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(channels, 10 if semantic else 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, disparity: torch.Tensor, logits: torch.Tensor, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = logits.shape[-2:]
        low_disparity = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        low_image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        delta = self.out(self.body(torch.cat((low_disparity, low_image, logits), dim=1)))
        full_delta = F.interpolate(delta[:, :1], size=disparity.shape[-2:], mode="bilinear", align_corners=False)
        refined_disparity = disparity + full_delta
        refined_logits = apply_semantic_residual(logits, delta[:, 1:]) if self.semantic else logits
        return refined_disparity, refined_logits
