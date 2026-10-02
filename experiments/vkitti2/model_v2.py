"""Small confidence-gated late-fusion heads over two frozen predictors."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.vkitti2.model import apply_semantic_residual


def spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    """Absolute horizontal and vertical finite differences (same H×W)."""
    dx = F.pad((x[..., 1:] - x[..., :-1]).abs(), (1, 0, 0, 0))
    dy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 1, 0))
    return dx + dy


def warp_right_to_left(right_low: torch.Tensor, disparity_full_pixels: torch.Tensor,
                       full_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample right(x−d) at low resolution, preserving full-pixel disparity units."""
    batch, _, height, width = right_low.shape
    scale_x = full_width / width
    y, x = torch.meshgrid(torch.arange(height, device=right_low.device),
                          torch.arange(width, device=right_low.device), indexing="ij")
    source_x = x[None, None].to(disparity_full_pixels.dtype) - disparity_full_pixels / scale_x
    valid = (source_x >= 0) & (source_x <= width - 1)
    gx = 2 * source_x[:, 0] / max(width - 1, 1) - 1
    gy = 2 * y[None].expand(batch, -1, -1) / max(height - 1, 1) - 1
    grid = torch.stack((gx, gy), dim=-1).to(right_low.dtype)
    warped = F.grid_sample(right_low, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)
    return warped, valid


def _zero_last(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class GatedResidual(nn.Module):
    """Two task-specific gated outputs, optional right-view and full-res cues."""

    def __init__(self, use_warp: bool = False, use_edge: bool = False):
        super().__init__()
        self.use_warp = use_warp
        self.use_edge = use_edge
        self.semantic_embed = nn.Sequential(nn.Conv2d(150, 16, 1),
                                            nn.GroupNorm(4, 16), nn.SiLU())
        self.geometry_embed = nn.Sequential(nn.Conv2d(8 if use_warp else 6, 16, 3, padding=1),
                                            nn.GroupNorm(4, 16), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1),
                                  nn.GroupNorm(8, 32), nn.SiLU())
        self.disparity_delta = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1),
                                             nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
        self.semantic_delta = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1),
                                            nn.SiLU(), nn.Conv2d(16, 9, 3, padding=1))
        self.disparity_gate = nn.Conv2d(32, 1, 1)
        self.semantic_gate = nn.Conv2d(32, 9, 1)
        _zero_last(self.disparity_delta)
        _zero_last(self.semantic_delta)
        if use_edge:
            self.edge_refine = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, groups=4),
                                             nn.Conv2d(4, 16, 1), nn.SiLU(),
                                             nn.Conv2d(16, 1, 3, padding=1))
            _zero_last(self.edge_refine)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor,
                image: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = logits.shape[-2:]
        d_low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        left_low = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        # Explicit normalization prevents raw 0–192 px disparity from dominating.
        d_norm = d_low / 192.0
        prob = logits.float().softmax(dim=1)
        entropy = -(prob * prob.clamp_min(1e-7).log()).sum(dim=1, keepdim=True) / math.log(150)
        geometry = [d_norm, left_low, spatial_gradient(d_norm), entropy.to(d_norm.dtype)]
        if self.use_warp:
            right_low = F.interpolate(right, size=size, mode="bilinear", align_corners=False)
            warped, valid = warp_right_to_left(right_low, d_low, image.shape[-1])
            photo_error = (left_low - warped).abs().mean(dim=1, keepdim=True)
            geometry += [photo_error * valid, valid.to(photo_error.dtype)]
        feature = self.fuse(torch.cat((self.geometry_embed(torch.cat(geometry, dim=1)),
                                       self.semantic_embed(logits)), dim=1))
        delta_d = torch.sigmoid(self.disparity_gate(feature)) * self.disparity_delta(feature)
        delta_s = torch.sigmoid(self.semantic_gate(feature)) * self.semantic_delta(feature)
        delta_full = F.interpolate(delta_d, size=disparity.shape[-2:], mode="bilinear",
                                   align_corners=False)
        if self.use_edge:
            gray = image.mean(dim=1, keepdim=True)
            rgb_edge = spatial_gradient(gray)
            semantic_edge_low = spatial_gradient(prob).sum(dim=1, keepdim=True) * 0.5
            sem_edge = F.interpolate(semantic_edge_low, size=disparity.shape[-2:],
                                     mode="bilinear", align_corners=False)
            edge_weight = (2 * rgb_edge + 2 * sem_edge).clamp(0, 1)
            edge_input = torch.cat((delta_full, rgb_edge, sem_edge, disparity / 192.0), dim=1)
            delta_full = delta_full + edge_weight * self.edge_refine(edge_input)
        return disparity + delta_full, apply_semantic_residual(logits, delta_s)
