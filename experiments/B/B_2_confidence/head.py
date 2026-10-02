"""Use semantic entropy and stereo photometric consistency as correction cues."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(18, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.SiLU(),
            nn.Conv2d(32, 16, 1), nn.SiLU(),
        )
        self.delta = nn.Conv2d(16, 1, 3, padding=1)
        self.gate = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    @staticmethod
    def _photometric(left: torch.Tensor, right: torch.Tensor,
                     disparity: torch.Tensor, size: tuple[int, int]):
        low_l = F.interpolate(left, size=size, mode="bilinear", align_corners=False)
        low_r = F.interpolate(right, size=size, mode="bilinear", align_corners=False)
        d_low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        height, width = size
        y, x = torch.meshgrid(torch.arange(height, device=left.device),
                              torch.arange(width, device=left.device), indexing="ij")
        source_x = x[None, None].to(d_low.dtype) - d_low / (left.shape[-1] / width)
        valid = (source_x >= 0) & (source_x <= width - 1)
        gx = 2 * source_x[:, 0] / max(width - 1, 1) - 1
        gy = 2 * y[None].expand(left.shape[0], -1, -1) / max(height - 1, 1) - 1
        grid = torch.stack((gx, gy), dim=-1).to(low_r.dtype)
        warped = F.grid_sample(low_r, grid, mode="bilinear", padding_mode="zeros",
                               align_corners=True)
        photo = (low_l - warped).abs().mean(dim=1, keepdim=True) * valid
        return photo, valid.to(photo.dtype)

    def forward(self, disparity: torch.Tensor, logits: torch.Tensor,
                left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        size = logits.shape[-2:]
        d_low = F.interpolate(disparity, size=size, mode="bilinear", align_corners=False)
        probabilities = logits.float().softmax(dim=1).to(d_low.dtype)
        entropy = -(probabilities.float() * probabilities.float().clamp_min(1e-7).log()
                    ).sum(dim=1, keepdim=True).to(d_low.dtype) / math.log(14)
        photo, valid = self._photometric(left, right, disparity, size)
        feature = self.features(torch.cat((d_low / 192.0, probabilities,
                                           entropy, photo, valid), dim=1))
        delta = 4.0 * torch.tanh(self.delta(feature)) * torch.sigmoid(self.gate(feature))
        return disparity + F.interpolate(delta, size=disparity.shape[-2:],
                                         mode="bilinear", align_corners=False)
