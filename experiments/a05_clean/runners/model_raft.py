"""A04b — Selective-RAFT-style iterative head on frozen ADE20K YOLO26s encoder.

Head is the user's own StereoLite_raftlike design (vendored tile_propagate.py,
byte-identical logic): TileInit at 1/16 + cost-lookup ConvGRU refinement at
1/16, 1/8, 1/4 (2+3+3 iters) + dual ConvexUpsample to full res. No ghost arm.

Encoder: frozen YOLO26s-sem ADE20K, layers 0-6 (f2/f4/f8/f16 =
32/128/256/256 ch). The head self-sizes from out_channels, so there is no
adapter at all — frozen semantic features go straight into init/refine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from tile_propagate import TileInit, TileRefineRAFTLike, TileUpsample

ROOT = Path(__file__).resolve().parents[3]

YOLO_WEIGHTS = ROOT / "models/segmentation/yolo26s-sem-ade20k.pt"
YOLO_OUT_CHANNELS = (32, 128, 256, 256)


def _safe_gn(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = max_groups
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def native_crop(left, right, disparity, *, height: int, width: int, top: int, left_offset: int):
    return (
        left[top:top + height, left_offset:left_offset + width],
        right[top:top + height, left_offset:left_offset + width],
        disparity[top:top + height, left_offset:left_offset + width],
    )


class FrozenYoloEncoder(nn.Module):
    """Truncated YOLO26s-sem: layers 0-6 chain, frozen, (f2,f4,f8,f16)."""

    def __init__(self, weights: Path = YOLO_WEIGHTS):
        super().__init__()
        trunk = YOLO(str(weights)).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(7)])
        for p in self.parameters():
            p.requires_grad = False
        self.out_channels = YOLO_OUT_CHANNELS
        self.layers.eval()

    def train(self, mode: bool = True) -> "FrozenYoloEncoder":
        super().train(mode)
        self.layers.eval()
        return self

    def forward(self, x: torch.Tensor):
        f2 = self.layers[0](x / 255.0)
        x4_ = self.layers[1](f2)
        f4 = self.layers[2](x4_)
        x8_ = self.layers[3](f4)
        f8 = self.layers[4](x8_)
        x16 = self.layers[5](f8)
        f16 = self.layers[6](x16)
        return f2, f4, f8, f16


class ConvexUpsample(nn.Module):
    def __init__(self, feat_ch: int, scale: int = 2, hidden: int = 48):
        super().__init__()
        self.scale = scale
        self.mask = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 3, padding=1, bias=False),
            _safe_gn(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 9 * scale * scale, 1),
        )

    def forward(self, disp, feat):
        B, _, H, W = disp.shape
        s = self.scale
        m = self.mask(feat).view(B, 1, 9, s, s, H, W).softmax(dim=2)
        up = F.unfold(disp * s, kernel_size=3, padding=1)
        up = up.view(B, 1, 9, 1, 1, H, W)
        out = (m * up).sum(dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(B, 1, s * H, s * W)


@dataclass
class SelectiveRaftConfig:
    base_ch: int = 24
    tile_feat_ch: int = 16
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    cost_lookup_half_range: int = 2
    cost_lookup_groups: int = 8


class YoloSelectiveRaft(nn.Module):
    """Frozen ADE20K YOLO encoder + RAFT-like iterative tile head."""

    def __init__(self, cfg: SelectiveRaftConfig | None = None):
        super().__init__()
        self.cfg = cfg or SelectiveRaftConfig()
        self.fnet = FrozenYoloEncoder()
        self.encoder = self.fnet

        ch2, ch4, ch8, ch16 = self.fnet.out_channels

        g = self.cfg.init_groups
        while ch16 % g != 0 and g > 1:
            g -= 1
        self.init_tile = TileInit(feat_ch=ch16,
                                   max_disp=self.cfg.init_max_disp,
                                   groups=g,
                                   feat_out=self.cfg.tile_feat_ch)

        def make_refine(feat_ch):
            return TileRefineRAFTLike(
                feat_ch=feat_ch, tile_feat_ch=self.cfg.tile_feat_ch,
                hidden=self.cfg.refine_hidden,
                half_range=self.cfg.cost_lookup_half_range,
                groups=self.cfg.cost_lookup_groups)

        self.refine_16 = make_refine(ch16)
        self.refine_8 = make_refine(ch8)
        self.refine_4 = make_refine(ch4)
        self.up_16_to_8 = TileUpsample(scale_factor=2)
        self.up_8_to_4 = TileUpsample(scale_factor=2)
        self.up_final_4_to_2 = ConvexUpsample(feat_ch=ch4, scale=2)
        self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch2, scale=2)

    def train(self, mode: bool = True) -> "YoloSelectiveRaft":
        super().train(mode)
        self.fnet.eval()
        return self

    def forward(self, left, right, aux: bool = False, semantic: bool = False):
        feats = self.fnet(torch.cat([left, right], dim=0))
        fL2, fR2 = feats[0].chunk(2, dim=0)
        fL4, fR4 = feats[1].chunk(2, dim=0)
        fL8, fR8 = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        tile = self.init_tile(fL16, fR16)
        for _ in range(self.cfg.iters_16):
            tile = self.refine_16(tile, fL16, fR16)

        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        for _ in range(self.cfg.iters_8):
            tile = self.refine_8(tile, fL8, fR8)

        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        for _ in range(self.cfg.iters_4):
            tile = self.refine_4(tile, fL4, fR4)

        d_half = self.up_final_4_to_2(tile.d, fL4)
        d_full = self.up_final_2_to_1(d_half, fL2)

        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)
        return d_full, None, None
