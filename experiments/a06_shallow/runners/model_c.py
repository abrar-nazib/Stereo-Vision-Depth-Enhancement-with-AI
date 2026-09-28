"""A06 — Step C: disparity branch off SHALLOW frozen YOLO26s-seg layers.

The seg model (neck/head) is never touched: encoder frozen, no seg labels used.
Tap points are backbone-only (all f=-1 linear chain, layers 0-6):

  layer 2 (C3k2) -> f4  128ch @ /4
  layer 4 (C3k2) -> f8  256ch @ /8
  layer 6 (C3k2) -> f16 256ch @ /16

Arms (TAP confg selects scales, always stride-ordered f4/f8[/f16]):
  C1: /4 only            (most texture-like)
  C2: /4 + /8            (SegStereo sharing depth)
  C3: /4 + /8 + /16      (where class semantics starts)

The RAFT-like head (vendored tile_propagate, byte-identical logic) self-sizes
from TAP_CHANNELS, so there is no adapter: frozen shallow features go straight
into TileInit + cost-lookup ConvGRU refinement + dual convex upsample.
Missing scales are synthesized by stride-2 convs on the deepest tapped map
(learned downsample, trainable) so every arm runs the same 1/16->1/8->1/4
refinement: C1 synthesizes /8,/16; C2 synthesizes /16.

Expected read-out: EPE slope across C1->C2->C3 tells us the sharing depth.
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

# layer index -> (name, channels, stride)
TAPS = {2: ("f4", 128, 4), 4: ("f8", 256, 8), 6: ("f16", 256, 16)}

ARMS = {
    "C1": (2,),
    "C2": (2, 4),
    "C3": (2, 4, 6),
}


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


class ShallowYoloEncoder(nn.Module):
    """Frozen YOLO backbone layers 0..max_tap; exposes tapped maps /255-normalized."""

    def __init__(self, tap_layers: tuple[int, ...], weights: Path = YOLO_WEIGHTS):
        super().__init__()
        self.tap_layers = tuple(tap_layers)
        trunk = YOLO(str(weights)).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(max(tap_layers) + 1)])
        for p in self.parameters():
            p.requires_grad = False
        self.out_channels = tuple(TAPS[i][1] for i in tap_layers)
        self.layers.eval()

    def train(self, mode: bool = True) -> "ShallowYoloEncoder":
        super().train(mode)
        self.layers.eval()
        return self

    def forward(self, x: torch.Tensor):
        v = x / 255.0
        saved = {}
        for i, layer in enumerate(self.layers):
            v = layer(v)
            if i in TAPS and i in self.tap_layers:
                saved[i] = v
        return [saved[i] for i in self.tap_layers]


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
class ShallowRaftConfig:
    tile_feat_ch: int = 16
    iters_16: int = 2
    iters_8: int = 3
    iters_4: int = 3
    init_max_disp: int = 24
    init_groups: int = 8
    refine_hidden: int = 48
    cost_lookup_half_range: int = 2
    cost_lookup_groups: int = 8


class ShallowYoloRaft(nn.Module):
    """RAFT-like head on shallow frozen YOLO taps (arm C1/C2/C3)."""

    def __init__(self, arm: str = "C2", cfg: ShallowRaftConfig | None = None):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        self.cfg = cfg or ShallowRaftConfig()
        taps = ARMS[arm]
        self.fnet = ShallowYoloEncoder(taps)
        self.encoder = self.fnet

        # Order tapped maps to (f4, f8, f16); synthesize missing deep scales.
        strides = [TAPS[i][2] for i in taps]
        chans = [TAPS[i][1] for i in taps]
        self.synths = nn.ModuleDict()
        if 8 not in strides:
            src = chans[strides.index(4)]
            self.synths["s8"] = nn.Sequential(
                nn.Conv2d(src, 128, 3, stride=2, padding=1, bias=False),
                _safe_gn(128), nn.SiLU(inplace=True))
            src = 128
        else:
            src = chans[strides.index(8)]
        if 16 not in strides:
            self.synths["s16"] = nn.Sequential(
                nn.Conv2d(src, 128, 3, stride=2, padding=1, bias=False),
                _safe_gn(128), nn.SiLU(inplace=True))
            ch16 = 128
        else:
            ch16 = chans[strides.index(16)]
        ch4 = chans[strides.index(4)]
        ch8 = chans[strides.index(8)] if 8 in strides else 128
        self.ch4, self.ch8, self.ch16 = ch4, ch8, ch16

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
        self.up_final_2_to_1 = ConvexUpsample(feat_ch=ch4, scale=2)

    def _pyramid(self, feats):
        by_stride = {}
        taps = ARMS[self.arm]
        for layer_idx, feat in zip(taps, feats):
            by_stride[TAPS[layer_idx][2]] = feat
        f4 = by_stride[4]
        if 8 in by_stride:
            f8 = by_stride[8]
        else:
            f8 = self.synths["s8"](by_stride[4])
        if 16 in by_stride:
            f16 = by_stride[16]
        else:
            f16 = self.synths["s16"](f8 if 8 not in by_stride else by_stride[8])
        return f4, f8, f16

    def train(self, mode: bool = True) -> "ShallowYoloRaft":
        super().train(mode)
        self.fnet.eval()
        return self

    def forward(self, left, right, aux: bool = False, semantic: bool = False):
        feats = self.fnet(torch.cat([left, right], dim=0))
        per_side = [f.chunk(2, dim=0) for f in feats]
        fL = [p[0] for p in per_side]
        fR = [p[1] for p in per_side]
        # Need f2 for final convex upsample: use tapped f4 downsampled? No —
        # reuse f4 (head's up_final_2_to_1 consumes ch4 features at /4 then /2->/1
        # via scale-2 convex steps from d_half). Keep A04b convention: two scale-2
        # convex steps driven by f4 features.
        fL4, fL8, fL16 = self._pyramid(fL)
        fR4, fR8, fR16 = self._pyramid(fR)

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
        fL2 = F.interpolate(fL4, size=d_half.shape[-2:], mode="bilinear", align_corners=False)
        d_full = self.up_final_2_to_1(d_half, fL2)

        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)
        return d_full, None, None
