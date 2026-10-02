"""A07v2 — fusion arms on the v2_hitnet host (the head that reproduces 0.84).

Host: A03's vendored StereoLite_v2_hitnet (hitnet_head package) + frozen
ADE20K YOLO26s encoder — the exact configuration that produced A03's 5.66 arm,
with the head that reproduces the user's 0.84 20-pair overfit.

Arms inject at TileInit (identical mechanism to model_a07.py, reviewed):
  E  CoEx GCE: gate = sigmoid(1x1(gL16→16)) on agg[:3](cv) output, broadcast /D
  V  SGNet:    conf = VetoConf(cv, fL16, fR16); conf ⊙ agg[:3](cv) → agg[3:]
  D  GGEV:     DynAgg rewrites cv pre-agg; GRU feat = feat_head(fL16)+h0(proj)
  B  no fusion (baseline with the same true multi-scale loss — isolates the
     fusion effect from the loss effect vs A03's 5.66 smooth_l1 arm)

Matching + guidance both come from the SAME frozen f16 map (layer 6): per
SGNet the disp-branch and semantic-branch correlations come from different
heads; with one shared frozen encoder the semantic correlation is built from
the same features that build the volume — the arm tests whether per-pixel
gating of the correlation helps even when the features are shared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

A03_DIR = Path(__file__).resolve().parents[2] / "hitnet_a03"
sys.path.insert(0, str(A03_DIR))

from hitnet_head.model import StereoLite, StereoLiteConfig  # noqa: E402
from hitnet_head.tile_propagate import TileState  # noqa: E402

from model_a07 import ExciteGate, VetoConf, DynAgg  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ADE20K = ROOT / "models/segmentation/yolo26s-sem-ade20k.pt"


class FusionStereoLite(StereoLite):
    """StereoLite with a fusion arm injected at the 1/16 init volume."""

    def __init__(self, arm: str, cfg: StereoLiteConfig | None = None,
                 encoder: str | Path | None = None):
        cfg = cfg or StereoLiteConfig(backbone=str(encoder or ADE20K), freeze_encoder=True)
        super().__init__(cfg)
        self.arm = arm
        ch16 = self.fnet.out_channels[3]
        self.arms = set(arm) if arm != "B" else set()
        for a in self.arms:
            assert a in ("E", "V", "D"), a
        if "E" in self.arms:
            self.fuse_e = ExciteGate(ch16, 16)
        if "V" in self.arms:
            self.fuse_v = VetoConf(ch16)
        if "D" in self.arms:
            self.fuse_d = DynAgg(self.init_tile.groups, 24)
            self.h0 = nn.Sequential(nn.Conv2d(24, self.cfg.tile_feat_ch, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, self.cfg.tile_feat_ch), nn.SiLU(inplace=True))
            self.guide_proj = nn.Conv2d(ch16, 24, 1)
        if arm != "B" and not self.arms:
            raise ValueError(arm)

    def forward(self, left, right, aux: bool = False, semantic: bool = False):
        feats = self.fnet(torch.cat([left, right], dim=0))
        fL2, fR2 = feats[0].chunk(2, dim=0)
        fL4, fR4 = feats[1].chunk(2, dim=0)
        fL8, fR8 = feats[2].chunk(2, dim=0)
        fL16, fR16 = feats[3].chunk(2, dim=0)

        tile = self._init_with_fusion(fL16, fR16)
        t16_init = tile
        tile = self.prop_16(tile, fL16, fR16)
        t16_prop = tile

        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        tile = self.prop_8(tile, fL8, fR8)
        t8_prop = tile

        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        tile = self.prop_4(tile, fL4, fR4)
        t4_prop = tile

        tile = self.up_4_to_2(tile, target_hw=fL2.shape[-2:])
        tile = self.prop_2(tile, fL2, fR2)
        d_half = tile.d

        tile_full = self.up_2_to_1(tile, target_hw=left.shape[-2:])
        d_full = tile_full.d
        if d_full.shape[-2:] != left.shape[-2:]:
            d_full = F.interpolate(d_full, size=left.shape[-2:],
                                    mode="bilinear", align_corners=True)
        if aux:
            return {
                "d_final": d_full, "d_half": d_half, "d4": t4_prop.d,
                "d8": t8_prop.d, "d8_cv": t16_prop.d, "d16": t16_prop.d,
                "d32": t16_init.d,
            }
        return d_full

    def _init_with_fusion(self, fL16, fR16):
        t = self.init_tile
        B, C, H, W = fL16.shape
        cg = C // t.groups
        fL_g = fL16.view(B, t.groups, cg, H, W)
        cv = fL16.new_zeros((B, t.groups, t.max_disp, H, W))
        for d in range(t.max_disp):
            if d == 0:
                fR_s = fR16
            else:
                fR_s = fL16.new_zeros(fR16.shape)
                fR_s[:, :, :, d:] = fR16[:, :, :, :-d]
            fR_g = fR_s.view(B, t.groups, cg, H, W)
            cv[:, :, d] = (fL_g * fR_g).mean(dim=2)

        if "D" in self.arms:
            cv = self.fuse_d(cv, self.guide_proj(fL16))
        if "E" in self.arms or "V" in self.arms:
            h = t.agg[:3](cv)
            if "E" in self.arms:
                a = torch.sigmoid(self.fuse_e.proj(fL16))
                h = h * a.unsqueeze(2)
            if "V" in self.arms:
                conf = self.fuse_v(cv, fL16, fR16)
                h = h * conf
            logits = t.agg[3:](h).squeeze(1)
        else:
            logits = t.agg(cv).squeeze(1)

        prob = F.softmax(logits, dim=1)
        d = (prob * t.disp_idx).sum(dim=1, keepdim=True)
        conf = prob.max(dim=1, keepdim=True).values
        if "D" in self.arms:
            feat = t.feat_head(fL16) + self.h0(self.guide_proj(fL16))
        else:
            feat = t.feat_head(fL16)
        return TileState(d=d, sx=torch.zeros_like(d), sy=torch.zeros_like(d),
                         feat=feat, conf=conf)
