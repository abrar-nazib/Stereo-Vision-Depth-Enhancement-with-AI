"""A07 — fusion-role arms: YOLO as gate/veto/guidance, never as matcher.

Matching path (fixed across arms): frozen YOLO layers 0-4 → f4@/4 (128ch),
f8@/8 (256ch) — the C2 shallow taps. Guidance source (fixed): frozen YOLO
layer 6 → g16@/16 (256ch), the deepest backbone map, projected per arm.

Arms (injection point = TileInit's groupwise correlation volume, D=24 @ /16):
  E (CoEx GCE, arXiv:2108.05773 Eq.1): a = σ(1x1(g16→G)); cv ← a ⊙ cv,
      per-pixel-per-channel gate broadcast over D ("weights shared across the
      disparity dimension ... broadcasted multiplication").
  V (SGNet confidence, ACCV2020 Sec.3.4): semCorr = channel-mean inner product
      of L/R guidance at each shift; conf = σ(TransConv3d(Conv3d(Conv3d(
      dispCorr ⊙ semCorr))) + skip) with 3x3 kernels s1/s2/sT2, zero-init
      transposed conv so conf≈skip≈1 early; cv ← conf ⊙ cv.
  D (GGEV DCA-lite, arXiv:2512.06793 Eq.2-6): per plane C_d ∈ (G,H,W):
      Q=Wq(C_d) (G→C=24), K=Wk(AvgPool(g16proj, S=4)) (C→C), group-split
      affinity A_g=Q_g^T K_g ∈ (HW,S²), M=softmax(A W_m) (S²→K², K=3),
      per-pixel dynamic depthwise conv via unfold on cat[C_d, f_da] groups,
      1x1 (G+C)→G projection; plus GRU hidden-state init from g16.

Spec deviations (documented, honest):
  * DCA applied at /16 (D=24) instead of GGEV's /4 (D=48): our init volume
    lives at /16. Mechanism identical, scale different.
  * Single-branch K=3 (no large+small dual branch — sizes not stated in paper).
  * GGEV-style per-iterate γ loss NOT ported (head returns final only).
  * Self-warp aux loss not included (user deferred).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_c import _safe_gn, native_crop  # noqa: F401
from tile_propagate import TileInit, TileRefineRAFTLike, TileUpsample

MATCH_LAYERS = (0, 1, 2, 3, 4)   # f4@/4 (layer2), f8@/8 (layer4)
GUIDE_LAYER = 6                   # g16@/16 (256ch)
MATCH_CHANNELS = (128, 256)
GUIDE_CHANNELS = 256


class FrozenYoloMatchGuide(nn.Module):
    """Frozen backbone; returns [f4, f8, g16]. Never trainable."""

    def __init__(self, weights_path):
        super().__init__()
        from ultralytics import YOLO
        trunk = YOLO(str(weights_path)).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(GUIDE_LAYER + 1)])
        for p in self.parameters():
            p.requires_grad = False
        self.out_channels = (*MATCH_CHANNELS, GUIDE_CHANNELS)
        self.layers.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.layers.eval()
        return self

    def forward(self, x):
        v = x / 255.0
        saved = {}
        for i, layer in enumerate(self.layers):
            v = layer(v)
            if i == 2:
                saved[2] = v
            elif i == 4:
                saved[4] = v
            elif i == GUIDE_LAYER:
                saved[6] = v
        return [saved[2], saved[4], saved[6]]


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


class ExciteGate(nn.Module):
    """CoEx Eq.1: per-pixel-per-channel sigmoid gate, broadcast over D."""

    def __init__(self, guide_ch: int, vol_groups: int):
        super().__init__()
        self.proj = nn.Conv2d(guide_ch, vol_groups, 1)

    def forward(self, cv: torch.Tensor, gL: torch.Tensor) -> torch.Tensor:
        # cv (B,G,D,H,W); gL (B,Cg,H,W) same H,W as cv
        a = torch.sigmoid(self.proj(gL))              # (B,G,H,W)
        return cv * a.unsqueeze(2)                    # broadcast over D


class VetoConf(nn.Module):
    """SGNet Sec.3.4: semCorr ⊙ dispCorr → 3×(3d conv k3, s1/s2/sT2) + skip → σ."""

    def __init__(self, guide_ch: int, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(1, hidden, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden, hidden, 3, stride=2, padding=1)
        self.tconv = nn.ConvTranspose3d(hidden, 1, 3, stride=2, padding=1, output_padding=1)
        nn.init.zeros_(self.tconv.weight)
        nn.init.constant_(self.tconv.bias, 2.0)  # sigma(2)=0.88: near-neutral, not 0.5

    def forward(self, cv: torch.Tensor, gL: torch.Tensor, gR: torch.Tensor) -> torch.Tensor:
        B, G, D, H, W = cv.shape
        disp_corr = cv.mean(dim=1, keepdim=True)      # (B,1,D,H,W)
        gl = F.normalize(gL, dim=1)
        gr = F.normalize(gR, dim=1)
        cg = gl.size(1)
        sem_corr = gl.new_zeros(B, 1, D, H, W)
        for d in range(D):
            if d == 0:
                rsh = gr
            else:
                rsh = gr.new_zeros(gr.shape)
                rsh[:, :, :, d:] = gr[:, :, :, :-d]
            sem_corr[:, 0, d] = (gl * rsh).mean(dim=1)  # 1/Nc via mean only
        x = disp_corr * sem_corr
        skip = x
        # fp32: the product can sit below fp16 min-normal (6.1e-5)
        with torch.autocast("cuda", enabled=False):
            h = F.relu(self.conv1(x.float()))
            h = F.relu(self.conv2(h))
            conf = torch.sigmoid(self.tconv(h) + skip.float())  # sigmoid AFTER residual add
        return conf  # (B,1,D,H,W); caller multiplies the post-conv3d map


class DynAgg(nn.Module):
    """GGEV Eq.2-6 lite: per-plane dynamic depthwise kernels on the volume.

    Q=Wq(C_d) G→C; K=Wk(AvgPool(f_da,S)) C→C; A_g=Q_gᵀK_g (HW,S²);
    M=softmax(A W_m) S²→K²; unfold-apply on cat[C_d, f_da]; proj (G+C)→G.
    """

    def __init__(self, vol_groups: int, guide_ch: int, C: int = 24,
                 S: int = 4, K: int = 3):
        super().__init__()
        self.G, self.C, self.S, self.K = vol_groups, C, S, K
        self.pad = K // 2
        self.Wq = nn.Conv2d(vol_groups, C, 1)
        self.Wk = nn.Conv2d(C, C, 1)
        self.Wm = nn.Linear(S * S, K * K)
        self.proj = nn.Conv2d(vol_groups + C, vol_groups, 1)
        self.pre = nn.Conv2d(guide_ch, C, 1)

    def forward(self, cv: torch.Tensor, gL: torch.Tensor) -> torch.Tensor:
        B, G, D, H, W = cv.shape
        Cc = self.C
        f_da = self.pre(gL)                             # (B,C,H,W)
        pooled = F.adaptive_avg_pool2d(f_da, self.S)    # (B,C,S,S)
        Kvec = self.Wk(pooled).flatten(2)               # (B,C,S²)
        Kg = Kvec.view(B, self.G, Cc // self.G, self.S * self.S)
        Ktaps = self.K * self.K
        out = cv.new_zeros(B, G, D, H, W)
        for d in range(D):
            Cd = cv[:, :, d]                            # (B,G,H,W)
            Q = self.Wq(Cd).flatten(2)                  # (B,C,HW)
            Qg = Q.view(B, self.G, Cc // self.G, H * W)
            A = torch.einsum("bgcp,bgcd->bgpd", Qg, Kg)  # (B,G,HW,S²)
            Mg = F.softmax(self.Wm(A), dim=-1)          # (B,G,HW,K²) shared W_m
            fin = torch.cat([Cd, f_da], dim=1)          # (B,G+C,H,W)
            ch_per_g = (G + Cc) // self.G
            patches = F.unfold(fin.reshape(B * self.G, ch_per_g, H, W),
                               self.K, padding=self.pad)  # (B*G, ch*K², HW)
            patches = patches.view(B, self.G, ch_per_g, Ktaps, H * W)
            agg = torch.einsum("bgckp,bgpk->bgcp", patches, Mg)  # (B,G,ch,HW)
            agg = agg.reshape(B, self.G * ch_per_g, H, W)
            out[:, :, d] = self.proj(agg)
        return out


class YoloGuidedRaft(nn.Module):
    """C2-matching skeleton + one fusion arm at the init volume."""

    def __init__(self, arm: str, weights_path):
        super().__init__()
        assert arm in ("E", "V", "D"), arm
        self.arm = arm
        self.fnet = FrozenYoloMatchGuide(weights_path)
        ch4, ch8, gch = self.fnet.out_channels
        ch16 = 128
        self.synth16 = nn.Sequential(
            nn.Conv2d(ch8, ch16, 3, stride=2, padding=1, bias=False),
            _safe_gn(ch16), nn.SiLU(inplace=True))
        self.guide_proj = nn.Conv2d(gch, 24, 1) if arm == "D" else None
        self.init_tile = TileInit(feat_ch=ch16, max_disp=24, groups=8, feat_out=16)
        if arm == "E":
            self.fuse = ExciteGate(gch, 16)  # gate ch = agg[:3] output width
        elif arm == "V":
            self.fuse = VetoConf(gch)
        else:
            self.fuse = DynAgg(8, 24)
            self.h0 = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1, bias=False),
                                    _safe_gn(16), nn.SiLU(inplace=True))
        def refine(c):
            return TileRefineRAFTLike(feat_ch=c, tile_feat_ch=16, hidden=48,
                                      half_range=2, groups=8)
        self.refine_16, self.refine_8, self.refine_4 = refine(ch16), refine(ch8), refine(ch4)
        self.up_16_to_8 = TileUpsample(2)
        self.up_8_to_4 = TileUpsample(2)
        self.up_4_to_2 = ConvexUpsample(ch4, 2)
        self.up_2_to_1 = ConvexUpsample(ch4, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.fnet.eval()
        return self

    def forward(self, left, right):
        feats = self.fnet(torch.cat([left, right], 0))
        fL4, fR4 = feats[0].chunk(2, 0)
        fL8, fR8 = feats[1].chunk(2, 0)
        gL16, gR16 = feats[2].chunk(2, 0)
        fL16, fR16 = self.synth16(fL8), self.synth16(fR8)

        # TileInit volume with fusion injected BEFORE aggregation:
        tile = self._init_with_fusion(fL16, fR16, gL16, gR16)
        for _ in range(2):
            tile = self.refine_16(tile, fL16, fR16)
        tile = self.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        for _ in range(3):
            tile = self.refine_8(tile, fL8, fR8)
        tile = self.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        for _ in range(3):
            tile = self.refine_4(tile, fL4, fR4)
        d_half = self.up_4_to_2(tile.d, fL4)
        fL2 = F.interpolate(fL4, size=d_half.shape[-2:], mode="bilinear", align_corners=False)
        d_full = self.up_2_to_1(d_half, fL2)
        return d_full

    def _init_with_fusion(self, fL16, fR16, gL16, gR16):
        """TileInit forward with arm-specific fusion at the paper-faithful site.

        E/V: gate/conf multiplies the post-conv3d feature map (CoEx: GCE sits
        after the first conv3d; SGNet: confidence multiplies the first
        hourglass volume) — NOT the raw correlation volume.
        D: DCA rewrites the volume pre-aggregation (GGEV applies DCA to the
        init volume itself), h0 = feat_head(fL) + h0(proj(g16)).
        """
        fL, fR = fL16, fR16
        B, C, H, W = fL.shape
        t = self.init_tile
        cg = C // t.groups
        fL_g = fL.view(B, t.groups, cg, H, W)
        cv = fL.new_zeros((B, t.groups, t.max_disp, H, W))
        for d in range(t.max_disp):
            if d == 0:
                fR_s = fR
            else:
                fR_s = fL.new_zeros(fR.shape)
                fR_s[:, :, :, d:] = fR[:, :, :, :-d]
            fR_g = fR_s.view(B, t.groups, cg, H, W)
            cv[:, :, d] = (fL_g * fR_g).mean(dim=2)
        if self.arm == "E":
            h = t.agg[:3](cv)                                   # (B,16,D,H,W)
            a = torch.sigmoid(self.fuse.proj(gL16))              # (B,16,H,W)
            logits = t.agg[3:](h * a.unsqueeze(2)).squeeze(1)
        elif self.arm == "V":
            h = t.agg[:3](cv)                                    # (B,16,D,H,W)
            conf = self.fuse(cv, gL16, gR16)                     # (B,1,D,H,W)
            logits = t.agg[3:](h * conf).squeeze(1)
        else:
            gp = self.guide_proj(gL16)
            cv2 = self.fuse(cv, gp)
            logits = t.agg(cv2).squeeze(1)
        prob = F.softmax(logits, dim=1)
        d = (prob * t.disp_idx).sum(dim=1, keepdim=True)
        conf = prob.max(dim=1, keepdim=True).values
        if self.arm == "D":
            feat = t.feat_head(fL) + self.h0(gp)
        else:
            feat = t.feat_head(fL)
        from tile_propagate import TileState
        return TileState(d=d, sx=torch.zeros_like(d), sy=torch.zeros_like(d),
                         feat=feat, conf=conf)
