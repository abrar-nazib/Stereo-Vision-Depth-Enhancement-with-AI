"""A07 pre-check — 20-pair overfit falsification test.

The user's StereoLite overfit record: 20 pairs, train==eval, 0.84 EPE @ 3k steps
with a random-init 0.487M encoder trained end-to-end. Our frozen-YOLO runs have
never been tested in that regime. If frozen features can't overfit 20 pairs that
a trained scratch encoder crushes, the frozen premise is the bug, not the head.

Two arms, identical RAFT-like head (A06 C2 skeleton, taps /4+/8):
  --encoder frozen   YOLO26s-seg layers 0-4, requires_grad=False
  --encoder scratch  4-stage conv encoder (~0.2M), trained
Both: 20 pairs from A05 train manifest, train==eval at native res, 3k steps,
AdamW 2e-4, modal loss, seed 42.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
A06 = HERE.parent
ROOT = A06.parents[1]
sys.path.insert(0, str(HERE))

from model_c import ARMS, TAPS, _safe_gn, native_crop  # noqa: E402
from losses import modal_loss  # noqa: E402
from tile_propagate import TileInit, TileRefineRAFTLike, TileUpsample  # noqa: E402
import importlib.util


def _sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a02 = _sibling("a02_run", ROOT / "experiments/lightstereo_s_a02/run.py")
read_pfm, metrics = a02.read_pfm, a02.metrics


class FrozenYoloTaps(nn.Module):
    """YOLO layers 0..4 frozen; out (f4@/4 128ch, f8@/8 256ch)."""

    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        trunk = YOLO(str(ROOT / "models/segmentation/yolo26s-sem-ade20k.pt")).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(5)])
        for p in self.parameters():
            p.requires_grad = False
        self.out_channels = (128, 256)
        self.layers.eval()

    def train(self, mode=True):
        super().train(mode)
        self.layers.eval()
        return self

    def forward(self, x):
        v = x / 255.0
        saved = []
        for i, layer in enumerate(self.layers):
            v = layer(v)
            if i in (2, 4):
                saved.append(v)
        return saved


class ScratchEncoder(nn.Module):
    """4-stage conv encoder, same channel contract, trained from scratch."""

    def __init__(self):
        super().__init__()
        def stage(ci, co, s):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=s, padding=1, bias=False),
                _safe_gn(co), nn.SiLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                _safe_gn(co), nn.SiLU(inplace=True))
        self.s2 = stage(3, 32, 2)
        self.s4 = stage(32, 128, 2)
        self.s8 = stage(128, 256, 2)
        self.out_channels = (128, 256)

    def forward(self, x):
        v = x / 255.0
        f4 = self.s4(self.s2(v))
        f8 = self.s8(f4)
        return [f4, f8]


class OverfitHead(nn.Module):
    """A06 C2 head skeleton; encoder swapped."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        ch4, ch8 = encoder.out_channels
        ch16 = 128
        self.synth16 = nn.Sequential(
            nn.Conv2d(ch8, ch16, 3, stride=2, padding=1, bias=False),
            _safe_gn(ch16), nn.SiLU(inplace=True))
        self.init_tile = TileInit(feat_ch=ch16, max_disp=24, groups=8, feat_out=16)
        def refine(c):
            return TileRefineRAFTLike(feat_ch=c, tile_feat_ch=16, hidden=48,
                                      half_range=2, groups=8)
        self.refine_16, self.refine_8, self.refine_4 = refine(ch16), refine(ch8), refine(ch4)
        self.up_16_to_8 = TileUpsample(2)
        self.up_8_to_4 = TileUpsample(2)
        from model_c import ConvexUpsample
        self.up_4_to_2 = ConvexUpsample(ch4, 2)
        self.up_2_to_1 = ConvexUpsample(ch4, 2)

    def forward(self, left, right):
        feats = self.encoder(torch.cat([left, right], 0))
        fL4, fR4 = feats[0].chunk(2, 0)
        fL8, fR8 = feats[1].chunk(2, 0)
        fL16, fR16 = self.synth16(fL8), self.synth16(fR8)

        tile = self.init_tile(fL16, fR16)
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


def to_t(img):
    return torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda()


def pad_lr(l, r):
    h, w = l.shape[-2:]
    top, rp = (-h) % 16, (-w) % 16
    return (F.pad(l, (0, rp, top, 0), mode="replicate"),
            F.pad(r, (0, rp, top, 0), mode="replicate"), top, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["frozen", "scratch"], required=True)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    tag = f"A07pre_{args.encoder}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = A06 / "runs" / tag
    out.mkdir(parents=True)
    log = open(out / "run.log", "a")

    def P(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    manifest = json.loads((A06 / "manifests/manifest_train.json").read_text())[:args.pairs]
    cache = []
    for r in manifest:
        l = cv2.cvtColor(cv2.imread(r["left"]), cv2.COLOR_BGR2RGB)
        r_ = cv2.cvtColor(cv2.imread(r["right"]), cv2.COLOR_BGR2RGB)
        cache.append((l, r_, read_pfm(r["disparity"])))
    P(f"{tag}: {len(cache)} pairs, train==eval, encoder={args.encoder}")

    enc = FrozenYoloTaps() if args.encoder == "frozen" else ScratchEncoder()
    model = OverfitHead(enc).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    P(f"trainable {sum(p.numel() for p in trainable)/1e6:.3f}M / total {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rnd = random.Random(args.seed)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        l, r_, gt = cache[rnd.randrange(len(cache))]
        h, w = gt.shape
        top = rnd.randint(0, h - 384); loff = rnd.randint(0, w - 640)
        l, r_, gt = native_crop(l, r_, gt, height=384, width=640, top=top, left_offset=loff)
        lt, rt = to_t(l), to_t(r_)
        target = torch.from_numpy(gt)[None, None].cuda()
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(lt, rt)
            loss, _ = modal_loss(pred, target, lt)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            errs = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for l, r_, gt in cache:
                    lt, rt = to_t(l), to_t(r_)
                    lt, rt, top, w = pad_lr(lt, rt)
                    p = model(lt, rt)[..., top:, :w].cpu()
                    errs.append(metrics(p, torch.from_numpy(gt)[None, None])["epe"])
            e = float(np.mean(errs))
            hist.append({"step": step, "train_eval_epe": e})
            P(f"step={step} loss={loss.item():.3f} train==eval EPE={e:.4f}")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    P(f"DONE {tag} final={hist[-1]['train_eval_epe']:.4f} in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
