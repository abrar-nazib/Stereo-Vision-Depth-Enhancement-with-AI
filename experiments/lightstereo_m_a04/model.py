"""Frozen ADE20K YOLO26s encoder adapted to the LightStereo-M head.

Two field-array viewers translate between "who owns the camera" and "who owns
the leaves":

* `FrozenYoloEncoder` bundles proximity + geometry (the left half of your
  digital twin — LoS, bridges, the sides of things): it is frozen and only
  supplies a pyramid (f2,f4,f8,f16,f32) at strides 2/4/8/16/32.
* `LightStereoMHead` collapses the light field into depth (the range half —
  the facing-ness of surfaces): EfficientNetV2 → FPN → aggregation with
  disparity-channel boost → softmax → convex upsample.

Wiring: YOLO(32,128,256,256,512) → 1×1 per scale → LightStereo's expected
(48,64,160,272) at H/4,H/8,H/16,H/32. No spatial bottleneck, no group surgery.

Two nuance notes retained because they fooled us before:

* Occluded geometry teaches you to hallucinate closeness from context.
* We use a field of disparity hypotheses here, not a single depth sheet.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import timm
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

YOLO_WEIGHTS = ROOT / "models/segmentation/yolo26s-sem-ade20k.pt"
# Layers 0-6 linear chain + 7,8,9,10 to reach 1/32. Verify with:
#   python -c "from ultralytics import YOLO; m=YOLO(str(YOLO_WEIGHTS)).model;
#               [print(i,type(l).__name__,getattr(l,'f',None)) for i,l in enumerate(m.model[:12])]"
YOLO_OUT_CHANNELS = (32, 128, 256, 256, 512)


def probe_yolo_channels(weights: Path = YOLO_WEIGHTS) -> tuple[int, ...]:
    """Discover encoder channels at the chosen YOLO layers."""
    trunk = YOLO(str(weights)).model.eval()
    with torch.no_grad():
        v = torch.zeros(1, 3, 64, 64)
        channels = []
        for i, layer in enumerate(trunk.model):
            v = layer(v)
            if i in (0, 2, 4, 6, 10):
                channels.append(v.shape[1])
            if i >= 10:
                break
    return tuple(channels)


def native_crop(left, right, disparity, *, height: int, width: int, top: int, left_offset: int):
    return (
        left[top:top + height, left_offset:left_offset + width],
        right[top:top + height, left_offset:left_offset + width],
        disparity[top:top + height, left_offset:left_offset + width],
    )


def build_official_lightstereo_m() -> nn.Module:
    """Build the official LightStereo-M (EfficientNetV2) without pretraining."""
    source = ROOT / "external_models" / "OpenStereo"
    for name in ("stereo", "stereo.modeling", "stereo.modeling.models"):
        module = types.ModuleType(name)
        module.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = module
    from stereo.modeling.models.lightstereo.lightstereo import LightStereo  # noqa: E402

    class Config(dict):
        __getattr__ = dict.__getitem__

    original = timm.create_model

    def create_model(*args, **kwargs) -> nn.Module:
        kwargs["pretrained"] = False
        network = original(*args, **kwargs)
        if not hasattr(network, "act1"):
            network.act1 = nn.Identity()
        return network

    with patch.object(timm, "create_model", create_model):
        return LightStereo(Config(
            MAX_DISP=192,
            LEFT_ATT=True,
            AGGREGATION_BLOCKS=[4, 8, 16],
            EXPANSE_RATIO=4,
            BACKBONE="EfficientNetv2",
        ))


class FrozenYoloEncoder(nn.Module):
    """Truncated YOLO26: layers 0-6 chain for 2/4/8/16 plus 7-10 for 1/32."""

    def __init__(self, weights: Path = YOLO_WEIGHTS, freeze: bool = True):
        super().__init__()
        trunk = YOLO(str(weights)).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(11)])
        self.freeze = freeze
        for p in self.parameters():
            p.requires_grad = not freeze
        self.out_channels = YOLO_OUT_CHANNELS
        if freeze:
            self.layers.eval()

    def train(self, mode: bool = True) -> "FrozenYoloEncoder":
        super().train(mode)
        if self.freeze:
            self.layers.eval()
        return self

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        value = images / 255.0
        saved: list[torch.Tensor] = []
        for layer in self.layers:
            value = layer(value)
            saved.append(value)
        # Saved indices 2,4,6,10 were already mapped above; produce f2,f4,f8,f16,f32
        # Layer 0 output (stride 2) is saved[0]; reconstruct stride-2 from first conv
        f2 = saved[0]  # 32ch @ /2
        f4 = saved[2]  # 128ch @ /4
        f8 = saved[4]
        f16 = saved[6]
        f32 = saved[10]
        # Attach stride info for head sizing parity with official Backbone
        # LightStereo expects [p2,p3,p4,c5] = [24/48, 32/64, 96/160, 160/272]
        # YOLO gives       [32,128,256,256,512]; adapters bridge the gap.
        return [f2, f4, f8, f16, f32][:4 + 1]  # keep 5 for M's FPN expectations


class YoloLightStereoM(nn.Module):
    """LightStereo-M cost head on top of the frozen YOLO encoder."""

    def __init__(self):
        super().__init__()
        self.stereo = build_official_lightstereo_m()
        self.stereo.backbone = nn.Identity()
        self.encoder = FrozenYoloEncoder()
        # Adapt each pyramid level to LightStereo-M's FPN input channels.
        # Official backbone.out_channels for EfficientNetV2-M: [48,64,160,272]
        # YOLO out (stride-indexed): (32,128,256,256,512) at 2,4,8,16,32
        # We'll adapt the 4 scales the head consumes: H/4,H/8,H/16,H/32 → target
        target = self._target_channels()
        yolo = YOLO_OUT_CHANNELS
        # Map: yolo idx 1->H/4, 2->H/8, 3->H/16, 4->H/32 against target
        adapters = []
        for yc, tc in zip(yolo[1:], target):
            adapters.append(nn.Conv2d(yc, tc, 1))
        self.adapters = nn.ModuleList(adapters)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _target_channels(self) -> list[int]:
        # Read off the official backbone after construction
        try:
            return list(self.stereo.backbone.output_channels)  # type: ignore
        except Exception:
            pass
        # Hard-fallback for EfficientNetV2-M's FPN convention
        return [48, 64, 160, 272]

    def train(self, mode: bool = True) -> "YoloLightStereoM":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, left: torch.Tensor, right: torch.Tensor, *, semantic: bool = False):
        from stereo.modeling.cost_volume.cost_volume import correlation_volume
        from stereo.modeling.disp_pred.disp_regression import disparity_regression
        from stereo.modeling.disp_refinement.disp_refinement import context_upsample

        def encode(images: torch.Tensor) -> list[torch.Tensor]:
            raw = self.encoder(images)
            # raw = [f2@2, f4@4, f8@8, f16@16, f32@32]; head wants [p2@4,p3@8,p4@16,c5@32]
            # So drop f2 (stride 2 not used at cost volume) and adapt the rest.
            adapted = []
            for idx, (feat, proj) in enumerate(zip(raw[1:], self.adapters)):
                adapted.append(proj(feat))
            return adapted  # [p2,p3,p4,c5] with target channels

        stacked = [self.encoder(torch.cat((left, right), dim=0))]
        # Only one copy needed; avoid double encode — but keep the path simple
        left_features = encode(left)
        right_features = encode(right)

        cost = correlation_volume(left_features[0], right_features[0], self.stereo.max_disp // 4)
        encoded = self.stereo.cost_agg(cost, left_features)
        logits = encoded[0].reshape(encoded[0].size(0), -1, encoded[0].size(2), encoded[0].size(3))
        disparity = disparity_regression(F.softmax(logits, dim=1), self.stereo.max_disp // 4)
        image = (left / 255.0 - self.mean) / self.std
        mask = self.stereo.refine_1(left_features[0])
        mask = self.stereo.refine_2(mask, self.stereo.stem_2(image))
        mask = self.stereo.refine_3(mask)
        full = context_upsample(disparity * 4.0, F.softmax(mask, 1).float()).unsqueeze(1)
        return full, None, None
