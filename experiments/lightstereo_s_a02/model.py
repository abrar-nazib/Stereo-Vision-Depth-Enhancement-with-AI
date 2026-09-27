"""Frozen YOLO26s-sem encoder with an adapted LightStereo-S matching head."""

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
YOLO_CHANNELS = (128, 256, 256, 512)
LIGHTSTEREO_CHANNELS = (24, 32, 96, 160)


def native_crop(
    left: object, right: object, disparity: object, *, height: int, width: int, top: int, left_offset: int
) -> tuple[object, object, object]:
    """Return one co-located native crop; no interpolation or disparity scaling."""
    return (
        left[top:top + height, left_offset:left_offset + width],
        right[top:top + height, left_offset:left_offset + width],
        disparity[top:top + height, left_offset:left_offset + width],
    )


def coarse_upsample(disparity: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Upsample the B×1×h×w auxiliary disparity without adding a channel."""
    return F.interpolate(disparity * 4.0, size=size, mode="bilinear", align_corners=False)


def build_official_lightstereo() -> nn.Module:
    """Build OpenStereo's LightStereo-S without trainer imports or ImageNet download."""
    source = ROOT / "external_models" / "OpenStereo"
    for name in ("stereo", "stereo.modeling", "stereo.modeling.models"):
        module = types.ModuleType(name)
        module.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = module
    from stereo.modeling.models.lightstereo.lightstereo import LightStereo

    class Config(dict):
        __getattr__ = dict.__getitem__

    original = timm.create_model

    def create_model(*args: object, **kwargs: object) -> nn.Module:
        kwargs["pretrained"] = False
        network = original(*args, **kwargs)
        if not hasattr(network, "act1"):
            network.act1 = nn.Identity()
        return network

    with patch.object(timm, "create_model", create_model):
        return LightStereo(Config(MAX_DISP=192, LEFT_ATT=True, AGGREGATION_BLOCKS=[1, 2, 4], EXPANSE_RATIO=4))


def load_official_lightstereo() -> nn.Module:
    model = build_official_lightstereo()
    checkpoint = ROOT / "models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = state.get("model_state", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return model


class FrozenYoloAdapter(nn.Module):
    """Expose YOLO feature maps in the LightStereo-S channel contract."""

    output_channels = list(LIGHTSTEREO_CHANNELS)

    def __init__(self) -> None:
        super().__init__()
        self.semantic = YOLO(str(ROOT / "models/segmentation/yolo26s-sem-cityscapes.pt")).model
        self.semantic.requires_grad_(False).eval()
        self.adapters = nn.ModuleList(nn.Conv2d(source, target, 1) for source, target in zip(YOLO_CHANNELS, LIGHTSTEREO_CHANNELS))

    def train(self, mode: bool = True) -> "FrozenYoloAdapter":
        super().train(mode)
        self.semantic.eval()
        return self

    def _raw_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            value = images / 255.0
            saved: list[torch.Tensor] = []
            for layer in self.semantic.model[:11]:
                value = layer(value)
                saved.append(value)
        return [saved[index] for index in (2, 4, 6, 10)]

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [adapter(feature) for adapter, feature in zip(self.adapters, self._raw_features(images))]

    def semantic_logits(self, left: torch.Tensor) -> torch.Tensor:
        """Run the frozen original semantic neck/head for equivalence checks or deployment."""
        with torch.no_grad():
            value = left / 255.0
            saved: list[torch.Tensor] = []
            for layer in self.semantic.model[:11]:
                value = layer(value)
                saved.append(value)
            value = saved[-1]
            for layer in self.semantic.model[11:]:
                inputs = value if layer.f == -1 else (
                    saved[layer.f] if isinstance(layer.f, int) else [value if item == -1 else saved[item] for item in layer.f]
                )
                value = layer(inputs)
                saved.append(value)
        return value


class YoloLightStereoS(nn.Module):
    """LightStereo-S cost head trained on frozen YOLO26s feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.stereo = load_official_lightstereo()
        # The original MobileNetV2 is deliberately not retained in the model
        # state: all matching-head inputs come from the YOLO adapter below.
        self.stereo.backbone = nn.Identity()
        self.encoder = FrozenYoloAdapter()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "YoloLightStereoS":
        super().train(mode)
        self.encoder.semantic.eval()
        return self

    def forward(self, left: torch.Tensor, right: torch.Tensor, *, semantic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from stereo.modeling.cost_volume.cost_volume import correlation_volume
        from stereo.modeling.disp_pred.disp_regression import disparity_regression
        from stereo.modeling.disp_refinement.disp_refinement import context_upsample

        left_features = self.encoder(left)
        right_features = self.encoder(right)
        cost = correlation_volume(left_features[0], right_features[0], self.stereo.max_disp // 4)
        encoded = self.stereo.cost_agg(cost, left_features)
        logits = encoded[0].reshape(encoded[0].size(0), -1, encoded[0].size(2), encoded[0].size(3))
        disparity = disparity_regression(F.softmax(logits, dim=1), self.stereo.max_disp // 4)
        image = (left / 255.0 - self.mean) / self.std
        mask = self.stereo.refine_1(left_features[0])
        mask = self.stereo.refine_2(mask, self.stereo.stem_2(image))
        mask = self.stereo.refine_3(mask)
        full = context_upsample(disparity * 4.0, F.softmax(mask, 1).float()).unsqueeze(1)
        coarse = coarse_upsample(disparity, left.shape[-2:])
        return full, coarse, self.encoder.semantic_logits(left) if semantic else None
