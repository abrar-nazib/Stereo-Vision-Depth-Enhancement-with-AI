"""Reuse one frozen YOLO26m trunk for A09 stereo and VKITTI semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO


_RUNNERS = Path(__file__).resolve().parents[2] / "a06_shallow" / "runners"
if str(_RUNNERS) not in sys.path:
    sys.path.insert(0, str(_RUNNERS))
from model_a07v2 import FusionStereoLite  # noqa: E402


class FusedStereoSemantic(nn.Module):
    """Frozen predictors with one left/right trunk call per stereo pair.

    Inputs are RGB tensors in [0,255] and spatial dimensions divisible by 32.
    Outputs are padded full-pixel disparity and low-resolution semantic logits.
    """

    def __init__(self, stereo_checkpoint: str | Path,
                 semantic_checkpoint: str | Path,
                 encoder_checkpoint: str | Path):
        super().__init__()
        self.stereo = FusionStereoLite("V", encoder=encoder_checkpoint)
        payload = torch.load(stereo_checkpoint, map_location="cpu", weights_only=False)
        try:
            self.stereo.load_state_dict(payload["model"], strict=True)
        except RuntimeError as exc:
            raise ValueError("shared trunk or stereo checkpoint is incompatible") from exc
        semantic = YOLO(str(semantic_checkpoint)).model.eval()
        if len(semantic.model) != 18:
            raise ValueError("unexpected semantic graph; expected YOLO26m layers 0–17")
        for index in range(7):
            source = self.stereo.fnet.layers[index].state_dict()
            target = semantic.model[index].state_dict()
            if source.keys() != target.keys() or any(
                    not torch.equal(value, target[key]) for key, value in source.items()):
                raise ValueError(f"shared trunk differs at layer {index}")
        self.semantic_tail = nn.ModuleList(semantic.model[7:])
        if [layer.f for layer in self.semantic_tail if isinstance(layer.f, list)] != [
                [-1, 6], [-1, 4], [16, 13]]:
            raise ValueError("unexpected semantic skip graph")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._cuda_streams = {}
        self.eval()

    @property
    def shared_layers(self) -> nn.ModuleList:
        return self.stereo.fnet.layers

    def train(self, mode: bool = True) -> "FusedStereoSemantic":
        # Base predictors must never switch BatchNorm to training mode.
        super().train(False)
        return self

    def _stereo_from_features(self, features: tuple[torch.Tensor, ...],
                              image_hw: tuple[int, int], return_features: bool = False):
        fL2, fR2 = features[0].chunk(2, dim=0)
        fL4, fR4 = features[1].chunk(2, dim=0)
        fL8, fR8 = features[2].chunk(2, dim=0)
        fL16, fR16 = features[3].chunk(2, dim=0)
        stereo = self.stereo
        tile = stereo._init_with_fusion(fL16, fR16)
        tile = stereo.prop_16(tile, fL16, fR16)
        tile = stereo.up_16_to_8(tile, target_hw=fL8.shape[-2:])
        tile = stereo.prop_8(tile, fL8, fR8)
        tile = stereo.up_8_to_4(tile, target_hw=fL4.shape[-2:])
        tile = stereo.prop_4(tile, fL4, fR4)
        tile4 = tile
        tile = stereo.up_4_to_2(tile, target_hw=fL2.shape[-2:])
        tile = stereo.prop_2(tile, fL2, fR2)
        tile = stereo.up_2_to_1(tile, target_hw=image_hw)
        disparity = tile.d
        if disparity.shape[-2:] != image_hw:
            disparity = F.interpolate(disparity, size=image_hw, mode="bilinear", align_corners=True)
        if return_features:
            return disparity, {"tile_f4": tile4.feat, "tile_conf4": tile4.conf}
        return disparity

    def _semantic_from_features(self, features: tuple[torch.Tensor, ...],
                                return_features: bool = False):
        left4 = features[2].chunk(2, dim=0)[0]
        left16 = features[3].chunk(2, dim=0)[0]
        saved = {4: left4, 6: left16}
        value = left16
        for index, layer in enumerate(self.semantic_tail, start=7):
            source = layer.f
            if isinstance(source, list):
                inputs = [value if j == -1 else saved[j] for j in source]
            elif source == -1:
                inputs = value
            else:
                inputs = saved[source]
            value = layer(inputs)
            if index in (13, 16):
                saved[index] = value
        if return_features:
            return value, {"semantic_f8": saved[16]}
        return value

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                parallel: bool = False, return_features: bool = False):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError("expected matching NCHW RGB stereo tensors")
        if left.shape[-2] % 32 or left.shape[-1] % 32:
            raise ValueError("pad both inputs to multiples of 32 before inference")
        features = self.stereo.fnet(torch.cat((left, right), dim=0))
        if return_features and parallel:
            raise ValueError("feature return uses the measured-faster single-stream path")
        if parallel and left.is_cuda:
            device = left.device.index
            if device not in self._cuda_streams:
                self._cuda_streams[device] = (torch.cuda.Stream(device=left.device),
                                              torch.cuda.Stream(device=left.device))
            depth_stream, semantic_stream = self._cuda_streams[device]
            caller = torch.cuda.current_stream(device=left.device)
            depth_stream.wait_stream(caller)
            semantic_stream.wait_stream(caller)
            with torch.cuda.stream(depth_stream):
                disparity = self._stereo_from_features(features, left.shape[-2:])
            with torch.cuda.stream(semantic_stream):
                semantics = self._semantic_from_features(features)
            caller.wait_stream(depth_stream)
            caller.wait_stream(semantic_stream)
            disparity.record_stream(caller)
            semantics.record_stream(caller)
        else:
            if return_features:
                disparity, stereo_features = self._stereo_from_features(
                    features, left.shape[-2:], return_features=True)
                semantics, semantic_features = self._semantic_from_features(
                    features, return_features=True)
                left4, _ = features[1].chunk(2, dim=0)
                left8, right8 = features[2].chunk(2, dim=0)
                return disparity, semantics, {"left_f4": left4, "left_f8": left8,
                                              "right_f8": right8, **stereo_features,
                                              **semantic_features}
            disparity = self._stereo_from_features(features, left.shape[-2:])
            semantics = self._semantic_from_features(features)
        return disparity, semantics
