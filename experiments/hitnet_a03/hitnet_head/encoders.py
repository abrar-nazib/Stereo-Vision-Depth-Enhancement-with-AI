"""Feature encoders for the StereoLite HITNet-style head.

`FrozenYoloEncoder` gives the head the same ``(f2, f4, f8, f16)`` contract that
its native ``TileFeatureEncoder`` provides, so the encoder can be swapped
without adapters: the head sizes its own convolutions from ``out_channels``.

The YOLO backbone is truncated to layers 0-6, which form a strict linear chain
(every layer's ``from`` is -1) covering strides 2/4/8/16. Layers 7+ (the 1/32
stage, the neck and the detection/segmentation head) are dropped.

Input contract: the head passes images in [0, 255], and every encoder is
responsible for its own scaling. YOLO expects [0, 1], so that division happens
here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def probe_channels(layers: nn.ModuleList, indices: tuple[int, ...] = (0, 2, 4, 6),
                   size: int = 64) -> tuple[int, ...]:
    """Discover the channel count at the given layer outputs."""
    channels: list[int] = []
    value = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        for index, layer in enumerate(layers):
            value = layer(value)
            if index in indices:
                channels.append(value.shape[1])
    return tuple(channels)


class FrozenYoloEncoder(nn.Module):
    """Truncated YOLO26 backbone exposing (f2, f4, f8, f16).

    Args:
        weights: path to the ultralytics checkpoint.
        freeze: when True the backbone is held in eval mode with no gradients,
            which is what the A03 ablation tests.
    """

    def __init__(self, weights, freeze: bool = True):
        super().__init__()
        from ultralytics import YOLO

        trunk = YOLO(str(weights)).model.eval()
        self.layers = nn.ModuleList([trunk.model[i] for i in range(7)])
        self.freeze = freeze
        for parameter in self.parameters():
            parameter.requires_grad = not freeze
        self.out_channels = probe_channels(self.layers)
        if freeze:
            self.layers.eval()

    def train(self, mode: bool = True) -> "FrozenYoloEncoder":
        super().train(mode)
        if self.freeze:
            self.layers.eval()
        return self

    def forward(self, images: torch.Tensor):
        value = images / 255.0
        f2 = self.layers[0](value)
        f4 = self.layers[2](self.layers[1](f2))
        f8 = self.layers[4](self.layers[3](f4))
        f16 = self.layers[6](self.layers[5](f8))
        return f2, f4, f8, f16
