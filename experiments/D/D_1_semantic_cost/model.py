"""SGNet-inspired semantic correlation at A09's 1/16 stereo cost volume.

This is an adaptation, not a reproduction of SGNet's PSMNet architecture.
The pretrained stereo and semantic predictors stay frozen; only D modules train.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.hitnet_a03.hitnet_head.tile_propagate import TileState


def semantic_agreement(left: torch.Tensor, right: torch.Tensor,
                       disparities: int) -> torch.Tensor:
    """Class-probability dot product for left(x) versus right(x-d)."""
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("expected equal BCHW semantic probabilities")
    batch, _, height, width = left.shape
    if disparities < 1:
        raise ValueError("disparities must be positive")
    agreement = left.new_zeros(batch, 1, disparities, height, width)
    for d in range(min(disparities, width)):
        agreement[:, 0, d, :, d:] = (left[:, :, :, d:] *
                                      right[:, :, :, :width - d]).sum(1)
    return agreement


class SemanticCostGate(nn.Module):
    """Identity-initialized correction to SGNet-style candidate confidence."""

    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.net = nn.Sequential(nn.Conv3d(2, hidden_channels, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(hidden_channels, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, volume: torch.Tensor, left_probs: torch.Tensor,
                right_probs: torch.Tensor, use_semantics: bool = True) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError("expected BGDHW cost volume")
        h, w = volume.shape[-2:]
        left = F.interpolate(left_probs.float(), (h, w), mode="bilinear", align_corners=False)
        right = F.interpolate(right_probs.float(), (h, w), mode="bilinear", align_corners=False)
        # Renormalize after interpolation so the score remains a class agreement.
        left = left / left.sum(1, keepdim=True).clamp_min(1e-6)
        right = right / right.sum(1, keepdim=True).clamp_min(1e-6)
        agreement = semantic_agreement(left, right, volume.shape[2])
        if not use_semantics:
            agreement = torch.zeros_like(agreement)
        evidence = torch.cat((volume.float().mean(1, keepdim=True), agreement), 1)
        return 1.0 + 0.5 * torch.tanh(self.net(evidence))


class ClassResidual(nn.Module):
    """SGNet-inspired category-wise depthwise refinement at native resolution."""

    def __init__(self, classes: int = 14, hidden_channels: int = 32):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.depthwise = nn.Conv2d(classes, classes, 3, padding=1, groups=classes)
        self.fuse = nn.Sequential(nn.Conv2d(2 * classes + 2, hidden_channels, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(hidden_channels, 1, 3, padding=1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, disparity: torch.Tensor, left_probs: torch.Tensor,
                confidence: torch.Tensor, use_semantics: bool) -> torch.Tensor:
        size = disparity.shape[-2:]
        probs = F.interpolate(left_probs.float(), size, mode="bilinear", align_corners=False)
        probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-6)
        if not use_semantics:
            probs = torch.zeros_like(probs)
        normalized = disparity.float() / 192.0
        confidence = F.interpolate(confidence.float(), size, mode="bilinear",
                                   align_corners=False)
        category_disparity = self.depthwise(probs * normalized)
        signal = torch.cat((category_disparity, probs, normalized, confidence), 1)
        return disparity + 4.0 * torch.tanh(self.fuse(signal))


class DModel(nn.Module):
    """One shared two-view encoder, two frozen semantic tails, trainable D modules."""

    def __init__(self, stereo_checkpoint, semantic_checkpoint, encoder_checkpoint,
                 *, refinement: bool, use_semantics: bool):
        super().__init__()
        self.base = FusedStereoSemantic(stereo_checkpoint, semantic_checkpoint,
                                        encoder_checkpoint)
        self.gate = SemanticCostGate()
        self.refiner = ClassResidual() if refinement else None
        self.use_semantics = use_semantics

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _semantic_tail(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        saved = {4: features[2], 6: features[3]}
        value = features[3]
        for index, layer in enumerate(self.base.semantic_tail, start=7):
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
        return value

    def _init_tile(self, f_left: torch.Tensor, f_right: torch.Tensor,
                   left_probs: torch.Tensor, right_probs: torch.Tensor) -> TileState:
        stereo = self.base.stereo
        tile = stereo.init_tile
        batch, channels, height, width = f_left.shape
        grouped_left = f_left.reshape(batch, tile.groups, channels // tile.groups, height, width)
        volume = f_left.new_zeros(batch, tile.groups, tile.max_disp, height, width)
        for d in range(tile.max_disp):
            shifted = f_right if d == 0 else F.pad(f_right[..., :-d], (d, 0))
            grouped_right = shifted.reshape(batch, tile.groups, channels // tile.groups, height, width)
            volume[:, :, d] = (grouped_left * grouped_right).mean(2)
        hidden = tile.agg[:3](volume)
        # Preserve the pretrained A09 veto exactly; the D gate starts at identity.
        hidden = hidden * stereo.fuse_v(volume, f_left, f_right)
        gate = self.gate(volume, left_probs, right_probs, self.use_semantics)
        logits = tile.agg[3:](hidden * gate.to(hidden.dtype)).squeeze(1)
        probabilities = F.softmax(logits, dim=1)
        disparity = (probabilities * tile.disp_idx).sum(1, keepdim=True)
        confidence = probabilities.max(1, keepdim=True).values
        return TileState(d=disparity, sx=torch.zeros_like(disparity),
                         sy=torch.zeros_like(disparity), feat=tile.feat_head(f_left),
                         conf=confidence)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError("expected paired NCHW RGB tensors")
        if left.shape[-2] % 32 or left.shape[-1] % 32:
            raise ValueError("pad paired inputs to multiples of 32")
        with torch.no_grad(), torch.autocast("cuda", enabled=left.is_cuda, dtype=torch.float16):
            features = self.base.stereo.fnet(torch.cat((left, right), 0))
            left_features = tuple(feature.chunk(2, 0)[0] for feature in features)
            right_features = tuple(feature.chunk(2, 0)[1] for feature in features)
            # Sequential tails keep the right-view overhead below two full YOLO passes.
            left_logits = self._semantic_tail(left_features)
            right_logits = self._semantic_tail(right_features)
            left_probs = left_logits.float().softmax(1)
            right_probs = right_logits.float().softmax(1)
        f_left = [feature.chunk(2, 0)[0] for feature in features]
        f_right = [feature.chunk(2, 0)[1] for feature in features]
        stereo = self.base.stereo
        with torch.autocast("cuda", enabled=left.is_cuda, dtype=torch.float16):
            state = self._init_tile(f_left[3], f_right[3], left_probs, right_probs)
            state = stereo.prop_16(state, f_left[3], f_right[3])
            state = stereo.up_16_to_8(state, target_hw=f_left[2].shape[-2:])
            state = stereo.prop_8(state, f_left[2], f_right[2])
            state = stereo.up_8_to_4(state, target_hw=f_left[1].shape[-2:])
            state = stereo.prop_4(state, f_left[1], f_right[1])
            state = stereo.up_4_to_2(state, target_hw=f_left[0].shape[-2:])
            state = stereo.prop_2(state, f_left[0], f_right[0])
            state = stereo.up_2_to_1(state, target_hw=left.shape[-2:])
            disparity = state.d
            if disparity.shape[-2:] != left.shape[-2:]:
                disparity = F.interpolate(disparity, size=left.shape[-2:],
                                          mode="bilinear", align_corners=True)
        if self.refiner is not None:
            disparity = self.refiner(disparity.float(), left_probs, state.conf,
                                      self.use_semantics)
        return disparity.float(), left_logits.float()

    def trainable_state(self) -> dict:
        return {"gate": self.gate.state_dict(),
                "refiner": self.refiner.state_dict() if self.refiner is not None else None}

    def load_trainable_state(self, state: dict) -> None:
        self.gate.load_state_dict(state["gate"])
        if self.refiner is not None:
            self.refiner.load_state_dict(state["refiner"])
