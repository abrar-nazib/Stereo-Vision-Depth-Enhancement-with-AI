"""HITNet (Tankovich et al., CVPR 2021) — implemented from the paper.

Written from the published paper and its equations; no third-party
implementation source is reused. Module and parameter names follow the
TensorFlow-converted tensor set (`hitnet_xl_sf_finalpass_from_tf.ckpt`,
Apache-2.0 upstream) so that it loads by name (verified 164/164 strict).

Layout conventions, all traced back to the paper:

* Sec 3.2  U-Net feature extractor; the **decoder** outputs at every resolution
  (``e0`` at 1/1 ... ``e4`` at 1/16) are what the rest of the network uses.
* Sec 3.3  a tile is a 4x4 region of a feature map, so at the working level the
  tile grid is 1/4 image resolution. The 4x4 conv uses **stride 4 for the left
  image and stride (4, 1) for the right image**, keeping the secondary embedding
  at full horizontal resolution for scan-line matching (Eq. 2/3).
* Sec 3.4  each hypothesis is expanded to its planar patch with the plane
  equation (Eq. 5); the right features are warped by those local disparities and
  the 4x4 L1 cost vector (Eq. 6) is concatenated at d-1, d, d+1 (Eq. 7). One
  residual CNN, shared across scales, predicts deltas plus a confidence (Eq. 8).
* The hierarchy is followed by three propagation passes at tile sizes 4x4, 2x2
  and 1x1; the 1x1 output is the final prediction.

A tile at grid spacing ``s`` covers ``tile`` pixels of the image; the plane
expansion of any grid therefore always resolves to the full image resolution,
which is what makes the 4x4 -> 2x2 -> 1x1 schedule consistent.

Residual blocks carry no batch normalization and use dilated convolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAK = 0.2
PLANE_DIMS = 3           # d, dx, dy
DESCRIPTOR_DIMS = 13     # tile feature descriptor p
HYPOTHESIS_DIMS = PLANE_DIMS + DESCRIPTOR_DIMS   # 16
COST_VECTOR_DIMS = 16    # 4x4 tile => 16 cost entries (Eq. 6)
TILE = 4


def _act() -> nn.Module:
    return nn.LeakyReLU(LEAK)


class UpsampleBlock(nn.Module):
    """U-Net decoder block: transposed conv, concat skip, 1x1 then two 3x3."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up_conv = nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 2, 2), _act())
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, 1), _act(),
            nn.Conv2d(c_out, c_out, 3, 1, 1), _act(),
            nn.Conv2d(c_out, c_out, 3, 1, 1), _act(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.merge_conv(torch.cat([self.up_conv(x), skip], dim=1))


class FeatureExtractor(nn.Module):
    """U-Net producing the decoder features used downstream."""

    def __init__(self, channels: tuple[int, int, int, int, int]):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.channels = channels
        self.down_0 = nn.Sequential(nn.Conv2d(3, c0, 3, 1, 1), _act())
        self.down_1 = nn.Sequential(nn.Conv2d(c0, c1, 4, 2, 1), _act(), nn.Conv2d(c1, c1, 3, 1, 1), _act())
        self.down_2 = nn.Sequential(nn.Conv2d(c1, c2, 4, 2, 1), _act(), nn.Conv2d(c2, c2, 3, 1, 1), _act())
        self.down_3 = nn.Sequential(nn.Conv2d(c2, c3, 4, 2, 1), _act(), nn.Conv2d(c3, c3, 3, 1, 1), _act())
        self.down_4 = nn.Sequential(
            nn.Conv2d(c3, c4, 4, 2, 1), _act(),
            nn.Conv2d(c4, c4, 3, 1, 1), _act(),
            nn.Conv2d(c4, c4, 3, 1, 1), _act(),
            nn.Conv2d(c4, c4, 3, 1, 1), _act(),
        )
        self.up_3 = UpsampleBlock(c4, c3)
        self.up_2 = UpsampleBlock(c3, c2)
        self.up_1 = UpsampleBlock(c2, c1)
        self.up_0 = UpsampleBlock(c1, c0)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        d0 = self.down_0(image)
        d1 = self.down_1(d0)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        e3 = self.up_3(d4, d3)
        e2 = self.up_2(e3, d2)
        e1 = self.up_1(e2, d1)
        e0 = self.up_0(e1, d0)
        return [e0, e1, e2, e3, d4]


class ResBlock(nn.Module):
    """Dilated residual block, no batch normalization (Sec 3.4)."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation), _act(),
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation),
        )
        self.relu = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


def _res_stack(channels: int, count: int, dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 1)) -> nn.Sequential:
    return nn.Sequential(*[ResBlock(channels, dilation=dilations[i % len(dilations)]) for i in range(count)])


def plane_expand(disparity: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """Eq. 5: expand a tile grid to per-pixel disparity with the plane equation.

    ``tile`` is how many image pixels one tile spans, so the result always lands
    at the full image resolution regardless of the current grid spacing.
    """
    batch, _, height, width = disparity.shape
    offsets = torch.arange(tile, device=disparity.device, dtype=disparity.dtype) - (tile - 1) / 2
    offset_y = offsets.view(1, 1, tile, 1)
    offset_x = offsets.view(1, 1, 1, tile)
    expanded = (disparity.unsqueeze(-1).unsqueeze(-1)
                + dx.unsqueeze(-1).unsqueeze(-1) * offset_y
                + dy.unsqueeze(-1).unsqueeze(-1) * offset_x)
    return expanded.permute(0, 1, 2, 4, 3, 5).reshape(batch, 1, height * tile, width * tile)


def warp_right(right_feature: torch.Tensor, reference: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Sample the right feature map at x - d using linear interpolation (Sec 3.4)."""
    height, width = reference.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    sample_x = (x.unsqueeze(0) - disparity.squeeze(1)) / max(width - 1, 1) * 2 - 1
    sample_y = (y.unsqueeze(0) / max(height - 1, 1) * 2 - 1).expand_as(sample_x)
    return F.grid_sample(right_feature, torch.stack([sample_x, sample_y], dim=-1),
                         align_corners=True, padding_mode="border")


def cost_vector(left_feature: torch.Tensor, right_feature: torch.Tensor,
                disparity: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """Eq. 6: the 16-entry 4x4 cost vector, folded to one cell per tile."""
    local = plane_expand(disparity, dx, dy, tile)
    warped = warp_right(right_feature, left_feature, local)
    difference = (left_feature - warped).abs().mean(dim=1, keepdim=True)
    folded = F.unfold(difference, kernel_size=TILE, stride=TILE)
    height, width = disparity.shape[-2:]
    return folded.reshape(difference.shape[0], COST_VECTOR_DIMS, height, width)


class TileInit(nn.Module):
    """Initialization stage (Sec 3.3): tile embeddings, L1 matching, descriptor."""

    def __init__(self, embed_channels: int = 16, feature_channels: int = 48,
                 descriptor_dims: int = DESCRIPTOR_DIMS, max_disp: int = 48):
        super().__init__()
        self.max_disp = max_disp
        self.conv_em = nn.Conv2d(32, embed_channels, TILE, TILE)
        self.relu_conv = nn.Sequential(_act(), nn.Conv2d(embed_channels, embed_channels, 1))
        self.tile_feautre = nn.Sequential(nn.Conv2d(feature_channels + 1, descriptor_dims, 1), _act())

    def embedding(self, feature: torch.Tensor, stride_x: int) -> torch.Tensor:
        """Shared 4x4 conv; the right image uses stride (4, 1) (Sec 3.3)."""
        embedded = F.conv2d(feature, self.conv_em.weight, self.conv_em.bias, stride=(TILE, stride_x))
        return self.relu_conv(embedded)

    def descriptor(self, feature: torch.Tensor, best_cost: torch.Tensor) -> torch.Tensor:
        return self.tile_feautre(torch.cat([feature, best_cost], dim=1))


class TilePropagation(nn.Module):
    """Shared propagation module U_l (Sec 3.4), applied once per resolution."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(HYPOTHESIS_DIMS + 3 * COST_VECTOR_DIMS, HYPOTHESIS_DIMS, 1), _act())
        self.conv1 = nn.Sequential(nn.Conv2d(HYPOTHESIS_DIMS * 2, hidden, 3, 1, 1), _act())
        self.res_block = _res_stack(hidden, 2)
        self.convn = nn.Conv2d(hidden, HYPOTHESIS_DIMS + 1, 3, 1, 1)

    def forward(self, augmented: torch.Tensor, hypothesis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.conv_neighbors(augmented)
        x = self.conv1(torch.cat([hypothesis, reduced], dim=1))
        out = self.convn(self.res_block(x))
        return out[:, :HYPOTHESIS_DIMS], out[:, HYPOTHESIS_DIMS:]


class Refine(nn.Module):
    """Final scale-specific propagation pass at 4x4, 2x2 and 1x1 tile sizes."""

    def __init__(self, feature_channels: int, hidden: int = 64):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(HYPOTHESIS_DIMS + feature_channels, hidden, 1), _act())
        self.conv1 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1), _act())
        self.res_block = _res_stack(hidden, 6)
        self.convn = nn.Conv2d(hidden, HYPOTHESIS_DIMS, 3, 1, 1)

    def forward(self, hypothesis: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        x = self.res_block(self.conv1(self.conv1x1(torch.cat([hypothesis, feature], dim=1))))
        return self.convn(x)


@dataclass
class TileHypothesis:
    """Planar tile state: disparity, its x/y gradients, descriptor, confidence."""

    d: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    descriptor: torch.Tensor
    confidence: torch.Tensor
    tile: int = TILE

    def stack(self) -> torch.Tensor:
        return torch.cat([self.d, self.dx, self.dy, self.descriptor], dim=1)

    def plane(self) -> torch.Tensor:
        return plane_expand(self.d, self.dx, self.dy, self.tile)

    @classmethod
    def from_update(cls, hypothesis: "TileHypothesis", delta: torch.Tensor) -> "TileHypothesis":
        updated = hypothesis.stack() + delta
        return cls(d=updated[:, :1], dx=updated[:, 1:2], dy=updated[:, 2:3], descriptor=updated[:, 3:],
                   confidence=hypothesis.confidence, tile=hypothesis.tile)

    def upsample(self) -> "TileHypothesis":
        """Halve the tile footprint: pool the plane at the new tile size.

        The plane expansion always resolves to full image resolution, so pooling
        it with the *next* tile size lands exactly on the next (finer) tile grid.
        """
        tile = max(self.tile // 2, 1)
        plane = self.plane()
        grid = F.avg_pool2d(plane, tile, tile)
        nearest = lambda t: F.interpolate(t.to(plane.dtype), size=grid.shape[-2:], mode="nearest")
        return TileHypothesis(d=grid, dx=nearest(self.dx), dy=nearest(self.dy),
                              descriptor=nearest(self.descriptor), confidence=nearest(self.confidence),
                              tile=tile)


@dataclass
class HITNetConfig:
    channels: tuple[int, int, int, int, int] = (32, 40, 48, 56, 64)
    init_max_disp: int = 48
    refine_scales: tuple[int, ...] = (2, 1, 0)   # image features e2, e1, e0


class HITNet(nn.Module):
    """HITNet with 5 levels (M = 4), as used for SceneFlow."""

    def __init__(self, config: HITNetConfig | None = None):
        super().__init__()
        self.config = config or HITNetConfig()
        c0, c1, c2, _, _ = self.config.channels
        self.feature_extractor = FeatureExtractor(self.config.channels)
        self.init_layer_0 = TileInit(embed_channels=16, feature_channels=c2, max_disp=self.config.init_max_disp)
        self.prop_layer_0 = TilePropagation(hidden=32)
        self.refine_l0 = Refine(c2)
        self.refine_l1 = Refine(c1)
        self.refine_l2 = Refine(c0)

    # -- Sec 3.3 -----------------------------------------------------------------
    def initialize(self, left_feature: torch.Tensor, right_feature: torch.Tensor,
                   descriptor_feature: torch.Tensor) -> TileHypothesis:
        left_embedding = self.init_layer_0.embedding(left_feature, stride_x=TILE)
        right_embedding = self.init_layer_0.embedding(right_feature, stride_x=1)
        best_cost, best_disparity = self._match(left_embedding, right_embedding)
        return TileHypothesis(
            d=best_disparity, dx=torch.zeros_like(best_disparity), dy=torch.zeros_like(best_disparity),
            descriptor=self.init_layer_0.descriptor(descriptor_feature, best_cost),
            confidence=torch.ones_like(best_disparity),
        )

    def _match(self, left_embedding: torch.Tensor, right_embedding: torch.Tensor):
        """Eq. 2/3: L1 cost along scan lines; the right map keeps full x-resolution."""
        width = left_embedding.shape[-1]
        positions = torch.arange(width, device=left_embedding.device) * TILE
        costs = []
        for disparity in range(self.config.init_max_disp):
            index = (positions - disparity).clamp(0, right_embedding.shape[-1] - 1)
            sampled = right_embedding.index_select(-1, index)
            costs.append((left_embedding - sampled).abs().mean(dim=1, keepdim=True))
        stacked = torch.cat(costs, dim=1)
        best_cost, index = stacked.min(dim=1, keepdim=True)
        return best_cost, index.to(left_embedding.dtype)

    # -- Sec 3.4 -----------------------------------------------------------------
    def propagate(self, hypothesis: TileHypothesis, left_feature: torch.Tensor,
                  right_feature: torch.Tensor) -> TileHypothesis:
        neighbours = [cost_vector(left_feature, right_feature, hypothesis.d + offset,
                                  hypothesis.dx, hypothesis.dy, hypothesis.tile)
                      for offset in (-1.0, 0.0, 1.0)]
        delta, confidence = self.prop_layer_0(
            torch.cat([hypothesis.stack()] + neighbours, dim=1), hypothesis.stack())
        return replace(TileHypothesis.from_update(hypothesis, delta), confidence=confidence)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                iters: int = 3, test_mode: bool = False) -> dict[str, torch.Tensor]:
        left_features = self.feature_extractor(left / 255.0)
        right_features = self.feature_extractor(right / 255.0)

        hypothesis = self.initialize(left_features[0], right_features[0], left_features[2])
        for _ in range(iters):
            hypothesis = self.propagate(hypothesis, left_features[0], right_features[0])

        # Three propagation passes at tile sizes 4x4, 2x2 and 1x1.
        for order, (feature_index, refine) in enumerate(
                zip(self.config.refine_scales, (self.refine_l0, self.refine_l1, self.refine_l2))):
            if order:
                hypothesis = hypothesis.upsample()
            hypothesis = TileHypothesis.from_update(
                hypothesis, refine(hypothesis.stack(), left_features[feature_index]))

        return {"disp_pred": hypothesis.plane(), "tile_disparity": hypothesis.d}


def load_hitnet(checkpoint: Path | str, config: HITNetConfig | None = None) -> HITNet:
    """Build HITNet and load the TensorFlow-converted tensor set by name."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("model."): value for key, value in state.items() if hasattr(value, "shape")}
    model = HITNet(config)
    model.load_state_dict(state, strict=True)
    return model.eval()
