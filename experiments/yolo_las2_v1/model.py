"""A01 v1.0: frozen YOLO26s-sem + learned adapters + pretrained LAS2-S."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'external_models/LiteAnyStereo'))
from core.liteanystereov2 import LiteAnyStereoV2


def load_stereo():
    model = LiteAnyStereoV2(model_size='s', fnet_pretrained=False)
    model.load_state_dict(torch.load(ROOT / 'models/stereo/liteanystereo/LAS2_S.pth',
                                    map_location='cpu', weights_only=True), strict=True)
    return model


class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic = YOLO(str(ROOT / 'models/segmentation/yolo26s-sem-cityscapes.pt')).model
        self.semantic.requires_grad_(False).eval()
        self.stereo = load_stereo()
        # Retain the trained LAS2 feature pyramid, remove its image backbone.
        self.stereo.fnet.backbone = nn.Identity()
        self.adapters = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in
                                      zip([128, 256, 256, 512], [40, 80, 160, 320])])

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()  # Freeze BatchNorm buffers as well as parameters.
        return self

    def forward(self, left, right, semantic=True):
        with torch.no_grad():
            x = torch.cat([left, right]) / 255.0
            saved = []
            for layer in self.semantic.model[:11]:
                x = layer(x)
                saved.append(x)
            raw = [saved[i] for i in (2, 4, 6, 10)]
            logits = None
            if semantic:
                # Run the original semantic neck/head on the left features only.
                ys = [v[:left.shape[0]] for v in saved]
                x = ys[-1]
                for layer in self.semantic.model[11:]:
                    inp = x if layer.f == -1 else (
                        ys[layer.f] if isinstance(layer.f, int) else
                        [x if j == -1 else ys[j] for j in layer.f])
                    x = layer(inp)
                    ys.append(x)
                logits = x
        features = self.stereo.fnet([a(v) for a, v in zip(self.adapters, raw)])
        lf = [v[:left.shape[0]] for v in features]
        rf = [v[left.shape[0]:] for v in features]
        from core.submodule import build_correlation_volume, disparity_regression, context_upsample
        cost = build_correlation_volume(lf[0], rf[0], 48)
        disp = disparity_regression(self.stereo.cost_agg(cost, lf).softmax(1), 48)
        mask = self.stereo.refine_1(lf[0])
        mask = self.stereo.refine_2(mask, self.stereo.stem_2(self.stereo.normalize_image(left)))
        mask = self.stereo.refine_3(mask).softmax(1)
        full = context_upsample(disp * 4, mask.float())
        coarse = F.interpolate(disp * 4, size=left.shape[-2:], mode='bilinear', align_corners=False)
        return full, coarse, logits
