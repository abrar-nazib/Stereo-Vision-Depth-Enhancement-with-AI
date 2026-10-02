"""Native-resolution left-view fused output: disparity, class ID, confidence.

Pixel coordinates ``u,v`` are the tensor column/row indices. The maps jointly
represent ``[u, v, d, [c, C]]`` without materializing redundant coordinate grids.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic


class FusedPredictor(nn.Module):
    def __init__(self, base: FusedStereoSemantic, head: nn.Module | None = None):
        super().__init__()
        self.base = base.eval()
        self.head = head.eval() if head is not None else None

    @torch.inference_mode()
    def forward(self, left: torch.Tensor, right: torch.Tensor,
                parallel: bool = False) -> dict[str, torch.Tensor]:
        if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3:
            raise ValueError("expected matching NCHW left/right RGB tensors")
        left = left.float()
        right = right.float()
        height, width = left.shape[-2:]
        top = (-height) % 32
        extra_right = (-width) % 32
        left_pad = F.pad(left, (0, extra_right, top, 0), mode="replicate")
        right_pad = F.pad(right, (0, extra_right, top, 0), mode="replicate")
        amp = torch.autocast("cuda", dtype=torch.float16) if left.is_cuda else nullcontext()
        with amp:
            disparity, logits = self.base(left_pad, right_pad, parallel=parallel)
        if self.head is not None:
            disparity = self.head(disparity.float(), logits.float(),
                                  left_pad / 255.0, right_pad / 255.0)
        disparity = disparity.float()[..., top:top + height, :width]
        class_logits = F.interpolate(logits.float(), size=left_pad.shape[-2:],
                                     mode="bilinear", align_corners=False)
        class_logits = class_logits[..., top:top + height, :width]
        confidence, class_id = class_logits.softmax(dim=1).max(dim=1, keepdim=True)
        return {"disparity_px": disparity, "class_id": class_id,
                "class_confidence": confidence}


def dense_uvdcc(result: dict[str, torch.Tensor]) -> torch.Tensor:
    """Materialize `[u, v, disparity_px, class_id, class_confidence]` per pixel.

    Use the separate maps for normal inference; this layout is convenient for
    point-cloud export and costs five full-resolution float channels.
    """
    disparity = result["disparity_px"][:, 0]
    class_id = result["class_id"][:, 0].to(disparity.dtype)
    confidence = result["class_confidence"][:, 0].to(disparity.dtype)
    batch, height, width = disparity.shape
    v, u = torch.meshgrid(torch.arange(height, device=disparity.device),
                          torch.arange(width, device=disparity.device), indexing="ij")
    return torch.stack((u.expand(batch, -1, -1).to(disparity.dtype),
                        v.expand(batch, -1, -1).to(disparity.dtype),
                        disparity, class_id, confidence), dim=-1)
