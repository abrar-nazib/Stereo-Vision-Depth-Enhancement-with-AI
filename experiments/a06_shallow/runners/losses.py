"""A05 — multi-scale stereo loss ported from the modal training script.

Source: stero_research_claude/model/scripts/train_arch_sceneflow.py
(train_loss + ms_l1 + grad_consistency + threshold_hinge + d1_hinge +
edge_smooth). Ported verbatim except:

* Heads here return full-res disparity only (no d_half/d4/d8/d16
  intermediates), so the multi-scale L1 terms are computed by
  downsampling prediction and GT — exactly what modal ms_l1 does
  internally when shapes differ (interpolate pred, scale-aware).
* guided_guard terms are omitted (gev4_guided heads only; none of the
  A05 heads emit d_pre_guided / guided_gate).
* smooth_w default 0.02 matches modal.

Weight recipe (modal defaults):
  1.0 * ms_l1(full) + 0.5 * ms_l1(1/2) + 0.3 * ms_l1(1/4)
  + 0.2 * grad_consistency + 0.2 * threshold_hinge + 0.2 * d1_hinge
  + 0.02 * edge_smooth
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def valid_mask(gt: torch.Tensor, max_disp: float) -> torch.Tensor:
    return ((gt > 0) & (gt < max_disp) & torch.isfinite(gt)).float()


def ms_l1(pred: torch.Tensor, gt: torch.Tensor, val: torch.Tensor,
          scale: float) -> torch.Tensor:
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, size=gt.shape[-2:],
                             mode="bilinear", align_corners=False) * scale
    return ((pred - gt).abs() * val).sum() / val.sum().clamp(min=1)


def grad_consistency(pred: torch.Tensor, gt: torch.Tensor,
                     val: torch.Tensor) -> torch.Tensor:
    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gx_g = gt[..., :, 1:] - gt[..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_g = gt[..., 1:, :] - gt[..., :-1, :]
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = ((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
    ly = ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def threshold_hinge(pred: torch.Tensor, gt: torch.Tensor,
                    val: torch.Tensor) -> torch.Tensor:
    err = (pred - gt).abs()
    total = (
        (err - 0.5).clamp(min=0) ** 2
        + (err - 1.0).clamp(min=0) ** 2
        + (err - 2.0).clamp(min=0) ** 2
        + (err - 3.0).clamp(min=0) ** 2
    )
    return (total * val).sum() / val.sum().clamp(min=1)


def d1_hinge(pred: torch.Tensor, gt: torch.Tensor,
             val: torch.Tensor) -> torch.Tensor:
    err = (pred - gt).abs()
    rel = err / gt.clamp(min=1e-6)
    is_d1 = ((err > 3.0) & (rel > 0.05)).float()
    return (((err - 3.0).clamp(min=0) ** 2) * is_d1 * val).sum() / (
        val.sum().clamp(min=1))


def to_unit_image(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0 if x.detach().amax() > 2.0 else x


def edge_smooth(pred: torch.Tensor, left: torch.Tensor,
                val: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    grey = to_unit_image(left).mean(dim=1, keepdim=True)
    gx_i = (grey[..., :, 1:] - grey[..., :, :-1]).abs()
    gy_i = (grey[..., 1:, :] - grey[..., :-1, :]).abs()
    gx_d = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
    gy_d = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    vx = val[..., :, 1:] * val[..., :, :-1]
    vy = val[..., 1:, :] * val[..., :-1, :]
    lx = (gx_d * torch.exp(-alpha * gx_i) * vx).sum() / vx.sum().clamp(min=1)
    ly = (gy_d * torch.exp(-alpha * gy_i) * vy).sum() / vy.sum().clamp(min=1)
    return lx + ly


def modal_loss(pred_full: torch.Tensor, gt: torch.Tensor,
               left: torch.Tensor, max_disp: float = 192.0,
               smooth_w: float = 0.02) -> tuple[torch.Tensor, dict[str, float]]:
    """Full modal recipe from a single full-res prediction.

    Multi-scale L1 is evaluated at full, 1/2 and 1/4 by average-pooling
    the prediction and GT (equivalent to modal ms_l1 with mismatched
    shapes, but cheaper than interpolate per scale).
    """
    val = valid_mask(gt, max_disp)
    d_full = pred_full
    if d_full.shape[-2:] != gt.shape[-2:]:
        d_full = F.interpolate(d_full, size=gt.shape[-2:],
                               mode="bilinear", align_corners=True)

    def down(x: torch.Tensor, s: int) -> torch.Tensor:
        return F.avg_pool2d(x, s, s, count_include_pad=False)

    d_half, gt_half, val_half = down(d_full, 2), down(gt, 2), down(val, 2)
    d_q, gt_q, val_q = down(d_full, 4), down(gt, 4), down(val, 4)

    loss = (
        1.0 * ((d_full - gt).abs() * val).sum() / val.sum().clamp(min=1)
        + 0.5 * ((d_half - gt_half).abs() * val_half).sum() / val_half.sum().clamp(min=1)
        + 0.3 * ((d_q - gt_q).abs() * val_q).sum() / val_q.sum().clamp(min=1)
        + 0.5 * grad_consistency(d_full, gt, val)
        + 0.2 * threshold_hinge(d_full, gt, val)
        + 0.2 * d1_hinge(d_full, gt, val)
    )
    if smooth_w > 0:
        loss = loss + smooth_w * edge_smooth(d_full, left, val)

    with torch.no_grad():
        err = (d_full - gt).abs()
        denom = val.sum().clamp(min=1)
        diag = {
            "epe": float((err * val).sum() / denom),
            "bad1": float((((err > 1.0).float() * val).sum() / denom) * 100),
        }
    return loss, diag
