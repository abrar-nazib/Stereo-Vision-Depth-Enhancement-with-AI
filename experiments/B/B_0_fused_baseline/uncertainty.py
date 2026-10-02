"""Paired source-frame bootstrap for B1 versus frozen B0 disparity error."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.B.B_0_fused_baseline.run import ADE, DATA, ROOT, SEMANTIC, STEREO, pad32, read_pair, to_tensors
from experiments.B.B_1_residual.head import ResidualHead


def bootstrap_delta(groups: dict, draws: int = 10000, seed: int = 42) -> dict:
    """Bootstrap frame groups, not their correlated weather/camera variants."""
    values = list(groups.values())
    if not values or draws < 1:
        raise ValueError("need frame groups and a positive draw count")
    base = np.array([item["baseline_abs_error"] for item in values], dtype=np.float64)
    fused = np.array([item["fused_abs_error"] for item in values], dtype=np.float64)
    pixels = np.array([item["valid_pixels"] for item in values], dtype=np.float64)
    if np.any(pixels <= 0):
        raise ValueError("each group needs valid disparity pixels")
    delta = float((fused.sum() - base.sum()) / pixels.sum())
    choices = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    samples = (fused[choices].sum(axis=1) - base[choices].sum(axis=1)) / pixels[choices].sum(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {"delta_epe_px": delta, "ci95_low_px": float(low),
            "ci95_high_px": float(high), "frame_groups": len(values),
            "bootstrap_draws": draws}


@torch.inference_mode()
def collect_groups(model, head, rows):
    groups = {}
    for row in rows:
        left, right, gt, _ = to_tensors(read_pair(DATA, row))
        lp, top = pad32(left)
        rp, _ = pad32(right)
        with torch.autocast("cuda", dtype=torch.float16):
            baseline, logits = model(lp, rp)
        corrected = head(baseline.float(), logits.float(), lp / 255.0, rp / 255.0)
        baseline = baseline[..., top:top + gt.shape[-2], :gt.shape[-1]]
        corrected = corrected[..., top:top + gt.shape[-2], :gt.shape[-1]]
        valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
        entry = groups.setdefault(str(row["frame"]),
                                  {"baseline_abs_error": 0.0, "fused_abs_error": 0.0,
                                   "valid_pixels": 0})
        entry["baseline_abs_error"] += (baseline.float() - gt).abs()[valid].sum().item()
        entry["fused_abs_error"] += (corrected.float() - gt).abs()[valid].sum().item()
        entry["valid_pixels"] += int(valid.sum())
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261001_b_full_seed42")
    args = parser.parse_args()
    run = ROOT / "experiments/B/B_0_fused_baseline/runs" / args.run_id
    split = json.loads((run / "split.json").read_text())
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
    head = ResidualHead().cuda().eval()
    payload = torch.load(ROOT / "experiments/B/B_1_residual/runs" / args.run_id / "best_epe.pth",
                         map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"])
    result = {}
    for partition in ("val", "test"):
        groups = collect_groups(model, head, split[partition])
        result[partition] = {"bootstrap": bootstrap_delta(groups), "groups": groups}
        print(partition, result[partition]["bootstrap"], flush=True)
    (run / "uncertainty.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
