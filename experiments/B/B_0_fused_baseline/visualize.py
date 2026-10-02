"""Render native-resolution held-out examples with the selected B1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.B.B_0_fused_baseline.predict import FusedPredictor
from experiments.B.B_0_fused_baseline.run import ADE, DATA, ROOT, SEMANTIC, STEREO, read_pair, to_tensors
from experiments.B.B_1_residual.head import ResidualHead
from experiments.vkitti2.semantic_data import CLASS_COLORS


def _color_disparity(values: np.ndarray, valid: np.ndarray, cap: float):
    rgba = plt.get_cmap("turbo")(np.clip(values / cap, 0, 1))[..., :3]
    rgba[~valid] = 0
    return rgba


def _color_labels(values: np.ndarray):
    palette = np.asarray(CLASS_COLORS, dtype=np.float32) / 255.0
    rgb = palette[np.clip(values, 0, 13)]
    rgb[values == 255] = 0
    return rgb


def render(run_id: str) -> list[Path]:
    run = ROOT / "experiments/B/B_0_fused_baseline/runs" / run_id
    split = json.loads((run / "split.json").read_text())
    chosen = []
    for variation in ("clone", "rain", "fog"):
        chosen.append(next(row for row in split["test"] if row["variation"] == variation))
    head = ResidualHead().cuda().eval()
    state = torch.load(ROOT / "experiments/B/B_1_residual/runs" / run_id / "best_epe.pth",
                       map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    base = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
    baseline, fused = FusedPredictor(base), FusedPredictor(base, head)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    paths = []
    for row in chosen:
        sample = read_pair(DATA, row)
        left, right, gt, labels = to_tensors(sample)
        before = baseline(left, right)
        after = fused(left, right)
        rgb = sample[0]
        true = sample[2]
        valid = (true > 0) & (true < 192)
        pred0 = before["disparity_px"][0, 0].cpu().numpy()
        pred1 = after["disparity_px"][0, 0].cpu().numpy()
        cap = float(np.percentile(true[valid], 99))
        cap = max(cap, 1.0)
        error_cap = max(float(np.percentile(np.abs(pred0[valid] - true[valid]), 99)), 1.0)
        fig, axes = plt.subplots(2, 4, figsize=(18, 4.75))
        panels = (
            (rgb, "Left RGB"),
            (_color_disparity(true, valid, cap), f"GT disparity (0–{cap:.0f} px)"),
            (_color_disparity(pred0, valid, cap), "B0 frozen disparity"),
            (_color_disparity(pred1, valid, cap), "B1 semantic-guided disparity"),
            (_color_labels(sample[3]), "GT semantic classes"),
            (_color_labels(after["class_id"][0, 0].cpu().numpy()), "Frozen YOLO 14-class prediction"),
            (_color_disparity(np.abs(pred0 - true), valid, error_cap), "B0 absolute error"),
            (_color_disparity(np.abs(pred1 - true), valid, error_cap), "B1 absolute error"),
        )
        for axis, (picture, title) in zip(axes.flat, panels):
            axis.imshow(picture)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        fig.suptitle(f"Scene20 · {row['variation']} · frame {row['frame']} · native {true.shape[1]}×{true.shape[0]}",
                     fontsize=13)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.015, top=0.87,
                            wspace=0.02, hspace=0.23)
        path = figures / f"Scene20_{row['variation']}_{row['frame']:05d}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261001_b_full_seed42")
    args = parser.parse_args()
    for path in render(args.run_id):
        print(path)


if __name__ == "__main__":
    main()
