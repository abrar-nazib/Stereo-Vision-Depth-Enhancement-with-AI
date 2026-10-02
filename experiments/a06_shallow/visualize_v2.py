"""A07v2 — 2x2 metric collages for A07v2 fusion-arm runs.

For each selected image and each saved checkpoint this renders

    +---------------------+-------------------------+
    | Left RGB            | GT disparity (px)       |
    +---------------------+-------------------------+
    | Predicted disparity | metrics text panel      |
    +---------------------+-------------------------+

GT and prediction share one colour scale so they are directly comparable.
Disparity is shown rather than metric depth: every metric we report is in native
disparity pixels, and the Drive maps are not metric without focal length and
baseline.

Five train and five held-out images are chosen at evenly spaced indices from the
run's own manifests, so the selection is deterministic. Output PNGs are build
artifacts (gitignored); `summary.csv` is the durable record.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "runners"))
from model_a07v2 import FusionStereoLite  # noqa: E402


def _sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a02 = _sibling("a02_vis", ROOT / "experiments/lightstereo_s_a02/run.py")

MAX_DISP = 192


def select(records: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Cannot select {count} of {len(records)}")
    return [records[i] for i in np.linspace(0, len(records) - 1, count, dtype=int)]


def load_image(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def predict(model, left: np.ndarray, right: np.ndarray, device) -> np.ndarray:
    left_tensor = torch.from_numpy(left).permute(2, 0, 1)[None].float().to(device)
    right_tensor = torch.from_numpy(right).permute(2, 0, 1)[None].float().to(device)
    height, width = left_tensor.shape[-2:]
    top, pad_right = (-height) % 16, (-width) % 16
    pad = lambda t: torch.nn.functional.pad(t, (0, pad_right, top, 0), mode="replicate")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = model(pad(left_tensor), pad(right_tensor))
    return prediction[..., top:, :width].float().cpu()[0, 0].numpy()


def collage(rgb, ground_truth, prediction, row, title, path: Path) -> None:
    valid = np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < MAX_DISP)
    lower = float(ground_truth[valid].min())
    upper = float(np.percentile(ground_truth[valid], 99))
    error = np.where(valid, np.abs(prediction - ground_truth), np.nan)

    figure, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    figure.suptitle(title, fontsize=13)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Left RGB")
    image = axes[0, 1].imshow(np.where(valid, ground_truth, np.nan), cmap="turbo", vmin=lower, vmax=upper)
    axes[0, 1].set_title("Ground-truth disparity (px)")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)
    image = axes[1, 0].imshow(np.where(valid, prediction, np.nan), cmap="turbo", vmin=lower, vmax=upper)
    axes[1, 0].set_title(f"Predicted disparity (px)   mean |err| {np.nanmean(error):.3f} px")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
    panel = axes[1, 1]
    panel.set_facecolor("black")
    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(0.06, 0.94, "\n".join([
        f"EPE      {row['epe']:8.3f} px",
        f"RMSE     {row['rmse']:8.3f} px",
        f"median   {row['median']:8.3f} px",
        f"bad-0.5  {row['bad_0.5']:8.2f} %",
        f"bad-1    {row['bad_1']:8.2f} %",
        f"bad-2    {row['bad_2']:8.2f} %",
        f"bad-3    {row['bad_3']:8.2f} %",
        f"D1       {row['d1']:8.2f} %",
        f"valid    {row['valid_fraction'] * 100:8.2f} %",
    ]), va="top", ha="left", color="white", family="monospace", fontsize=13, transform=panel.transAxes)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--checkpoints", default="all", help="'all', 'final', or comma-separated steps")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    train_records = json.loads((run / "manifest_train.json").read_text())
    validation_records = json.loads((run / "manifest_validation.json").read_text())
    selections = {"train": select(train_records, args.train_count),
                  "validation": select(validation_records, args.val_count)}

    paths = sorted((run / "checkpoints").glob("step_*.pth"))
    if args.checkpoints == "all":
        chosen = paths
    elif args.checkpoints == "final":
        chosen = paths[-1:]
    else:
        wanted = {int(x) for x in args.checkpoints.split(",")}
        chosen = [p for p in paths if int(p.stem.split("_")[1]) in wanted]

    output = run / "visualizations"
    rows: list[dict[str, object]] = []
    cache = {split: [(load_image(r["left"]), load_image(r["right"]), a02.read_pfm(r["disparity"]),
                      Path(r["left"]).name) for r in records]
             for split, records in selections.items()}

    for path in chosen:
        step = int(path.stem.split("_")[1])
        model = FusionStereoLite(config["arm"],
                                 encoder=config.get("encoder")).to(device).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        tag = f"step_{step:06d}"
        (output / tag).mkdir(parents=True, exist_ok=True)
        for split, items in cache.items():
            for index, (left, right, ground_truth, name) in enumerate(items):
                prediction = predict(model, left, right, device)
                values = a02.metrics(torch.from_numpy(prediction)[None, None],
                                     torch.from_numpy(ground_truth)[None, None])
                values["valid_fraction"] = values["valid_pixels"] / ground_truth.size
                collage(left, ground_truth, prediction, values,
                        f"A07v2{config['arm']}  |  step {step}  |  {split}  |  {name}",
                        output / tag / f"{split}_{index:02d}.png")
                rows.append({"step": step, "split": split, "index": index, "image": name, **values})
        print(f"rendered {tag}: {len(cache['train']) + len(cache['validation'])} collages", flush=True)
        del model
        torch.cuda.empty_cache()

    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"COMPLETE {output}  ({len(rows)} collages)")


if __name__ == "__main__":
    main()
