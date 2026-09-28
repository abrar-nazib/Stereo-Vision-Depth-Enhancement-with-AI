"""Post-run visual comparison for the A02B ablation: 2x2 collages per image.

For each selected checkpoint (and the official LightStereo-S control) this
renders, for a handful of train and held-out images, a 2x2 collage of

    +---------------------+---------------------+
    | Left RGB            | GT disparity (px)   |
    +---------------------+---------------------+
    | Predicted disparity | metrics text panel  |
    +---------------------+---------------------+

Disparity is shown, not metric depth: every metric we report is in native
disparity pixels, and the Drive disparity maps are not metric without the
dataset's focal length and baseline. The GT and prediction panels share one
colour scale so they are directly comparable.

Generated PNGs are build artifacts and are gitignored; the metrics are also
written to ``summary.csv`` next to them.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import ROOT, YoloLightStereoS, load_official_lightstereo  # noqa: E402
from run import metrics, read_pfm  # noqa: E402
from run_yolo_backbone import pad_top_right, to_tensor  # noqa: E402

MAX_DISP = 192


def latest_run() -> Path:
    runs = sorted((ROOT / "experiments/lightstereo_s_a02/runs").glob("A02B_*"))
    if not runs:
        raise FileNotFoundError("No A02B run directories found")
    return runs[-1]


def load_record(record: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = cv2.cvtColor(cv2.imread(record["left"]), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(record["right"]), cv2.COLOR_BGR2RGB)
    return left, right, read_pfm(record["disparity"])


def select(records: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Cannot select {count} of {len(records)} records")
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    return [records[index] for index in indices]


def predict_hybrid(model: YoloLightStereoS, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_tensor, right_tensor, top, width = pad_top_right(to_tensor(left), to_tensor(right))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        prediction = model(left_tensor, right_tensor)[0]
    return prediction[..., top:, :width].float().cpu()[0, 0].numpy()


def predict_baseline(model: torch.nn.Module, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    left_tensor, right_tensor, top, width = pad_top_right(to_tensor(left), to_tensor(right))
    with torch.inference_mode():
        prediction = model({"left": (left_tensor / 255 - mean) / std, "right": (right_tensor / 255 - mean) / std})["disp_pred"]
    return prediction[..., top:, :width].float().cpu()[0, 0].numpy()


def metrics_text(row: dict[str, float]) -> str:
    return "\n".join([
        f"EPE      {row['epe']:8.3f} px",
        f"RMSE     {row['rmse']:8.3f} px",
        f"median   {row['median']:8.3f} px",
        f"bad-0.5  {row['bad_0.5']:8.2f} %",
        f"bad-1    {row['bad_1']:8.2f} %",
        f"bad-2    {row['bad_2']:8.2f} %",
        f"bad-3    {row['bad_3']:8.2f} %",
        f"D1       {row['d1']:8.2f} %",
        f"valid    {row['valid_fraction'] * 100:8.2f} %",
    ])


def collage(rgb: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray, row: dict[str, float], title: str, path: Path) -> None:
    valid = np.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < MAX_DISP)
    lower = float(ground_truth[valid].min())
    upper = float(np.percentile(ground_truth[valid], 99))
    ground_truth_panel = np.where(valid, ground_truth, np.nan)
    prediction_panel = np.where(valid, prediction, np.nan)
    error_panel = np.where(valid, np.abs(prediction - ground_truth), np.nan)

    figure, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    figure.suptitle(title, fontsize=14)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Left RGB")
    ground_truth_image = axes[0, 1].imshow(ground_truth_panel, cmap="turbo", vmin=lower, vmax=upper)
    axes[0, 1].set_title("Ground-truth disparity (px)")
    figure.colorbar(ground_truth_image, ax=axes[0, 1], fraction=0.046)
    prediction_image = axes[1, 0].imshow(prediction_panel, cmap="turbo", vmin=lower, vmax=upper)
    axes[1, 0].set_title(f"Predicted disparity (px)   |error| mean {np.nanmean(error_panel):.3f} px")
    figure.colorbar(prediction_image, ax=axes[1, 0], fraction=0.046)
    panel = axes[1, 1]
    panel.set_facecolor("black")
    panel.set_xticks([])
    panel.set_yticks([])
    panel.text(0.06, 0.94, metrics_text(row), va="top", ha="left", color="white",
               family="monospace", fontsize=13, transform=panel.transAxes)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def evaluate_one(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    values = metrics(torch.from_numpy(prediction)[None, None], torch.from_numpy(ground_truth)[None, None])
    values["valid_fraction"] = values["valid_pixels"] / ground_truth.size
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="A02B run directory (default: latest)")
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--val-count", type=int, default=5)
    parser.add_argument("--checkpoints", type=str, default="all",
                        help="'all', 'final', or a comma-separated list of step numbers")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the RTX 3050 visualizations")

    run = (args.run or latest_run()).resolve()
    train_records = json.loads((run / "manifest_train.json").read_text())
    validation_records = json.loads((run / "manifest_validation.json").read_text())
    selections = {
        "train": select(train_records, args.train_count),
        "validation": select(validation_records, args.val_count),
    }
    checkpoints = sorted((run / "checkpoints").glob("step_*.pth"))
    if args.checkpoints == "all":
        chosen = checkpoints
    elif args.checkpoints == "final":
        chosen = checkpoints[-1:]
    else:
        wanted = {int(item) for item in args.checkpoints.split(",")}
        chosen = [path for path in checkpoints if int(path.stem.split("_")[1]) in wanted]
    if not chosen:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoints!r} in {run / 'checkpoints'}")

    output = run / "visualizations"
    rows: list[dict[str, object]] = []
    loaded = {split: {name: load_record(record) for name, record in enumerate(records)} for split, records in selections.items()}

    def render(tag: str, step: int, predict) -> None:
        (output / tag).mkdir(parents=True, exist_ok=True)
        for split, records in selections.items():
            for name, record in enumerate(records):
                left, right, ground_truth = loaded[split][name]
                prediction = predict(left, right)
                values = evaluate_one(prediction, ground_truth)
                target = output / tag / f"{split}_{name:02d}.png"
                collage(left, ground_truth, prediction, values,
                        f"{tag}  |  step {step}  |  {split}  |  {Path(record['left']).name}", target)
                rows.append({"tag": tag, "step": step, "split": split, "index": name, **values})
            print(f"rendered {tag} {split}", flush=True)

    if not args.skip_baseline:
        baseline = load_official_lightstereo().cuda().eval()
        render("baseline", 0, lambda left, right: predict_baseline(baseline, left, right))
        del baseline
        torch.cuda.empty_cache()

    for path in chosen:
        step = int(path.stem.split("_")[1])
        model = YoloLightStereoS().cuda()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        model.eval()
        render(f"step_{step:06d}", step, lambda left, right: predict_hybrid(model, left, right))
        del model
        torch.cuda.empty_cache()

    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"COMPLETE {output}")


if __name__ == "__main__":
    main()
