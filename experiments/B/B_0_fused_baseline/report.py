"""Summarize a completed B-series run and render its learning curves.

    uv run python -m experiments.B.B_0_fused_baseline.report --run-id 20261001_b_full_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.B.B_0_fused_baseline.run import ARM_DIRS, ROOT
from experiments.vkitti2.semantic_data import CLASS_NAMES


def collect(run_id: str):
    base = ROOT / "experiments/B/B_0_fused_baseline/runs" / run_id
    baseline = {"validation": json.loads((base / "validation.json").read_text()),
                "test": json.loads((base / "test.json").read_text())}
    arms = {name: json.loads((folder / "runs" / run_id / "summary.json").read_text())
            for name, folder in ARM_DIRS.items()}
    return base, baseline, arms


def render(run_id: str) -> Path:
    base, baseline, arms = collect(run_id)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].axhline(baseline["validation"]["epe"], color="black", linestyle="--",
                    label="B0 frozen baseline")
    for name, folder in ARM_DIRS.items():
        path = folder / "runs" / run_id / "history.jsonl"
        history = [json.loads(line) for line in path.read_text().splitlines()]
        val = [row for row in history if row["phase"] == "val"]
        axes[0].plot([row["step"] for row in val], [row["epe"] for row in val],
                     marker="o", label=name)
        train = [row for row in history if row["phase"] == "train"]
        axes[1].plot([row["step"] for row in train], [row["loss"] for row in train],
                     alpha=0.7, label=name)
    axes[0].set(ylabel="Validation EPE (px)", xlabel="Optimizer step",
                title="Scene20 validation: 100 native-resolution pairs")
    axes[1].set(ylabel="Training smooth-L1 loss", xlabel="Optimizer step",
                title="One crop per step; raw loss is scene-dependent")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    plot = base / "B_series_progress.png"
    fig.savefig(plot, dpi=180)
    plt.close(fig)

    header = "| Arm | Selected step | Val EPE | Test EPE | Test RMSE | Test bad-1 | Test bad-3 | Test D1 | Test mIoU | Trainable params |"
    lines = ["# B-series 1,000-pair ablation report", "",
             f"Run ID: `{run_id}`. 800 train / 100 validation / 100 test; source-frame grouped.",
             "Both pretrained predictors were frozen; only B1–B3 disparity heads trained.",
             "All disparity metrics are native-resolution; EPE and RMSE are in pixels, bad/D1 in percent.",
             "", header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    b_val, b_test = baseline["validation"], baseline["test"]
    lines.append(f"| B0 frozen | — | {b_val['epe']:.4f} | {b_test['epe']:.4f} | "
                 f"{b_test['rmse']:.4f} | {b_test['bad_1']:.2f} | {b_test['bad_3']:.2f} | "
                 f"{b_test['d1']:.2f} | {b_test['miou']:.4f} | 0 |")
    for name, result in arms.items():
        test = result["test"]
        lines.append(f"| {name} | {result['selected_step']} | {result['best_validation_epe']:.4f} | "
                     f"{test['epe']:.4f} | {test['rmse']:.4f} | {test['bad_1']:.2f} | "
                     f"{test['bad_3']:.2f} | {test['d1']:.2f} | {test['miou']:.4f} | "
                     f"{result['trainable_params']:,} |")
    lines += ["", "## Frozen semantic output on Scene20 test", "",
              "IoU is reported only for classes with nonzero prediction/label union;"
              " an absent class is shown as a dash. The disparity heads cannot change these values.",
              "", "| Class | IoU |", "| --- | ---: |"]
    for class_id, label in enumerate(CLASS_NAMES):
        value = b_test["class_iou"].get(str(class_id))
        lines.append(f"| {label} | {value:.4f} |" if value is not None else f"| {label} | — |")
    lines += ["", "![B-series training and validation curves](B_series_progress.png)", "",
              "## Causal controls (validation EPE)", "",
              "Uniform or spatially shifted semantic logits are passed to the same selected head.",
              "A useful semantic head should lose its gain when guidance is invalid.", "",
              "| Arm | Normal | Uniform | Shifted |", "| --- | ---: | ---: | ---: |"]
    for name, result in arms.items():
        controls = result["validation_controls"]
        lines.append(f"| {name} | {result['best_validation_epe']:.4f} | "
                     f"{controls['uniform']['epe']:.4f} | {controls['shifted']['epe']:.4f} |")
    uncertainty_path = base / "uncertainty.json"
    if uncertainty_path.exists():
        uncertainty = json.loads(uncertainty_path.read_text())
        lines += ["", "## Paired frame-group uncertainty for B1 − B0 EPE", "",
                  "Ten Scene20 source frames per partition were resampled with replacement (10,000 draws);"
                  " all ten variants of a frame stay together.", "",
                  "| Partition | EPE delta (px) | 95% bootstrap interval (px) |",
                  "| --- | ---: | ---: |"]
        for partition in ("val", "test"):
            score = uncertainty[partition]["bootstrap"]
            lines.append(f"| {partition} | {score['delta_epe_px']:.4f} | "
                         f"[{score['ci95_low_px']:.4f}, {score['ci95_high_px']:.4f}] |")
    latency_path = base / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text())
        lines += ["", "## End-to-end GPU latency", "",
                  "Native 375×1242 input; includes fusion and class-confidence output, excludes image I/O.",
                  "The extra photometric warp makes B2 substantially slower on this RTX 3050.", "",
                  "| Arm | Median (ms) | p95 (ms) |", "| --- | ---: | ---: |"]
        for name, numbers in latency["results"].items():
            lines.append(f"| {name} | {numbers['median_ms']:.2f} | {numbers['p95_ms']:.2f} |")
    lines += ["", "## Qualitative held-out examples", "",
              "Each panel uses the same disparity color scale for GT, B0 and B1; error scales are shared too."]
    for variation in ("clone", "rain", "fog"):
        matching = sorted((base / "figures").glob(f"Scene20_{variation}_*.png"))
        if matching:
            lines.append(f"![Scene20 {variation}](figures/{matching[0].name})")
    lines += ["", "## Interpretation limits", "",
              "The semantic checkpoint was fine-tuned using a random full-VKITTI split that likely overlaps this subset."
              " This is an in-domain fusion feasibility test, not independent real-world generalization.",
              "Scene20 validation and test each contain only ten independent source frames; their ten"
              " variations per frame are correlated.", "",
              "Do not compare these EPE values directly to the previous 800/200 split or to the semantic"
              " checkpoint's full-dataset random-split mIoU.", ""]
    report = base / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261001_b_full_seed42")
    args = parser.parse_args()
    print(render(args.run_id))


if __name__ == "__main__":
    main()
