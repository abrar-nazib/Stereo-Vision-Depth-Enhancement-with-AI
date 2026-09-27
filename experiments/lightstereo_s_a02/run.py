"""Ablation Test 2: native-resolution LightStereo-S benchmark on Driving."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_manifest(data_root: Path, count: int) -> list[dict[str, str]]:
    """Choose unique frames evenly across SceneFlow Driving's sequences."""
    groups = sorted((data_root / "frames_finalpass").glob("*/*/*/left"))
    if not groups:
        raise FileNotFoundError(f"No Driving left-image sequences below {data_root}")
    records: list[dict[str, str]] = []
    for sequence_index, group in enumerate(groups):
        paths = sorted(group.glob("*.png"))
        n = count // len(groups) + (sequence_index < count % len(groups))
        if n > len(paths):
            raise ValueError(f"{group} has only {len(paths)} frames, requires {n}")
        for index in np.linspace(0, len(paths) - 1, n, dtype=int):
            left = paths[index]
            relative = left.relative_to(data_root / "frames_finalpass")
            records.append({
                "left": str(left),
                "right": str(left.parent.parent / "right" / left.name),
                "disparity": str((data_root / "disparity" / relative).with_suffix(".pfm")),
                "sequence": str(group.relative_to(data_root)),
            })
    if len(records) != count or len({record["left"] for record in records}) != count:
        raise RuntimeError("Manifest does not contain the requested number of unique left frames")
    return records


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"{path} is not a grayscale PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        array = np.fromfile(stream, "<f4" if scale < 0 else ">f4").reshape(height, width)
    return np.flipud(array).copy() * abs(scale)


def metrics(prediction: torch.Tensor, ground_truth: torch.Tensor) -> dict[str, float]:
    valid = torch.isfinite(ground_truth) & (ground_truth > 0) & (ground_truth < 192)
    error = (prediction.float() - ground_truth).abs()[valid]
    truth = ground_truth[valid]
    values = {
        "epe": error.mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "median": error.median().item(),
        "valid_pixels": float(error.numel()),
    }
    for threshold in (0.5, 1.0, 2.0, 3.0):
        values[f"bad_{threshold:g}"] = (error > threshold).float().mean().item() * 100
    values["d1"] = ((error > 3) & (error > 0.05 * truth)).float().mean().item() * 100
    return values


def load_model() -> torch.nn.Module:
    import sys

    adapter_dir = ROOT / "experiments" / "yolo_las2_v1"
    sys.path.insert(0, str(adapter_dir))
    from lightstereo_adapter import build_lightstereo

    checkpoint = ROOT / "models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt"
    model = build_lightstereo()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = state.get("model_state", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return model.cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--data", type=Path, default=Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving"))
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this requested RTX 3050 benchmark")

    output = ROOT / "experiments/lightstereo_s_a02/runs" / (
        "A02_lightstereo_s_native_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(args.data, args.count)
    write_json(output / "manifest.json", manifest)
    checkpoint = ROOT / "models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt"
    write_json(output / "config.json", {
        "experiment": "Ablation Test 2",
        "architecture": "Official LightStereo-S control; unmodified MobileNetV2 backbone",
        "protocol": "native 540x960, 200 fixed SceneFlow Driving pairs, no resize, no training",
        "training_checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
        "count": args.count,
        "metric_mask": "finite GT, 0 < disparity < 192 native pixels",
        "padding": "official RightTopPad equivalent: replicate top/right to multiple of 32, removed before scoring",
        "precision": "FP32",
        "gpu": torch.cuda.get_device_name(),
    })

    model = load_model()
    rows: list[dict[str, float]] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, record in enumerate(manifest):
            images = [cv2.cvtColor(cv2.imread(record[key]), cv2.COLOR_BGR2RGB) for key in ("left", "right")]
            left, right = [torch.from_numpy(image).permute(2, 0, 1)[None].float().cuda() for image in images]
            ground_truth = torch.from_numpy(read_pfm(record["disparity"]))[None, None]
            height, width = left.shape[-2:]
            top, right_pad = (-height) % 32, (-width) % 32
            left = torch.nn.functional.pad(left, (0, right_pad, top, 0), mode="replicate")
            right = torch.nn.functional.pad(right, (0, right_pad, top, 0), mode="replicate")
            if index == 0:
                for _ in range(10):
                    model({"left": (left / 255 - mean) / std, "right": (right / 255 - mean) / std})
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model({"left": (left / 255 - mean) / std, "right": (right / 255 - mean) / std})["disp_pred"]
            torch.cuda.synchronize()
            prediction = prediction[..., top:, :width].cpu()
            row = {"index": float(index), "latency_ms": (time.perf_counter() - start) * 1000,
                   **metrics(prediction, ground_truth)}
            row["valid_fraction"] = row["valid_pixels"] / ground_truth.numel()
            rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"{index + 1}/{len(manifest)}: EPE={np.mean([item['epe'] for item in rows]):.4f}", flush=True)

    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "index"}
    latencies = [row["latency_ms"] for row in rows]
    summary.update({
        "model": "LightStereo-S-SceneFlow",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fps": 1000 / summary["latency_ms"],
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
    })
    write_json(output / "results.json", summary)
    print(f"COMPLETE {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
