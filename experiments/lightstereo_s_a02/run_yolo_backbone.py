"""Ablation Test 2B: frozen YOLO26s encoder + LightStereo-S head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from model import ROOT, YoloLightStereoS, load_official_lightstereo, native_crop
from run import build_manifest, metrics, read_pfm, write_json


def split_manifest(records: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for record in records:
        groups.setdefault(record["sequence"], []).append(record)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    randomizer = random.Random(seed)
    for sequence in sorted(groups):
        group = sorted(groups[sequence], key=lambda item: item["left"])
        randomizer.shuffle(group)
        validation_count = round(len(group) * 0.2)
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    if len(train) + len(validation) != len(records) or not train or not validation:
        raise RuntimeError("Invalid 80/20 split")
    return train, validation


def load_record(record: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = cv2.cvtColor(cv2.imread(record["left"]), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(record["right"]), cv2.COLOR_BGR2RGB)
    return left, right, read_pfm(record["disparity"])


def preload(records: list[dict[str, str]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode every pair once so no step pays external-drive image I/O."""
    return [load_record(record) for record in records]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)[None].float().cuda(non_blocking=True)


def pad_top_right(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    height, width = left.shape[-2:]
    top, right_pad = (-height) % 32, (-width) % 32
    return (
        F.pad(left, (0, right_pad, top, 0), mode="replicate"),
        F.pad(right, (0, right_pad, top, 0), mode="replicate"),
        top,
        width,
    )


def evaluate_hybrid(model: YoloLightStereoS, cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]], *, semantic: bool = False) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for index, (left, right, ground_truth) in enumerate(cache):
            left_tensor, right_tensor, top, width = pad_top_right(to_tensor(left), to_tensor(right))
            prediction, _, _ = model(left_tensor, right_tensor, semantic=semantic and index == 0)
            prediction = prediction[..., top:, :width].cpu()
            values = {"index": float(index), **metrics(prediction, torch.from_numpy(ground_truth)[None, None])}
            values["valid_fraction"] = values["valid_pixels"] / ground_truth.size
            rows.append(values)
    return ({key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "index"}, rows)


def evaluate_baseline(cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, float]:
    model = load_official_lightstereo().cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for left, right, ground_truth in cache:
            left_tensor, right_tensor, top, width = pad_top_right(to_tensor(left), to_tensor(right))
            prediction = model({"left": (left_tensor / 255 - mean) / std, "right": (right_tensor / 255 - mean) / std})["disp_pred"]
            rows.append(metrics(prediction[..., top:, :width].cpu(), torch.from_numpy(ground_truth)[None, None]))
    del model
    torch.cuda.empty_cache()
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def plot_training(path: Path, train_rows: list[dict[str, float]], validation_rows: list[dict[str, float]]) -> None:
    fig, (loss_axis, epe_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], label="crop Smooth L1")
    loss_axis.set(xlabel="step", ylabel="loss", title="Training loss")
    loss_axis.grid(alpha=0.3)
    epe_axis.plot([row["step"] for row in train_rows], [row["epe"] for row in train_rows], alpha=0.45, label="train crop EPE")
    epe_axis.plot([row["step"] for row in validation_rows], [row["epe"] for row in validation_rows], marker="o", label="held-out native EPE")
    epe_axis.set(xlabel="step", ylabel="EPE (px)", title="No-resize convergence")
    epe_axis.grid(alpha=0.3)
    epe_axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-height", type=int, default=384)
    parser.add_argument("--crop-width", type=int, default=640)
    parser.add_argument("--data", type=Path, default=Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested RTX 3050 ablation")
    if args.crop_height % 32 or args.crop_width % 32:
        raise ValueError("Native crop dimensions must be divisible by 32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = ROOT / "experiments/lightstereo_s_a02/runs" / (
        "A02B_yolo26s_frozen_native_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    all_records = build_manifest(args.data, 200)
    train_records, validation_records = split_manifest(all_records, args.seed)
    write_json(output / "manifest_all.json", all_records)
    write_json(output / "manifest_train.json", train_records)
    write_json(output / "manifest_validation.json", validation_records)
    train_cache = preload(train_records)
    validation_cache = preload(validation_records)
    checkpoint_paths = [
        ROOT / "models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt",
        ROOT / "models/segmentation/yolo26s-sem-cityscapes.pt",
    ]
    config = {
        "experiment": "Ablation Test 2B",
        "architecture": "frozen YOLO26s-sem encoder; four learned 1x1 adapters [128,256,256,512]->[24,32,96,160]; pretrained LightStereo-S aggregation/refinement head fine-tuned",
        "data": "SceneFlow Driving 200 fixed pairs; stratified deterministic 160 train / 40 validation",
        "training": {"steps": args.steps, "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 1, "precision": "AMP float16", "loss": "smooth_l1(full)+0.3*smooth_l1(coarse)", "crop": [args.crop_height, args.crop_width]},
        "evaluation": "native 540x960 full image, top/right replicate pad to /32, no resize",
        "metric_mask": "finite GT, 0 < disparity < 192 native pixels",
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint_sha256": {str(path.relative_to(ROOT)): hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for path in checkpoint_paths},
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("model.py", "run_yolo_backbone.py")},
    }
    write_json(output / "config.json", config)
    baseline = evaluate_baseline(validation_cache)
    write_json(output / "baseline_official_lightstereo_s_validation.json", baseline)

    model = YoloLightStereoS().cuda()
    parameter_counts = {"total": sum(parameter.numel() for parameter in model.parameters()), "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    frozen_before = {key: value.detach().cpu().clone() for key, value in model.encoder.semantic.state_dict().items()}
    initial, initial_rows = evaluate_hybrid(model, validation_cache, semantic=True)
    write_json(output / "hybrid_initial_validation.json", initial)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    randomizer = random.Random(args.seed)
    order: list[int] = []
    train_rows: list[dict[str, float]] = []
    validation_rows: list[dict[str, float]] = [{"step": 0.0, "epe": initial["epe"]}]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / "train.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "epe", "elapsed_seconds"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            if not order:
                order = list(range(len(train_records)))
                randomizer.shuffle(order)
            left, right, ground_truth = train_cache[order.pop()]
            maximum_top = left.shape[0] - args.crop_height
            maximum_left = left.shape[1] - args.crop_width
            crop_top, crop_left = randomizer.randint(0, maximum_top), randomizer.randint(0, maximum_left)
            left, right, ground_truth = native_crop(left, right, ground_truth, height=args.crop_height, width=args.crop_width, top=crop_top, left_offset=crop_left)
            left_tensor, right_tensor = to_tensor(left), to_tensor(right)
            target = torch.from_numpy(ground_truth)[None, None].cuda()
            valid = torch.isfinite(target) & (target > 0) & (target < 192)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction, coarse, _ = model(left_tensor, right_tensor)
                loss = F.smooth_l1_loss(prediction[valid], target[valid]) + 0.3 * F.smooth_l1_loss(coarse[valid], target[valid])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {"step": float(step), "loss": loss.item(), "epe": (prediction[valid] - target[valid]).abs().mean().item(), "elapsed_seconds": time.perf_counter() - started}
            writer.writerow(row)
            train_rows.append(row)
            if step % args.eval_every == 0 or step == args.steps:
                values, _ = evaluate_hybrid(model, validation_cache)
                validation_row = {"step": float(step), "epe": values["epe"]}
                validation_rows.append(validation_row)
                checkpoint_payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": step, "validation": values}
                torch.save(checkpoint_payload, checkpoints / f"step_{step:06d}.pth")
                torch.save(checkpoint_payload, checkpoints / "latest.pth")
                print(f"step={step} loss={row['loss']:.4f} crop_epe={row['epe']:.4f} val_native_epe={values['epe']:.4f}", flush=True)
    final, final_rows = evaluate_hybrid(model, validation_cache, semantic=True)
    frozen_unchanged = all(torch.equal(value.cpu(), frozen_before[key]) for key, value in model.encoder.semantic.state_dict().items())
    if not frozen_unchanged:
        raise RuntimeError("Frozen YOLO semantic weights or buffers changed")
    elapsed = time.perf_counter() - started
    write_json(output / "hybrid_final_validation.json", final)
    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "epe"])
        writer.writeheader()
        writer.writerows(validation_rows)
    with (output / "validation_final_per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)
    plot_training(output / "training_curve.png", train_rows, validation_rows)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": args.steps}, output / "final.pth")
    results = {"baseline": baseline, "initial": initial, "final": final, "parameters": parameter_counts, "training_seconds": elapsed, "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20, "semantic_weights_and_buffers_unchanged": frozen_unchanged}
    write_json(output / "results.json", results)
    report = f"""# Ablation Test 2B — frozen YOLO26s encoder + LightStereo-S head\n\n- Fixed split: 160 train / 40 held-out validation frames from the 200-pair A02 manifest.\n- Training: {args.steps} updates of co-located native {args.crop_height}×{args.crop_width} crops; no image or disparity resizing.\n- Validation: full native 540×960 frames with only top/right stride-32 padding.\n- Frozen YOLO semantic weights/buffers unchanged: {frozen_unchanged}.\n\n| Model | EPE | RMSE | bad-1 % | bad-3 % | D1 % |\n|---|---:|---:|---:|---:|---:|\n| Official LightStereo-S control | {baseline['epe']:.3f} | {baseline['rmse']:.3f} | {baseline['bad_1']:.3f} | {baseline['bad_3']:.3f} | {baseline['d1']:.3f} |\n| YOLO-adapted initial | {initial['epe']:.3f} | {initial['rmse']:.3f} | {initial['bad_1']:.3f} | {initial['bad_3']:.3f} | {initial['d1']:.3f} |\n| YOLO-adapted after {args.steps} updates | {final['epe']:.3f} | {final['rmse']:.3f} | {final['bad_1']:.3f} | {final['bad_3']:.3f} | {final['d1']:.3f} |\n\n![Training curve](training_curve.png)\n\nThis is a capacity test, not a final SceneFlow generalization result: the 40 validation frames are held out from optimization, but all 200 pairs come from the Driving subset. See `config.json`, manifests, CSV traces, and `results.json` for reproducibility.\n"""
    (output / "REPORT.md").write_text(report)
    print(f"COMPLETE {output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
