"""A03 — encoder swap on the StereoLite HITNet-style head.

Two arms, identical head, identical protocol, identical loss:
  ghost   native TileFeatureEncoder, trained (control; reproduces v2_hitnet)
  ade20k  frozen YOLO26s-sem (ADE20K) backbone via the same (f2,f4,f8,f16) contract

Everything runs at native SceneFlow Driving resolution: training on co-located
384x640 native crops, validation on full 540x960 frames with replicate padding
only. No image or disparity is ever resized.

Logging goes to both stdout and ``run.log`` inside the run directory, so the run
is pollable from a single durable file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.nn import functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from hitnet_head.model import StereoLite, StereoLiteConfig  # noqa: E402


def _sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a02 = _sibling("a02_run", ROOT / "experiments/lightstereo_s_a02/run.py")
build_manifest, read_pfm, metrics = a02.build_manifest, a02.read_pfm, a02.metrics

ARMS = {
    "ghost": dict(backbone="ghost", freeze_encoder=False),
    "ade20k": dict(backbone=str(ROOT / "models/segmentation/yolo26s-sem-ade20k.pt"), freeze_encoder=True),
}


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"a03.{output.name}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(output / "run.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers[:] = [file_handler, stream_handler]
    return logger


def split(records: list[dict[str, str]], seed: int) -> tuple[list, list]:
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record["sequence"], []).append(record)
    randomizer = random.Random(seed)
    train, validation = [], []
    for sequence in sorted(groups):
        group = sorted(groups[sequence], key=lambda item: item["left"])
        randomizer.shuffle(group)
        count = round(len(group) * 0.2)
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation


def load_record(record: dict[str, str]):
    left = cv2.cvtColor(cv2.imread(record["left"]), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(record["right"]), cv2.COLOR_BGR2RGB)
    return left, right, read_pfm(record["disparity"])


def preload(records: list[dict[str, str]]):
    return [load_record(record) for record in records]


def pad_to_16(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    top, right = (-height) % 16, (-width) % 16
    return F.pad(image, (0, right, top, 0), mode="replicate"), top, width


def evaluate(model, cache, device) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for left, right, ground_truth in cache:
            left_tensor = torch.from_numpy(left).permute(2, 0, 1)[None].float().to(device)
            right_tensor = torch.from_numpy(right).permute(2, 0, 1)[None].float().to(device)
            left_tensor, top, width = pad_to_16(left_tensor)
            right_tensor, _, _ = pad_to_16(right_tensor)
            prediction = model(left_tensor, right_tensor)[..., top:, :width].float().cpu()
            rows.append(metrics(prediction, torch.from_numpy(ground_truth)[None, None]))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-height", type=int, default=384)
    parser.add_argument("--crop-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data", type=Path, default=Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = HERE / "runs" / f"A03_{args.arm}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    logger = setup_logging(output)
    logger.info("run=%s arm=%s steps=%d", output.name, args.arm, args.steps)

    records = build_manifest(args.data, 200)
    train_records, validation_records = split(records, args.seed)
    (output / "manifest_train.json").write_text(json.dumps(train_records, indent=2) + "\n")
    (output / "manifest_validation.json").write_text(json.dumps(validation_records, indent=2) + "\n")
    train_cache = preload(train_records)
    validation_cache = preload(validation_records)
    logger.info("pairs: %d train / %d held-out", len(train_cache), len(validation_cache))

    model = StereoLite(StereoLiteConfig(**ARMS[args.arm])).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    config = {
        "experiment": "A03 encoder swap on StereoLite v2_hitnet head",
        "arm": args.arm,
        "head": "StereoLite_v2_hitnet (ported from the author's thesis repo)",
        "encoder": ARMS[args.arm]["backbone"],
        "encoder_frozen": ARMS[args.arm]["freeze_encoder"],
        "encoder_out_channels": list(model.fnet.out_channels),
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in trainable)},
        "data": "SceneFlow Driving 200 fixed pairs; deterministic stratified 160/40",
        "training": {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                     "optimizer": "AdamW", "precision": "AMP float16",
                     "crop": [args.crop_height, args.crop_width],
                     "loss": "smooth_l1(full), mask finite & 0<d<192"},
        "evaluation": "native 540x960, replicate pad to /16, no resize",
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("run.py", "hitnet_head/model.py", "hitnet_head/encoders.py")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("params: total %.3f M, trainable %.3f M, encoder %s",
                config["parameters"]["total"] / 1e6, config["parameters"]["trainable"] / 1e6,
                tuple(model.fnet.out_channels))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    order: list[int] = []
    randomizer = random.Random(args.seed)
    validation_rows = [{"step": 0.0, "epe": evaluate(model, validation_cache, device)["epe"]}]
    logger.info("initial held-out EPE: %.4f", validation_rows[0]["epe"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with (output / "train.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "epe", "elapsed_seconds"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            if not order:
                order = list(range(len(train_cache)))
                randomizer.shuffle(order)
            left, right, ground_truth = train_cache[order.pop()]
            top = randomizer.randint(0, left.shape[0] - args.crop_height)
            left_off = randomizer.randint(0, left.shape[1] - args.crop_width)
            sl = (slice(top, top + args.crop_height), slice(left_off, left_off + args.crop_width))
            left_tensor = torch.from_numpy(left[sl]).permute(2, 0, 1)[None].float().to(device)
            right_tensor = torch.from_numpy(right[sl]).permute(2, 0, 1)[None].float().to(device)
            target = torch.from_numpy(ground_truth[sl])[None, None].to(device)
            valid = torch.isfinite(target) & (target > 0) & (target < 192)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(left_tensor, right_tensor)
                loss = F.smooth_l1_loss(prediction[valid], target[valid])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            row = {"step": float(step), "loss": loss.item(),
                   "epe": (prediction[valid] - target[valid]).abs().mean().item(),
                   "elapsed_seconds": time.perf_counter() - started}
            writer.writerow(row)
            if step % args.eval_every == 0 or step == args.steps:
                values = evaluate(model, validation_cache, device)
                validation_rows.append({"step": float(step), "epe": values["epe"]})
                logger.info("step=%d loss=%.4f crop_epe=%.4f val_epe=%.4f", step, row["loss"], row["epe"], values["epe"])
            if step % args.checkpoint_every == 0 or step == args.steps:
                payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "config": config, "step": step}
                torch.save(payload, output / "checkpoints" / f"step_{step:06d}.pth")
                torch.save(payload, output / "checkpoints" / "latest.pth")

    final = evaluate(model, validation_cache, device)
    elapsed = time.perf_counter() - started
    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "epe"])
        writer.writeheader()
        writer.writerows(validation_rows)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([r["step"] for r in validation_rows], [r["epe"] for r in validation_rows], marker="o")
    axis.set(xlabel="step", ylabel="held-out EPE (px)", title=f"A03 {args.arm}")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "training_curve.png", dpi=160)
    plt.close(figure)
    results = {"arm": args.arm, "config": config, "initial": validation_rows[0], "final": final,
               "best": min(validation_rows[1:] or validation_rows, key=lambda r: r["epe"]),
               "training_seconds": elapsed, "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    logger.info("COMPLETE %s final EPE=%.4f best=%.4f (step %.0f)", output, final["epe"],
                results["best"]["epe"], results["best"]["step"])
    print(json.dumps(results["final"], indent=2))


if __name__ == "__main__":
    main()
