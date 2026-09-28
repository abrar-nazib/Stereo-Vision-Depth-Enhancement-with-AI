"""A04b — Selective-RAFT-style iterative head with frozen ADE20K YOLO26 encoder.

Protocol (identical across A04):
  200 fixed SceneFlow Driving pairs, stratified 160 train / 40 held-out (seed 42),
  native 384x640 co-located crops, same window on L/R/D, batch 1, AMP fp16,
  AdamW 1e-4, 15k steps, eval every 100, checkpoint every 500, run.log + stdout.

Lane: frozen ADE20K only (ghost control removed per request).
Head self-sizes from YOLO channels — no adapter at all.
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
A06 = HERE.parent
ROOT = A06.parents[1]
sys.path.insert(0, str(HERE))

from model_c import ROOT as MODEL_ROOT, ShallowYoloRaft, native_crop  # noqa: E402
from losses import modal_loss  # noqa: E402


def _sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a02 = _sibling("a02_run", ROOT / "experiments/lightstereo_s_a02/run.py")
build_manifest, metrics, read_pfm = a02.build_manifest, a02.metrics, a02.read_pfm


def split(records, seed: int):
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(r["sequence"], []).append(r)
    rnd = random.Random(seed)
    train, val = [], []
    for seq in sorted(groups):
        g = sorted(groups[seq], key=lambda x: x["left"])
        rnd.shuffle(g)
        k = round(len(g) * 0.2)
        val.extend(g[:k])
        train.extend(g[k:])
    return train, val


def load_record(rec):
    left = cv2.cvtColor(cv2.imread(rec["left"]), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(rec["right"]), cv2.COLOR_BGR2RGB)
    return left, right, read_pfm(rec["disparity"])


def preload(records):
    return [load_record(r) for r in records]


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda(non_blocking=True)


def pad_top_right(left: torch.Tensor, right: torch.Tensor):
    h, w = left.shape[-2:]
    top, rp = (-h) % 16, (-w) % 16
    return (
        F.pad(left, (0, rp, top, 0), mode="replicate"),
        F.pad(right, (0, rp, top, 0), mode="replicate"),
        top, w,
    )


def evaluate(model, cache, *, semantic=False):
    model.eval()
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for idx, (left, right, gt) in enumerate(cache):
            lt, rt, top, w = pad_top_right(to_tensor(left), to_tensor(right))
            pred, _, _ = model(lt, rt, semantic=semantic and idx == 0)
            pred = pred[..., top:, :w].cpu()
            row = {"index": float(idx), **metrics(pred, torch.from_numpy(gt)[None, None])}
            row["valid_fraction"] = row["valid_pixels"] / gt.size
            rows.append(row)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "index"}
    return mean, rows


def setup_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"a04b.{output.name}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(output / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.handlers[:] = [fh, sh]
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-height", type=int, default=384)
    parser.add_argument("--crop-width", type=int, default=640)
    parser.add_argument("--data", type=Path, default=Path("/media/abrar/AbrarSSD/Datasets/sceneflow_driving"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    output = A06 / "runs" / f"A06C1_shallow_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    logger = setup_logging(output)
    logger.info("run=%s steps=%d", output.name, args.steps)

    manifest_dir = A06 / "manifests"
    train_records = json.loads((manifest_dir / "manifest_train.json").read_text())
    val_records = json.loads((manifest_dir / "manifest_validation.json").read_text())
    (output / "manifest_train.json").write_text(json.dumps(train_records, indent=2) + "\n")
    (output / "manifest_validation.json").write_text(json.dumps(val_records, indent=2) + "\n")
    train_cache = preload(train_records)
    val_cache = preload(val_records)
    logger.info("pairs: %d train / %d held-out", len(train_cache), len(val_cache))

    model = ShallowYoloRaft(arm="C1").cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    config = {
        "experiment": "A06C1 shallow-tap C1 + RAFT head (frozen YOLO, modal loss)",
        "encoder": str((ROOT / "models/segmentation/yolo26s-sem-ade20k.pt")),
        "encoder_frozen": True,
        "arm": "C1", "taps": "RECORD", "encoder_out_channels": list(model.encoder.out_channels),
        "adapter": "none — head self-sizes from shallow taps C1",
        "head": "RAFT-like TileInit + 2/3/3 GRU iters + convex upsample on shallow taps C1",
        "parameters": {"total": total, "trainable": sum(p.numel() for p in trainable)},
        "data": "SceneFlow Driving 200 fixed pairs; stratified 160/40",
        "training": {"steps": args.steps, "batch_size": 1, "lr": 1e-4, "optimizer": "AdamW",
                     "precision": "AMP fp16", "crop": [args.crop_height, args.crop_width],
                     "loss": "modal multi-scale (1.0/0.5/0.3 + grad 0.5 + hinge 0.2/0.2 + smooth 0.02)"},
        "evaluation": "native 540x960, replicate pad to /16, no resize",
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("model_c.py", "tile_propagate.py", "run_C1.py", "losses.py")},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("params: total %.3f M, trainable %.3f M, encoder %s", total/1e6, sum(p.numel() for p in trainable)/1e6, tuple(model.encoder.out_channels))

    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rnd = random.Random(args.seed)
    order: list[int] = []
    val_rows = [{"step": 0.0, "epe": evaluate(model, val_cache)[0]["epe"]}]
    logger.info("initial held-out EPE: %.4f", val_rows[0]["epe"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_rows: list[dict] = []

    with (output / "train.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "epe", "elapsed_seconds"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            if not order:
                order = list(range(len(train_cache)))
                rnd.shuffle(order)
            left, right, gt = train_cache[order.pop()]
            top = rnd.randint(0, left.shape[0] - args.crop_height)
            loff = rnd.randint(0, left.shape[1] - args.crop_width)
            left, right, gt = native_crop(left, right, gt, height=args.crop_height, width=args.crop_width, top=top, left_offset=loff)
            lt, rt = to_tensor(left), to_tensor(right)
            target = torch.from_numpy(gt)[None, None].cuda()
            valid = torch.isfinite(target) & (target > 0) & (target < 192)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred, _, _ = model(lt, rt)
                loss, _diag = modal_loss(pred, target, lt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            row = {"step": float(step), "loss": loss.item(), "epe": (pred[valid]-target[valid]).abs().mean().item(), "elapsed_seconds": time.perf_counter()-started}
            writer.writerow(row)
            train_rows.append(row)
            if step % args.eval_every == 0 or step == args.steps:
                vals, _ = evaluate(model, val_cache)
                val_rows.append({"step": float(step), "epe": vals["epe"]})
                logger.info("step=%d loss=%.4f crop_epe=%.4f val_epe=%.4f", step, row["loss"], row["epe"], vals["epe"])
            if step % args.checkpoint_every == 0 or step == args.steps:
                payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": step}
                torch.save(payload, output / "checkpoints" / f"step_{step:06d}.pth")
                torch.save(payload, output / "checkpoints" / "latest.pth")

    final, final_rows = evaluate(model, val_cache, semantic=True)
    elapsed = time.perf_counter() - started
    with (output / "validation.csv").open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=["step", "epe"]); w.writeheader(); w.writerows(val_rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([r["step"] for r in train_rows], [r["loss"] for r in train_rows]); ax1.set(xlabel="step", ylabel="loss", title="Training loss"); ax1.grid(alpha=0.3)
    ax2.plot([r["step"] for r in train_rows], [r["epe"] for r in train_rows], alpha=0.45, label="train crop EPE")
    ax2.plot([r["step"] for r in val_rows], [r["epe"] for r in val_rows], marker="o", label="held-out native EPE"); ax2.set(xlabel="step", ylabel="EPE (px)", title="No-resize convergence"); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(output / "training_curve.png", dpi=160); plt.close(fig)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": args.steps}, output / "final.pth")
    results = {"final": final, "best": min(val_rows[1:] or val_rows, key=lambda r: r["epe"]), "initial": val_rows[0], "parameters": config["parameters"], "training_seconds": elapsed, "peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20}
    (output / "results.json").write_text(json.dumps(results, indent=2)+"\n")
    logger.info("COMPLETE %s final EPE=%.4f best=%.4f @ %.0f", output, final["epe"], results["best"]["epe"], results["best"]["step"])
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
