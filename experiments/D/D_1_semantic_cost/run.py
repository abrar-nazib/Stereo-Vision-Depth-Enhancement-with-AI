"""D-series SGNet-inspired, frozen-predictor VKITTI2 ablation runner.

Use ``uv run --no-sync python -m experiments.D.D_1_semantic_cost.queue`` to
launch all arms sequentially. Stereo images and disparity are never resized.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.run import (
    ADE, DATA, ROOT, SEMANTIC, STEREO, crop_valid_mask, pad32, read_pair,
    sha256, split_rows, to_tensors,
)
from experiments.C.C_1_large_control.run import (
    append_jsonl, atomic_json, semantic_metrics_from_confusion,
)
from experiments.D.D_1_semantic_cost.model import DModel
from experiments.vkitti2.data import paired_crop
from experiments.vkitti2.run import DisparityMeter


ARMS = {
    "D_1_semantic_cost": (False, True),
    "D_2_semantic_cost_residual": (True, True),
    "D_3_no_semantic_cost": (False, False),
    "D_4_no_semantic_residual": (True, False),
}


def construct(arm: str) -> DModel:
    refine, semantics = ARMS[arm]
    return DModel(STEREO, SEMANTIC, ADE, refinement=refine,
                  use_semantics=semantics)


@torch.inference_mode()
def evaluate(model: DModel, rows: list[dict], data: Path) -> dict:
    disparity_meter = DisparityMeter()
    confusion = np.zeros((14, 14), dtype=np.int64)
    pair_results = []
    model.eval()
    for row in rows:
        left, right, target, labels = to_tensors(read_pair(data, row))
        left_padded, top = pad32(left)
        right_padded, _ = pad32(right)
        predicted, logits = model(left_padded, right_padded)
        predicted = predicted[..., top:top + target.shape[-2], :target.shape[-1]]
        disparity_meter.add(predicted, target)
        pair_meter = DisparityMeter()
        pair_meter.add(predicted, target)
        pair_results.append({"scene": row["scene"], "frame": row["frame"],
                             "variation": row["variation"], **pair_meter.result()})
        classes = F.interpolate(logits, size=left_padded.shape[-2:], mode="bilinear",
                                align_corners=False)
        classes = classes[..., top:top + target.shape[-2], :target.shape[-1]].argmax(1)
        prediction = classes[0].cpu().numpy()
        truth = labels[0, 0].cpu().numpy()
        valid = truth < 14
        confusion += np.bincount(14 * truth[valid].astype(np.int64) +
                                 prediction[valid].astype(np.int64),
                                 minlength=196).reshape(14, 14)
    return {**disparity_meter.result(), **semantic_metrics_from_confusion(confusion),
            "pairs": len(rows), "pair_results": pair_results,
            "selection_metric": "bad_3"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", default="d_v1_seed42_20261002")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Smoke only; truncates each split and must not be cited")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D-series requires local CUDA")
    if min(args.steps, args.eval_every, args.patience) < 1:
        raise ValueError("steps, eval-every and patience must be positive")
    if not args.data.joinpath("manifest.json").is_file() or not STEREO.is_file():
        raise FileNotFoundError("VKITTI ablation1000 or A09 stereo checkpoint is unavailable")
    run_dir = ROOT / "experiments" / "D" / args.arm / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    status_path = run_dir / "status.json"
    started = time.time()
    atomic_json(status_path, {"state": "initializing", "arm": args.arm,
                              "started_unix": started})
    try:
        rows = json.loads((args.data / "manifest.json").read_text())
        train_rows, val_rows, test_rows = split_rows(rows, 42)
        if args.eval_limit is not None:
            limit = max(1, args.eval_limit)
            train_rows, val_rows, test_rows = (train_rows[:max(2, limit)],
                                              val_rows[:limit], test_rows[:limit])
        torch.manual_seed(42)
        rng = random.Random(42)
        model = construct(args.arm).cuda().train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if any(p.requires_grad for p in model.base.parameters()):
            raise AssertionError("a frozen predictor became trainable")
        metadata = {
            "arm": args.arm, "run_id": args.run_id, "seed": 42,
            "max_steps": args.steps, "eval_every": args.eval_every,
            "patience": args.patience, "crop_hw": [256, 512],
            "train_pairs": len(train_rows), "val_pairs": len(val_rows),
            "test_pairs": len(test_rows), "dataset_manifest_sha256": sha256(args.data / "manifest.json"),
            "stereo_sha256": sha256(STEREO), "semantic_sha256": sha256(SEMANTIC),
            "trainable_params": sum(p.numel() for p in trainable),
            "gpu": torch.cuda.get_device_name(),
            "split": "B/C-series Scene20 frame-grouped 800/100/100 seed42",
            "semantic_metric": "GT-present 14-class mIoU", "smoke_limit": args.eval_limit,
            "use_semantics": model.use_semantics, "refinement": model.refiner is not None,
            "architecture": "SGNet-inspired 1/16 semantic correlation gate before A09 aggregation"
        }
        atomic_json(run_dir / "manifest.json", metadata)
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                         total_steps=args.steps, pct_start=0.1)
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        cursor, best_bad3, stale = 0, float("inf"), 0
        atomic_json(status_path, {"state": "training", "arm": args.arm,
                                  "step": 0, "max_steps": args.steps})
        for step in range(1, args.steps + 1):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            sample = read_pair(args.data, train_rows[order[cursor]])
            cursor += 1
            height, width = sample[0].shape[:2]
            top = rng.randrange(height - 256 + 1)
            left_x = rng.randrange(width - 512 + 1)
            left, right, target, _ = to_tensors(
                paired_crop(*sample, top, left_x, 256, 512))
            valid = crop_valid_mask(target)
            if not valid.any():
                raise ValueError("training crop has no valid disparity")
            left_padded, _ = pad32(left)
            right_padded, _ = pad32(right)
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(left_padded, right_padded)
            error = (predicted[valid] - target[valid]).abs()
            # Same outlier-aware objective for every arm; evaluate full-resolution bad-3.
            loss = F.smooth_l1_loss(predicted[valid], target[valid]) + \
                0.2 * F.relu(error - 3.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0:
                record = {"phase": "train", "step": step,
                          "loss": float(loss.detach()),
                          "lr": scheduler.get_last_lr()[0],
                          "elapsed_s": time.time() - started}
                append_jsonl(run_dir / "history.jsonl", record)
                atomic_json(status_path, {"state": "training", "arm": args.arm,
                                          "step": step, "max_steps": args.steps,
                                          "best_val_bad_3": best_bad3 if np.isfinite(best_bad3) else None,
                                          "latest_loss": record["loss"],
                                          "elapsed_s": record["elapsed_s"]})
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val_rows, args.data)
                record = {"phase": "val", "step": step, "epe": metrics["epe"],
                          "bad_1": metrics["bad_1"], "bad_3": metrics["bad_3"],
                          "d1": metrics["d1"], "miou": metrics["miou"],
                          "elapsed_s": time.time() - started}
                append_jsonl(run_dir / "history.jsonl", record)
                atomic_json(run_dir / f"validation_step_{step:05d}.json", metrics)
                payload = {"trainable": model.trainable_state(), "step": step,
                           "metrics": metrics, "manifest": metadata}
                torch.save(payload, run_dir / "checkpoints" / f"step_{step:05d}.pth")
                if metrics["bad_3"] < best_bad3 - 0.01:
                    best_bad3, stale = metrics["bad_3"], 0
                    torch.save(payload, run_dir / "checkpoints" / "best.pth")
                    atomic_json(run_dir / "best_validation.json", metrics)
                else:
                    stale += 1
                atomic_json(status_path, {"state": "training", "arm": args.arm,
                                          "step": step, "max_steps": args.steps,
                                          "best_val_bad_3": best_bad3,
                                          "latest_val_epe": metrics["epe"],
                                          "latest_val_bad_3": metrics["bad_3"],
                                          "stale_evals": stale,
                                          "elapsed_s": record["elapsed_s"]})
                print(json.dumps(record), flush=True)
                model.train()
                if stale >= args.patience and step >= 3000 and args.eval_limit is None:
                    print(json.dumps({"phase": "early_stop", "step": step}), flush=True)
                    break
        checkpoint = torch.load(run_dir / "checkpoints" / "best.pth",
                                map_location="cpu", weights_only=False)
        model.load_trainable_state(checkpoint["trainable"])
        test_metrics = evaluate(model, test_rows, args.data)
        atomic_json(run_dir / "test.json", test_metrics)
        atomic_json(status_path, {"state": "complete", "arm": args.arm,
                                  "step": step, "selected_step": checkpoint["step"],
                                  "best_val_bad_3": best_bad3,
                                  "test_epe": test_metrics["epe"],
                                  "test_bad_3": test_metrics["bad_3"],
                                  "elapsed_s": time.time() - started})
        print(json.dumps({"phase": "complete", "arm": args.arm,
                          "test_epe": test_metrics["epe"],
                          "test_bad_3": test_metrics["bad_3"]}), flush=True)
    except BaseException as error:
        atomic_json(status_path, {"state": "failed", "arm": args.arm,
                                  "error": repr(error),
                                  "elapsed_s": time.time() - started})
        raise


if __name__ == "__main__":
    main()
