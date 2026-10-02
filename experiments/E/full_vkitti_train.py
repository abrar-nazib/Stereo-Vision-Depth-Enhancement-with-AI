"""Full VKITTI2 frozen-predictor depth-head training, with exact resume."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.run import crop_valid_mask, pad32, sha256, to_tensors
from experiments.C.C_1_large_control.run import atomic_json, semantic_metrics_from_confusion
from experiments.D.D_1_semantic_cost.model import DModel
from experiments.E.E_1_wide2_semantics.model import EModel
from experiments.vkitti2.data import depth_to_disparity, paired_crop
from experiments.vkitti2.run import DisparityMeter


ARM_CONFIGS = {"E3": (32, 128, True), "E4": (32, 128, False),
               "D2": (8, 32, True)}


def read_full_pair(root: Path, row: dict) -> tuple[np.ndarray, ...]:
    stem, split = row["stem"], row["split"]
    left = cv2.imread(str(root / "dataset/images" / split / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(root / row["files"]["rgb_right"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / row["files"]["depth"]), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(root / "dataset/masks" / split / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if any(value is None for value in (left, right, depth, labels)):
        raise FileNotFoundError(f"incomplete stereo pair: {stem}")
    if (left.shape != right.shape or left.shape[:2] != depth.shape or
            depth.dtype != np.uint16 or labels.shape != depth.shape or labels.dtype != np.uint8):
        raise ValueError(f"incompatible RGB, depth or indexed-mask shape: {stem}")
    if np.any((labels > 13) & (labels != 255)):
        raise ValueError(f"out-of-range semantic class: {stem}")
    disparity, _ = depth_to_disparity(depth, row["fx"], row["baseline_m"], 192)
    return (cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
            disparity, labels)


def construct(arm: str, stereo: Path, semantic: Path, encoder: Path):
    gate_hidden, residual_hidden, use_semantics = ARM_CONFIGS[arm]
    if arm == "D2":
        return DModel(stereo, semantic, encoder, refinement=True, use_semantics=True)
    return EModel(stereo, semantic, encoder, gate_hidden=gate_hidden,
                  residual_hidden=residual_hidden, use_semantics=use_semantics)


@torch.inference_mode()
def evaluate(model, rows: list[dict], data: Path) -> dict:
    disparity_meter = DisparityMeter()
    confusion = np.zeros((14, 14), dtype=np.int64)
    pair_results = []
    model.eval()
    for index, row in enumerate(rows, start=1):
        left, right, target, labels = to_tensors(read_full_pair(data, row))
        lp, top = pad32(left)
        rp, _ = pad32(right)
        prediction, logits = model(lp, rp)
        prediction = prediction[..., top:top + target.shape[-2], :target.shape[-1]]
        disparity_meter.add(prediction, target)
        pair_meter = DisparityMeter()
        pair_meter.add(prediction, target)
        pair_results.append({"scene": row["scene"], "frame": row["frame"],
                             "variation": row["variation"], **pair_meter.result()})
        classes = F.interpolate(logits, size=lp.shape[-2:], mode="bilinear",
                                align_corners=False)
        classes = classes[..., top:top + target.shape[-2], :target.shape[-1]].argmax(1)
        predicted = classes[0].cpu().numpy()
        truth = labels[0, 0].cpu().numpy()
        valid = truth < 14
        confusion += np.bincount(14 * truth[valid].astype(np.int64) +
                                 predicted[valid].astype(np.int64),
                                 minlength=196).reshape(14, 14)
        if index % 200 == 0 or index == len(rows):
            print(json.dumps({"phase": "evaluation_progress", "pairs": index,
                              "total_pairs": len(rows)}), flush=True)
    return {**disparity_meter.result(), **semantic_metrics_from_confusion(confusion),
            "pairs": len(rows), "pair_results": pair_results,
            "selection_metric": "validation bad_3"}


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_CONFIGS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stereo", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Probe only; nonpublishable metrics")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("full VKITTI trainer requires CUDA")
    if min(args.steps, args.batch, args.eval_every, args.patience) < 1:
        raise ValueError("steps, batch, eval-every and patience must be positive")
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    rows = json.loads(args.manifest.read_text())["rows"]
    if args.eval_limit is not None:
        rows = {name: values[:args.eval_limit] for name, values in rows.items()}
    if any(not values for values in rows.values()):
        raise ValueError("empty train, val or test partition")
    torch.manual_seed(42)
    rng = random.Random(42)
    model = construct(args.arm, args.stereo, args.semantic, args.encoder).cuda().train()
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise AssertionError("frozen predictor has trainable parameters")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    metadata = {"arm": args.arm, "seed": 42, "gpu": torch.cuda.get_device_name(),
                "steps": args.steps, "batch": args.batch, "eval_every": args.eval_every,
                "patience": args.patience, "crop_hw": [256, 512],
                "train_pairs": len(rows["train"]), "val_pairs": len(rows["val"]),
                "test_pairs": len(rows["test"]), "split": "semantic teacher grouped seed42",
                "stereo_sha256": sha256(args.stereo), "semantic_sha256": sha256(args.semantic),
                "encoder_sha256": sha256(args.encoder), "dataset_manifest_sha256": sha256(args.manifest),
                "trainable_params": sum(p.numel() for p in trainable),
                "stereo_resize": False, "validation_native_resolution": True,
                "use_semantics": ARM_CONFIGS[args.arm][2],
                "selection_metric": "validation bad_3", "eval_limit": args.eval_limit}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != metadata:
            raise ValueError("resume manifest differs from frozen model/data/training settings")
    else:
        atomic_json(manifest_path, metadata)
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                    total_steps=args.steps, pct_start=0.1)
    order = list(range(len(rows["train"])))
    rng.shuffle(order)
    cursor, best_bad3, stale, first_step = 0, float("inf"), 0, 1
    latest = args.output / "checkpoints/latest.pth"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["manifest"] != metadata:
            raise ValueError("latest checkpoint belongs to another configuration")
        model.load_trainable_state(saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        order, cursor, best_bad3, stale = (saved["order"], saved["cursor"],
                                           saved["best_bad3"], saved["stale"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        first_step = saved["step"] + 1
        print(json.dumps({"phase": "resume", "step": saved["step"]}), flush=True)
    status = args.output / "status.json"
    atomic_json(status, {"state": "training", "step": first_step - 1, "arm": args.arm})
    for step in range(first_step, args.steps + 1):
        samples = []
        for _ in range(args.batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            sample = read_full_pair(args.data, rows["train"][order[cursor]])
            cursor += 1
            height, width = sample[0].shape[:2]
            top = rng.randrange(height - 256 + 1)
            left_x = rng.randrange(width - 512 + 1)
            samples.append(to_tensors(paired_crop(*sample, top, left_x, 256, 512)))
        left, right, target, _ = (torch.cat(items, 0) for items in zip(*samples))
        valid = crop_valid_mask(target)
        if not valid.any():
            raise ValueError("training batch has no valid disparity")
        lp, _ = pad32(left)
        rp, _ = pad32(right)
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = model(lp, rp)
        error = (predicted[valid] - target[valid]).abs()
        loss = F.smooth_l1_loss(predicted[valid], target[valid]) + 0.2 * F.relu(error - 3.0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        if step == first_step or step % 50 == 0:
            record = {"phase": "train", "step": step, "loss": float(loss.detach()),
                      "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started}
            _append(args.output / "history.jsonl", record)
            atomic_json(status, {"state": "training", "arm": args.arm, **record,
                                 "best_val_bad_3": best_bad3 if np.isfinite(best_bad3) else None})
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            atomic_json(status, {"state": "validating", "arm": args.arm,
                                 "step": step, "total_pairs": len(rows["val"])})
            print(json.dumps({"phase": "validation_start", "step": step,
                              "pairs": len(rows["val"])}), flush=True)
            metrics = evaluate(model, rows["val"], args.data)
            record = {"phase": "val", "step": step, "epe": metrics["epe"],
                      "bad_3": metrics["bad_3"], "d1": metrics["d1"],
                      "elapsed_s": time.time() - started}
            _append(args.output / "history.jsonl", record)
            atomic_json(args.output / f"validation_step_{step:06d}.json", metrics)
            improved = metrics["bad_3"] < best_bad3 - 0.01
            if improved:
                best_bad3, stale = metrics["bad_3"], 0
            else:
                stale += 1
            payload = {"manifest": metadata, "trainable": model.trainable_state(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "step": step, "order": order, "cursor": cursor,
                       "best_bad3": best_bad3, "stale": stale,
                       "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state()}
            _save_checkpoint(latest, payload)
            if improved:
                _save_checkpoint(args.output / "checkpoints/best.pth", payload)
                atomic_json(args.output / "best_validation.json", metrics)
            atomic_json(status, {"state": "training", "arm": args.arm, **record,
                                 "best_val_bad_3": best_bad3, "stale_evals": stale,
                                 "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30})
            print(json.dumps(record), flush=True)
            model.train()
            if stale >= args.patience and args.eval_limit is None:
                print(json.dumps({"phase": "early_stop", "step": step}), flush=True)
                break
    checkpoint = torch.load(args.output / "checkpoints/best.pth", map_location="cpu",
                            weights_only=False)
    model.load_trainable_state(checkpoint["trainable"])
    test = evaluate(model, rows["test"], args.data)
    atomic_json(args.output / "test.json", test)
    atomic_json(status, {"state": "complete", "arm": args.arm,
                         "step": step, "selected_step": checkpoint["step"],
                         "test_epe": test["epe"], "test_bad_3": test["bad_3"],
                         "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                         "elapsed_s": time.time() - started})
    print(json.dumps({"phase": "complete", "arm": args.arm,
                      "test_epe": test["epe"], "test_bad_3": test["bad_3"]}), flush=True)


if __name__ == "__main__":
    main()
