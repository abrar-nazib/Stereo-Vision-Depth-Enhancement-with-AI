"""C-series frozen-predictor ablations on the versioned VKITTI 800/100/100 split.

Run one arm with ``uv run --no-sync python -m experiments.C.C_1_large_control.run``.
The queue driver launches all arms sequentially on one RTX 3050.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.B.B_0_fused_baseline.run import (
    ADE, DATA, SEMANTIC, STEREO, ROOT, crop_valid_mask, pad32, read_pair,
    sha256, split_rows, to_tensors,
)
from experiments.C.C_1_large_control.head import LargeControlHead
from experiments.C.C_2_feature_fusion.head import FeatureFusionHead
from experiments.C.C_3_semantic_match.head import SemanticMatchHead
from experiments.vkitti2.data import paired_crop
from experiments.vkitti2.run import DisparityMeter


ARMS = {
    "C_1_large_control": LargeControlHead,
    "C_2_feature_fusion": FeatureFusionHead,
    "C_3_semantic_match": SemanticMatchHead,
}
CONTROL_ARMS = {
    "C_4_no_semantics": FeatureFusionHead,
    "C_5_misaligned_semantics": FeatureFusionHead,
}
ALL_ARMS = {**ARMS, **CONTROL_ARMS}


def select_head(name: str) -> torch.nn.Module:
    return ALL_ARMS[name]()


def guidance_mode_for_arm(arm: str) -> str:
    return {"C_4_no_semantics": "none",
            "C_5_misaligned_semantics": "misaligned"}.get(arm, "aligned")


def apply_guidance(logits: torch.Tensor, features: dict[str, torch.Tensor],
                   mode: str, rng: random.Random):
    """Keep frozen outputs intact; transform only inputs to the trainable head.

    Misalignment draws independent nonzero vertical/horizontal circular shifts
    for each pair. The same shift applies to logits and semantic decoder maps;
    stereo and shared trunk maps remain at their original pixel coordinates.
    """
    transformed = dict(features)
    if mode == "aligned":
        return logits, transformed, (0, 0)
    if mode == "none":
        transformed["semantic_f8"] = torch.zeros_like(features["semantic_f8"])
        return torch.zeros_like(logits), transformed, (0, 0)
    if mode != "misaligned":
        raise ValueError(f"unknown guidance mode: {mode}")
    height, width = logits.shape[-2:]
    dy = rng.choice((-1, 1)) * rng.randint(max(1, height // 4), max(1, height // 2))
    dx = rng.choice((-1, 1)) * rng.randint(max(1, width // 4), max(1, width // 2))
    offsets = (dy, dx)
    transformed["semantic_f8"] = features["semantic_f8"].roll(offsets, dims=(-2, -1))
    return logits.roll(offsets, dims=(-2, -1)), transformed, offsets


def semantic_metrics(pairs, classes: int = 14) -> dict:
    """GT-present class mIoU, matching Ultralytics' aggregation convention."""
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for prediction, target in pairs:
        valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
        confusion += np.bincount(
            classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
            minlength=classes * classes).reshape(classes, classes)
    intersection = confusion.diagonal()
    support = confusion.sum(1)
    union = support + confusion.sum(0) - intersection
    present = np.flatnonzero(support > 0)
    ious = np.divide(intersection, union, out=np.zeros(classes, dtype=float), where=union > 0)
    return {"miou": float(ious[present].mean()) if len(present) else 0.0,
            "pixel_accuracy": float(intersection.sum() / max(confusion.sum(), 1)),
            "present_classes": present.tolist(),
            "class_iou": {str(i): float(ious[i]) for i in present},
            "class_pixels": {str(i): int(support[i]) for i in present},
            "semantic_pixels": int(confusion.sum())}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
        stream.flush()


def _predict(model, left: torch.Tensor, right: torch.Tensor):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        disparity, logits, features = model(left, right, return_features=True)
    return disparity.float(), logits.float(), {key: value.float() for key, value in features.items()}


@torch.inference_mode()
def evaluate(model, head, rows: list[dict], data: Path, control: str = "normal",
             guidance_mode: str = "aligned") -> dict:
    disparity_meter = DisparityMeter()
    confusion = np.zeros((14, 14), dtype=np.int64)
    guidance_rng = random.Random(42)
    for row in rows:
        left, right, gt, labels = to_tensors(read_pair(data, row))
        lp, top = pad32(left)
        rp, _ = pad32(right)
        base, logits, features = _predict(model, lp, rp)
        if guidance_mode != "aligned":
            guidance, features, _ = apply_guidance(logits, features, guidance_mode,
                                                    guidance_rng)
        elif control == "uniform":
            guidance = torch.zeros_like(logits)
            features["semantic_f8"] = torch.zeros_like(features["semantic_f8"])
        elif control == "shifted":
            shift = max(1, logits.shape[-1] // 3)
            guidance = logits.roll(shift, dims=-1)
            features["semantic_f8"] = features["semantic_f8"].roll(shift, dims=-1)
        else:
            guidance = logits
        prediction = head(base, guidance, **features)
        prediction = prediction[..., top:top + gt.shape[-2], :gt.shape[-1]]
        disparity_meter.add(prediction, gt)
        full_logits = F.interpolate(logits, size=lp.shape[-2:], mode="bilinear",
                                    align_corners=False)
        pred_classes = full_logits[..., top:top + gt.shape[-2], :gt.shape[-1]].argmax(1)
        pred_np = pred_classes[0].cpu().numpy()
        gt_np = labels[0, 0].cpu().numpy()
        valid = gt_np < 14
        confusion += np.bincount(14 * gt_np[valid].astype(np.int64) +
                                 pred_np[valid].astype(np.int64), minlength=196).reshape(14, 14)
    sem = semantic_metrics_from_confusion(confusion)
    return {**disparity_meter.result(), **sem, "pairs": len(rows),
            "control": control, "guidance_mode": guidance_mode}


def semantic_metrics_from_confusion(confusion: np.ndarray) -> dict:
    intersection = confusion.diagonal()
    support = confusion.sum(1)
    union = support + confusion.sum(0) - intersection
    present = np.flatnonzero(support > 0)
    ious = np.divide(intersection, union, out=np.zeros(14, dtype=float), where=union > 0)
    return {"miou": float(ious[present].mean()) if len(present) else 0.0,
            "pixel_accuracy": float(intersection.sum() / max(confusion.sum(), 1)),
            "present_classes": present.tolist(),
            "class_iou": {str(i): float(ious[i]) for i in present},
            "class_pixels": {str(i): int(support[i]) for i in present},
            "semantic_pixels": int(confusion.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ALL_ARMS, required=True)
    parser.add_argument("--run-id", default="c_v1_seed42_20261001")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Smoke only; never use for paper metrics")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C-series training requires CUDA")
    if args.steps < 1 or args.eval_every < 1 or args.patience < 1:
        raise ValueError("steps, eval-every and patience must be positive")
    run_dir = ROOT / "experiments" / "C" / args.arm / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    started = time.time()
    status_path = run_dir / "status.json"
    atomic_json(status_path, {"state": "initializing", "arm": args.arm, "started_unix": started})
    try:
        rows = json.loads((args.data / "manifest.json").read_text())
        train_rows, val_rows, test_rows = split_rows(rows, 42)
        if args.eval_limit is not None:
            train_rows = train_rows[:max(2, args.eval_limit)]
            val_rows, test_rows = val_rows[:args.eval_limit], test_rows[:args.eval_limit]
        torch.manual_seed(42)
        rng = random.Random(42)
        model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
        head = select_head(args.arm).cuda().train()
        guidance_mode = guidance_mode_for_arm(args.arm)
        guidance_rng = random.Random(314159)
        assert not any(p.requires_grad for p in model.parameters())
        metadata = {"arm": args.arm, "run_id": args.run_id, "seed": 42,
                    "max_steps": args.steps, "eval_every": args.eval_every,
                    "patience": args.patience, "crop_hw": [256, 512],
                    "train_pairs": len(train_rows), "val_pairs": len(val_rows),
                    "test_pairs": len(test_rows), "data_manifest_sha256": sha256(args.data / "manifest.json"),
                    "stereo_sha256": sha256(STEREO), "semantic_sha256": sha256(SEMANTIC),
                    "trainable_params": sum(p.numel() for p in head.parameters()),
                    "gpu": torch.cuda.get_device_name(), "semantic_metric": "GT-present mIoU",
                    "split": "B-series 800/100/100 Scene20 frame-grouped seed42",
                    "smoke_limit": args.eval_limit,
                    "guidance_mode": guidance_mode,
                    "reference_run": "C_2_feature_fusion/runs/c_v1_seed42_20261001"
                    if args.arm in CONTROL_ARMS else None,
                    "guidance_shift_seed": 314159 if guidance_mode == "misaligned" else None}
        atomic_json(run_dir / "manifest.json", metadata)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                         total_steps=args.steps, pct_start=0.1)
        indices = list(range(len(train_rows)))
        rng.shuffle(indices)
        cursor, best_epe, stale = 0, float("inf"), 0
        atomic_json(status_path, {"state": "training", "arm": args.arm,
                                  "step": 0, "max_steps": args.steps, "started_unix": started})
        for step in range(1, args.steps + 1):
            if cursor >= len(indices):
                rng.shuffle(indices)
                cursor = 0
            row = train_rows[indices[cursor]]
            cursor += 1
            sample = read_pair(args.data, row)
            h, w = sample[0].shape[:2]
            top, x = rng.randrange(h - 256 + 1), rng.randrange(w - 512 + 1)
            left, right, gt, _ = to_tensors(paired_crop(*sample, top, x, 256, 512))
            valid = crop_valid_mask(gt)
            lp, _ = pad32(left)
            rp, _ = pad32(right)
            base, logits, features = _predict(model, lp, rp)
            logits, features, _ = apply_guidance(logits, features, guidance_mode,
                                                 guidance_rng)
            optimizer.zero_grad(set_to_none=True)
            predicted = head(base, logits, **features)
            loss = F.smooth_l1_loss(predicted[valid], gt[valid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 50 == 0:
                record = {"phase": "train", "step": step, "loss": float(loss.detach()),
                          "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started}
                append_jsonl(run_dir / "history.jsonl", record)
                atomic_json(status_path, {"state": "training", "arm": args.arm,
                                          "step": step, "max_steps": args.steps,
                                          "best_val_epe": best_epe if np.isfinite(best_epe) else None,
                                          "latest_loss": record["loss"],
                                          "elapsed_s": record["elapsed_s"]})
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                head.eval()
                metrics = evaluate(model, head, val_rows, args.data,
                                   guidance_mode=guidance_mode)
                record = {"phase": "val", "step": step, "epe": metrics["epe"],
                          "miou": metrics["miou"], "elapsed_s": time.time() - started}
                append_jsonl(run_dir / "history.jsonl", record)
                atomic_json(run_dir / f"validation_step_{step:05d}.json", metrics)
                torch.save({"head": head.state_dict(), "step": step, "metrics": metrics,
                            "manifest": metadata}, run_dir / "checkpoints" / f"step_{step:05d}.pth")
                if metrics["epe"] < best_epe - 0.002:
                    best_epe, stale = metrics["epe"], 0
                    torch.save({"head": head.state_dict(), "step": step, "metrics": metrics,
                                "manifest": metadata}, run_dir / "checkpoints" / "best.pth")
                    atomic_json(run_dir / "best_validation.json", metrics)
                else:
                    stale += 1
                atomic_json(status_path, {"state": "training", "arm": args.arm,
                                          "step": step, "max_steps": args.steps,
                                          "best_val_epe": best_epe, "latest_val_epe": metrics["epe"],
                                          "latest_val_miou": metrics["miou"], "stale_evals": stale,
                                          "elapsed_s": record["elapsed_s"]})
                print(json.dumps(record), flush=True)
                head.train()
                if stale >= args.patience and step >= 3000 and args.eval_limit is None:
                    print(json.dumps({"phase": "early_stop", "step": step}), flush=True)
                    break
        checkpoint = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cuda",
                                weights_only=False)
        head.load_state_dict(checkpoint["head"])
        head.eval()
        test_metrics = evaluate(model, head, test_rows, args.data,
                                guidance_mode=guidance_mode)
        atomic_json(run_dir / "test.json", test_metrics)
        if guidance_mode == "aligned":
            controls = {control: evaluate(model, head, val_rows, args.data, control=control)
                        for control in ("normal", "uniform", "shifted")}
        else:
            controls = {"trained_mode": evaluate(model, head, val_rows, args.data,
                                                   guidance_mode=guidance_mode),
                        "aligned_probe": evaluate(model, head, val_rows, args.data)}
        atomic_json(run_dir / "controls.json", controls)
        atomic_json(status_path, {"state": "complete", "arm": args.arm,
                                  "step": step, "selected_step": checkpoint["step"],
                                  "best_val_epe": best_epe, "test_epe": test_metrics["epe"],
                                  "test_miou": test_metrics["miou"],
                                  "elapsed_s": time.time() - started})
        print(json.dumps({"phase": "complete", "arm": args.arm,
                          "selected_step": checkpoint["step"],
                          "test_epe": test_metrics["epe"]}), flush=True)
    except BaseException as error:
        atomic_json(status_path, {"state": "failed", "arm": args.arm,
                                  "error": repr(error), "elapsed_s": time.time() - started})
        raise


if __name__ == "__main__":
    main()
