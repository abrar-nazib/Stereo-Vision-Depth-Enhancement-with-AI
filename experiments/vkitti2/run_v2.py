"""Versioned 1,000-pair local comparison of literature-inspired residual heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO

from experiments.vkitti2.data import paired_crop
from experiments.vkitti2.model import JointResidual
from experiments.vkitti2.model_v2 import GatedResidual
from experiments.vkitti2.run import (
    DEFAULT_CKPT, REPO, YOLO_WEIGHTS, DisparityMeter, FusionStereoLite,
    SemanticMeter, append_history, frozen_outputs, pad16, read_pair,
    save_plot, tensors, unpad_semantic_logits,
)


DEFAULT_DATA = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation1000")


def disparity_edge_mask(gt: torch.Tensor, threshold: float = 4.0,
                        radius: int = 5) -> torch.Tensor:
    """GT disparity jumps > threshold, dilated within radius valid pixels."""
    valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
    edge_x = (gt[..., 1:] - gt[..., :-1]).abs() > threshold
    edge_x &= valid[..., 1:] & valid[..., :-1]
    edge_y = (gt[..., 1:, :] - gt[..., :-1, :]).abs() > threshold
    edge_y &= valid[..., 1:, :] & valid[..., :-1, :]
    edge = F.pad(edge_x, (1, 0, 0, 0)) | F.pad(edge_y, (0, 0, 1, 0))
    if radius:
        edge = F.max_pool2d(edge.float(), 2 * radius + 1, stride=1,
                            padding=radius) > 0
    return edge & valid


def epoch_record_indices(count: int, steps: int, seed: int):
    """Yield reproducible shuffled full passes through the training pool."""
    if count <= 0 or steps < 0:
        raise ValueError("count must be positive and steps nonnegative")
    rng = random.Random(seed)
    emitted = 0
    while emitted < steps:
        indices = list(range(count))
        rng.shuffle(indices)
        for index in indices:
            if emitted == steps:
                break
            yield index
            emitted += 1


def _forward_head(name, head, disparity, logits, left, right):
    if head is None:
        return disparity, logits
    if name == "v1":
        return head(disparity, logits, left)
    return head(disparity, logits, left, right)


@torch.inference_mode()
def evaluate(name, stereo, semantic, head, records: list[dict], root: Path) -> dict:
    if head is not None:
        head.eval()
    disp_meter, sem_meter = DisparityMeter(), SemanticMeter()
    edge_sum = flat_sum = 0.0
    edge_n = flat_n = 0
    for row in records:
        left, right, gt, labels = tensors(read_pair(root, row), "cuda")
        lp, base_d, base_s, top, _ = frozen_outputs(stereo, semantic, left, right)
        rp, _, _ = pad16(right)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_d, pred_s = _forward_head(name, head, base_d, base_s,
                                           lp / 255.0, rp / 255.0)
        pred_d = pred_d[..., top:top + gt.shape[-2], :gt.shape[-1]]
        pred_s = unpad_semantic_logits(pred_s, lp.shape[-2:], top,
                                       gt.shape[-2], gt.shape[-1])
        disp_meter.add(pred_d, gt)
        sem_meter.add(pred_s, labels)
        valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
        edge = disparity_edge_mask(gt)
        flat = valid & ~edge
        error = (pred_d.float() - gt).abs()
        edge_sum += error[edge].sum().item()
        flat_sum += error[flat].sum().item()
        edge_n += int(edge.sum())
        flat_n += int(flat.sum())
    return {**disp_meter.result(), **sem_meter.result(),
            "edge_epe": edge_sum / max(edge_n, 1),
            "nonedge_epe": flat_sum / max(flat_n, 1),
            "edge_pixels": edge_n, "nonedge_pixels": flat_n,
            "pairs": len(records)}


@torch.inference_mode()
def benchmark_latency(name, stereo, semantic, head, row, root: Path,
                      repeats: int = 30) -> dict:
    left, right, _, _ = tensors(read_pair(root, row), "cuda")
    rp, _, _ = pad16(right)
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(repeats + 5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lp, d, s, _, _ = frozen_outputs(stereo, semantic, left, right)
        with torch.autocast("cuda", dtype=torch.float16):
            _forward_head(name, head, d, s, lp / 255.0, rp / 255.0)
        end.record()
        torch.cuda.synchronize()
        if index >= 5:
            durations.append(start.elapsed_time(end))
    return {"median_latency_ms": statistics.median(durations),
            "p95_latency_ms": sorted(durations)[int(0.95 * (len(durations) - 1))],
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20}


def _new_head(name: str):
    if name == "v1":
        return JointResidual(channels=32, semantic=True)
    if name == "gate":
        return GatedResidual(use_warp=False)
    if name == "warp":
        return GatedResidual(use_warp=True)
    raise ValueError(name)


def train_arm(name, stereo, semantic, train_rows, val_rows, root: Path,
              run_dir: Path, steps: int, eval_every: int, seed: int,
              crop: tuple[int, int], seg_weight: float) -> dict:
    arm_dir = run_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    head = _new_head(name).cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                    total_steps=steps, pct_start=0.1)
    crop_rng = random.Random(seed + 1)
    indices = epoch_record_indices(len(train_rows), steps, seed)
    history = []
    best_epe, best_miou = float("inf"), float("-inf")
    start = time.perf_counter()
    for step, row_index in enumerate(indices, start=1):
        sample = read_pair(root, train_rows[row_index])
        height, width = sample[0].shape[:2]
        crop_h, crop_w = crop
        top = crop_rng.randrange(height - crop_h + 1)
        x = crop_rng.randrange(width - crop_w + 1)
        sample = paired_crop(*sample, top, x, crop_h, crop_w)
        left, right, gt, labels = tensors(sample, "cuda")
        # Preserve native disparity units and mask matches cut off by the crop.
        column = torch.arange(crop_w, device="cuda")[None, None, None, :]
        gt = torch.where(column >= gt, gt, 0.0)
        lp, base_d, base_s, _, _ = frozen_outputs(stereo, semantic, left, right)
        rp, _, _ = pad16(right)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_d, pred_s = _forward_head(name, head, base_d, base_s,
                                           lp / 255.0, rp / 255.0)
            valid = gt > 0
            if not bool(valid.any()):
                raise ValueError(f"no valid disparity in training crop at step {step}")
            disp_loss = F.smooth_l1_loss(pred_d[valid].float(), gt[valid])
            low_labels = F.interpolate(labels.float(), size=pred_s.shape[-2:],
                                       mode="nearest").long()[:, 0]
            seg_loss = F.cross_entropy(pred_s.float(), low_labels, ignore_index=255)
            loss = disp_loss + seg_weight * seg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 40 == 0:
            record = {"phase": "train", "step": step,
                      "train_disp_loss": disp_loss.detach().item(),
                      "train_sem_loss": seg_loss.detach().item(),
                      "lr": scheduler.get_last_lr()[0],
                      "elapsed_s": time.perf_counter() - start}
            history.append(record)
            append_history(arm_dir / "history.jsonl", record)
            print(json.dumps({"arm": name, **record}), flush=True)
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(name, stereo, semantic, head, val_rows, root)
            record = {"phase": "val", "step": step, **metrics,
                      "elapsed_s": time.perf_counter() - start}
            history.append(record)
            append_history(arm_dir / "history.jsonl", record)
            print(json.dumps({"arm": name, "validation": {k: v for k, v in record.items()
                                                           if k != "class_iou"}}), flush=True)
            payload = {"head": head.state_dict(), "arm": name, "step": step,
                       "metrics": metrics, "optimizer": optimizer.state_dict()}
            torch.save(payload, arm_dir / f"step_{step:05d}.pth")
            if metrics["epe"] < best_epe:
                best_epe = metrics["epe"]
                torch.save(payload, arm_dir / "best_epe.pth")
            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                torch.save(payload, arm_dir / "best_miou.pth")
            save_plot(history, arm_dir / "curves.png")
            head.train()
    head.eval()
    latency = benchmark_latency(name, stereo, semantic, head, val_rows[0], root)
    return {"best_epe": best_epe, "best_miou": best_miou,
            "last": metrics, "trainable_params": sum(p.numel() for p in head.parameters()),
            "elapsed_s": time.perf_counter() - start, **latency}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260930)
    parser.add_argument("--seg-weight", type=float, default=0.5)
    parser.add_argument("--run-dir", type=Path,
                        default=REPO / "experiments/vkitti2/runs/ablation_v2_1000")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v2 ablation requires the local NVIDIA GPU")
    if args.steps < 20:
        raise ValueError("OneCycle requires at least 20 steps at 10% warm-up")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    manifest = args.data / "manifest.json"
    rows = json.loads(manifest.read_text())
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 800 or len(val_rows) != 200:
        raise ValueError("expected exactly 800 train and 200 held-out pairs")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data": str(args.data), "ckpt": str(args.ckpt),
              "run_dir": str(args.run_dir), "manifest_sha256": _sha256(manifest),
              "checkpoint_sha256": _sha256(args.ckpt),
              "v1_head_code_sha256": _sha256(REPO / "experiments/vkitti2/model.py"),
              "v2_head_code_sha256": _sha256(REPO / "experiments/vkitti2/model_v2.py"),
              "runner_code_sha256": _sha256(Path(__file__)),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "resolution": "native 375x1242 validation",
              "arms": ["baseline", "v1", "gate", "warp"],
              "edge_arm_skipped": "Prior 500-pair held-out edge EPE was 1.19x non-edge EPE",
              "right_crop_validity": "x >= disparity_px"}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stereo = FusionStereoLite("V", encoder=YOLO_WEIGHTS).cuda().eval()
    stereo.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    semantic = YOLO(str(YOLO_WEIGHTS)).model.cuda().eval()
    for model in (stereo, semantic):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    assert not any(p.requires_grad for p in stereo.parameters())
    assert not any(p.requires_grad for p in semantic.parameters())
    summary = {}
    baseline = evaluate("baseline", stereo, semantic, None, val_rows, args.data)
    summary["baseline"] = {**baseline, "trainable_params": 0,
                           **benchmark_latency("baseline", stereo, semantic,
                                               None, val_rows[0], args.data)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": "baseline", **{k: v for k, v in summary["baseline"].items()
                                                if k != "class_iou"}}), flush=True)
    for name in ("v1", "gate", "warp"):
        summary[name] = train_arm(name, stereo, semantic, train_rows, val_rows,
                                  args.data, args.run_dir, args.steps,
                                  args.eval_every, args.seed,
                                  (args.crop_height, args.crop_width), args.seg_weight)
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[complete] all v2 1,000-pair arms finished", flush=True)


if __name__ == "__main__":
    main()
