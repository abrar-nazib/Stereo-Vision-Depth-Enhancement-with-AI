"""Shared B-series data protocol and local RTX 3050 runner.

Run from repository root with ``uv run python -m experiments.B.B_0_fused_baseline.run``.
No resize is applied to stereo images or disparity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.model import FusedStereoSemantic
from experiments.B.B_1_residual.head import ResidualHead
from experiments.B.B_2_confidence.head import ConfidenceHead
from experiments.B.B_3_classaware.head import ClassAwareHead
from experiments.vkitti2.data import depth_to_disparity, paired_crop
from experiments.vkitti2.run import DisparityMeter
from experiments.vkitti2.semantic_data import decode_class_mask


ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation1000")
STEREO = Path("/media/abrar/AbrarSSD/ResearchArtifacts/SVDE/final_pass/"
              "a09m_fullsf_a10_v1_20260929/checkpoints/best.pth")
SEMANTIC = ROOT / "models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt"
ADE = ROOT / "models/segmentation/yolo26m-sem-ade20k.pt"
ARM_DIRS = {
    "B_1_residual": ROOT / "experiments/B/B_1_residual",
    "B_2_confidence": ROOT / "experiments/B/B_2_confidence",
    "B_3_classaware": ROOT / "experiments/B/B_3_classaware",
}
HEADS = {"B_1_residual": ResidualHead,
         "B_2_confidence": ConfidenceHead,
         "B_3_classaware": ClassAwareHead}


def split_rows(rows: list[dict], seed: int = 42):
    """800 training pairs and two Scene20 frame-grouped sets of 100."""
    train = [row for row in rows if row["scene"] != "Scene20"]
    held = [row for row in rows if row["scene"] == "Scene20"]
    frames = sorted({row["frame"] for row in held})
    if len(train) != 800 or len(held) != 200 or len(frames) != 20:
        raise ValueError("expected the complete 1,000-pair five-scene subset")
    random.Random(seed).shuffle(frames)
    val_frames = set(frames[:10])
    validation = [row for row in held if row["frame"] in val_frames]
    test = [row for row in held if row["frame"] not in val_frames]
    if len(validation) != 100 or len(test) != 100:
        raise ValueError("each Scene20 frame group must have ten variations")
    return train, validation, test


def label_mask(rgb: np.ndarray) -> np.ndarray:
    return decode_class_mask(rgb, "B-series VKITTI class mask")


def crop_valid_mask(disparity: torch.Tensor) -> torch.Tensor:
    """Mask left pixels whose same-x right crop excludes their match."""
    columns = torch.arange(disparity.shape[-1], device=disparity.device)[None, None, None, :]
    return torch.isfinite(disparity) & (disparity > 0) & (disparity < 192) & (columns >= disparity)


class SemanticMeter14:
    def __init__(self):
        self.intersection = np.zeros(14, dtype=np.int64)
        self.union = np.zeros(14, dtype=np.int64)
        self.correct = 0
        self.pixels = 0

    def add(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pred = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear",
                             align_corners=False).argmax(dim=1)
        target = labels[:, 0]
        valid = target != 255
        self.correct += int(((pred == target) & valid).sum())
        self.pixels += int(valid.sum())
        for class_id in range(14):
            p = (pred == class_id) & valid
            g = (target == class_id) & valid
            self.intersection[class_id] += int((p & g).sum())
            self.union[class_id] += int((p | g).sum())

    def result(self) -> dict:
        ious = {str(c): float(self.intersection[c] / self.union[c])
                for c in range(14) if self.union[c]}
        return {"miou": sum(ious.values()) / max(len(ious), 1),
                "pixel_accuracy": self.correct / max(self.pixels, 1),
                "class_iou": ious, "semantic_pixels": self.pixels}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pair(root: Path, row: dict):
    paths = row["files"]
    left = cv2.imread(str(root / paths["rgb_left"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(root / paths["rgb_right"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / paths["depth"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(root / paths["class_seg"]), cv2.IMREAD_COLOR)
    if any(item is None for item in (left, right, depth, mask)):
        raise FileNotFoundError(f"incomplete VKITTI pair: {row['scene']}/{row['variation']}/{row['frame']}")
    if depth.dtype != np.uint16 or left.shape != right.shape or left.shape[:2] != depth.shape:
        raise ValueError("invalid VKITTI shape or depth dtype")
    disparity, _ = depth_to_disparity(depth, row["fx"], row["baseline_m"], 192)
    labels = label_mask(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB), disparity, labels


def to_tensors(sample):
    left, right, disparity, labels = sample
    device = "cuda"
    def image(value):
        return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).to(device).float()[None]
    d = torch.from_numpy(np.ascontiguousarray(disparity))[None, None].to(device).float()
    s = torch.from_numpy(np.ascontiguousarray(labels))[None, None].to(device).long()
    return image(left), image(right), d, s


def pad32(image: torch.Tensor):
    top = (-image.shape[-2]) % 32
    right = (-image.shape[-1]) % 32
    return F.pad(image, (0, right, top, 0), mode="replicate"), top


@torch.inference_mode()
def evaluate(model, head, rows, root: Path, control: str = "normal") -> dict:
    disparity_meter, semantic_meter = DisparityMeter(), SemanticMeter14()
    class_error = np.zeros(14, dtype=np.float64)
    class_pixels = np.zeros(14, dtype=np.int64)
    for row in rows:
        left, right, gt, labels = to_tensors(read_pair(root, row))
        lp, top = pad32(left)
        rp, _ = pad32(right)
        with torch.autocast("cuda", dtype=torch.float16):
            base, logits = model(lp, rp)
        if head is not None:
            if control == "uniform":
                guidance = torch.zeros_like(logits)
            elif control == "shifted":
                guidance = logits.roll(shifts=max(1, logits.shape[-1] // 3), dims=-1)
            else:
                guidance = logits
            prediction = head(base.float(), guidance.float(), lp / 255.0, rp / 255.0)
        else:
            prediction = base
        prediction = prediction[..., top:top + gt.shape[-2], :gt.shape[-1]]
        disparity_meter.add(prediction, gt)
        full_logits = F.interpolate(logits.float(), size=lp.shape[-2:], mode="bilinear",
                                    align_corners=False)
        semantic_meter.add(full_logits[..., top:top + gt.shape[-2], :gt.shape[-1]], labels)
        valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
        error = (prediction.float() - gt).abs()
        for class_id in range(14):
            region = valid & (labels == class_id)
            class_pixels[class_id] += int(region.sum())
            class_error[class_id] += error[region].sum().item()
    return {**disparity_meter.result(), **semantic_meter.result(),
            "class_disparity_epe": {str(c): float(class_error[c] / class_pixels[c])
                                    for c in range(14) if class_pixels[c]},
            "class_disparity_pixels": {str(c): int(class_pixels[c])
                                       for c in range(14) if class_pixels[c]},
            "pairs": len(rows), "guidance_control": control}


def _class_boundary_smoothness(delta, labels, valid):
    """Constrain corrections inside a class, never force disparity edges at labels."""
    same_x = (labels[..., 1:] == labels[..., :-1]) & valid[..., 1:] & valid[..., :-1]
    same_y = (labels[..., 1:, :] == labels[..., :-1, :]) & valid[..., 1:, :] & valid[..., :-1, :]
    dx = (delta[..., 1:] - delta[..., :-1]).abs()
    dy = (delta[..., 1:, :] - delta[..., :-1, :]).abs()
    return dx[same_x].mean() + dy[same_y].mean() if same_x.any() and same_y.any() else delta.new_zeros(())


def _append(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def train_arm(name: str, model, train_rows, val_rows, test_rows, data: Path,
              run_dir: Path, steps: int, eval_every: int, seed: int,
              crop_h: int, crop_w: int, eval_limit: int | None = None):
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    head = HEADS[name]().cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                     total_steps=steps, pct_start=0.1)
    rng = random.Random(seed)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    cursor = 0
    best_epe = float("inf")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if cursor >= len(indices):
            rng.shuffle(indices)
            cursor = 0
        row = train_rows[indices[cursor]]
        cursor += 1
        sample = read_pair(data, row)
        h, w = sample[0].shape[:2]
        top, x = rng.randrange(h - crop_h + 1), rng.randrange(w - crop_w + 1)
        left, right, gt, labels = to_tensors(paired_crop(*sample, top, x, crop_h, crop_w))
        valid = crop_valid_mask(gt)
        if not bool(valid.any()):
            raise ValueError(f"empty valid crop at step {step}")
        lp, _ = pad32(left)
        rp, _ = pad32(right)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            base, logits = model(lp, rp)
        optimizer.zero_grad(set_to_none=True)
        predicted = head(base.float(), logits.float(), lp / 255.0, rp / 255.0)
        loss_disp = F.smooth_l1_loss(predicted[valid], gt[valid])
        if name == "B_3_classaware":
            loss_smooth = _class_boundary_smoothness(predicted - base.float(), labels, valid)
            loss = loss_disp + 0.01 * loss_smooth
        else:
            loss_smooth = loss_disp.new_zeros(())
            loss = loss_disp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 40 == 0:
            record = {"phase": "train", "step": step,
                      "loss": float(loss.detach()), "disp_loss": float(loss_disp.detach()),
                      "smooth_loss": float(loss_smooth.detach()),
                      "lr": scheduler.get_last_lr()[0],
                      "elapsed_s": time.perf_counter() - start}
            _append(run_dir / "history.jsonl", record)
            print(json.dumps({"arm": name, **record}), flush=True)
        if step % eval_every == 0 or step == steps:
            head.eval()
            metrics = evaluate(model, head, val_rows[:eval_limit], data)
            record = {"phase": "val", "step": step, **metrics,
                      "elapsed_s": time.perf_counter() - start}
            _append(run_dir / "history.jsonl", record)
            print(json.dumps({"arm": name, "phase": "val", "step": step,
                              "epe": metrics["epe"], "bad_1": metrics["bad_1"],
                              "miou": metrics["miou"]}), flush=True)
            payload = {"head": head.state_dict(), "step": step, "val": metrics,
                       "arm": name}
            torch.save(payload, run_dir / f"step_{step:05d}.pth")
            if metrics["epe"] < best_epe:
                best_epe = metrics["epe"]
                torch.save(payload, run_dir / "best_epe.pth")
            head.train()
    head.load_state_dict(torch.load(run_dir / "best_epe.pth", map_location="cpu",
                                    weights_only=False)["head"])
    head.eval()
    test_metrics = evaluate(model, head, test_rows[:eval_limit], data)
    controls = {control: evaluate(model, head, val_rows[:eval_limit], data, control)
                for control in ("uniform", "shifted")}
    summary = {"arm": name, "best_validation_epe": best_epe,
               "selected_step": torch.load(run_dir / "best_epe.pth", map_location="cpu",
                                           weights_only=False)["step"],
               "test": test_metrics, "validation_controls": controls,
               "trainable_params": sum(p.numel() for p in head.parameters()),
               "elapsed_s": time.perf_counter() - start}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--run-id", default="20261001_seed42")
    parser.add_argument("--arms", nargs="+", choices=list(ARM_DIRS), default=list(ARM_DIRS))
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="smoke-test only: restrict validation/test image count")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B-series requires the local CUDA GPU")
    if args.steps < 20 or not 1 <= args.eval_every <= args.steps:
        raise ValueError("OneCycle requires at least 20 steps and a valid evaluation interval")
    rows = json.loads((args.data / "manifest.json").read_text())
    train, val, test = split_rows(rows, args.seed)
    manifest = {"seed": args.seed, "train": train, "val": val, "test": test}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = FusedStereoSemantic(STEREO, SEMANTIC, ADE).cuda().eval()
    protocol = {"seed": args.seed, "steps": args.steps, "eval_every": args.eval_every,
                "crop": [args.crop_height, args.crop_width],
                "native_evaluation": True, "resize": False,
                "semantic_classes": 14, "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "manifest_sha256": sha256(args.data / "manifest.json"),
                "stereo_sha256": sha256(STEREO), "semantic_sha256": sha256(SEMANTIC),
                "fused_code_sha256": sha256(Path(__file__).with_name("model.py")),
                "runner_code_sha256": sha256(Path(__file__)),
                "eval_limit": args.eval_limit,
                "limitations": "VKITTI semantic teacher's full-dataset random split likely overlaps this subset"}
    base_dir = ROOT / "experiments/B/B_0_fused_baseline/runs" / args.run_id
    base_dir.mkdir(parents=True, exist_ok=False)
    (base_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (base_dir / "config.json").write_text(json.dumps(protocol, indent=2) + "\n")
    baseline_val = evaluate(model, None, val[:args.eval_limit], args.data)
    (base_dir / "validation.json").write_text(json.dumps(baseline_val, indent=2) + "\n")
    print(json.dumps({"arm": "B_0_fused_baseline", "phase": "val",
                      "epe": baseline_val["epe"], "miou": baseline_val["miou"]}), flush=True)
    for arm in args.arms:
        arm_dir = ARM_DIRS[arm] / "runs" / args.run_id
        arm_dir.parent.mkdir(parents=True, exist_ok=True)
        (arm_dir.parent / "README.md").write_text(
            f"{arm}: frozen shared YOLO trunk and frozen A09 stereo; depth-only trainable head.\n")
        summary = train_arm(arm, model, train, val, test, args.data, arm_dir,
                            args.steps, args.eval_every, args.seed,
                            args.crop_height, args.crop_width, args.eval_limit)
        (arm_dir / "config.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print(json.dumps({"arm": arm, "phase": "test", "epe": summary["test"]["epe"],
                          "best_val_epe": summary["best_validation_epe"]}), flush=True)
    baseline_test = evaluate(model, None, test[:args.eval_limit], args.data)
    (base_dir / "test.json").write_text(json.dumps(baseline_test, indent=2) + "\n")
    print(json.dumps({"arm": "B_0_fused_baseline", "phase": "test",
                      "epe": baseline_test["epe"], "miou": baseline_test["miou"]}), flush=True)


if __name__ == "__main__":
    main()
