"""Local RTX 3050 feasibility ablation: frozen baseline, depth-only, joint.

No image resizing. Training uses co-located native-pixel crops; validation is
at the original 375x1242 resolution with replicate padding only.

    uv run python -m experiments.vkitti2.run --steps 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

from experiments.vkitti2.data import depth_to_disparity, map_vkitti_to_ade, paired_crop
from experiments.vkitti2.model import JointResidual, MAPPED_ADE_IDS


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/a06_shallow/runners"))
from model_a07v2 import FusionStereoLite  # noqa: E402

DEFAULT_DATA = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation500")
DEFAULT_CKPT = Path("/media/abrar/AbrarSSD/ResearchArtifacts/SVDE/final_pass/"
                    "a09m_fullsf_a10_v1_20260929/checkpoints/best.pth")
YOLO_WEIGHTS = REPO / "models/segmentation/yolo26m-sem-ade20k.pt"


class DisparityMeter:
    def __init__(self):
        self.n = 0
        self.totals = {key: 0.0 for key in
                       ("epe", "sqerr", "bad_0.5", "bad_1", "bad_2", "bad_3", "d1")}

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        valid = torch.isfinite(target) & (target > 0) & (target < 192)
        err = (prediction.float() - target.float()).abs()[valid]
        gt = target[valid]
        self.n += int(err.numel())
        self.totals["epe"] += err.sum().item()
        self.totals["sqerr"] += err.square().sum().item()
        for t in (0.5, 1, 2, 3):
            self.totals[f"bad_{t}"] += (err > t).sum().item()
        self.totals["d1"] += ((err > 3) & (err / gt.clamp_min(1e-6) > 0.05)).sum().item()

    def result(self) -> dict:
        n = max(self.n, 1)
        return {"n": self.n, "epe": self.totals["epe"] / n,
                "rmse": (self.totals["sqerr"] / n) ** 0.5,
                **{key: 100 * self.totals[key] / n for key in
                   ("bad_0.5", "bad_1", "bad_2", "bad_3", "d1")}}


class SemanticMeter:
    def __init__(self):
        self.intersection = {class_id: 0 for class_id in MAPPED_ADE_IDS}
        self.union = {class_id: 0 for class_id in MAPPED_ADE_IDS}
        self.correct = 0
        self.n = 0

    def add(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear",
                             align_corners=False).argmax(dim=1)
        target = target[:, 0]
        valid = target != 255
        self.correct += int(((pred == target) & valid).sum())
        self.n += int(valid.sum())
        for class_id in MAPPED_ADE_IDS:
            p = (pred == class_id) & valid
            g = target == class_id
            self.intersection[class_id] += int((p & g).sum())
            self.union[class_id] += int((p | g).sum())

    def result(self) -> dict:
        ious = {str(c): self.intersection[c] / self.union[c]
                for c in MAPPED_ADE_IDS if self.union[c]}
        return {"miou": sum(ious.values()) / max(len(ious), 1),
                "pixel_accuracy": self.correct / max(self.n, 1),
                "class_iou": ious, "semantic_pixels": self.n}


def read_pair(root: Path, row: dict) -> tuple[np.ndarray, ...]:
    paths = row["files"]
    left = cv2.imread(str(root / paths["rgb_left"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(root / paths["rgb_right"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / paths["depth"]), cv2.IMREAD_UNCHANGED)
    class_mask = cv2.imread(str(root / paths["class_seg"]), cv2.IMREAD_COLOR)
    if any(item is None for item in (left, right, depth, class_mask)):
        raise FileNotFoundError(f"incomplete pair {row['scene']}/{row['variation']}/{row['frame']}")
    if depth.dtype != np.uint16 or left.shape != right.shape or depth.shape != left.shape[:2]:
        raise ValueError(f"incompatible shapes/dtype in {row['scene']}/{row['variation']}")
    disparity, _ = depth_to_disparity(depth, row["fx"], row["baseline_m"], 192)
    labels = map_vkitti_to_ade(cv2.cvtColor(class_mask, cv2.COLOR_BGR2RGB))
    return (cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right, cv2.COLOR_BGR2RGB), disparity, labels)


def tensors(sample, device: str):
    left, right, disparity, labels = sample
    def image(x):
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).to(device).float()[None]
    gt = torch.from_numpy(np.ascontiguousarray(disparity))[None, None].to(device).float()
    seg = torch.from_numpy(np.ascontiguousarray(labels.astype(np.int64)))[None, None].to(device)
    return image(left), image(right), gt, seg


def pad16(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    top, right = (-x.shape[-2]) % 16, (-x.shape[-1]) % 16
    return F.pad(x, (0, right, top, 0), mode="replicate"), top, right


def unpad_semantic_logits(logits: torch.Tensor, padded_size: tuple[int, int],
                          top: int, height: int, width: int) -> torch.Tensor:
    full = F.interpolate(logits.float(), size=padded_size, mode="bilinear",
                         align_corners=False)
    return full[..., top:top + height, :width]


def frozen_outputs(stereo: nn.Module, semantic: nn.Module,
                   left: torch.Tensor, right: torch.Tensor):
    lp, top, rp = pad16(left)
    rt, _, _ = pad16(right)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        disparity = stereo(lp, rt)
        logits = semantic(lp / 255.0)
    return lp, disparity, logits, top, rp


@torch.no_grad()
def evaluate(stereo, semantic, head, rows: list[dict], root: Path) -> dict:
    d_meter, s_meter = DisparityMeter(), SemanticMeter()
    if head is not None:
        head.eval()
    for row in rows:
        left, right, gt, labels = tensors(read_pair(root, row), "cuda")
        lp, base_d, base_s, top, _ = frozen_outputs(stereo, semantic, left, right)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_d, pred_s = (head(base_d, base_s, lp / 255.0) if head is not None
                              else (base_d, base_s))
        pred_d = pred_d[..., top:top + gt.shape[-2], :gt.shape[-1]]
        pred_s = unpad_semantic_logits(pred_s, lp.shape[-2:], top,
                                       gt.shape[-2], gt.shape[-1])
        d_meter.add(pred_d, gt)
        s_meter.add(pred_s, labels)
    return {**d_meter.result(), **s_meter.result(), "pairs": len(rows)}


def save_plot(history: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ("train_disp_loss", "epe", "miou"),
                              ("training disparity loss", "held-out EPE (px)", "held-out mIoU")):
        points = [(r["step"], r[key]) for r in history if key in r]
        if points:
            ax.plot(*zip(*points), marker="." if key != "train_disp_loss" else None)
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def append_history(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def train_arm(arm: str, stereo, semantic, train_rows, val_rows,
              root: Path, run_dir: Path, steps: int, eval_every: int,
              crop: tuple[int, int], seed: int, seg_weight: float):
    arm_dir = run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    head = JointResidual(channels=32, semantic=(arm == "joint")).cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                    total_steps=steps, pct_start=0.1)
    rng = random.Random(seed)
    history = []
    best_epe = float("inf")
    begin = time.perf_counter()
    for step in range(1, steps + 1):
        row = rng.choice(train_rows)
        sample = read_pair(root, row)
        h, w = sample[0].shape[:2]
        ch, cw = crop
        top, x = rng.randrange(h - ch + 1), rng.randrange(w - cw + 1)
        sample = paired_crop(*sample, top, x, ch, cw)
        left, right, gt, labels = tensors(sample, "cuda")
        # A same-x crop excludes right-image matches for left pixels x < disparity.
        col = torch.arange(cw, device="cuda")[None, None, None, :]
        gt = torch.where(col >= gt, gt, 0.0)
        lp, base_d, base_s, _, _ = frozen_outputs(stereo, semantic, left, right)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred_d, pred_s = head(base_d, base_s, lp / 255.0)
            valid = gt > 0
            disp_loss = F.smooth_l1_loss(pred_d[valid].float(), gt[valid])
            if arm == "joint":
                low_labels = F.interpolate(labels.float(), size=pred_s.shape[-2:],
                                           mode="nearest").long()[:, 0]
                seg_loss = F.cross_entropy(pred_s.float(), low_labels, ignore_index=255)
            else:
                seg_loss = disp_loss.new_zeros(())
            loss = disp_loss + seg_weight * seg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0:
            record = {"phase": "train", "step": step,
                      "train_disp_loss": disp_loss.detach().item(),
                      "train_sem_loss": seg_loss.detach().item(), "lr": scheduler.get_last_lr()[0],
                      "elapsed_s": time.perf_counter() - begin}
            history.append(record)
            append_history(arm_dir / "history.jsonl", record)
            print(json.dumps({"arm": arm, **record}), flush=True)
        if step % eval_every == 0 or step == steps:
            result = evaluate(stereo, semantic, head, val_rows, root)
            record = {"phase": "val", "step": step, **result,
                      "elapsed_s": time.perf_counter() - begin}
            history.append(record)
            print(json.dumps({"arm": arm, "validation": record}), flush=True)
            append_history(arm_dir / "history.jsonl", record)
            payload = {"head": head.state_dict(), "arm": arm, "step": step,
                       "metrics": result, "optimizer": optimizer.state_dict()}
            torch.save(payload, arm_dir / f"step_{step:05d}.pth")
            if result["epe"] < best_epe:
                best_epe = result["epe"]
                torch.save(payload, arm_dir / "best_epe.pth")
            save_plot(history, arm_dir / "curves.png")
            head.train()
    return {"best_epe": best_epe, "last": result, "trainable_params":
            sum(p.numel() for p in head.parameters()), "elapsed_s": time.perf_counter() - begin}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260930)
    parser.add_argument("--seg-weight", type=float, default=0.5)
    parser.add_argument("--run-dir", type=Path, default=REPO / "experiments/vkitti2/runs/ablation_v1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this ablation requires the local NVIDIA GPU")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    manifest_path = args.data / "manifest.json"
    rows = json.loads(manifest_path.read_text())
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 400 or len(val_rows) != 100:
        raise ValueError("expected the locked 400/100 manifest")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = {**vars(args), "data": str(args.data), "ckpt": str(args.ckpt),
              "run_dir": str(args.run_dir), "manifest_sha256": sha256(manifest_path),
              "checkpoint_sha256": sha256(args.ckpt), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "input_resolution": "native 375x1242 val",
              "arms": ["baseline", "depth_only", "joint"]}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stereo = FusionStereoLite("V", encoder=YOLO_WEIGHTS).cuda().eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stereo.load_state_dict(checkpoint["model"])
    semantic = YOLO(str(YOLO_WEIGHTS)).model.cuda().eval()
    for model in (stereo, semantic):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    assert not any(p.requires_grad for p in stereo.parameters())
    assert not any(p.requires_grad for p in semantic.parameters())
    base = evaluate(stereo, semantic, None, val_rows, args.data)
    print(json.dumps({"arm": "baseline", "validation": base}), flush=True)
    summary = {"baseline": {**base, "trainable_params": 0}}
    (args.run_dir / "baseline.json").write_text(json.dumps(summary["baseline"], indent=2) + "\n")
    for arm in ("depth_only", "joint"):
        summary[arm] = train_arm(arm, stereo, semantic, train_rows, val_rows,
                                 args.data, args.run_dir, args.steps, args.eval_every,
                                 (args.crop_height, args.crop_width), args.seed,
                                 args.seg_weight)
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
