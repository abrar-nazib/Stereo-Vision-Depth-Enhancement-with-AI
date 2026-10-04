"""Local G-series visual-edge ablations on the frozen E3 model and VKITTI-1k."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from experiments.B.B_0_fused_baseline.run import DATA, crop_valid_mask, read_pair, sha256, split_rows, to_tensors
from experiments.G.model import ARMS, EdgeHead
from experiments.vkitti2.data import paired_crop
from experiments.vkitti2.run import DisparityMeter
from semtilestereo.core import ModelPaths, load_model


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/G/runs"


def edge_mask(gt: torch.Tensor) -> torch.Tensor:
    """5-pixel neighborhood of a >2 px GT jump, excluding invalid depth."""
    valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
    horizontal = (gt[..., 1:] - gt[..., :-1]).abs() > 2
    vertical = (gt[..., 1:, :] - gt[..., :-1, :]).abs() > 2
    edges = torch.zeros_like(valid)
    edges[..., 1:] |= horizontal & valid[..., 1:] & valid[..., :-1]
    edges[..., :-1] |= horizontal & valid[..., 1:] & valid[..., :-1]
    edges[..., 1:, :] |= vertical & valid[..., 1:, :] & valid[..., :-1, :]
    edges[..., :-1, :] |= vertical & valid[..., 1:, :] & valid[..., :-1, :]
    return F.max_pool2d(edges.float(), 5, stride=1, padding=2).bool() & valid


class FrozenE3:
    def __init__(self):
        self.model = load_model(ModelPaths(), "cuda")
        self.model.requires_grad_(False)
        self.half: torch.Tensor | None = None
        self.model.base.stereo.up_2_to_1.register_forward_pre_hook(self._capture)

    def _capture(self, _module, args):
        self.half = args[0].d.detach()

    @torch.no_grad()
    def __call__(self, left: torch.Tensor, right: torch.Tensor):
        self.half = None
        with torch.autocast("cuda", dtype=torch.float16):
            base, _ = self.model(left, right)
        if self.half is None or self.half.shape[-2:] != tuple(n // 2 for n in base.shape[-2:]):
            raise RuntimeError("failed to capture E3's half-resolution tile disparity")
        return base.detach().float(), self.half.float()


def pad_pair(left: torch.Tensor, right: torch.Tensor):
    h, w = left.shape[-2:]
    top, right_pad = (-h) % 32, (-w) % 32
    return (F.pad(left, (0, right_pad, top, 0), mode="replicate"),
            F.pad(right, (0, right_pad, top, 0), mode="replicate"), top)


@torch.no_grad()
def evaluate(e3: FrozenE3, head: EdgeHead, rows: list[dict], data: Path,
             visuals: Path | None = None) -> dict:
    head.eval()
    meter, baseline = DisparityMeter(), DisparityMeter()
    edge_sum = edge_n = 0
    pairs = []
    visual_count = 0
    for row in rows:
        left, right, gt, labels = to_tensors(read_pair(data, row))
        lp, rp, top = pad_pair(left, right)
        base, half = e3(lp, rp)
        pred = head(base, half, lp / 255, rp / 255)
        h, w = gt.shape[-2:]
        base = base[..., top:top + h, :w]
        pred = pred[..., top:top + h, :w]
        meter.add(pred, gt)
        baseline.add(base, gt)
        valid_edges = edge_mask(gt)
        edge_sum += (pred[valid_edges] - gt[valid_edges]).abs().sum().item()
        edge_n += int(valid_edges.sum())
        pair = DisparityMeter()
        pair.add(pred, gt)
        pairs.append({"scene": row["scene"], "variation": row["variation"],
                      "frame": row["frame"], **pair.result()})
        if visuals is not None and row["variation"] == "clone" and visual_count < 3:
            visuals.mkdir(parents=True, exist_ok=True)
            def disparity_image(tensor):
                array = tensor[0, 0].float().cpu().numpy()
                image = cv2.applyColorMap(np.uint8(np.clip(array / 80, 0, 1) * 255),
                                          cv2.COLORMAP_TURBO)
                image[array <= 0] = 0
                return image
            rgb = cv2.cvtColor(left[0].permute(1, 2, 0).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
            panels = [rgb, disparity_image(gt), disparity_image(base), disparity_image(pred)]
            labels_text = ("Left RGB", "Ground truth", "Frozen E3", head.arm)
            stamped = []
            for panel, label in zip(panels, labels_text):
                banner = np.full((38, w, 3), 255, np.uint8)
                cv2.putText(banner, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 2, cv2.LINE_AA)
                stamped.append(np.vstack((banner, panel)))
            cv2.imwrite(str(visuals / f"{row['scene']}__{row['variation']}__{row['frame']:05d}.png"),
                        np.hstack(stamped))
            visual_count += 1
    return {**meter.result(), "baseline": baseline.result(), "edge_epe": edge_sum / max(edge_n, 1),
            "edge_pixels": edge_n, "pairs": len(rows), "pair_results": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", default="g_edge_native_seed42_10k_v2_20261004")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=None, help="Smoke only: restrict val/test pairs")
    parser.add_argument("--train-limit", type=int, default=None, help="Smoke only: restrict train pairs")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU required")
    rows = json.loads((args.data / "manifest.json").read_text())
    train, val, test = split_rows(rows, 42)
    if args.train_limit is not None:
        train = train[:args.train_limit]
    if args.eval_limit is not None:
        val, test = val[:args.eval_limit], test[:args.eval_limit]
    if not train or not val or not test:
        raise ValueError("empty split")
    output = RUNS / args.arm / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    random.seed(42)
    rng = random.Random(42)
    e3 = FrozenE3()
    head = EdgeHead(args.arm).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4,
                                                    total_steps=args.steps, pct_start=0.1)
    paths = ModelPaths()
    manifest = {"arm": args.arm, "run_id": args.run_id, "seed": 42,
                "split": {"train": len(train), "val": len(val), "test": len(test)},
                "split_rule": "B-series frame-grouped Scene20 holdout", "steps": args.steps,
                "eval_every": args.eval_every, "crop_hw": [256, 512], "resize": False,
                "base_frozen": True, "trainable_params": sum(p.numel() for p in head.parameters()),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "optimizer": "AdamW lr2e-4 wd1e-4, OneCycle 10% warmup",
                "loss": "valid SmoothL1 + 0.5 edge SmoothL1 + 0.2 bad1 hinge",
                "selection": "lowest validation edge_epe", "test_only_after_selection": True,
                "dataset_sha256": sha256(args.data / "manifest.json"),
                "e3_head_sha256": sha256(paths.head), "stereo_sha256": sha256(paths.stereo),
                "semantic_sha256": sha256(paths.semantic), "encoder_sha256": sha256(paths.encoder),
                "code_sha256": {"runner": sha256(Path(__file__)),
                                "head": sha256(Path(__file__).with_name("model.py"))},
                "limitations": "E3 semantic/depth teacher saw full VKITTI; this is visual in-domain screening"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps({"train": train, "val": val, "test": test}) + "\n")
    history = output / "history.jsonl"
    started = time.monotonic()
    best = float("inf")
    for step in range(1, args.steps + 1):
        row = train[rng.randrange(len(train))]
        sample = read_pair(args.data, row)
        h, w = sample[0].shape[:2]
        top = rng.randrange((h - 256) // 32 + 1) * 32
        x = rng.randrange((w - 512) // 32 + 1) * 32
        left, right, gt, _ = to_tensors(paired_crop(*sample, top, x, 256, 512))
        base, half = e3(left, right)
        head.train()
        pred = head(base, half, left / 255, right / 255)
        valid = crop_valid_mask(gt)
        boundary = edge_mask(gt) & valid
        error = (pred[valid] - gt[valid]).abs()
        loss = F.smooth_l1_loss(pred[valid], gt[valid]) + 0.2 * F.relu(error - 1).mean()
        if boundary.any():
            loss = loss + 0.5 * F.smooth_l1_loss(pred[boundary], gt[boundary])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            record = {"phase": "train", "step": step, "loss": float(loss.detach()),
                      "lr": scheduler.get_last_lr()[0], "elapsed_s": time.monotonic() - started}
            with history.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_metrics = evaluate(e3, head, val, args.data)
            (output / f"validation_step_{step:06d}.json").write_text(json.dumps(val_metrics) + "\n")
            record = {"phase": "val", "step": step, "edge_epe": val_metrics["edge_epe"],
                      "epe": val_metrics["epe"], "bad_1": val_metrics["bad_1"],
                      "bad_3": val_metrics["bad_3"], "elapsed_s": time.monotonic() - started}
            with history.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if val_metrics["edge_epe"] < best:
                best = val_metrics["edge_epe"]
                torch.save({"head": head.state_dict(), "step": step, "manifest": manifest},
                           output / "best.pth")
    saved = torch.load(output / "best.pth", map_location="cuda", weights_only=False)
    head.load_state_dict(saved["head"])
    test_metrics = evaluate(e3, head, test, args.data, output / "visuals")
    (output / "test.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"selected_step": saved["step"],
                                                     "best_val_edge_epe": best,
                                                     "test": {k: v for k, v in test_metrics.items()
                                                              if k != "pair_results"}}, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "arm": args.arm,
                      "selected_step": saved["step"], "test_epe": test_metrics["epe"],
                      "test_edge_epe": test_metrics["edge_epe"]}), flush=True)


if __name__ == "__main__":
    main()
