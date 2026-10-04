"""Stereo-only A09 fine-tuning and native-resolution upsampler comparison."""

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

from experiments.B.B_0_fused_baseline.run import pad32, sha256
from experiments.G.run import edge_mask
from experiments.H.model import ARMS, StereoUpsampleHead
from experiments.vkitti2.data import depth_to_disparity
from experiments.vkitti2.run import DisparityMeter


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/ablation2000")
STEREO = ROOT / "models/stereo/semtilestereo/a09m_sceneflow_best.pth"
ENCODER = ROOT / "models/segmentation/yolo26m-sem-ade20k.pt"
RUNS = ROOT / "experiments/H/runs"


def split_rows_2000(rows: list[dict], seed: int = 42):
    train = [row for row in rows if row["scene"] != "Scene20"]
    held = [row for row in rows if row["scene"] == "Scene20"]
    frames = sorted({row["frame"] for row in held})
    if len(train) != 1800 or len(held) != 200 or len(frames) != 20:
        raise ValueError("expected 1800 non-Scene20 train and 200 Scene20 held-out pairs")
    if any(sum(row["frame"] == frame for row in held) != 10 for frame in frames):
        raise ValueError("Scene20 frame groups must contain ten variations")
    random.Random(seed).shuffle(frames)
    validation_frames = set(frames[:10])
    val = [row for row in held if row["frame"] in validation_frames]
    test = [row for row in held if row["frame"] not in validation_frames]
    return train, val, test


def read_stereo(root: Path, row: dict):
    files = row["files"]
    left = cv2.imread(str(root / files["rgb_left"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(root / files["rgb_right"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / files["depth"]), cv2.IMREAD_UNCHANGED)
    if left is None or right is None or depth is None:
        raise FileNotFoundError(f"missing VKITTI files for {row['scene']}/{row['variation']}/{row['frame']}")
    if depth.dtype != np.uint16 or left.shape != right.shape or left.shape[:2] != depth.shape:
        raise ValueError("invalid stereo/depth shape or dtype")
    disparity, _ = depth_to_disparity(depth, row["fx"], row["baseline_m"], 192)
    return (cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right, cv2.COLOR_BGR2RGB), disparity)


def tensors(sample):
    left, right, disparity = sample
    def rgb(image):
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].cuda().float()
    gt = torch.from_numpy(np.ascontiguousarray(disparity))[None, None].cuda().float()
    return rgb(left), rgb(right), gt


def initialize_stereo(base, initialization: str, checkpoint: Path) -> None:
    if initialization == "scratch":
        return
    if initialization != "sceneflow":
        raise ValueError(f"unknown stereo initialization: {initialization}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(payload["model"], strict=True)


def stereo_state_sha256(base) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(base.state_dict().items()):
        if key.startswith("fnet."):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(arm: str, initialization: str = "scratch"):
    from experiments.B.B_0_fused_baseline.model import FusionStereoLite
    base = FusionStereoLite("V", encoder=ENCODER)
    initialize_stereo(base, initialization, STEREO)
    if any(parameter.requires_grad for parameter in base.fnet.parameters()):
        raise RuntimeError("YOLO encoder must remain frozen")
    return base.cuda(), StereoUpsampleHead(arm).cuda()


def predict(base, head, left, right):
    auxiliary = base(left, right, aux=True)
    output = head(auxiliary["d_final"], auxiliary["d_half"], left / 255)
    return output, auxiliary


def stereo_loss(pred, auxiliary, gt):
    valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
    if not valid.any():
        raise ValueError("crop has no valid disparity")
    error = (pred.float() - gt).abs()
    loss = error[valid].mean()
    for key, scale, weight in (("d_half", 2, 0.5), ("d4", 4, 0.3),
                               ("d8", 8, 0.2), ("d16", 16, 0.1)):
        scaled = F.interpolate(auxiliary[key].float(), size=gt.shape[-2:],
                               mode="bilinear", align_corners=False) * scale
        loss = loss + weight * (scaled - gt).abs()[valid].mean()
    vx = valid[..., :, 1:] & valid[..., :, :-1]
    vy = valid[..., 1:, :] & valid[..., :-1, :]
    gx = (pred[..., :, 1:].float() - pred[..., :, :-1].float()) - (gt[..., :, 1:] - gt[..., :, :-1])
    gy = (pred[..., 1:, :].float() - pred[..., :-1, :].float()) - (gt[..., 1:, :] - gt[..., :-1, :])
    loss = loss + 0.5 * (gx.abs()[vx].mean() + gy.abs()[vy].mean())
    loss = loss + 0.2 * (F.relu(error[valid] - 1) ** 2).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    return loss


@torch.inference_mode()
def evaluate(base, head, rows, data: Path, visuals: Path | None = None):
    base.eval(); head.eval()
    meter = DisparityMeter()
    edge_sum = edge_n = 0
    pairs = []
    visual_count = 0
    for row in rows:
        left, right, gt = tensors(read_stereo(data, row))
        lp, top = pad32(left)
        rp, _ = pad32(right)
        with torch.autocast("cuda", dtype=torch.float16):
            pred, _ = predict(base, head, lp, rp)
        pred = pred.float()[..., top:top + gt.shape[-2], :gt.shape[-1]]
        meter.add(pred, gt)
        edges = edge_mask(gt)
        edge_sum += (pred[edges] - gt[edges]).abs().sum().item()
        edge_n += int(edges.sum())
        pair = DisparityMeter(); pair.add(pred, gt)
        pairs.append({"scene": row["scene"], "variation": row["variation"],
                      "frame": row["frame"], **pair.result()})
        if visuals is not None and row["variation"] == "clone" and visual_count < 3:
            visuals.mkdir(parents=True, exist_ok=True)
            def color(tensor):
                array = tensor[0, 0].float().cpu().numpy()
                return cv2.applyColorMap(np.uint8(np.clip(array / 80, 0, 1) * 255), cv2.COLORMAP_TURBO)
            image = cv2.cvtColor(left[0].byte().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(visuals / f"{row['scene']}__{row['frame']:05d}.png"),
                        np.hstack((image, color(gt), color(pred))))
            visual_count += 1
    return {**meter.result(), "edge_epe": edge_sum / max(edge_n, 1),
            "edge_pixels": edge_n, "pairs": len(rows), "pair_results": pairs}


def save_checkpoint(path: Path, base, head, optimizer, scheduler, scaler, step: int, manifest: dict):
    temp = path.with_suffix(".tmp")
    torch.save({"base": base.state_dict(), "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "manifest": manifest}, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", default="h_stereo_only_2k_25k_scratchhead_seed42_v2")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--initialization", choices=("scratch", "sceneflow"), default="scratch")
    parser.add_argument("--eval-limit", type=int, default=None, help="smoke only")
    parser.add_argument("--train-limit", type=int, default=None, help="smoke only")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 3050 CUDA GPU required")
    rows = json.loads((args.data / "manifest.json").read_text())
    train, val, test = split_rows_2000(rows)
    if args.train_limit is not None:
        train = train[:args.train_limit]
    if args.eval_limit is not None:
        val, test = val[:args.eval_limit], test[:args.eval_limit]
    output = RUNS / args.arm / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    rng = random.Random(42)
    base, head = build_model(args.arm, args.initialization)
    initial_stereo_sha256 = stereo_state_sha256(base)
    trainable = [p for p in (*base.parameters(), *head.parameters()) if p.requires_grad]
    manifest = {"arm": args.arm, "run_id": args.run_id, "seed": 42,
                "split": {"train": len(train), "validation": len(val), "test": len(test)},
                "steps": args.steps, "eval_every": args.eval_every,
                "crop_hw": [256, 512], "resize": False, "batch": 1,
                "encoder_frozen": True, "stereo_trainable": True,
                "stereo_initialization": args.initialization,
                "initial_stereo_state_sha256": initial_stereo_sha256,
                "trainable_params": sum(p.numel() for p in trainable),
                "gpu": torch.cuda.get_device_name(),
                "optimizer": "AdamW 1e-4 wd1e-4; OneCycle 10% warmup",
                "loss": "A09 multiscale L1 + gradient + bad1 hinge",
                "selection": "lowest validation full-image EPE; test after selection",
                "dataset_sha256": sha256(args.data / "manifest.json"),
                "stereo_source_sha256": sha256(STEREO) if args.initialization == "sceneflow" else None,
                "encoder_sha256": sha256(ENCODER),
                "code_sha256": {"runner": sha256(Path(__file__)),
                                "head": sha256(Path(__file__).with_name("model.py"))},
                "limitations": "VKITTI in-domain training; Scene20 held out from new training only"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps({"train": train, "validation": val, "test": test}) + "\n")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                                                    total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    latest = output / "latest.pth"
    start = 0
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload["manifest"] != manifest:
            raise RuntimeError("resume provenance differs from the requested run")
        base.load_state_dict(payload["base"]); head.load_state_dict(payload["head"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start = payload["step"]
    history = output / "history.jsonl"
    best = float("inf")
    best_file = output / "best.pth"
    if best_file.exists():
        best = torch.load(best_file, map_location="cpu", weights_only=False)["validation_epe"]
    begun = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        row = train[rng.randrange(len(train))]
        sample = read_stereo(args.data, row)
        h, w = sample[0].shape[:2]
        top = rng.randrange((h - 256) // 32 + 1) * 32
        x = rng.randrange((w - 512) // 32 + 1) * 32
        crop = np.s_[top:top + 256, x:x + 512]
        left, right, gt = tensors(tuple(item[crop] for item in sample))
        base.train(); head.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred, auxiliary = predict(base, head, left, right)
            loss = stereo_loss(pred, auxiliary, gt)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        if step == 1 or step % 100 == 0:
            record = {"phase": "train", "step": step, "loss": float(loss.detach()),
                      "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.monotonic() - begun}
            with history.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(base, head, val, args.data)
            (output / f"validation_step_{step:06d}.json").write_text(json.dumps(metrics) + "\n")
            record = {"phase": "validation", "step": step, "epe": metrics["epe"],
                      "edge_epe": metrics["edge_epe"], "bad_1": metrics["bad_1"],
                      "elapsed_s": time.monotonic() - begun}
            with history.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if metrics["epe"] < best:
                best = metrics["epe"]
                save_checkpoint(best_file, base, head, optimizer, scheduler, scaler, step, manifest)
                payload = torch.load(best_file, map_location="cpu", weights_only=False)
                payload["validation_epe"] = best
                temp = best_file.with_suffix(".tmp")
                torch.save(payload, temp); temp.replace(best_file)
            save_checkpoint(latest, base, head, optimizer, scheduler, scaler, step, manifest)
    selected = torch.load(best_file, map_location="cpu", weights_only=False)
    base.load_state_dict(selected["base"]); head.load_state_dict(selected["head"])
    result = evaluate(base, head, test, args.data, output / "visuals")
    (output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"selected_step": selected["step"],
                                                     "best_validation_epe": best,
                                                     "test": {k: v for k, v in result.items()
                                                              if k != "pair_results"}}, indent=2) + "\n")
    print(json.dumps({"phase": "complete", "arm": args.arm, "selected_step": selected["step"],
                      "test_epe": result["epe"], "test_edge_epe": result["edge_epe"]}), flush=True)


if __name__ == "__main__":
    main()
