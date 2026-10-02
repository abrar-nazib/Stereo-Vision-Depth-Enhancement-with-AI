"""Final pass — shard-streaming trainer: FusionStereoLite V (frozen yolo26m-sem)
on the OFFICIAL SceneFlow split (sceneflow_split_v1: 35,454 train pairs =
FT3D-train + Monkaa + Driving; 4,370 FT3D test), streamed from the
sceneflow-shards volume with the stereolite ShardStream loader.

Training strategy starts from the A08/A09 one:
  frozen M encoder (eval mode, no grads) + SGNet veto + v2_hitnet propagation,
  native 384x640 co-located crops (the loader's native_crop mode — byte-level
  same window in L/R/D), multi-scale loss (1.0/0.5/0.3/0.2/0.1 + grad 0.5 +
  hinge 0.2), AdamW 1e-4, AMP fp16 + GradScaler, grad clip 1.0.
  Validation-EPE plateau LR decay begins from the step-1000 resume of the
  current A10 run; the original A09 Driving experiment held LR constant.

Differences from run_A07v2: data streams from shards (no 185 GB preload —
the official split cannot be preloaded), eval runs on the split's fixed
val_subset of FT3D-TEST at native 960x540 pad-16, and --probe sweeps physical
batch sizes through the REAL data pipeline (decode included).

Usage (probe):
    python3 train_shards.py --shards_dir /shards/v1 --out_dir /results/... \
        --probe 8,16,32
Usage (train):
    python3 train_shards.py --shards_dir /shards/v1 --out_dir /results/... \
        --steps 120000 --batch 16 --ckpt-every 1000 --eval-every 2000
"""

from __future__ import annotations

import argparse
import csv
import cv2
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

STEREOLITE_SCRIPTS = "/workspace/stereolite/scripts"
RUNNERS = "/workspace/experiments/a06_shallow/runners"
sys.path.insert(0, STEREOLITE_SCRIPTS)
sys.path.insert(0, RUNNERS)

from train_full_sceneflow import ShardStream, decode_record, load_test_pairs  # noqa: E402

from model_a07v2 import FusionStereoLite  # noqa: E402
from lr_schedule import make_scheduler, after_training_step, after_validation, description  # noqa: E402

ENCODER = "/workspace/models/segmentation/yolo26m-sem-ade20k.pt"


def native_pair(rec: dict, device: str):
    """Shard record -> (left, right, gt) float tensors on device, 0-255 RGB."""
    left = rec["L"].to(device, non_blocking=True).float()
    right = rec["R"].to(device, non_blocking=True).float()
    gt = rec["D"].to(device, non_blocking=True).float()
    return left, right, gt


def pad16(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    top, rp = (-h) % 16, (-w) % 16
    return F.pad(t, (0, rp, top, 0), mode="replicate")


@torch.no_grad()
def evaluate(model, pairs, device: str, bs: int = 4) -> dict:
    """Native-resolution pad-16 eval over valid pixels, without resizing."""
    model.eval()
    sums = {key: 0.0 for key in ("epe", "sqerr", "bad_0.5", "bad_1", "bad_2", "bad_3", "d1")}
    n = 0
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        left = torch.stack([p["L"] for p in chunk]).to(device).float()
        right = torch.stack([p["R"] for p in chunk]).to(device).float()
        top = (-left.shape[-2]) % 16
        lt = pad16(left)
        rt = pad16(right)
        gt = torch.stack([p["D"] for p in chunk]).to(device).float()
        with torch.autocast("cuda", dtype=torch.float16):
            disp = model(lt, rt, aux=False)
        disp = disp[..., top:top + gt.shape[-2], :gt.shape[-1]].float()
        valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
        err = (disp - gt).abs()[valid]
        n += int(valid.sum())
        sums["epe"] += err.sum().item()
        sums["sqerr"] += err.square().sum().item()
        for threshold in (0.5, 1, 2, 3):
            sums[f"bad_{threshold}"] += (err > threshold).sum().item()
        sums["d1"] += ((err > 3) & (err / gt[valid].clamp_min(1e-6) > 0.05)).sum().item()
    model.train()
    if n == 0:
        raise RuntimeError("validation has no valid disparity pixels")
    return {"epe": sums["epe"] / n, "rmse": (sums["sqerr"] / n) ** 0.5,
            **{key: 100 * sums[key] / n for key in ("bad_0.5", "bad_1", "bad_2", "bad_3", "d1")},
            "n": n}


@torch.no_grad()
def evaluate_full_test(model, shards_dir: Path, device: str) -> dict:
    """Score all FT3D test shards, keeping only one decoded shard in RAM."""
    totals = {key: 0.0 for key in ("epe", "sqerr", "bad_0.5", "bad_1", "bad_2", "bad_3", "d1")}
    total_pixels = 0
    total_pairs = 0
    for shard in sorted(shards_dir.glob("test_ft3d_*.pkl")):
        with shard.open("rb") as stream:
            records = pickle.load(stream)
        pairs = [decode_record(rec, native=True) for rec in records]
        del records
        metrics = evaluate(model, pairs, device)
        n = metrics["n"]
        total_pixels += n
        total_pairs += len(pairs)
        totals["epe"] += metrics["epe"] * n
        totals["sqerr"] += metrics["rmse"] ** 2 * n
        for key in ("bad_0.5", "bad_1", "bad_2", "bad_3", "d1"):
            totals[key] += metrics[key] * n / 100
        print(f"[full-test] {shard.name}: {len(pairs)} pairs, EPE={metrics['epe']:.4f}", flush=True)
        del pairs
    return {"pairs": total_pairs, "n": total_pixels,
            "epe": totals["epe"] / total_pixels,
            "rmse": (totals["sqerr"] / total_pixels) ** 0.5,
            **{key: 100 * totals[key] / total_pixels
               for key in ("bad_0.5", "bad_1", "bad_2", "bad_3", "d1")}}


@torch.no_grad()
def save_progress_images(model, pairs, out_dir: Path, step: int, device: str) -> None:
    """Fixed FT3D test scenes, raw disparity and consistent-scale color preview."""
    image_dir = out_dir / "visualizations"
    model.eval()
    for index in np.linspace(0, len(pairs) - 1, min(12, len(pairs)), dtype=int):
        pair = pairs[int(index)]
        scene_dir = image_dir / f"val_{index:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        if not (scene_dir / "left.png").exists():
            rgb = pair["L"].permute(1, 2, 0).numpy()
            cv2.imwrite(str(scene_dir / "left.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            gt = pair["D"][0].float().numpy()
            cv2.imwrite(str(scene_dir / "gt_u16.png"),
                        np.clip(gt * 256, 0, 65535).astype(np.uint16))
            (scene_dir / "scene.json").write_text(json.dumps(
                {"index": int(index), "sequence": pair["seq"], "frame": pair["t"],
                 "disparity_units": "uint16 / 256 = pixels",
                 "color_scale_px": [0, 192]}, indent=2) + "\n")
        left = pair["L"][None].to(device).float()
        right = pair["R"][None].to(device).float()
        top = (-left.shape[-2]) % 16
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(pad16(left), pad16(right), aux=False)
        disp = pred[0, 0, top:top + left.shape[-2], :left.shape[-1]].float().cpu().numpy()
        raw = np.clip(disp * 256, 0, 65535).astype(np.uint16)
        color = cv2.applyColorMap(np.clip(disp / 192 * 255, 0, 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        cv2.imwrite(str(scene_dir / f"step_{step:06d}_u16.png"), raw)
        cv2.imwrite(str(scene_dir / f"step_{step:06d}_color.png"), color)
    model.train()


def train_step(model, batch, device, opt, scaler):
    left, right, gt = native_pair(batch, device)
    valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(left, right, aux=True)
    d_full = out["d_final"]

    def ms_l1(pred, g, v, scale):
        if pred.shape[-2:] != g.shape[-2:]:
            pred = F.interpolate(pred, size=g.shape[-2:], mode="bilinear",
                                 align_corners=False) * scale
        return ((pred - g).abs() * v).sum() / v.sum().clamp(min=1)

    def gc(pred, g, v):
        gx_p = pred[..., :, 1:] - pred[..., :, :-1]
        gx_g = g[..., :, 1:] - g[..., :, :-1]
        gy_p = pred[..., 1:, :] - pred[..., :-1, :]
        gy_g = g[..., 1:, :] - g[..., :-1, :]
        vx = v[..., :, 1:] * v[..., :, :-1]
        vy = v[..., 1:, :] * v[..., :-1, :]
        return (((gx_p - gx_g).abs() * vx).sum() / vx.sum().clamp(min=1)
                + ((gy_p - gy_g).abs() * vy).sum() / vy.sum().clamp(min=1))

    err = (d_full - gt).abs()
    loss = (1.0 * ms_l1(d_full, gt, valid, 1.0)
            + 0.5 * ms_l1(out["d_half"], gt, valid, 2.0)
            + 0.3 * ms_l1(out["d4"], gt, valid, 4.0)
            + 0.2 * ms_l1(out["d8"], gt, valid, 8.0)
            + 0.1 * ms_l1(out["d16"], gt, valid, 16.0)
            + 0.5 * gc(d_full, gt, valid)
            + 0.2 * (((err - 1.0).clamp(min=0) ** 2) * valid).sum()
            / valid.sum().clamp(min=1))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    prior_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    optimizer_updated = scaler.get_scale() >= prior_scale
    return loss.item(), err[valid].mean().item(), optimizer_updated


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_probe_sweep(model, shards, device, args) -> None:
    """Through the REAL shard pipeline: ms/step, ms/sample, peak VRAM per batch."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"[probe] GPU={torch.cuda.get_device_name()} VRAM={total_memory/2**30:.2f} GiB "
          f"85%-budget={0.85*total_memory/2**30:.2f} GiB", flush=True)
    for batch in [int(b) for b in args.probe.split(",")]:
        loader = DataLoader(ShardStream(shards, args.seed, args.input_mode, 0),
                            batch_size=batch, num_workers=args.workers,
                            pin_memory=True, prefetch_factor=4, drop_last=True)
        it = iter(loader)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(10):
                if i == 3:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                batch_data = next(it)
                train_step(model, batch_data, device, opt, scaler)
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / 7
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            print(f"batch {batch:2d}: {per*1000:6.0f} ms/step = "
                  f"{per/batch*1000:5.1f} ms/sample | peak "
                  f"{peak/2**30:.2f} GiB allocated, {reserved/2**30:.2f} GiB reserved "
                  f"({100*reserved/total_memory:.1f}% VRAM) | "
                  f"{batch/per:5.1f} samples/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch {batch:2d}: OOM", flush=True)
        del loader, it, opt, scaler
        torch.cuda.empty_cache()


def main(commit=None, argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", default="/shards/v1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_schedule", choices=("plateau", "onecycle"), default="plateau")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--input_mode", default="native_crop")
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--val_subset", type=int, default=400)
    ap.add_argument("--split_json", default="/workspace/stereolite/configs/sceneflow_split_v1.json.gz")
    ap.add_argument("--probe", default=None, help='e.g. "8,16,32": sweep + exit')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    args = ap.parse_args(argv)

    import gzip

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    shards_dir = Path(args.shards_dir)
    shards = sorted(str(p) for p in shards_dir.glob("train_*.pkl"))
    if not shards or not list(shards_dir.glob("test_ft3d_*.pkl")):
        raise FileNotFoundError(f"missing train/test shards in {shards_dir}; expected /shards/v1")
    total_bytes = sum(p.stat().st_size for p in shards_dir.glob("train_*.pkl"))
    split = json.load(gzip.open(args.split_json))
    print(f"[data] train shards: {len(shards)} files, "
          f"{total_bytes / 2**30:.1f} GiB on volume; split counts: "
          f"{split['counts']}", flush=True)

    model = FusionStereoLite(arm="V", encoder=args.encoder).to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for p in model.fnet.parameters()):
        raise RuntimeError("YOLO encoder must be frozen")
    print(f"[model] out_channels {tuple(model.fnet.out_channels)} | "
          f"trainable {sum(p.numel() for p in trainable)/1e6:.3f} M | "
          f"encoder frozen: {not any(p.requires_grad for p in model.fnet.parameters())}",
          flush=True)

    if args.probe:
        run_probe_sweep(model, shards, device, args)
        print(f"[gpu] {torch.cuda.get_device_name()}", flush=True)
        # do NOT return: the probe continues into the short training leg so we
        # also measure real steps, eval, and checkpoint writing
        args.probe = None

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = make_scheduler(args.lr_schedule, opt, args.steps, args.lr)
    scaler = torch.amp.GradScaler("cuda")
    start_step = 0
    if args.resume and (out_dir / "checkpoints" / "latest.pth").exists():
        payload = torch.load(out_dir / "checkpoints" / "latest.pth",
                             map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_step = int(payload["step"])
        print(f"[resume] from step {start_step}", flush=True)

    subset_keys = set(split["val_subset"][:args.val_subset])
    val_pairs = load_test_pairs(shards_dir, subset_keys, native=True)
    print(f"[eval] {len(val_pairs)} held-out FT3D test pairs (subset of 4,370)",
          flush=True)

    loader = DataLoader(ShardStream(shards, args.seed, args.input_mode, 0),
                        batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4, drop_last=True)

    csv_path = out_dir / "train.csv"
    validation_path = out_dir / "validation.csv"
    lr_path = out_dir / "learning_rate.csv"
    config_path = out_dir / "config.json"
    config = {"architecture": "A09_V_yolo26m_ade20k", "encoder": args.encoder,
              "encoder_frozen": True, "data_split": args.split_json,
              "train_pairs": split["counts"]["train_total"],
              "test_pairs": split["counts"]["test_total"],
              "crop": [384, 640], "resize": False, "batch": args.batch,
              "steps": args.steps, "lr": args.lr, "seed": args.seed,
              "loss": "A09 five-scale L1 + 0.5 gradient + 0.2 bad-1 hinge",
              "amp": "fp16", "validation_pairs": len(val_pairs),
              "lr_schedule": description(args.lr_schedule, args.lr, args.steps)}
    if start_step > 0 and config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior.get("lr_schedule") != config["lr_schedule"]:
            raise RuntimeError("cannot resume checkpoint with a different LR schedule")
        prior["lr_schedule"] = config["lr_schedule"]
        prior["steps"] = args.steps
        prior["resume_step"] = start_step
        config = prior
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if start_step == 0:
        csv_path.write_text("step,loss,crop_epe,val_step,val_epe,lr,elapsed_seconds\n")
    else:
        with csv_path.open(newline="") as stream:
            old = list(csv.DictReader(stream))
        temporary = csv_path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "loss", "crop_epe",
                                                      "val_step", "val_epe", "lr", "elapsed_seconds"])
            writer.writeheader()
            for row in old:
                if int(row["step"]) <= start_step:
                    writer.writerow({**row, "val_step": row.get("val_step", ""),
                                     "val_epe": row.get("val_epe", ""),
                                     "lr": row.get("lr", args.lr)})
        os.replace(temporary, csv_path)
    if not validation_path.exists():
        validation_path.write_text("step,lr,epe,rmse,bad_0.5,bad_1,bad_2,bad_3,d1,n\n")
    if not lr_path.exists():
        lr_path.write_text("step,lr\n")
    elif start_step > 0:
        with lr_path.open(newline="") as stream:
            old_lrs = list(csv.DictReader(stream))
        temporary = lr_path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "lr"])
            writer.writeheader()
            writer.writerows(row for row in old_lrs if int(row["step"]) <= start_step)
        os.replace(temporary, lr_path)

    def _commit():
        if commit is not None:
            try:
                commit()
            except Exception as exc:  # noqa: BLE001 - never kill training for a commit hiccup
                print(f"[committer] {exc}", flush=True)

    import threading
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            _commit()

    threading.Thread(target=committer, daemon=True).start()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step = start_step
    best_epe = float("inf")
    latest_val_epe = None
    latest_val_step = None
    best_path = out_dir / "best_eval.json"
    if best_path.exists():
        best_epe = json.loads(best_path.read_text())["epe"]
    if start_step > 0:
        existing_eval = out_dir / "latest_eval.json"
        if existing_eval.exists():
            latest_val_epe = json.loads(existing_eval.read_text())["epe"]
            if validation_path.exists():
                with validation_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    latest_val_step = int(rows[-1]["step"])
        else:
            baseline = evaluate(model, val_pairs, device)
            after_validation(args.lr_schedule, scheduler, baseline["epe"])
            latest_val_epe, latest_val_step = baseline["epe"], start_step
            print(f"EVAL step={start_step} val_epe={baseline['epe']:.4f} "
                  f"(resume baseline)", flush=True)
            existing_eval.write_text(json.dumps(baseline, indent=1) + "\n")
            with validation_path.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=["step", "lr", *baseline.keys()]).writerow(
                    {"step": start_step, "lr": opt.param_groups[0]["lr"], **baseline})
            if baseline["epe"] < best_epe:
                best_epe = baseline["epe"]
                best_path.write_text(json.dumps({"step": start_step, **baseline}, indent=1) + "\n")
                atomic_save({"model": model.state_dict(), "step": start_step, "metrics": baseline},
                            out_dir / "checkpoints" / "best.pth")
            try:
                save_progress_images(model, val_pairs, out_dir, start_step, device)
            except Exception as exc:
                print(f"[visualizations] step={start_step}: {exc}", flush=True)
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "step": start_step}, out_dir / "checkpoints" / "latest.pth")
            _commit()
    it = iter(loader)
    model.train()
    try:
        while step < args.steps:
            step += 1
            batch = next(it)
            loss, crop_epe, optimizer_updated = train_step(model, batch, device, opt, scaler)
            after_training_step(args.lr_schedule, scheduler, optimizer_updated)
            if step % 50 == 0 or step == args.steps:
                el = time.perf_counter() - started
                rate = (step - start_step) / max(el, 1e-6)
                eta_h = (args.steps - step) * (1 / max(rate, 1e-6)) / 3600
                val_text = (f"{latest_val_epe:.4f}@{latest_val_step}"
                            if latest_val_epe is not None else "pending")
                with csv_path.open("a") as s:
                    s.write(f"{step},{loss:.4f},{crop_epe:.4f},"
                            f"{latest_val_step or ''},{latest_val_epe if latest_val_epe is not None else ''},"
                            f"{opt.param_groups[0]['lr']:.9g},{el:.1f}\n")
                with lr_path.open("a") as s:
                    s.write(f"{step},{opt.param_groups[0]['lr']:.9g}\n")
                print(f"step={step} loss={loss:.4f} crop_epe={crop_epe:.4f} "
                      f"val_epe={val_text} "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"rate={rate:.2f} it/s eta={eta_h:.1f}h", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val_pairs, device)
                after_validation(args.lr_schedule, scheduler, m["epe"])
                latest_val_epe, latest_val_step = m["epe"], step
                print(f"EVAL step={step} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in m.items())
                      + f" lr={opt.param_groups[0]['lr']:.2e}", flush=True)
                (out_dir / "latest_eval.json").write_text(json.dumps(m, indent=1) + "\n")
                with validation_path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["step", "lr", *m.keys()])
                    writer.writerow({"step": step, "lr": opt.param_groups[0]["lr"], **m})
                if m["epe"] < best_epe:
                    best_epe = m["epe"]
                    best_path.write_text(json.dumps({"step": step, **m}, indent=1) + "\n")
                    atomic_save({"model": model.state_dict(), "step": step, "metrics": m},
                                out_dir / "checkpoints" / "best.pth")
                try:
                    save_progress_images(model, val_pairs, out_dir, step, device)
                except Exception as exc:
                    print(f"[visualizations] step={step}: {exc}", flush=True)
            if step % args.ckpt_every == 0 or step == args.steps:
                payload = {"model": model.state_dict(),
                           "optimizer": opt.state_dict(),
                           "scheduler": scheduler.state_dict(),
                           "scaler": scaler.state_dict(),
                           "step": step}
                atomic_save(payload, out_dir / "checkpoints" / "latest.pth")
                atomic_save(payload, out_dir / "checkpoints" / f"step_{step:06d}.pth")
                _commit()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        stop.set()

    del val_pairs
    best_payload = torch.load(out_dir / "checkpoints" / "best.pth",
                              map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    full_metrics = evaluate_full_test(model, shards_dir, device)
    full_metrics["checkpoint_step"] = best_payload["step"]
    (out_dir / "full_test.json").write_text(json.dumps(full_metrics, indent=2) + "\n")
    print("FULL_TEST " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in full_metrics.items()), flush=True)
    commit()
    print(f"COMPLETE {out_dir} at step {step} "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
