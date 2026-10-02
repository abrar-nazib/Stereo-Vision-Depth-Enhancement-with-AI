"""Live dual-window demo: stereo depth + ADE20K segmentation, same encoder.

Window 1: left camera view
Window 2: colorized disparity from the A08 V-chassis checkpoint
          (frozen yolo26s-seg trunk + SGNet veto + v2_hitnet propagation)
Window 3: ADE20K segmentation from the FULL yolo26s-seg model that lives in
          the SAME checkpoint file — the original seg head running on the
          pretrained trunk. This is the "dual side" test: one pretrained
          encoder serving both a depth consumer and a segmentation consumer.

Usage:
    uv run python experiments/final_pass/live_dual.py --rot 180
    uv run python experiments/final_pass/live_dual.py --frames 1 --save smoke.png
Press q / ESC to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/a06_shallow/runners"))

from live_depth import (DEFAULT_CKPT, colorize, load_model,  # noqa: E402
                        open_camera, rotate, split_stereo)


def golden_palette(n: int) -> np.ndarray:
    """Deterministic distinct colors, BGR, one per class id."""
    colors = np.zeros((n, 3), np.uint8)
    for i in range(n):
        hue = int((i * 137.508) % 180)
        colors[i] = cv2.cvtColor(np.array([[[hue, 230, 220]]], np.uint8),
                                 cv2.COLOR_HSV2BGR)[0, 0]
    return colors


def seg_frame(seg_model, left_bgr: np.ndarray, palette: np.ndarray,
              imgsz: int) -> tuple[np.ndarray, list[str]]:
    """Decode the yolo26s '-sem' head: predict() returns a dense per-pixel
    class map at result.semantic_mask.data (this checkpoint is a semantic
    segmentation model, NOT an instance segmentation model)."""
    res = seg_model.predict(left_bgr, imgsz=imgsz, retina_masks=True, verbose=False)[0]
    sm = getattr(res, "semantic_mask", None)
    if sm is None:
        raise RuntimeError("checkpoint returned no semantic mask — is this a '-sem' model?")
    mask = sm.data
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape[:2] != left_bgr.shape[:2]:
        mask = cv2.resize(mask, left_bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    names = res.names or {}
    n_names = len(names)
    color = palette[np.clip(mask, 0, len(palette) - 1)]
    overlay = (0.45 * left_bgr + 0.55 * color).astype(np.uint8)

    ids, counts = np.unique(mask, return_counts=True)
    labels = []
    for i in np.argsort(counts)[::-1]:
        cid = int(ids[i])
        if 0 <= cid < n_names and counts[i] / mask.size > 0.02:
            labels.append(f"{names[cid]} {counts[i] / mask.size:.0%}")
    return overlay, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--video", default="/dev/video2")
    parser.add_argument("--mode", default="1280x480")
    parser.add_argument("--left-view", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--seg-imgsz", type=int, default=480,
                        help="inference size for the semantic head")
    parser.add_argument("--seg-every", type=int, default=1,
                        help="run segmentation every Nth frame")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ultralytics import YOLO

    model, arm = load_model(args.ckpt, args.device)
    seg_model = YOLO(str(REPO / "models/segmentation/yolo26s-sem-ade20k.pt"))
    palette = golden_palette(150)

    cap = open_camera(args.video, args.mode)
    print(f"arm={arm} ckpt={args.ckpt.name} video={args.video} mode={args.mode}")

    last = None
    frame_idx = 0
    t0 = time.perf_counter()
    seg_overlay, labels = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        left, right = split_stereo(frame, args.left_view)
        left, right = rotate(left, args.rot), rotate(right, args.rot)
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        lt, rt = left_rgb.transpose(2, 0, 1)[None].astype(np.float32), \
            right.transpose(2, 0, 1)[None].astype(np.float32)
        lt = torch.from_numpy(lt).cuda()
        rt = torch.from_numpy(rt).cuda()
        h, w = lt.shape[-2:]
        top, rp = (-h) % 16, (-w) % 16
        lt = torch.nn.functional.pad(lt, (0, rp, top, 0), mode="replicate")
        rt = torch.nn.functional.pad(rt, (0, rp, top, 0), mode="replicate")

        with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
            disp = model(lt, rt, aux=False)
        disp = disp[0, 0].float().cpu().numpy()

        if frame_idx % args.seg_every == 0:
            seg_overlay, labels = seg_frame(seg_model, left, palette, args.seg_imgsz)

        depth_img = colorize(disp)
        if frame_idx == 0:
            t0 = time.perf_counter()
        fps = frame_idx / max(time.perf_counter() - t0, 1e-6)
        cv2.putText(depth_img, f"{arm} | {fps:.1f} fps", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        seg_view = seg_overlay if seg_overlay is not None else left.copy()
        if labels:
            cv2.putText(seg_view, " | ".join(labels[:5])[:110], (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(seg_view, " | ".join(labels[:5])[:110], (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("left view", left)
        cv2.imshow("depth", depth_img)
        cv2.imshow("segmentation (ADE20K)", seg_view)
        last = (left, depth_img, seg_view)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        if args.frames and frame_idx >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    if args.save and last is not None:
        cv2.imwrite(str(args.save), np.hstack([last[0], last[1], last[2]]))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
