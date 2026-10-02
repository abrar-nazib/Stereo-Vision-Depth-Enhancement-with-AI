"""Live stereo-depth inference with the A08 V-chassis checkpoint.

Grabs the rig's side-by-side frame (1280x480 = two 640x480 views), splits it
into left/right, and runs the trained FusionStereoLite V arm on every frame.
One window shows the left view, one the colorized disparity.

The encoder trunk is fully convolutional and the eval protocol is native
resolution with /16 replicate padding, so the 640x480 rig frames work without
any resize (480 and 640 are both divisible by 16).

Usage:
    uv run python experiments/final_pass/live_depth.py                 # GUI
    uv run python experiments/final_pass/live_depth.py --frames 1 --save smoke.png
    uv run python experiments/final_pass/live_depth.py --video /dev/video2 \
        --mode 1280x480 --left-view 0 --rot 0
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
sys.path.insert(0, str(REPO / "experiments/a06_shallow/runners"))

from model_a07v2 import FusionStereoLite  # noqa: E402
from run_A07v2 import pad_top_right, to_tensor  # noqa: E402

DEFAULT_CKPT = REPO / "experiments/a06_shallow/runs/A07v2V_20260928T154934Z/final.pth"


def load_model(ckpt_path: Path, device: str) -> tuple[FusionStereoLite, str]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    arm = payload["config"]["arm"]
    model = FusionStereoLite(arm=arm).to(device).eval()
    model.load_state_dict(payload["model"])
    return model, arm


def colorize(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    if not valid.any():
        return np.zeros((*disparity.shape, 3), np.uint8)
    lo, hi = np.percentile(disparity[valid], [2, 98])
    span = max(hi - lo, 1e-6)
    norm = np.clip((disparity - lo) / span, 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def rotate(img: np.ndarray, rot: int) -> np.ndarray:
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def open_camera(video: str, mode: str):
    width, height = (int(v) for v in mode.split("x"))
    cap = cv2.VideoCapture(video, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    return cap


def split_stereo(frame: np.ndarray, left_view: int) -> tuple[np.ndarray, np.ndarray]:
    half = frame.shape[1] // 2
    a, b = frame[:, :half], frame[:, half:]
    return (a, b) if left_view == 0 else (b, a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--video", default="/dev/video2")
    parser.add_argument("--mode", default="1280x480", help="rig side-by-side mode")
    parser.add_argument("--left-view", type=int, default=0, choices=[0, 1],
                        help="which half of the frame is the left camera")
    parser.add_argument("--rot", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--frames", type=int, default=0, help="0 = run until 'q'")
    parser.add_argument("--save", type=Path, default=None, help="save the last frame composite")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, arm = load_model(args.ckpt, args.device)
    cap = open_camera(args.video, args.mode)
    print(f"arm={arm} ckpt={args.ckpt.name} video={args.video} mode={args.mode}")

    last = None
    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        left, right = split_stereo(frame, args.left_view)
        left, right = rotate(left, args.rot), rotate(right, args.rot)
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)   # trunk was trained on RGB
        lt, rt = to_tensor(left_rgb), to_tensor(right)
        lt, rt, _, _ = pad_top_right(lt, rt)

        with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
            disp = model(lt, rt, aux=False)          # [1, 1, H, W]
        disp = disp[0, 0].float().cpu().numpy()

        depth_img = colorize(disp)
        left_bgr = left  # already native BGR for display
        if frame_idx == 0:
            t0 = time.perf_counter()  # skip warmup in the FPS average
        fps = frame_idx / max(time.perf_counter() - t0, 1e-6)
        cv2.putText(depth_img, f"{arm} | {fps:.1f} fps | disp {np.nanmin(disp):.1f}"
                    f"..{np.nanmax(disp):.1f} px", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("left view", left_bgr)
        cv2.imshow("depth", depth_img)
        last = (left_bgr, depth_img, disp)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        if args.frames and frame_idx >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    if args.save and last is not None:
        left_bgr, depth_img, _ = last
        cv2.imwrite(str(args.save), np.hstack([left_bgr, depth_img]))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
