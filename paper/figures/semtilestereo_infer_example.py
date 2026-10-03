"""Render genuine E3/F2 output thumbnails for the architecture figure.

The A09 stereo checkpoint is downloaded separately from the existing Modal
results volume; this script does not train or download any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from experiments.E.E_1_wide2_semantics.model import EModel


ROOT = Path(__file__).resolve().parents[2]
HEAD = ROOT / "experiments/E/E_3_wide4_semantics/runs/e3_e4_d2_full_t4_v1_20261002/checkpoints/best.pth"
SEMANTIC = ROOT / "models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt"
ENCODER = ROOT / "models/segmentation/yolo26m-sem-ade20k.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo", type=Path, required=True, help="A09 full-SceneFlow best.pth")
    parser.add_argument("--left", type=Path, required=True, help="VKITTI2 training Camera_0 RGB")
    parser.add_argument("--right", type=Path, required=True, help="matching Camera_1 RGB")
    parser.add_argument("--stem", required=True, help="output basename; scene/variation/frame")
    parser.add_argument("--display-max", type=float, default=80.0,
                        help="fixed disparity color range in pixels (default: 80)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures/semtilestereo")
    args = parser.parse_args()
    left_bgr = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None or left_bgr.shape != right_bgr.shape:
        raise FileNotFoundError("matching VKITTI2 left/right input images are required")
    if not args.stereo.is_file():
        raise FileNotFoundError(args.stereo)
    if args.display_max <= 0:
        raise ValueError("--display-max must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EModel(args.stereo, SEMANTIC, ENCODER, gate_hidden=32,
                   residual_hidden=128, use_semantics=True)
    payload = torch.load(HEAD, map_location="cpu", weights_only=False)
    model.load_trainable_state(payload["trainable"])
    model = model.to(device).eval()

    height, width = left_bgr.shape[:2]
    pad_h, pad_w = (-height) % 32, (-width) % 32

    def tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        value = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        return F.pad(value.float(), (0, pad_w, 0, pad_h), mode="replicate").to(device)

    with torch.inference_mode():
        disparity, logits = model(tensor(left_bgr), tensor(right_bgr))
        logits = F.interpolate(logits, size=disparity.shape[-2:],
                               mode="bilinear", align_corners=False)
    disp = disparity[0, 0, :height, :width].cpu().numpy()
    classes = logits[0, :, :height, :width].argmax(0).byte().cpu().numpy()
    if not np.isfinite(disp).all():
        raise ValueError("non-finite disparity in example inference")

    # Fixed display range makes the color image reproducible across reruns.
    disparity_u8 = np.clip(disp / args.display_max * 255.0, 0, 255).astype(np.uint8)
    disparity_bgr = cv2.applyColorMap(disparity_u8, cv2.COLORMAP_TURBO)
    palette_rgb = np.asarray([
        (70, 70, 70), (190, 153, 153), (250, 170, 160), (220, 20, 60),
        (153, 153, 153), (157, 234, 50), (128, 64, 128), (244, 35, 232),
        (107, 142, 35), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (119, 11, 32),
    ], dtype=np.uint8)
    semantic_bgr = cv2.cvtColor(palette_rgb[classes], cv2.COLOR_RGB2BGR)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir / f"disparity_{args.stem}_e3.png"), disparity_bgr):
        raise OSError("could not save disparity thumbnail")
    if not cv2.imwrite(str(args.output_dir / f"semantic_{args.stem}_e3.png"), semantic_bgr):
        raise OSError("could not save semantic thumbnail")
    print(f"device={device} shape={width}x{height} disparity_px=[{disp.min():.3f}, {disp.max():.3f}] display_max={args.display_max}")
    print(f"classes={np.unique(classes).tolist()} checkpoint_step={payload['step']}")


if __name__ == "__main__":
    main()
