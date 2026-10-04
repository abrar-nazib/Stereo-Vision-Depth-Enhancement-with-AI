"""Native-pixel, single-frame diagnostic comparison; not an ablation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
SSD = Path("/media/abrar/AbrarSSD/ResearchArtifacts/SVDE/external_weights")
PAIR = ROOT / "paper/figures/semtilestereo/candidates"
LEFT = PAIR / "Scene18__clone__00259_left.jpg"
RIGHT = PAIR / "Scene18__clone__00259_right.jpg"
FOUNDATION_SOURCE = Path("/home/abrar/Research/stero_research_claude/external_models/FoundationStereo")
HITNET_SOURCE = Path("/home/abrar/Research/stero_research_claude/external_models/TinyHITNet")
NAMES = ("LightStereo-S", "LightStereo-M", "SemTileStereo-E3", "HITNet-XL", "FoundationStereo-ViT-S")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gt_disparity() -> np.ndarray:
    from experiments.vkitti2.data import depth_to_disparity

    depth = cv2.imread(str(HERE / "depth_00259.png"), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.shape != (375, 1242):
        raise ValueError("Expected the native 375x1242 VKITTI uint16 depth PNG")
    disparity, _ = depth_to_disparity(depth, 725.0087, 0.532725, 192)
    return disparity


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} != GT {gt.shape}")
    x = np.arange(gt.shape[1])[None, :]
    valid = np.isfinite(gt) & (gt > 0) & (gt < 192) & (x >= gt)
    valid &= np.isfinite(pred)
    err = np.abs(pred[valid].astype(np.float64) - gt[valid])
    truth = gt[valid]
    if not err.size:
        raise ValueError("No valid pixels")
    values: dict[str, float | int] = {
        "valid_pixels": int(err.size), "epe_px": float(err.mean()),
        "rmse_px": float(np.sqrt(np.mean(err * err))),
        "median_px": float(np.median(err)),
    }
    for threshold in (0.5, 1, 2, 3):
        values[f"bad_{threshold:g}_pct"] = float(np.mean(err > threshold) * 100)
    values["d1_pct"] = float(np.mean((err > 3) & (err > 0.05 * truth)) * 100)
    return values


def lightstereo(name: str, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Path]:
    sys.path.insert(0, str(ROOT / "experiments/yolo_las2_v1"))
    from lightstereo_adapter import build_lightstereo

    variant = name[-1]
    weight = SSD / "lightstereo" / f"LightStereo-{variant}-SceneFlow.ckpt"
    model = build_lightstereo(variant=variant)
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model_state", payload))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.cuda().eval()
    h, w = left.shape[:2]
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    def tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().cuda()
        t = F.pad(t, (0, (-w) % 32, (-h) % 32, 0), mode="replicate")
        return (t / 255 - mean) / std
    with torch.inference_mode():
        result = model({"left": tensor(left), "right": tensor(right)})["disp_pred"]
    return result[0, 0, (-h) % 32:, :w].float().cpu().numpy(), weight


def semtilestereo(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Path]:
    from semtilestereo.core import ModelPaths, infer_pair, load_model
    paths = ModelPaths()
    model = load_model(paths, "cuda")
    result = infer_pair(model, left, right, "cuda")
    return result.disparity_px, paths.head


def hitnet(name: str, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Path]:
    sys.path.insert(0, str(HITNET_SOURCE))
    from models.hit_net_sf import HITNetXL_SF, HITNet_SF

    xl = name == "HITNet-XL"
    weight = SSD / "hitnet" / ("hitnet_xl_sf_finalpass_from_tf.ckpt" if xl else "hitnet_sf_finalpass.ckpt")
    model = HITNetXL_SF() if xl else HITNet_SF()
    payload = torch.load(weight, map_location="cpu", weights_only=xl)
    state = payload if xl else {key.removeprefix("model."): value
                                for key, value in payload["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    def tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        disp = model(tensor(left), tensor(right))["disp"]
    return disp[0, 0].float().cpu().numpy(), weight


def foundation(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Path]:
    sys.path.insert(0, str(FOUNDATION_SOURCE))
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    weight = SSD / "foundationstereo_vits/model_best_bp2.pth"
    cfg = OmegaConf.load(weight.parent / "cfg.yaml")
    cfg.low_memory = 1
    cfg.mixed_precision = True
    model = FoundationStereo(cfg)
    payload = torch.load(weight, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.cuda().eval()
    def tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().cuda()
    left_t, right_t = tensor(left), tensor(right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pred = model(left_t, right_t, iters=int(cfg.valid_iters), test_mode=True, low_memory=True)
    pred = padder.unpad(pred.float())
    return pred[0, 0].cpu().numpy(), weight


def colorize(disparity: np.ndarray, max_px: float = 80) -> np.ndarray:
    scaled = np.clip(disparity / max_px, 0, 1)
    color = cv2.applyColorMap(np.uint8(np.round(scaled * 255)), cv2.COLORMAP_TURBO)
    color[~np.isfinite(disparity) | (disparity <= 0)] = 0
    return color


def montage(results: dict[str, dict], gt: np.ndarray) -> None:
    panels = []
    for name in ("Ground truth", *NAMES):
        if name != "Ground truth" and name not in results:
            continue
        disparity = gt if name == "Ground truth" else np.load(HERE / f"{name}.npy")
        image = colorize(disparity)
        banner = np.full((54, image.shape[1], 3), 255, np.uint8)
        label = name if name == "Ground truth" else f"{name}  EPE {results[name]['metrics']['epe_px']:.3f} px"
        cv2.putText(banner, label, (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        panels.append(np.vstack([banner, image]))
    if panels:
        cv2.imwrite(str(HERE / "side_by_side.png"), np.hstack(panels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*NAMES, "HITNet-SF", "montage"), required=True)
    args = parser.parse_args()
    left = cv2.imread(str(LEFT))
    right = cv2.imread(str(RIGHT))
    if left is None or right is None or left.shape != right.shape or left.shape[:2] != (375, 1242):
        raise ValueError("Missing or mismatched native Scene18 pair")
    gt = gt_disparity()
    report_file = HERE / "results.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    if args.model == "montage":
        montage(report, gt)
        return
    torch.cuda.empty_cache()
    start = time.perf_counter()
    func = {"LightStereo-S": lightstereo, "LightStereo-M": lightstereo,
            "SemTileStereo-E3": semtilestereo, "HITNet-SF": hitnet, "HITNet-XL": hitnet,
            "FoundationStereo-ViT-S": foundation}[args.model]
    pred, weight = func(args.model, left, right) if args.model.startswith(("LightStereo", "HITNet")) else func(left, right)
    elapsed = time.perf_counter() - start
    report[args.model] = {"metrics": metrics(pred, gt), "elapsed_s_including_load": elapsed,
                          "checkpoint": str(weight), "checkpoint_sha256": sha256(weight),
                          "pred_min_px": float(np.nanmin(pred)), "pred_max_px": float(np.nanmax(pred))}
    np.save(HERE / f"{args.model}.npy", pred.astype(np.float32))
    cv2.imwrite(str(HERE / f"{args.model}.png"), colorize(pred))
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    montage(report, gt)
    print(json.dumps({args.model: report[args.model]}, indent=2))


if __name__ == "__main__":
    main()
