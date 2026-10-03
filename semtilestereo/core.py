"""Native-resolution inference for the released E3 semantic stereo model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import cv2
import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
A09_SHA256 = "5ceaad4c84d941bd23be0d4c045f6d94934f6c17d0e6f5a9bc4cbc18f7f226b7"


@dataclass(frozen=True)
class ModelPaths:
    stereo: Path = ROOT / "models/stereo/semtilestereo/a09m_sceneflow_best.pth"
    head: Path = ROOT / "experiments/E/E_3_wide4_semantics/runs/e3_e4_d2_full_t4_v1_20261002/checkpoints/best.pth"
    semantic: Path = ROOT / "models/segmentation/yolo26m-sem-vkitti2-14class-freeze7-best.pt"
    encoder: Path = ROOT / "models/segmentation/yolo26m-sem-ade20k.pt"


@dataclass(frozen=True)
class InferenceResult:
    disparity_px: np.ndarray
    semantic_logits: np.ndarray
    class_id: np.ndarray
    class_confidence: np.ndarray
    left_rgb: np.ndarray


def load_model(paths: ModelPaths, device: str):
    """Load the frozen shared predictors and trained E3 fusion head."""
    for name in ("stereo", "head", "semantic", "encoder"):
        path = Path(getattr(paths, name))
        if not path.is_file():
            hint = " Retrieve the A09 checkpoint from the svde-results Modal volume." if name == "stereo" else ""
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}.{hint}")
    if Path(paths.stereo) == ModelPaths.stereo:
        with Path(paths.stereo).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != A09_SHA256:
            raise ValueError(f"A09 checkpoint hash mismatch at {paths.stereo}: {digest}")
    from experiments.E.E_1_wide2_semantics.model import EModel

    model = EModel(paths.stereo, paths.semantic, paths.encoder,
                   gate_hidden=32, residual_hidden=128, use_semantics=True)
    payload = torch.load(paths.head, map_location="cpu", weights_only=False)
    if "trainable" not in payload:
        raise KeyError(f"Missing 'trainable' state in fusion checkpoint: {paths.head}")
    model.load_trainable_state(payload["trainable"])
    return model.to(torch.device(device)).eval()


def infer_pair(model, left_bgr: np.ndarray, right_bgr: np.ndarray, device: str) -> InferenceResult:
    """Infer one rectified pair without resizing; pad only the right/bottom edges."""
    if (not isinstance(left_bgr, np.ndarray) or not isinstance(right_bgr, np.ndarray)
            or left_bgr.ndim != 3 or left_bgr.shape[2] != 3
            or right_bgr.shape != left_bgr.shape or left_bgr.dtype != np.uint8
            or right_bgr.dtype != np.uint8 or not all(left_bgr.shape[:2])):
        raise ValueError("left/right must be matching nonempty uint8 BGR images with three channels")
    height, width = left_bgr.shape[:2]
    pad_h, pad_w = (-height) % 32, (-width) % 32
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    def as_tensor(image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate").to(device)

    with torch.inference_mode():
        disparity, logits = model(as_tensor(left_bgr), as_tensor(right_bgr))
        if disparity.ndim != 4 or disparity.shape[1] != 1 or logits.ndim != 4:
            raise ValueError("model returned invalid disparity/logit tensor shapes")
        full_logits = F.interpolate(logits.float(), size=disparity.shape[-2:],
                                    mode="bilinear", align_corners=False)
        full_logits = full_logits[0, :, :height, :width]
        probabilities = full_logits.softmax(dim=0)
        confidence, classes = probabilities.max(dim=0)
    disparity_px = disparity[0, 0, :height, :width].float().cpu().numpy().copy()
    return InferenceResult(
        disparity_px=disparity_px,
        semantic_logits=logits[0].float().cpu().numpy().copy(),
        class_id=classes.byte().cpu().numpy().copy(),
        class_confidence=confidence.float().cpu().numpy().copy(),
        left_rgb=left_rgb,
    )
