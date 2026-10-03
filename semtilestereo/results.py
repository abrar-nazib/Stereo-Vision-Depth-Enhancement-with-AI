"""Portable, versioned numeric result bundles and preview artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from experiments.vkitti2.semantic_data import CLASS_NAMES
from semtilestereo.core import InferenceResult
from semtilestereo.visuals import colorize_disparity, segmentation_overlay


SCHEMA_VERSION = 1
ARRAY_KEYS = ("disparity_px", "semantic_logits", "class_id", "class_confidence", "left_rgb")


@dataclass(frozen=True)
class CameraParameters:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float

    def validate(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("camera parameters must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.baseline_m <= 0:
            raise ValueError("fx, fy and baseline_m must be positive")


def _validate_result(result: InferenceResult) -> None:
    disparity, logits, labels, confidence, image = (
        result.disparity_px, result.semantic_logits, result.class_id,
        result.class_confidence, result.left_rgb,
    )
    if disparity.ndim != 2 or disparity.dtype != np.float32:
        raise ValueError("disparity_px must be a float32 H×W array")
    h, w = disparity.shape
    if h == 0 or w == 0 or logits.ndim != 3 or logits.shape[0] != len(CLASS_NAMES) or logits.dtype != np.float32:
        raise ValueError("semantic_logits must be a float32 14×h×w array")
    if labels.shape != (h, w) or labels.dtype != np.uint8 or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError("class_id must be a uint8 H×W array with IDs 0–13")
    if confidence.shape != (h, w) or confidence.dtype != np.float32 or not np.isfinite(confidence).all():
        raise ValueError("class_confidence must be a finite float32 H×W array")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("class_confidence must be within [0,1]")
    if image.shape != (h, w, 3) or image.dtype != np.uint8:
        raise ValueError("left_rgb must be a uint8 H×W×3 array")


def save_result(result: InferenceResult, out_dir: Path, *, camera: CameraParameters | None,
                metadata: dict, display_max: float) -> Path:
    _validate_result(result)
    if camera is not None:
        camera.validate()
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError("display_max must be positive and finite")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    h, w = result.disparity_px.shape
    document = dict(metadata)
    document.update(schema_version=SCHEMA_VERSION, image_size=[w, h],
                    class_names=list(CLASS_NAMES), display_max=float(display_max),
                    camera=asdict(camera) if camera is not None else None)
    metadata_json = json.dumps(document, allow_nan=False, sort_keys=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "disparity.npy", result.disparity_px, allow_pickle=False)
    np.save(out_dir / "semantic_logits.npy", result.semantic_logits, allow_pickle=False)
    if not cv2.imwrite(str(out_dir / "disparity_color.png"), colorize_disparity(result.disparity_px, display_max)):
        raise OSError("could not write disparity_color.png")
    if not cv2.imwrite(str(out_dir / "segmentation_overlay.png"), segmentation_overlay(result.left_rgb, result.class_id)):
        raise OSError("could not write segmentation_overlay.png")
    path = out_dir / "result.npz"
    np.savez_compressed(path, **{key: getattr(result, key) for key in ARRAY_KEYS},
                        metadata_json=np.array(metadata_json))
    return path


def load_bundle(path: Path) -> tuple[InferenceResult, dict]:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = set((*ARRAY_KEYS, "metadata_json")) - set(archive.files)
        if missing:
            raise ValueError(f"bundle missing keys: {sorted(missing)}")
        try:
            document = json.loads(str(archive["metadata_json"].item()))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("invalid bundle metadata JSON") from exc
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema: {document.get('schema_version')}")
        result = InferenceResult(**{key: archive[key].copy() for key in ARRAY_KEYS})
    _validate_result(result)
    h, w = result.disparity_px.shape
    if document.get("image_size") != [w, h] or document.get("class_names") != list(CLASS_NAMES):
        raise ValueError("bundle metadata does not match numeric arrays or class taxonomy")
    if document.get("camera") is not None:
        CameraParameters(**document["camera"]).validate()
    return result, document
