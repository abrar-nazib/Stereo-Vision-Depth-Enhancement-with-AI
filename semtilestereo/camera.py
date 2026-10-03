"""Side-by-side stereo capture helpers and optional OpenCV rectification."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def split_stereo(frame: np.ndarray, left_view: str = "left") -> tuple[np.ndarray, np.ndarray]:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[1] % 2:
        raise ValueError("stereo frame must have three channels and even width")
    if left_view not in ("left", "right"):
        raise ValueError("left_view must be 'left' or 'right'")
    width = frame.shape[1] // 2
    first, second = frame[:, :width].copy(), frame[:, width:].copy()
    return (first, second) if left_view == "left" else (second, first)


@dataclass(frozen=True)
class RectificationMaps:
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray

    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (cv2.remap(left, self.left_x, self.left_y, cv2.INTER_LINEAR),
                cv2.remap(right, self.right_x, self.right_y, cv2.INTER_LINEAR))


def load_rectification(path: Path, image_shape: tuple[int, int]) -> RectificationMaps:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"could not open rectification XML: {path}")
    try:
        maps = [storage.getNode(key).mat() for key in
                ("stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y")]
    finally:
        storage.release()
    if any(value is None or value.shape != tuple(image_shape) or value.dtype != np.float32 for value in maps):
        raise ValueError(f"rectification map shape/type does not match image {image_shape}; expected float32 maps")
    return RectificationMaps(*maps)
