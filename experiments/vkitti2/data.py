"""Geometry and label conversions for the small VKITTI2 feasibility split."""

from __future__ import annotations

import numpy as np


# IDs are the zero-based ADE20K labels in yolo26m-sem-ade20k.pt.
VKITTI_RGB_TO_ADE = {
    (140, 140, 140): 1,   # building
    (90, 200, 255): 2,    # sky
    (0, 199, 0): 4,       # tree
    (100, 60, 100): 6,    # road
    (255, 127, 80): 20,   # car
    (160, 60, 60): 83,    # truck
    (255, 130, 0): 93,    # pole
    (0, 139, 139): 102,   # van
    (200, 200, 0): 136,   # traffic light
}


def depth_to_disparity(
    depth_cm: np.ndarray, fx: float, baseline_m: float, max_disp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert VKITTI's uint16 depth (centimetres) to pixel disparity."""
    depth = depth_cm.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = (100.0 * fx * baseline_m) / depth
    valid = (depth > 0) & (depth < 65535) & np.isfinite(disparity) & (disparity < max_disp)
    return np.where(valid, disparity, 0.0).astype(np.float32), valid


def map_vkitti_to_ade(rgb: np.ndarray) -> np.ndarray:
    """Map unambiguous VKITTI colors; all other labels are ignored (255)."""
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in VKITTI_RGB_TO_ADE.items():
        result[np.all(rgb == color, axis=-1)] = class_id
    return result


def paired_crop(left, right, disparity, labels, top, x, height, width):
    """Crop co-located pixels without rescaling either image or disparity."""
    if top < 0 or x < 0 or height <= 0 or width <= 0:
        raise ValueError("invalid crop coordinates")
    if top + height > left.shape[0] or x + width > left.shape[1]:
        raise ValueError("crop exceeds image bounds")
    sl = np.s_[top : top + height, x : x + width]
    return left[sl], right[sl], disparity[sl], labels[sl]
