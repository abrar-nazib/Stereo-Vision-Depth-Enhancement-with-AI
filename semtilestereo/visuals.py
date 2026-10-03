"""Canonical disparity and semantic preview rendering (BGR for OpenCV)."""

import cv2
import numpy as np

from experiments.vkitti2.semantic_data import CLASS_COLORS


def colorize_disparity(disparity_px: np.ndarray, display_max: float) -> np.ndarray:
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError("display_max must be a positive finite disparity in pixels")
    if disparity_px.ndim != 2:
        raise ValueError("disparity must be a 2D array")
    valid = np.isfinite(disparity_px) & (disparity_px > 0)
    scaled = np.clip(np.where(valid, disparity_px, 0) / display_max * 255, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def segmentation_overlay(left_rgb: np.ndarray, class_id: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if (left_rgb.dtype != np.uint8 or left_rgb.ndim != 3 or left_rgb.shape[2] != 3
            or class_id.shape != left_rgb.shape[:2] or class_id.dtype != np.uint8
            or np.any(class_id >= len(CLASS_COLORS))):
        raise ValueError("left_rgb and class_id must be matching RGB/14-class uint8 arrays")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    rgb = cv2.addWeighted(left_rgb, 1 - alpha, palette[class_id], alpha, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
