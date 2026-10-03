"""Metric left-camera point generation from rectified disparity."""

import numpy as np

from experiments.vkitti2.semantic_data import CLASS_COLORS, CLASS_NAMES
from semtilestereo.core import InferenceResult
from semtilestereo.results import CameraParameters


VKITTI_CAMERA = CameraParameters(725.0087, 725.0087, 620.5, 187.0, 0.532725)


def points_from_result(result: InferenceResult, camera: CameraParameters, *,
                       visible_classes: set[int], min_depth_m: float, max_depth_m: float,
                       stride: int, semantic_colors: bool) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    camera.validate()
    if (not np.isfinite(min_depth_m) or not np.isfinite(max_depth_m)
            or min_depth_m <= 0 or max_depth_m <= min_depth_m or stride < 1):
        raise ValueError("depth range must be finite/positive and stride must be at least 1")
    if not visible_classes.issubset(set(range(len(CLASS_NAMES)))):
        raise ValueError("visible_classes contains an unknown class ID")
    disparity = result.disparity_px
    if disparity.ndim != 2 or result.class_id.shape != disparity.shape or result.left_rgb.shape != (*disparity.shape, 3):
        raise ValueError("result disparity, class and RGB dimensions do not match")
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.zeros(disparity.shape, dtype=np.float64)
    depth[valid] = camera.fx * camera.baseline_m / disparity[valid]
    valid &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    labels = result.class_id
    counts = {int(class_id): int(np.count_nonzero(valid & (labels == class_id)))
              for class_id in np.unique(labels) if np.count_nonzero(valid & (labels == class_id))}
    sample = np.zeros(disparity.shape, dtype=bool)
    sample[::stride, ::stride] = True
    mask = valid & sample & np.isin(labels, list(visible_classes))
    v, u = np.nonzero(mask)
    z = depth[mask]
    points = np.column_stack(((u - camera.cx) * z / camera.fx,
                              (v - camera.cy) * z / camera.fy, z))
    if semantic_colors:
        colors = np.asarray(CLASS_COLORS, dtype=np.float64)[labels[mask]] / 255.0
    else:
        colors = result.left_rgb[mask].astype(np.float64) / 255.0
    return points, colors, counts
