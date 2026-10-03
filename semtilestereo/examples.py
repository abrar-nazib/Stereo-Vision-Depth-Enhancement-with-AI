"""Regenerate three ignored VKITTI example bundles with released E3 weights."""

from pathlib import Path

import cv2
import torch

from semtilestereo.core import ModelPaths, infer_pair, load_model
from semtilestereo.geometry import VKITTI_CAMERA
from semtilestereo.results import save_result


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "paper/figures/semtilestereo/candidates"
DESTINATION = ROOT / "examples/semtilestereo"
STEMS = ("Scene01__clone__00246", "Scene06__clone__00019", "Scene18__clone__00259")


def make_examples():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ModelPaths(), device)
    for stem in STEMS:
        left_path, right_path = SOURCES / f"{stem}_left.jpg", SOURCES / f"{stem}_right.jpg"
        left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        if left is None or right is None or left.shape != right.shape:
            raise FileNotFoundError(f"matching VKITTI pair is missing: {stem}")
        if left.shape[:2] != (375, 1242):
            raise ValueError(f"{stem}: unexpected VKITTI image shape {left.shape}")
        result = infer_pair(model, left, right, device)
        output = save_result(result, DESTINATION / stem.split("__")[0], camera=VKITTI_CAMERA,
                             metadata={"example_vkitti": True, "source_stem": stem,
                                       "left_source": str(left_path), "right_source": str(right_path)},
                             display_max=80)
        print(f"{stem}: {output} disparity [{result.disparity_px.min():.3f}, {result.disparity_px.max():.3f}] px")


if __name__ == "__main__":
    make_examples()
