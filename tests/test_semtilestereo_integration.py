"""Local checkpoint/asset acceptance; run explicitly on a CUDA workstation."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from semtilestereo.core import A09_SHA256, ModelPaths, infer_pair, load_model
from semtilestereo.results import load_bundle


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "paper/figures/semtilestereo/candidates"
EXAMPLES = ROOT / "examples/semtilestereo"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_checkpoint_inventory_and_sources():
    paths = ModelPaths()
    for path in (paths.stereo, paths.head, paths.semantic, paths.encoder):
        assert path.is_file(), path
    assert sha(paths.stereo) == A09_SHA256
    assert sha(paths.head) == "2493d46d17eceb51ab133b73f9b8fd54fb4ccc1b9b2d87a9a05810f15b727b87"
    for stem in ("Scene01__clone__00246", "Scene06__clone__00019", "Scene18__clone__00259"):
        assert (SOURCES / f"{stem}_left.jpg").is_file()
        assert (SOURCES / f"{stem}_right.jpg").is_file()


def test_three_saved_examples():
    for scene in ("Scene01", "Scene06", "Scene18"):
        path = EXAMPLES / scene / "result.npz"
        result, metadata = load_bundle(path)
        assert result.disparity_px.shape == (375, 1242)
        assert result.class_id.shape == (375, 1242)
        assert result.semantic_logits.shape[0] == 14
        assert metadata["camera"]["baseline_m"] == 0.532725
        for name in ("disparity.npy", "semantic_logits.npy", "disparity_color.png", "segmentation_overlay.png"):
            assert (path.parent / name).is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real checkpoint parity needs CUDA")
def test_scene01_saved_matches_direct_inference():
    model = load_model(ModelPaths(), "cuda")
    left = cv2.imread(str(SOURCES / "Scene01__clone__00246_left.jpg"))
    right = cv2.imread(str(SOURCES / "Scene01__clone__00246_right.jpg"))
    direct = infer_pair(model, left, right, "cuda")
    saved, _ = load_bundle(EXAMPLES / "Scene01/result.npz")
    # CUDA stereo aggregation may differ slightly across processes; saved arrays
    # themselves are exact float32, but a fresh forward is checked numerically.
    np.testing.assert_allclose(saved.disparity_px, direct.disparity_px, atol=0.05, rtol=0)
    np.testing.assert_array_equal(saved.semantic_logits, direct.semantic_logits)
    np.testing.assert_array_equal(saved.class_id, direct.class_id)


def test_missing_stereo_has_retrieval_hint(tmp_path):
    paths = ModelPaths(stereo=tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="Modal volume"):
        load_model(paths, "cpu")
