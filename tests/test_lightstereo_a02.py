"""Tests for the LightStereo-S A02 benchmark manifest."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.lightstereo_s_a02.run import build_manifest
from experiments.lightstereo_s_a02.model import coarse_upsample, native_crop


class ManifestTests(unittest.TestCase):
    def test_build_manifest_selects_requested_unique_pairs_evenly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for sequence in range(2):
                left_dir = root / "frames_finalpass" / "a" / "b" / str(sequence) / "left"
                right_dir = left_dir.parent / "right"
                disparity_dir = root / "disparity" / "a" / "b" / str(sequence) / "left"
                left_dir.mkdir(parents=True)
                right_dir.mkdir(parents=True)
                disparity_dir.mkdir(parents=True)
                for index in range(10):
                    name = f"{index:04d}.png"
                    (left_dir / name).touch()
                    (right_dir / name).touch()
                    (disparity_dir / f"{index:04d}.pfm").touch()

            manifest = build_manifest(root, count=12)

        self.assertEqual(len(manifest), 12)
        self.assertEqual(len({record["left"] for record in manifest}), 12)
        self.assertEqual({Path(record["right"]).name for record in manifest},
                         {Path(record["left"]).name for record in manifest})


class NativeCropTests(unittest.TestCase):
    def test_native_crop_preserves_co_located_pixels_without_resizing(self) -> None:
        left = np.arange(5 * 8 * 3, dtype=np.uint8).reshape(5, 8, 3)
        right = left + 1
        disparity = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)

        cropped_left, cropped_right, cropped_disparity = native_crop(
            left, right, disparity, height=3, width=4, top=1, left_offset=2
        )

        np.testing.assert_array_equal(cropped_left, left[1:4, 2:6])
        np.testing.assert_array_equal(cropped_right, right[1:4, 2:6])
        np.testing.assert_array_equal(cropped_disparity, disparity[1:4, 2:6])


class AuxiliaryDisparityTests(unittest.TestCase):
    def test_coarse_upsample_keeps_a_single_disparity_channel(self) -> None:
        import torch

        coarse = coarse_upsample(torch.zeros(1, 1, 4, 8), (16, 32))

        self.assertEqual(coarse.shape, (1, 1, 16, 32))


if __name__ == "__main__":
    unittest.main()
