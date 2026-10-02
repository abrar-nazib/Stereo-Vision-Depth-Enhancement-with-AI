"""Contracts for the local Virtual KITTI 2 joint-refinement ablation."""

import importlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def subject(name):
    try:
        return importlib.import_module(f"experiments.vkitti2.{name}")
    except ModuleNotFoundError as exc:
        raise AssertionError(f"missing vkitti2 {name} implementation") from exc


class GeometryAndLabelsTests(unittest.TestCase):
    def test_depth_cm_becomes_pixel_disparity_with_invalid_values_masked(self):
        data = subject("data")
        depth = np.array([[1000, 0, 65535, 10]], dtype=np.uint16)

        disp, valid = data.depth_to_disparity(depth, fx=100.0, baseline_m=0.5, max_disp=192)

        np.testing.assert_array_equal(disp, [[5.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(valid, [[True, False, False, False]])

    def test_only_exact_vkitti_colors_map_to_ade_ids(self):
        data = subject("data")
        rgb = np.array([[(90, 200, 255), (255, 127, 80), (210, 0, 200), (1, 2, 3)]],
                       dtype=np.uint8)

        labels = data.map_vkitti_to_ade(rgb)

        np.testing.assert_array_equal(labels, [[2, 20, 255, 255]])

    def test_paired_crop_preserves_alignment_and_disparity_units(self):
        data = subject("data")
        grid = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
        left = np.repeat(grid[..., None], 3, axis=2)
        right = left + 100
        labels = grid.astype(np.uint8)

        lc, rc, dc, sc = data.paired_crop(left, right, grid, labels,
                                          top=1, x=2, height=3, width=4)

        self.assertEqual(lc.shape, (3, 4, 3))
        np.testing.assert_array_equal(lc[..., 0], grid[1:4, 2:6])
        np.testing.assert_array_equal(rc[..., 0], grid[1:4, 2:6] + 100)
        np.testing.assert_array_equal(dc, grid[1:4, 2:6])
        np.testing.assert_array_equal(sc, labels[1:4, 2:6])


class RefinerTests(unittest.TestCase):
    def test_zero_initialized_joint_head_preserves_both_frozen_outputs(self):
        model = subject("model")
        refiner = model.JointResidual(channels=16)
        disparity = torch.ones(1, 1, 32, 64) * 10
        logits = torch.randn(1, 150, 4, 8)
        image = torch.zeros(1, 3, 32, 64)

        d_new, s_new = refiner(disparity, logits, image)

        torch.testing.assert_close(d_new, disparity)
        torch.testing.assert_close(s_new, logits)

    def test_joint_head_changes_only_mapped_semantic_channels(self):
        model = subject("model")
        logits = torch.zeros(1, 150, 2, 2)
        delta = torch.ones(1, 9, 2, 2)

        changed = model.apply_semantic_residual(logits, delta)

        self.assertEqual(int((changed != 0).sum()), 9 * 2 * 2)
        for class_id in (1, 2, 4, 6, 20, 83, 93, 102, 136):
            torch.testing.assert_close(changed[:, class_id], torch.ones(1, 2, 2))

    def test_semantic_residual_accepts_mixed_precision_inputs(self):
        model = subject("model")
        logits = torch.zeros(1, 150, 2, 2, dtype=torch.float16)
        delta = torch.ones(1, 9, 2, 2, dtype=torch.float32)
        changed = model.apply_semantic_residual(logits, delta)
        self.assertEqual(changed.dtype, torch.float16)
        torch.testing.assert_close(changed[:, 1], torch.ones(1, 2, 2, dtype=torch.float16))


class SubsetTests(unittest.TestCase):
    def test_evenly_spaced_frames_include_sequence_endpoints(self):
        subset = subject("subset")
        self.assertEqual(subset.evenly_spaced_frames(91, 10),
                         [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    @unittest.skipUnless(Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/from_modal/"
                              "vkitti_2.0.3_textgt.tar.gz").exists(), "local metadata unavailable")
    def test_twenty_frames_per_variation_produce_800_train_200_val(self):
        subset = subject("subset")
        metadata = Path("/media/abrar/AbrarSSD/Datasets/VirtualKitti2/from_modal/"
                        "vkitti_2.0.3_textgt.tar.gz")
        records = subset.manifest_from_metadata(metadata, frames_per_variation=20)
        self.assertEqual(len(records), 1000)
        self.assertEqual(sum(row["split"] == "train" for row in records), 800)
        self.assertEqual(sum(row["split"] == "val" for row in records), 200)

    def test_every_pair_requires_two_rgb_views_and_left_targets(self):
        subset = subject("subset")
        names = subset.required_members("Scene20", "clone", 7)
        self.assertEqual(names["rgb_left"],
                         "Scene20/clone/frames/rgb/Camera_0/rgb_00007.jpg")
        self.assertEqual(names["rgb_right"],
                         "Scene20/clone/frames/rgb/Camera_1/rgb_00007.jpg")
        self.assertEqual(names["depth"],
                         "Scene20/clone/frames/depth/Camera_0/depth_00007.png")
        self.assertEqual(names["class_seg"],
                         "Scene20/clone/frames/classSegmentation/Camera_0/classgt_00007.png")

    def test_archive_extractor_copies_only_requested_member(self):
        subset = subject("subset")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sample.tar"
            with tarfile.open(archive_path, "w") as archive:
                for name in ("Scene20/clone/keep.png", "Scene20/clone/skip.png"):
                    payload = name.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            output = root / "output"
            subset.extract_selected(archive_path, {"Scene20/clone/keep.png"}, output)
            self.assertEqual((output / "Scene20/clone/keep.png").read_bytes(),
                             b"Scene20/clone/keep.png")
            self.assertFalse((output / "Scene20/clone/skip.png").exists())


class EvaluationTests(unittest.TestCase):
    def test_disparity_metrics_ignore_invalid_pixels(self):
        run = subject("run")
        meter = run.DisparityMeter()
        prediction = torch.tensor([[[[1.0, 4.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 2.0, 0.0]]]])
        meter.add(prediction, target)
        result = meter.result()
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["epe"], 1.0)
        self.assertAlmostEqual(result["bad_1"], 50.0)

    def test_semantic_logits_are_unpadded_after_upsampling(self):
        run = subject("run")
        logits = torch.zeros(1, 2, 2, 2)
        logits[:, 1, 0] = 1
        full = run.unpad_semantic_logits(logits, padded_size=(16, 16), top=2,
                                         height=13, width=15)
        self.assertEqual(full.shape, (1, 2, 13, 15))
        self.assertGreater(float(full[0, 1, 0, 0]), float(full[0, 0, 0, 0]))

    def test_history_append_keeps_training_and_validation_records(self):
        run = subject("run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            run.append_history(path, {"phase": "train", "step": 1, "loss": 2.0})
            run.append_history(path, {"phase": "val", "step": 1, "epe": 1.0})
            self.assertEqual([row["phase"] for row in map(__import__("json").loads,
                            path.read_text().splitlines())], ["train", "val"])


if __name__ == "__main__":
    unittest.main()
