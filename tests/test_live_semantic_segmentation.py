"""Behavior tests for the live stereo semantic-segmentation CLI helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.live_semantic_segmentation import crop_stereo_frame, parse_args, resolve_model, should_exit


class StereoCropTests(unittest.TestCase):
    def test_left_side_returns_the_left_half_of_an_sbs_frame(self) -> None:
        frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)

        cropped = crop_stereo_frame(frame, "left")

        np.testing.assert_array_equal(cropped, frame[:, :4])

    def test_right_side_returns_the_right_half_of_an_sbs_frame(self) -> None:
        frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)

        cropped = crop_stereo_frame(frame, "right")

        np.testing.assert_array_equal(cropped, frame[:, 4:])

    def test_odd_width_sbs_frame_is_rejected(self) -> None:
        frame = np.zeros((4, 7, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "even width"):
            crop_stereo_frame(frame, "left")


class ExitConditionTests(unittest.TestCase):
    def test_q_escape_or_closed_window_requests_exit(self) -> None:
        self.assertTrue(should_exit(ord("q"), True))
        self.assertTrue(should_exit(27, True))
        self.assertTrue(should_exit(-1, False))

    def test_open_window_with_another_key_continues(self) -> None:
        self.assertFalse(should_exit(ord("a"), True))


class ArgumentTests(unittest.TestCase):
    def test_model_argument_accepts_a_local_checkpoint_path(self) -> None:
        arguments = parse_args(["--model", "/tmp/custom-sem.pt"])

        self.assertEqual(arguments.model, "/tmp/custom-sem.pt")

    def test_stereo_model_selects_a_fast_foundation_stereo_preset(self) -> None:
        arguments = parse_args(["--model", "none", "--stereo-model", "fastfs-20-30-48"])

        self.assertEqual(arguments.stereo_model, "fastfs-20-30-48")

    def test_fastfs_can_run_without_a_semantic_model(self) -> None:
        arguments = parse_args(["--model", "none", "--stereo-model", "fastfs-20-30-48"])

        self.assertEqual(arguments.model, "none")
        self.assertEqual(arguments.stereo_model, "fastfs-20-30-48")
        self.assertIsNone(resolve_model(arguments.model))

    def test_joint_ticoss_is_the_only_selected_pipeline(self) -> None:
        arguments = parse_args(["--model", "none", "--joint-model", "ticoss"])

        self.assertEqual(arguments.joint_model, "ticoss")
        self.assertEqual(arguments.stereo_model, "none")

    def test_rejects_multiple_simultaneous_pipelines(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--joint-model", "ticoss"])


if __name__ == "__main__":
    unittest.main()
