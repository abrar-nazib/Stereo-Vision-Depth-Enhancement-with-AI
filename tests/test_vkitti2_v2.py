"""Contracts for confidence-gated, right-aware and edge-aware residual heads."""

import unittest

import torch

from experiments.vkitti2 import model_v2


class V2ModelTests(unittest.TestCase):
    def test_right_warp_uses_pixel_disparity_at_low_resolution(self):
        right = torch.arange(5, dtype=torch.float32)[None, None, None, :]
        disparity_full_pixels = torch.ones(1, 1, 1, 5) * 2
        warped, valid = model_v2.warp_right_to_left(right, disparity_full_pixels,
                                                    full_width=10)
        torch.testing.assert_close(warped[0, 0, 0, 1:], torch.tensor([0., 1., 2., 3.]))
        self.assertFalse(bool(valid[0, 0, 0, 0]))
        self.assertTrue(bool(valid[0, 0, 0, 1]))

    def test_zero_initialized_variants_preserve_both_predictions(self):
        disparity = torch.ones(1, 1, 32, 64) * 10
        logits = torch.randn(1, 150, 4, 8)
        image = torch.rand(1, 3, 32, 64)
        for use_warp, use_edge in ((False, False), (True, False), (True, True)):
            with self.subTest(warp=use_warp, edge=use_edge):
                model = model_v2.GatedResidual(use_warp=use_warp, use_edge=use_edge)
                result_d, result_s = model(disparity, logits, image, image)
                torch.testing.assert_close(result_d, disparity)
                torch.testing.assert_close(result_s, logits)
                self.assertLess(sum(p.numel() for p in model.parameters()), 200_000)


class EdgeMetricTests(unittest.TestCase):
    def test_edge_mask_marks_disparity_jump_and_excludes_invalid(self):
        from experiments.vkitti2 import run_v2
        gt = torch.tensor([[[[1., 1., 1., 10., 10., 0.]]]])
        edge = run_v2.disparity_edge_mask(gt, threshold=4, radius=0)
        self.assertTrue(bool(edge[0, 0, 0, 3]))
        self.assertFalse(bool(edge[0, 0, 0, 1]))
        self.assertFalse(bool(edge[0, 0, 0, 5]))

    def test_each_arm_visits_every_train_record_each_epoch(self):
        from experiments.vkitti2 import run_v2
        indices = list(run_v2.epoch_record_indices(4, steps=8, seed=260930))
        self.assertEqual(len(indices), 8)
        self.assertEqual(set(indices[:4]), set(range(4)))
        self.assertEqual(set(indices[4:]), set(range(4)))
        self.assertEqual(indices, list(run_v2.epoch_record_indices(4, 8, 260930)))
