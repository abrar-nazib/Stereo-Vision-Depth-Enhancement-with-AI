"""Behavior checks for the full-SceneFlow LR schedule switch."""

import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/final_pass/lr_schedule.py"
SPEC = importlib.util.spec_from_file_location("final_pass_lr_schedule", MODULE_PATH)
lr_schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lr_schedule)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.AdamW([self.weight], lr=1e-4)

    def test_onecycle_changes_lr_on_training_steps_and_not_validation(self):
        scheduler = lr_schedule.make_scheduler("onecycle", self.optimizer, 1000, 1e-4)
        initial = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        lr_schedule.after_training_step("onecycle", scheduler)
        after_step = self.optimizer.param_groups[0]["lr"]
        lr_schedule.after_validation("onecycle", scheduler, 4.0)
        self.assertNotEqual(initial, after_step)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], after_step)

    def test_plateau_changes_only_after_validation_plateau(self):
        scheduler = lr_schedule.make_scheduler("plateau", self.optimizer, 100, 1e-4)
        lr_schedule.after_training_step("plateau", scheduler)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 1e-4)
        for _ in range(5):
            lr_schedule.after_validation("plateau", scheduler, 4.0)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 5e-5)

    def test_onecycle_state_resumes_at_same_lr(self):
        scheduler = lr_schedule.make_scheduler("onecycle", self.optimizer, 1000, 1e-4)
        for _ in range(11):
            self.optimizer.step()
            lr_schedule.after_training_step("onecycle", scheduler)
        lr = self.optimizer.param_groups[0]["lr"]
        restored_opt = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-4)
        restored_sched = lr_schedule.make_scheduler("onecycle", restored_opt, 1000, 1e-4)
        restored_opt.load_state_dict(self.optimizer.state_dict())
        restored_sched.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored_opt.param_groups[0]["lr"], lr)
        self.assertEqual(restored_sched.last_epoch, scheduler.last_epoch)


if __name__ == "__main__":
    unittest.main()
