"""Keep selected experiment checkpoints eligible for version control."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "should_be_ignored"),
    [
        ("experiments/B/B_1_residual/runs/full/best_epe.pth", False),
        ("experiments/C/C_1_large_control/runs/full/checkpoints/best.pth", False),
        ("experiments/D/D_1_semantic_cost/runs/full/checkpoints/best_model.pt", False),
        ("experiments/C/C_1_large_control/runs/full/checkpoints/step_001000.pth", True),
        ("experiments/D/D_1_semantic_cost/runs/full/checkpoints/step_001000.pt", True),
        ("models/stereo/example/model_best_weights.pth", True),
        ("models/segmentation/example-best.pt", True),
    ],
)
def test_only_best_experiment_run_checkpoints_are_git_eligible(path, should_be_ignored):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == (0 if should_be_ignored else 1)
