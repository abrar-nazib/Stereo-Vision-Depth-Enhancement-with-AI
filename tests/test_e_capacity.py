import torch
import json
import subprocess
import sys
from pathlib import Path

from experiments.D.D_1_semantic_cost.model import ClassResidual, SemanticCostGate


ROOT = Path(__file__).resolve().parents[1]


def test_wider_d2_modules_preserve_identity_and_increase_only_head_parameters():
    volume = torch.randn(1, 16, 4, 2, 5)
    probs = torch.softmax(torch.randn(1, 14, 2, 5), 1)
    disparity = torch.randn(1, 1, 8, 20)
    confidence = torch.rand(1, 1, 2, 5)

    for gate_width, residual_width, expected in (
        (8, 32, 9758),
        (16, 64, 19374),
        (32, 128, 38606),
    ):
        gate = SemanticCostGate(hidden_channels=gate_width)
        residual = ClassResidual(hidden_channels=residual_width)
        actual = sum(p.numel() for p in gate.parameters()) + sum(
            p.numel() for p in residual.parameters()
        )
        assert actual == expected
        assert torch.allclose(gate(volume, probs, probs), torch.ones(1, 1, 4, 2, 5))
        assert torch.allclose(residual(disparity, probs, confidence, True), disparity)


def test_e_runner_lists_matched_width_and_semantics_arms_without_loading_weights():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.E.E_1_wide2_semantics.run", "--list-arms"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "E_1_wide2_semantics": {"gate_hidden": 16, "residual_hidden": 64, "use_semantics": True},
        "E_2_wide2_control": {"gate_hidden": 16, "residual_hidden": 64, "use_semantics": False},
        "E_3_wide4_semantics": {"gate_hidden": 32, "residual_hidden": 128, "use_semantics": True},
        "E_4_wide4_control": {"gate_hidden": 32, "residual_hidden": 128, "use_semantics": False},
    }


def test_e_queue_dry_run_is_sequential_and_does_not_create_runs():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.E.E_1_wide2_semantics.queue",
         "--dry-run", "--run-id", "e_dry_run_no_write"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert [entry["arm"] for entry in plan["jobs"]] == [
        "E_1_wide2_semantics", "E_2_wide2_control",
        "E_3_wide4_semantics", "E_4_wide4_control",
    ]
    assert all(entry["command"][2] == "experiments.E.E_1_wide2_semantics.run"
               for entry in plan["jobs"])
    assert not (ROOT / "experiments/E/E_1_wide2_semantics/queue/e_dry_run_no_write").exists()
