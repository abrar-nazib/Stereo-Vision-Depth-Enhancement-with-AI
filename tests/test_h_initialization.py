from pathlib import Path

import torch
from torch import nn

from experiments.H.run import initialize_stereo


def test_scratch_mode_keeps_fresh_parameters_and_does_not_read_checkpoint(tmp_path: Path):
    model = nn.Linear(2, 1)
    original = model.weight.detach().clone()
    initialize_stereo(model, "scratch", tmp_path / "nonexistent.pth")
    torch.testing.assert_close(model.weight, original)


def test_sceneflow_mode_loads_checkpoint_when_explicit(tmp_path: Path):
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "pretrained.pth"
    target = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
    torch.save({"model": target}, checkpoint)
    initialize_stereo(model, "sceneflow", checkpoint)
    torch.testing.assert_close(model.weight, target["weight"])
