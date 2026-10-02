from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.vkitti2.semantic_train import (
    assert_trunk_unchanged,
    format_semantic_metrics,
    training_options,
    checkpoint_for_run,
    evaluation_options,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList((nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3)), nn.Conv2d(3, 3, 1)))


def test_trunk_parameters_and_buffers_identical():
    before = TinyModel()
    after = TinyModel()
    after.load_state_dict(before.state_dict())
    assert_trunk_unchanged(before, after, end_layer=0)
    with torch.no_grad():
        after.model[0][0].weight.add_(1)
    with pytest.raises(AssertionError, match="model.0"):
        assert_trunk_unchanged(before, after, end_layer=0)


def test_trunk_batchnorm_buffer_change_is_detected():
    before = TinyModel()
    after = TinyModel()
    after.load_state_dict(before.state_dict())
    with torch.no_grad():
        after.model[0][1].running_mean.add_(1)
    with pytest.raises(AssertionError, match="running_mean"):
        assert_trunk_unchanged(before, after, end_layer=0)


def test_training_options_freeze_exact_shared_trunk(tmp_path):
    options = training_options(tmp_path / "dataset.yaml", tmp_path / "runs", epochs=60, batch=4)
    assert options["freeze"] == 7
    assert options["task"] == "semantic"
    assert options["imgsz"] == 1248
    assert options["rect"] is True
    assert options["optimizer"] == "AdamW"
    assert options["batch"] == 4
    assert Path(options["data"]) == tmp_path / "dataset.yaml"


def test_format_semantic_metrics_preserves_class_support():
    class Metrics:
        miou = 0.6
        pixel_accuracy = 0.9
        per_class_iou = [0.4, 0.8]
        nt_per_class = [10, 0]
        names = {0: "Road", 1: "Sky"}

    result = format_semantic_metrics(Metrics())
    assert result["miou"] == 0.6
    assert result["per_class"]["Sky"] == {"iou": 0.8, "pixels": 0}


def test_checkpoint_for_run_resumes_existing_last_checkpoint(tmp_path):
    source = tmp_path / "source.pt"
    out = tmp_path / "run"
    last = out / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"checkpoint")
    assert checkpoint_for_run(out, source, probe=False) == (last, True)
    assert checkpoint_for_run(out, source, probe=True) == (source, False)


def test_separate_evaluation_uses_held_out_test_split(tmp_path):
    dataset_yaml = tmp_path / "dataset.yaml"
    out_dir = tmp_path / "run"
    options = evaluation_options(dataset_yaml, out_dir, batch=32)
    assert options["data"] == str(dataset_yaml)
    assert options["split"] == "test"
    assert options["imgsz"] == 1248
    assert options["rect"] is True
    assert options["batch"] == 32
    assert options["project"] == str(out_dir)
    assert options["name"] == "final_test"
