import pytest

from experiments.vkitti2.batch_benchmark import (
    BatchBenchmarkComplete,
    BatchTimer,
    benchmark_options,
    select_fastest,
)


def test_benchmark_options_preserve_training_protocol(tmp_path):
    options = benchmark_options(tmp_path / "data.yaml", tmp_path / "batch16", batch=16)
    assert options["batch"] == 16
    assert options["imgsz"] == 1248
    assert options["freeze"] == 7
    assert options["optimizer"] == "AdamW"
    assert options["fraction"] == 0.25
    assert options["val"] is False
    assert options["save"] is False


def test_timer_excludes_warmup_and_counts_images():
    ticks = iter((10.0, 30.0))
    timer = BatchTimer(batch=16, warmup_steps=2, measured_steps=3,
                       clock=lambda: next(ticks), synchronize=lambda: None)
    for _ in range(4):
        timer.on_batch_end(None)
    with pytest.raises(BatchBenchmarkComplete):
        timer.on_batch_end(None)
    assert timer.summary()["steps"] == 3
    assert timer.summary()["images"] == 48
    assert timer.summary()["images_per_second"] == pytest.approx(2.4)


def test_select_fastest_ignores_failed_batches():
    records = [
        {"batch": 4, "status": "ok", "images_per_second": 20.0},
        {"batch": 16, "status": "ok", "images_per_second": 29.0},
        {"batch": 64, "status": "oom", "images_per_second": None},
    ]
    assert select_fastest(records) == 16
