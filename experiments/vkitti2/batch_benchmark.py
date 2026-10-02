"""Short A10 throughput sweep for the existing VKITTI semantic trainer.

This is not a model-quality experiment: it trains fresh weights for each batch
and aborts before validation/checkpointing. The measured interval excludes the
first few CUDA/dataloader batches and uses the same image and train recipe.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable

from experiments.vkitti2.semantic_train import training_options


class BatchBenchmarkComplete(Exception):
    """Intentional exit after the fixed number of measured batches."""


class BatchShrunk(Exception):
    """Ultralytics automatically reduced an out-of-memory batch."""


class BatchTimer:
    def __init__(self, batch: int, *, warmup_steps: int = 8,
                 measured_steps: int = 40,
                 clock: Callable[[], float] = time.perf_counter,
                 synchronize: Callable[[], None] = lambda: None,
                 on_warmup_end: Callable[[], None] = lambda: None):
        self.batch = batch
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.clock = clock
        self.synchronize = synchronize
        self.on_warmup_end = on_warmup_end
        self.completed = 0
        self.started: float | None = None
        self.elapsed: float | None = None

    def on_batch_end(self, trainer) -> None:
        if trainer is not None and trainer.batch_size != self.batch:
            raise BatchShrunk(f"requested {self.batch}, actual {trainer.batch_size}")
        self.completed += 1
        if self.completed == self.warmup_steps:
            self.synchronize()
            self.on_warmup_end()
            self.started = self.clock()
        if self.completed == self.warmup_steps + self.measured_steps:
            self.synchronize()
            self.elapsed = self.clock() - self.started
            raise BatchBenchmarkComplete

    def summary(self) -> dict:
        if self.elapsed is None or self.elapsed <= 0:
            raise ValueError("benchmark did not finish")
        images = self.batch * self.measured_steps
        return {
            "batch": self.batch,
            "warmup_steps": self.warmup_steps,
            "steps": self.measured_steps,
            "images": images,
            "seconds": self.elapsed,
            "seconds_per_step": self.elapsed / self.measured_steps,
            "images_per_second": images / self.elapsed,
        }


def benchmark_options(dataset_yaml: Path, out_dir: Path, batch: int) -> dict:
    options = training_options(dataset_yaml, out_dir, epochs=60, batch=batch)
    options.update(fraction=0.25, val=False, save=False, plots=False,
                   save_period=-1, patience=0, workers=8)
    return options


def select_fastest(records: list[dict]) -> int:
    successful = [r for r in records if r["status"] == "ok"]
    if not successful:
        raise ValueError("no batch size completed the benchmark")
    return max(successful, key=lambda r: r["images_per_second"])["batch"]


def benchmark_batches(dataset_yaml: Path, source_weights: Path, out_dir: Path,
                      batches: tuple[int, ...] = (4, 8, 16, 32, 64),
                      commit: Callable[[], None] | None = None) -> dict:
    """Measure steady-state *training* throughput; persist after every candidate."""
    import torch
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark.json"
    report = {
        "purpose": "batch-size throughput only; no validation or quality comparison",
        "dataset_yaml": str(dataset_yaml),
        "source_weights": str(source_weights),
        "batches": list(batches),
        "records": [],
    }
    for batch in batches:
        model = None
        timer = BatchTimer(batch, synchronize=torch.cuda.synchronize,
                           on_warmup_end=torch.cuda.reset_peak_memory_stats)
        try:
            model = YOLO(str(source_weights))
            model.add_callback("on_train_batch_end", timer.on_batch_end)
            model.train(**benchmark_options(dataset_yaml, out_dir / f"batch_{batch}", batch))
            raise RuntimeError(f"batch {batch} ended before the measurement completed")
        except BatchBenchmarkComplete:
            record = {"status": "ok", **timer.summary(),
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        except (BatchShrunk, torch.cuda.OutOfMemoryError) as exc:
            record = {"batch": batch, "status": "oom", "error": str(exc),
                      "images_per_second": None}
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            record = {"batch": batch, "status": "oom", "error": str(exc),
                      "images_per_second": None}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["records"].append(record)
        successful = [r for r in report["records"] if r["status"] == "ok"]
        report["fastest_batch"] = select_fastest(report["records"]) if successful else None
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if commit:
            commit()
        print(f"[batch-benchmark] {record}", flush=True)
    return report
