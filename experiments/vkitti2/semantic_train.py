"""Frozen-shared-trunk Ultralytics semantic training for Virtual KITTI 2."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import torch


def assert_trunk_unchanged(source: torch.nn.Module, trained: torch.nn.Module,
                           end_layer: int = 6) -> None:
    """Require bit-exact equality for shared trunk parameters *and* BN buffers."""
    source_state = source.state_dict()
    trained_state = trained.state_dict()
    seen = 0
    for name, original in source_state.items():
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        if int(parts[1]) > end_layer:
            continue
        seen += 1
        updated = trained_state.get(name)
        if updated is None or not torch.equal(original.cpu(), updated.cpu()):
            raise AssertionError(f"shared trunk changed: {name}")
    if seen == 0:
        raise AssertionError("no shared-trunk tensors found")


def training_options(dataset_yaml: Path, out_dir: Path, *, epochs: int = 60,
                     batch: int = 4) -> dict:
    """Conservative fine-tuning settings; batch is adjusted only after A10 probe."""
    return {
        "task": "semantic", "data": str(dataset_yaml), "project": str(out_dir.parent),
        "name": out_dir.name, "exist_ok": True, "epochs": epochs,
        "imgsz": 1248, "rect": True, "batch": batch, "freeze": 7,
        "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.05,
        "cos_lr": True, "warmup_epochs": 1.0, "patience": 12,
        "save": True, "save_period": 5, "val": True,
        "workers": 8, "amp": True, "seed": 42, "deterministic": True,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "fliplr": 0.5, "flipud": 0.0, "multi_scale": 0.0,
        "plots": True,
    }


def evaluation_options(dataset_yaml: Path, out_dir: Path, *, batch: int) -> dict:
    """Keep independent best-checkpoint evaluation on the held-out test set."""
    return {
        "data": str(dataset_yaml), "split": "test", "imgsz": 1248,
        "rect": True, "batch": batch, "device": 0,
        "project": str(out_dir), "name": "final_test",
        "exist_ok": True, "plots": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_semantic_metrics(metrics) -> dict:
    """Keep absent-class support visible beside per-class IoU."""
    return {
        "miou": float(metrics.miou),
        "pixel_accuracy": float(metrics.pixel_accuracy),
        "per_class": {
            str(metrics.names[i]): {
                "iou": float(metrics.per_class_iou[i]),
                "pixels": int(metrics.nt_per_class[i]),
            }
            for i in range(len(metrics.per_class_iou))
        },
    }


def checkpoint_for_run(out_dir: Path, source_weights: Path,
                       *, probe: bool) -> tuple[Path, bool]:
    """Resume full runs from committed last.pt; probes always start fresh."""
    last = out_dir / "weights" / "last.pt"
    return (last, True) if last.is_file() and not probe else (source_weights, False)


def evaluate_best(dataset_yaml: Path, source_weights: Path, out_dir: Path,
                  *, batch: int = 32,
                  commit: Callable[[], None] | None = None) -> dict:
    """Test saved best.pt without resuming training or changing model weights."""
    from ultralytics import YOLO

    checkpoint = out_dir / "weights" / "best.pt"
    if not dataset_yaml.is_file() or not source_weights.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing dataset, source, or best checkpoint: {checkpoint}")
    source = YOLO(str(source_weights))
    best = YOLO(str(checkpoint))
    if source.task != "semantic" or best.task != "semantic":
        raise ValueError("separate evaluation requires semantic checkpoints")
    assert_trunk_unchanged(source.model, best.model)
    metrics = best.val(**evaluation_options(dataset_yaml, out_dir, batch=batch))
    result = format_semantic_metrics(metrics)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint)
    result["trunk_verified"] = True
    (out_dir / "final_test.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata_path = out_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["final_test"] = result
        metadata["separate_test_batch"] = batch
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if commit is not None:
        commit()
    return result


def train_semantic(dataset_yaml: Path, source_weights: Path, out_dir: Path,
                   *, probe: bool = False, batch: int = 4,
                   commit: Callable[[], None] | None = None) -> dict:
    """Train a new semantic head while proving the shared stereo trunk is fixed."""
    from ultralytics import YOLO
    import ultralytics

    if not dataset_yaml.is_file() or not source_weights.is_file():
        raise FileNotFoundError(f"missing dataset or source: {dataset_yaml}, {source_weights}")
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint, resumed = checkpoint_for_run(out_dir, source_weights, probe=probe)
    source = YOLO(str(resume_checkpoint))
    if source.task != "semantic":
        raise ValueError(f"expected semantic checkpoint, got {source.task}")
    reference = deepcopy(YOLO(str(source_weights)).model).cpu().eval()
    if resumed:
        assert_trunk_unchanged(reference, source.model)
    options = training_options(dataset_yaml, out_dir,
                               epochs=1 if probe else 60, batch=batch)
    if probe:
        options.update(fraction=0.02, patience=1, save_period=-1, plots=False)
    if resumed:
        options["resume"] = str(resume_checkpoint)
    record = {
        "source_weights": str(source_weights),
        "source_sha256": _sha256(source_weights),
        "dataset_yaml": str(dataset_yaml),
        "dataset_audit": json.loads((dataset_yaml.parent / "audit.json").read_text()),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "options": options,
        "probe": probe,
        "resumed_from": str(resume_checkpoint) if resumed else None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    if commit is not None:
        commit()

    def verify_epoch(trainer) -> None:
        assert_trunk_unchanged(reference, trainer.model)

    def commit_saved(trainer) -> None:
        verify_epoch(trainer)
        if commit is not None:
            commit()

    source.add_callback("on_train_epoch_end", verify_epoch)
    source.add_callback("on_model_save", commit_saved)
    source.train(**options)
    best = out_dir / "weights" / "best.pt"
    last = out_dir / "weights" / "last.pt"
    checkpoint = best if best.is_file() else last
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no Ultralytics checkpoint in {out_dir}")
    assert_trunk_unchanged(reference, YOLO(str(checkpoint)).model)
    record["checkpoint"] = str(checkpoint)
    record["checkpoint_sha256"] = _sha256(checkpoint)
    record["trunk_verified"] = True
    if not probe:
        final_metrics = YOLO(str(checkpoint)).val(
            data=str(dataset_yaml), split="test", imgsz=1248, rect=True,
            batch=batch, device=0, project=str(out_dir), name="final_test",
            exist_ok=True, plots=True,
        )
        record["final_test"] = format_semantic_metrics(final_metrics)
        (out_dir / "final_test.json").write_text(
            json.dumps(record["final_test"], indent=2) + "\n"
        )
    (out_dir / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    if commit is not None:
        commit()
    return record
