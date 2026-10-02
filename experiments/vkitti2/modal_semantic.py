"""Modal A10 runner for frozen-trunk full-VKITTI2 semantic fine-tuning.

Sequence (from repository root):
    uv run modal run experiments/vkitti2/modal_semantic.py::prepare_full
    uv run modal run experiments/vkitti2/modal_semantic.py::probe_a10
    uv run modal run -d experiments/vkitti2/modal_semantic.py::launch_a10 \
        --run-name vkitti2sem_a10_v1_20260930 --batch 4
"""

from __future__ import annotations

from pathlib import Path

import modal


REPO = Path("/home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI")
app = modal.App("svde-vkitti2-semantic")
data_volume = modal.Volume.from_name("svde-vkitti2")
results_volume = modal.Volume.from_name("svde-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_sync(str(REPO), frozen=True)
    .env({"PYTHONPATH": "/workspace", "MPLBACKEND": "Agg"})
    .add_local_dir(
        str(REPO / "experiments/vkitti2"), "/workspace/experiments/vkitti2",
        ignore=["runs/**", "**/__pycache__/**", "*.pth", "*.png", "*.jpg"],
    )
    .add_local_file(
        str(REPO / "models/segmentation/yolo26m-sem-ade20k.pt"),
        "/workspace/models/segmentation/yolo26m-sem-ade20k.pt",
    )
)

DATASET_ARCHIVE = Path("/vkitti/semantic/v1/dataset.tar")
DATASET_AUDIT = Path("/vkitti/semantic/v1/audit.json")
RANDOM_ARCHIVE = Path("/vkitti/semantic/v3_random14_seed42/dataset.tar")
RANDOM_AUDIT = Path("/vkitti/semantic/v3_random14_seed42/audit.json")
SOURCE_WEIGHTS = Path("/workspace/models/segmentation/yolo26m-sem-ade20k.pt")


@app.function(image=image, volumes={"/vkitti": data_volume}, cpu=4, memory=8192,
              timeout=3600)
def audit_clone_support() -> dict:
    """Cheap CPU audit: class support by scene on the unmodified clone variation."""
    import tarfile
    import cv2
    import numpy as np
    from experiments.vkitti2.semantic_data import CLASS_NAMES, decode_class_mask

    hist = {scene: np.zeros(256, dtype=np.int64)
            for scene in ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")}
    source = Path("/vkitti/archives/vkitti_2.0.3_classSegmentation.tar")
    with tarfile.open(source, "r:") as archive:
        for member in archive:
            if not member.isfile() or "/clone/frames/classSegmentation/Camera_0/" not in member.name:
                continue
            scene = member.name.split("/", 1)[0]
            stream = archive.extractfile(member)
            if scene not in hist or stream is None:
                continue
            bgr = cv2.imdecode(np.frombuffer(stream.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            mask = decode_class_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), member.name)
            hist[scene] += np.bincount(mask.ravel(), minlength=256)
    result = {scene: {name: int(values[i]) for i, name in enumerate(CLASS_NAMES)}
              for scene, values in hist.items()}
    print(result, flush=True)
    return result


@app.function(image=image, volumes={"/vkitti": data_volume}, cpu=8, memory=32768,
              timeout=24 * 3600)
def prepare_full() -> dict:
    """CPU-only full conversion; source and output archives remain on Volume."""
    import json
    import tarfile
    from experiments.vkitti2.semantic_data import prepare_dataset

    if DATASET_ARCHIVE.is_file() and DATASET_AUDIT.is_file():
        audit = json.loads(DATASET_AUDIT.read_text())
        print(f"[prepare] already exists: {audit['split_counts']}", flush=True)
        return audit
    root = Path("/tmp/vkitti_semantic_v1")
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "dataset"
    rgb_tar = Path("/vkitti/archives/vkitti_2.0.3_rgb.tar")
    mask_tar = Path("/vkitti/archives/vkitti_2.0.3_classSegmentation.tar")
    audit = prepare_dataset(rgb_tar, mask_tar, dataset)
    if any(audit["split_counts"][split] == 0 for split in ("train", "val", "test")):
        raise AssertionError(f"empty scene split: {audit['split_counts']}")
    rare = {split: [name for name in ("TrafficSign", "TrafficLight", "Pole", "Truck", "Van")
                    if audit["pixel_counts"][split].get(name, 0) == 0]
            for split in ("train", "val", "test")}
    print(f"[prepare] pairs: {audit['split_counts']}; absent selected classes: {rare}", flush=True)
    DATASET_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATASET_ARCHIVE.with_suffix(".tar.part")
    with tarfile.open(temporary, "w") as archive:
        archive.add(dataset, arcname="dataset")
    temporary.replace(DATASET_ARCHIVE)
    DATASET_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    data_volume.commit()
    print(f"[prepare] dataset archive {DATASET_ARCHIVE.stat().st_size} bytes", flush=True)
    return audit


@app.function(image=image, volumes={"/vkitti": data_volume}, cpu=8, memory=32768,
              timeout=24 * 3600)
def prepare_random_14() -> dict:
    """CPU-only seeded frame-group shuffle of all 14 VKITTI classes."""
    import json
    import tarfile
    from experiments.vkitti2.semantic_data import (
        CLASS_NAMES, audit_converted_masks, reshuffle_prepared_dataset,
    )

    if RANDOM_ARCHIVE.is_file() and RANDOM_AUDIT.is_file():
        return json.loads(RANDOM_AUDIT.read_text())
    if not DATASET_ARCHIVE.is_file():
        raise FileNotFoundError(DATASET_ARCHIVE)
    root = Path("/tmp/vkitti_semantic_random_14")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DATASET_ARCHIVE, "r:") as archive:
        archive.extractall(root, filter="data")
    dataset = root / "dataset"
    split_summary = reshuffle_prepared_dataset(dataset, seed=42)
    audit = audit_converted_masks(dataset, CLASS_NAMES)
    audit["split_protocol"] = {
        "kind": "source-frame grouped random 80/10/10",
        "seed": 42,
        "group_counts": split_summary["group_counts"],
        "manifest": "split_manifest.json",
    }
    if audit["split_counts"] != split_summary["split_counts"]:
        raise AssertionError("split count changed after mask audit")
    missing = {
        split: [name for name in CLASS_NAMES
                if audit["pixel_counts"][split][name] == 0]
        for split in ("train", "val", "test")
    }
    if any(missing.values()):
        raise ValueError(f"random split lacks classes: {missing}")
    (dataset / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    RANDOM_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RANDOM_ARCHIVE.with_suffix(".tar.part")
    with tarfile.open(temporary, "w") as archive:
        archive.add(dataset, arcname="dataset")
    temporary.replace(RANDOM_ARCHIVE)
    RANDOM_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    data_volume.commit()
    print(f"[random14] pairs {audit['split_counts']}, archive "
          f"{RANDOM_ARCHIVE.stat().st_size} bytes", flush=True)
    return audit


def _extract_dataset() -> Path:
    import tarfile
    import yaml

    if not RANDOM_ARCHIVE.is_file():
        raise FileNotFoundError(f"run prepare_random_14 first: {RANDOM_ARCHIVE}")
    root = Path("/tmp/vkitti_semantic_run")
    dataset = root / "dataset"
    extraction_complete = dataset / ".extraction_complete"
    if not extraction_complete.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(RANDOM_ARCHIVE, "r:") as archive:
            archive.extractall(root, filter="data")
        extraction_complete.write_text("complete\n")
    dataset_yaml = dataset / "dataset.yaml"
    config = yaml.safe_load(dataset_yaml.read_text())
    config["path"] = str(dataset)
    dataset_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return dataset_yaml


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=24 * 3600)
def probe_a10(batch: int = 4) -> dict:
    """One epoch on 2% of train, with full random validation."""
    from experiments.vkitti2.semantic_train import train_semantic

    dataset_yaml = _extract_dataset()
    return train_semantic(dataset_yaml, SOURCE_WEIGHTS,
                          Path("/results/vkitti2_semantic/a10_probe_random14_v1"),
                          probe=True, batch=batch, commit=results_volume.commit)


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=24 * 3600, retries=2)
def train_a10(run_name: str, batch: int = 4) -> dict:
    """Full semantic run; retries resume from Volume checkpoint when available."""
    from experiments.vkitti2.semantic_train import train_semantic

    if not run_name or "/" in run_name or run_name in (".", ".."):
        raise ValueError(f"invalid run name {run_name!r}")
    dataset_yaml = _extract_dataset()
    out_dir = Path("/results/vkitti2_semantic") / run_name
    if (out_dir / "final_test.json").is_file():
        import json
        return json.loads((out_dir / "run_metadata.json").read_text())
    return train_semantic(dataset_yaml, SOURCE_WEIGHTS, out_dir,
                          probe=False, batch=batch, commit=results_volume.commit)


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=2 * 3600)
def benchmark_a10() -> dict:
    """Sweep batch 4/8/16/32/64 on one A10; do not launch full training."""
    from experiments.vkitti2.batch_benchmark import benchmark_batches

    dataset_yaml = _extract_dataset()
    return benchmark_batches(
        dataset_yaml, SOURCE_WEIGHTS,
        Path("/results/vkitti2_semantic/batch_benchmark_a10_20260930"),
        commit=results_volume.commit,
    )


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=2 * 3600)
def evaluate_best_a10(run_name: str, batch: int = 32) -> dict:
    """Independent held-out test evaluation of the saved best checkpoint."""
    from experiments.vkitti2.semantic_train import evaluate_best

    if not run_name or "/" in run_name or run_name in (".", ".."):
        raise ValueError(f"invalid run name {run_name!r}")
    dataset_yaml = _extract_dataset()
    out_dir = Path("/results/vkitti2_semantic") / run_name
    return evaluate_best(dataset_yaml, SOURCE_WEIGHTS, out_dir,
                         batch=batch, commit=results_volume.commit)


@app.local_entrypoint()
def launch_a10(run_name: str, batch: int = 4) -> None:
    call = train_a10.spawn(run_name, batch)
    print(f"Detached A10 call: {call.object_id}", flush=True)


@app.local_entrypoint()
def launch_batch_benchmark() -> None:
    call = benchmark_a10.spawn()
    print(f"Detached A10 batch benchmark: {call.object_id}", flush=True)


@app.local_entrypoint()
def launch_best_evaluation(run_name: str, batch: int = 32) -> None:
    call = evaluate_best_a10.spawn(run_name, batch)
    print(f"Detached A10 best-checkpoint evaluation: {call.object_id}", flush=True)
