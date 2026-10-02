"""CPU data staging and detached T4 training for full VKITTI E3/E4/D2."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import modal


REPO = Path("/home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI")
app = modal.App("svde-e-full-vkitti-t4")
data_volume = modal.Volume.from_name("svde-vkitti2")
results_volume = modal.Volume.from_name("svde-results")
STAGED = Path("/vkitti/full_stereo/teacher42_right_depth_v1.tar")
MANIFEST = Path("/vkitti/full_stereo/teacher42_manifest_v1.json")
SEMANTIC_DATA = Path("/vkitti/semantic/v3_random14_seed42/dataset.tar")

stage_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "experiments.E.full_vkitti_data", "experiments.vkitti2.subset")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_sync(str(REPO), frozen=True)
    .env({"PYTHONPATH": "/workspace", "MPLBACKEND": "Agg"})
    .add_local_dir(str(REPO / "experiments/E"), "/workspace/experiments/E",
                   ignore=["**/runs/**", "**/queue/**", "**/__pycache__/**", "*.pth", "*.log"])
    .add_local_dir(str(REPO / "experiments/D"), "/workspace/experiments/D",
                   ignore=["**/runs/**", "**/queue/**", "**/__pycache__/**", "*.pth"])
    .add_local_dir(str(REPO / "experiments/B"), "/workspace/experiments/B",
                   ignore=["**/runs/**", "**/__pycache__/**", "*.pth"])
    .add_local_dir(str(REPO / "experiments/C"), "/workspace/experiments/C",
                   ignore=["**/runs/**", "**/queue/**", "**/__pycache__/**", "*.pth"])
    .add_local_dir(str(REPO / "experiments/vkitti2"), "/workspace/experiments/vkitti2",
                   ignore=["**/runs/**", "**/__pycache__/**", "*.pth", "*.jpg", "*.png"])
    .add_local_dir(str(REPO / "experiments/hitnet_a03"), "/workspace/experiments/hitnet_a03",
                   ignore=["**/runs/**", "**/__pycache__/**", "*.pth"])
    .add_local_dir(str(REPO / "experiments/a06_shallow/runners"),
                   "/workspace/experiments/a06_shallow/runners",
                   ignore=["**/__pycache__/**", "*.pth"])
    .add_local_file(str(REPO / "models/segmentation/yolo26m-sem-ade20k.pt"),
                    "/workspace/models/segmentation/yolo26m-sem-ade20k.pt")
)

STEREO = Path("/results/final_pass/a09m_fullsf_a10_v1_20260929/checkpoints/best.pth")
SEMANTIC = Path("/results/vkitti2_semantic/vkitti2sem_random14_a10_v1_20260930/weights/best.pt")
ENCODER = Path("/workspace/models/segmentation/yolo26m-sem-ade20k.pt")
RUN_ROOT = Path("/results/E_full_vkitti")


@app.function(image=stage_image, volumes={"/vkitti": data_volume}, cpu=6,
              memory=16384, timeout=24 * 3600)
def stage_full() -> dict:
    """Audit teacher split and package only right RGB/depth on CPU."""
    from experiments.E.full_vkitti_data import (
        copy_selected_members, load_intrinsics, rows_from_teacher_split, teacher_split,
    )

    if STAGED.exists() and MANIFEST.exists():
        payload = json.loads(MANIFEST.read_text())
        print(f"[stage] existing complete archive: {payload['counts']}", flush=True)
        return payload["counts"]
    split = teacher_split(SEMANTIC_DATA)
    fx = load_intrinsics(Path("/vkitti/archives/vkitti_2.0.3_textgt.tar.gz"))
    rows = rows_from_teacher_split(split, fx)
    counts = {name: len(group) for name, group in rows.items()}
    if counts != {"train": 17000, "val": 2120, "test": 2140}:
        raise ValueError(f"unexpected teacher-aligned stereo counts: {counts}")
    all_rows = sum(rows.values(), [])
    with tarfile.open(SEMANTIC_DATA, "r:") as semantic:
        available = {member.name for member in semantic if member.isfile()}
    needed_semantic = {
        f"dataset/{kind}/{row['split']}/{row['stem']}.{suffix}"
        for row in all_rows for kind, suffix in (("images", "jpg"), ("masks", "png"))
    }
    missing = needed_semantic - available
    if missing:
        raise FileNotFoundError(f"semantic archive misses {len(missing)} stereo frames: {sorted(missing)[:3]}")
    STAGED.parent.mkdir(parents=True, exist_ok=True)
    temporary = STAGED.with_suffix(".tar.part")
    with tarfile.open(temporary, "w") as output:
        for modality, key in (("rgb", "rgb_right"), ("depth", "depth")):
            wanted = {row["files"][key] for row in all_rows}
            source = Path(f"/vkitti/archives/vkitti_2.0.3_{modality}.tar")
            print(f"[stage] copying {len(wanted)} {modality} members", flush=True)
            copy_selected_members(source, wanted, output)
    temporary.replace(STAGED)
    payload = {"schema": 1, "source": "semantic/v3_random14_seed42/dataset.tar",
               "grouping": split["grouping"], "seed": split["seed"],
               "counts": counts, "rows": rows, "archive_bytes": STAGED.stat().st_size}
    MANIFEST.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    data_volume.commit()
    print(f"[stage] complete: {counts}; archive bytes={payload['archive_bytes']}", flush=True)
    return counts


def _extract_data() -> Path:
    root = Path("/tmp/e_full_vkitti")
    complete = root / ".complete"
    if complete.exists():
        return root
    if not STAGED.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("run stage_full before T4 training")
    root.mkdir(parents=True, exist_ok=True)
    for archive_path in (SEMANTIC_DATA, STAGED):
        print(f"[extract] {archive_path}", flush=True)
        with tarfile.open(archive_path, "r:") as archive:
            archive.extractall(root, filter="data")
    complete.write_text("complete\n")
    return root


def _run_worker(arm: str, run_id: str, batch: int, steps: int,
                eval_every: int, eval_limit: int | None = None) -> None:
    import os
    import subprocess
    import threading

    from experiments.E.full_vkitti_train import ARM_CONFIGS

    if arm not in ARM_CONFIGS:
        raise ValueError(f"unrecognized arm: {arm}")
    if not all(path.is_file() for path in (STEREO, SEMANTIC, ENCODER)):
        raise FileNotFoundError("one of the frozen predictor checkpoints is missing")
    root = _extract_data()
    out = RUN_ROOT / run_id / arm
    if eval_limit is not None:
        out = RUN_ROOT / "probes" / run_id / arm
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", "-u", "-m", "experiments.E.full_vkitti_train",
           "--arm", arm, "--data", str(root), "--manifest", str(MANIFEST),
           "--stereo", str(STEREO), "--semantic", str(SEMANTIC),
           "--encoder", str(ENCODER), "--output", str(out),
           "--batch", str(batch), "--steps", str(steps),
           "--eval-every", str(eval_every), "--patience", "5"]
    if eval_limit is not None:
        cmd += ["--eval-limit", str(eval_limit)]
    os.chdir("/workspace")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    stop = threading.Event()

    def commit_loop() -> None:
        while not stop.wait(120):
            try:
                results_volume.commit()
            except Exception as error:
                print(f"[commit] {error!r}", flush=True)

    thread = threading.Thread(target=commit_loop, daemon=True)
    thread.start()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    finally:
        stop.set()
        thread.join()
        results_volume.commit()
    if code:
        raise RuntimeError(f"{arm} exited {code}; outputs retained at {out}")
    print(f"[worker] complete arm={arm} output={out}", flush=True)


@app.function(image=train_image, gpu="T4", cpu=8, memory=32768,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=24 * 3600, retries=0)
def probe_t4(run_id: str = "e_full_t4_probe_v1", batch: int = 4,
             steps: int = 30) -> None:
    """Real full-data extraction, train/eval/checkpoint smoke on T4."""
    _run_worker("E3", run_id, batch=batch, steps=steps, eval_every=steps, eval_limit=2)


@app.function(image=train_image, gpu="T4", cpu=8, memory=32768,
              volumes={"/vkitti": data_volume, "/results": results_volume},
              timeout=24 * 3600, retries=modal.Retries(max_retries=2, initial_delay=10))
def train_t4(arm: str, run_id: str, batch: int, steps: int,
             eval_every: int) -> None:
    _run_worker(arm, run_id, batch, steps, eval_every)


@app.local_entrypoint()
def launch(arm: str, run_id: str = "e3_e4_d2_full_t4_v1_20261002",
           batch: int = 8, steps: int = 30000, eval_every: int = 2000) -> None:
    """Submit exactly one detached long GPU task per invocation."""
    if arm not in ("E3", "E4", "D2"):
        raise ValueError("arm must be E3, E4 or D2")
    call = train_t4.spawn(arm, run_id, batch, steps, eval_every)
    print(f"submitted arm={arm} run_id={run_id} function_call_id={call.object_id}")
