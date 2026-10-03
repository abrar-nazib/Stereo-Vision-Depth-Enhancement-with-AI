"""Detached, inference-only KITTI 2015 F-series evaluation on Modal T4."""

from __future__ import annotations

import json
from pathlib import Path

import modal


REPO = Path("/home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI")
app = modal.App("svde-f-kitti2015-t4")
datasets_volume = modal.Volume.from_name("stereo-datasets")
results_volume = modal.Volume.from_name("svde-results")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_sync(str(REPO), frozen=True)
    .env({"PYTHONPATH": "/workspace", "MPLBACKEND": "Agg"})
    .add_local_dir(str(REPO / "experiments/F"), "/workspace/experiments/F",
                   ignore=["**/runs/**", "**/__pycache__/**", "*.pth", "*.log"])
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

ARCHIVE = Path("/datasets/kitti/data_scene_flow.zip")
STEREO = Path("/results/final_pass/a09m_fullsf_a10_v1_20260929/checkpoints/best.pth")
SEMANTIC = Path("/results/vkitti2_semantic/vkitti2sem_random14_a10_v1_20260930/weights/best.pt")
ENCODER = Path("/workspace/models/segmentation/yolo26m-sem-ade20k.pt")
E3_HEAD = Path("/results/E_full_vkitti/e3_e4_d2_full_t4_v1_20261002/E3/checkpoints/best.pth")
E4_HEAD = Path("/results/E_full_vkitti/e3_e4_d2_full_t4_v1_20261002/E4/checkpoints/best.pth")
RUN_ROOT = Path("/results/F_kitti2015")


def worker_command(arm: str, *, archive: Path, stereo: Path, semantic: Path,
                   encoder: Path, head: Path, output: Path,
                   limit: int | None = None) -> list[str]:
    if arm not in ("F1", "F2", "F3"):
        raise ValueError(f"unsupported F arm: {arm}")
    command = ["python3", "-u", "-m", "experiments.F.evaluate_kitti",
               "--arm", arm, "--archive", str(archive), "--stereo", str(stereo),
               "--semantic", str(semantic), "--encoder", str(encoder),
               "--output", str(output)]
    if arm in ("F2", "F3"):
        command += ["--head", str(head)]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def _run(run_id: str, limit: int | None = None,
         arms: tuple[str, ...] = ("F1", "F2")) -> None:
    import os
    import shutil
    import subprocess

    if not run_id or "/" in run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single non-empty path component")
    required = [ARCHIVE, STEREO, SEMANTIC, ENCODER]
    if "F2" in arms:
        required.append(E3_HEAD)
    if "F3" in arms:
        required.append(E4_HEAD)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("KITTI archive or frozen model/head checkpoint missing")
    local_archive = Path("/tmp/f_kitti2015_scene_flow.zip")
    if not local_archive.is_file() or local_archive.stat().st_size != ARCHIVE.stat().st_size:
        shutil.copyfile(ARCHIVE, local_archive)
    root = RUN_ROOT / ("probes" if limit is not None else "full") / run_id
    root.mkdir(parents=True, exist_ok=True)
    os.chdir("/workspace")
    for arm in arms:
        out = root / arm
        if (out / "result.json").is_file():
            print(json.dumps({"phase": "skip_complete", "arm": arm, "output": str(out)}),
                  flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = worker_command(arm, archive=local_archive, stereo=STEREO,
                                 semantic=SEMANTIC, encoder=ENCODER,
                                 head=E4_HEAD if arm == "F3" else E3_HEAD,
                                 output=out, limit=limit)
        print(json.dumps({"phase": "start", "arm": arm, "output": str(out)}), flush=True)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith('{"arm":'):
                    results_volume.commit()
            code = process.wait()
        results_volume.commit()
        if code:
            raise RuntimeError(f"{arm} evaluation exited {code}; inspect {out}/status.json")
    print(json.dumps({"phase": "complete", "run_id": run_id, "root": str(root)}), flush=True)


@app.function(image=image, gpu="T4", cpu=4, memory=16384,
              volumes={"/datasets": datasets_volume, "/results": results_volume},
              timeout=4 * 3600, retries=0)
def probe_t4(run_id: str = "f_kitti2015_probe_v1", limit: int = 2) -> None:
    _run(run_id, limit=limit)


@app.function(image=image, gpu="T4", cpu=4, memory=16384,
              volumes={"/datasets": datasets_volume, "/results": results_volume},
              timeout=4 * 3600, retries=0)
def evaluate_t4(run_id: str) -> None:
    _run(run_id)


@app.function(image=image, gpu="T4", cpu=4, memory=16384,
              volumes={"/datasets": datasets_volume, "/results": results_volume},
              timeout=4 * 3600, retries=0)
def probe_control_t4(run_id: str = "f3_kitti2015_probe_v1", limit: int = 2) -> None:
    _run(run_id, limit=limit, arms=("F3",))


@app.function(image=image, gpu="T4", cpu=4, memory=16384,
              volumes={"/datasets": datasets_volume, "/results": results_volume},
              timeout=4 * 3600, retries=0)
def evaluate_control_t4(run_id: str) -> None:
    _run(run_id, arms=("F3",))


@app.local_entrypoint()
def launch(run_id: str = "f1_f2_kitti2015_v1_20261003") -> None:
    call = evaluate_t4.spawn(run_id)
    print(f"submitted run_id={run_id} function_call_id={call.object_id}")


@app.local_entrypoint()
def launch_control(run_id: str = "f1_f2_kitti2015_v1_20261003") -> None:
    call = evaluate_control_t4.spawn(run_id)
    print(f"submitted arm=F3 run_id={run_id} function_call_id={call.object_id}")
