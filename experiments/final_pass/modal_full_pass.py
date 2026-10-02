"""Modal entry points for the A09 V-arm full SceneFlow run.

Use ``probe_shards_a10`` to measure real training throughput and VRAM on the
35,454-pair official split, then ``launch_a10`` with ``modal run -d`` to submit
the full run and disconnect. The worker reads /shards/v1 from the
sceneflow-shards volume; checkpoints and metrics go to svde-results. Each
retry receives the same run_name and resumes from the last atomic checkpoint.

The older Driving-only T4/L4 entry points below remain for historical runs.
See experiments/final_pass/RUN_A09M_A10.md for the exact current command and
probe measurements.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("svde-final-pass")

raw_vol = modal.Volume.from_name("stereo-datasets")
results_vol = modal.Volume.from_name("svde-results", create_if_missing=True)
shards_vol = modal.Volume.from_name("sceneflow-shards")

HERE = Path(__file__).resolve().parent
# Absolute repo root, hardcoded per the stereolite template: on the container the
# entry script is imported from /root/modal_full_pass.py, so __file__-derived
# paths are meaningless there. REPO is only used client-side (image mounts).
REPO = Path("/home/abrar/Research/Stereo-Vision-Depth-Enhancement-with-AI")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("lbzip2", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0", "torchvision", "numpy<2",
        "opencv-python-headless", "ultralytics==8.4.163",
        "scipy", "matplotlib", "zstandard",
    )
    .add_local_dir(f"{REPO}/experiments/a06_shallow", "/workspace/experiments/a06_shallow",
                   ignore=["runs/**", "manifests/**", "**/__pycache__/**",
                           "*.pth", "*.png", "*.drawio", "*.excalidraw"])
    .add_local_dir(f"{REPO}/experiments/hitnet_a03", "/workspace/experiments/hitnet_a03",
                   ignore=["runs/**", "**/__pycache__/**", "*.pth"])
    .add_local_dir(f"{REPO}/experiments/lightstereo_s_a02", "/workspace/experiments/lightstereo_s_a02",
                   ignore=["runs/**", "**/__pycache__/**", "*.pth", "*.png", "*.drawio"])
    .add_local_dir(f"{REPO}/experiments/final_pass", "/workspace/experiments/final_pass",
                   ignore=["**/__pycache__/**"])
    .add_local_file(f"{REPO}/models/segmentation/yolo26m-sem-ade20k.pt",
                    "/workspace/models/segmentation/yolo26m-sem-ade20k.pt")
    .add_local_dir("/home/abrar/Research/stero_research_claude/model/scripts",
                   "/workspace/stereolite/scripts",
                   ignore=["**/__pycache__/**", "igev_stereo_repo/**",
                           "lite_any_stereo_repo/**"])
    .add_local_dir("/home/abrar/Research/stero_research_claude/model/designs",
                   "/workspace/stereolite/designs",
                   ignore=["**/__pycache__/**", "*.pth", "*.pdf"])
    .add_local_dir("/home/abrar/Research/stero_research_claude/model/configs",
                   "/workspace/stereolite/configs")
)

DATA_DIR = Path("/data/sceneflow_driving")
RAW_ROOT = Path("/raw/sceneflow/driving")


def _prepare_data() -> str:
    """Extract Driving tarballs to container-local disk, symlink expected layout.

    Idempotent: a reused container skips straight through. The tarballs may or
    may not contain a top-level wrapper directory, so the expected dirs are
    located by name rather than assumed.
    """
    import shutil
    import subprocess

    if (DATA_DIR / "frames_finalpass").is_dir() and (DATA_DIR / "disparity").is_dir():
        return "already extracted"

    scratch = Path("/data_local")
    scratch.mkdir(exist_ok=True)
    frames_tar = RAW_ROOT / "driving__frames_finalpass.tar"
    disparity_tbz = RAW_ROOT / "driving__disparity.tar.bz2"
    for src in (frames_tar, disparity_tbz):
        if not src.exists():
            raise FileNotFoundError(f"{src} missing from the stereo-datasets volume")
    subprocess.check_call(["tar", "-xf", str(frames_tar), "-C", str(scratch)])
    subprocess.check_call(["tar", "-I", "lbzip2", "-xf", str(disparity_tbz), "-C", str(scratch)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for wanted in ("frames_finalpass", "disparity"):
        hits = sorted(scratch.rglob(wanted))
        hits = [h for h in hits if h.is_dir()]
        if not hits:
            raise RuntimeError(f"{wanted} not found after extraction")
        src = hits[0]
        dst = DATA_DIR / wanted
        if dst.exists():
            continue
        if src.parent == scratch:  # already at top level
            shutil.move(str(src), str(dst))
        else:                      # wrapper dir: symlink
            dst.symlink_to(src)
    shutil.rmtree(scratch, ignore_errors=True)
    return "extracted fresh"


def _swap_manifests() -> tuple[int, int]:
    """Copy the full-pass manifests over the runner's hardcoded manifests dir."""
    import json
    import shutil

    src = Path("/workspace/experiments/final_pass/manifests")
    dst = Path("/workspace/experiments/a06_shallow/manifests")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("manifest_train.json", "manifest_validation.json"):
        shutil.copy(src / name, dst / name)
    n_train = len(json.loads((dst / "manifest_train.json").read_text()))
    n_val = len(json.loads((dst / "manifest_validation.json").read_text()))
    return n_train, n_val


def _run_streaming(cmd: list[str], out_dir: Path, commit_seconds: int = 120) -> int:
    """Run the training subprocess, stream its output, commit the volume in a
    background thread so checkpoints are durable even mid-run."""
    import os
    import subprocess
    import threading
    import time

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir("/workspace")

    stop = threading.Event()

    def committer():
        while not stop.wait(commit_seconds):
            try:
                results_vol.commit()
            except Exception as exc:  # noqa: BLE001 - never kill training for a commit hiccup
                print(f"[committer] {exc}", flush=True)

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    finally:
        stop.set()
        thread.join()
    results_vol.commit()
    print(f"[committer] final commit done; out_dir={out_dir}", flush=True)
    return rc


ENCODER = "/workspace/models/segmentation/yolo26m-sem-ade20k.pt"


def _base_cmd(out_dir: str, steps: int, eval_every: int, checkpoint_every: int) -> list[str]:
    return [
        "python3", "-u", "/workspace/experiments/a06_shallow/runners/run_A07v2.py",
        "--arm", "V",
        "--encoder", ENCODER,
        "--steps", str(steps),
        "--eval-every", str(eval_every),
        "--checkpoint-every", str(checkpoint_every),
        "--seed", "42",
        "--out-dir", out_dir,
        "--resume",
    ]


@app.function(image=image, gpu="T4", cpu=8, memory=36864,
              volumes={"/raw": raw_vol, "/results": results_vol},
              timeout=3 * 3600, retries=0)
def probe(run_name: str, steps: int = 120, ckpt_every: int = 40) -> int:
    """Extraction + real train steps: ms/step, peak VRAM, 3 checkpoints, storage."""
    return _probe_body(run_name, steps, ckpt_every)


@app.function(image=image, gpu="L4", cpu=8, memory=36864,
              volumes={"/raw": raw_vol, "/results": results_vol},
              timeout=3 * 3600, retries=0)
def probe_l4(run_name: str, steps: int = 120, ckpt_every: int = 40) -> int:
    """Same probe on an L4 (24 GB) — the 'slightly better than T4' option."""
    return _probe_body(run_name, steps, ckpt_every)


def _probe_body(run_name: str, steps: int, ckpt_every: int) -> int:
    import subprocess

    out_dir = f"/results/final_pass/{run_name}"
    print("data:", _prepare_data(), flush=True)
    n_train, n_val = _swap_manifests()
    print(f"manifests swapped: {n_train} train / {n_val} held-out", flush=True)
    rc = _run_streaming(_base_cmd(out_dir, steps=steps, eval_every=1000,
                                  checkpoint_every=ckpt_every),
                        Path(out_dir))
    dataset = subprocess.run(["du", "-sh", "/data/sceneflow_driving"],
                             capture_output=True, text=True).stdout.strip()
    rundir = subprocess.run(["du", "-sh", out_dir],
                            capture_output=True, text=True).stdout.strip()
    print(f"[storage] extracted dataset (container disk): {dataset}", flush=True)
    print(f"[storage] run dir on results volume: {rundir}", flush=True)
    ckpts = sorted((Path(out_dir) / "checkpoints").glob("*.pth"))
    for p in ckpts:
        print(f"[storage] {p.name}: {p.stat().st_size / 2**20:.1f} MiB", flush=True)
    print(f"[storage] checkpoints written: {len(ckpts)}", flush=True)
    print(f"PROBE done rc={rc} out_dir={out_dir}", flush=True)
    return rc


@app.function(image=image, gpu="T4", cpu=8, memory=36864,
              volumes={"/raw": raw_vol, "/results": results_vol}, timeout=1800)
def probe_batches(run_name: str = "batches_t4") -> int:
    """Synthetic batch sweep with the real M-V model: peak VRAM + ms/sample."""
    return _batch_sweep()


@app.function(image=image, gpu="L4", cpu=8, memory=36864,
              volumes={"/raw": raw_vol, "/results": results_vol}, timeout=1800)
def probe_batches_l4(run_name: str = "batches_l4") -> int:
    return _batch_sweep()


def _batch_sweep() -> int:
    import sys, time
    sys.path.insert(0, "/workspace/experiments/a06_shallow/runners")
    import torch
    from model_a07v2 import FusionStereoLite

    model = FusionStereoLite(arm="V",
                             encoder="/workspace/models/segmentation/yolo26m-sem-ade20k.pt").cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    print(f"out_channels: {tuple(model.fnet.out_channels)}", flush=True)
    for batch in (1, 2, 4, 8, 16, 32):
        try:
            l = torch.rand(batch, 3, 384, 640, device="cuda")
            r = torch.rand(batch, 3, 384, 640, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            for i in range(6):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(l, r, aux=True)
                scaler.scale(out["d_final"].abs().mean()).backward()
                scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / 4
            print(f"batch {batch:2d}: {per*1000:6.0f} ms/step = "
                  f"{per/batch*1000:5.1f} ms/sample | peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
            del l, r
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch {batch:2d}: OOM", flush=True)
            break
    print(f"gpu: {torch.cuda.get_device_name()}", flush=True)
    return 0


@app.function(image=image, gpu="T4", cpu=8, memory=36864,
              volumes={"/raw": raw_vol, "/results": results_vol},
              timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                    backoff_coefficient=1.0))
def train(run_name: str, steps: int = 60000, eval_every: int = 1000,
          checkpoint_every: int = 2000) -> int:
    """The real run. Same 24h ceiling + retry+resume contract as the
    stereolite full-pass script; a preemption costs at most eval_every steps."""
    out_dir = f"/results/final_pass/{run_name}"
    print("data:", _prepare_data(), flush=True)
    n_train, n_val = _swap_manifests()
    print(f"manifests swapped: {n_train} train / {n_val} held-out", flush=True)
    return _run_streaming(_base_cmd(out_dir, steps=steps, eval_every=eval_every,
                                    checkpoint_every=checkpoint_every),
                          Path(out_dir))


# ---------------------------------------------------------------------------
# Official-split shard probes / training (sceneflow_split_v1, 35,454 train /
# 4,370 FT3D test) — streams from the sceneflow-shards volume.
# ---------------------------------------------------------------------------

SHARDS_VOLUMES = {"/raw": raw_vol, "/results": results_vol, "/shards": shards_vol}


@app.function(image=image, gpu="T4", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=3 * 3600, retries=0)
def probe_shards_t4(run_name: str = "shards_probe_t4") -> int:
    """Official-split probe on T4: real-shard batch sweep + 150-step leg."""
    return _shards_run(run_name, ["--probe", "8,16,32"], 150, 50, 8)


@app.function(image=image, gpu="L4", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=3 * 3600, retries=0)
def probe_shards_l4(run_name: str = "shards_probe_l4") -> int:
    """Official-split probe on L4: real-shard batch sweep + 150-step leg."""
    return _shards_run(run_name, ["--probe", "8,16,32,64"], 150, 50, 8)


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=3 * 3600, retries=0)
def probe_shards_a10(run_name: str = "shards_probe_a10") -> int:
    """A10 probe using actual shard decoding, A09 loss, and backward passes."""
    return _shards_run(run_name, ["--probe", "8,16,24,32,48"], 50, 50, 8)


@app.function(image=image, gpu="T4", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                    backoff_coefficient=1.0))
def train_shards_t4(run_name: str, steps: int = 100000, batch: int = 8,
                    eval_every: int = 5000, ckpt_every: int = 2000) -> int:
    """The real unattended run on T4 (official split, shard streaming)."""
    return _shards_run(run_name, [], steps, ckpt_every, batch, eval_every)


@app.function(image=image, gpu="L4", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                    backoff_coefficient=1.0))
def train_shards_l4(run_name: str, steps: int = 100000, batch: int = 8,
                    eval_every: int = 5000, ckpt_every: int = 2000) -> int:
    """The real unattended run on L4 (official split, shard streaming)."""
    return _shards_run(run_name, [], steps, ckpt_every, batch, eval_every)


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                    backoff_coefficient=1.0))
def train_shards_a10(run_name: str, steps: int = 120000, batch: int = 16,
                     eval_every: int = 2000, ckpt_every: int = 1000) -> int:
    """Full SceneFlow A09 architecture on A10, with detached launch and resume."""
    return _shards_run(run_name, [], steps, ckpt_every, batch, eval_every)


@app.local_entrypoint()
def launch_a10(run_name: str, steps: int = 120000, batch: int = 16,
               eval_every: int = 2000, ckpt_every: int = 1000) -> None:
    """Submit a durable input and return without holding a laptop connection."""
    call = train_shards_a10.spawn(run_name, steps, batch, eval_every, ckpt_every)
    print(f"Submitted {run_name}: call={call.object_id}; "
          f"results=svde-results:/final_pass/{run_name}", flush=True)


def _shards_run(run_name: str, extra_args: list[str], steps: int,
                ckpt_every: int, batch: int, eval_every: int = 5000) -> int:
    import subprocess
    import sys as _sys
    _sys.path.insert(0, "/workspace/experiments/final_pass")
    import train_shards
    results_vol.reload()
    out_dir = f"/results/final_pass/{run_name}"
    argv = ["--shards_dir", "/shards/v1", "--out_dir", out_dir,
            "--steps", str(steps), "--batch", str(batch),
            "--ckpt_every", str(ckpt_every), "--eval_every", str(eval_every),
            "--workers", "6"] + extra_args
    rc = train_shards.main(commit=results_vol.commit, argv=argv)
    if extra_args:  # probes: report storage footprint
        listing = subprocess.run(["du", "-sh", "/shards", out_dir],
                                 capture_output=True, text=True).stdout.strip()
        print(f"[storage]\n{listing}", flush=True)
        for p in sorted((Path(out_dir) / "checkpoints").glob("*.pth")):
            print(f"[storage] {p.name}: {p.stat().st_size / 2**20:.1f} MiB", flush=True)
    return rc


@app.function(image=image, gpu="A10", cpu=12, memory=36864,
              volumes=SHARDS_VOLUMES, timeout=24 * 3600,
              retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                    backoff_coefficient=1.0))
def train_shards_a10_onecycle(run_name: str, steps: int = 120000, batch: int = 16,
                              eval_every: int = 2000, ckpt_every: int = 1000) -> int:
    """Isolated scheduler comparison with the plateau run's data and model."""
    return _shards_run(run_name, ["--lr_schedule", "onecycle"],
                       steps, ckpt_every, batch, eval_every)


@app.local_entrypoint()
def launch_a10_onecycle(run_name: str, steps: int = 120000, batch: int = 16,
                        eval_every: int = 2000, ckpt_every: int = 1000) -> None:
    call = train_shards_a10_onecycle.spawn(run_name, steps, batch, eval_every, ckpt_every)
    print(f"Submitted OneCycle {run_name}: call={call.object_id}; "
          f"results=svde-results:/final_pass/{run_name}", flush=True)
