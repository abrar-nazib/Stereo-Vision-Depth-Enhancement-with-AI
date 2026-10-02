"""Fetch the Virtual KITTI 2 modalities needed for joint stereo/semantic work.

Run detached with ``uv run modal run -d experiments/vkitti2/modal_download.py::download_all``.
Files are published to the volume only after their official MD5 matches.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import modal


app = modal.App("svde-vkitti2-download")
volume = modal.Volume.from_name("svde-vkitti2")
base_url = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
archives = {
    "vkitti_2.0.3_rgb.tar": (7532472320, "1e00a143a397c2c53aa9720a868fe34a"),
    "vkitti_2.0.3_depth.tar": (8145715200, "1d34a96f870fc5d33b482df87844f5e4"),
    "vkitti_2.0.3_classSegmentation.tar": (1015961600, "e8658ab49250e61f1caa174625563263"),
    "vkitti_2.0.3_textgt.tar.gz": (24451078, "855f7f37746e5508f094e522c9cf41f4"),
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=2,
    memory=4096,
    volumes={"/vkitti": volume},
    timeout=3 * 3600,
    retries=1,
)
def download_all() -> None:
    root = Path("/vkitti/archives")
    root.mkdir(parents=True, exist_ok=True)
    for name, (expected_size, expected_md5) in archives.items():
        destination = root / name
        if destination.is_file() and destination.stat().st_size == expected_size:
            if md5_file(destination) == expected_md5:
                print(f"[verified] {name} (already present)", flush=True)
                continue

        temporary = root / f"{name}.part"
        print(f"[download] {name} ({expected_size / 1e9:.2f} GB)", flush=True)
        digest = hashlib.md5()
        received = 0
        with urllib.request.urlopen(f"{base_url}/{name}", timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"{name}: unexpected HTTP status {response.status}")
            with temporary.open("wb") as stream:
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if received // (512 * 1024 * 1024) != (received - len(chunk)) // (512 * 1024 * 1024):
                        print(f"[progress] {name}: {received / 1e9:.2f} GB", flush=True)
                stream.flush()
                os.fsync(stream.fileno())
        if received != expected_size or digest.hexdigest() != expected_md5:
            raise RuntimeError(
                f"{name}: size/MD5 mismatch: {received} bytes, {digest.hexdigest()}"
            )
        os.replace(temporary, destination)
        volume.commit()
        print(f"[verified] {name}: {received} bytes, MD5 {expected_md5}", flush=True)
    print("[complete] all Virtual KITTI 2 stereo/semantic archives verified", flush=True)
