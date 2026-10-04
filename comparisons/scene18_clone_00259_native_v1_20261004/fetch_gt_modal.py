"""Retrieve only the Scene18 clone frame-259 depth PNG from the Modal VKITTI volume."""

from pathlib import Path
import tarfile

import modal


app = modal.App("svde-scene18-single-depth-read")
volume = modal.Volume.from_name("svde-vkitti2")
MEMBER = "Scene18/clone/frames/depth/Camera_0/depth_00259.png"
OUTPUT = Path(__file__).resolve().parent / "depth_00259.png"


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
              volumes={"/vkitti": volume}, cpu=1, timeout=1800)
def fetch_depth() -> bytes:
    archive = "/vkitti/full_stereo/teacher42_right_depth_v1.tar"
    with tarfile.open(archive, "r:") as stream:
        source = stream.extractfile(MEMBER)
        if source is None:
            raise FileNotFoundError(MEMBER)
        return source.read()


@app.local_entrypoint()
def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    contents = fetch_depth.remote()
    OUTPUT.write_bytes(contents)
    print(f"Retrieved {MEMBER} to {OUTPUT}: {len(contents)} bytes")
