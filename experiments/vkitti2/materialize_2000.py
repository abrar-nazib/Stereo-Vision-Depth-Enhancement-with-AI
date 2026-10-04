"""Verify chunked delta and hardlink the old 1000 pairs into ablation2000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    old = root / "ablation1000"
    chunks = root / "ablation2000_chunks"
    target = root / "ablation2000"
    staging = root / "ablation2000.part"
    archive = root / "ablation2000_verified.tar"
    if not old.is_dir() or not chunks.is_dir():
        raise FileNotFoundError("old subset or chunk directory missing")
    if target.exists() or staging.exists() or archive.exists():
        raise FileExistsError("target, staging, or verified archive already exists")
    manifest = json.loads((chunks / "manifest.json").read_text())
    with archive.open("wb") as output:
        for item in manifest["chunks"]:
            part = chunks / item["name"]
            if part.stat().st_size != item["bytes"] or sha256(part) != item["sha256"]:
                raise ValueError(f"corrupt transfer chunk: {part}")
            with part.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    output.write(block)
    if sha256(archive) != manifest["sha256"]:
        raise ValueError("reassembled archive checksum differs from Modal")
    staging.mkdir()
    for source in old.rglob("*"):
        if not source.is_file() or source.name == "manifest.json":
            continue
        dest = staging / source.relative_to(old)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, dest)
    prefix = "ablation2000_delta/"
    with tarfile.open(archive, "r:") as package:
        for member in package:
            if not member.isfile():
                continue
            if not member.name.startswith(prefix):
                raise ValueError(f"unexpected archive member: {member.name}")
            relative = Path(member.name[len(prefix):])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            destination = staging / relative
            if destination.exists():
                raise ValueError(f"new data overlaps old subset: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            with destination.open("wb") as stream:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    stream.write(block)
    rows = json.loads((staging / "manifest.json").read_text())
    if len(rows) != 2000 or len({(row["scene"], row["variation"], row["frame"])
                                  for row in rows}) != 2000:
        raise ValueError("manifest is not 2000 unique stereo pairs")
    for row in rows:
        if not all((staging / filename).is_file() for filename in row["files"].values()):
            raise FileNotFoundError(f"incomplete pair: {row['scene']}/{row['variation']}/{row['frame']}")
    staging.rename(target)
    print(f"[complete] {target}: 2000 unique pairs, archive sha256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
