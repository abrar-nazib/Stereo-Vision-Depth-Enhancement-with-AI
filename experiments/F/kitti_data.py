"""Read the public KITTI 2015 labeled stereo pairs without resizing."""

from __future__ import annotations

from zipfile import ZipFile

import cv2
import numpy as np


def list_pair_ids(archive: ZipFile) -> list[str]:
    names = set(archive.namelist())
    prefix = "training/image_2/"
    ids = sorted(name[len(prefix):-4] for name in names
                 if name.startswith(prefix) and name.endswith("_10.png"))
    if not ids:
        raise ValueError("KITTI archive has no training left-view pairs")
    for stem in ids:
        required = (f"training/image_3/{stem}.png",
                    f"training/disp_occ_0/{stem}.png",
                    f"training/disp_noc_0/{stem}.png")
        absent = [path for path in required if path not in names]
        if absent:
            raise ValueError(f"KITTI pair {stem} missing {absent}")
    return ids


def read_pair(archive: ZipFile, stem: str) -> tuple[np.ndarray, ...]:
    def decode(path: str, flags: int) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(archive.read(path), dtype=np.uint8), flags)
        if image is None:
            raise ValueError(f"failed decoding KITTI {path}")
        return image

    left = decode(f"training/image_2/{stem}.png", cv2.IMREAD_COLOR)
    right = decode(f"training/image_3/{stem}.png", cv2.IMREAD_COLOR)
    occ = decode(f"training/disp_occ_0/{stem}.png", cv2.IMREAD_UNCHANGED)
    noc = decode(f"training/disp_noc_0/{stem}.png", cv2.IMREAD_UNCHANGED)
    if (left.shape != right.shape or left.shape[:2] != occ.shape or
            occ.shape != noc.shape or occ.dtype != np.uint16 or noc.dtype != np.uint16):
        raise ValueError(f"KITTI pair {stem} has incompatible shape or disparity dtype")
    return (cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
            occ.astype(np.float32) / 256.0,
            noc.astype(np.float32) / 256.0)
