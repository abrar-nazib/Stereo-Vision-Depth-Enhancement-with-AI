"""Run released SemTileStereo on a live stereo camera or one rectified image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from semtilestereo.camera import load_rectification, split_stereo
from semtilestereo.core import ModelPaths, infer_pair, load_model
from semtilestereo.results import CameraParameters, save_result
from semtilestereo.visuals import colorize_disparity, segmentation_overlay


DISPARITY_WINDOW = "SemTileStereo disparity"
SEGMENTATION_WINDOW = "SemTileStereo segmentation"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("live", "pair"):
        command = sub.add_parser(name)
        command.add_argument("--stereo", type=Path, default=ModelPaths.stereo)
        command.add_argument("--head", type=Path, default=ModelPaths.head)
        command.add_argument("--semantic", type=Path, default=ModelPaths.semantic)
        command.add_argument("--encoder", type=Path, default=ModelPaths.encoder)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--display-max", type=float, default=80.0)
        command.add_argument("--rectification", type=Path, help="optional matching OpenCV stereoMap XML")
    live = sub.choices["live"]
    live.add_argument("--camera", default="/dev/video2")
    live.add_argument("--width", type=int, default=1280)
    live.add_argument("--height", type=int, default=480)
    live.add_argument("--left-view", choices=("left", "right"), default="left")
    live.add_argument("--rotate", choices=("none", "180"), default="none")
    pair = sub.choices["pair"]
    pair.add_argument("left", type=Path)
    pair.add_argument("right", type=Path)
    pair.add_argument("output", type=Path)
    pair.add_argument("--vkitti-camera", action="store_true", help="use VKITTI2 Camera 0 calibration for 1242×375 images")
    for key in ("fx", "fy", "cx", "cy", "baseline-m"):
        pair.add_argument(f"--{key}", type=float)
    return parser.parse_args(argv)


def _model_paths(args) -> ModelPaths:
    return ModelPaths(args.stereo, args.head, args.semantic, args.encoder)


def _camera_parameters(args, shape: tuple[int, int]) -> CameraParameters | None:
    keys = ("fx", "fy", "cx", "cy", "baseline_m")
    values = [getattr(args, key) for key in keys]
    if args.vkitti_camera and any(value is not None for value in values):
        raise ValueError("choose --vkitti-camera or explicit camera fields, not both")
    if args.vkitti_camera:
        if shape != (375, 1242):
            raise ValueError("VKITTI2 preset requires native 1242×375 image; supply custom calibration")
        return CameraParameters(725.0087, 725.0087, 620.5, 187.0, 0.532725)
    if any(value is not None for value in values):
        if any(value is None for value in values):
            raise ValueError("all of --fx --fy --cx --cy --baseline-m are required together")
        camera = CameraParameters(*values)
        camera.validate()
        return camera
    return None


def run_pair(args) -> Path:
    left, right = cv2.imread(str(args.left)), cv2.imread(str(args.right))
    if left is None or right is None:
        raise FileNotFoundError("left and right image paths must be readable")
    if left.shape != right.shape:
        raise ValueError("left and right images must have identical dimensions")
    camera = _camera_parameters(args, left.shape[:2])
    if args.rectification:
        maps = load_rectification(args.rectification, left.shape[:2])
        left, right = maps.apply(left, right)
    model = load_model(_model_paths(args), args.device)
    result = infer_pair(model, left, right, args.device)
    metadata = {"rectification": str(args.rectification) if args.rectification else None,
                "checkpoints": {name: str(getattr(args, name)) for name in ("stereo", "head", "semantic", "encoder")},
                "left_source": str(args.left), "right_source": str(args.right)}
    return save_result(result, args.output, camera=camera, metadata=metadata, display_max=args.display_max)


def should_stop(key: int, disparity_open: bool, segmentation_open: bool) -> bool:
    return key in (ord("q"), ord("Q"), 27) or not disparity_open or not segmentation_open


def run_live(args) -> None:
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open stereo camera {args.camera}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (args.width, args.height):
            raise RuntimeError(f"camera negotiated {actual}, requested {(args.width, args.height)}")
        model = load_model(_model_paths(args), args.device)
        maps = None
        cv2.namedWindow(DISPARITY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(SEGMENTATION_WINDOW, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera read failed")
            left, right = split_stereo(frame, args.left_view)
            if args.rotate == "180":
                left, right = cv2.rotate(left, cv2.ROTATE_180), cv2.rotate(right, cv2.ROTATE_180)
            if args.rectification:
                if maps is None:
                    maps = load_rectification(args.rectification, left.shape[:2])
                left, right = maps.apply(left, right)
            result = infer_pair(model, left, right, args.device)
            cv2.imshow(DISPARITY_WINDOW, colorize_disparity(result.disparity_px, args.display_max))
            cv2.imshow(SEGMENTATION_WINDOW, segmentation_overlay(result.left_rgb, result.class_id))
            key = cv2.waitKey(1) & 0xFF
            if should_stop(key, cv2.getWindowProperty(DISPARITY_WINDOW, cv2.WND_PROP_VISIBLE) >= 1,
                           cv2.getWindowProperty(SEGMENTATION_WINDOW, cv2.WND_PROP_VISIBLE) >= 1):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    import sys
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.mode == "pair":
            print(run_pair(args))
        else:
            run_live(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
