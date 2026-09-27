"""Live semantic segmentation for a side-by-side UVC stereo camera.

Run with the project environment:
    uv run python src/live_semantic_segmentation.py --model cityscapes-n
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "segmentation"
FASTFS_MODEL_DIRECTORY = REPOSITORY_ROOT / "models" / "stereo" / "fast_foundationstereo"
FASTFS_SOURCE_DIRECTORY = REPOSITORY_ROOT / "external_models" / "Fast-FoundationStereo"
TICOSS_MODEL_PATH = REPOSITORY_ROOT / "models" / "stereo" / "ticoss" / "ticoss_kitti.pth"
TICOSS_SOURCE_DIRECTORY = REPOSITORY_ROOT / "external_models" / "TiCoSS"
MODEL_PRESETS = {
    "cityscapes-n": MODEL_DIRECTORY / "yolo26n-sem-cityscapes.pt",
    "cityscapes-s": MODEL_DIRECTORY / "yolo26s-sem-cityscapes.pt",
    "ade20k-n": MODEL_DIRECTORY / "yolo26n-sem-ade20k.pt",
    "ade20k-s": MODEL_DIRECTORY / "yolo26s-sem-ade20k.pt",
}
STEREO_MODEL_PRESETS = {
    "fastfs-23-36-37": FASTFS_MODEL_DIRECTORY / "23-36-37" / "model_best_bp2_serialize.pth",
    "fastfs-20-26-39": FASTFS_MODEL_DIRECTORY / "20-26-39" / "model_best_bp2_serialize.pth",
    "fastfs-20-30-48": FASTFS_MODEL_DIRECTORY / "20-30-48" / "model_best_bp2_serialize.pth",
    "fastfs-15-44-51": FASTFS_MODEL_DIRECTORY / "15-44-51" / "model_best_bp2_serialize.pth",
}
WINDOW_NAME = "Stereo Semantic Segmentation"


def crop_stereo_frame(frame: np.ndarray, side: str) -> np.ndarray:
    """Return one eye from an equal-width side-by-side stereo frame."""
    if frame.ndim < 2:
        raise ValueError("camera frame must have at least two dimensions")
    width = frame.shape[1]
    if width % 2:
        raise ValueError("side-by-side stereo input must have an even width")

    midpoint = width // 2
    if side == "left":
        return frame[:, :midpoint]
    if side == "right":
        return frame[:, midpoint:]
    raise ValueError(f"unsupported stereo side: {side}")


def split_stereo_frame(frame: np.ndarray, swap_stereo: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return the explicit left/right pair from an equal-width SBS frame."""
    left = crop_stereo_frame(frame, "left")
    right = crop_stereo_frame(frame, "right")
    return (right, left) if swap_stereo else (left, right)


def should_exit(key: int, window_visible: bool) -> bool:
    """Return whether a keypress or window close requests loop termination."""
    return not window_visible or (key & 0xFF) in (ord("q"), ord("Q"), 27)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="cityscapes-n",
        help=(
            "Preset: cityscapes-n, cityscapes-s, ade20k-n, ade20k-s, or none; "
            "or path to a YOLO26 semantic checkpoint. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--stereo-model",
        choices=("none", *STEREO_MODEL_PRESETS),
        default="none",
        help="Optional disparity model shown beside segmentation. Default: %(default)s.",
    )
    parser.add_argument(
        "--joint-model",
        choices=("none", "ticoss"),
        default="none",
        help="One joint semantic-stereo model. Default: %(default)s.",
    )
    parser.add_argument(
        "--fastfs-iters",
        type=int,
        default=4,
        help="FastFS recurrent refinement iterations. Default: %(default)s.",
    )
    parser.add_argument(
        "--fastfs-max-disp",
        type=int,
        default=192,
        help="FastFS maximum disparity in pixels. Default: %(default)s.",
    )
    parser.add_argument(
        "--swap-stereo",
        action="store_true",
        help="Explicitly reverse the SBS left/right order (only after checking rectification).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=2,
        help="V4L2 camera index. Default: 2 (/dev/video2, the CCB stereo camera).",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Eye selected from the side-by-side camera frame. Default: left.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2560,
        help="Requested combined stereo-frame width. Default: 2560.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested combined stereo-frame height. Default: 720.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size; lower values improve live FPS. Default: 640.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device, for example 0, cpu, or cuda:0. Default: 0.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Segmentation overlay opacity in [0, 1]. Default: 0.45.",
    )
    arguments = parser.parse_args(argv)
    if not 0.0 <= arguments.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if arguments.fastfs_iters < 1:
        parser.error("--fastfs-iters must be positive")
    if arguments.fastfs_max_disp < 4 or arguments.fastfs_max_disp % 4:
        parser.error("--fastfs-max-disp must be a multiple of 4 and at least 4")
    selected_pipelines = sum(
        (arguments.model != "none", arguments.stereo_model != "none", arguments.joint_model != "none")
    )
    if selected_pipelines != 1:
        parser.error("select exactly one pipeline: --model, --stereo-model, or --joint-model")
    if (arguments.stereo_model != "none" or arguments.joint_model != "none") and arguments.side != "left":
        parser.error("stereo pipelines are aligned with the left eye; use --side left")
    return arguments


def resolve_model(model_name: str) -> Path | None:
    """Resolve a named checkpoint preset and ensure it is locally available."""
    if model_name == "none":
        return None
    model_path = MODEL_PRESETS.get(model_name, Path(model_name).expanduser())
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Download it into models/segmentation first."
        )
    return model_path


def resolve_stereo_model(model_name: str) -> Path | None:
    """Resolve an optional Fast-FoundationStereo checkpoint preset."""
    if model_name == "none":
        return None
    model_path = STEREO_MODEL_PRESETS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"FastFS checkpoint not found: {model_path}")
    if not FASTFS_SOURCE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"FastFS source checkout not found: {FASTFS_SOURCE_DIRECTORY}. "
            "Clone the official Fast-FoundationStereo repository there."
        )
    return model_path


def resolve_joint_model(model_name: str) -> Path | None:
    """Resolve the checkpoint for a joint segmentation-and-stereo pipeline."""
    if model_name == "none":
        return None
    if not TICOSS_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"TiCoSS checkpoint not found: {TICOSS_MODEL_PATH}. "
            "Download the official inference checkpoint into that path."
        )
    if not TICOSS_SOURCE_DIRECTORY.is_dir():
        raise FileNotFoundError(f"TiCoSS source checkout not found: {TICOSS_SOURCE_DIRECTORY}")
    return TICOSS_MODEL_PATH


class FastFoundationStereoRunner:
    """Minimal live-inference adapter for upstream serialized FastFS checkpoints."""

    def __init__(self, checkpoint: Path, iterations: int, max_disparity: int) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FastFS live inference requires a CUDA-enabled PyTorch installation")
        source_path = str(FASTFS_SOURCE_DIRECTORY)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        from core.utils.utils import InputPadder

        self.torch = torch
        self.input_padder = InputPadder
        self.iterations = iterations
        self.device = torch.device("cuda:0")
        self.model = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.args.valid_iters = iterations
        self.model.args.max_disp = max_disparity
        self.model.args.mixed_precision = True
        self.model.to(self.device).eval()

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Produce a pixel-disparity map for an already-rectified BGR pair."""
        import cv2

        if left_bgr.shape != right_bgr.shape:
            raise ValueError("FastFS requires left and right frames of identical shape")
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        left = self.torch.from_numpy(left_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        right = self.torch.from_numpy(right_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        padder = self.input_padder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with self.torch.inference_mode(), self.torch.amp.autocast("cuda", dtype=self.torch.float16):
            disparity = self.model(
                left,
                right,
                iters=self.iterations,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disparity = padder.unpad(disparity).squeeze().float().cpu().numpy()
        return np.asarray(disparity, dtype=np.float32)


class TiCoSSRunner:
    """Adapter for the official TiCoSS joint semantic/stereo checkpoint."""

    def __init__(self, checkpoint: Path) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TiCoSS live inference requires CUDA-enabled PyTorch")
        source_path = str(TICOSS_SOURCE_DIRECTORY)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        from core.JLNet import JLnet
        from core.utils.utils import InputPadder

        self.torch, self.input_padder = torch, InputPadder
        self.device = torch.device("cuda:0")
        args = argparse.Namespace(
            mixed_precision=True, hidden_dims=[128] * 3, corr_implementation="reg",
            shared_backbone=False, corr_levels=4, corr_radius=4, n_downsample=2,
            slow_fast_gru=False, n_gru_layers=3, scale=1,
        )
        self.model = JLnet(args)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        if all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        left = self.torch.from_numpy(left_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        right = self.torch.from_numpy(right_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        padder = self.input_padder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with self.torch.inference_mode(), self.torch.amp.autocast("cuda", dtype=self.torch.float16):
            flow, segmentation = self.model(left, right, left, right, threshold=1.0, iters=32, test_mode=True)
        disparity = -padder.unpad(flow).squeeze().float().cpu().numpy()
        labels = padder.unpad(segmentation).argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return np.asarray(disparity, dtype=np.float32), labels


def class_palette() -> np.ndarray:
    """Create a fixed, high-contrast BGR palette for up to 256 semantic IDs."""
    ids = np.arange(256, dtype=np.uint32)
    return np.column_stack(
        (
            (37 * ids + 29) % 256,
            (17 * ids + 101) % 256,
            (97 * ids + 53) % 256,
        )
    ).astype(np.uint8)


def semantic_mask_from_result(result: Any, target_shape: tuple[int, int]) -> np.ndarray:
    """Extract and resize Ultralytics' dense class map to the displayed frame."""
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError(
            "The selected model did not return a semantic mask. "
            "Use a YOLO26 '-sem' checkpoint, not a '-seg' instance model."
        )

    mask: Any = semantic_mask.data
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise RuntimeError(f"Expected a two-dimensional class map, got {mask.shape}")

    import cv2

    height, width = target_shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def render_overlay(frame: np.ndarray, class_map: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a class-coloured semantic map over a BGR camera frame."""
    import cv2

    colour_map = class_palette()[class_map]
    return cv2.addWeighted(frame, 1.0 - alpha, colour_map, alpha, 0.0)


def render_disparity(disparity: np.ndarray) -> np.ndarray:
    """Create a robust false-colour view of a positive disparity map."""
    import cv2

    valid = np.isfinite(disparity) & (disparity > 0)
    visualization = np.zeros(disparity.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(disparity[valid], (2, 98))
        if high > low:
            scaled = (disparity - low) * 255.0 / (high - low)
            visualization[valid] = np.clip(scaled[valid], 0, 255).astype(np.uint8)
    return cv2.applyColorMap(visualization, cv2.COLORMAP_TURBO)


def open_camera(camera_index: int, width: int, height: int, fps: int) -> Any:
    """Open the requested UVC camera and request the side-by-side MJPEG mode."""
    import cv2

    backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera {camera_index}")
    return capture


def window_is_visible(cv2: Any) -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def run(arguments: argparse.Namespace) -> int:
    """Start capture, inference, and display until an exit condition occurs."""
    import cv2
    model_path = resolve_model(arguments.model)
    stereo_model_path = resolve_stereo_model(arguments.stereo_model)
    joint_model_path = resolve_joint_model(arguments.joint_model)
    model = None
    if model_path is not None:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    stereo_runner = (
        FastFoundationStereoRunner(
            stereo_model_path,
            arguments.fastfs_iters,
            arguments.fastfs_max_disp,
        )
        if stereo_model_path is not None
        else None
    )
    joint_runner = TiCoSSRunner(joint_model_path) if joint_model_path is not None else None
    capture = open_camera(arguments.camera, arguments.width, arguments.height, arguments.fps)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    last_timestamp = time.perf_counter()
    fps = 0.0

    print(
        f"Camera {arguments.camera}; requesting {arguments.width}x{arguments.height} MJPEG; "
        f"displaying {arguments.side} eye. Press q, Esc, or close the window to stop."
    )
    try:
        while True:
            ok, stereo_frame = capture.read()
            if not ok or stereo_frame is None:
                raise RuntimeError("Camera frame read failed")

            left_frame, right_frame = split_stereo_frame(stereo_frame, arguments.swap_stereo)
            eye_frame = left_frame if arguments.side == "left" else right_frame
            if model is not None:
                result = model(eye_frame, imgsz=arguments.imgsz, device=arguments.device, verbose=False)[0]
                class_map = semantic_mask_from_result(result, eye_frame.shape[:2])
                display = render_overlay(eye_frame, class_map, arguments.alpha)
            else:
                display = eye_frame.copy()
            if stereo_runner is not None:
                disparity = stereo_runner.infer(left_frame, right_frame)
                display = np.hstack((display, render_disparity(disparity)))
            if joint_runner is not None:
                disparity, labels = joint_runner.infer(left_frame, right_frame)
                display = np.hstack((render_overlay(left_frame, labels, arguments.alpha), render_disparity(disparity)))

            now = time.perf_counter()
            elapsed = max(now - last_timestamp, 1e-6)
            fps = 0.9 * fps + 0.1 / elapsed if fps else 1.0 / elapsed
            last_timestamp = now
            cv2.putText(
                display,
                f"{model_path.name if model_path else arguments.stereo_model if stereo_model_path else arguments.joint_model} | "
                f"{arguments.side} | {fps:.1f} FPS | q/Esc: quit",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1)
            if should_exit(key, window_is_visible(cv2)):
                break
    except KeyboardInterrupt:
        print("Interrupted; closing camera.")
        return 130
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
