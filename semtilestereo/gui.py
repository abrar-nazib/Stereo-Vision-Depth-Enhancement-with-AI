"""Qt6 SemTileStereo result reviewer with an embedded Open3D point-cloud view."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from experiments.vkitti2.semantic_data import CLASS_NAMES
from semtilestereo.core import ModelPaths, infer_pair, load_model
from semtilestereo.geometry import VKITTI_CAMERA, points_from_result
from semtilestereo.results import CameraParameters, load_bundle, save_result
from semtilestereo.viewport import Open3DViewport


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/semtilestereo"


class InferenceWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, left: Path, right: Path, model, paths: ModelPaths, device: str):
        super().__init__()
        self.left, self.right, self.model, self.paths, self.device = left, right, model, paths, device

    def run(self):
        try:
            left, right = cv2.imread(str(self.left)), cv2.imread(str(self.right))
            if left is None or right is None or left.shape != right.shape:
                raise ValueError("matching readable left/right images are required")
            model = self.model or load_model(self.paths, self.device)
            result = infer_pair(model, left, right, self.device)
            self.succeeded.emit(result, model)
        except Exception as exc:
            self.failed.emit(f"Inference failed: {exc}")


class ReviewWindow(QMainWindow):
    def __init__(self, *, viewport=None):
        super().__init__()
        self.setWindowTitle("SemTileStereo · semantic point-cloud reviewer")
        self.resize(1320, 850)
        self.viewport = viewport if viewport is not None else Open3DViewport()
        self.result = None
        self.metadata = {}
        self.model = None
        self.worker = None
        self._pending_camera = None
        self._filter_error = False
        self.class_checks = {}
        self._build_ui()

    def _build_ui(self):
        shell = QWidget()
        row = QHBoxLayout(shell)
        row.addWidget(self.viewport, 3)
        panel = QWidget()
        form = QVBoxLayout(panel)

        open_button = QPushButton("Open result…")
        open_button.clicked.connect(self._choose_result)
        form.addWidget(open_button)
        self.examples = QComboBox()
        self.examples.addItem("Example results…", None)
        for path in sorted(EXAMPLES.glob("*/result.npz")):
            self.examples.addItem(path.parent.name, path)
        self.examples.currentIndexChanged.connect(self._open_example)
        form.addWidget(self.examples)

        self.left_field = QLineEdit()
        self.right_field = QLineEdit()
        self.output_field = QLineEdit(str(EXAMPLES / "user_inference"))
        for label, field in (("Left image", self.left_field), ("Right image", self.right_field),
                             ("Output folder", self.output_field)):
            form.addWidget(QLabel(label))
            input_row = QHBoxLayout()
            input_row.addWidget(field)
            if field is not self.output_field:
                browse = QPushButton("Browse…")
                browse.clicked.connect(lambda _checked=False, target=field: self._choose_image(target))
                input_row.addWidget(browse)
                if field is self.left_field:
                    self.left_browse = browse
                else:
                    self.right_browse = browse
            form.addLayout(input_row)
        infer_button = QPushButton("Infer pair")
        infer_button.clicked.connect(lambda: self.infer_pair(Path(self.left_field.text()), Path(self.right_field.text())))
        form.addWidget(infer_button)

        self.camera_fields = {}
        camera_form = QFormLayout()
        for name in ("fx", "fy", "cx", "cy", "baseline_m"):
            field = QLineEdit()
            self.camera_fields[name] = field
            camera_form.addRow(name if name != "baseline_m" else "baseline (m)", field)
        form.addLayout(camera_form)
        self.fx_field = self.camera_fields["fx"]
        self.use_camera_for_inference = QCheckBox("Apply these camera parameters to next inference")
        form.addWidget(self.use_camera_for_inference)
        preset = QPushButton("VKITTI2 Camera 0 preset (1242×375)")
        preset.clicked.connect(self._preset_clicked)
        form.addWidget(preset)
        self.min_depth_field = QLineEdit("0.1")
        self.max_depth_field = QLineEdit("80")
        self.stride_field = QLineEdit("2")
        for label, field in (("Minimum depth (m)", self.min_depth_field),
                             ("Maximum depth (m)", self.max_depth_field),
                             ("Preview stride", self.stride_field)):
            form.addWidget(QLabel(label))
            form.addWidget(field)
        self.semantic_colors = QCheckBox("Use semantic colors")
        self.semantic_colors.toggled.connect(self.apply_filters)
        form.addWidget(self.semantic_colors)
        update_button = QPushButton("Update point cloud")
        update_button.clicked.connect(self.apply_filters)
        form.addWidget(update_button)
        reset_button = QPushButton("Reset view")
        reset_button.clicked.connect(self.viewport.reset_view)
        form.addWidget(reset_button)
        form.addWidget(QLabel("Classes in result"))
        self.class_panel = QWidget()
        self.class_layout = QVBoxLayout(self.class_panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.class_panel)
        form.addWidget(scroll, 1)
        self.status = QLabel("Open a result or infer a stereo pair. Camera calibration is required for metric depth.")
        self.status.setWordWrap(True)
        form.addWidget(self.status)
        row.addWidget(panel, 1)
        self.setCentralWidget(shell)

    def _choose_result(self):
        name, _ = QFileDialog.getOpenFileName(self, "Open SemTileStereo result", str(EXAMPLES), "Result bundles (*.npz)")
        if name:
            try:
                self.open_result(Path(name))
            except Exception as exc:
                self.show_error(str(exc))

    def _choose_image(self, target: QLineEdit):
        name, _ = QFileDialog.getOpenFileName(self, "Select stereo image", str(ROOT), "Images (*.jpg *.jpeg *.png)")
        if name:
            target.setText(name)

    def _open_example(self, index):
        path = self.examples.itemData(index)
        if path is not None:
            try:
                self.open_result(Path(path))
            except Exception as exc:
                self.show_error(str(exc))

    def _preset_clicked(self):
        try:
            self.use_vkitti_preset()
        except ValueError as exc:
            self.show_error(str(exc))

    def use_vkitti_preset(self):
        if self.result is None or self.result.disparity_px.shape != (375, 1242):
            raise ValueError("VKITTI preset requires a loaded native 1242×375 result")
        self._set_camera(VKITTI_CAMERA)
        self.apply_filters()

    def _set_camera(self, camera: CameraParameters):
        for name, field in self.camera_fields.items():
            field.setText(str(getattr(camera, name)))

    def _read_camera(self) -> CameraParameters:
        try:
            camera = CameraParameters(*(float(self.camera_fields[key].text()) for key in
                                        ("fx", "fy", "cx", "cy", "baseline_m")))
            camera.validate()
            return camera
        except ValueError as exc:
            raise ValueError("Enter valid fx, fy, cx, cy and baseline (metres) for this camera") from exc

    def open_result(self, path: Path):
        result, metadata = load_bundle(path)
        self._set_result(result, metadata)
        if metadata.get("camera") is None and not metadata.get("example_vkitti"):
            self.status.setText(f"Loaded {path} · enter camera calibration to view metric points")
        elif not self._filter_error:
            self.status.setText(f"Loaded {path} · {result.disparity_px.shape[1]}×{result.disparity_px.shape[0]}")

    def _set_result(self, result, metadata):
        self.result, self.metadata = result, metadata
        self.viewport.set_cloud(np.empty((0, 3)), np.empty((0, 3)))
        self.use_camera_for_inference.setChecked(False)
        for box in self.class_checks.values():
            box.deleteLater()
        self.class_checks.clear()
        for class_id in np.unique(result.class_id):
            class_id = int(class_id)
            box = QCheckBox(CLASS_NAMES[class_id])
            box.setChecked(True)
            box.toggled.connect(self.apply_filters)
            self.class_layout.addWidget(box)
            self.class_checks[class_id] = box
        camera_data = metadata.get("camera")
        if camera_data:
            self._set_camera(CameraParameters(**camera_data))
        elif result.disparity_px.shape == (375, 1242) and metadata.get("example_vkitti"):
            self._set_camera(VKITTI_CAMERA)
        else:
            for field in self.camera_fields.values():
                field.clear()
        self.apply_filters()

    def apply_filters(self):
        if self.result is None:
            return
        try:
            camera = self._read_camera()
            visible = {class_id for class_id, box in self.class_checks.items() if box.isChecked()}
            points, colors, counts = points_from_result(
                self.result, camera, visible_classes=visible,
                min_depth_m=float(self.min_depth_field.text()),
                max_depth_m=float(self.max_depth_field.text()),
                stride=int(self.stride_field.text()),
                semantic_colors=self.semantic_colors.isChecked())
            for class_id, box in self.class_checks.items():
                box.setText(f"{CLASS_NAMES[class_id]} ({counts.get(class_id, 0):,})")
            self.viewport.set_cloud(points, colors)
            self._filter_error = False
            self.status.setText(f"Showing {len(points):,} points · {len(visible)} classes")
        except (ValueError, TypeError) as exc:
            self.viewport.set_cloud(np.empty((0, 3)), np.empty((0, 3)))
            self._filter_error = True
            self.show_error(str(exc))

    def infer_pair(self, left_path: Path, right_path: Path):
        if self.worker is not None and self.worker.isRunning():
            self.show_error("Inference is already running")
            return
        try:
            self._pending_camera = self._read_camera() if self.use_camera_for_inference.isChecked() else None
        except ValueError as exc:
            self.show_error(str(exc))
            return
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.worker = InferenceWorker(left_path, right_path, self.model, ModelPaths(), device)
        self.worker.succeeded.connect(self._inference_done)
        self.worker.failed.connect(self.show_error)
        self.status.setText("Inferring stereo pair…")
        self.worker.start()

    def _inference_done(self, result, model):
        self.model = model
        try:
            path = save_result(result, Path(self.output_field.text()), camera=self._pending_camera,
                               metadata={"checkpoints": {key: str(getattr(ModelPaths(), key)) for key in
                                                         ("stereo", "head", "semantic", "encoder")}}, display_max=80)
            self.open_result(path)
        except Exception as exc:
            self.show_error(f"Could not save inference: {exc}")
        finally:
            self._pending_camera = None

    def show_error(self, message: str):
        self.status.setText(f"Error: {message}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ReviewWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
