"""Mouse-interactive Open3D-rendered viewport embedded in Qt Widgets."""

import math

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget


class Open3DViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(800, 500)
        self._cloud = o3d.geometry.PointCloud()
        self._center = np.zeros(3)
        self._distance = 10.0
        self._yaw, self._pitch = 0.0, -0.25
        self._last = None
        self._image = None
        self._render()

    def set_cloud(self, points: np.ndarray, colors: np.ndarray) -> None:
        self._renderer.scene.clear_geometry()
        self._cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        self._cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        if len(points):
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2.0
            self._renderer.scene.add_geometry("cloud", self._cloud, material)
            self._center = np.median(points, axis=0)
            self._distance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 1.2, 1.0)
        self._render()

    def reset_view(self) -> None:
        if len(self._cloud.points):
            points = np.asarray(self._cloud.points)
            self._center = np.median(points, axis=0)
            self._distance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 1.2, 1.0)
        self._yaw, self._pitch = 0.0, -0.25
        self._render()

    def _render(self) -> None:
        direction = np.array([math.sin(self._yaw) * math.cos(self._pitch),
                              math.sin(self._pitch), math.cos(self._yaw) * math.cos(self._pitch)])
        eye = self._center - direction * self._distance
        self._renderer.scene.camera.look_at(self._center, eye, [0, -1, 0])
        image = np.asarray(self._renderer.render_to_image())
        height, width = image.shape[:2]
        fmt = QImage.Format_RGB888 if image.shape[2] == 3 else QImage.Format_RGBA8888
        self._image = QImage(image.data, width, height, image.strides[0], fmt).copy()
        self.update()

    def paintEvent(self, event):
        if self._image is not None:
            painter = QPainter(self)
            painter.drawImage(self.rect(), self._image)

    def mousePressEvent(self, event):
        self._last = event.position()

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        if event.buttons() & Qt.LeftButton:
            self._yaw += delta.x() * 0.007
            self._pitch = float(np.clip(self._pitch - delta.y() * 0.007, -1.45, 1.45))
        elif event.buttons() & Qt.RightButton:
            self._center += np.array([-delta.x(), delta.y(), 0]) * self._distance * 0.002
        self._render()

    def mouseReleaseEvent(self, event):
        self._last = None

    def wheelEvent(self, event):
        self._distance = float(np.clip(self._distance * (0.88 ** (event.angleDelta().y() / 120)), 0.05, 1e5))
        self._render()
