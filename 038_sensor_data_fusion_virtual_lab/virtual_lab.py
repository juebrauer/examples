"""
Stereo Train Data Collector (Qt + OpenGL)

- No physics engine.
- Scene rendered with OpenGL.
- Two stereo cameras capture images during a scripted drive.
- A minimal UI shows both camera previews and one collect button.
"""

from __future__ import annotations

import json
import math
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

MATPLOTLIB_IMPORT_ERROR = ""
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception as exc:  # noqa: BLE001
    MATPLOTLIB_IMPORT_ERROR = str(exc)

TORCH_IMPORT_ERROR = ""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception as exc:  # noqa: BLE001
    TORCH_IMPORT_ERROR = str(exc)

OPENGL_IMPORT_ERROR = ""
try:
    from OpenGL.GL import (  # type: ignore[import-not-found]
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_COMPONENT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_FLOAT,
        GL_LEQUAL,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGBA,
        GL_SMOOTH,
        GL_UNSIGNED_BYTE,
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glDepthFunc,
        glEnable,
        glEnd,
        glLoadIdentity,
        glMatrixMode,
        glPopMatrix,
        glPushMatrix,
        glReadPixels,
        glRotatef,
        glScalef,
        glShadeModel,
        glTranslatef,
        glVertex3f,
        glViewport,
    )
    from OpenGL.GLU import gluLookAt, gluPerspective  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    OPENGL_IMPORT_ERROR = str(exc)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class CameraPose:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]


@dataclass(frozen=True)
class BoxObject:
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    yaw_deg: float
    color: tuple[float, float, float]


class StereoPreview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(f"{title}: no image")
        self._title = title
        self.setFixedSize(360, 220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#171717; color:#e8e8e8; border:1px solid #3a3a3a;")

    def set_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    def reset(self) -> None:
        self.clear()
        self.setText(f"{self._title}: no image")


def qimage_to_rgb_array(image: QImage) -> np.ndarray:
    rgb_image = image.convertToFormat(QImage.Format_RGB888)
    width = rgb_image.width()
    height = rgb_image.height()
    bytes_per_line = rgb_image.bytesPerLine()
    data = np.frombuffer(rgb_image.bits(), dtype=np.uint8, count=bytes_per_line * height)
    return data.reshape((height, bytes_per_line))[:, : width * 3].reshape((height, width, 3)).copy()


class StereoDepthDataset(Dataset):
    def __init__(self, run_dir: Path, records: list[dict], target_size: tuple[int, int]) -> None:
        self.run_dir = run_dir
        self.records = records
        self.target_size = target_size  # (height, width)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        left_path = self.run_dir / record["left_image"]
        right_path = self.run_dir / record["right_image"]
        depth_path = self.run_dir / record["depth_array"]

        left_qimg = QImage(str(left_path))
        right_qimg = QImage(str(right_path))
        if left_qimg.isNull() or right_qimg.isNull():
            raise RuntimeError(f"Could not load RGB images for sample {index}.")

        left = qimage_to_rgb_array(left_qimg).astype(np.float32) / 255.0
        right = qimage_to_rgb_array(right_qimg).astype(np.float32) / 255.0

        depth_u16 = np.load(depth_path)
        if depth_u16.ndim == 2:
            depth_u16 = depth_u16[:, :, None]
        depth_u16 = depth_u16.astype(np.float32)

        depth_min_m = float(record.get("depth_min_m", 0.0))
        depth_max_m = float(record.get("depth_max_m", 50.0))
        depth_span = max(1e-6, depth_max_m - depth_min_m)
        depth_m = (depth_u16 / 65535.0) * depth_span + depth_min_m
        depth_norm = np.clip((depth_m - depth_min_m) / depth_span, 0.0, 1.0)

        left_t = torch.from_numpy(left).permute(2, 0, 1)
        right_t = torch.from_numpy(right).permute(2, 0, 1)
        x = torch.cat([left_t, right_t], dim=0)
        y = torch.from_numpy(depth_norm).permute(2, 0, 1)

        x = F.interpolate(x.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        return x, y


class StereoDepthFusionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(kernel_size=2)
        self.mid = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        m = self.mid(self.down(e1))
        u = self.up(m)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([e1, u], dim=1))
        return torch.sigmoid(y)


class OpenGLWorldWidget(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(960, 620)
        self._objects = self._generate_world_objects(320)
        self._main_view_angle = 0.0

    def _generate_world_objects(self, count: int) -> list[BoxObject]:
        objects: list[BoxObject] = []
        for _ in range(count):
            x = random.uniform(-24.0, 24.0)
            y = random.uniform(-24.0, 24.0)
            z = random.uniform(0.15, 3.4)
            sx = random.uniform(0.2, 1.1)
            sy = random.uniform(0.2, 1.1)
            sz = random.uniform(0.2, 2.8)
            yaw = random.uniform(0.0, 360.0)
            color = (
                random.uniform(0.15, 0.95),
                random.uniform(0.15, 0.95),
                random.uniform(0.15, 0.95),
            )
            objects.append(BoxObject((x, y, z), (sx, sy, sz), yaw, color))
        return objects

    def advance_main_view(self) -> None:
        self._main_view_angle = (self._main_view_angle + 0.25) % 360.0
        self.update()

    def initializeGL(self) -> None:
        glClearColor(0.05, 0.07, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_SMOOTH)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self) -> None:
        width = max(1, self.width())
        height = max(1, self.height())

        eye_x = 33.0 * math.cos(math.radians(self._main_view_angle))
        eye_y = 33.0 * math.sin(math.radians(self._main_view_angle))
        eye = (eye_x, eye_y, 21.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)

        self._render_scene(
            width=width,
            height=height,
            fov_deg=58.0,
            camera=CameraPose(eye=eye, target=target, up=up),
        )

    def _draw_unit_cube(self) -> None:
        glBegin(GL_QUADS)

        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)

        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)

        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)

        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)

        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)

        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)

        glEnd()

    def _draw_ground_tiles(self, half_tiles: int = 40, tile_size: float = 1.2) -> None:
        for tx in range(-half_tiles, half_tiles):
            for ty in range(-half_tiles, half_tiles):
                # Subtle checkerboard pattern for spatial orientation.
                if (tx + ty) % 2 == 0:
                    glColor3f(0.11, 0.14, 0.19)
                else:
                    glColor3f(0.07, 0.1, 0.15)

                x0 = tx * tile_size
                y0 = ty * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                glBegin(GL_QUADS)
                glVertex3f(x0, y0, 0.0)
                glVertex3f(x1, y0, 0.0)
                glVertex3f(x1, y1, 0.0)
                glVertex3f(x0, y1, 0.0)
                glEnd()

    def _draw_world(self) -> None:
        self._draw_ground_tiles()
        for obj in self._objects:
            glPushMatrix()
            glTranslatef(obj.position[0], obj.position[1], obj.position[2])
            glRotatef(obj.yaw_deg, 0.0, 0.0, 1.0)
            glScalef(obj.scale[0], obj.scale[1], obj.scale[2])
            glColor3f(obj.color[0], obj.color[1], obj.color[2])
            self._draw_unit_cube()
            glPopMatrix()

    def _render_scene(self, width: int, height: int, fov_deg: float, camera: CameraPose) -> None:
        glViewport(0, 0, max(1, width), max(1, height))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov_deg, width / max(1.0, float(height)), 0.05, 250.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            camera.eye[0],
            camera.eye[1],
            camera.eye[2],
            camera.target[0],
            camera.target[1],
            camera.target[2],
            camera.up[0],
            camera.up[1],
            camera.up[2],
        )
        self._draw_world()

    def render_camera_image(self, camera: CameraPose, width: int, height: int, fov_deg: float = 72.0) -> QImage:
        self.makeCurrent()
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo = QOpenGLFramebufferObject(width, height, fmt)
        fbo.bind()

        self._render_scene(width=width, height=height, fov_deg=fov_deg, camera=camera)
        pixel_data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = QImage(pixel_data, width, height, 4 * width, QImage.Format_RGBA8888).mirrored(False, True).copy()

        fbo.release()
        self.doneCurrent()
        return image

    def render_depth_array(
        self,
        camera: CameraPose,
        width: int,
        height: int,
        fov_deg: float = 72.0,
        near_plane: float = 0.05,
        far_plane: float = 250.0,
    ) -> np.ndarray:
        self.makeCurrent()
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        fbo = QOpenGLFramebufferObject(width, height, fmt)
        fbo.bind()

        self._render_scene(width=width, height=height, fov_deg=fov_deg, camera=camera)
        depth_data = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
        if hasattr(depth_data, "tobytes"):
            depth_bytes = depth_data.tobytes()
        else:
            depth_bytes = bytes(depth_data)
        depth_values = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
        z_ndc = 2.0 * depth_values - 1.0
        denom = far_plane + near_plane - z_ndc * (far_plane - near_plane)
        linear_depth = (2.0 * near_plane * far_plane) / np.maximum(1e-6, denom)
        linear_depth = np.clip(linear_depth, near_plane, far_plane).astype(np.float32)
        linear_depth = np.flipud(linear_depth)

        fbo.release()
        self.doneCurrent()
        return linear_depth[:, :, None]

    @staticmethod
    def depth_array_to_colormap_image(depth_array: np.ndarray, min_depth_m: float, max_depth_m: float) -> QImage:
        depth = depth_array[:, :, 0]
        depth_span = max(1e-6, max_depth_m - min_depth_m)
        t = np.clip((depth - min_depth_m) / depth_span, 0.0, 1.0)

        blue = np.array([59.0, 76.0, 192.0], dtype=np.float32)
        mid = np.array([221.0, 221.0, 221.0], dtype=np.float32)
        red = np.array([180.0, 4.0, 38.0], dtype=np.float32)

        rgb = np.empty((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
        low_mask = t <= 0.5
        high_mask = ~low_mask

        t_low = (t[low_mask] * 2.0)[:, None]
        t_high = ((t[high_mask] - 0.5) * 2.0)[:, None]

        rgb[low_mask] = blue[None, :] * (1.0 - t_low) + mid[None, :] * t_low
        rgb[high_mask] = mid[None, :] * (1.0 - t_high) + red[None, :] * t_high

        rgba = np.empty((depth.shape[0], depth.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 3] = 255
        return QImage(rgba.data, depth.shape[1], depth.shape[0], 4 * depth.shape[1], QImage.Format_RGBA8888).copy()

    @staticmethod
    def stereo_from_rig(
        rig_pos: tuple[float, float, float],
        forward_xy: tuple[float, float],
        baseline: float,
        look_distance: float,
    ) -> tuple[CameraPose, CameraPose]:
        fx, fy = forward_xy
        norm = math.hypot(fx, fy)
        if norm < 1e-8:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = fx / norm, fy / norm

        left_vec = (-fy, fx)
        half = 0.5 * baseline

        left_eye = (rig_pos[0] + half * left_vec[0], rig_pos[1] + half * left_vec[1], rig_pos[2])
        right_eye = (rig_pos[0] - half * left_vec[0], rig_pos[1] - half * left_vec[1], rig_pos[2])

        left_target = (left_eye[0] + look_distance * fx, left_eye[1] + look_distance * fy, rig_pos[2] - 0.04)
        right_target = (right_eye[0] + look_distance * fx, right_eye[1] + look_distance * fy, rig_pos[2] - 0.04)

        up = (0.0, 0.0, 1.0)
        return CameraPose(left_eye, left_target, up), CameraPose(right_eye, right_target, up)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stereo Train Data Collector - OpenGL")
        self.resize(1500, 900)

        self.world_view = OpenGLWorldWidget()
        self.left_preview = StereoPreview("Left")
        self.right_preview = StereoPreview("Right")
        self.depth_preview = StereoPreview("Depth")
        self.gt_depth_preview = StereoPreview("GT depth")
        self.pred_depth_preview = StereoPreview("Pred depth")
        self.diff_depth_preview = StereoPreview("Abs diff")
        self.collect_button = QPushButton("Collect train data")
        self.train_model_button = QPushButton("Train sensor data fusion model")
        self.test_model_button = QPushButton("Test trained depth model")
        self.frame_count_label = QLabel("Frames to collect")
        self.frame_count_spinbox = QSpinBox()
        self.frame_count_spinbox.setRange(1, 200000)
        self.frame_count_spinbox.setValue(1000)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        self.training_eta_label = QLabel("ETA: -")
        self.training_epoch_label = QLabel("Epochs: -")
        self.training_batch_label = QLabel("Mini-batches: -")
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.loss_canvas = None
        self.loss_figure = None
        self.loss_ax = None
        if not MATPLOTLIB_IMPORT_ERROR:
            self.loss_figure = Figure(figsize=(4.0, 2.8), tight_layout=True)
            self.loss_ax = self.loss_figure.add_subplot(111)
            self.loss_canvas = FigureCanvas(self.loss_figure)
            self.loss_canvas.setMinimumSize(420, 280)
            self._update_loss_plot()
        else:
            self.loss_placeholder_label = QLabel(
                "Matplotlib not available.\n"
                "Install with: pip install matplotlib"
            )
            self.loss_placeholder_label.setWordWrap(True)
            self.loss_placeholder_label.setStyleSheet("color:#bbbbbb;")

        self.collect_button.clicked.connect(self.toggle_collection)
        self.train_model_button.clicked.connect(self.train_sensor_data_fusion_model)
        self.test_model_button.clicked.connect(self.test_trained_depth_model)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Stereo camera previews"))
        controls_layout.addWidget(self.left_preview)
        controls_layout.addWidget(self.right_preview)
        controls_layout.addWidget(self.depth_preview)
        controls_layout.addWidget(QLabel("Model test previews"))
        controls_layout.addWidget(self.gt_depth_preview)
        controls_layout.addWidget(self.pred_depth_preview)
        controls_layout.addWidget(self.diff_depth_preview)
        controls_layout.addWidget(self.frame_count_label)
        controls_layout.addWidget(self.frame_count_spinbox)
        controls_layout.addWidget(self.collect_button)
        controls_layout.addWidget(self.train_model_button)
        controls_layout.addWidget(self.test_model_button)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        training_layout = QVBoxLayout()
        training_layout.addWidget(QLabel("Training progress"))
        training_layout.addWidget(self.training_eta_label)
        training_layout.addWidget(self.training_epoch_label)
        training_layout.addWidget(self.training_batch_label)
        if self.loss_canvas is not None:
            training_layout.addWidget(self.loss_canvas)
        else:
            training_layout.addWidget(self.loss_placeholder_label)
        training_layout.addStretch(1)

        right_panel_layout = QHBoxLayout()
        right_panel_layout.addLayout(controls_layout, stretch=2)
        right_panel_layout.addLayout(training_layout, stretch=1)

        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self.world_view, stretch=1)
        root_layout.addLayout(right_panel_layout, stretch=0)

        self.drive_timer = QTimer(self)
        self.drive_timer.setInterval(45)
        self.drive_timer.timeout.connect(self._collect_next_frame)

        self.preview_animation_timer = QTimer(self)
        self.preview_animation_timer.setInterval(33)
        self.preview_animation_timer.timeout.connect(self.world_view.advance_main_view)
        self.preview_animation_timer.start()

        self.output_root = Path(__file__).resolve().parent / "traindata"
        self.current_run_dir: Path | None = None
        self.current_meta_path: Path | None = None

        self.total_frames = 1000
        self.frame_index = 0
        self.capture_size = (640, 400)
        self.stereo_baseline = 0.48
        self.look_distance = 8.0
        self.depth_min_m = 0.0
        self.depth_max_m = 50.0

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        sec = max(0, int(round(seconds)))
        hours, rem = divmod(sec, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _reset_training_progress_ui(self) -> None:
        self.training_eta_label.setText("ETA: -")
        self.training_epoch_label.setText("Epochs: -")
        self.training_batch_label.setText("Mini-batches: -")
        self.train_loss_history = []
        self.val_loss_history = []
        self._update_loss_plot()

    def _update_loss_plot(self) -> None:
        if self.loss_ax is None or self.loss_canvas is None:
            return
        self.loss_ax.clear()
        self.loss_ax.set_title("Loss per epoch")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("MSE")
        self.loss_ax.grid(True, alpha=0.3)

        if self.train_loss_history:
            epochs_axis = list(range(1, len(self.train_loss_history) + 1))
            self.loss_ax.plot(epochs_axis, self.train_loss_history, marker="o", label="train")
            self.loss_ax.plot(epochs_axis, self.val_loss_history, marker="o", label="val")
            self.loss_ax.legend(loc="best")

        self.loss_canvas.draw_idle()

    def _path_pose(self, t: float) -> tuple[tuple[float, float, float], tuple[float, float]]:
        x = -22.0 + 44.0 * t
        y = 7.0 * math.sin(2.0 * math.pi * 1.2 * t)
        z = 2.2 + 0.45 * math.sin(2.0 * math.pi * 2.1 * t)

        t2 = min(1.0, t + 0.01)
        x2 = -22.0 + 44.0 * t2
        y2 = 7.0 * math.sin(2.0 * math.pi * 1.2 * t2)

        fx = x2 - x
        fy = y2 - y
        norm = math.hypot(fx, fy)
        if norm < 1e-8:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = fx / norm, fy / norm

        return (x, y, z), (fx, fy)

    def toggle_collection(self) -> None:
        if self.drive_timer.isActive():
            self._stop_collection("Collection stopped by user.")
            return
        self._start_collection()

    def _start_collection(self) -> None:
        self.total_frames = int(self.frame_count_spinbox.value())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_dir = run_dir
        self.current_meta_path = run_dir / "metadata.jsonl"
        self.current_meta_path.write_text("", encoding="utf-8")

        self.frame_index = 0
        self.frame_count_spinbox.setEnabled(False)
        self.collect_button.setText("Stop collection")
        self.train_model_button.setEnabled(False)
        self.status_label.setText(f"Collecting stereo data into:\n{run_dir}")
        self.drive_timer.start()
        self._collect_next_frame()

    def _stop_collection(self, message: str) -> None:
        if self.drive_timer.isActive():
            self.drive_timer.stop()
        self.frame_count_spinbox.setEnabled(True)
        self.collect_button.setText("Collect train data")
        self.train_model_button.setEnabled(True)
        self.status_label.setText(message)

    def _set_training_ui_running(self, running: bool) -> None:
        self.train_model_button.setEnabled(not running)
        self.test_model_button.setEnabled(not running)
        self.collect_button.setEnabled(not running)
        self.frame_count_spinbox.setEnabled(not running)
        if running:
            self._reset_training_progress_ui()

    def _set_testing_ui_running(self, running: bool) -> None:
        self.train_model_button.setEnabled(not running)
        self.test_model_button.setEnabled(not running)
        self.collect_button.setEnabled(not running)
        self.frame_count_spinbox.setEnabled(not running)

    def _load_training_records(self, run_dir: Path) -> list[dict]:
        metadata_path = run_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise RuntimeError("metadata.jsonl not found in selected training directory.")

        records: list[dict] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not all(key in record for key in ("left_image", "right_image", "depth_array")):
                    continue
                if not (run_dir / record["left_image"]).exists():
                    continue
                if not (run_dir / record["right_image"]).exists():
                    continue
                if not (run_dir / record["depth_array"]).exists():
                    continue
                records.append(record)
        if not records:
            raise RuntimeError("No valid training samples found in metadata.jsonl.")
        return records

    def _prepare_stereo_input_tensor(self, run_dir: Path, record: dict, target_size: tuple[int, int]) -> torch.Tensor:
        left_path = run_dir / record["left_image"]
        right_path = run_dir / record["right_image"]
        left_qimg = QImage(str(left_path))
        right_qimg = QImage(str(right_path))
        if left_qimg.isNull() or right_qimg.isNull():
            raise RuntimeError("Could not load stereo RGB images.")

        left = qimage_to_rgb_array(left_qimg).astype(np.float32) / 255.0
        right = qimage_to_rgb_array(right_qimg).astype(np.float32) / 255.0

        left_t = torch.from_numpy(left).permute(2, 0, 1)
        right_t = torch.from_numpy(right).permute(2, 0, 1)
        x = torch.cat([left_t, right_t], dim=0)
        x = F.interpolate(x.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _load_depth_meters(run_dir: Path, record: dict) -> np.ndarray:
        depth_path = run_dir / record["depth_array"]
        depth_u16 = np.load(depth_path)
        if depth_u16.ndim == 3 and depth_u16.shape[2] == 1:
            depth_u16 = depth_u16[:, :, 0]
        if depth_u16.ndim != 2:
            raise RuntimeError("Depth array has invalid shape. Expected HxW or HxWx1.")

        depth_min_m = float(record.get("depth_min_m", 0.0))
        depth_max_m = float(record.get("depth_max_m", 50.0))
        depth_span = max(1e-6, depth_max_m - depth_min_m)
        depth_m = (depth_u16.astype(np.float32) / 65535.0) * depth_span + depth_min_m
        return depth_m

    @staticmethod
    def _to_uint16_depth(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
        depth_span = max(1e-6, max_depth_m - min_depth_m)
        clipped = np.clip(depth_m, min_depth_m, max_depth_m)
        return np.rint(((clipped - min_depth_m) / depth_span) * 65535.0).astype(np.uint16)

    def _test_model(self, model_path: Path, run_dir: Path) -> tuple[float, int]:
        if TORCH_IMPORT_ERROR:
            raise RuntimeError(
                "PyTorch is not available. Install with: pip install torch\n"
                f"Import error: {TORCH_IMPORT_ERROR}"
            )
        if not model_path.exists():
            raise RuntimeError("Selected model file does not exist.")

        records = self._load_training_records(run_dir)
        checkpoint = torch.load(model_path, map_location="cpu")
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is None:
            raise RuntimeError("Checkpoint does not contain model_state_dict.")

        target_size_raw = checkpoint.get("target_size", (128, 192))
        if not isinstance(target_size_raw, (tuple, list)) or len(target_size_raw) != 2:
            raise RuntimeError("Checkpoint target_size is invalid.")
        target_size = (int(target_size_raw[0]), int(target_size_raw[1]))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StereoDepthFusionNet().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        maes: list[float] = []
        total = len(records)
        self.status_label.setText(
            f"Testing model on {total} samples.\n"
            f"Model: {model_path.name}\n"
            f"Dataset: {run_dir.name}"
        )
        QApplication.processEvents()

        with torch.no_grad():
            for idx, record in enumerate(records, start=1):
                x = self._prepare_stereo_input_tensor(run_dir, record, target_size).to(device)
                pred_norm = model(x).detach().cpu()[0, 0].numpy().astype(np.float32)

                gt_depth_m_full = self._load_depth_meters(run_dir, record)
                gt_depth_t = torch.from_numpy(gt_depth_m_full).unsqueeze(0).unsqueeze(0)
                gt_depth_m = F.interpolate(
                    gt_depth_t,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy().astype(np.float32)

                depth_min_m = float(record.get("depth_min_m", checkpoint.get("depth_min_m", self.depth_min_m)))
                depth_max_m = float(record.get("depth_max_m", checkpoint.get("depth_max_m", self.depth_max_m)))
                depth_span = max(1e-6, depth_max_m - depth_min_m)

                pred_depth_m = np.clip(pred_norm, 0.0, 1.0) * depth_span + depth_min_m
                abs_diff_m = np.abs(pred_depth_m - gt_depth_m)
                mae = float(abs_diff_m.mean())
                maes.append(mae)

                gt_u16 = self._to_uint16_depth(gt_depth_m, depth_min_m, depth_max_m)
                pred_u16 = self._to_uint16_depth(pred_depth_m, depth_min_m, depth_max_m)

                # Keep diff visualization robust across easy and hard samples.
                diff_vis_max = max(1e-3, float(np.percentile(abs_diff_m, 95.0)))
                diff_u16 = self._to_uint16_depth(abs_diff_m, 0.0, diff_vis_max)

                gt_img = self.world_view.depth_array_to_colormap_image(gt_u16[:, :, None], 0.0, 65535.0)
                pred_img = self.world_view.depth_array_to_colormap_image(pred_u16[:, :, None], 0.0, 65535.0)
                diff_img = self.world_view.depth_array_to_colormap_image(diff_u16[:, :, None], 0.0, 65535.0)

                self.gt_depth_preview.set_image(gt_img)
                self.pred_depth_preview.set_image(pred_img)
                self.diff_depth_preview.set_image(diff_img)

                self.training_epoch_label.setText(f"Test samples: {idx}/{total}")
                self.training_batch_label.setText(f"Image MAE: {mae:.4f} m")
                self.training_eta_label.setText(f"Running avg MAE: {float(np.mean(maes)):.4f} m")
                self.status_label.setText(
                    f"Testing model: {model_path.name}\n"
                    f"Sample {idx}/{total} - frame {record.get('frame_index', idx - 1)}\n"
                    f"MAE={mae:.4f} m"
                )
                QApplication.processEvents()

        mean_mae = float(np.mean(maes)) if maes else 0.0
        return mean_mae, total

    def _train_model(self, run_dir: Path) -> Path:
        if TORCH_IMPORT_ERROR:
            raise RuntimeError(
                "PyTorch is not available. Install with: pip install torch\n"
                f"Import error: {TORCH_IMPORT_ERROR}"
            )

        records = self._load_training_records(run_dir)
        target_size = (128, 192)
        dataset = StereoDepthDataset(run_dir, records, target_size)

        if len(dataset) < 2:
            raise RuntimeError("Need at least 2 samples for training.")

        val_len = max(1, int(0.1 * len(dataset)))
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StereoDepthFusionNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        epochs = 8
        num_train_batches = max(1, len(train_loader))
        total_steps = epochs * num_train_batches
        start_time = time.perf_counter()

        self.status_label.setText(
            f"Training started on {device.type} with {len(records)} samples.\n"
            f"Train/Val: {train_len}/{val_len}"
        )
        self.training_epoch_label.setText(f"Epochs: 0/{epochs}")
        self.training_batch_label.setText(f"Mini-batches: 0/{num_train_batches}")
        self.training_eta_label.setText("ETA: calculating...")
        QApplication.processEvents()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batches += 1

                completed_steps = (epoch - 1) * num_train_batches + batch_idx
                elapsed = time.perf_counter() - start_time
                seconds_per_step = elapsed / max(1, completed_steps)
                eta = seconds_per_step * (total_steps - completed_steps)

                self.training_epoch_label.setText(f"Epochs: {epoch}/{epochs}")
                self.training_batch_label.setText(f"Mini-batches: {batch_idx}/{num_train_batches}")
                self.training_eta_label.setText(f"ETA: {self._format_seconds(eta)}")

                if batch_idx % 5 == 0 or batch_idx == num_train_batches:
                    QApplication.processEvents()

            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss_sum += float(loss.item())
                    val_batches += 1

            train_loss = train_loss_sum / max(1, train_batches)
            val_loss = val_loss_sum / max(1, val_batches)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self._update_loss_plot()
            self.status_label.setText(
                f"Training model in {run_dir.name}\n"
                f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )
            QApplication.processEvents()

        self.training_epoch_label.setText(f"Epochs: {epochs}/{epochs}")
        self.training_batch_label.setText(f"Mini-batches: {num_train_batches}/{num_train_batches}")
        self.training_eta_label.setText("ETA: 00:00")

        model_path = run_dir / "sensor_fusion_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_channels": 6,
                "output_channels": 1,
                "target_size": target_size,
                "depth_min_m": self.depth_min_m,
                "depth_max_m": self.depth_max_m,
                "epochs": epochs,
            },
            model_path,
        )
        return model_path

    def train_sensor_data_fusion_model(self) -> None:
        if self.drive_timer.isActive():
            self.status_label.setText("Stop data collection before training the model.")
            return

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select training data directory",
            str(self.output_root),
        )
        if not selected_dir:
            return

        run_dir = Path(selected_dir)
        self._set_training_ui_running(True)
        try:
            model_path = self._train_model(run_dir)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Training failed:\n{exc}")
        else:
            self.status_label.setText(f"Training complete. Model saved to:\n{model_path}")
        finally:
            self._set_training_ui_running(False)

    def test_trained_depth_model(self) -> None:
        if self.drive_timer.isActive():
            self.status_label.setText("Stop data collection before testing a model.")
            return

        model_file, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select trained model checkpoint",
            str(self.output_root),
            "PyTorch model (*.pth);;All files (*)",
        )
        if not model_file:
            return

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select test data directory",
            str(self.output_root),
        )
        if not selected_dir:
            return

        model_path = Path(model_file)
        run_dir = Path(selected_dir)

        self._set_testing_ui_running(True)
        try:
            mean_mae, sample_count = self._test_model(model_path, run_dir)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Model test failed:\n{exc}")
        else:
            self.status_label.setText(
                f"Model test complete.\n"
                f"Samples: {sample_count}\n"
                f"Average image MAE: {mean_mae:.4f} m"
            )
            QMessageBox.information(
                self,
                "Test complete",
                f"Evaluated {sample_count} samples.\n"
                f"Average image MAE: {mean_mae:.4f} m",
            )
        finally:
            self._set_testing_ui_running(False)

    def _collect_next_frame(self) -> None:
        if self.current_run_dir is None or self.current_meta_path is None:
            self._stop_collection("Error: output directory not available.")
            return

        if self.frame_index >= self.total_frames:
            self._stop_collection(f"Collection complete. Saved to:\n{self.current_run_dir}")
            return

        t = self.frame_index / max(1, self.total_frames - 1)
        rig_pos, forward_xy = self._path_pose(t)
        left_cam, right_cam = self.world_view.stereo_from_rig(
            rig_pos=rig_pos,
            forward_xy=forward_xy,
            baseline=self.stereo_baseline,
            look_distance=self.look_distance,
        )
        center_cam = CameraPose(
            eye=rig_pos,
            target=(
                rig_pos[0] + self.look_distance * forward_xy[0],
                rig_pos[1] + self.look_distance * forward_xy[1],
                rig_pos[2] - 0.04,
            ),
            up=(0.0, 0.0, 1.0),
        )

        width, height = self.capture_size
        left_image = self.world_view.render_camera_image(left_cam, width, height)
        right_image = self.world_view.render_camera_image(right_cam, width, height)
        depth_array = self.world_view.render_depth_array(
            center_cam,
            width,
            height,
        )
        depth_preview_image = self.world_view.depth_array_to_colormap_image(
            depth_array,
            self.depth_min_m,
            self.depth_max_m,
        )

        depth_span = max(1e-6, self.depth_max_m - self.depth_min_m)
        depth_clipped = np.clip(depth_array, self.depth_min_m, self.depth_max_m)
        depth_uint16 = np.rint(((depth_clipped - self.depth_min_m) / depth_span) * 65535.0).astype(np.uint16)

        left_name = f"frame_{self.frame_index:04d}_left.png"
        right_name = f"frame_{self.frame_index:04d}_right.png"
        depth_name = f"depth_{self.frame_index:04d}.npy"
        left_path = self.current_run_dir / left_name
        right_path = self.current_run_dir / right_name
        depth_path = self.current_run_dir / depth_name

        if not left_image.save(str(left_path)) or not right_image.save(str(right_path)):
            self._stop_collection("Error: could not save one or more images.")
            return

        try:
            np.save(depth_path, depth_uint16)
        except Exception:  # noqa: BLE001
            self._stop_collection("Error: could not save depth array.")
            return

        self.left_preview.set_image(left_image)
        self.right_preview.set_image(right_image)
        self.depth_preview.set_image(depth_preview_image)

        meta = {
            "frame_index": self.frame_index,
            "left_image": left_name,
            "right_image": right_name,
            "depth_array": depth_name,
            "depth_shape": [height, width, 1],
            "depth_dtype": "uint16",
            "rig_position": [rig_pos[0], rig_pos[1], rig_pos[2]],
            "forward_xy": [forward_xy[0], forward_xy[1]],
            "baseline": self.stereo_baseline,
            "look_distance": self.look_distance,
            "depth_reference_camera": "center",
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
        }
        with self.current_meta_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta) + "\n")

        self.frame_index += 1
        self.status_label.setText(
            f"Collecting stereo data into:\n{self.current_run_dir}\n"
            f"Frame {self.frame_index}/{self.total_frames}"
        )


def main() -> int:
    if OPENGL_IMPORT_ERROR:
        print("PyOpenGL is required. Install with: pip install PyOpenGL PyOpenGL_accelerate")
        print(f"Import error: {OPENGL_IMPORT_ERROR}")
        return 1

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
