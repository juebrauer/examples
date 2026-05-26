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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

OPENGL_IMPORT_ERROR = ""
try:
    from OpenGL.GL import (  # type: ignore[import-not-found]
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LEQUAL,
        GL_LINES,
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
    QHBoxLayout,
    QLabel,
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
        aspect = width / max(1, height)

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

    def _draw_ground_grid(self, grid_size: int = 30, spacing: float = 1.6) -> None:
        glColor3f(0.18, 0.2, 0.22)
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            v = i * spacing
            glVertex3f(-grid_size * spacing, v, 0.0)
            glVertex3f(grid_size * spacing, v, 0.0)
            glVertex3f(v, -grid_size * spacing, 0.0)
            glVertex3f(v, grid_size * spacing, 0.0)
        glEnd()

    def _draw_world(self) -> None:
        self._draw_ground_grid()
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
        self.collect_button = QPushButton("Collect train data")
        self.frame_count_label = QLabel("Frames to collect")
        self.frame_count_spinbox = QSpinBox()
        self.frame_count_spinbox.setRange(1, 200000)
        self.frame_count_spinbox.setValue(1000)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        self.collect_button.clicked.connect(self.toggle_collection)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Stereo camera previews"))
        right_layout.addWidget(self.left_preview)
        right_layout.addWidget(self.right_preview)
        right_layout.addWidget(self.frame_count_label)
        right_layout.addWidget(self.frame_count_spinbox)
        right_layout.addWidget(self.collect_button)
        right_layout.addWidget(self.status_label)
        right_layout.addStretch(1)

        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self.world_view, stretch=1)
        root_layout.addLayout(right_layout, stretch=0)

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
        self.status_label.setText(f"Collecting stereo data into:\n{run_dir}")
        self.drive_timer.start()
        self._collect_next_frame()

    def _stop_collection(self, message: str) -> None:
        if self.drive_timer.isActive():
            self.drive_timer.stop()
        self.frame_count_spinbox.setEnabled(True)
        self.collect_button.setText("Collect train data")
        self.status_label.setText(message)

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

        width, height = self.capture_size
        left_image = self.world_view.render_camera_image(left_cam, width, height)
        right_image = self.world_view.render_camera_image(right_cam, width, height)

        left_name = f"frame_{self.frame_index:04d}_left.png"
        right_name = f"frame_{self.frame_index:04d}_right.png"
        left_path = self.current_run_dir / left_name
        right_path = self.current_run_dir / right_name

        if not left_image.save(str(left_path)) or not right_image.save(str(right_path)):
            self._stop_collection("Error: could not save one or more images.")
            return

        self.left_preview.set_image(left_image)
        self.right_preview.set_image(right_image)

        meta = {
            "frame_index": self.frame_index,
            "left_image": left_name,
            "right_image": right_name,
            "rig_position": [rig_pos[0], rig_pos[1], rig_pos[2]],
            "forward_xy": [forward_xy[0], forward_xy[1]],
            "baseline": self.stereo_baseline,
            "look_distance": self.look_distance,
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
