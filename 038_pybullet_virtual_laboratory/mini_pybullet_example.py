"""
Mini PyBullet + PySide6 Example
--------------------------------
PyBullet laeuft im DIRECT-Modus, die Darstellung erfolgt im PySide6-Fenster.

Steuerung im 3D-Bereich:
  - Linke Maustaste ziehen: Kamera rotieren (Yaw/Pitch)
  - Rechte Maustaste ziehen: Kamera verschieben (Pan)
  - Mausrad: Zoom
"""

import math
import random
import sys
from colorsys import hsv_to_rgb

import pybullet as p
import pybullet_data
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PyBulletView(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(800, 500)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #111; color: #ddd;")
        self.setText("Warte auf Render-Bild...")

        self.camera_target = [0.0, 0.0, 0.5]
        self.camera_distance = 5.0
        self.camera_yaw = 35.0
        self.camera_pitch = -25.0
        self.camera_roll = 0.0

        self._last_mouse_pos = QPoint()

    def get_camera_matrices(self) -> tuple[list[float], list[float]]:
        width = max(64, self.width())
        height = max(64, self.height())
        aspect = width / height

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=self.camera_roll,
            upAxisIndex=2,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=aspect,
            nearVal=0.1,
            farVal=100.0,
        )
        return view_matrix, projection_matrix

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._last_mouse_pos = event.position().toPoint()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        current_pos = event.position().toPoint()
        delta = current_pos - self._last_mouse_pos
        self._last_mouse_pos = current_pos

        if event.buttons() & Qt.LeftButton:
            self.camera_yaw += delta.x() * 0.4
            self.camera_pitch += delta.y() * 0.35
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
        elif event.buttons() & Qt.RightButton:
            pan_factor = self.camera_distance * 0.003
            yaw_rad = math.radians(self.camera_yaw)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)

            self.camera_target[0] += (-delta.x() * cos_yaw + delta.y() * sin_yaw) * pan_factor
            self.camera_target[1] += (-delta.x() * sin_yaw - delta.y() * cos_yaw) * pan_factor

        event.accept()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        angle = event.angleDelta().y()
        zoom_factor = 0.9 if angle > 0 else 1.1
        self.camera_distance *= zoom_factor
        self.camera_distance = max(0.6, min(40.0, self.camera_distance))
        event.accept()

    def update_frame(self, rgba_bytes: bytes, width: int, height: int) -> None:
        image = QImage(rgba_bytes, width, height, 4 * width, QImage.Format_RGBA8888).copy()
        self.setPixmap(QPixmap.fromImage(image))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mini PyBullet in PySide6")
        self.resize(1150, 650)

        self.view = PyBulletView()
        self.start_stop_button = QPushButton("Start Simulation")
        self.reset_button = QPushButton("Reset Welt")
        self.spawn_button = QPushButton("Spawn N=30 Objekte")
        self.time_label = QLabel("Simulierte Zeit: 0.000 s")
        self.info_label = QLabel(
            "3D-Steuerung:\n"
            "- Links ziehen: Rotate\n"
            "- Rechts ziehen: Pan\n"
            "- Mausrad: Zoom"
        )

        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.reset_button.clicked.connect(self.reset_world)
        self.spawn_button.clicked.connect(self.spawn_objects_cluster)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.start_stop_button)
        side_layout.addWidget(self.reset_button)
        side_layout.addWidget(self.spawn_button)
        side_layout.addWidget(self.time_label)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(1)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.view, stretch=1)
        main_layout.addLayout(side_layout, stretch=0)

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.populate_initial_demo_world()

        self.is_running = False
        self.sim_time = 0.0
        self.dt = 1.0 / 240.0
        self.substeps_per_tick = 12
        self.render_every_n_ticks = 2
        self._tick_counter = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)

        self.render_scene()

    def toggle_simulation(self) -> None:
        self.is_running = not self.is_running
        if self.is_running:
            self.start_stop_button.setText("Stopp Simulation")
        else:
            self.start_stop_button.setText("Start Simulation")

    def populate_initial_demo_world(self) -> None:
        p.loadURDF("plane.urdf")

    def _generate_distinct_colors(self, count: int) -> list[list[float]]:
        colors: list[list[float]] = []
        for i in range(count):
            hue = i / max(1, count)
            r, g, b = hsv_to_rgb(hue, 0.75, 0.95)
            colors.append([r, g, b, 0.9])
        random.shuffle(colors)
        return colors

    def _find_non_overlapping_position(
        self,
        placed_objects: list[tuple[float, float, float, float]],
        radius: float,
        center_xy: tuple[float, float] = (0.0, 0.0),
        xy_span: float = 1.0,
        z_min: float = 5.0,
        z_max: float = 5.6,
        clearance: float = 0.03,
        max_tries: int = 400,
    ) -> tuple[float, float, float] | None:
        cx, cy = center_xy
        for _ in range(max_tries):
            x = random.uniform(cx - xy_span / 2.0, cx + xy_span / 2.0)
            y = random.uniform(cy - xy_span / 2.0, cy + xy_span / 2.0)
            z = random.uniform(z_min, z_max)

            overlap_found = False
            for ox, oy, oz, oradius in placed_objects:
                dx = x - ox
                dy = y - oy
                dz = z - oz
                min_dist = radius + oradius + clearance
                if dx * dx + dy * dy + dz * dz < min_dist * min_dist:
                    overlap_found = True
                    break

            if not overlap_found:
                return (x, y, z)

        return None

    def reset_world(self) -> None:
        self.is_running = False
        self.start_stop_button.setText("Start Simulation")
        self.sim_time = 0.0
        self.time_label.setText("Simulierte Zeit: 0.000 s")

        body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        for body_id in body_ids:
            p.removeBody(body_id)

        # Nach Reset bleibt immer ein Boden in der Welt.
        p.loadURDF("plane.urdf")

        self.render_scene()

    def spawn_objects_cluster(self) -> None:
        # Falls die Welt leer ist, wird ein Boden hinzugefuegt, damit die Objekte sichtbar reagieren.
        if p.getNumBodies() == 0:
            p.loadURDF("plane.urdf")

        sphere_radius = 0.12
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)

        box_half_extents = [0.16, 0.09, 0.12]
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        box_bounding_radius = math.sqrt(
            box_half_extents[0] ** 2 + box_half_extents[1] ** 2 + box_half_extents[2] ** 2
        )

        num_spheres = 20
        num_boxes = 10
        total_objects = num_spheres + num_boxes
        colors = self._generate_distinct_colors(total_objects)
        placed_objects: list[tuple[float, float, float, float]] = []

        for i in range(num_spheres):
            pos = None
            for attempt in range(6):
                pos = self._find_non_overlapping_position(
                    placed_objects=placed_objects,
                    radius=sphere_radius,
                    xy_span=1.0 + 0.2 * attempt,
                    z_min=5.0,
                    z_max=5.8 + 0.3 * attempt,
                    max_tries=300,
                )
                if pos is not None:
                    break
            if pos is None:
                max_top = max((oz + oradius) for _, _, oz, oradius in placed_objects) if placed_objects else 5.0
                pos = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), max_top + sphere_radius + 0.25)
            x, y, z = pos

            sphere_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=sphere_radius,
                rgbaColor=colors[i],
            )
            body_id = p.createMultiBody(
                baseMass=0.35,
                baseCollisionShapeIndex=sphere_collision,
                baseVisualShapeIndex=sphere_visual,
                basePosition=[x, y, z],
                baseOrientation=[0, 0, 0, 1],
            )
            p.changeDynamics(body_id, -1, lateralFriction=0.6, restitution=0.45)
            placed_objects.append((x, y, z, sphere_radius))

        for i in range(num_boxes):
            pos = None
            for attempt in range(6):
                pos = self._find_non_overlapping_position(
                    placed_objects=placed_objects,
                    radius=box_bounding_radius,
                    xy_span=1.0 + 0.2 * attempt,
                    z_min=5.0,
                    z_max=5.8 + 0.3 * attempt,
                    max_tries=300,
                )
                if pos is not None:
                    break
            if pos is None:
                max_top = max((oz + oradius) for _, _, oz, oradius in placed_objects) if placed_objects else 5.0
                pos = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), max_top + box_bounding_radius + 0.25)
            x, y, z = pos

            box_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=box_half_extents,
                rgbaColor=colors[num_spheres + i],
            )
            yaw = random.uniform(0.0, 2.0 * math.pi)
            orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            body_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=box_collision,
                baseVisualShapeIndex=box_visual,
                basePosition=[x, y, z],
                baseOrientation=orientation,
            )
            p.changeDynamics(body_id, -1, lateralFriction=0.7, restitution=0.3)
            placed_objects.append((x, y, z, box_bounding_radius))

        self.render_scene()

    def on_timer(self) -> None:
        self._tick_counter += 1

        if self.is_running:
            for _ in range(self.substeps_per_tick):
                p.stepSimulation()
                self.sim_time += self.dt
            self.time_label.setText(f"Simulierte Zeit: {self.sim_time:.3f} s")

        # Rendering ist der teuerste Schritt; geringere Renderfrequenz beschleunigt die Simulation.
        if (not self.is_running) or (self._tick_counter % self.render_every_n_ticks == 0):
            self.render_scene()

    def render_scene(self) -> None:
        width = max(64, self.view.width())
        height = max(64, self.view.height())
        view_matrix, projection_matrix = self.view.get_camera_matrices()

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        self.view.update_frame(bytes(rgba), width, height)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
