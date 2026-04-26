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
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
        self.setFocusPolicy(Qt.StrongFocus)

        self.view = PyBulletView()
        self.start_stop_button = QPushButton("Start Simulation")
        self.reset_button = QPushButton("Reset World")
        self.spawn_button = QPushButton("Spawn N=30 Objects")
        self.time_label = QLabel("Simulated Time: 0.000 s")
        self.info_label = QLabel(
            "3D Controls:\n"
            "- Left drag: Rotate\n"
            "- Right drag: Pan\n"
            "- Scroll wheel: Zoom"
        )

        # Mobile robot selection
        self.robot_combo = QComboBox()
        self._robot_options: list[tuple[str, str]] = [
            ("Husky UGV (wheeled)",          "husky/husky.urdf"),
            ("MIT Racecar (wheeled)",        "racecar/racecar.urdf"),
            ("MIT Racecar Diff. (wheeled)",  "racecar/racecar_differential.urdf"),
            ("R2D2 (wheeled)",               "r2d2.urdf"),
            ("Unitree A1 (quadruped)",       "a1/a1.urdf"),
            ("Unitree AlienGo (quadruped)",  "aliengo/aliengo.urdf"),
            ("Laikago (quadruped)",          "laikago/laikago.urdf"),
            ("MIT Mini Cheetah (quadruped)", "mini_cheetah/mini_cheetah.urdf"),
            ("Minitaur (quadruped)",         "quadruped/minitaur.urdf"),
            ("Spirit 40 (quadruped)",        "quadruped/spirit40.urdf"),
            ("Vision 60 (quadruped)",        "quadruped/vision60.urdf"),
            ("Humanoid (biped)",             "humanoid/humanoid.urdf"),
        ]
        for label, _ in self._robot_options:
            self.robot_combo.addItem(label)
        self.add_robot_button = QPushButton("Add Robot")
        self.add_husky_cam_button = QPushButton("Add Husky With Camera")
        self.robot_camera_overlay_checkbox = QCheckBox("Show Robot Camera Overlay")
        self.robot_camera_overlay_checkbox.setChecked(False)

        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.reset_button.clicked.connect(self.reset_world)
        self.spawn_button.clicked.connect(self.spawn_objects_cluster)
        self.add_robot_button.clicked.connect(self.add_robot)
        self.add_husky_cam_button.clicked.connect(self.add_husky_with_camera)
        self.robot_camera_overlay_checkbox.stateChanged.connect(self.on_overlay_checkbox_changed)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.start_stop_button)
        side_layout.addWidget(self.reset_button)
        side_layout.addWidget(self.spawn_button)
        side_layout.addWidget(self.robot_combo)
        side_layout.addWidget(self.add_robot_button)
        side_layout.addWidget(self.add_husky_cam_button)
        side_layout.addWidget(self.robot_camera_overlay_checkbox)
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
        self.robot_camera_body_id: int | None = None
        self.husky_drive_body_id: int | None = None
        self.camera_marker_body_id: int | None = None
        self.camera_dir_body_id: int | None = None

        self._husky_wheel_joints = [2, 3, 4, 5]
        self._husky_max_wheel_speed = 14.0
        self._husky_turn_gain = 0.55
        self._husky_wheel_force = 90.0
        self._husky_accel_up = 18.0
        self._husky_accel_down = 28.0
        self._husky_left_speed_cmd = 0.0
        self._husky_right_speed_cmd = 0.0

        self._key_up_pressed = False
        self._key_down_pressed = False
        self._key_left_pressed = False
        self._key_right_pressed = False

        self.is_running = False
        self.sim_time = 0.0
        self.dt = 1.0 / 240.0
        self.substeps_per_tick = 12
        self.render_every_n_ticks = 2
        self._tick_counter = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.render_scene()

    def toggle_simulation(self) -> None:
        self.is_running = not self.is_running
        if self.is_running:
            self.start_stop_button.setText("Stop Simulation")
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
        self.time_label.setText("Simulated Time: 0.000 s")
        self.robot_camera_body_id = None
        self.husky_drive_body_id = None
        self.camera_marker_body_id = None
        self.camera_dir_body_id = None
        self._key_up_pressed = False
        self._key_down_pressed = False
        self._key_left_pressed = False
        self._key_right_pressed = False
        self._husky_left_speed_cmd = 0.0
        self._husky_right_speed_cmd = 0.0

        body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        for body_id in body_ids:
            p.removeBody(body_id)

        # After reset the ground plane is always present.
        p.loadURDF("plane.urdf")

        self.render_scene()

    def spawn_objects_cluster(self) -> None:
        # If the world is empty, add a ground plane so objects interact visibly.
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

    def add_robot(self) -> None:
        idx = self.robot_combo.currentIndex()
        _, urdf_path = self._robot_options[idx]
        # Spawn slightly above the ground so it settles cleanly
        spawn_pos = [0.0, 0.0, 0.2]
        try:
            p.loadURDF(urdf_path, basePosition=spawn_pos, baseOrientation=[0, 0, 0, 1])
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"Could not load URDF:\n{exc}")
            return
        self.render_scene()

    def add_husky_with_camera(self) -> None:
        # Ensure there is a ground plane.
        if p.getNumBodies() == 0:
            p.loadURDF("plane.urdf")

        spawn_pos = [0.0, 0.0, 0.2]
        try:
            husky_id = p.loadURDF(
                "husky/husky.urdf",
                basePosition=spawn_pos,
                baseOrientation=[0, 0, 0, 1],
            )
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"Could not load Husky:\n{exc}")
            return

        self.robot_camera_body_id = husky_id
        self.husky_drive_body_id = husky_id
        self._ensure_camera_visualization_bodies()
        self._update_camera_visualization_bodies()
        self._apply_husky_keyboard_control()
        self.info_label.setText(
            "3D Controls:\n"
            "- Left drag: Rotate\n"
            "- Right drag: Pan\n"
            "- Scroll wheel: Zoom\n"
            "- Arrow keys: Drive Husky\n"
            "Robot camera active (Husky).\n"
            "Enable overlay checkbox to show camera image."
        )
        self.render_scene()

    def on_overlay_checkbox_changed(self, _state: int) -> None:
        self.render_scene()

    def _apply_husky_keyboard_control(self) -> None:
        if self.husky_drive_body_id is None or self.husky_drive_body_id >= p.getNumBodies():
            return

        throttle = float(self._key_up_pressed) - float(self._key_down_pressed)
        turn = float(self._key_left_pressed) - float(self._key_right_pressed)

        left_target = self._husky_max_wheel_speed * (throttle - self._husky_turn_gain * turn)
        right_target = self._husky_max_wheel_speed * (throttle + self._husky_turn_gain * turn)

        # Smooth acceleration and braking to avoid abrupt launch/wheelie behavior.
        control_dt = max(1e-3, self.timer.interval() / 1000.0)
        left_accel = self._husky_accel_up if abs(left_target) > abs(self._husky_left_speed_cmd) else self._husky_accel_down
        right_accel = self._husky_accel_up if abs(right_target) > abs(self._husky_right_speed_cmd) else self._husky_accel_down
        left_step = left_accel * control_dt
        right_step = right_accel * control_dt

        self._husky_left_speed_cmd = max(
            self._husky_left_speed_cmd - left_step,
            min(self._husky_left_speed_cmd + left_step, left_target),
        )
        self._husky_right_speed_cmd = max(
            self._husky_right_speed_cmd - right_step,
            min(self._husky_right_speed_cmd + right_step, right_target),
        )

        p.setJointMotorControlArray(
            bodyUniqueId=self.husky_drive_body_id,
            jointIndices=[2, 4],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self._husky_left_speed_cmd, self._husky_left_speed_cmd],
            forces=[self._husky_wheel_force, self._husky_wheel_force],
        )
        p.setJointMotorControlArray(
            bodyUniqueId=self.husky_drive_body_id,
            jointIndices=[3, 5],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self._husky_right_speed_cmd, self._husky_right_speed_cmd],
            forces=[self._husky_wheel_force, self._husky_wheel_force],
        )

    def eventFilter(self, _obj, event):  # type: ignore[override]
        if not self.isActiveWindow():
            return super().eventFilter(_obj, event)

        event_type = event.type()
        if event_type in (QEvent.KeyPress, QEvent.KeyRelease):
            if event.isAutoRepeat():
                return False

            is_pressed = event_type == QEvent.KeyPress
            handled = True
            key = event.key()

            if key == Qt.Key_Up:
                self._key_up_pressed = is_pressed
            elif key == Qt.Key_Down:
                self._key_down_pressed = is_pressed
            elif key == Qt.Key_Left:
                self._key_left_pressed = is_pressed
            elif key == Qt.Key_Right:
                self._key_right_pressed = is_pressed
            else:
                handled = False

            if handled:
                self._apply_husky_keyboard_control()
                return True

        return super().eventFilter(_obj, event)

    def _compute_robot_camera_pose(self) -> tuple[list[float], list[float], list[float], list[float]] | None:
        if self.robot_camera_body_id is None or self.robot_camera_body_id >= p.getNumBodies():
            return None

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_camera_body_id)
        rot = p.getMatrixFromQuaternion(base_orn)
        forward = [rot[0], rot[3], rot[6]]
        up = [rot[2], rot[5], rot[8]]

        # Mount the camera above the robot body so the view is outside the chassis.
        eye = [
            base_pos[0] + 0.42 * forward[0] + 0.58 * up[0],
            base_pos[1] + 0.42 * forward[1] + 0.58 * up[1],
            base_pos[2] + 0.42 * forward[2] + 0.58 * up[2],
        ]
        target = [
            eye[0] + 2.4 * forward[0],
            eye[1] + 2.4 * forward[1],
            eye[2] + 2.4 * forward[2],
        ]
        return eye, target, up, base_orn

    def _ensure_camera_visualization_bodies(self) -> None:
        if self.camera_marker_body_id is None:
            marker_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=[1.0, 0.15, 0.15, 0.95],
            )
            self.camera_marker_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

        if self.camera_dir_body_id is None:
            dir_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.16, 0.015, 0.015],
                rgbaColor=[1.0, 0.8, 0.1, 0.95],
            )
            self.camera_dir_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=dir_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

    def _update_camera_visualization_bodies(self) -> None:
        pose = self._compute_robot_camera_pose()
        if pose is None:
            return
        eye, target, _, base_orn = pose

        self._ensure_camera_visualization_bodies()
        if self.camera_marker_body_id is not None:
            p.resetBasePositionAndOrientation(self.camera_marker_body_id, eye, [0, 0, 0, 1])

        if self.camera_dir_body_id is not None:
            forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]
            norm = math.sqrt(forward[0] ** 2 + forward[1] ** 2 + forward[2] ** 2)
            if norm > 1e-9:
                forward = [forward[0] / norm, forward[1] / norm, forward[2] / norm]
            center = [
                eye[0] + 0.16 * forward[0],
                eye[1] + 0.16 * forward[1],
                eye[2] + 0.16 * forward[2],
            ]
            p.resetBasePositionAndOrientation(self.camera_dir_body_id, center, base_orn)

    def on_timer(self) -> None:
        self._tick_counter += 1

        self._apply_husky_keyboard_control()

        if self.is_running:
            for _ in range(self.substeps_per_tick):
                p.stepSimulation()
                self.sim_time += self.dt
            self.time_label.setText(f"Simulated Time: {self.sim_time:.3f} s")

        # Rendering is the most expensive step; lower frequency speeds up simulation.
        if (not self.is_running) or (self._tick_counter % self.render_every_n_ticks == 0):
            self.render_scene()

    def render_scene(self) -> None:
        width = max(64, self.view.width())
        height = max(64, self.view.height())

        if self.robot_camera_body_id is not None and self.robot_camera_body_id < p.getNumBodies():
            self._update_camera_visualization_bodies()

        # Main view always uses free camera navigation.
        view_matrix, projection_matrix = self.view.get_camera_matrices()
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        main_image = QImage(bytes(rgba), width, height, 4 * width, QImage.Format_RGBA8888).copy()

        has_robot_cam = (
            self.robot_camera_body_id is not None
            and self.robot_camera_body_id < p.getNumBodies()
            and self.robot_camera_overlay_checkbox.isChecked()
        )

        if has_robot_cam:
            pose = self._compute_robot_camera_pose()
            if pose is not None:
                eye, target, up, _ = pose
                robot_view = p.computeViewMatrix(eye, target, up)

                overlay_w = max(180, min(320, width // 3))
                overlay_h = max(120, min(220, height // 3))
                robot_projection = p.computeProjectionMatrixFOV(
                    fov=80.0,
                    aspect=overlay_w / max(1, overlay_h),
                    nearVal=0.05,
                    farVal=100.0,
                )
                _, _, robot_rgba, _, _ = p.getCameraImage(
                    width=overlay_w,
                    height=overlay_h,
                    viewMatrix=robot_view,
                    projectionMatrix=robot_projection,
                    renderer=p.ER_TINY_RENDERER,
                )

                overlay_image = QImage(
                    bytes(robot_rgba),
                    overlay_w,
                    overlay_h,
                    4 * overlay_w,
                    QImage.Format_RGBA8888,
                ).copy()

                painter = QPainter(main_image)
                margin = 12
                painter.drawImage(margin, margin, overlay_image)
                painter.setPen(QColor(0, 0, 0, 230))
                painter.drawRect(margin - 1, margin - 1, overlay_w + 1, overlay_h + 1)
                painter.end()
        self.view.setPixmap(QPixmap.fromImage(main_image))

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
