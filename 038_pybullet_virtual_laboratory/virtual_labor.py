"""
Mini PyBullet + PySide6 Example
--------------------------------
PyBullet laeuft im DIRECT-Modus, die Darstellung erfolgt im PySide6-Fenster.

Steuerung im 3D-Bereich:
  - Linke Maustaste ziehen: Kamera rotieren (Yaw/Pitch)
  - Rechte Maustaste ziehen: Kamera verschieben (Pan)
  - Mausrad: Zoom
"""

import json
import math
import random
import signal
import sys
from colorsys import hsv_to_rgb
from datetime import datetime
from pathlib import Path

import pybullet as p
import pybullet_data
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
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


class CameraPreview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(f"{title} camera inactive")
        self.title = title
        self.setFixedSize(250, 160)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #181818; color: #ddd; border: 1px solid #555;")

    def set_frame(self, rgba_bytes: bytes, width: int, height: int) -> None:
        image = QImage(rgba_bytes, width, height, 4 * width, QImage.Format_RGBA8888).copy()
        self.set_image(image)

    def set_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.setPixmap(pixmap)

    def clear_frame(self) -> None:
        self.clear()
        self.setText(f"{self.title} camera inactive")


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
        self.spawn_training_objects_button = QPushButton("Spawn 20 Near Robot")
        self.collect_train_data_button = QPushButton("Collect Train Data")
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
        self.add_husky_cam_button = QPushButton("Add Husky With Stereo Cameras")
        self.robot_camera_overlay_checkbox = QCheckBox("Show Robot Camera Views")
        self.robot_camera_overlay_checkbox.setChecked(False)
        self.capture_robot_camera_views_button = QPushButton("Capture Camera Views")
        self.auto_robot_camera_views_checkbox = QCheckBox("Auto Refresh Camera Views")
        self.auto_robot_camera_views_checkbox.setChecked(False)
        self.left_camera_preview = CameraPreview("Left")
        self.right_camera_preview = CameraPreview("Right")

        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.reset_button.clicked.connect(self.reset_world)
        self.spawn_button.clicked.connect(self.spawn_objects_cluster)
        self.spawn_training_objects_button.clicked.connect(self.spawn_training_objects_near_robot)
        self.collect_train_data_button.clicked.connect(self.toggle_train_data_collection)
        self.add_robot_button.clicked.connect(self.add_robot)
        self.add_husky_cam_button.clicked.connect(self.add_husky_with_camera)
        self.robot_camera_overlay_checkbox.stateChanged.connect(self.on_overlay_checkbox_changed)
        self.capture_robot_camera_views_button.clicked.connect(self.capture_robot_camera_views)
        self.auto_robot_camera_views_checkbox.stateChanged.connect(self.on_auto_camera_refresh_changed)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.start_stop_button)
        side_layout.addWidget(self.reset_button)
        side_layout.addWidget(self.spawn_button)
        side_layout.addWidget(self.spawn_training_objects_button)
        side_layout.addWidget(self.collect_train_data_button)
        side_layout.addWidget(self.robot_combo)
        side_layout.addWidget(self.add_robot_button)
        side_layout.addWidget(self.add_husky_cam_button)
        side_layout.addWidget(self.robot_camera_overlay_checkbox)
        side_layout.addWidget(self.capture_robot_camera_views_button)
        side_layout.addWidget(self.auto_robot_camera_views_checkbox)
        side_layout.addWidget(self.time_label)
        side_layout.addWidget(self.info_label)
        side_layout.addStretch(1)
        side_layout.addWidget(QLabel("Robot camera views"))
        side_layout.addWidget(self.left_camera_preview)
        side_layout.addWidget(self.right_camera_preview)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.view, stretch=1)
        main_layout.addLayout(side_layout, stretch=0)

        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.populate_initial_demo_world()
        self.robot_camera_body_id: int | None = None
        self.husky_drive_body_id: int | None = None
        self.latest_robot_body_id: int | None = None
        self.left_camera_marker_body_id: int | None = None
        self.left_camera_dir_body_id: int | None = None
        self.right_camera_marker_body_id: int | None = None
        self.right_camera_dir_body_id: int | None = None
        self.stereo_camera_baseline = 0.44
        self.training_object_body_ids: list[int] = []

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
        self.idle_render_every_n_ticks = 6
        self.robot_camera_preview_width = 160
        self.robot_camera_preview_height = 100
        self._next_robot_camera_preview_index = 0
        self._tick_counter = 0
        self.data_collection_active = False
        self.data_collection_total_frames = 80
        self.data_collection_frame_index = 0
        self.data_collection_run_dir: Path | None = None
        self.data_collection_metadata_path: Path | None = None
        self.data_collection_start_pos = [0.0, 0.0, 0.2]
        self.data_collection_end_pos = [0.0, 0.0, 0.2]
        self.data_collection_yaw = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(16)

        self.robot_camera_timer = QTimer(self)
        self.robot_camera_timer.timeout.connect(self.update_next_robot_camera_view)
        self.robot_camera_timer.setInterval(5000)

        self.data_collection_timer = QTimer(self)
        self.data_collection_timer.timeout.connect(self.collect_train_data_step)
        self.data_collection_timer.setInterval(160)

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
        self.latest_robot_body_id = None
        self.left_camera_marker_body_id = None
        self.left_camera_dir_body_id = None
        self.right_camera_marker_body_id = None
        self.right_camera_dir_body_id = None
        self.training_object_body_ids.clear()
        self._key_up_pressed = False
        self._key_down_pressed = False
        self._key_left_pressed = False
        self._key_right_pressed = False
        self._husky_left_speed_cmd = 0.0
        self._husky_right_speed_cmd = 0.0
        self.robot_camera_timer.stop()
        self.stop_train_data_collection(update_label=False)
        self.robot_camera_overlay_checkbox.setChecked(False)
        self.auto_robot_camera_views_checkbox.setChecked(False)
        self.left_camera_preview.clear_frame()
        self.right_camera_preview.clear_frame()

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

    def _body_exists(self, body_id: int | None) -> bool:
        if body_id is None:
            return False
        try:
            p.getBodyInfo(body_id)
        except p.error:
            return False
        return True

    def _current_robot_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        for body_id in (self.husky_drive_body_id, self.robot_camera_body_id, self.latest_robot_body_id):
            if self._body_exists(body_id):
                return p.getBasePositionAndOrientation(body_id)
        return None

    def spawn_training_objects_near_robot(self) -> None:
        # Ensure there is a ground plane before placing the training objects.
        if p.getNumBodies() == 0:
            p.loadURDF("plane.urdf")

        robot_pose = self._current_robot_pose()
        if robot_pose is None:
            self.info_label.setText(
                "Add a robot first, then spawn the 20 nearby training objects."
            )
            return

        robot_pos, robot_orn = robot_pose
        robot_yaw = p.getEulerFromQuaternion(robot_orn)[2]
        forward = (math.cos(robot_yaw), math.sin(robot_yaw))
        right = (-math.sin(robot_yaw), math.cos(robot_yaw))
        colors = self._generate_distinct_colors(20)
        object_scale = 2.0
        placed_objects: list[tuple[float, float, float, float]] = []

        shape_specs = [
            "box",
            "sphere",
            "cylinder",
            "capsule",
        ]

        for i in range(20):
            shape_kind = shape_specs[i % len(shape_specs)]
            size_factor = object_scale * (0.75 + 0.7 * random.random())

            if shape_kind == "sphere":
                radius = 0.07 * size_factor
                height = 2.0 * radius
                collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
                visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=colors[i])
                bounding_radius = radius
            elif shape_kind == "cylinder":
                radius = 0.055 * size_factor
                height = 0.18 * size_factor
                collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=colors[i],
                )
                bounding_radius = math.sqrt(radius * radius + (0.5 * height) ** 2)
            elif shape_kind == "capsule":
                radius = 0.045 * size_factor
                height = 0.18 * size_factor
                collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
                visual = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=radius,
                    length=height,
                    rgbaColor=colors[i],
                )
                bounding_radius = 0.5 * height + radius
            else:
                half_extents = [
                    random.uniform(0.05, 0.11) * size_factor,
                    random.uniform(0.04, 0.10) * size_factor,
                    random.uniform(0.05, 0.14) * size_factor,
                ]
                height = 2.0 * half_extents[2]
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=colors[i],
                )
                bounding_radius = math.sqrt(sum(value * value for value in half_extents))

            pos = None
            for _ in range(400):
                distance_forward = random.uniform(1.0, 3.2)
                lateral_offset = random.uniform(-1.4, 1.4)
                x = robot_pos[0] + distance_forward * forward[0] + lateral_offset * right[0]
                y = robot_pos[1] + distance_forward * forward[1] + lateral_offset * right[1]
                z = 0.5 * height + 0.005

                overlap_found = False
                for ox, oy, _, oradius in placed_objects:
                    dx = x - ox
                    dy = y - oy
                    min_dist = bounding_radius + oradius + 0.04
                    if dx * dx + dy * dy < min_dist * min_dist:
                        overlap_found = True
                        break

                if not overlap_found:
                    pos = (x, y, z)
                    break

            if pos is None:
                angle = (2.0 * math.pi * i) / 20.0
                radius_from_robot = 2.0 + 0.12 * (i % 5)
                pos = (
                    robot_pos[0] + radius_from_robot * math.cos(angle),
                    robot_pos[1] + radius_from_robot * math.sin(angle),
                    0.5 * height + 0.005,
                )

            yaw = random.uniform(0.0, 2.0 * math.pi)
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pos,
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            )
            p.changeDynamics(body_id, -1, lateralFriction=0.8, restitution=0.15)
            placed_objects.append((pos[0], pos[1], pos[2], bounding_radius))
            self.training_object_body_ids.append(body_id)

        self.info_label.setText(
            "Placed 20 varied training objects near the robot for RGB/depth experiments."
        )
        self.render_scene()

    def _current_training_object_positions(self) -> list[tuple[float, float, float]]:
        positions: list[tuple[float, float, float]] = []
        live_body_ids: list[int] = []
        for body_id in self.training_object_body_ids:
            if self._body_exists(body_id):
                pos, _ = p.getBasePositionAndOrientation(body_id)
                positions.append((pos[0], pos[1], pos[2]))
                live_body_ids.append(body_id)
        self.training_object_body_ids = live_body_ids
        return positions

    def _training_object_center_xy(self) -> tuple[float, float] | None:
        positions = self._current_training_object_positions()
        if not positions:
            return None
        center_x = sum(pos[0] for pos in positions) / len(positions)
        center_y = sum(pos[1] for pos in positions) / len(positions)
        return center_x, center_y

    def toggle_train_data_collection(self) -> None:
        if self.data_collection_active:
            self.stop_train_data_collection(update_label=True)
        else:
            self.start_train_data_collection()

    def start_train_data_collection(self) -> None:
        if self.robot_camera_body_id is None or not self._body_exists(self.robot_camera_body_id):
            self.info_label.setText("Add Husky With Stereo Cameras before collecting train data.")
            return

        object_center = self._training_object_center_xy()
        if object_center is None:
            self.info_label.setText("Spawn 20 nearby training objects before collecting train data.")
            return

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_camera_body_id)
        direction_x = object_center[0] - base_pos[0]
        direction_y = object_center[1] - base_pos[1]
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm < 1e-6:
            current_yaw = p.getEulerFromQuaternion(base_orn)[2]
            direction_x = math.cos(current_yaw)
            direction_y = math.sin(current_yaw)
        else:
            direction_x /= direction_norm
            direction_y /= direction_norm

        approach_start_distance = 5.0
        approach_end_distance = 1.0
        start_z = max(0.2, base_pos[2])
        self.data_collection_start_pos = [
            object_center[0] - approach_start_distance * direction_x,
            object_center[1] - approach_start_distance * direction_y,
            start_z,
        ]
        self.data_collection_end_pos = [
            object_center[0] - approach_end_distance * direction_x,
            object_center[1] - approach_end_distance * direction_y,
            start_z,
        ]
        self.data_collection_yaw = math.atan2(direction_y, direction_x)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_collection_run_dir = Path(__file__).resolve().parent / "traindata" / timestamp
        self.data_collection_run_dir.mkdir(parents=True, exist_ok=True)
        self.data_collection_metadata_path = self.data_collection_run_dir / "metadata.jsonl"
        self.data_collection_metadata_path.write_text("", encoding="utf-8")

        self.data_collection_frame_index = 0
        self.data_collection_active = True
        self.collect_train_data_button.setText("Stop Collect Train Data")
        self.is_running = False
        self.start_stop_button.setText("Start Simulation")
        self._key_up_pressed = False
        self._key_down_pressed = False
        self._key_left_pressed = False
        self._key_right_pressed = False
        self._husky_left_speed_cmd = 0.0
        self._husky_right_speed_cmd = 0.0
        self.robot_camera_overlay_checkbox.setChecked(True)
        self.auto_robot_camera_views_checkbox.setChecked(False)
        self.robot_camera_timer.stop()
        self.info_label.setText(f"Collecting stereo train data in:\n{self.data_collection_run_dir}")
        self.collect_train_data_step()
        self.data_collection_timer.start()

    def stop_train_data_collection(self, update_label: bool) -> None:
        if self.data_collection_timer.isActive():
            self.data_collection_timer.stop()
        was_active = self.data_collection_active
        self.data_collection_active = False
        self.collect_train_data_button.setText("Collect Train Data")
        if update_label and was_active and self.data_collection_run_dir is not None:
            self.info_label.setText(
                f"Train data collection stopped.\nSaved frames in:\n{self.data_collection_run_dir}"
            )

    def collect_train_data_step(self) -> None:
        if not self.data_collection_active:
            return

        if self.robot_camera_body_id is None or not self._body_exists(self.robot_camera_body_id):
            self.stop_train_data_collection(update_label=False)
            self.info_label.setText("Train data collection stopped: robot is no longer available.")
            return

        if self.data_collection_run_dir is None or self.data_collection_metadata_path is None:
            self.stop_train_data_collection(update_label=False)
            self.info_label.setText("Train data collection stopped: output directory is missing.")
            return

        if self.data_collection_frame_index >= self.data_collection_total_frames:
            run_dir = self.data_collection_run_dir
            self.stop_train_data_collection(update_label=False)
            self.info_label.setText(f"Train data collection complete.\nSaved in:\n{run_dir}")
            return

        t = self.data_collection_frame_index / max(1, self.data_collection_total_frames - 1)
        x = self.data_collection_start_pos[0] * (1.0 - t) + self.data_collection_end_pos[0] * t
        y = self.data_collection_start_pos[1] * (1.0 - t) + self.data_collection_end_pos[1] * t
        z = self.data_collection_start_pos[2] * (1.0 - t) + self.data_collection_end_pos[2] * t
        orientation = p.getQuaternionFromEuler([0.0, 0.0, self.data_collection_yaw])

        p.resetBasePositionAndOrientation(self.robot_camera_body_id, [x, y, z], orientation)
        p.resetBaseVelocity(self.robot_camera_body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self._update_camera_visualization_bodies()

        frame_name = f"frame_{self.data_collection_frame_index:04d}"
        saved_metadata = self.save_stereo_camera_images(frame_name)
        if saved_metadata is None:
            self.stop_train_data_collection(update_label=False)
            self.info_label.setText("Train data collection stopped: could not render stereo cameras.")
            return

        metadata = {
            "frame_index": self.data_collection_frame_index,
            "sim_time": self.sim_time,
            "robot_position": [x, y, z],
            "robot_yaw": self.data_collection_yaw,
            **saved_metadata,
        }
        with self.data_collection_metadata_path.open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(metadata) + "\n")

        if self.data_collection_frame_index % 5 == 0:
            self.render_scene()
        self.data_collection_frame_index += 1

    def save_stereo_camera_images(self, frame_name: str) -> dict | None:
        if self.data_collection_run_dir is None:
            return None

        stereo_poses = self._compute_robot_stereo_camera_poses()
        if stereo_poses is None:
            return None

        image_width = 320
        image_height = 200
        projection = p.computeProjectionMatrixFOV(
            fov=80.0,
            aspect=image_width / image_height,
            nearVal=0.05,
            farVal=100.0,
        )
        camera_info: dict[str, dict] = {}
        previews = {
            "left": self.left_camera_preview,
            "right": self.right_camera_preview,
        }
        for camera_name, pose in zip(("left", "right"), stereo_poses, strict=True):
            eye, target, up, _ = pose
            view = p.computeViewMatrix(eye, target, up)
            _, _, rgba, _, _ = p.getCameraImage(
                width=image_width,
                height=image_height,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
            )
            relative_path = Path(f"{frame_name}_{camera_name}.png")
            image_path = self.data_collection_run_dir / relative_path
            image = QImage(bytes(rgba), image_width, image_height, 4 * image_width, QImage.Format_RGBA8888).copy()
            if not image.save(str(image_path)):
                return None

            previews[camera_name].set_image(image)
            camera_info[camera_name] = {
                "image": str(relative_path),
                "eye": eye,
                "target": target,
                "up": up,
                "view_matrix": list(view),
                "projection_matrix": list(projection),
            }

        return {"cameras": camera_info}

    def add_robot(self) -> None:
        idx = self.robot_combo.currentIndex()
        _, urdf_path = self._robot_options[idx]
        # Spawn slightly above the ground so it settles cleanly
        spawn_pos = [0.0, 0.0, 0.2]
        try:
            robot_id = p.loadURDF(urdf_path, basePosition=spawn_pos, baseOrientation=[0, 0, 0, 1])
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"Could not load URDF:\n{exc}")
            return
        self.latest_robot_body_id = robot_id
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
        self.latest_robot_body_id = husky_id
        self._ensure_camera_visualization_bodies()
        self._update_camera_visualization_bodies()
        self._apply_husky_keyboard_control()
        self.info_label.setText(
            "3D Controls:\n"
            "- Left drag: Rotate\n"
            "- Right drag: Pan\n"
            "- Scroll wheel: Zoom\n"
            "- Arrow keys: Drive Husky\n"
            "Stereo robot cameras active (Husky).\n"
            "Enable camera views checkbox to show both camera images."
        )
        self.render_scene()

    def on_overlay_checkbox_changed(self, _state: int) -> None:
        if not self.robot_camera_overlay_checkbox.isChecked():
            self.robot_camera_timer.stop()
            self.auto_robot_camera_views_checkbox.setChecked(False)
            self.left_camera_preview.clear_frame()
            self.right_camera_preview.clear_frame()
        elif self.auto_robot_camera_views_checkbox.isChecked():
            self.robot_camera_timer.start()
        self.render_scene()

    def on_auto_camera_refresh_changed(self, _state: int) -> None:
        if self.auto_robot_camera_views_checkbox.isChecked():
            self.robot_camera_overlay_checkbox.setChecked(True)
            self._next_robot_camera_preview_index = 0
            self.robot_camera_timer.start()
        else:
            self.robot_camera_timer.stop()

    def capture_robot_camera_views(self) -> None:
        self.robot_camera_overlay_checkbox.setChecked(True)
        self._next_robot_camera_preview_index = 0
        self.update_next_robot_camera_view()
        self.update_next_robot_camera_view()

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

    def _compute_robot_camera_pose(
        self,
        lateral_offset: float = 0.0,
    ) -> tuple[list[float], list[float], list[float], list[float]] | None:
        if self.robot_camera_body_id is None or self.robot_camera_body_id >= p.getNumBodies():
            return None

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_camera_body_id)
        rot = p.getMatrixFromQuaternion(base_orn)
        forward = [rot[0], rot[3], rot[6]]
        left = [rot[1], rot[4], rot[7]]
        up = [rot[2], rot[5], rot[8]]

        # Mount the camera above the robot body so the view is outside the chassis.
        eye = [
            base_pos[0] + 0.42 * forward[0] + lateral_offset * left[0] + 0.58 * up[0],
            base_pos[1] + 0.42 * forward[1] + lateral_offset * left[1] + 0.58 * up[1],
            base_pos[2] + 0.42 * forward[2] + lateral_offset * left[2] + 0.58 * up[2],
        ]
        target = [
            eye[0] + 2.4 * forward[0],
            eye[1] + 2.4 * forward[1],
            eye[2] + 2.4 * forward[2],
        ]
        return eye, target, up, base_orn

    def _compute_robot_stereo_camera_poses(
        self,
    ) -> tuple[
        tuple[list[float], list[float], list[float], list[float]],
        tuple[list[float], list[float], list[float], list[float]],
    ] | None:
        half_baseline = 0.5 * self.stereo_camera_baseline
        left_pose = self._compute_robot_camera_pose(half_baseline)
        right_pose = self._compute_robot_camera_pose(-half_baseline)
        if left_pose is None or right_pose is None:
            return None
        return left_pose, right_pose

    def _ensure_camera_visualization_bodies(self) -> None:
        if self.left_camera_marker_body_id is None:
            marker_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=[0.15, 0.45, 1.0, 0.95],
            )
            self.left_camera_marker_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

        if self.left_camera_dir_body_id is None:
            dir_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.16, 0.015, 0.015],
                rgbaColor=[0.15, 0.45, 1.0, 0.95],
            )
            self.left_camera_dir_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=dir_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

        if self.right_camera_marker_body_id is None:
            marker_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.04,
                rgbaColor=[1.0, 0.25, 0.15, 0.95],
            )
            self.right_camera_marker_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

        if self.right_camera_dir_body_id is None:
            dir_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.16, 0.015, 0.015],
                rgbaColor=[1.0, 0.25, 0.15, 0.95],
            )
            self.right_camera_dir_body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=dir_visual,
                basePosition=[0.0, 0.0, -100.0],
                baseOrientation=[0, 0, 0, 1],
            )

    def _update_camera_visualization_bodies(self) -> None:
        poses = self._compute_robot_stereo_camera_poses()
        if poses is None:
            return

        self._ensure_camera_visualization_bodies()
        visualization_bodies = (
            (self.left_camera_marker_body_id, self.left_camera_dir_body_id, poses[0]),
            (self.right_camera_marker_body_id, self.right_camera_dir_body_id, poses[1]),
        )
        for marker_body_id, dir_body_id, pose in visualization_bodies:
            eye, target, _, base_orn = pose
            if marker_body_id is not None:
                p.resetBasePositionAndOrientation(marker_body_id, eye, [0, 0, 0, 1])

            if dir_body_id is None:
                continue
            forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]
            norm = math.sqrt(forward[0] ** 2 + forward[1] ** 2 + forward[2] ** 2)
            if norm > 1e-9:
                forward = [forward[0] / norm, forward[1] / norm, forward[2] / norm]
            center = [
                eye[0] + 0.16 * forward[0],
                eye[1] + 0.16 * forward[1],
                eye[2] + 0.16 * forward[2],
            ]
            p.resetBasePositionAndOrientation(dir_body_id, center, base_orn)

    def on_timer(self) -> None:
        self._tick_counter += 1

        self._apply_husky_keyboard_control()

        if self.is_running:
            for _ in range(self.substeps_per_tick):
                p.stepSimulation()
                self.sim_time += self.dt
            self.time_label.setText(f"Simulated Time: {self.sim_time:.3f} s")

        # Rendering is the most expensive step; lower frequency speeds up simulation.
        if self.data_collection_active:
            render_interval = 30
        else:
            render_interval = self.render_every_n_ticks if self.is_running else self.idle_render_every_n_ticks
        if self._tick_counter % render_interval == 0:
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

        self.view.setPixmap(QPixmap.fromImage(main_image))

    def update_next_robot_camera_view(self) -> None:
        if not self.robot_camera_overlay_checkbox.isChecked():
            return

        stereo_poses = self._compute_robot_stereo_camera_poses()
        if stereo_poses is None:
            self.left_camera_preview.clear_frame()
            self.right_camera_preview.clear_frame()
            return

        preview_w = self.robot_camera_preview_width
        preview_h = self.robot_camera_preview_height
        robot_projection = p.computeProjectionMatrixFOV(
            fov=80.0,
            aspect=preview_w / preview_h,
            nearVal=0.05,
            farVal=100.0,
        )
        previews = (self.left_camera_preview, self.right_camera_preview)
        preview_index = self._next_robot_camera_preview_index
        preview = previews[preview_index]
        pose = stereo_poses[preview_index]
        self._next_robot_camera_preview_index = (preview_index + 1) % len(previews)

        eye, target, up, _ = pose
        robot_view = p.computeViewMatrix(eye, target, up)
        _, _, robot_rgba, _, _ = p.getCameraImage(
            width=preview_w,
            height=preview_h,
            viewMatrix=robot_view,
            projectionMatrix=robot_projection,
            renderer=p.ER_TINY_RENDERER,
        )
        preview.set_frame(bytes(robot_rgba), preview_w, preview_h)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_train_data_collection(update_label=False)
        self.robot_camera_timer.stop()
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
