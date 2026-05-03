import math
import random
import sys
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Geometry helpers
# ============================================================


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def point_to_rect_distance(px: float, py: float, rect: QRectF) -> float:
    cx = clamp(px, rect.left(), rect.right())
    cy = clamp(py, rect.top(), rect.bottom())
    return math.hypot(px - cx, py - cy)


def ray_segment_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Optional[float]:
    vx = x2 - x1
    vy = y2 - y1

    det = dx * vy - dy * vx
    if abs(det) < 1e-9:
        return None

    wx = x1 - ox
    wy = y1 - oy

    t = (wx * vy - wy * vx) / det
    u = (wx * dy - wy * dx) / det

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


# ============================================================
# KD-tree for scan vectors
# ============================================================


@dataclass
class KDNode:
    vec: List[float]
    label_xy: Tuple[float, float]
    axis: int
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class ScanKDTree:
    def __init__(self, samples: List[Tuple[List[float], Tuple[float, float]]]):
        self.dim = len(samples[0][0]) if samples else 0
        self.root = self._build(samples, depth=0)

    def _build(
        self,
        samples: List[Tuple[List[float], Tuple[float, float]]],
        depth: int,
    ) -> Optional[KDNode]:
        if not samples:
            return None

        axis = depth % self.dim
        samples_sorted = sorted(samples, key=lambda item: item[0][axis])
        median = len(samples_sorted) // 2

        vec, label_xy = samples_sorted[median]
        return KDNode(
            vec=vec,
            label_xy=label_xy,
            axis=axis,
            left=self._build(samples_sorted[:median], depth + 1),
            right=self._build(samples_sorted[median + 1 :], depth + 1),
        )

    @staticmethod
    def _sqdist(a: List[float], b: List[float]) -> float:
        return sum((x - y) * (x - y) for x, y in zip(a, b))

    def knearest(self, query: List[float], k: int) -> List[Tuple[float, Tuple[float, float]]]:
        if self.root is None or k <= 0:
            return []

        # Max-heap with entries (-distance, label)
        heap: List[Tuple[float, Tuple[float, float]]] = []

        def visit(node: Optional[KDNode]) -> None:
            if node is None:
                return

            d2 = self._sqdist(query, node.vec)
            item = (-d2, node.label_xy)

            if len(heap) < k:
                heapq.heappush(heap, item)
            elif d2 < -heap[0][0]:
                heapq.heapreplace(heap, item)

            axis = node.axis
            diff = query[axis] - node.vec[axis]

            first, second = (node.left, node.right) if diff < 0.0 else (node.right, node.left)
            visit(first)

            worst_d2 = -heap[0][0] if heap else float("inf")
            if len(heap) < k or (diff * diff) < worst_d2:
                visit(second)

        visit(self.root)

        result = [(-neg_d2, label) for neg_d2, label in heap]
        result.sort(key=lambda t: t[0])
        return result


# ============================================================
# World and robot
# ============================================================


@dataclass
class Robot:
    x: float
    y: float
    theta: float
    radius: float = 12.0


class WorldModel:
    def __init__(self) -> None:
        self.width = 900.0
        self.height = 620.0

        self.wall_rects: List[QRectF] = []
        self.object_rects: List[QRectF] = []
        self._build_world()

        self.segments = self._all_segments()

    def _build_world(self) -> None:
        t = 16.0  # wall thickness

        # Outer boundary walls
        self.wall_rects.extend(
            [
                QRectF(0.0, 0.0, self.width, t),
                QRectF(0.0, self.height - t, self.width, t),
                QRectF(0.0, 0.0, t, self.height),
                QRectF(self.width - t, 0.0, t, self.height),
            ]
        )

        # Interior walls with multiple explicit doors so all rooms are reachable.
        # Vertical partition x=300 with two doors
        self.wall_rects.extend(
            [
                QRectF(300.0, t, t, 120.0),
                QRectF(300.0, 240.0, t, 120.0),
                QRectF(300.0, 460.0, t, self.height - 460.0 - t),
            ]
        )

        # Vertical partition x=600 with two doors
        self.wall_rects.extend(
            [
                QRectF(600.0, t, t, 120.0),
                QRectF(600.0, 240.0, t, 120.0),
                QRectF(600.0, 460.0, t, self.height - 460.0 - t),
            ]
        )

        # Horizontal partition y=320 with two doors
        self.wall_rects.extend(
            [
                QRectF(t, 320.0, 220.0, t),
                QRectF(320.0, 320.0, 260.0, t),
                QRectF(680.0, 320.0, self.width - 680.0 - t, t),
            ]
        )

        # Room objects
        self.object_rects.extend(
            [
                QRectF(90.0, 90.0, 85.0, 45.0),
                QRectF(195.0, 180.0, 60.0, 80.0),
                QRectF(380.0, 120.0, 95.0, 95.0),
                QRectF(470.0, 420.0, 80.0, 55.0),
                QRectF(710.0, 110.0, 120.0, 60.0),
                QRectF(760.0, 410.0, 65.0, 120.0),
                QRectF(130.0, 430.0, 130.0, 75.0),
            ]
        )

    def _rect_segments(self, rect: QRectF) -> List[Tuple[float, float, float, float]]:
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
        return [
            (l, t, r, t),
            (r, t, r, b),
            (r, b, l, b),
            (l, b, l, t),
        ]

    def _all_segments(self) -> List[Tuple[float, float, float, float]]:
        segments: List[Tuple[float, float, float, float]] = []
        for rect in self.wall_rects + self.object_rects:
            segments.extend(self._rect_segments(rect))
        return segments

    def all_obstacle_rects(self) -> List[QRectF]:
        return self.wall_rects + self.object_rects

    def is_free(self, x: float, y: float, radius: float) -> bool:
        for rect in self.all_obstacle_rects():
            if point_to_rect_distance(x, y, rect) <= radius:
                return False
        return True

    def cast_scan(
        self,
        x: float,
        y: float,
        theta: float,
        beam_count: int,
        max_range: float,
        noise_std: float = 0.0,
    ) -> List[float]:
        scan: List[float] = []

        # Use robot-local beams rotated by theta. With compass, theta is known at query time.
        for i in range(beam_count):
            angle = theta + 2.0 * math.pi * (i / beam_count)
            dx = math.cos(angle)
            dy = math.sin(angle)

            best = max_range
            for x1, y1, x2, y2 in self.segments:
                hit = ray_segment_distance(x, y, dx, dy, x1, y1, x2, y2)
                if hit is not None and hit < best:
                    best = hit

            if noise_std > 0.0:
                best += random.gauss(0.0, noise_std)
                best = clamp(best, 0.0, max_range)

            scan.append(best / max_range)

        return scan


# ============================================================
# Visualization widget
# ============================================================


class WorldWidget(QWidget):
    def __init__(self, world: WorldModel, parent=None) -> None:
        super().__init__(parent)
        self.world = world
        self.robot: Optional[Robot] = None
        self.current_scan: List[float] = []
        self.beam_count = 48
        self.max_range = 260.0

        self.heatmap: List[List[float]] = []
        self.heat_cols = 0
        self.heat_rows = 0
        self.cell = 12.0
        self.show_belief_map = False
        self.belief_cells: List[Tuple[float, float]] = []
        self.belief_values: Dict[Tuple[float, float], float] = {}

        self.candidate_points: List[Tuple[float, float, float]] = []
        self.estimate_xy: Optional[Tuple[float, float]] = None
        self.db_probe_xy: Optional[Tuple[float, float]] = None
        self.db_scanned_points: List[Tuple[float, float]] = []
        self.db_route_points: List[Tuple[float, float]] = []
        self.db_route_active = False

        self.setMinimumSize(940, 680)

    def init_heatmap(self) -> None:
        self.heat_cols = int(math.ceil(self.world.width / self.cell))
        self.heat_rows = int(math.ceil(self.world.height / self.cell))
        self.heatmap = [[0.0 for _ in range(self.heat_cols)] for _ in range(self.heat_rows)]

    def set_belief_cells(self, cells: List[Tuple[float, float]]) -> None:
        self.belief_cells = cells
        self.belief_values = {(x, y): 0.0 for x, y in cells}

    def clear_heatmap(self) -> None:
        if not self.belief_values:
            return
        for key in list(self.belief_values.keys()):
            self.belief_values[key] = 0.0

    def update_belief_heatmap(self, match_points: List[Tuple[float, float]]) -> None:
        # Belief update rule: +1.0 for matched cells, -0.25 for non-matched cells, clipped at 0.
        if not self.belief_values:
            return

        for key in list(self.belief_values.keys()):
            self.belief_values[key] = max(0.0, self.belief_values[key] - 0.25)

        for x, y in match_points:
            key = (x, y)
            if key in self.belief_values:
                self.belief_values[key] += 1.0

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor("#f4f3ef"))

        # Fit world with margin
        margin = 20.0
        sx = (self.width() - 2.0 * margin) / self.world.width
        sy = (self.height() - 2.0 * margin) / self.world.height
        scale = min(sx, sy)
        ox = (self.width() - self.world.width * scale) * 0.5
        oy = (self.height() - self.world.height * scale) * 0.5

        p.save()
        p.translate(ox, oy)
        p.scale(scale, scale)

        # World floor
        p.fillRect(QRectF(0.0, 0.0, self.world.width, self.world.height), QColor("#faf8f2"))

        # Heatmap overlay
        self._draw_heatmap(p)

        # Optional route overlay for route-based DB generation
        self._draw_route_overlay(p)

        # Visualize database generation sampling positions
        self._draw_db_generation(p)

        # Walls and objects
        wall_brush = QColor("#4a4e55")
        object_brush = QColor("#8c6d62")

        p.setPen(Qt.NoPen)
        p.setBrush(wall_brush)
        for rect in self.world.wall_rects:
            p.drawRect(rect)

        p.setBrush(object_brush)
        for rect in self.world.object_rects:
            p.drawRect(rect)

        # Candidate locations from nearest neighbors
        self._draw_candidates(p)

        # Robot and beams
        if self.robot is not None:
            self._draw_scan_beams(p)
            self._draw_robot(p)

        # Border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#1b1d20"), 1.4))
        p.drawRect(QRectF(0.0, 0.0, self.world.width, self.world.height))

        p.restore()

    def _draw_heatmap(self, p: QPainter) -> None:
        if not self.show_belief_map:
            return

        if not self.belief_cells:
            return

        max_v = max(self.belief_values.values()) if self.belief_values else 0.0
        max_v = max(max_v, 1e-9)

        for x, y in self.belief_cells:
            v = self.belief_values.get((x, y), 0.0)
            t = v / max_v
            color = self._cool_warm_color(t)

            # Larger beliefs are rendered with larger circles up to a cap.
            radius = 3.2 + 6.8 * t
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(x, y), radius, radius)

            if v > 0.0:
                f = QFont(p.font())
                f.setPointSizeF(4.2 + 3.6 * t)
                p.setFont(f)
                p.setPen(QPen(QColor("white"), 0.5))
                text_box = QRectF(x - radius * 1.6, y - radius * 0.95, radius * 3.2, radius * 1.9)
                p.drawText(text_box, Qt.AlignCenter, f"{v:.2f}")

    def _cool_warm_color(self, t: float) -> QColor:
        t = clamp(t, 0.0, 1.0)

        # Cool-warm: belief 0 -> blue, high belief -> red.
        r = int(35 + t * (230 - 35))
        g = int(90 + t * (45 - 90))
        b = int(230 + t * (35 - 230))
        return QColor(r, g, b, 165)

    def _draw_candidates(self, p: QPainter) -> None:
        if not self.candidate_points:
            return

        for x, y, weight in self.candidate_points:
            radius = 6.0 + 14.0 * weight
            alpha = int(60 + 160 * weight)
            p.setBrush(QColor(240, 70, 70, alpha))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, y), radius, radius)

        if self.estimate_xy is not None:
            ex, ey = self.estimate_xy
            p.setPen(QPen(QColor("#0f7f6e"), 2.5))
            p.setBrush(QColor(15, 127, 110, 70))
            p.drawEllipse(QPointF(ex, ey), 12.0, 12.0)

    def _draw_db_generation(self, p: QPainter) -> None:
        if self.db_scanned_points:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(220, 120, 25, 80))
            for x, y in self.db_scanned_points:
                p.drawEllipse(QPointF(x, y), 4.5, 4.5)

        if self.db_probe_xy is not None:
            x, y = self.db_probe_xy
            p.setPen(QPen(QColor("#cc5f0a"), 2.0))
            p.setBrush(QColor(230, 140, 30, 50))
            p.drawEllipse(QPointF(x, y), 15.0, 15.0)

    def _draw_route_overlay(self, p: QPainter) -> None:
        if not self.db_route_active or len(self.db_route_points) < 2:
            return

        p.setPen(QPen(QColor(32, 105, 172, 130), 2.0))
        for i in range(len(self.db_route_points) - 1):
            x1, y1 = self.db_route_points[i]
            x2, y2 = self.db_route_points[i + 1]
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(32, 105, 172, 150))
        for x, y in self.db_route_points:
            p.drawEllipse(QPointF(x, y), 5.0, 5.0)

    def _draw_scan_beams(self, p: QPainter) -> None:
        if self.robot is None or not self.current_scan:
            return

        x = self.robot.x
        y = self.robot.y

        p.setPen(QPen(QColor(30, 30, 30, 55), 1.0))
        for i, value in enumerate(self.current_scan):
            angle = self.robot.theta + 2.0 * math.pi * (i / self.beam_count)
            dist = value * self.max_range
            bx = x + dist * math.cos(angle)
            by = y + dist * math.sin(angle)
            p.drawLine(QPointF(x, y), QPointF(bx, by))

    def _draw_robot(self, p: QPainter) -> None:
        assert self.robot is not None

        x = self.robot.x
        y = self.robot.y
        r = self.robot.radius

        p.setBrush(QColor("#1d8fdb"))
        p.setPen(QPen(QColor("#0b4a72"), 2.0))
        p.drawEllipse(QPointF(x, y), r, r)

        hx = x + (r + 10.0) * math.cos(self.robot.theta)
        hy = y + (r + 10.0) * math.sin(self.robot.theta)
        p.setPen(QPen(QColor("#0b4a72"), 3.0))
        p.drawLine(QPointF(x, y), QPointF(hx, hy))


# ============================================================
# Main Window
# ============================================================


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D Scan-Matching Demo (KD-Tree)")
        self.resize(1380, 820)

        self.world = WorldModel()
        self.robot = Robot(x=80.0, y=80.0, theta=0.0)

        self.beam_count = 8
        self.max_range = 260.0

        self.db_samples: List[Tuple[List[float], Tuple[float, float], float]] = []
        self.kdtree: Optional[ScanKDTree] = None
        self.db_generation_running = False
        self.db_generation_points: List[Tuple[float, float]] = []
        self.db_generation_angles: List[float] = []
        self.db_generation_samples: List[Tuple[List[float], Tuple[float, float], float]] = []
        self.db_generation_index = 0
        self.db_generation_total_steps = 0
        self.db_saved_pose: Optional[Tuple[float, float, float]] = None
        self.db_generation_mode = "grid"
        self.db_route_phase = "scan"
        self.db_route_scan_point_index = 0
        self.db_route_scan_angle_index = 0

        self.route_move_step_px = 8
        self.route_patrol_targets: List[Tuple[float, float]] = [
            (70.0, 70.0),
            (160.0, 70.0),
            (255.0, 90.0),
            (255.0, 260.0),
            (420.0, 260.0),
            (420.0, 520.0),
            (560.0, 520.0),
            (740.0, 520.0),
            (820.0, 420.0),
            (820.0, 260.0),
            (740.0, 120.0),
            (640.0, 120.0),
            (520.0, 120.0),
            (520.0, 260.0),
            (430.0, 260.0),
            (430.0, 500.0),
            (260.0, 500.0),
            (140.0, 500.0),
        ]
        self.route_cached_points: List[Tuple[float, float]] = []
        self.route_coverage_targets_count = 0

        self.scan_step_x = 28
        self.scan_step_y = 28
        self.scan_step_ms = 50
        self.scan_theta_step_deg = 25
        self.scan_density = 60
        self.orientation_feature_weight = 1.25
        self.match_mode = "top_n"
        self.match_top_n = 6
        self.match_theta = 0.22
        self.match_display_mode = "current"
        self.match_orientation_tolerance_deg = 15.0

        self.matching_running = False
        self.tick_counter = 0

        self.world_widget = WorldWidget(self.world)
        self.world_widget.robot = self.robot
        self.world_widget.beam_count = self.beam_count
        self.world_widget.max_range = self.max_range
        self.world_widget.init_heatmap()

        self.btn_generate_db = QPushButton("Generate scan database")
        self.btn_generate_db.clicked.connect(self.generate_database)

        self.btn_match = QPushButton("Scan-Matching")
        self.btn_match.setCheckable(True)
        self.btn_match.clicked.connect(self.toggle_matching)

        self.lbl_db = QLabel("Database: not generated")
        self.lbl_pose = QLabel("Robot pose: x=?, y=?, theta=?")
        self.lbl_estimate = QLabel("Estimated x,y: n/a")

        self.radio_match_topn = QRadioButton("Top-N neighbors")
        self.radio_match_threshold = QRadioButton("Distance threshold Theta")
        self.radio_match_topn.setChecked(True)
        self.radio_match_topn.toggled.connect(self._on_match_mode_changed)
        self.radio_match_threshold.toggled.connect(self._on_match_mode_changed)

        self.radio_display_current = QRadioButton("Only show current matches")
        self.radio_display_belief = QRadioButton("Show accumulated belief")
        self.radio_display_current.setChecked(True)
        self.radio_display_current.toggled.connect(self._on_display_mode_changed)
        self.radio_display_belief.toggled.connect(self._on_display_mode_changed)

        self.lbl_match_topn = QLabel(f"N neighbors: {self.match_top_n}")
        self.slider_match_topn = QSlider(Qt.Horizontal)
        self.slider_match_topn.setRange(1, 40)
        self.slider_match_topn.setSingleStep(1)
        self.slider_match_topn.setValue(self.match_top_n)
        self.slider_match_topn.valueChanged.connect(self._on_match_topn_changed)

        self.lbl_match_theta = QLabel(f"Theta: {self.match_theta:.2f}")
        self.slider_match_theta = QSlider(Qt.Horizontal)
        self.slider_match_theta.setRange(1, 100)
        self.slider_match_theta.setSingleStep(1)
        self.slider_match_theta.setValue(int(round(self.match_theta * 100.0)))
        self.slider_match_theta.valueChanged.connect(self._on_match_theta_changed)

        self.lbl_speed = QLabel("Generation step time: 50 ms")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, 300)
        self.slider_speed.setSingleStep(5)
        self.slider_speed.setValue(self.scan_step_ms)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)

        self.lbl_density = QLabel("Sampling density: 60")
        self.slider_density = QSlider(Qt.Horizontal)
        self.slider_density.setRange(10, 100)
        self.slider_density.setSingleStep(1)
        self.slider_density.setValue(self.scan_density)
        self.slider_density.valueChanged.connect(self._on_density_changed)

        self.radio_mode_grid = QRadioButton("Use grid locations")
        self.radio_mode_route = QRadioButton("Use route traversal")
        self.radio_mode_grid.setChecked(True)
        self.radio_mode_grid.toggled.connect(self._on_generation_mode_changed)
        self.radio_mode_route.toggled.connect(self._on_generation_mode_changed)

        self.lbl_route_info = QLabel(
            "Route preview is generated from free-space path planning"
        )
        self.lbl_route_info.setWordWrap(True)

        self.lbl_sensor_count = QLabel("Distance sensors: 8")
        self.slider_sensor_count = QSlider(Qt.Horizontal)
        self.slider_sensor_count.setRange(4, 72)
        self.slider_sensor_count.setSingleStep(1)
        self.slider_sensor_count.setValue(self.beam_count)
        self.slider_sensor_count.valueChanged.connect(self._on_sensor_count_changed)

        group_robot = QGroupBox("Robot configuration")
        group_robot_layout = QVBoxLayout()
        group_robot_layout.addWidget(self.lbl_sensor_count)
        group_robot_layout.addWidget(self.slider_sensor_count)
        group_robot.setLayout(group_robot_layout)

        group_db = QGroupBox("Database generation")
        group_db_layout = QVBoxLayout()
        group_db_layout.addWidget(self.btn_generate_db)
        group_db_layout.addWidget(self.lbl_db)
        group_db_layout.addWidget(self.lbl_speed)
        group_db_layout.addWidget(self.slider_speed)
        group_db_layout.addWidget(self.lbl_density)
        group_db_layout.addWidget(self.slider_density)
        group_db_layout.addWidget(self.radio_mode_grid)
        group_db_layout.addWidget(self.radio_mode_route)
        group_db_layout.addWidget(self.lbl_route_info)
        group_db.setLayout(group_db_layout)

        group_match = QGroupBox("Scan-Matching")
        group_match_layout = QVBoxLayout()
        group_match_layout.addWidget(self.btn_match)
        group_match_layout.addWidget(self.radio_match_topn)
        group_match_layout.addWidget(self.lbl_match_topn)
        group_match_layout.addWidget(self.slider_match_topn)
        group_match_layout.addWidget(self.radio_match_threshold)
        group_match_layout.addWidget(self.lbl_match_theta)
        group_match_layout.addWidget(self.slider_match_theta)
        group_match_layout.addWidget(self.radio_display_current)
        group_match_layout.addWidget(self.radio_display_belief)
        group_match_layout.addWidget(self.lbl_pose)
        group_match_layout.addWidget(self.lbl_estimate)
        group_match.setLayout(group_match_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(group_robot)
        right_layout.addWidget(group_db)
        right_layout.addWidget(group_match)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.hide()
        right_layout.addStretch(1)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setFixedWidth(360)

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.addWidget(self.world_widget, stretch=1)
        lay.addWidget(right_panel, stretch=0)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(45)

        self.db_timer = QTimer(self)
        self.db_timer.setInterval(self.scan_step_ms)
        self.db_timer.timeout.connect(self._on_db_generation_step)

        self.setStyleSheet(
            "QGroupBox {"
            "  border: 1px solid #7a7f87;"
            "  border-radius: 6px;"
            "  margin-top: 10px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 10px;"
            "  padding: 0 3px 0 3px;"
            "}"
        )

        self._refresh_route_preview()
        self._update_matching_controls_state()
        self._update_scan_only()
        self._update_labels()

    def _update_scan_only(self) -> None:
        self.world_widget.current_scan = self.world.cast_scan(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            beam_count=self.beam_count,
            max_range=self.max_range,
            noise_std=0.0,
        )
        self.world_widget.update()

    def _make_descriptor(self, scan: List[float], theta: float) -> List[float]:
        # Compass is known, so include theta as circular features to avoid wrap-around issues.
        # Weight controls how strongly orientation gates matching.
        w = self.orientation_feature_weight
        return scan + [
            w * 0.5 * (math.cos(theta) + 1.0),
            w * 0.5 * (math.sin(theta) + 1.0),
        ]

    def _angle_diff_deg(self, a: float, b: float) -> float:
        return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))

    def _set_generation_controls_enabled(self, enabled: bool) -> None:
        # Keep speed adjustable even during generation for live teaching demos.
        self.slider_speed.setEnabled(True)
        self.slider_density.setEnabled(enabled)
        self.slider_sensor_count.setEnabled(enabled)
        self.radio_mode_grid.setEnabled(enabled)
        self.radio_mode_route.setEnabled(enabled)

    def _on_speed_changed(self, value: int) -> None:
        self.scan_step_ms = int(value)
        if self.scan_step_ms == 0:
            self.lbl_speed.setText("Generation step time: 0 ms (max speed)")
        else:
            self.lbl_speed.setText(f"Generation step time: {self.scan_step_ms} ms")
        self.db_timer.setInterval(self.scan_step_ms)

    def _on_density_changed(self, value: int) -> None:
        if self.db_generation_running:
            return
        self.scan_density = int(value)
        self.lbl_density.setText(f"Sampling density: {self.scan_density}")
        self._refresh_route_preview()

        self.db_samples = []
        self.kdtree = None
        self.world_widget.set_belief_cells([])
        self.lbl_db.setText("Database: invalidated (density changed)")

        if self.matching_running:
            self.btn_match.setChecked(False)
            self.toggle_matching(False)

    def _grid_step_from_density(self) -> float:
        # 10 -> sparse (52 px), 100 -> dense (14 px)
        return 52.0 - (self.scan_density - 10.0) * (38.0 / 90.0)

    def _route_stop_spacing_from_density(self) -> float:
        # 10 -> sparse stops, 100 -> dense stops
        return 16.0 - (self.scan_density - 10.0) * (10.0 / 90.0)

    def _on_generation_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        self.db_generation_mode = "route" if self.radio_mode_route.isChecked() else "grid"
        if self.db_generation_mode == "route":
            self.world_widget.db_route_active = True
            self.world_widget.db_route_points = list(self.route_cached_points)
            self.world_widget.update()
        else:
            self.world_widget.db_route_active = False
            self.world_widget.update()
        self.lbl_db.setText(f"Database mode: {self.db_generation_mode} (re-generate database)")

    def _on_match_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        self.match_mode = "threshold" if self.radio_match_threshold.isChecked() else "top_n"
        self._update_matching_controls_state()

    def _on_display_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        self.match_display_mode = "belief" if self.radio_display_belief.isChecked() else "current"
        self.world_widget.show_belief_map = self.match_display_mode == "belief"
        if self.match_display_mode == "current":
            self.world_widget.clear_heatmap()
        self.world_widget.update()

    def _on_match_topn_changed(self, value: int) -> None:
        self.match_top_n = int(value)
        self.lbl_match_topn.setText(f"N neighbors: {self.match_top_n}")

    def _on_match_theta_changed(self, value: int) -> None:
        self.match_theta = max(0.01, float(value) / 100.0)
        self.lbl_match_theta.setText(f"Theta: {self.match_theta:.2f}")

    def _update_matching_controls_state(self) -> None:
        topn_active = self.match_mode == "top_n"
        self.lbl_match_topn.setEnabled(topn_active)
        self.slider_match_topn.setEnabled(topn_active)
        self.lbl_match_theta.setEnabled(not topn_active)
        self.slider_match_theta.setEnabled(not topn_active)

    def _refresh_route_preview(self) -> None:
        self.route_cached_points = self._build_route_path_points()
        grid_step = self._grid_step_from_density()
        self.lbl_route_info.setText(
            f"Grid step (grid mode): {grid_step:.1f} px\n"
            f"Coverage targets (route mode): {self.route_coverage_targets_count}\n"
            f"Collision-safe route points: {len(self.route_cached_points)}\n"
            "Robot drives this route and scans by rotation at each route point"
        )
        self.world_widget.db_route_active = self.radio_mode_route.isChecked()
        self.world_widget.db_route_points = list(self.route_cached_points)

    def _build_route_path_points(self) -> List[Tuple[float, float]]:
        step = max(14.0, self._grid_step_from_density() * 0.9)
        margin = 22.0
        radius = self.robot.radius

        max_ix = int((self.world.width - 2.0 * margin) // step)
        max_iy = int((self.world.height - 2.0 * margin) // step)

        def node_to_xy(node: Tuple[int, int]) -> Tuple[float, float]:
            ix, iy = node
            return margin + ix * step, margin + iy * step

        free_nodes = set()
        for iy in range(max_iy + 1):
            for ix in range(max_ix + 1):
                x, y = node_to_xy((ix, iy))
                if self.world.is_free(x, y, radius):
                    free_nodes.add((ix, iy))

        if not free_nodes:
            return []

        # Build a serpentine set of coverage targets row by row.
        row_groups: Dict[int, List[Tuple[int, int]]] = {}
        for ix, iy in free_nodes:
            row_groups.setdefault(iy, []).append((ix, iy))

        density_ratio = (self.scan_density - 10.0) / 90.0
        stride = max(2, int(round(9.0 - 7.0 * density_ratio)))
        row_stride = max(1, int(round(4.0 - 3.0 * density_ratio)))

        sweep_nodes: List[Tuple[int, int]] = []
        chosen_rows = sorted(row_groups.keys())[::row_stride]
        for row_idx, iy in enumerate(chosen_rows):
            if row_idx % row_stride != 0:
                continue
            xs = sorted(n[0] for n in row_groups[iy])
            if not xs:
                continue

            # Split row into contiguous free segments.
            segments: List[List[int]] = []
            seg: List[int] = [xs[0]]
            for x in xs[1:]:
                if x == seg[-1] + 1:
                    seg.append(x)
                else:
                    segments.append(seg)
                    seg = [x]
            segments.append(seg)

            left_to_right = (row_idx % 2 == 0)
            seg_iter = segments if left_to_right else list(reversed(segments))
            for segment in seg_iter:
                ordered = segment if left_to_right else list(reversed(segment))
                sampled_xs = ordered[::stride]
                if ordered[-1] not in sampled_xs:
                    sampled_xs.append(ordered[-1])
                for ix in sampled_xs:
                    sweep_nodes.append((ix, iy))

        self.route_coverage_targets_count = len(sweep_nodes)
        if not sweep_nodes:
            return []

        def astar(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
            if start == goal:
                return [start]

            open_heap: List[Tuple[float, Tuple[int, int]]] = []
            heapq.heappush(open_heap, (0.0, start))
            came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
            g_score: Dict[Tuple[int, int], float] = {start: 0.0}

            def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

            while open_heap:
                _, current = heapq.heappop(open_heap)
                if current == goal:
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()
                    return path

                cx, cy = current
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    neighbor = (nx, ny)
                    if neighbor not in free_nodes:
                        continue

                    tentative_g = g_score[current] + 1.0
                    if tentative_g < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_heap, (f, neighbor))

            return []

        route_nodes: List[Tuple[int, int]] = []
        route_nodes.append(sweep_nodes[0])
        for next_node in sweep_nodes[1:]:
            seg = astar(route_nodes[-1], next_node)
            if not seg:
                continue
            if route_nodes and seg and route_nodes[-1] == seg[0]:
                route_nodes.extend(seg[1:])
            else:
                route_nodes.extend(seg)

        # Remove immediate A->B->A backtracking artifacts.
        simplified: List[Tuple[int, int]] = []
        for node in route_nodes:
            simplified.append(node)
            while len(simplified) >= 3 and simplified[-1] == simplified[-3]:
                simplified.pop()
                simplified.pop()
        route_nodes = simplified

        coarse_points = [node_to_xy(node) for node in route_nodes]
        return self._densify_polyline(coarse_points, spacing=self._route_stop_spacing_from_density())

    def _densify_polyline(
        self,
        points: List[Tuple[float, float]],
        spacing: float,
    ) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points

        dense: List[Tuple[float, float]] = [points[0]]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue

            steps = max(1, int(dist // spacing))
            for s in range(1, steps + 1):
                t = s / steps
                nx = x1 + t * dx
                ny = y1 + t * dy
                if self.world.is_free(nx, ny, self.robot.radius):
                    dense.append((nx, ny))

        return dense

    def _on_sensor_count_changed(self, value: int) -> None:
        if self.db_generation_running:
            return
        self.beam_count = int(value)
        self.world_widget.beam_count = self.beam_count
        self.lbl_sensor_count.setText(f"Distance sensors: {self.beam_count}")

        # Existing database was generated with a different scan vector dimension.
        self.db_samples = []
        self.kdtree = None
        self.world_widget.set_belief_cells([])
        self.lbl_db.setText("Database: invalidated (sensor count changed)")

        if self.matching_running:
            self.btn_match.setChecked(False)
            self.toggle_matching(False)

        self._update_scan_only()

    def generate_database(self) -> None:
        if self.db_generation_running:
            return

        if self.matching_running:
            self.btn_match.setChecked(False)
            self.toggle_matching(False)

        points: List[Tuple[float, float]] = []
        r = self.robot.radius

        if self.db_generation_mode == "grid":
            step_x = self._grid_step_from_density()
            step_y = self._grid_step_from_density()
            y = 22.0
            while y < self.world.height - 22.0:
                x = 22.0
                while x < self.world.width - 22.0:
                    if self.world.is_free(x, y, r):
                        points.append((x, y))
                    x += step_x
                y += step_y
        else:
            if not self.route_cached_points:
                self._refresh_route_preview()
            points = list(self.route_cached_points)

        if not points:
            self.db_samples = []
            self.kdtree = None
            self.world_widget.set_belief_cells([])
            self.lbl_db.setText("Database: no valid acquisition points found")
            return

        self.db_generation_running = True
        self.db_generation_points = points
        self.db_generation_angles = [
            math.radians(deg) for deg in range(0, 360, self.scan_theta_step_deg)
        ]
        self.db_generation_samples = []
        self.db_generation_index = 0
        self.db_generation_total_steps = len(self.db_generation_points) * len(self.db_generation_angles)
        self.db_saved_pose = (self.robot.x, self.robot.y, self.robot.theta)
        self.db_route_phase = "scan"
        self.db_route_scan_point_index = 0
        self.db_route_scan_angle_index = 0

        self.btn_generate_db.setEnabled(False)
        self.btn_generate_db.setText("Generating database...")
        self.btn_match.setEnabled(False)
        self._set_generation_controls_enabled(False)

        self.world_widget.estimate_xy = None
        self.world_widget.candidate_points = []
        self.world_widget.db_scanned_points = []
        self.world_widget.db_probe_xy = None
        self.world_widget.db_route_active = self.db_generation_mode == "route"
        self.world_widget.db_route_points = list(self.db_generation_points)

        if self.db_generation_mode == "route":
            sx, sy = self.db_generation_points[0]
            self.robot.x = sx
            self.robot.y = sy
            self.robot.theta = self.db_generation_angles[0]

        self.lbl_db.setText(f"Database: generating 0/{self.db_generation_total_steps}")
        self.db_timer.start()

    def toggle_matching(self, checked: bool) -> None:
        if self.db_generation_running:
            self.btn_match.setChecked(False)
            self.matching_running = False
            self.lbl_estimate.setText("Estimated x,y: wait for DB generation to finish")
            return

        if checked and self.kdtree is None:
            self.btn_match.setChecked(False)
            self.matching_running = False
            self.lbl_estimate.setText("Estimated x,y: first generate database")
            return

        self.matching_running = checked
        self.btn_match.setText("Stop Scan-Matching" if checked else "Scan-Matching")
        if not checked:
            # Keep last matching result visible for teaching/inspection.
            self.world_widget.update()

    def _on_db_generation_step(self) -> None:
        if self.db_generation_mode == "route":
            self._on_db_generation_step_route()
        else:
            self._on_db_generation_step_grid()

    def _finish_db_generation(self) -> None:
        self.db_timer.stop()
        self.db_samples = self.db_generation_samples
        if self.db_samples:
            self.kdtree = ScanKDTree([(vec, xy) for vec, xy, _ in self.db_samples])
            unique_xy = sorted({xy for _, xy, _ in self.db_samples})
            self.world_widget.set_belief_cells(unique_xy)
        else:
            self.kdtree = None
            self.world_widget.set_belief_cells([])

        if self.db_saved_pose is not None:
            sx, sy, st = self.db_saved_pose
            self.robot.x = sx
            self.robot.y = sy
            self.robot.theta = st
        self.db_saved_pose = None

        self.world_widget.db_probe_xy = None
        self.world_widget.db_route_active = False
        self._update_scan_only()

        self.db_generation_running = False
        self.btn_generate_db.setEnabled(True)
        self.btn_generate_db.setText("Generate scan database")
        self.btn_match.setEnabled(True)
        self._set_generation_controls_enabled(True)
        self.lbl_db.setText(f"Database: {len(self.db_samples)} scans in KD-tree")

    def _on_db_generation_step_grid(self) -> None:
        total = self.db_generation_total_steps
        idx = self.db_generation_index

        if idx >= total:
            self._finish_db_generation()
            return

        point_count = len(self.db_generation_points)
        angle_count = len(self.db_generation_angles)
        point_idx = idx // angle_count
        angle_idx = idx % angle_count

        x, y = self.db_generation_points[point_idx]
        theta = self.db_generation_angles[angle_idx]
        self.robot.x = x
        self.robot.y = y
        self.robot.theta = theta

        scan = self.world.cast_scan(
            x,
            y,
            theta,
            beam_count=self.beam_count,
            max_range=self.max_range,
            noise_std=0.0,
        )
        descriptor = self._make_descriptor(scan, theta)
        self.db_generation_samples.append((descriptor, (x, y), theta))

        self.world_widget.current_scan = scan
        self.world_widget.db_probe_xy = (x, y)
        if angle_idx == 0:
            self.world_widget.db_scanned_points.append((x, y))
        self.world_widget.update()

        self.db_generation_index += 1
        self.lbl_db.setText(f"Database: generating {self.db_generation_index}/{total}")

    def _on_db_generation_step_route(self) -> None:
        point_count = len(self.db_generation_points)
        angle_count = len(self.db_generation_angles)
        total = self.db_generation_total_steps

        if self.db_route_scan_point_index >= point_count:
            self._finish_db_generation()
            return

        if self.db_route_phase == "move":
            tx, ty = self.db_generation_points[self.db_route_scan_point_index]
            dx = tx - self.robot.x
            dy = ty - self.robot.y
            dist = math.hypot(dx, dy)

            if dist <= self.route_move_step_px:
                self.robot.x = tx
                self.robot.y = ty
                self.db_route_phase = "scan"
                self.db_route_scan_angle_index = 0
                self.world_widget.db_probe_xy = (tx, ty)
            else:
                ux = dx / dist
                uy = dy / dist
                nx = self.robot.x + ux * self.route_move_step_px
                ny = self.robot.y + uy * self.route_move_step_px
                if self.world.is_free(nx, ny, self.robot.radius):
                    self.robot.x = nx
                    self.robot.y = ny
                else:
                    # Safety fallback in case of numeric edge cases near obstacles.
                    self.robot.x = tx
                    self.robot.y = ty
                    self.db_route_phase = "scan"
                    self.db_route_scan_angle_index = 0
                self.robot.theta = math.atan2(uy, ux)
                moving_scan = self.world.cast_scan(
                    self.robot.x,
                    self.robot.y,
                    self.robot.theta,
                    beam_count=self.beam_count,
                    max_range=self.max_range,
                    noise_std=0.0,
                )
                self.world_widget.current_scan = moving_scan
                self.world_widget.db_probe_xy = (self.robot.x, self.robot.y)
                self.world_widget.update()
                self.lbl_db.setText(f"Database: generating {self.db_generation_index}/{total}")
                return

        x, y = self.db_generation_points[self.db_route_scan_point_index]
        theta = self.db_generation_angles[self.db_route_scan_angle_index]
        self.robot.x = x
        self.robot.y = y
        self.robot.theta = theta

        scan = self.world.cast_scan(
            x,
            y,
            theta,
            beam_count=self.beam_count,
            max_range=self.max_range,
            noise_std=0.0,
        )
        descriptor = self._make_descriptor(scan, theta)
        self.db_generation_samples.append((descriptor, (x, y), theta))

        if self.db_route_scan_angle_index == 0:
            self.world_widget.db_scanned_points.append((x, y))

        self.world_widget.current_scan = scan
        self.world_widget.db_probe_xy = (x, y)
        self.world_widget.update()

        self.db_generation_index += 1
        self.lbl_db.setText(f"Database: generating {self.db_generation_index}/{total}")

        self.db_route_scan_angle_index += 1
        if self.db_route_scan_angle_index >= angle_count:
            self.db_route_scan_angle_index = 0
            self.db_route_scan_point_index += 1
            if self.db_route_scan_point_index < point_count:
                self.db_route_phase = "move"

    def on_timer(self) -> None:
        self.tick_counter += 1

        if self.matching_running:
            self._move_robot()
            self._scan_and_match()
        else:
            self._update_scan_only()

        self._update_labels()

    def _move_robot(self) -> None:
        front_dist = self._front_distance()

        if front_dist < 35.0:
            self.robot.theta += random.uniform(0.7, 1.8)
        else:
            if random.random() < 0.06:
                self.robot.theta += random.uniform(-0.28, 0.28)

            speed = 2.3
            nx = self.robot.x + speed * math.cos(self.robot.theta)
            ny = self.robot.y + speed * math.sin(self.robot.theta)

            if self.world.is_free(nx, ny, self.robot.radius):
                self.robot.x = nx
                self.robot.y = ny
            else:
                self.robot.theta += random.uniform(1.0, 2.3)

        # Keep theta in [-pi, pi] for display readability
        self.robot.theta = math.atan2(math.sin(self.robot.theta), math.cos(self.robot.theta))

    def _front_distance(self) -> float:
        dx = math.cos(self.robot.theta)
        dy = math.sin(self.robot.theta)

        best = self.max_range
        for x1, y1, x2, y2 in self.world.segments:
            hit = ray_segment_distance(self.robot.x, self.robot.y, dx, dy, x1, y1, x2, y2)
            if hit is not None and hit < best:
                best = hit
        return best

    def _scan_and_match(self) -> None:
        scan = self.world.cast_scan(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            beam_count=self.beam_count,
            max_range=self.max_range,
            noise_std=0.01,
        )
        self.world_widget.current_scan = scan
        query_descriptor = self._make_descriptor(scan, self.robot.theta)

        neighbors: List[Tuple[float, Tuple[float, float]]] = []

        oriented_samples: List[Tuple[List[float], Tuple[float, float], float]] = []
        for vec, xy, theta_db in self.db_samples:
            if self._angle_diff_deg(self.robot.theta, theta_db) <= self.match_orientation_tolerance_deg:
                oriented_samples.append((vec, xy, theta_db))

        if not oriented_samples:
            self.world_widget.candidate_points = []
            self.world_widget.estimate_xy = None
            if self.match_display_mode == "belief":
                self.world_widget.update_belief_heatmap([])
            self.world_widget.update()
            return

        if self.match_mode == "top_n":
            scored: List[Tuple[float, Tuple[float, float]]] = []
            for vec, xy, _ in oriented_samples:
                d2 = ScanKDTree._sqdist(query_descriptor, vec)
                scored.append((d2, xy))
            scored.sort(key=lambda t: t[0])
            k = max(1, min(self.match_top_n, len(scored)))
            neighbors = scored[:k]
        else:
            theta2 = self.match_theta * self.match_theta
            for vec, xy, _ in oriented_samples:
                d2 = ScanKDTree._sqdist(query_descriptor, vec)
                if d2 <= theta2:
                    neighbors.append((d2, xy))
            neighbors.sort(key=lambda t: t[0])
            if len(neighbors) > 80:
                neighbors = neighbors[:80]

        if not neighbors:
            self.world_widget.candidate_points = []
            self.world_widget.estimate_xy = None
            if self.match_display_mode == "belief":
                self.world_widget.update_belief_heatmap([])
            self.world_widget.update()
            return

        d2s = [d2 for d2, _ in neighbors]
        best_d2 = max(1e-9, d2s[0])

        weighted: List[Tuple[float, float, float]] = []
        sx = 0.0
        sy = 0.0
        sw = 0.0

        for d2, (x, y) in neighbors:
            rel = math.exp(-d2 / (best_d2 * 6.0))
            weighted.append((x, y, rel))
            sx += rel * x
            sy += rel * y
            sw += rel

        if sw > 1e-9:
            ex = sx / sw
            ey = sy / sw
            self.world_widget.estimate_xy = (ex, ey)
        else:
            self.world_widget.estimate_xy = None

        max_w = max(w for _, _, w in weighted)
        if self.match_display_mode == "current" and max_w > 1e-9:
            self.world_widget.candidate_points = [(x, y, w / max_w) for x, y, w in weighted]
        else:
            self.world_widget.candidate_points = []

        if self.match_display_mode == "belief":
            self.world_widget.update_belief_heatmap([(x, y) for _, (x, y) in neighbors])

        self.world_widget.update()

    def _update_labels(self) -> None:
        self.lbl_pose.setText(
            f"Robot pose: x={self.robot.x:6.1f}, y={self.robot.y:6.1f}, "
            f"theta={math.degrees(self.robot.theta):6.1f} deg"
        )

        est = self.world_widget.estimate_xy
        if est is None:
            self.lbl_estimate.setText("Estimated x,y: n/a")
        else:
            dx = est[0] - self.robot.x
            dy = est[1] - self.robot.y
            err = math.hypot(dx, dy)
            self.lbl_estimate.setText(
                f"Estimated x,y: ({est[0]:.1f}, {est[1]:.1f}) | error={err:.1f} px"
            )


def main() -> None:
    random.seed(7)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
