import math
import sys
from dataclasses import dataclass

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def odometry_step(pose: Pose2D, v_l: float, v_r: float, wheel_base: float, delta_t: float) -> Pose2D:
    if wheel_base <= 0.0:
        raise ValueError("wheel_base must be > 0")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be > 0")

    delta_s = 0.5 * (v_l + v_r) * delta_t
    delta_theta = ((v_r - v_l) / wheel_base) * delta_t

    # Midpoint integration is a practical odometry approximation for one time step.
    theta_mid = pose.theta + 0.5 * delta_theta
    x_next = pose.x + delta_s * math.cos(theta_mid)
    y_next = pose.y + delta_s * math.sin(theta_mid)
    theta_next = wrap_angle(pose.theta + delta_theta)

    return Pose2D(x=x_next, y=y_next, theta=theta_next)


class OdomCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(720, 540)
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.trajectory = [Pose2D(0.0, 0.0, 0.0)]
        self.wheel_base = 0.4
        self.zoom_factor = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 8.0
        self.base_world_span = 20.0

    def set_state(self, pose: Pose2D, trajectory: list[Pose2D], wheel_base: float) -> None:
        self.pose = pose
        self.trajectory = trajectory
        self.wheel_base = max(wheel_base, 0.05)
        self.update()

    def world_to_screen(self, x: float, y: float, bounds: tuple[float, float, float, float], w: int, h: int, margin: int) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = bounds
        extent_x = max(max_x - min_x, 1e-6)
        extent_y = max(max_y - min_y, 1e-6)
        scale = min((w - 2 * margin) / extent_x, (h - 2 * margin) / extent_y)

        sx = margin + (x - min_x) * scale
        sy = h - (margin + (y - min_y) * scale)
        return sx, sy

    def compute_bounds(self) -> tuple[float, float, float, float]:
        xs = [p.x for p in self.trajectory]
        ys = [p.y for p in self.trajectory]

        if not xs:
            xs = [self.pose.x]
            ys = [self.pose.y]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        # Keep some visible space around the trajectory and robot shape.
        padding = max(self.wheel_base * 2.0, 0.6)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        # Guarantee a larger visible world window and apply zoom.
        min_span = self.base_world_span / self.zoom_factor
        span_x = max(max_x - min_x, min_span)
        span_y = max(max_y - min_y, min_span)

        center_x = 0.5 * (max_x + min_x)
        center_y = 0.5 * (max_y + min_y)
        min_x = center_x - 0.5 * span_x
        max_x = center_x + 0.5 * span_x
        min_y = center_y - 0.5 * span_y
        max_y = center_y + 0.5 * span_y

        return min_x, max_x, min_y, max_y

    def set_zoom(self, zoom: float) -> None:
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, zoom))
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_factor * 1.2)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_factor / 1.2)

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.angleDelta().y() > 0:
            self.zoom_in()
        elif event.angleDelta().y() < 0:
            self.zoom_out()
        event.accept()

    def compute_tick_step(self, lower: float, upper: float, target_ticks: int = 10) -> float:
        span = max(upper - lower, 1e-9)
        raw = span / max(target_ticks, 2)
        exponent = math.floor(math.log10(raw))
        base = 10.0 ** exponent

        for factor in (1.0, 2.0, 5.0, 10.0):
            step = factor * base
            if step >= raw:
                return step
        return 10.0 * base

    def tick_values(self, lower: float, upper: float, step: float) -> list[float]:
        if step <= 0.0:
            return []

        values: list[float] = []
        start = math.ceil(lower / step) * step
        v = start
        eps = step * 1e-6
        while v <= upper + eps:
            if abs(v) < eps:
                v = 0.0
            values.append(v)
            v += step
        return values

    def paintEvent(self, event) -> None:  # noqa: N802
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        margin = 35

        painter.fillRect(0, 0, w, h, QColor("#f7fafc"))
        bounds = self.compute_bounds()

        min_x, max_x, min_y, max_y = bounds
        extent_x = max(max_x - min_x, 1e-6)
        extent_y = max(max_y - min_y, 1e-6)
        scale = min((w - 2 * margin) / extent_x, (h - 2 * margin) / extent_y)

        # Draw world axes if visible in current window.
        axis_pen = QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine)
        painter.setPen(axis_pen)
        axis_y_screen = None
        axis_x_screen = None
        if min_y <= 0.0 <= max_y:
            x0, y0 = self.world_to_screen(min_x, 0.0, bounds, w, h, margin)
            x1, y1 = self.world_to_screen(max_x, 0.0, bounds, w, h, margin)
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
            axis_y_screen = y0
        if min_x <= 0.0 <= max_x:
            x0, y0 = self.world_to_screen(0.0, min_y, bounds, w, h, margin)
            x1, y1 = self.world_to_screen(0.0, max_y, bounds, w, h, margin)
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
            axis_x_screen = x0

        # Draw tick marks and numeric labels on visible axes.
        painter.setPen(QPen(QColor("#64748b"), 1))
        x_step = self.compute_tick_step(min_x, max_x)
        y_step = self.compute_tick_step(min_y, max_y)

        if axis_y_screen is not None:
            for tick_x in self.tick_values(min_x, max_x, x_step):
                sx, sy = self.world_to_screen(tick_x, 0.0, bounds, w, h, margin)
                painter.drawLine(int(sx), int(sy - 4), int(sx), int(sy + 4))
                if abs(tick_x) > 1e-9:
                    painter.drawText(int(sx - 16), int(sy + 18), f"{tick_x:.2g}")

        if axis_x_screen is not None:
            for tick_y in self.tick_values(min_y, max_y, y_step):
                sx, sy = self.world_to_screen(0.0, tick_y, bounds, w, h, margin)
                painter.drawLine(int(sx - 4), int(sy), int(sx + 4), int(sy))
                if abs(tick_y) > 1e-9:
                    painter.drawText(int(sx + 8), int(sy + 4), f"{tick_y:.2g}")

        # Draw trajectory.
        if len(self.trajectory) >= 2:
            traj_pen = QPen(QColor("#2563eb"), 2)
            painter.setPen(traj_pen)
            for i in range(1, len(self.trajectory)):
                p0 = self.trajectory[i - 1]
                p1 = self.trajectory[i]
                x0, y0 = self.world_to_screen(p0.x, p0.y, bounds, w, h, margin)
                x1, y1 = self.world_to_screen(p1.x, p1.y, bounds, w, h, margin)
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Draw start marker.
        if self.trajectory:
            start = self.trajectory[0]
            sx, sy = self.world_to_screen(start.x, start.y, bounds, w, h, margin)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#16a34a"))
            painter.drawEllipse(int(sx - 5), int(sy - 5), 10, 10)

        # Draw robot body as circle.
        cx, cy = self.world_to_screen(self.pose.x, self.pose.y, bounds, w, h, margin)
        radius_world = max(self.wheel_base * 0.12, 0.03)
        radius_px = max(int(radius_world * scale), 6)

        painter.setPen(QPen(QColor("#0f172a"), 2))
        painter.setBrush(QColor("#e2e8f0"))
        painter.drawEllipse(int(cx - radius_px), int(cy - radius_px), 2 * radius_px, 2 * radius_px)

        # Draw orientation line.
        heading_len_px = int(radius_px * 1.5)
        hx = cx + heading_len_px * math.cos(self.pose.theta)
        hy = cy - heading_len_px * math.sin(self.pose.theta)
        painter.setPen(QPen(QColor("#dc2626"), 3))
        painter.drawLine(int(cx), int(cy), int(hx), int(hy))

        # Labels in corner.
        painter.setPen(QPen(QColor("#0f172a"), 1))
        painter.drawText(12, 22, "x/y in meters")
        painter.drawText(12, 40, f"trajectory points: {len(self.trajectory)}")
        painter.drawText(12, 58, f"zoom: {self.zoom_factor:.2f}x")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D Differential Drive Odometry Simulator")
        self.resize(1220, 700)

        self.current_pose = Pose2D(0.0, 0.0, 0.0)
        self.trajectory = [Pose2D(0.0, 0.0, 0.0)]
        self.sim_time = 0.0

        self.canvas = OdomCanvas()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.single_step)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        root_layout.addWidget(self.canvas, 3)
        root_layout.addWidget(self.build_controls(), 2)

        self.setCentralWidget(root)

        self.reset_pose()

    def make_float_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 4,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def make_output(self) -> QLineEdit:
        line = QLineEdit()
        line.setReadOnly(True)
        line.setAlignment(Qt.AlignmentFlag.AlignRight)
        return line

    def build_controls(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)

        group_init = QGroupBox("Input Pose")
        init_form = QFormLayout(group_init)
        self.x0_spin = self.make_float_spin(-1000.0, 1000.0, 0.0, 0.1)
        self.y0_spin = self.make_float_spin(-1000.0, 1000.0, 0.0, 0.1)
        self.theta0_spin = self.make_float_spin(-10.0 * math.pi, 10.0 * math.pi, 0.0, 0.05)
        init_form.addRow("x0 [m]", self.x0_spin)
        init_form.addRow("y0 [m]", self.y0_spin)
        init_form.addRow("theta0 [rad]", self.theta0_spin)

        group_model = QGroupBox("Model Parameters")
        model_form = QFormLayout(group_model)
        self.v_l_spin = self.make_float_spin(-10.0, 10.0, 0.4, 0.05)
        self.v_r_spin = self.make_float_spin(-10.0, 10.0, 0.2, 0.05)
        self.b_spin = self.make_float_spin(0.05, 10.0, 0.5, 0.01)
        self.delta_t_spin = self.make_float_spin(0.001, 10.0, 0.1, 0.01, decimals=3)
        self.timer_ms_spin = QSpinBox()
        self.timer_ms_spin.setRange(10, 5000)
        self.timer_ms_spin.setValue(100)
        self.timer_ms_spin.setSingleStep(10)
        model_form.addRow("vL [m/s]", self.v_l_spin)
        model_form.addRow("vR [m/s]", self.v_r_spin)
        model_form.addRow("b [m]", self.b_spin)
        model_form.addRow("delta_t [s]", self.delta_t_spin)
        model_form.addRow("timer [ms]", self.timer_ms_spin)

        group_output = QGroupBox("Output Pose")
        out_form = QFormLayout(group_output)
        self.x1_out = self.make_output()
        self.y1_out = self.make_output()
        self.theta1_out = self.make_output()
        self.time_out = self.make_output()
        out_form.addRow("x1 [m]", self.x1_out)
        out_form.addRow("y1 [m]", self.y1_out)
        out_form.addRow("theta1 [rad]", self.theta1_out)
        out_form.addRow("t [s]", self.time_out)

        group_log = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout(group_log)
        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels(["step", "t [s]", "x [m]", "y [m]", "theta [rad]"])
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.verticalHeader().setVisible(False)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)

        log_layout.addWidget(self.log_table)
        log_layout.addWidget(self.clear_log_btn)

        button_row_1 = QHBoxLayout()
        self.single_step_btn = QPushButton("Single Step")
        self.single_step_btn.clicked.connect(self.single_step)
        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.clicked.connect(self.toggle_start_stop)
        button_row_1.addWidget(self.single_step_btn)
        button_row_1.addWidget(self.start_stop_btn)

        button_row_2 = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Pose")
        self.reset_btn.clicked.connect(self.reset_pose)
        self.clear_traj_btn = QPushButton("Clear Trajectory")
        self.clear_traj_btn.clicked.connect(self.clear_trajectory)
        button_row_2.addWidget(self.reset_btn)
        button_row_2.addWidget(self.clear_traj_btn)

        button_row_3 = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_in_btn.clicked.connect(self.canvas.zoom_in)
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_out_btn.clicked.connect(self.canvas.zoom_out)
        self.zoom_reset_btn = QPushButton("Zoom Reset")
        self.zoom_reset_btn.clicked.connect(self.canvas.reset_zoom)
        button_row_3.addWidget(self.zoom_in_btn)
        button_row_3.addWidget(self.zoom_out_btn)
        button_row_3.addWidget(self.zoom_reset_btn)

        panel_layout.addWidget(group_init)
        panel_layout.addWidget(group_model)
        panel_layout.addLayout(button_row_1)
        panel_layout.addLayout(button_row_2)
        panel_layout.addLayout(button_row_3)
        panel_layout.addWidget(group_output)
        panel_layout.addWidget(group_log)
        panel_layout.addStretch(1)

        hint = QLabel(
            "Hinweis: Bei Start wird fortlaufend mit den aktuellen\n"
            "vL, vR, b und delta_t simuliert; die Trajektorie\n"
            "bleibt sichtbar, bis Reset oder Clear gedrueckt wird."
        )
        hint.setStyleSheet("color: #334155;")
        panel_layout.addWidget(hint)

        return panel

    def refresh_output_and_canvas(self) -> None:
        self.x1_out.setText(f"{self.current_pose.x:.5f}")
        self.y1_out.setText(f"{self.current_pose.y:.5f}")
        self.theta1_out.setText(f"{self.current_pose.theta:.5f}")
        self.time_out.setText(f"{self.sim_time:.3f}")
        self.canvas.set_state(self.current_pose, self.trajectory, self.b_spin.value())

    def append_log_row(self) -> None:
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)

        values = [
            str(row),
            f"{self.sim_time:.3f}",
            f"{self.current_pose.x:.5f}",
            f"{self.current_pose.y:.5f}",
            f"{self.current_pose.theta:.5f}",
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.log_table.setItem(row, col, item)

        self.log_table.scrollToBottom()

    def clear_log(self) -> None:
        self.log_table.setRowCount(0)
        self.append_log_row()

    def reset_pose(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.start_stop_btn.setText("Start")

        self.current_pose = Pose2D(
            x=self.x0_spin.value(),
            y=self.y0_spin.value(),
            theta=wrap_angle(self.theta0_spin.value()),
        )
        self.sim_time = 0.0
        self.trajectory = [Pose2D(self.current_pose.x, self.current_pose.y, self.current_pose.theta)]
        self.log_table.setRowCount(0)
        self.append_log_row()
        self.refresh_output_and_canvas()

    def clear_trajectory(self) -> None:
        self.trajectory = [Pose2D(self.current_pose.x, self.current_pose.y, self.current_pose.theta)]
        self.refresh_output_and_canvas()

    def single_step(self) -> None:
        self.current_pose = odometry_step(
            pose=self.current_pose,
            v_l=self.v_l_spin.value(),
            v_r=self.v_r_spin.value(),
            wheel_base=self.b_spin.value(),
            delta_t=self.delta_t_spin.value(),
        )
        self.sim_time += self.delta_t_spin.value()
        self.trajectory.append(Pose2D(self.current_pose.x, self.current_pose.y, self.current_pose.theta))
        self.append_log_row()
        self.refresh_output_and_canvas()

    def toggle_start_stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.start_stop_btn.setText("Start")
            return

        self.timer.start(self.timer_ms_spin.value())
        self.start_stop_btn.setText("Stop")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
