from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


WINDOW_TITLE = "Kalman Filter 1D Demo"
IMG_WIDTH = 900
IMG_HEIGHT = 620
VISIBLE_X_WIDTH = 1000.0
LANE_COUNT = 4
LINE_WIDTH = 3
CONTROL_SIGNAL = 3.0
NOISE_PROCESS = 10.0
NOISE_MEASUREMENT = 20.0
DT_SECONDS = 1.0
INITIAL_MU = 5.0
INITIAL_SIGMA = 0.75

COL_BACKGROUND = QColor(18, 22, 28)
COL_PANEL = QColor(28, 34, 42)
COL_TEXT = QColor(235, 237, 240)
COL_SUBTLE = QColor(146, 156, 168)
COL_TIME_AXIS = QColor(90, 100, 114)
COL_GT_POS = QColor(245, 245, 245)
COL_NAIVE_EST_POS = QColor(255, 215, 64)
COL_MEASUREMENT = QColor(235, 87, 87)
COL_KF_EST_POS = QColor(86, 204, 118)
COL_KF_UNCERTAINTY = QColor(80, 210, 235)


class KalmanFilter1D:
    def __init__(self, init_mu: float, init_sigma: float, process_noise: float, measurement_noise: float) -> None:
        self.mu = init_mu
        self.sigma = init_sigma
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_signal: float) -> None:
        self.mu = self.mu + control_signal
        self.sigma = self.sigma + self.process_noise

    def correct_by_measurement(self, measurement: float) -> None:
        self.mu = (
            self.measurement_noise * self.mu + self.sigma * measurement
        ) / (self.measurement_noise + self.sigma)
        self.sigma = 1.0 / (1.0 / self.measurement_noise + 1.0 / self.sigma)

    def get_current_state_estimate(self) -> float:
        return self.mu

    def get_current_uncertainty(self) -> float:
        return self.sigma


@dataclass
class SimulationState:
    step: int = 0
    sim_time_seconds: float = 0.0
    ground_truth_pos: float = INITIAL_MU
    naive_est_pos: float = INITIAL_MU
    measurement: float = INITIAL_MU
    kf_est_pos: float = INITIAL_MU
    kf_uncertainty: float = INITIAL_SIGMA
    error_measurement_avg: float = 0.0
    error_naive_avg: float = 0.0
    error_kf_avg: float = 0.0
    measurement_errors: list[float] = field(default_factory=list)
    naive_errors: list[float] = field(default_factory=list)
    kf_errors: list[float] = field(default_factory=list)


class KalmanSimulation:
    def __init__(self) -> None:
        self.process_rng = random.Random()
        self.measurement_rng = random.Random()
        self.process_noise = NOISE_PROCESS
        self.measurement_noise = NOISE_MEASUREMENT
        self.reset()

    def reset(self) -> None:
        self.state = SimulationState()
        self.kf = KalmanFilter1D(
            init_mu=INITIAL_MU,
            init_sigma=INITIAL_SIGMA,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )

    def set_process_noise(self, value: float) -> None:
        self.process_noise = float(value)
        self.kf.process_noise = self.process_noise

    def set_measurement_noise(self, value: float) -> None:
        self.measurement_noise = float(value)
        self.kf.measurement_noise = self.measurement_noise

    def step_once(self) -> SimulationState:
        state = self.state
        state.naive_est_pos += CONTROL_SIGNAL
        state.ground_truth_pos += CONTROL_SIGNAL + self.process_rng.gauss(0.0, self.process_noise)
        state.measurement = state.ground_truth_pos + self.measurement_rng.gauss(0.0, self.measurement_noise)

        self.kf.predict(CONTROL_SIGNAL)
        self.kf.correct_by_measurement(state.measurement)

        state.kf_est_pos = self.kf.get_current_state_estimate()
        state.kf_uncertainty = self.kf.get_current_uncertainty()

        state.measurement_errors.append(abs(state.measurement - state.ground_truth_pos))
        state.naive_errors.append(abs(state.naive_est_pos - state.ground_truth_pos))
        state.kf_errors.append(abs(state.kf_est_pos - state.ground_truth_pos))

        state.error_measurement_avg = sum(state.measurement_errors) / len(state.measurement_errors)
        state.error_naive_avg = sum(state.naive_errors) / len(state.naive_errors)
        state.error_kf_avg = sum(state.kf_errors) / len(state.kf_errors)

        state.step += 1
        state.sim_time_seconds = state.step * DT_SECONDS
        return state


class VisualizationWidget(QWidget):
    def __init__(self, simulation: KalmanSimulation) -> None:
        super().__init__()
        self.simulation = simulation
        self.setMinimumSize(IMG_WIDTH, IMG_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), COL_BACKGROUND)

        plot_rect = self.rect().adjusted(24, 24, -24, -24)
        painter.fillRect(plot_rect, QColor(24, 29, 36))

        lane_height = plot_rect.height() / LANE_COUNT
        labels = [
            ("Ground truth pos", COL_GT_POS, None),
            ("Kalman filtered pos", COL_KF_EST_POS, self.simulation.state.error_kf_avg),
            ("Naive estimated pos", COL_NAIVE_EST_POS, self.simulation.state.error_naive_avg),
            ("Measured pos", COL_MEASUREMENT, self.simulation.state.error_measurement_avg),
        ]

        self._draw_axes(painter, plot_rect, lane_height)
        min_x, max_x = self._compute_world_window()

        for lane_index, (label, color, error_value) in enumerate(labels):
            y_top = plot_rect.top() + lane_index * lane_height
            label_text = label
            if error_value is not None:
                label_text = f"{label}: error = {error_value:.2f}"
            self._draw_label(painter, plot_rect.left() + 14, y_top + lane_height * 0.34, label_text, color)

        state = self.simulation.state
        self._draw_position_line(
            painter,
            plot_rect,
            lane_height,
            0,
            state.ground_truth_pos,
            min_x,
            max_x,
            COL_GT_POS,
        )
        self._draw_position_line(
            painter,
            plot_rect,
            lane_height,
            1,
            state.kf_est_pos,
            min_x,
            max_x,
            COL_KF_EST_POS,
        )
        self._draw_gaussian(
            painter,
            plot_rect,
            lane_height,
            1,
            state.kf_est_pos,
            max(state.kf_uncertainty, 1e-6),
            min_x,
            max_x,
        )
        self._draw_position_line(
            painter,
            plot_rect,
            lane_height,
            2,
            state.naive_est_pos,
            min_x,
            max_x,
            COL_NAIVE_EST_POS,
        )
        self._draw_position_line(
            painter,
            plot_rect,
            lane_height,
            3,
            state.measurement,
            min_x,
            max_x,
            COL_MEASUREMENT,
        )

        self._draw_world_range(painter, plot_rect, min_x, max_x)

    def _compute_world_window(self) -> tuple[float, float]:
        car_pos = max(0.0, self.simulation.state.ground_truth_pos)
        interval_index = int(car_pos // VISIBLE_X_WIDTH)
        min_x = interval_index * VISIBLE_X_WIDTH
        max_x = min_x + VISIBLE_X_WIDTH
        return min_x, max_x

    def _world_to_screen(self, plot_rect: QRectF, value: float, min_x: float, max_x: float) -> float:
        ratio = (value - min_x) / (max_x - min_x)
        ratio = max(0.0, min(1.0, ratio))
        return plot_rect.left() + ratio * plot_rect.width()

    def _draw_axes(self, painter: QPainter, plot_rect: QRectF, lane_height: float) -> None:
        axis_pen = QPen(COL_TIME_AXIS, 1)
        painter.setPen(axis_pen)
        for lane_index in range(1, LANE_COUNT):
            y = plot_rect.top() + lane_index * lane_height
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)

    def _draw_label(self, painter: QPainter, x: float, y: float, text: str, color: QColor) -> None:
        painter.setPen(color)
        painter.drawText(QPointF(x, y), text)

    def _draw_position_line(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        lane_height: float,
        lane_index: int,
        world_x: float,
        min_x: float,
        max_x: float,
        color: QColor,
    ) -> None:
        x = self._world_to_screen(plot_rect, world_x, min_x, max_x)
        y0 = plot_rect.top() + lane_index * lane_height
        y1 = y0 + lane_height
        painter.setPen(QPen(color, LINE_WIDTH))
        painter.drawLine(QPointF(x, y0), QPointF(x, y1))

    def _draw_gaussian(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        lane_height: float,
        lane_index: int,
        mu: float,
        sigma: float,
        min_x: float,
        max_x: float,
    ) -> None:
        baseline = plot_rect.top() + (lane_index + 1) * lane_height - 8.0
        amplitude = lane_height * 0.72
        path = QPainterPath()
        sample_count = max(120, int(plot_rect.width() / 4))
        max_pdf = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        if max_pdf <= 0.0:
            return

        for index in range(sample_count + 1):
            ratio = index / sample_count
            world_x = min_x + ratio * (max_x - min_x)
            pdf = 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * ((world_x - mu) / sigma) ** 2)
            x = self._world_to_screen(plot_rect, world_x, min_x, max_x)
            y = baseline - amplitude * (pdf / max_pdf)
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QPen(COL_KF_UNCERTAINTY, 2))
        painter.drawPath(path)

    def _draw_world_range(self, painter: QPainter, plot_rect: QRectF, min_x: float, max_x: float) -> None:
        painter.setPen(COL_SUBTLE)
        text = f"visible x-range: [{min_x:.1f}, {max_x:.1f}] m"
        painter.drawText(QPointF(plot_rect.left() + 14, plot_rect.bottom() - 8), text)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.simulation = KalmanSimulation()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 760)
        self._build_ui()
        self._install_shortcuts()
        self._refresh_ui()
        QTimer.singleShot(0, self._set_initial_focus)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        self.visualization = VisualizationWidget(self.simulation)
        root_layout.addWidget(self.visualization, stretch=4)

        side_panel = QFrame()
        side_panel.setStyleSheet(
            f"QFrame {{ background-color: {COL_PANEL.name()}; border-radius: 10px; }}"
            f"QLabel {{ color: {COL_TEXT.name()}; }}"
            f"QPushButton {{ padding: 10px; font-size: 14px; }}"
        )
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(14)

        headline = QLabel("1D Kalman Filter Demo")
        headline.setStyleSheet("font-size: 22px; font-weight: 700;")
        side_layout.addWidget(headline)

        intro = QLabel(
            "Space: ein Simulationsschritt\n"
            "Der Button 'Nächster Schritt' macht dasselbe wie Space."
        )
        intro.setStyleSheet(f"color: {COL_SUBTLE.name()};")
        side_layout.addWidget(intro)

        form_wrapper = QFrame()
        form_layout = QFormLayout(form_wrapper)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)

        self.lbl_sim_time = QLabel()
        self.lbl_step = QLabel()
        self.lbl_truth = QLabel()
        self.lbl_measurement = QLabel()
        self.lbl_naive = QLabel()
        self.lbl_kf = QLabel()
        self.lbl_sigma = QLabel()
        self.lbl_error_meas = QLabel()
        self.lbl_error_naive = QLabel()
        self.lbl_error_kf = QLabel()
        self.spin_process_noise = QDoubleSpinBox()
        self.spin_measurement_noise = QDoubleSpinBox()

        for text, widget in [
            ("Simulationszeit", self.lbl_sim_time),
            ("Simulationsschritt", self.lbl_step),
            ("Ground truth", self.lbl_truth),
            ("Messung", self.lbl_measurement),
            ("Naive Schätzung", self.lbl_naive),
            ("Kalman-Schätzung", self.lbl_kf),
            ("KF Unsicherheit", self.lbl_sigma),
            ("Ø Fehler Messung", self.lbl_error_meas),
            ("Ø Fehler Naiv", self.lbl_error_naive),
            ("Ø Fehler Kalman", self.lbl_error_kf),
        ]:
            label = QLabel(text)
            label.setStyleSheet(f"color: {COL_SUBTLE.name()};")
            form_layout.addRow(label, widget)

        self.spin_process_noise.setRange(0.0, 500.0)
        self.spin_process_noise.setDecimals(2)
        self.spin_process_noise.setSingleStep(0.5)
        self.spin_process_noise.setValue(self.simulation.process_noise)
        self.spin_process_noise.valueChanged.connect(self.set_process_noise)
        form_layout.addRow("Prozessrauschen", self.spin_process_noise)

        self.spin_measurement_noise.setRange(0.0, 500.0)
        self.spin_measurement_noise.setDecimals(2)
        self.spin_measurement_noise.setSingleStep(0.5)
        self.spin_measurement_noise.setValue(self.simulation.measurement_noise)
        self.spin_measurement_noise.valueChanged.connect(self.set_measurement_noise)
        form_layout.addRow("Messrauschen", self.spin_measurement_noise)

        side_layout.addWidget(form_wrapper)

        self.btn_next = QPushButton("Nächster Schritt")
        self.btn_next.clicked.connect(self.advance_one_step)
        side_layout.addWidget(self.btn_next)

        self.btn_reset = QPushButton("Simulation zurücksetzen")
        self.btn_reset.clicked.connect(self.reset_simulation)
        side_layout.addWidget(self.btn_reset)

        note = QLabel(
            f"Parameter:\n"
            f"u = {CONTROL_SIGNAL:.1f} m/Schritt"
        )
        note.setStyleSheet(f"color: {COL_SUBTLE.name()};")
        side_layout.addWidget(note)

        side_layout.addStretch(1)
        root_layout.addWidget(side_panel, stretch=1)

        file_menu = self.menuBar().addMenu("Datei")
        reset_action = QAction("Zurücksetzen", self)
        reset_action.triggered.connect(self.reset_simulation)
        file_menu.addAction(reset_action)

        next_action = QAction("Nächster Schritt", self)
        next_action.triggered.connect(self.advance_one_step)
        file_menu.addAction(next_action)
        self.addAction(next_action)

    def _install_shortcuts(self) -> None:
        shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(self.advance_one_step)
        self.step_shortcut = shortcut

    def _set_initial_focus(self) -> None:
        self.activateWindow()
        self.raise_()
        self.btn_next.setFocus(Qt.OtherFocusReason)

    def advance_one_step(self) -> None:
        self.simulation.step_once()
        self._refresh_ui()

    def reset_simulation(self) -> None:
        self.simulation.reset()
        self._refresh_ui()

    def set_process_noise(self, value: float) -> None:
        self.simulation.set_process_noise(value)

    def set_measurement_noise(self, value: float) -> None:
        self.simulation.set_measurement_noise(value)

    def _refresh_ui(self) -> None:
        state = self.simulation.state
        self.lbl_sim_time.setText(f"{state.sim_time_seconds:.1f} s")
        self.lbl_step.setText(str(state.step))
        self.lbl_truth.setText(f"{state.ground_truth_pos:.2f} m")
        self.lbl_measurement.setText(f"{state.measurement:.2f} m")
        self.lbl_naive.setText(f"{state.naive_est_pos:.2f} m")
        self.lbl_kf.setText(f"{state.kf_est_pos:.2f} m")
        self.lbl_sigma.setText(f"{state.kf_uncertainty:.6f}")
        self.lbl_error_meas.setText(f"{state.error_measurement_avg:.2f} m")
        self.lbl_error_naive.setText(f"{state.error_naive_avg:.2f} m")
        self.lbl_error_kf.setText(f"{state.error_kf_avg:.2f} m")
        self.visualization.update()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())