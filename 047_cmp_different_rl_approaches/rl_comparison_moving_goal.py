"""RL comparison demo with PySide6.

This demo compares exactly three Deep RL approaches:
- DQN
- REINFORCE
- PPO

Each approach lives in its own file/class:
- dqn_agent.py -> DQNAgent
- reinforce_agent.py -> ReinforceAgent
- ppo_agent.py -> PPOAgent

Task:
The agent (black circle) has to follow a moving target (red circle).
State space: [dx_t-1, dy_t-1, dx_t, dy_t].
Action space: UP, LEFT, RIGHT, DOWN.
Reward: absolute normalized distance in [0, 1].
"""

from __future__ import annotations

import random
import sys
from collections import deque
from typing import Deque, List

import numpy as np

from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from reinforce_agent import ReinforceAgent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

Action = int
State = np.ndarray


class MovingTargetEnv:
    """A small 2D world with a target that bounces back and forth."""

    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_DOWN = 3

    def __init__(self, width: int = 760, height: int = 500) -> None:
        self.width = width
        self.height = height
        self.agent_radius = 10
        self.target_radius = 12
        self.capture_radius = 2 * self.target_radius
        self.agent_speed = 5.0
        self.target_speed = 2.0
        self.max_steps_per_episode = 100
        self.max_distance = float(np.hypot(self.width, self.height))

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.curr_dx = 0.0
        self.curr_dy = 0.0
        self.steps = 0
        self.episode_reward = 0.0

        self.reset()

    def reset(self) -> State:
        margin = max(self.agent_radius, self.target_radius) + 12
        self.agent_x = self.width * 0.20
        self.agent_y = self.height * 0.50
        self.target_x = random.uniform(margin, self.width - margin)
        self.target_y = random.uniform(margin, self.height - margin)

        vx = random.choice([-1.0, 1.0]) * self.target_speed
        vy = random.choice([-1.0, 1.0]) * self.target_speed * 0.75
        self.target_vx = vx
        self.target_vy = vy

        dx, dy = self._offset_to_target()
        self.prev_dx = dx
        self.prev_dy = dy
        self.curr_dx = dx
        self.curr_dy = dy

        self.steps = 0
        self.episode_reward = 0.0
        return self.state()

    def state(self) -> State:
        return np.array([self.prev_dx, self.prev_dy, self.curr_dx, self.curr_dy], dtype=np.float32)

    def _offset_to_target(self) -> tuple[float, float]:
        dx = (self.target_x - self.agent_x) / float(self.width)
        dy = (self.target_y - self.agent_y) / float(self.height)
        return float(dx), float(dy)

    def distance_to_target(self) -> float:
        return float(np.hypot(self.target_x - self.agent_x, self.target_y - self.agent_y))

    def in_reward_zone(self) -> bool:
        return self.distance_to_target() <= float(self.capture_radius)

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        if action == self.ACTION_UP:
            self.agent_y -= self.agent_speed
        elif action == self.ACTION_LEFT:
            self.agent_x -= self.agent_speed
        elif action == self.ACTION_RIGHT:
            self.agent_x += self.agent_speed
        elif action == self.ACTION_DOWN:
            self.agent_y += self.agent_speed

        self.agent_x = float(np.clip(self.agent_x, self.agent_radius, self.width - self.agent_radius))
        self.agent_y = float(np.clip(self.agent_y, self.agent_radius, self.height - self.agent_radius))

        self._move_target()

        next_dx, next_dy = self._offset_to_target()
        self.prev_dx = self.curr_dx
        self.prev_dy = self.curr_dy
        self.curr_dx = next_dx
        self.curr_dy = next_dy

        distance = self.distance_to_target()
        reward = 1.0 - (distance / self.max_distance)

        self.episode_reward += reward
        self.steps += 1

        done = self.steps >= self.max_steps_per_episode
        info = {
            "distance": distance,
            "in_zone": self.in_reward_zone(),
            "steps": self.steps,
        }
        return self.state(), reward, done, info

    def _move_target(self) -> None:
        self.target_x += self.target_vx
        self.target_y += self.target_vy

        left = self.target_radius
        right = self.width - self.target_radius
        top = self.target_radius
        bottom = self.height - self.target_radius

        if self.target_x < left:
            self.target_x = left
            self.target_vx = abs(self.target_vx)
        elif self.target_x > right:
            self.target_x = right
            self.target_vx = -abs(self.target_vx)

        if self.target_y < top:
            self.target_y = top
            self.target_vy = abs(self.target_vy)
        elif self.target_y > bottom:
            self.target_y = bottom
            self.target_vy = -abs(self.target_vy)


class SimulationView(QWidget):
    def __init__(self, env: MovingTargetEnv) -> None:
        super().__init__()
        self.env = env
        self.state = env.state()
        self.setMinimumSize(520, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, state: State) -> None:
        self.state = state
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(248, 250, 252))

        margin = 18
        world = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.setPen(QPen(QColor(180, 186, 196), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawRoundedRect(world, 12, 12)

        painter.setPen(QPen(QColor(235, 237, 241), 1))
        for frac in (0.25, 0.5, 0.75):
            x = int(world.left() + frac * world.width())
            y = int(world.top() + frac * world.height())
            painter.drawLine(x, world.top(), x, world.bottom())
            painter.drawLine(world.left(), y, world.right(), y)

        scale_x = world.width() / float(self.env.width)
        scale_y = world.height() / float(self.env.height)

        def wx(x: float) -> int:
            return int(world.left() + x * scale_x)

        def wy(y: float) -> int:
            return int(world.top() + y * scale_y)

        capture_pen = QPen(QColor(231, 76, 60, 160), 2)
        capture_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(capture_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        capture_w = int(self.env.capture_radius * 2 * scale_x)
        capture_h = int(self.env.capture_radius * 2 * scale_y)
        painter.drawEllipse(
            wx(self.env.target_x) - capture_w // 2,
            wy(self.env.target_y) - capture_h // 2,
            capture_w,
            capture_h,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(220, 40, 40))
        target_w = int(self.env.target_radius * 2 * scale_x)
        target_h = int(self.env.target_radius * 2 * scale_y)
        painter.drawEllipse(
            wx(self.env.target_x) - target_w // 2,
            wy(self.env.target_y) - target_h // 2,
            target_w,
            target_h,
        )

        painter.setBrush(QColor(20, 20, 20))
        painter.drawEllipse(
            wx(self.env.agent_x) - int(self.env.agent_radius * scale_x),
            wy(self.env.agent_y) - int(self.env.agent_radius * scale_y),
            int(self.env.agent_radius * 2 * scale_x),
            int(self.env.agent_radius * 2 * scale_y),
        )


class RewardPlot(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(5.4, 3.6), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.steps: List[int] = []
        self.step_rewards: List[float] = []
        self.setMinimumHeight(220)

    def clear(self) -> None:
        self.steps.clear()
        self.step_rewards.clear()
        self.redraw()

    def add_point(self, step: int, step_reward: float, redraw: bool = True) -> None:
        self.steps.append(step)
        self.step_rewards.append(step_reward)
        if len(self.steps) > 5000:
            self.steps = self.steps[-5000:]
            self.step_rewards = self.step_rewards[-5000:]
        if redraw:
            self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self.ax.set_title("Reward per training step + rolling average (100)")
        self.ax.set_xlabel("Training steps")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, alpha=0.25)

        if self.steps:
            self.ax.plot(self.steps, self.step_rewards, color="#1565c0", linewidth=1.2, label="Step reward")

            rolling_rewards: List[float] = []
            for i in range(len(self.step_rewards)):
                start = max(0, i - 99)
                rolling_rewards.append(float(np.mean(self.step_rewards[start : i + 1])))

            self.ax.plot(self.steps, rolling_rewards, color="#ef6c00", linewidth=2.0, label="Avg last 100")
            self.ax.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
            self.ax.set_xlim(self.steps[0], self.steps[-1] + 1)
            y_min = min(0.0, min(self.step_rewards) * 1.2)
            y_max = max(0.2, max(self.step_rewards) * 1.2)
            if y_max - y_min < 1e-6:
                y_max = y_min + 0.1
            self.ax.set_ylim(y_min, y_max)
            self.ax.legend(loc="lower right")

        self.draw_idle()


class MainWindow(QMainWindow):
    ALGORITHM_NAMES = ("DQN", "REINFORCE", "PPO")

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("RL comparison demo: DQN / REINFORCE / PPO")
        self.resize(1280, 760)

        self.env = MovingTargetEnv()
        self.agent = self._create_agent("DQN")
        self.running = False
        self.training_steps = 0
        self.episode_index = 1
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_loss: float | None = None
        self.steps_per_tick = 2
        self.render_enabled = True

        self.distance_window: Deque[float] = deque(maxlen=5000)

        self.view = SimulationView(self.env)
        self.plot = RewardPlot()

        self.combo_algorithm = QComboBox()
        for algorithm_name in self.ALGORITHM_NAMES:
            self.combo_algorithm.addItem(algorithm_name)
        self.combo_algorithm.currentIndexChanged.connect(self._algorithm_changed)

        self.btn_start_stop = QPushButton("Start")
        self.btn_start_stop.clicked.connect(self.toggle_running)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_demo)

        self.chk_rendering = QCheckBox("Rendering enabled")
        self.chk_rendering.setChecked(True)
        self.chk_rendering.toggled.connect(self.set_rendering_enabled)

        self.lbl_algorithm = QLabel()
        self.lbl_steps = QLabel()
        self.lbl_episode = QLabel()
        self.lbl_episode_reward = QLabel()
        self.lbl_state = QLabel()
        self.lbl_distance = QLabel()
        self.lbl_avg_distance_5000 = QLabel()
        self.lbl_epsilon = QLabel()
        self.lbl_replay = QLabel()
        self.lbl_loss = QLabel()
        self.lbl_ppo_entropy = QLabel()
        self.lbl_ppo_ratio = QLabel()
        self.lbl_ppo_clip_frac = QLabel()
        self.lbl_last_reward = QLabel()
        self.lbl_device = QLabel(str(self.agent.device))

        self._build_ui()
        self._sync_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(16)

        self.menuBar().addMenu("File").addAction(self._quit_action())

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left = QFrame()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self.view, stretch=3)
        left_layout.addWidget(self.plot, stretch=2)
        layout.addWidget(left, stretch=3)

        side_container = QFrame()
        side_layout = QVBoxLayout(side_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(side_container)
        layout.addWidget(scroll, stretch=2)

        settings_box = QGroupBox("Controls")
        settings_layout = QVBoxLayout(settings_box)

        form = QFormLayout()
        form.addRow("Agent type", self.combo_algorithm)
        settings_layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addWidget(self.btn_start_stop)
        button_row.addWidget(self.btn_reset)
        settings_layout.addLayout(button_row)
        settings_layout.addWidget(self.chk_rendering)

        hint = QLabel(
            "Three standalone Deep RL variants are available:\n"
            "- DQN: value-based Q-learning with replay + target network\n"
            "- REINFORCE: Monte-Carlo policy gradient, one update per episode\n"
            "- PPO: clipped actor-critic updates over mini-batches\n"
            "Reward is fixed to: reward = 1 - d/max_d (range [0, 1]).\n"
            "Comparison metric: mean distance to goal over the last 5000 steps."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #475569;")
        settings_layout.addWidget(hint)

        side_layout.addWidget(settings_box)

        stats_box = QGroupBox("Status")
        stats_layout = QFormLayout(stats_box)
        stats_layout.addRow("Algorithm", self.lbl_algorithm)
        stats_layout.addRow("Training steps", self.lbl_steps)
        stats_layout.addRow("Episode", self.lbl_episode)
        stats_layout.addRow("Episode reward", self.lbl_episode_reward)
        stats_layout.addRow("State [dx_t-1, dy_t-1, dx_t, dy_t]", self.lbl_state)
        stats_layout.addRow("Current distance", self.lbl_distance)
        stats_layout.addRow("Avg distance (last 5000)", self.lbl_avg_distance_5000)
        stats_layout.addRow("Epsilon", self.lbl_epsilon)
        stats_layout.addRow("Replay size", self.lbl_replay)
        stats_layout.addRow("Last loss", self.lbl_loss)
        stats_layout.addRow("PPO entropy", self.lbl_ppo_entropy)
        stats_layout.addRow("PPO ratio mean", self.lbl_ppo_ratio)
        stats_layout.addRow("PPO clip fraction", self.lbl_ppo_clip_frac)
        stats_layout.addRow("Last reward", self.lbl_last_reward)
        stats_layout.addRow("Device", self.lbl_device)
        side_layout.addWidget(stats_box)

        side_layout.addStretch(1)

    def _quit_action(self) -> QAction:
        action = QAction("Quit", self)
        action.triggered.connect(self.close)
        return action

    def _selected_algorithm(self) -> str:
        return str(self.combo_algorithm.currentText())

    def _create_agent(self, algorithm_name: str):
        if algorithm_name == "DQN":
            return DQNAgent()
        if algorithm_name == "REINFORCE":
            return ReinforceAgent()
        if algorithm_name == "PPO":
            return PPOAgent()
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    def _algorithm_changed(self, _index: int) -> None:
        if self.running:
            self.running = False
            self.btn_start_stop.setText("Start")
        self.reset_demo()

    def reset_demo(self) -> None:
        self.env = MovingTargetEnv()
        self.agent = self._create_agent(self._selected_algorithm())
        self.training_steps = 0
        self.episode_index = 1
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_loss = None
        self.distance_window.clear()
        self.plot.clear()
        self.view.env = self.env
        self.view.set_state(self.env.state())
        self._sync_ui()

    def toggle_running(self) -> None:
        self.running = not self.running
        self.btn_start_stop.setText("Stop" if self.running else "Start")

    def set_rendering_enabled(self, enabled: bool) -> None:
        self.render_enabled = enabled
        self.view.setVisible(enabled)

        if enabled:
            self.view.set_state(self.env.state())
            self.plot.redraw()

    def on_tick(self) -> None:
        if not self.running:
            return

        for _ in range(self.steps_per_tick):
            episode_done = self.training_step()
            if episode_done:
                break

        self._sync_ui()

    def _avg_distance_last_5000(self) -> float | None:
        if not self.distance_window:
            return None
        return float(np.mean(self.distance_window))

    def training_step(self) -> bool:
        state = self.env.state()
        action = self.agent.act(state)
        next_state, reward, done, info = self.env.step(action)

        self.agent.store(state, action, reward, next_state, done)
        self.agent.register_env_step()
        loss = self.agent.learn(episode_done=done)
        if loss is not None:
            self.last_loss = loss

        self.training_steps += 1
        self.episode_reward += reward
        self.last_reward = reward

        self.distance_window.append(float(info["distance"]))

        self.plot.add_point(self.training_steps, reward, redraw=self.render_enabled)

        if self.render_enabled:
            self.view.set_state(next_state)

        if done:
            avg_distance = self._avg_distance_last_5000()
            avg_text = f"{avg_distance:.2f} px" if avg_distance is not None else "n/a"
            print(
                f"[{self._selected_algorithm()}] Episode {self.episode_index} finished | "
                f"avg distance over last {len(self.distance_window)} steps: {avg_text}"
            )

            self.episode_index += 1
            self.episode_reward = 0.0
            self.env.reset()
            if self.render_enabled:
                self.view.set_state(self.env.state())

        return done

    def _sync_ui(self) -> None:
        self.lbl_algorithm.setText(self._selected_algorithm())
        self.lbl_steps.setText(str(self.training_steps))
        self.lbl_episode.setText(str(self.episode_index))
        self.lbl_episode_reward.setText(f"{self.episode_reward:.2f}")
        state = self.env.state()
        self.lbl_state.setText(f"[{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}, {state[3]:+.3f}]")
        self.lbl_distance.setText(f"{self.env.distance_to_target():.1f} px")

        avg_distance = self._avg_distance_last_5000()
        if avg_distance is None:
            self.lbl_avg_distance_5000.setText("n/a")
        else:
            self.lbl_avg_distance_5000.setText(f"{avg_distance:.2f} px")

        if isinstance(self.agent, DQNAgent):
            self.lbl_epsilon.setText(f"{self.agent.epsilon:.3f}")
            self.lbl_replay.setText(str(len(self.agent.replay)))
        else:
            self.lbl_epsilon.setText("-")
            self.lbl_replay.setText("-")

        if isinstance(self.agent, PPOAgent):
            if self.agent.last_entropy is None:
                self.lbl_ppo_entropy.setText("-")
            else:
                self.lbl_ppo_entropy.setText(f"{self.agent.last_entropy:.4f}")

            if self.agent.last_ratio_mean is None:
                self.lbl_ppo_ratio.setText("-")
            else:
                self.lbl_ppo_ratio.setText(f"{self.agent.last_ratio_mean:.4f}")

            if self.agent.last_clip_fraction is None:
                self.lbl_ppo_clip_frac.setText("-")
            else:
                self.lbl_ppo_clip_frac.setText(f"{self.agent.last_clip_fraction:.3f}")
        else:
            self.lbl_ppo_entropy.setText("-")
            self.lbl_ppo_ratio.setText("-")
            self.lbl_ppo_clip_frac.setText("-")

        self.lbl_loss.setText("-" if self.last_loss is None else f"{self.last_loss:.4f}")
        self.lbl_last_reward.setText(f"{self.last_reward:.3f}")
        self.lbl_device.setText(str(self.agent.device))


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
