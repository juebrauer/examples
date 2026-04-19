import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This demo requires PyTorch. Install it in your environment, e.g. 'pip install torch'."
    ) from exc


@dataclass
class RewardConfig:
    goal_reached: float = 100.0
    distance_reduction: float = 1.0
    obstacle_collision: float = -5.0


@dataclass
class Obstacle:
    shape: str
    x: float
    y: float
    w: float
    h: float


class Simple2DWorld:
    ACTION_TURN_LEFT = 0
    ACTION_TURN_RIGHT = 1
    ACTION_FORWARD = 2
    ACTION_BACKWARD = 3

    def __init__(self, world_size: int = 160, max_steps: int = 260) -> None:
        self.world_size = world_size
        self.max_steps = max_steps
        self.robot_radius = 3.5
        self.goal_radius = 5.0
        self.move_step = 2.8
        self.turn_step = math.radians(18.0)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.obstacles: List[Obstacle] = []
        self.step_count = 0

        self.reset()

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.obstacles = self._generate_random_obstacles()
        self.robot_x, self.robot_y = self._sample_free_point(min_border_dist=12.0)
        self.goal_x, self.goal_y = self._sample_free_point(min_border_dist=12.0)
        while self._distance(self.robot_x, self.robot_y, self.goal_x, self.goal_y) < 40.0:
            self.goal_x, self.goal_y = self._sample_free_point(min_border_dist=12.0)
        self.robot_theta = random.uniform(-math.pi, math.pi)
        return self.render_image()

    def step(self, action: int, rewards: RewardConfig) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        old_dist = self._distance(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        reward = 0.0
        done = False
        collision = False

        if action == self.ACTION_TURN_LEFT:
            self.robot_theta -= self.turn_step
        elif action == self.ACTION_TURN_RIGHT:
            self.robot_theta += self.turn_step
        elif action in (self.ACTION_FORWARD, self.ACTION_BACKWARD):
            direction = 1.0 if action == self.ACTION_FORWARD else -1.0
            candidate_x = self.robot_x + direction * self.move_step * math.cos(self.robot_theta)
            candidate_y = self.robot_y + direction * self.move_step * math.sin(self.robot_theta)

            if self._is_collision(candidate_x, candidate_y):
                collision = True
                reward += rewards.obstacle_collision
            else:
                self.robot_x = candidate_x
                self.robot_y = candidate_y

        self.robot_theta = ((self.robot_theta + math.pi) % (2.0 * math.pi)) - math.pi

        new_dist = self._distance(self.robot_x, self.robot_y, self.goal_x, self.goal_y)
        if new_dist < old_dist:
            reward += rewards.distance_reduction

        if new_dist <= self.goal_radius + self.robot_radius:
            reward += rewards.goal_reached
            done = True

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "collision": collision,
            "distance": new_dist,
            "steps": self.step_count,
        }

        return self.render_image(), reward, done, info

    def _generate_random_obstacles(self) -> List[Obstacle]:
        obs: List[Obstacle] = []
        num_obstacles = random.randint(5, 10)
        for _ in range(num_obstacles):
            shape = random.choice(["rect", "ellipse"])
            w = random.uniform(9.0, 20.0)
            h = random.uniform(8.0, 18.0)
            x = random.uniform(8.0, self.world_size - w - 8.0)
            y = random.uniform(8.0, self.world_size - h - 8.0)
            obs.append(Obstacle(shape=shape, x=x, y=y, w=w, h=h))
        return obs

    def _sample_free_point(self, min_border_dist: float) -> Tuple[float, float]:
        for _ in range(300):
            x = random.uniform(min_border_dist, self.world_size - min_border_dist)
            y = random.uniform(min_border_dist, self.world_size - min_border_dist)
            if not self._is_collision(x, y):
                return x, y
        return self.world_size * 0.5, self.world_size * 0.5

    def _is_collision(self, x: float, y: float) -> bool:
        r = self.robot_radius

        if x - r < 0.0 or y - r < 0.0 or x + r >= self.world_size or y + r >= self.world_size:
            return True

        for ob in self.obstacles:
            if ob.shape == "rect":
                if self._circle_rect_collision(x, y, r, ob):
                    return True
            else:
                if self._circle_ellipse_collision(x, y, r, ob):
                    return True
        return False

    @staticmethod
    def _circle_rect_collision(cx: float, cy: float, r: float, ob: Obstacle) -> bool:
        nearest_x = max(ob.x, min(cx, ob.x + ob.w))
        nearest_y = max(ob.y, min(cy, ob.y + ob.h))
        dx = cx - nearest_x
        dy = cy - nearest_y
        return (dx * dx + dy * dy) <= (r * r)

    @staticmethod
    def _circle_ellipse_collision(cx: float, cy: float, r: float, ob: Obstacle) -> bool:
        ex = ob.x + ob.w * 0.5
        ey = ob.y + ob.h * 0.5
        a = ob.w * 0.5 + r
        b = ob.h * 0.5 + r
        if a <= 0.0 or b <= 0.0:
            return False
        nx = (cx - ex) / a
        ny = (cy - ey) / b
        return (nx * nx + ny * ny) <= 1.0

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def render_image(self) -> np.ndarray:
        img = np.ones((self.world_size, self.world_size, 3), dtype=np.float32)

        for ob in self.obstacles:
            x1 = int(max(0, ob.x))
            y1 = int(max(0, ob.y))
            x2 = int(min(self.world_size - 1, ob.x + ob.w))
            y2 = int(min(self.world_size - 1, ob.y + ob.h))
            if ob.shape == "rect":
                img[y1:y2, x1:x2, :] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            else:
                if x2 <= x1 or y2 <= y1:
                    continue
                yy, xx = np.mgrid[y1:y2, x1:x2]
                cx = ob.x + ob.w * 0.5
                cy = ob.y + ob.h * 0.5
                rx = max(ob.w * 0.5, 1e-3)
                ry = max(ob.h * 0.5, 1e-3)
                mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
                img[y1:y2, x1:x2, :][mask] = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        self._draw_filled_circle_to_array(
            img,
            int(self.goal_x),
            int(self.goal_y),
            int(self.goal_radius),
            color=np.array([0.1, 0.75, 0.2], dtype=np.float32),
        )

        self._draw_filled_circle_to_array(
            img,
            int(self.robot_x),
            int(self.robot_y),
            int(self.robot_radius),
            color=np.array([0.85, 0.1, 0.1], dtype=np.float32),
        )

        head_x = int(self.robot_x + self.robot_radius * 2.2 * math.cos(self.robot_theta))
        head_y = int(self.robot_y + self.robot_radius * 2.2 * math.sin(self.robot_theta))
        self._draw_line_to_array(
            img,
            int(self.robot_x),
            int(self.robot_y),
            head_x,
            head_y,
            color=np.array([0.95, 0.95, 0.95], dtype=np.float32),
        )

        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _draw_filled_circle_to_array(
        arr: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray
    ) -> None:
        h, w, _ = arr.shape
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)
        if x_max <= x_min or y_max <= y_min:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        arr[y_min:y_max, x_min:x_max, :][mask] = color

    @staticmethod
    def _draw_line_to_array(
        arr: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: np.ndarray,
    ) -> None:
        h, w, _ = arr.shape
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < w and 0 <= y < h:
                arr[y, x, :] = color
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNCNN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNTrainer:
    def __init__(self, image_size: int = 128, device: str | None = None) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.target_sync_every = 300
        self.learn_starts = 600

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 24000

        self.num_actions = 4
        self.step_idx = 0

        self.policy_net = DQNCNN(in_channels=3, num_actions=self.num_actions).to(self.device)
        self.target_net = DQNCNN(in_channels=3, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(capacity=60_000)

        self.image_size = image_size

    def epsilon(self) -> float:
        t = min(1.0, self.step_idx / float(self.epsilon_decay_steps))
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    @staticmethod
    def to_chw(state_hwc: np.ndarray) -> np.ndarray:
        return np.transpose(state_hwc, (2, 0, 1)).astype(np.float32)

    def select_action(self, state_chw: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.num_actions)

        with torch.no_grad():
            x = torch.from_numpy(state_chw[None, ...]).to(self.device)
            q_values = self.policy_net(x)
            return int(torch.argmax(q_values, dim=1).item())

    def add_transition(
        self,
        state_chw: np.ndarray,
        action: int,
        reward: float,
        next_state_chw: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(state_chw, action, reward, next_state_chw, done)

    def train_step(self) -> float | None:
        self.step_idx += 1
        if len(self.replay) < max(self.learn_starts, self.batch_size):
            return None

        states_np, actions_np, rewards_np, next_states_np, dones_np = self.replay.sample(self.batch_size)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        q_pred = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.step_idx % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())


class WorldView(QWidget):
    def __init__(self, env: Simple2DWorld) -> None:
        super().__init__()
        self.env = env
        self.draw_trajectory = False
        self.trajectory_points: List[Tuple[float, float]] = []
        self.setMinimumSize(600, 600)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        s = min(self.width(), self.height())
        margin = 14
        draw_size = s - margin * 2
        if draw_size <= 10:
            return
        x0 = (self.width() - draw_size) // 2
        y0 = (self.height() - draw_size) // 2

        p.fillRect(self.rect(), QColor("#ece7df"))
        p.fillRect(x0, y0, draw_size, draw_size, QColor("#dbe5de"))
        p.setPen(QPen(QColor("#333333"), 2))
        p.drawRect(x0, y0, draw_size, draw_size)

        scale = draw_size / self.env.world_size

        for ob in self.env.obstacles:
            ox = x0 + int(ob.x * scale)
            oy = y0 + int(ob.y * scale)
            ow = max(2, int(ob.w * scale))
            oh = max(2, int(ob.h * scale))
            p.setBrush(QColor("#444444"))
            p.setPen(Qt.NoPen)
            if ob.shape == "rect":
                p.drawRect(ox, oy, ow, oh)
            else:
                p.drawEllipse(ox, oy, ow, oh)

        gx = x0 + int(self.env.goal_x * scale)
        gy = y0 + int(self.env.goal_y * scale)
        gr = max(4, int(self.env.goal_radius * scale))
        p.setBrush(QColor("#18b14a"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(gx - gr, gy - gr, gr * 2, gr * 2)

        if self.draw_trajectory and len(self.trajectory_points) >= 2:
            p.setPen(QPen(QColor("#2070b8"), 2))
            for i in range(1, len(self.trajectory_points)):
                x_prev, y_prev = self.trajectory_points[i - 1]
                x_curr, y_curr = self.trajectory_points[i]
                px = x0 + int(x_prev * scale)
                py = y0 + int(y_prev * scale)
                qx = x0 + int(x_curr * scale)
                qy = y0 + int(y_curr * scale)
                p.drawLine(px, py, qx, qy)

        rx = x0 + int(self.env.robot_x * scale)
        ry = y0 + int(self.env.robot_y * scale)
        rr = max(4, int(self.env.robot_radius * scale))
        p.setBrush(QColor("#d13232"))
        p.setPen(QPen(QColor("#7d1414"), 2))
        p.drawEllipse(rx - rr, ry - rr, rr * 2, rr * 2)

        hx = rx + int(rr * math.cos(self.env.robot_theta))
        hy = ry + int(rr * math.sin(self.env.robot_theta))
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(rx, ry, hx, hy)


class RewardEditor(QWidget):
    def __init__(self, defaults: RewardConfig) -> None:
        super().__init__()

        self.defaults = defaults

        layout = QFormLayout(self)
        self.goal_check, self.goal_spin = self._make_item(defaults.goal_reached)
        self.dist_check, self.dist_spin = self._make_item(defaults.distance_reduction)
        self.obs_check, self.obs_spin = self._make_item(defaults.obstacle_collision)

        layout.addRow("Goal reward", self._wrap_pair(self.goal_check, self.goal_spin))
        layout.addRow("Distance reward", self._wrap_pair(self.dist_check, self.dist_spin))
        layout.addRow("Collision penalty", self._wrap_pair(self.obs_check, self.obs_spin))

    @staticmethod
    def _wrap_pair(check: QCheckBox, spin: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(check)
        row.addWidget(spin)
        row.addStretch(1)
        return w

    @staticmethod
    def _make_item(value: float) -> Tuple[QCheckBox, QDoubleSpinBox]:
        check = QCheckBox("use")
        check.setChecked(True)

        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-500.0, 500.0)
        spin.setSingleStep(0.5)
        spin.setValue(value)

        check.toggled.connect(spin.setEnabled)
        return check, spin

    def current_rewards(self) -> RewardConfig:
        return RewardConfig(
            goal_reached=self.goal_spin.value() if self.goal_check.isChecked() else self.defaults.goal_reached,
            distance_reduction=(
                self.dist_spin.value() if self.dist_check.isChecked() else self.defaults.distance_reduction
            ),
            obstacle_collision=(
                self.obs_spin.value() if self.obs_check.isChecked() else self.defaults.obstacle_collision
            ),
        )


class DistanceProgressPlot(QWidget):
    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5.0, 2.2), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Max distance fraction per episode", fontsize=10)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Fraction")
        self.ax.set_ylim(0.0, 1.05)
        self.ax.grid(True, alpha=0.25)
        (self.line,) = self.ax.plot([], [], color="#2070b8", linewidth=1.8)

        self._x: List[int] = []
        self._y: List[float] = []

    def append_point(self, episode_idx: int, fraction: float) -> None:
        self._x.append(episode_idx)
        self._y.append(float(np.clip(fraction, 0.0, 1.0)))

        self.line.set_data(self._x, self._y)
        if self._x:
            left = max(1, self._x[0])
            right = max(left + 1, self._x[-1])
            self.ax.set_xlim(left, right)
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DQN + CNN Navigation Demo (PySide6)")

        self.env = Simple2DWorld(world_size=160, max_steps=260)
        self.trainer = DQNTrainer(image_size=160)

        self.current_state_hwc = self.env.render_image()
        self.current_state_chw = self.trainer.to_chw(self.current_state_hwc)

        self.episode_idx = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_loss = None
        self.total_steps = 0
        self.running = False
        self.recent_episode_rewards: Deque[float] = deque(maxlen=20)
        self.trajectory_points: List[Tuple[float, float]] = [(self.env.robot_x, self.env.robot_y)]
        self.episode_start_distance = self._current_start_goal_distance()
        self.episode_min_distance = self.episode_start_distance

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.world_view = WorldView(self.env)
        self.world_view.trajectory_points = self.trajectory_points
        layout.addWidget(self.world_view, stretch=2)

        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setSpacing(8)

        rewards_group = QGroupBox("Reward Shaping")
        rewards_layout = QVBoxLayout(rewards_group)
        self.reward_editor = RewardEditor(defaults=RewardConfig())
        rewards_layout.addWidget(self.reward_editor)

        self.step_limit_spin = QSpinBox()
        self.step_limit_spin.setRange(20, 2000)
        self.step_limit_spin.setSingleStep(10)
        self.step_limit_spin.setValue(self.env.max_steps)
        self.step_limit_spin.valueChanged.connect(self._on_step_limit_changed)

        rewards_layout.addWidget(QLabel("Episode step limit (truncation):"))
        rewards_layout.addWidget(self.step_limit_spin)

        self.chk_draw_trajectory = QCheckBox("Draw robot trajectory")
        self.chk_draw_trajectory.setChecked(False)
        self.chk_draw_trajectory.toggled.connect(self._on_draw_trajectory_toggled)
        rewards_layout.addWidget(self.chk_draw_trajectory)
        right.addWidget(rewards_group)

        status_group = QGroupBox("Training Status")
        status_layout = QGridLayout(status_group)

        self.lbl_episode = QLabel("Episode: 0")
        self.lbl_ep_reward = QLabel("Episode reward: 0.00")
        self.lbl_last_reward = QLabel("Last reward: 0.00")
        self.lbl_avg_reward = QLabel("Avg reward (20): 0.00")
        self.lbl_epsilon = QLabel("Epsilon: 1.000")
        self.lbl_loss = QLabel("Loss: -")
        self.lbl_steps = QLabel("Total steps: 0")
        self.lbl_episode_steps = QLabel(f"Episode steps: 0 / {self.env.max_steps}")

        status_layout.addWidget(self.lbl_episode, 0, 0)
        status_layout.addWidget(self.lbl_ep_reward, 1, 0)
        status_layout.addWidget(self.lbl_last_reward, 2, 0)
        status_layout.addWidget(self.lbl_avg_reward, 3, 0)
        status_layout.addWidget(self.lbl_epsilon, 4, 0)
        status_layout.addWidget(self.lbl_loss, 5, 0)
        status_layout.addWidget(self.lbl_steps, 6, 0)
        status_layout.addWidget(self.lbl_episode_steps, 7, 0)

        right.addWidget(status_group)

        progress_group = QGroupBox("Learning Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_plot = DistanceProgressPlot()
        progress_layout.addWidget(self.progress_plot)
        right.addWidget(progress_group)

        self.btn_start_stop = QPushButton("Start Training")
        self.btn_start_stop.clicked.connect(self.toggle_training)
        right.addWidget(self.btn_start_stop)
        right.addStretch(1)

        layout.addWidget(right_panel, stretch=1)

        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.training_tick)

        self._refresh_labels()

    def toggle_training(self) -> None:
        self.running = not self.running
        if self.running:
            self.btn_start_stop.setText("Stop Training")
            self.timer.start()
        else:
            self.btn_start_stop.setText("Start Training")
            self.timer.stop()

    def training_tick(self) -> None:
        rewards_cfg = self.reward_editor.current_rewards()

        # Run multiple env transitions per timer event to keep learning reasonably fast.
        for _ in range(4):
            action = self.trainer.select_action(self.current_state_chw)
            next_state_hwc, reward, done, _ = self.env.step(action, rewards_cfg)
            next_state_chw = self.trainer.to_chw(next_state_hwc)

            self.trainer.add_transition(self.current_state_chw, action, reward, next_state_chw, done)
            self.last_loss = self.trainer.train_step()

            self.current_state_chw = next_state_chw
            self.current_state_hwc = next_state_hwc
            self.trajectory_points.append((self.env.robot_x, self.env.robot_y))

            current_dist = self.env._distance(
                self.env.robot_x, self.env.robot_y, self.env.goal_x, self.env.goal_y
            )
            self.episode_min_distance = min(self.episode_min_distance, current_dist)

            self.episode_reward += reward
            self.last_reward = reward
            self.total_steps += 1

            if done:
                frac = self._episode_max_distance_fraction()
                self.progress_plot.append_point(self.episode_idx + 1, frac)
                self.recent_episode_rewards.append(self.episode_reward)
                self.episode_idx += 1
                self.episode_reward = 0.0
                self.current_state_hwc = self.env.reset()
                self.current_state_chw = self.trainer.to_chw(self.current_state_hwc)
                self.trajectory_points = [(self.env.robot_x, self.env.robot_y)]
                self.world_view.trajectory_points = self.trajectory_points
                self.episode_start_distance = self._current_start_goal_distance()
                self.episode_min_distance = self.episode_start_distance

        self.world_view.update()
        self._refresh_labels()

    def _on_step_limit_changed(self, value: int) -> None:
        self.env.max_steps = int(value)
        self._refresh_labels()

    def _on_draw_trajectory_toggled(self, checked: bool) -> None:
        self.world_view.draw_trajectory = checked
        self.world_view.update()

    def _current_start_goal_distance(self) -> float:
        return self.env._distance(self.env.robot_x, self.env.robot_y, self.env.goal_x, self.env.goal_y)

    def _episode_max_distance_fraction(self) -> float:
        d0 = max(self.episode_start_distance, 1e-6)
        return (d0 - self.episode_min_distance) / d0

    def _refresh_labels(self) -> None:
        avg_reward = (
            sum(self.recent_episode_rewards) / len(self.recent_episode_rewards)
            if self.recent_episode_rewards
            else 0.0
        )
        self.lbl_episode.setText(f"Episode: {self.episode_idx}")
        self.lbl_ep_reward.setText(f"Episode reward: {self.episode_reward:.2f}")
        self.lbl_last_reward.setText(f"Last reward: {self.last_reward:.2f}")
        self.lbl_avg_reward.setText(f"Avg reward (20): {avg_reward:.2f}")
        self.lbl_epsilon.setText(f"Epsilon: {self.trainer.epsilon():.3f}")
        self.lbl_loss.setText("Loss: -" if self.last_loss is None else f"Loss: {self.last_loss:.4f}")
        self.lbl_steps.setText(f"Total steps: {self.total_steps}")
        self.lbl_episode_steps.setText(f"Episode steps: {self.env.step_count} / {self.env.max_steps}")


def main() -> None:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1200, 760)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
