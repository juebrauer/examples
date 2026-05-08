"""
DQN Demo mit PySide6-Visualisierung

Agent: schwarzer Kreis
Ziel: roter Kreis, zufällig pro Episode
Observation: kleines RGB-Bild

Wichtig:
- Das Bild, das groß angezeigt wird, ist EXAKT das gleiche Bild, das das CNN bekommt.
- Es wird immer in Originalgröße angezeigt, z.B. 64 x 64 Pixel.
- Keine Skalierung, keine Interpolation, keine separate "schöne" Grid-Darstellung.

Installation:
    pip install PySide6 matplotlib numpy torch

Start:
    python dqn_pyside6_visual_demo.py
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# -----------------------------
# Environment
# -----------------------------

Action = int
Position = Tuple[int, int]


class ImageGoalEnv:
    """Kleine Grid-World, die ein RGB-Bild als Observation liefert."""

    MAX_STEPS_FACTOR = 10
    GOAL_RADIUS_FACTOR = 0.72
    AGENT_RADIUS_FACTOR = 0.68
    MIN_DISC_RADIUS_PX = 3.5

    ACTIONS = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }

    def __init__(self, grid_size: int = 64, obs_size: int = 64):
        self.grid_size = grid_size
        self.obs_size = obs_size
        self.agent: Position = (0, 0)
        self.goal: Position = (0, 0)
        self.min_steps = 1
        self.max_steps = 50
        self.steps = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent = (
            random.randrange(self.grid_size),
            random.randrange(self.grid_size),
        )

        self.goal = self.agent
        while self.is_touching_goal(self.agent, self.goal):
            self.goal = (
                random.randrange(self.grid_size),
                random.randrange(self.grid_size),
            )

        self.steps = 0
        self.min_steps = self.contact_distance(self.agent, self.goal)
        self.max_steps = self.MAX_STEPS_FACTOR * self.min_steps
        return self.render_observation()

    def manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def goal_radius_px(self) -> float:
        cell = self.obs_size / self.grid_size
        return max(cell * self.GOAL_RADIUS_FACTOR, self.MIN_DISC_RADIUS_PX)

    def agent_radius_px(self) -> float:
        cell = self.obs_size / self.grid_size
        return max(cell * self.AGENT_RADIUS_FACTOR, self.MIN_DISC_RADIUS_PX)

    def touch_radius_px(self) -> float:
        return self.goal_radius_px() + self.agent_radius_px()

    def contact_distance(self, a: Position, b: Position) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        touch_radius_sq = self.touch_radius_px() ** 2

        if dx * dx + dy * dy <= touch_radius_sq:
            return 0

        best_inside_l1 = 0
        max_x = min(dx, int(self.touch_radius_px()))

        for inside_x in range(max_x + 1):
            remaining_sq = touch_radius_sq - inside_x * inside_x
            if remaining_sq < 0:
                continue

            inside_y = min(dy, int(np.floor(np.sqrt(remaining_sq))))
            best_inside_l1 = max(best_inside_l1, inside_x + inside_y)

        return dx + dy - best_inside_l1

    def is_touching_goal(self, a: Position, b: Position) -> bool:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy <= self.touch_radius_px() ** 2

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        dx, dy = self.ACTIONS[action]
        old_dist = self.contact_distance(self.agent, self.goal)

        x = int(np.clip(self.agent[0] + dx, 0, self.grid_size - 1))
        y = int(np.clip(self.agent[1] + dy, 0, self.grid_size - 1))
        self.agent = (x, y)
        self.steps += 1

        new_dist = self.contact_distance(self.agent, self.goal)
        reached = self.is_touching_goal(self.agent, self.goal)
        truncated = self.steps >= self.max_steps

        # Reward-Shaping:
        # - jeder Schritt kostet leicht
        # - Annäherung an das Ziel wird deutlich belohnt
        # - Stillstand wird leicht bestraft
        # - Entfernen vom Ziel wird stärker bestraft
        # - Ziel gibt großen Bonus
        # - Truncation gibt Strafe
        distance_delta = float(old_dist - new_dist)

        reward = -0.02

        if distance_delta > 0.0:
            reward += 0.60 * distance_delta
        elif distance_delta < 0.0:
            reward += 0.90 * distance_delta
        else:
            reward -= 0.03

        if reached:
            reward += 5.0

        if truncated:
            reward -= 2.0

        done = reached or truncated

        info = {
            "reached": reached,
            "truncated": truncated,
            "min_steps": self.min_steps,
            "steps": self.steps,
        }

        return self.render_observation(), reward, done, truncated, info

    def render_observation(self) -> np.ndarray:
        """
        Das ist die einzige visuelle Repräsentation.

        Genau dieses Bild:
        - bekommt das CNN als Input
        - wird groß links angezeigt
        - wird rechts im Preview angezeigt

        Format:
        - shape: obs_size x obs_size x 3
        - dtype: float32
        - Wertebereich: [0, 1]
        - Layout: NHWC
        """
        img = np.ones((self.obs_size, self.obs_size, 3), dtype=np.float32)
        cell = self.obs_size / self.grid_size

        def draw_disc(
            pos: Position,
            color: Tuple[float, float, float],
            radius_px: float,
        ):
            cx = (pos[0] + 0.5) * cell
            cy = (pos[1] + 0.5) * cell
            rr = radius_px

            # Rastere gegen Pixel-Zentren statt Pixel-Ecken.
            # Sonst verschwinden die Punkte bei 1-Pixel-Zellen komplett.
            yy, xx = np.mgrid[0:self.obs_size, 0:self.obs_size].astype(np.float32)
            yy += 0.5
            xx += 0.5
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr**2
            img[mask] = color

        # Reihenfolge ist wichtig:
        # Erst Ziel, dann Agent. Falls sie sich überlagern würden,
        # wäre der Agent sichtbar. In normalen Episoden sind sie nicht gleich.
        draw_disc(self.goal, (1.0, 0.0, 0.0), self.goal_radius_px())
        draw_disc(self.agent, (0.0, 0.0, 0.0), self.agent_radius_px())

        return img


# -----------------------------
# DQN
# -----------------------------

class DQN(nn.Module):
    """
    CNN with full receptive field over the 64x64 input image.

    Architecture rationale:
    - 4 conv layers with kernel=5, stride=2 give a receptive field of ~61 px,
      covering the entire 64x64 image.  Every output cell therefore "sees" both
      the agent and the goal, no matter where they are.
    - Global Average Pooling (GAP) collapses the spatial dimensions to a single
      128-d vector, forcing the network to produce a global summary of the scene.
    - The head then maps that vector to Q-values for each action.

    This design is translation-equivariant: the network learns
    "goal is to the right of me → go right" and generalises that rule to all
    positions, without needing absolute coordinate hints (no CoordConv).
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()

        h, w, c = input_shape

        # 3x3 kernels are more precise for detecting small 3-4px dots.
        # Receptive field with kernel=3, stride=2, padding=1:
        #   conv1: RF=3   output 32x32
        #   conv2: RF=7   output 16x16
        #   conv3: RF=15  output  8x8
        #   conv4: RF=31  output  4x4
        #   conv5: RF=63  output  2x2  -> covers full 64px image
        #
        # Flatten (not GAP!) preserves the spatial layout so the head can
        # learn WHERE agent and goal are relative to each other.
        self.conv = nn.Sequential(
            nn.Conv2d(c,   8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, c, h, w)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))

    def receptive_field_summary(self) -> List[str]:
        rf = 1
        jump = 1
        lines: List[str] = []

        for idx, layer in enumerate(self.conv):
            if not isinstance(layer, nn.Conv2d):
                continue

            kernel = layer.kernel_size[0]
            stride = layer.stride[0]
            rf = rf + (kernel - 1) * jump
            jump *= stride
            lines.append(
                f"conv{len(lines) + 1}: kernel={kernel}x{kernel}, stride={stride}, receptive field={rf}px"
            )

        return lines


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        num_actions: int = 4,
        gamma: float = 0.97,
        lr: float = 3e-4,
        batch_size: int = 32,
        replay_size: int = 20_000,
        target_update_steps: int = 2000,
        learning_starts: int = 2000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay_steps: int = 100_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.learning_starts = learning_starts
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)

        self.obs_shape = obs_shape

        self.policy = DQN(obs_shape, num_actions).to(self.device)
        self.target = DQN(obs_shape, num_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_size)

        self.env_steps = 0
        self.training_steps = 0
        self.epsilon = self.epsilon_start

    def register_env_step(self):
        self.env_steps += 1
        progress = min(1.0, self.env_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)

    def to_tensor(self, states: np.ndarray) -> torch.Tensor:
        # Input from environment: NHWC -> NCHW for Conv2D
        x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return x.permute(0, 3, 1, 2).contiguous()

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            s = self.to_tensor(state[None, ...])
            q = self.policy(s)
            return int(torch.argmax(q, dim=1).item())

    def optimize(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.learning_starts):
            return None

        batch = self.replay.sample(self.batch_size)

        states = np.stack([t.state for t in batch])
        actions = torch.as_tensor(
            [t.action for t in batch],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = np.stack([t.next_state for t in batch])
        dones = torch.as_tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        s = self.to_tensor(states)
        ns = self.to_tensor(next_states)

        q_values = self.policy(s).gather(1, actions)

        with torch.no_grad():
            # Double-DQN:
            # Aktion aus policy-Netz, Bewertung aus target-Netz.
            next_actions = self.policy(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())


# -----------------------------
# UI Helpers
# -----------------------------

def observation_to_qimage(obs: np.ndarray) -> QImage:
    """
    Konvertiert die float32-NHWC-Observation in ein QImage.

    Keine Größenänderung.
    Keine Interpolation.
    Keine alternative Renderlogik.
    """
    arr = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
    h, w, c = arr.shape

    if c != 3:
        raise ValueError(f"Erwartet RGB-Bild mit 3 Kanälen, bekam shape={arr.shape}")

    return QImage(arr.data, w, h, c * w, QImage.Format_RGB888).copy()


class CnnInputView(QWidget):
    """
    Große Anzeige links.

    Wichtig:
    Diese Ansicht zeigt exakt self.state.
    self.state ist exakt der CNN-Input.
    Das Bild wird immer in Originalgröße gezeichnet.
    """

    def __init__(self):
        super().__init__()
        self.obs: np.ndarray | None = None
        self.display_scale = 2
        # Kleinere Mindestgröße, damit das Gesamtfenster auf kleineren Displays passt.
        self.setMinimumSize(300, 240)
        self.setSizePolicy(
            self.sizePolicy().Policy.Expanding,
            self.sizePolicy().Policy.Expanding,
        )

    def set_observation(self, obs: np.ndarray):
        self.obs = obs
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(245, 245, 245))

        if self.obs is None:
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(self.rect(), Qt.AlignCenter, "No observation yet")
            return

        qimg = observation_to_qimage(self.obs)
        scaled = qimg.scaled(
            qimg.width() * self.display_scale,
            qimg.height() * self.display_scale,
            Qt.IgnoreAspectRatio,
            Qt.FastTransformation,
        )
        w = scaled.width()
        h = scaled.height()

        # Nur die linke User-Anzeige wird als Pixelverdopplung vergroessert.
        # Der eigentliche CNN-Input self.obs bleibt unveraendert bei 64 x 64.
        x = int((self.width() - w) / 2)
        y = int((self.height() - h) / 2)
        painter.drawImage(x, y, scaled)

        painter.setPen(QColor(70, 70, 70))
        painter.drawText(
            12,
            24,
            f"Large view = 2x user view, CNN input remains {qimg.width()} x {qimg.height()} px",
        )


class ObservationPreview(QLabel):
    """
    Kleine Anzeige rechts.

    Auch hier:
    exakt das CNN-Input-Bild, exakt Originalgröße.
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: white; border: 1px solid #ccc;")
        self.setText("No observation")

    def set_observation(self, obs: np.ndarray):
        qimg = observation_to_qimage(obs)
        pixmap = QPixmap.fromImage(qimg)

        # Exakt Originalgröße.
        # Kein scaled().
        self.setPixmap(pixmap)

        # +2 wegen Border.
        self.setFixedSize(qimg.width() + 2, qimg.height() + 2)


class LearningPlot(FigureCanvas):
    """Balkendiagramm: tatsächliche Schritte relativ zur optimalen Mindestanzahl."""

    def __init__(self):
        self.fig = Figure(figsize=(5.0, 4.2), tight_layout=True)
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self.episodes: List[int] = []
        self.steps: List[int] = []
        self.minimums: List[int] = []
        self.ratios: List[float] = []

        self.setMinimumHeight(160)

        self.redraw()

    def add_episode(self, episode: int, steps: int, minimum: int):
        minimum = max(1, minimum)

        self.episodes.append(episode)
        self.steps.append(steps)
        self.minimums.append(minimum)
        self.ratios.append(steps / minimum)

        # Nur die letzten 50 Episoden behalten.
        if len(self.steps) > 50:
            self.episodes = self.episodes[-50:]
            self.steps = self.steps[-50:]
            self.minimums = self.minimums[-50:]
            self.ratios = self.ratios[-50:]

        self.redraw()

    def clear(self):
        self.episodes.clear()
        self.steps.clear()
        self.minimums.clear()
        self.ratios.clear()
        self.redraw()

    def redraw(self):
        self.ax.clear()

        if self.steps:
            x = np.array(self.episodes, dtype=np.int32)
            ratios = np.array(self.ratios, dtype=np.float32)

            self.ax.bar(
                x,
                ratios,
                width=0.85,
                color="#1f77b4",
                edgecolor="#0f3a5a",
                linewidth=0.6,
                label="Steps / Minimum",
            )
            self.ax.axhline(
                1.0,
                linestyle="--",
                linewidth=1.4,
                color="#c0392b",
            )
            visible = min(50, len(ratios))
            self.ax.set_xlim(
                x[-visible] - 0.5,
                x[-1] + 0.5,
            )

            visible_max = float(np.max(ratios[-visible:]))
            self.ax.set_ylim(
                bottom=0.0,
                top=max(2.0, min(50.0, visible_max * 1.15)),
            )

        self.ax.set_title("Episode: Steps relative to minimum", fontsize=11)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Factor")
        self.ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)

        self.draw_idle()


# -----------------------------
# Main Window
# -----------------------------

class MainWindow(QMainWindow):
    WORLD_SIZE_PX = 64

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DQN Demo: CNN image input, random goal, PySide6")
        self.resize(1100, 600)

        self.env = self.create_env()
        self.agent = DQNAgent(
            obs_shape=(self.env.obs_size, self.env.obs_size, 3),
        )

        self.state = self.env.render_observation()

        self.episode = 1
        self.last_loss: float | None = None
        self.running = False
        self.visualization_enabled = True
        self.train_steps_per_tick = 16

        self._build_ui()

        self.step_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.step_shortcut.setContext(Qt.WindowShortcut)
        self.step_shortcut.activated.connect(self.step_once)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(1)

        self.refresh_visuals()
        self.update_labels()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        # Links: Weltanzeige oben, Lernplot darunter.
        left = QFrame()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.input_view = CnnInputView()
        left_layout.addWidget(self.input_view, stretch=3)

        self.plot = LearningPlot()
        left_layout.addWidget(self.plot, stretch=2)

        layout.addWidget(left, stretch=3)

        # Rechts: Controls und Status.
        side = QFrame()
        side.setFrameShape(QFrame.StyledPanel)
        side_layout = QVBoxLayout(side)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setWidget(side)
        layout.addWidget(side_scroll, stretch=2)

        stats_box = QGroupBox("Status")
        stats_layout = QGridLayout(stats_box)

        self.lbl_episode = QLabel()
        self.lbl_step = QLabel()
        self.lbl_min = QLabel()
        self.lbl_ratio = QLabel()
        self.lbl_epsilon = QLabel()
        self.lbl_replay = QLabel()
        self.lbl_loss = QLabel()
        self.lbl_device = QLabel(str(self.agent.device))
        self.lbl_obs_shape = QLabel()

        rows = [
            ("Episode", self.lbl_episode),
            ("Steps", self.lbl_step),
            ("Minimum", self.lbl_min),
            ("Factor", self.lbl_ratio),
            ("Epsilon", self.lbl_epsilon),
            ("Replay", self.lbl_replay),
            ("Loss", self.lbl_loss),
            ("Device", self.lbl_device),
            ("CNN input", self.lbl_obs_shape),
        ]

        for r, (name, label) in enumerate(rows):
            stats_layout.addWidget(QLabel(name + ":"), r, 0)
            stats_layout.addWidget(label, r, 1)

        side_layout.addWidget(stats_box)

        controls_box = QGroupBox("Training")
        controls = QVBoxLayout(controls_box)

        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("Start")
        self.btn_start.setToolTip("Start or pause training")
        self.btn_start.clicked.connect(self.toggle_running)
        self.btn_start.setMinimumHeight(28)
        btn_row.addWidget(self.btn_start)

        self.btn_single = QPushButton("Step")
        self.btn_single.setToolTip("Run exactly one training step")
        self.btn_single.clicked.connect(self.step_once)
        self.btn_single.setMinimumHeight(28)
        btn_row.addWidget(self.btn_single)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset network and statistics")
        self.btn_reset.clicked.connect(self.reset_training)
        self.btn_reset.setMinimumHeight(28)
        btn_row.addWidget(self.btn_reset)

        self.btn_reward_info = QPushButton("Reward")
        self.btn_reward_info.setToolTip("Show reward shaping")
        self.btn_reward_info.clicked.connect(self.show_reward_shaping_dialog)
        self.btn_reward_info.setMinimumHeight(28)
        btn_row.addWidget(self.btn_reward_info)

        self.btn_rf_info = QPushButton("RF")
        self.btn_rf_info.setToolTip("Show receptive field per convolution layer")
        self.btn_rf_info.clicked.connect(self.show_receptive_field_dialog)
        self.btn_rf_info.setMinimumHeight(28)
        btn_row.addWidget(self.btn_rf_info)

        controls.addLayout(btn_row)

        self.chk_visual = QCheckBox("Visualization enabled")
        self.chk_visual.setChecked(True)
        self.chk_visual.toggled.connect(self.set_visualization_enabled)
        controls.addWidget(self.chk_visual)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Training steps per tick"))

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 5000)
        self.spin_steps.setValue(16)
        self.spin_steps.valueChanged.connect(self.set_train_steps_per_tick)

        speed_row.addWidget(self.spin_steps)
        controls.addLayout(speed_row)

        eps_row = QHBoxLayout()
        eps_row.addWidget(QLabel("Epsilon min"))

        self.spin_eps_min = QDoubleSpinBox()
        self.spin_eps_min.setRange(0.0, 0.5)
        self.spin_eps_min.setDecimals(3)
        self.spin_eps_min.setSingleStep(0.01)
        self.spin_eps_min.setValue(self.agent.epsilon_min)
        self.spin_eps_min.valueChanged.connect(self.set_epsilon_min)

        eps_row.addWidget(self.spin_eps_min)
        controls.addLayout(eps_row)

        side_layout.addWidget(controls_box)

        obs_box = QGroupBox()
        obs_layout = QVBoxLayout(obs_box)

        self.obs_preview = ObservationPreview()
        obs_layout.addWidget(self.obs_preview, alignment=Qt.AlignCenter)

        side_layout.addWidget(obs_box)

        menu = self.menuBar().addMenu("File")
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def show_reward_shaping_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Reward-Shaping")
        dialog.setModal(True)
        dialog.resize(420, 260)

        layout = QVBoxLayout(dialog)

        text = QLabel(
            "Current reward shaping:\n\n"
            "-0.02 per step\n"
            "+0.60 per distance reduction\n"
            "-0.03 extra for no movement\n"
            "-0.90 per distance increase\n"
            "+5.0 on goal\n"
            "-2.0 on truncation\n\n"
            "Truncate: steps >= 10 x minimum steps."
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def show_receptive_field_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Receptive Field")
        dialog.setModal(True)
        dialog.resize(500, 260)

        layout = QVBoxLayout(dialog)

        lines = self.agent.policy.receptive_field_summary()
        text = QLabel(
            "Receptive field after each convolution layer:\n\n"
            + "\n".join(lines)
            + "\n\nFinal receptive field covers almost the full 64x64 image."
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def toggle_running(self):
        self.running = not self.running
        self.btn_start.setText("Pause" if self.running else "Start")

    def set_visualization_enabled(self, checked: bool):
        self.visualization_enabled = checked

        # Visualisierung aus = linke große Anzeige nicht aktualisieren / ausblenden.
        # Das Training läuft weiter.
        self.input_view.setVisible(checked)

        # Das rechte Preview wird bei manuellen Updates und Labels weiter korrekt gesetzt.
        if checked:
            self.refresh_visuals()

    def set_train_steps_per_tick(self, value: int):
        self.train_steps_per_tick = value

    def set_epsilon_min(self, value: float):
        self.agent.epsilon_min = float(value)

    def create_env(self) -> ImageGoalEnv:
        return ImageGoalEnv(
            grid_size=self.WORLD_SIZE_PX,
            obs_size=self.WORLD_SIZE_PX,
        )

    def step_once(self):
        self.training_step(force_visual_update=True)

    def reset_training(self):
        self.running = False
        self.btn_start.setText("Start")

        self.env = self.create_env()
        self.agent = DQNAgent(
            obs_shape=(self.env.obs_size, self.env.obs_size, 3),
        )

        self.state = self.env.render_observation()
        self.episode = 1
        self.last_loss = None

        self.plot.clear()

        self.refresh_visuals()
        self.update_labels()

    def on_timer(self):
        if not self.running:
            return

        episode_finished = False
        for _ in range(self.train_steps_per_tick):
            episode_finished = self.training_step(force_visual_update=False) or episode_finished

        if self.visualization_enabled:
            self.update_labels()
            self.refresh_visuals()
        elif episode_finished:
            self.update_labels()

    def training_step(self, force_visual_update: bool = False) -> bool:
        action = self.agent.select_action(self.state)

        next_state, reward, done, truncated, info = self.env.step(action)

        self.agent.replay.push(
            self.state,
            action,
            reward,
            next_state,
            done,
        )
        self.agent.register_env_step()

        self.state = next_state

        loss = self.agent.optimize()
        if loss is not None:
            self.last_loss = loss

        episode_finished = False
        if done:
            self.plot.add_episode(self.episode, info["steps"], info["min_steps"])
            self.episode += 1
            self.state = self.env.reset()
            episode_finished = True

        if force_visual_update:
            self.refresh_visuals()
            self.update_labels()

        return episode_finished

    def refresh_visuals(self):
        # Beide Anzeigen bekommen exakt dasselbe Objekt self.state.
        self.input_view.set_observation(self.state)
        self.obs_preview.set_observation(self.state)

    def update_labels(self):
        steps = self.env.steps
        minimum = max(1, self.env.min_steps)
        ratio = steps / minimum

        self.lbl_episode.setText(str(self.episode))
        self.lbl_step.setText(f"{steps} / max {self.env.max_steps}")
        self.lbl_min.setText(str(minimum))
        self.lbl_ratio.setText(f"{ratio:.2f}x")
        self.lbl_epsilon.setText(f"{self.agent.epsilon:.3f}")
        self.lbl_replay.setText(str(len(self.agent.replay)))
        self.lbl_loss.setText(
            "-" if self.last_loss is None else f"{self.last_loss:.4f}"
        )
        self.lbl_device.setText(str(self.agent.device))

        h, w, c = self.state.shape
        self.lbl_obs_shape.setText(f"{h} x {w} x {c}")


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()