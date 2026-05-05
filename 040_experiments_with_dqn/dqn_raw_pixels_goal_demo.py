import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
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
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class GridWorldRawPixels:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, size: int = 800) -> None:
        self.size = size
        self.agent_radius = 18
        self.goal_radius = 22
        self.step_size = 12
        self._D_max = float(np.hypot(size, size))  # max possible distance, used for PBRS

        self.start_pos = (120, 120)
        self.goal_pos = (680, 660)

        self.agent_x = float(self.start_pos[0])
        self.agent_y = float(self.start_pos[1])

    def reset(self) -> np.ndarray:
        self.agent_x = float(self.start_pos[0])
        self.agent_y = float(self.start_pos[1])
        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, float, float]:
        """Returns (obs, base_reward, done, old_dist, new_dist).
        Distances are provided for optional external reward shaping (e.g. PBRS);
        they are NOT part of the agent observation."""
        old_dx = self.agent_x - self.goal_pos[0]
        old_dy = self.agent_y - self.goal_pos[1]
        old_dist = float(np.hypot(old_dx, old_dy))

        if action == self.ACTION_UP:
            self.agent_y -= self.step_size
        elif action == self.ACTION_DOWN:
            self.agent_y += self.step_size
        elif action == self.ACTION_LEFT:
            self.agent_x -= self.step_size
        elif action == self.ACTION_RIGHT:
            self.agent_x += self.step_size

        self.agent_x = float(np.clip(self.agent_x, self.agent_radius, self.size - self.agent_radius))
        self.agent_y = float(np.clip(self.agent_y, self.agent_radius, self.size - self.agent_radius))

        reward = -0.01
        done = False

        dx = self.agent_x - self.goal_pos[0]
        dy = self.agent_y - self.goal_pos[1]
        new_dist = float(np.hypot(dx, dy))

        if (dx * dx + dy * dy) <= (self.agent_radius + self.goal_radius) ** 2:
            reward = 1.0
            done = True

        return self.render(), reward, done, old_dist, new_dist

    def render(self) -> np.ndarray:
        img = np.ones((self.size, self.size, 3), dtype=np.float32)

        self._draw_filled_circle(
            img,
            int(self.goal_pos[0]),
            int(self.goal_pos[1]),
            self.goal_radius,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )

        self._draw_filled_circle(
            img,
            int(self.agent_x),
            int(self.agent_y),
            self.agent_radius,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

        return img

    @staticmethod
    def _draw_filled_circle(arr: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray) -> None:
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


class DQNCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),  # 3 RGB channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class RewardPlotCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(4.8, 3.6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def update_plot(self, rewards_last_1000: List[float]) -> None:
        self.ax.clear()
        self.ax.set_title("Rewards der letzten 1000 Schritte")
        self.ax.set_xlabel("Schritt-Index")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, alpha=0.3)
        if rewards_last_1000:
            x = np.arange(len(rewards_last_1000))
            self.ax.plot(x, rewards_last_1000, color="#2B6CB0", linewidth=1.3)
        self.draw_idle()


class DQNRawPixelsDemo(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DQN Demo mit Rohbild-Input")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = GridWorldRawPixels(size=800)
        self.n_actions = 4

        self.gamma = 0.99
        self.batch_size = 64
        self.lr = 1e-3
        self.target_sync_every = 250
        self.learning_starts = 300

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.replay = ReplayBuffer(capacity=40_000)

        self.policy_net = DQNCNN(self.n_actions).to(self.device)
        self.target_net = DQNCNN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.total_steps = 0
        self.rewards_window: Deque[float] = deque(maxlen=1000)

        self.current_observation = self.env.reset()

        self.timer = QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.run_one_training_step)

        self.world_label = QLabel()
        self.world_label.setFixedSize(800, 800)
        self.world_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #555;")

        self.steps_label = QLabel("Lernschritte: 0")
        self.steps_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.status_label = QLabel("Status: gestoppt")
        self.status_label.setStyleSheet("font-size: 14px;")

        self.start_stop_button = QPushButton("Start Lernen")
        self.start_stop_button.clicked.connect(self.toggle_learning)

        self.space_hint_label = QLabel("Space: einen Einzelschritt ausfuehren")

        # Checkbox: häufiges oder seltenes UI-Update
        self.slow_ui_checkbox = QCheckBox("UI nur alle 100 Schritte aktualisieren (schnelleres Lernen)")
        self.slow_ui_checkbox.setChecked(False)

        # Device-Anzeige
        device_name = "GPU (CUDA)" if self.device.type == "cuda" else "CPU"
        self.device_label = QLabel(f"Modell laeuft auf: {device_name}")
        self.device_label.setStyleSheet("font-size: 13px; color: #1a6e2e; font-weight: bold;")

        # PBRS-Checkbox
        self.pbrs_checkbox = QCheckBox("Potential-Based Reward Shaping (PBRS) aktiv")
        self.pbrs_checkbox.setChecked(True)

        # CNN-Input-Vorschau (84x84 RGB)
        self.cnn_input_title = QLabel("CNN-Input (84 x 84 x 3, RGB):")
        self.cnn_input_label = QLabel()
        self.cnn_input_label.setFixedSize(84, 84)
        self.cnn_input_label.setStyleSheet("border: 1px solid #888;")

        self.plot_canvas = RewardPlotCanvas()
        self.plot_canvas.setMinimumHeight(280)

        self._setup_layout()
        self._refresh_world_view()
        self.plot_canvas.update_plot(list(self.rewards_window))

    def _setup_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.world_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.device_label)
        right_layout.addWidget(self.steps_label)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.start_stop_button)
        right_layout.addWidget(self.space_hint_label)
        right_layout.addWidget(self.slow_ui_checkbox)
        right_layout.addWidget(self.pbrs_checkbox)
        right_layout.addWidget(self.cnn_input_title)
        right_layout.addWidget(self.cnn_input_label)
        right_layout.addWidget(self.plot_canvas)
        right_layout.addStretch()

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

    def toggle_learning(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Status: gestoppt")
            self.start_stop_button.setText("Start Lernen")
        else:
            self.timer.start()
            self.status_label.setText("Status: laeuft")
            self.start_stop_button.setText("Stop Lernen")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space:
            self.run_one_training_step()
            event.accept()
            return
        super().keyPressEvent(event)

    def run_one_training_step(self) -> None:
        state_small = self._obs_to_state_tensor(self.current_observation)
        action = self._select_action(state_small)

        next_observation, base_reward, done, old_dist, new_dist = self.env.step(action)
        next_state_small = self._obs_to_state_tensor(next_observation)

        # Potential-Based Reward Shaping: F(s,s') = gamma*Phi(s') - Phi(s)
        # Phi(s) = -dist/D_max  =>  approaching goal gives positive F,
        # oscillating gives net-negative due to gamma < 1 (reward-hack-proof).
        if self.pbrs_checkbox.isChecked():
            D_max = self.env._D_max
            phi_s = -old_dist / D_max
            phi_s_next = 0.0 if done else -new_dist / D_max  # terminal potential = 0
            reward = base_reward + self.gamma * phi_s_next - phi_s
        else:
            reward = base_reward

        self.replay.push(
            Transition(
                state=state_small,
                action=action,
                reward=reward,
                next_state=next_state_small,
                done=done,
            )
        )

        self.total_steps += 1
        self.rewards_window.append(reward)

        if len(self.replay) >= self.learning_starts:
            self._optimize_model()

        if self.total_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if done:
            self.current_observation = self.env.reset()
        else:
            self.current_observation = next_observation

        self.steps_label.setText(f"Lernschritte: {self.total_steps}")

        # UI-Update: immer oder nur alle 100 Schritte (je nach Checkbox)
        slow_mode = self.slow_ui_checkbox.isChecked()
        if not slow_mode or self.total_steps % 100 == 0:
            self._refresh_world_view()
            self.plot_canvas.update_plot(list(self.rewards_window))

    def _obs_to_state_tensor(self, obs: np.ndarray) -> np.ndarray:
        # Keep full RGB information; only spatial downscaling to 84x84.
        # obs shape: (800, 800, 3)  ->  output shape: (3, 84, 84)
        tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,800,800)
        resized = torch.nn.functional.interpolate(
            tensor,
            size=(84, 84),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).numpy()  # (3, 84, 84)

    def _select_action(self, state_small: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_tensor = torch.from_numpy(state_small).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _optimize_model(self) -> None:
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch])).float().to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float().to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _refresh_world_view(self) -> None:
        arr = np.clip(self.current_observation * 255.0, 0.0, 255.0).astype(np.uint8)
        h, w, _ = arr.shape
        bytes_per_line = 3 * w
        image = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image.copy())

        # Draw border directly into the pixmap.
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(70, 70, 70), 2))
        painter.drawRect(1, 1, w - 2, h - 2)
        painter.end()

        self.world_label.setPixmap(pixmap)

        # CNN-Input-Vorschau: identisch zu dem, was das Netz sieht (84x84, RGB)
        state_np = self._obs_to_state_tensor(self.current_observation)  # shape (3, 84, 84)
        rgb_84 = np.clip(state_np.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8).copy()  # (84,84,3)
        cnn_image = QImage(rgb_84.data, 84, 84, 3 * 84, QImage.Format_RGB888)
        self.cnn_input_label.setPixmap(QPixmap.fromImage(cnn_image.copy()))


def main() -> None:
    app = QApplication(sys.argv)
    win = DQNRawPixelsDemo()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
