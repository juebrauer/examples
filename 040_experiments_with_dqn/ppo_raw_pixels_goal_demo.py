import random
import sys
from typing import List

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
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This demo requires PyTorch. Install it in your environment, e.g. 'pip install torch'."
    ) from exc


class GridWorldRawPixels:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, world_size: int = 800, obs_size: int = 84) -> None:
        self.world_size = world_size
        self.obs_size = obs_size
        self.agent_radius = 18
        self.goal_radius = 22
        self.step_size = 12
        self._D_max = float(np.hypot(world_size, world_size))

        self.start_pos = (120, 120)
        self.goal_pos = (680, 660)
        self.agent_x = float(self.start_pos[0])
        self.agent_y = float(self.start_pos[1])

    def reset(self) -> np.ndarray:
        self._sample_new_goal_position()
        self.agent_x = float(self.start_pos[0])
        self.agent_y = float(self.start_pos[1])
        return self.render_observation()

    def _sample_new_goal_position(self) -> None:
        min_xy = self.goal_radius
        max_xy = self.world_size - self.goal_radius
        reach_sq = (self.agent_radius + self.goal_radius) ** 2
        while True:
            gx = random.randint(min_xy, max_xy)
            gy = random.randint(min_xy, max_xy)
            dx = gx - self.start_pos[0]
            dy = gy - self.start_pos[1]
            if dx * dx + dy * dy > reach_sq:
                self.goal_pos = (gx, gy)
                return

    def step(self, action: int) -> tuple[np.ndarray, bool, float, float]:
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

        self.agent_x = float(np.clip(self.agent_x, self.agent_radius, self.world_size - self.agent_radius))
        self.agent_y = float(np.clip(self.agent_y, self.agent_radius, self.world_size - self.agent_radius))

        dx = self.agent_x - self.goal_pos[0]
        dy = self.agent_y - self.goal_pos[1]
        new_dist = float(np.hypot(dx, dy))
        reached_goal = (dx * dx + dy * dy) <= (self.agent_radius + self.goal_radius) ** 2
        return self.render_observation(), reached_goal, old_dist, new_dist

    def render_world(self) -> np.ndarray:
        image = np.ones((self.world_size, self.world_size, 3), dtype=np.float32)
        self._draw_filled_circle(
            image,
            int(self.goal_pos[0]),
            int(self.goal_pos[1]),
            self.goal_radius,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        self._draw_filled_circle(
            image,
            int(self.agent_x),
            int(self.agent_y),
            self.agent_radius,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        return image

    def render_observation(self) -> np.ndarray:
        image = np.ones((self.obs_size, self.obs_size, 3), dtype=np.float32)
        scale = self.obs_size / self.world_size
        goal_radius = max(2, int(round(self.goal_radius * scale)))
        agent_radius = max(2, int(round(self.agent_radius * scale)))
        goal_x = int(np.clip(round(self.goal_pos[0] * scale), 0, self.obs_size - 1))
        goal_y = int(np.clip(round(self.goal_pos[1] * scale), 0, self.obs_size - 1))
        agent_x = int(np.clip(round(self.agent_x * scale), 0, self.obs_size - 1))
        agent_y = int(np.clip(round(self.agent_y * scale), 0, self.obs_size - 1))

        self._draw_filled_circle(
            image,
            goal_x,
            goal_y,
            goal_radius,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        self._draw_filled_circle(
            image,
            agent_x,
            agent_y,
            agent_radius,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        return image

    def compute_min_steps(self) -> int:
        from collections import deque as _deque

        reach_sq = (self.agent_radius + self.goal_radius) ** 2
        sx, sy = int(self.start_pos[0]), int(self.start_pos[1])
        dx0 = sx - self.goal_pos[0]
        dy0 = sy - self.goal_pos[1]
        if dx0 * dx0 + dy0 * dy0 <= reach_sq:
            return 0
        queue = _deque([(sx, sy, 0)])
        visited = {(sx, sy)}
        deltas = [(0, -self.step_size), (0, self.step_size), (-self.step_size, 0), (self.step_size, 0)]
        while queue:
            x, y, steps = queue.popleft()
            for ddx, ddy in deltas:
                nx = int(np.clip(x + ddx, self.agent_radius, self.world_size - self.agent_radius))
                ny = int(np.clip(y + ddy, self.agent_radius, self.world_size - self.agent_radius))
                if (nx, ny) in visited:
                    continue
                ex = nx - self.goal_pos[0]
                ey = ny - self.goal_pos[1]
                if ex * ex + ey * ey <= reach_sq:
                    return steps + 1
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))
        return 1

    @staticmethod
    def _draw_filled_circle(arr: np.ndarray, cx: int, cy: int, radius: int, color: np.ndarray) -> None:
        height, width, _ = arr.shape
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        if x_max <= x_min or y_max <= y_min:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        arr[y_min:y_max, x_min:x_max, :][mask] = color


class PPOActorCriticCNN(nn.Module):
    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(1)
        return logits, value


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def __len__(self) -> int:
        return len(self.states)


class EpisodeStepsRatioCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(4.8, 3.6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def update_plot(self, ratios: List[float]) -> None:
        self.ax.clear()
        self.ax.set_title("Schritte pro Episode / Mindestschritte")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Verhaeltnis (Schritte / Min-Schritte)")
        self.ax.axhline(y=1.0, color="green", linestyle="--", linewidth=1.2, label="Optimal (= 1)")
        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.3, axis="y")
        if ratios:
            x_values = np.arange(1, len(ratios) + 1)
            self.ax.bar(x_values, ratios, color="#2B6CB0", width=0.8)
        self.draw_idle()


class PPORawPixelsDemo(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PPO Demo mit Rohbild-Input")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = GridWorldRawPixels(world_size=800, obs_size=84)
        self.n_actions = 4

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr = 2.5e-4
        self.rollout_steps = 256
        self.ppo_epochs = 4
        self.minibatch_size = 64
        self.env_steps_per_tick = 8
        self.max_episode_factor = 50

        self.policy_net = PPOActorCriticCNN(self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.rollout_buffer = RolloutBuffer()

        self.total_steps = 0
        self.total_updates = 0
        self.episode_steps = 0
        self.episode_count = 0
        self.episode_steps_history: List[float] = []

        self.last_r_energy = 0.0
        self.last_r_goal = 0.0
        self.last_r_dist = 0.0
        self.last_r_reversal = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_approx_kl = 0.0
        self.reversal_penalty = 0.03
        self.last_action: int | None = None

        self.current_observation = self.env.reset()
        self.current_episode_min_steps = max(1, self.env.compute_min_steps())

        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.run_training_batch)

        self.world_label = QLabel()
        self.world_label.setFixedSize(self.env.world_size, self.env.world_size)
        self.world_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #555;")

        self.steps_label = QLabel("Umweltschritte: 0")
        self.steps_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.updates_label = QLabel("PPO-Updates: 0")
        self.updates_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.status_label = QLabel("Status: gestoppt")
        self.status_label.setStyleSheet("font-size: 14px;")

        self.start_stop_button = QPushButton("Start Lernen")
        self.start_stop_button.clicked.connect(self.toggle_learning)

        self.space_hint_label = QLabel("Space: einen Einzelschritt ausfuehren")

        self.slow_ui_checkbox = QCheckBox("UI nur alle 1000 Schritte aktualisieren (maximale Lerngeschwindigkeit)")
        self.slow_ui_checkbox.setChecked(True)

        device_name = "GPU (CUDA)" if self.device.type == "cuda" else "CPU"
        self.device_label = QLabel(f"Modell laeuft auf: {device_name}")
        self.device_label.setStyleSheet("font-size: 13px; color: #1a6e2e; font-weight: bold;")

        self.pbrs_checkbox = QCheckBox("Potential-Based Reward Shaping (PBRS) aktiv")
        self.pbrs_checkbox.setChecked(True)

        self.cnn_input_title = QLabel("Policy-Input (84 x 84 x 3, RGB):")
        self.cnn_input_label = QLabel()
        self.cnn_input_label.setFixedSize(self.env.obs_size, self.env.obs_size)
        self.cnn_input_label.setStyleSheet("border: 1px solid #888;")

        self.plot_canvas = EpisodeStepsRatioCanvas()
        self.plot_canvas.setMinimumHeight(280)

        self.reward_breakdown_title = QLabel("Reward-Aufschluesselung (letzter Schritt):")
        self.reward_breakdown_title.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 6px;")
        self.r_energy_label = QLabel("  Energieverbrauch: -0.0100")
        self.r_goal_label = QLabel("  Zielerreichung:   +0.0000")
        self.r_dist_label = QLabel("  Distanzreduktion: +0.000000")
        self.r_reversal_label = QLabel("  Richtungswechsel: +0.0000")

        self.ppo_stats_title = QLabel("PPO-Statistik (letztes Update):")
        self.ppo_stats_title.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 6px;")
        self.policy_loss_label = QLabel("  Policy-Loss: +0.0000")
        self.value_loss_label = QLabel("  Value-Loss:  +0.0000")
        self.entropy_label = QLabel("  Entropie:    +0.0000")
        self.kl_label = QLabel("  Approx-KL:   +0.000000")
        for label in (
            self.r_energy_label,
            self.r_goal_label,
            self.r_dist_label,
            self.r_reversal_label,
            self.policy_loss_label,
            self.value_loss_label,
            self.entropy_label,
            self.kl_label,
        ):
            label.setStyleSheet("font-size: 13px; font-family: monospace;")

        self._setup_layout()
        self._refresh_world_view()
        self._refresh_input_preview()

    def _setup_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.world_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.device_label)
        right_layout.addWidget(self.steps_label)
        right_layout.addWidget(self.updates_label)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.start_stop_button)
        right_layout.addWidget(self.space_hint_label)
        right_layout.addWidget(self.slow_ui_checkbox)
        right_layout.addWidget(self.pbrs_checkbox)
        right_layout.addWidget(self.cnn_input_title)
        right_layout.addWidget(self.cnn_input_label)
        right_layout.addWidget(self.plot_canvas)
        right_layout.addWidget(self.reward_breakdown_title)
        right_layout.addWidget(self.r_energy_label)
        right_layout.addWidget(self.r_goal_label)
        right_layout.addWidget(self.r_dist_label)
        right_layout.addWidget(self.r_reversal_label)
        right_layout.addWidget(self.ppo_stats_title)
        right_layout.addWidget(self.policy_loss_label)
        right_layout.addWidget(self.value_loss_label)
        right_layout.addWidget(self.entropy_label)
        right_layout.addWidget(self.kl_label)
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
            self._run_env_steps(1)
            self._maybe_update_ui()
            event.accept()
            return
        super().keyPressEvent(event)

    def run_training_batch(self) -> None:
        self._run_env_steps(self.env_steps_per_tick)
        self._maybe_update_ui()

    def _run_env_steps(self, num_steps: int) -> None:
        for _ in range(num_steps):
            state_small = self._obs_to_state_tensor(self.current_observation)
            action, log_prob, value = self._sample_action_and_value(state_small)

            next_observation, reached_goal, old_dist, new_dist = self.env.step(action)

            r_energy = -0.01
            r_goal = 1.0 if reached_goal else 0.0
            if self.pbrs_checkbox.isChecked():
                phi_s = -old_dist / self.env._D_max
                phi_s_next = 0.0 if reached_goal else -new_dist / self.env._D_max
                r_dist = 20.0 * (self.gamma * phi_s_next - phi_s)
            else:
                r_dist = 0.0

            is_immediate_reversal = (
                self.last_action is not None
                and (
                    (self.last_action == self.env.ACTION_LEFT and action == self.env.ACTION_RIGHT)
                    or (self.last_action == self.env.ACTION_RIGHT and action == self.env.ACTION_LEFT)
                    or (self.last_action == self.env.ACTION_UP and action == self.env.ACTION_DOWN)
                    or (self.last_action == self.env.ACTION_DOWN and action == self.env.ACTION_UP)
                )
            )
            r_reversal = -self.reversal_penalty if is_immediate_reversal else 0.0
            reward = r_energy + r_goal + r_dist + r_reversal

            self.last_r_energy = r_energy
            self.last_r_goal = r_goal
            self.last_r_dist = r_dist
            self.last_r_reversal = r_reversal

            self.total_steps += 1
            self.episode_steps += 1

            max_episode_steps = self.max_episode_factor * self.current_episode_min_steps
            truncated = (not reached_goal) and (self.episode_steps >= max_episode_steps)
            done = reached_goal or truncated

            self.rollout_buffer.add(
                state=state_small,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
            )

            if done:
                ratio = self.episode_steps / self.current_episode_min_steps
                self.episode_steps_history.append(ratio)
                self.episode_count += 1
                self.episode_steps = 0
                self.current_observation = self.env.reset()
                self.current_episode_min_steps = max(1, self.env.compute_min_steps())
                self.last_action = None
            else:
                self.current_observation = next_observation
                self.last_action = action

            if len(self.rollout_buffer) >= self.rollout_steps:
                self._finish_rollout_and_update()

    def _sample_action_and_value(self, state_small: np.ndarray) -> tuple[int, float, float]:
        state_tensor = torch.from_numpy(state_small).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy_net(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _predict_value(self, observation: np.ndarray) -> float:
        state_small = self._obs_to_state_tensor(observation)
        state_tensor = torch.from_numpy(state_small).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.policy_net(state_tensor)
        return float(value.item())

    def _finish_rollout_and_update(self) -> None:
        buffer = self.rollout_buffer
        if not buffer.states:
            return

        bootstrap_value = 0.0 if buffer.dones[-1] else self._predict_value(self.current_observation)
        values = np.asarray(buffer.values + [bootstrap_value], dtype=np.float32)
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
        returns = advantages + values[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.from_numpy(np.stack(buffer.states)).float().to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        num_samples = states.shape[0]
        minibatch_size = min(self.minibatch_size, num_samples)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                logits, values_pred = self.policy_net(states[mb_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[mb_idx]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages_tensor[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values_pred, returns_tensor[mb_idx])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().abs()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))

        self.total_updates += 1
        self.last_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        self.last_entropy = float(np.mean(entropies)) if entropies else 0.0
        self.last_approx_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
        buffer.clear()

    def _obs_to_state_tensor(self, obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1).astype(np.float32)

    def _maybe_update_ui(self) -> None:
        slow_mode = self.slow_ui_checkbox.isChecked()
        if slow_mode and self.total_steps % 1000 != 0:
            return

        self.steps_label.setText(f"Umweltschritte: {self.total_steps}")
        self.updates_label.setText(f"PPO-Updates: {self.total_updates}")
        self.r_energy_label.setText(f"  Energieverbrauch: {self.last_r_energy:+.4f}")
        self.r_goal_label.setText(f"  Zielerreichung:   {self.last_r_goal:+.4f}")
        self.r_dist_label.setText(f"  Distanzreduktion: {self.last_r_dist:+.6f}")
        self.r_reversal_label.setText(f"  Richtungswechsel: {self.last_r_reversal:+.4f}")
        self.policy_loss_label.setText(f"  Policy-Loss: {self.last_policy_loss:+.4f}")
        self.value_loss_label.setText(f"  Value-Loss:  {self.last_value_loss:+.4f}")
        self.entropy_label.setText(f"  Entropie:    {self.last_entropy:+.4f}")
        self.kl_label.setText(f"  Approx-KL:   {self.last_approx_kl:+.6f}")
        self._refresh_world_view()
        self._refresh_input_preview()
        self.plot_canvas.update_plot(self.episode_steps_history[-50:])

    def _refresh_world_view(self) -> None:
        arr = np.clip(self.env.render_world() * 255.0, 0.0, 255.0).astype(np.uint8)
        height, width, _ = arr.shape
        bytes_per_line = 3 * width
        image = QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image.copy())
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(70, 70, 70), 2))
        painter.drawRect(1, 1, width - 2, height - 2)
        painter.end()

        self.world_label.setPixmap(pixmap)

    def _refresh_input_preview(self) -> None:
        rgb_obs = np.clip(self.current_observation * 255.0, 0.0, 255.0).astype(np.uint8).copy()
        obs_size = self.env.obs_size
        cnn_image = QImage(rgb_obs.data, obs_size, obs_size, 3 * obs_size, QImage.Format_RGB888)
        self.cnn_input_label.setPixmap(QPixmap.fromImage(cnn_image.copy()))


def main() -> None:
    app = QApplication(sys.argv)
    win = PPORawPixelsDemo()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
