"""RL comparison demo with PySide6.

The user can switch between seven agent variants:
- DQN
- DDQN
- Dueling DQN
- PER (implemented as DDQN with prioritized replay)
- REINFORCE (Monte-Carlo policy gradient)
- REINFORCE fast (higher learning rate)
- REINFORCE with baseline

Task:
The agent (black circle) has to follow a moving target (red circle).
State space: only the relative offset (dx, dy) to the current target.
Action space: UP, LEFT, RIGHT, DOWN.
Reward: negative normalized distance to the target.

The UI shows the simulation on the left, and controls plus a rolling reward
plot on the right.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This demo requires PyTorch. Install it in your environment, e.g. 'pip install torch'."
    ) from exc

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen, QBrush
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


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    use_double: bool
    use_dueling: bool
    use_per: bool
    use_reinforce: bool
    use_baseline: bool
    reinforce_alpha: float
    normalize_advantage: bool


ALGORITHMS: Sequence[AlgorithmSpec] = (
    AlgorithmSpec(
        "DQN",
        use_double=False,
        use_dueling=False,
        use_per=False,
        use_reinforce=False,
        use_baseline=False,
        reinforce_alpha=3e-4,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "DDQN",
        use_double=True,
        use_dueling=False,
        use_per=False,
        use_reinforce=False,
        use_baseline=False,
        reinforce_alpha=3e-4,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "Dueling DQN",
        use_double=True,
        use_dueling=True,
        use_per=False,
        use_reinforce=False,
        use_baseline=False,
        reinforce_alpha=3e-4,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "PER",
        use_double=True,
        use_dueling=False,
        use_per=True,
        use_reinforce=False,
        use_baseline=False,
        reinforce_alpha=3e-4,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "REINFORCE",
        use_double=False,
        use_dueling=False,
        use_per=False,
        use_reinforce=True,
        use_baseline=False,
        reinforce_alpha=3e-4,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "REINFORCE fast",
        use_double=False,
        use_dueling=False,
        use_per=False,
        use_reinforce=True,
        use_baseline=False,
        reinforce_alpha=1e-3,
        normalize_advantage=False,
    ),
    AlgorithmSpec(
        "REINFORCE with baseline",
        use_double=False,
        use_dueling=False,
        use_per=False,
        use_reinforce=True,
        use_baseline=True,
        reinforce_alpha=1e-3,
        normalize_advantage=True,
    ),
)


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
        self.agent_speed = 8.0
        self.target_speed = 3.5
        self.max_steps_per_episode = 100
        self.max_distance = float(np.hypot(self.width, self.height))

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
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

        self.steps = 0
        self.episode_reward = 0.0
        return self.state()

    def state(self) -> State:
        dx = (self.target_x - self.agent_x) / float(self.width)
        dy = (self.target_y - self.agent_y) / float(self.height)
        return np.array([dx, dy], dtype=np.float32)

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

        distance = self.distance_to_target()
        reward = -distance / self.max_distance
        self.episode_reward += reward
        self.steps += 1

        done = self.steps >= self.max_steps_per_episode
        info = {
            "distance": self.distance_to_target(),
            "in_zone": reward > 0.0,
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


@dataclass
class Transition:
    state: State
    action: int
    reward: float
    next_state: State
    done: bool


@dataclass
class SampleBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    weights: np.ndarray
    indices: np.ndarray | None = None


class UniformReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> SampleBatch:
        batch = random.sample(self.buffer, batch_size)
        return SampleBatch(
            states=np.stack([item.state for item in batch]),
            actions=np.array([item.action for item in batch], dtype=np.int64),
            rewards=np.array([item.reward for item in batch], dtype=np.float32),
            next_states=np.stack([item.next_state for item in batch]),
            dones=np.array([item.done for item in batch], dtype=np.float32),
            weights=np.ones((batch_size, 1), dtype=np.float32),
            indices=None,
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, priority_epsilon: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        max_priority = float(self.priorities.max()) if self.buffer else 1.0
        if max_priority <= 0.0:
            max_priority = 1.0
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> SampleBatch:
        size = len(self.buffer)
        scaled = self.priorities[:size] ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(size, batch_size, replace=size < batch_size, p=probs)
        samples = [self.buffer[int(idx)] for idx in indices]

        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return SampleBatch(
            states=np.stack([item.state for item in samples]),
            actions=np.array([item.action for item in samples], dtype=np.int64),
            rewards=np.array([item.reward for item in samples], dtype=np.float32),
            next_states=np.stack([item.next_state for item in samples]),
            dones=np.array([item.done for item in samples], dtype=np.float32),
            weights=weights.astype(np.float32).reshape(-1, 1),
            indices=indices.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities, strict=False):
            self.priorities[int(idx)] = float(abs(priority) + self.priority_epsilon)

    def __len__(self) -> int:
        return len(self.buffer)


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLearningAgent:
    def __init__(self, spec: AlgorithmSpec) -> None:
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = 4
        self.gamma = 0.98
        self.batch_size = 64
        self.learning_starts = 800
        self.target_update_interval = 500
        self.replay_size = 30_000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 25_000
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta_decay_steps = 40_000

        network_cls = DuelingQNetwork if self.spec.use_dueling else MLPQNetwork
        self.policy_net = network_cls(2, self.num_actions).to(self.device)
        self.target_net = network_cls(2, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        if self.spec.use_per:
            self.replay: UniformReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                self.replay_size,
                alpha=self.per_alpha,
            )
        else:
            self.replay = UniformReplayBuffer(self.replay_size)

        self.env_steps = 0
        self.training_steps = 0
        self.epsilon = self.epsilon_start

    def register_env_step(self) -> None:
        self.env_steps += 1
        progress = min(1.0, self.env_steps / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def beta(self) -> float:
        if not self.spec.use_per:
            return 1.0
        progress = min(1.0, self.training_steps / float(self.per_beta_decay_steps))
        return self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)

    def act(self, state: State, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(x)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        self.replay.add(
            Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
            )
        )

    def learn(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.learning_starts):
            return None

        if self.spec.use_per:
            batch = self.replay.sample(self.batch_size, beta=self.beta())
        else:
            batch = self.replay.sample(self.batch_size)

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.spec.use_double:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True).values

            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        td_errors = targets - q_values
        loss = (self.loss_fn(q_values, targets) * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.spec.use_per and batch.indices is not None:
            priorities = td_errors.detach().abs().squeeze(1).cpu().numpy()
            self.replay.update_priorities(batch.indices, priorities)

        return float(loss.item())


class ReinforceAgent:
    def __init__(self, spec: AlgorithmSpec) -> None:
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = 4
        self.alpha = self.spec.reinforce_alpha
        self.gamma = 0.98
        self.epsilon = 0.0
        self.policy_net = PolicyNetwork(2, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.training_steps = 0
        self.env_steps = 0

        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        self._pending_log_prob: torch.Tensor | None = None

    def register_env_step(self) -> None:
        self.env_steps += 1

    def act(self, state: State, greedy: bool = False) -> int:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(x)

        if greedy:
            action = int(torch.argmax(logits, dim=1).item())
            self._pending_log_prob = torch.log_softmax(logits, dim=1).squeeze(0)[action]
            return action

        dist = Categorical(logits=logits)
        action_tensor = dist.sample().squeeze(0)
        self._pending_log_prob = dist.log_prob(action_tensor)
        return int(action_tensor.item())

    def store(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        del state, action, next_state, done
        if self._pending_log_prob is None:
            return
        self.episode_log_probs.append(self._pending_log_prob)
        self.episode_rewards.append(float(reward))
        self._pending_log_prob = None

    def learn(self, episode_done: bool = False) -> float | None:
        if not episode_done:
            return None
        if not self.episode_rewards:
            return None

        # Pseudocode-like REINFORCE:
        # 1) For each t, compute R_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        # 2) Accumulate objective J = sum_t R_t * log pi(a_t | s_t)
        # 3) Perform one gradient-ascent update per episode.
        returns: List[float] = []
        horizon = len(self.episode_rewards)
        for t in range(horizon):
            r_t = 0.0
            for tp in range(t, horizon):
                r_t += (self.gamma ** (tp - t)) * self.episode_rewards[tp]
            returns.append(r_t)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        log_probs_t = torch.stack(self.episode_log_probs)

        baseline = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.spec.use_baseline:
            baseline = returns_t.mean()

        advantages = returns_t - baseline
        if self.spec.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        objective = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in range(horizon):
            objective = objective + advantages[t] * log_probs_t[t]

        # Optimizers minimize by default, so we minimize -J instead of maximizing J.
        loss = -objective

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1
        self.episode_log_probs.clear()
        self.episode_rewards.clear()

        return float(loss.item())


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

        painter.setPen(QColor(60, 60, 60))
        painter.setFont(QFont("DejaVu Sans", 10))
        info = (
            f"Agent (black) follows moving target (red)\n"
            f"State = [dx, dy] = [{self.state[0]:+.3f}, {self.state[1]:+.3f}]"
        )
        painter.drawText(world.adjusted(14, 14, -14, -14), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, info)


class RewardPlot(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(5.4, 3.6), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.steps: List[int] = []
        self.rolling_avg: List[float] = []
        self.setMinimumHeight(220)

    def clear(self) -> None:
        self.steps.clear()
        self.rolling_avg.clear()
        self.redraw()

    def add_point(self, step: int, rolling_average: float) -> None:
        self.steps.append(step)
        self.rolling_avg.append(rolling_average)
        if len(self.steps) > 5000:
            self.steps = self.steps[-5000:]
            self.rolling_avg = self.rolling_avg[-5000:]
        self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self.ax.set_title("Rolling average reward over the last 1000 steps")
        self.ax.set_xlabel("Training steps")
        self.ax.set_ylabel("Average reward")
        self.ax.grid(True, alpha=0.25)

        if self.steps:
            self.ax.plot(self.steps, self.rolling_avg, color="#1565c0", linewidth=1.8)
            self.ax.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
            self.ax.set_xlim(self.steps[0], self.steps[-1] + 1)
            y_max = max(0.05, max(self.rolling_avg) * 1.2)
            self.ax.set_ylim(0.0, max(0.2, y_max))

        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "RL comparison demo: DQN / DDQN / Dueling DQN / PER / REINFORCE / REINFORCE-fast / REINFORCE+Baseline"
        )
        self.resize(1280, 760)

        self.env = MovingTargetEnv()
        self.agent: QLearningAgent | ReinforceAgent = QLearningAgent(ALGORITHMS[0])
        self.running = False
        self.training_steps = 0
        self.episode_index = 1
        self.episode_reward = 0.0
        self.recent_rewards: Deque[float] = deque(maxlen=1000)
        self.last_loss: float | None = None
        self.steps_per_tick = 8
        self.render_enabled = True

        self.view = SimulationView(self.env)
        self.plot = RewardPlot()

        self.combo_algorithm = QComboBox()
        for spec in ALGORITHMS:
            self.combo_algorithm.addItem(spec.name, spec)
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
        self.lbl_distance = QLabel()
        self.lbl_epsilon = QLabel()
        self.lbl_replay = QLabel()
        self.lbl_loss = QLabel()
        self.lbl_rolling_reward = QLabel()
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
        side_container.setFrameShape(QFrame.Shape.StyledPanel)
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
            "PER uses prioritized replay; the other variants use uniform replay.\n"
            "REINFORCE updates once per episode. REINFORCE fast uses a higher learning rate.\n"
            "REINFORCE with baseline uses trajectory-mean baseline and normalized advantages.\n"
            "The agent only observes the relative offset to the moving target."
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
        stats_layout.addRow("Distance", self.lbl_distance)
        stats_layout.addRow("Epsilon", self.lbl_epsilon)
        stats_layout.addRow("Replay size", self.lbl_replay)
        stats_layout.addRow("Last loss", self.lbl_loss)
        stats_layout.addRow("Rolling reward", self.lbl_rolling_reward)
        stats_layout.addRow("Device", self.lbl_device)
        side_layout.addWidget(stats_box)

        side_layout.addStretch(1)

    def _quit_action(self) -> QAction:
        action = QAction("Quit", self)
        action.triggered.connect(self.close)
        return action

    def selected_spec(self) -> AlgorithmSpec:
        spec = self.combo_algorithm.currentData()
        assert isinstance(spec, AlgorithmSpec)
        return spec

    def _algorithm_changed(self, _index: int) -> None:
        if self.running:
            self.running = False
            self.btn_start_stop.setText("Start")
        self.reset_demo()

    def reset_demo(self) -> None:
        spec = self.selected_spec()
        self.env = MovingTargetEnv()
        if spec.use_reinforce:
            self.agent = ReinforceAgent(spec)
        else:
            self.agent = QLearningAgent(spec)
        self.training_steps = 0
        self.episode_index = 1
        self.episode_reward = 0.0
        self.recent_rewards.clear()
        self.last_loss = None
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
            # Refresh both canvases once when rendering is turned on again.
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

    def training_step(self) -> bool:
        state = self.env.state()
        action = self.agent.act(state)
        next_state, reward, done, info = self.env.step(action)
        del info

        self.agent.store(state, action, reward, next_state, done)
        self.agent.register_env_step()
        loss = self.agent.learn(episode_done=done) if isinstance(self.agent, ReinforceAgent) else self.agent.learn()
        if loss is not None:
            self.last_loss = loss

        self.training_steps += 1
        self.episode_reward += reward
        self.recent_rewards.append(reward)

        rolling_avg = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        if self.render_enabled and self.training_steps % 2 == 0:
            self.plot.add_point(self.training_steps, rolling_avg)

        if self.render_enabled:
            self.view.set_state(next_state)

        if done:
            self.episode_index += 1
            self.episode_reward = 0.0
            self.env.reset()
            if self.render_enabled:
                self.view.set_state(self.env.state())

        return done

    def _sync_ui(self) -> None:
        self.lbl_algorithm.setText(self.selected_spec().name)
        self.lbl_steps.setText(str(self.training_steps))
        self.lbl_episode.setText(str(self.episode_index))
        self.lbl_episode_reward.setText(f"{self.episode_reward:.2f}")
        self.lbl_distance.setText(f"{self.env.distance_to_target():.1f} px")
        if isinstance(self.agent, QLearningAgent):
            self.lbl_epsilon.setText(f"{self.agent.epsilon:.3f}")
            self.lbl_replay.setText(str(len(self.agent.replay)))
        else:
            self.lbl_epsilon.setText("-")
            self.lbl_replay.setText("-")
        self.lbl_loss.setText("-" if self.last_loss is None else f"{self.last_loss:.4f}")
        rolling_avg = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        self.lbl_rolling_reward.setText(f"{rolling_avg:.3f} (window=1000 steps)")
        self.lbl_device.setText(str(self.agent.device))


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()