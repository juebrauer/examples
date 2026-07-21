import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent_rl import RLInteractiveAgent
from agent_supervised import SupervisedNavigationAgent


WORLD_SIZE = 50
MAX_STEPS_PER_EPISODE = 250
TEST_EPISODES = 100
TRAIN_SAMPLES = 10000
TRAIN_BATCH_SIZE = 128
AGENT_MARKER_SIZE = 3
GOAL_MARKER_SIZE = 3
LOCAL_SAMPLE_PROBABILITY = 0.6
LOCAL_RADIUS = 6

# RL hyperparameters.  These are intentionally collected here so the
# experiment can be changed without touching the agent implementation.
RL_ACTOR_LR = 3e-4
RL_CRITIC_LR = 1e-3
RL_HISTORY_DECAY = 0.90
RL_ENTROPY_COEFFICIENT = 0.01
RL_ADVANTAGE_CLIP = 2.0

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def draw_square_marker(
    img: np.ndarray,
    channel: int,
    center_x: int,
    center_y: int,
    marker_size: int,
    value: float = 1.0,
) -> None:
    half = marker_size // 2
    y0 = max(0, center_y - half)
    y1 = min(img.shape[1], center_y + half + 1)
    x0 = max(0, center_x - half)
    x1 = min(img.shape[2], center_x + half + 1)
    img[channel, y0:y1, x0:x1] = value


def build_state_image(
    size: int,
    agent_x: int,
    agent_y: int,
    goal_x: int,
    goal_y: int,
) -> np.ndarray:
    img = np.zeros((3, size, size), dtype=np.float32)
    draw_square_marker(
        img=img,
        channel=2,
        center_x=agent_x,
        center_y=agent_y,
        marker_size=AGENT_MARKER_SIZE,
        value=1.0,
    )  # Agent blue channel
    draw_square_marker(
        img=img,
        channel=0,
        center_x=goal_x,
        center_y=goal_y,
        marker_size=GOAL_MARKER_SIZE,
        value=1.0,
    )  # Goal red channel
    return img


def expert_action_from_positions(agent_x: int, agent_y: int, goal_x: int, goal_y: int) -> int:
    dx = goal_x - agent_x
    dy = goal_y - agent_y

    if dx == 0 and dy == 0:
        return ACTION_UP

    if abs(dx) > abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    if abs(dy) > abs(dx):
        return ACTION_DOWN if dy > 0 else ACTION_UP

    return ACTION_DOWN if dy > 0 else ACTION_UP


def expert_action_distribution(
    agent_x: int,
    agent_y: int,
    goal_x: int,
    goal_y: int,
) -> np.ndarray:
    target = np.zeros((4,), dtype=np.float32)

    dx = goal_x - agent_x
    dy = goal_y - agent_y

    if dx > 0:
        target[ACTION_RIGHT] = 1.0
    elif dx < 0:
        target[ACTION_LEFT] = 1.0

    if dy > 0:
        target[ACTION_DOWN] = 1.0
    elif dy < 0:
        target[ACTION_UP] = 1.0

    s = float(target.sum())
    if s <= 0.0:
        target[ACTION_UP] = 1.0
        s = 1.0

    target /= s
    return target


def sample_training_positions(size: int, use_local_sampling: bool) -> tuple[int, int, int, int]:
    agent_x = random.randint(0, size - 1)
    agent_y = random.randint(0, size - 1)

    if use_local_sampling and random.random() < LOCAL_SAMPLE_PROBABILITY:
        while True:
            dx = random.randint(-LOCAL_RADIUS, LOCAL_RADIUS)
            dy = random.randint(-LOCAL_RADIUS, LOCAL_RADIUS)

            goal_x = agent_x + dx
            goal_y = agent_y + dy

            if (
                (dx != 0 or dy != 0)
                and 0 <= goal_x < size
                and 0 <= goal_y < size
            ):
                return agent_x, agent_y, goal_x, goal_y

    while True:
        goal_x = random.randint(0, size - 1)
        goal_y = random.randint(0, size - 1)
        if goal_x != agent_x or goal_y != agent_y:
            return agent_x, agent_y, goal_x, goal_y


@dataclass
class StepResult:
    done: bool
    reached_goal: bool


class GridWorld:
    def __init__(self, size: int = WORLD_SIZE, max_steps: int = MAX_STEPS_PER_EPISODE):
        self.size = size
        self.max_steps = max_steps
        self.agent_x = 0
        self.agent_y = 0
        self.goal_x = 0
        self.goal_y = 0
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        self.agent_x = random.randint(0, self.size - 1)
        self.agent_y = random.randint(0, self.size - 1)

        while True:
            self.goal_x = random.randint(0, self.size - 1)
            self.goal_y = random.randint(0, self.size - 1)
            if self.goal_x != self.agent_x or self.goal_y != self.agent_y:
                break

        self.step_count = 0

    def get_distance(self) -> float:
        dx = self.goal_x - self.agent_x
        dy = self.goal_y - self.agent_y
        return math.sqrt(dx * dx + dy * dy)

    def get_state_image(self) -> np.ndarray:
        return build_state_image(
            size=self.size,
            agent_x=self.agent_x,
            agent_y=self.agent_y,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
        )

    def expert_action(self) -> int:
        return expert_action_from_positions(
            self.agent_x,
            self.agent_y,
            self.goal_x,
            self.goal_y,
        )

    def step(self, action: int) -> StepResult:
        if action == ACTION_UP:
            self.agent_y = clamp(self.agent_y - 1, 0, self.size - 1)
        elif action == ACTION_DOWN:
            self.agent_y = clamp(self.agent_y + 1, 0, self.size - 1)
        elif action == ACTION_LEFT:
            self.agent_x = clamp(self.agent_x - 1, 0, self.size - 1)
        elif action == ACTION_RIGHT:
            self.agent_x = clamp(self.agent_x + 1, 0, self.size - 1)

        self.step_count += 1
        reached_goal = (self.agent_x == self.goal_x and self.agent_y == self.goal_y)
        done = reached_goal or self.step_count >= self.max_steps
        return StepResult(done=done, reached_goal=reached_goal)


class TrainingWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        mode: str,
        total_train_epochs: int,
        total_test_episodes: int,
        supervised_agent: SupervisedNavigationAgent,
        rl_agent: RLInteractiveAgent,
        device: torch.device,
        world_size: int,
        max_steps: int,
        test_step_delay_ms: int,
        deterministic_action_selection: bool,
        cycle_blocker_enabled: bool,
        local_sampling_enabled: bool,
        soft_targets_enabled: bool,
        rl_history_length: int,
        fast_mode_enabled: bool,
    ):
        super().__init__()
        self.mode = mode
        self.total_train_epochs = total_train_epochs
        self.total_test_episodes = total_test_episodes
        self.supervised_agent = supervised_agent
        self.rl_agent = rl_agent
        self.device = device
        self.world_size = world_size
        self.max_steps = max_steps
        self.test_step_delay_ms = test_step_delay_ms
        self.deterministic_action_selection = deterministic_action_selection
        self.cycle_blocker_enabled = cycle_blocker_enabled
        self.local_sampling_enabled = local_sampling_enabled
        self.soft_targets_enabled = soft_targets_enabled
        self.rl_history_length = max(1, rl_history_length)
        self.fast_mode_enabled = fast_mode_enabled
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _sleep_with_stop(self, total_ms: int) -> None:
        remaining = max(0, total_ms)
        while remaining > 0 and not self._stop_requested:
            chunk = min(remaining, 5)
            time.sleep(chunk / 1000.0)
            remaining -= chunk

    @Slot()
    def run(self) -> None:
        if self.mode == "training_supervised":
            self._run_training_supervised()
            return
        if self.mode == "testing_supervised":
            self._run_testing_supervised()
            return
        if self.mode == "training_rl":
            self._run_training_rl()
            return
        self._run_testing_rl()

    def _compute_reward(self, prev_dist: float, dist: float, reached_goal: bool) -> float:
        reward = -0.02
        if dist < prev_dist - 1e-9:
            reward += 0.15
        elif dist > prev_dist + 1e-9:
            reward -= 0.15
        if reached_goal:
            reward += 1.0
        return reward

    def _run_training_supervised(self) -> None:
        epochs_done = self.supervised_agent.train(
            total_epochs=self.total_train_epochs,
            should_stop=lambda: self._stop_requested,
            progress_callback=lambda payload: self.progress.emit(payload),
            local_sampling_enabled=self.local_sampling_enabled,
            soft_targets_enabled=self.soft_targets_enabled,
            sample_training_positions_fn=sample_training_positions,
            build_state_image_fn=build_state_image,
            expert_action_from_positions_fn=expert_action_from_positions,
            expert_action_distribution_fn=expert_action_distribution,
        )

        self.finished.emit(
            {
                "mode": "training_supervised",
                "stopped": self._stop_requested,
                "epochs_done": epochs_done,
                "epochs_target": self.total_train_epochs,
            }
        )

    def _select_action_from_scores(
        self,
        scores: torch.Tensor,
        deterministic: bool,
    ) -> int:
        if deterministic:
            return int(torch.argmax(scores).item())

        probs = torch.softmax(scores, dim=0)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _run_testing_supervised(self) -> None:
        env = GridWorld(size=self.world_size, max_steps=self.max_steps)
        self.supervised_agent.model.eval()

        current_episode = 1
        global_step = 0
        successes = 0
        steps_sum = 0

        with torch.no_grad():
            while current_episode <= self.total_test_episodes and not self._stop_requested:
                env.reset()
                prev_dist = env.get_distance()
                position_history: deque[tuple[int, int]] = deque(maxlen=2)
                last_action = None
                opposite_action = {
                    ACTION_UP: ACTION_DOWN,
                    ACTION_DOWN: ACTION_UP,
                    ACTION_LEFT: ACTION_RIGHT,
                    ACTION_RIGHT: ACTION_LEFT,
                }

                while True:
                    state = env.get_state_image()
                    scores = self.supervised_agent.predict_scores_from_state_image(state).clone()
                    current_position = (env.agent_x, env.agent_y)

                    if self.cycle_blocker_enabled:
                        if (
                            len(position_history) == 2
                            and current_position == position_history[0]
                            and last_action is not None
                        ):
                            forbidden_action = opposite_action[last_action]
                            scores[forbidden_action] = float("-inf")

                    action = self._select_action_from_scores(
                        scores,
                        deterministic=self.deterministic_action_selection,
                    )

                    result = env.step(action)
                    position_history.append(current_position)
                    last_action = action
                    state_after_action = env.get_state_image()

                    global_step += 1
                    dist = env.get_distance()
                    if dist < prev_dist - 1e-9:
                        dist_trend = 1
                    elif dist > prev_dist + 1e-9:
                        dist_trend = -1
                    else:
                        dist_trend = 0
                    prev_dist = dist

                    self.progress.emit(
                        {
                            "mode": "testing_supervised",
                            "episode": current_episode,
                            "step": env.step_count,
                            "global_step": global_step,
                            "distance": dist,
                            "distance_trend": dist_trend,
                            "action": action,
                            "loss": None,
                            "agent_x": env.agent_x,
                            "agent_y": env.agent_y,
                            "goal_x": env.goal_x,
                            "goal_y": env.goal_y,
                            "state_image": state_after_action,
                            "episode_done": result.done,
                        }
                    )

                    # Pace testing so every single step is visible in the UI.
                    if self.test_step_delay_ms > 0:
                        self._sleep_with_stop(self.test_step_delay_ms)

                    if result.done or self._stop_requested:
                        if result.reached_goal:
                            successes += 1
                        steps_sum += env.step_count
                        break

                current_episode += 1

        episodes_done = min(current_episode - 1, self.total_test_episodes)
        success_rate = 0.0
        avg_steps = 0.0
        if episodes_done > 0:
            success_rate = 100.0 * successes / episodes_done
            avg_steps = steps_sum / episodes_done

        self.finished.emit(
            {
                "mode": "testing_supervised",
                "stopped": self._stop_requested,
                "episodes_done": episodes_done,
                "episodes_target": self.total_test_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
            }
        )

    def _run_training_rl(self) -> None:
        env = GridWorld(size=self.world_size, max_steps=self.max_steps)
        current_episode = 1
        global_step = 0
        successes = 0
        steps_sum = 0

        while current_episode <= self.total_train_epochs and not self._stop_requested:
            env.reset()
            self.rl_agent.begin_episode(self.rl_history_length)
            prev_dist = env.get_distance()

            episode_actor_loss_sum = 0.0
            episode_critic_loss_sum = 0.0
            episode_reward_sum = 0.0
            episode_abs_advantage_sum = 0.0
            episode_entropy_sum = 0.0
            episode_step_count = 0

            while True:
                if self._stop_requested:
                    break

                state = env.get_state_image()
                action, _ = self.rl_agent.act(state, deterministic=False)
                result = env.step(action)
                state_after_action = env.get_state_image()

                global_step += 1
                episode_step_count += 1
                dist = env.get_distance()

                if dist < prev_dist - 1e-9:
                    dist_trend = 1
                elif dist > prev_dist + 1e-9:
                    dist_trend = -1
                else:
                    dist_trend = 0

                reward = self._compute_reward(prev_dist, dist, result.reached_goal)
                prev_dist = dist

                learning_metrics = self.rl_agent.observe_and_learn(state, action, reward)
                actor_loss = float(learning_metrics["actor_loss"])
                critic_loss = float(learning_metrics["critic_loss"])

                episode_actor_loss_sum += actor_loss
                episode_critic_loss_sum += critic_loss
                episode_reward_sum += reward
                episode_abs_advantage_sum += abs(float(learning_metrics["advantage"]))
                episode_entropy_sum += float(learning_metrics["entropy"])

                if not self.fast_mode_enabled:
                    self.progress.emit(
                        {
                            "mode": "training_rl",
                            "episode": current_episode,
                            "step": env.step_count,
                            "global_step": global_step,
                            "distance": dist,
                            "distance_trend": dist_trend,
                            "action": action,
                            "loss": actor_loss + critic_loss,
                            "agent_x": env.agent_x,
                            "agent_y": env.agent_y,
                            "goal_x": env.goal_x,
                            "goal_y": env.goal_y,
                            "state_image": state_after_action,
                            "episode_done": result.done,
                        }
                    )

                if result.done:
                    if result.reached_goal:
                        successes += 1
                    steps_sum += env.step_count
                    break

            if episode_step_count > 0:
                avg_actor_loss = episode_actor_loss_sum / episode_step_count
                avg_critic_loss = episode_critic_loss_sum / episode_step_count
                avg_abs_advantage = episode_abs_advantage_sum / episode_step_count
                avg_entropy = episode_entropy_sum / episode_step_count
            else:
                avg_actor_loss = 0.0
                avg_critic_loss = 0.0
                avg_abs_advantage = 0.0
                avg_entropy = 0.0

            self.progress.emit(
                {
                    "mode": "training_rl_episode",
                    "episode": current_episode,
                    "episodes_total": self.total_train_epochs,
                    "avg_actor_loss": avg_actor_loss,
                    "avg_critic_loss": avg_critic_loss,
                    "avg_abs_advantage": avg_abs_advantage,
                    "avg_entropy": avg_entropy,
                    "reward_sum": episode_reward_sum,
                }
            )

            current_episode += 1

        episodes_done = min(current_episode - 1, self.total_train_epochs)
        success_rate = 0.0
        avg_steps = 0.0
        if episodes_done > 0:
            success_rate = 100.0 * successes / episodes_done
            avg_steps = steps_sum / episodes_done

        self.finished.emit(
            {
                "mode": "training_rl",
                "stopped": self._stop_requested,
                "episodes_done": episodes_done,
                "episodes_target": self.total_train_epochs,
                "successes": successes,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
            }
        )

    def _run_testing_rl(self) -> None:
        env = GridWorld(size=self.world_size, max_steps=self.max_steps)
        current_episode = 1
        global_step = 0
        successes = 0
        steps_sum = 0

        while current_episode <= self.total_test_episodes and not self._stop_requested:
            env.reset()
            self.rl_agent.begin_episode(self.rl_history_length)
            prev_dist = env.get_distance()

            position_history: deque[tuple[int, int]] = deque(maxlen=2)
            last_action = None
            opposite_action = {
                ACTION_UP: ACTION_DOWN,
                ACTION_DOWN: ACTION_UP,
                ACTION_LEFT: ACTION_RIGHT,
                ACTION_RIGHT: ACTION_LEFT,
            }

            while True:
                if self._stop_requested:
                    break

                state = env.get_state_image()
                action, probs = self.rl_agent.act(
                    state,
                    deterministic=self.deterministic_action_selection,
                )
                current_position = (env.agent_x, env.agent_y)

                if self.cycle_blocker_enabled:
                    if (
                        len(position_history) == 2
                        and current_position == position_history[0]
                        and last_action is not None
                    ):
                        forbidden_action = opposite_action[last_action]
                        probs = probs.copy()
                        probs[forbidden_action] = 0.0
                        probs_sum = float(probs.sum())
                        if probs_sum > 1e-12:
                            probs = probs / probs_sum

                            if self.deterministic_action_selection:
                                action = int(np.argmax(probs))
                            else:
                                action = int(np.random.choice(4, p=probs))

                result = env.step(action)
                position_history.append(current_position)
                last_action = action
                state_after_action = env.get_state_image()

                global_step += 1
                dist = env.get_distance()
                if dist < prev_dist - 1e-9:
                    dist_trend = 1
                elif dist > prev_dist + 1e-9:
                    dist_trend = -1
                else:
                    dist_trend = 0
                prev_dist = dist

                self.progress.emit(
                    {
                        "mode": "testing_rl",
                        "episode": current_episode,
                        "step": env.step_count,
                        "global_step": global_step,
                        "distance": dist,
                        "distance_trend": dist_trend,
                        "action": action,
                        "loss": None,
                        "agent_x": env.agent_x,
                        "agent_y": env.agent_y,
                        "goal_x": env.goal_x,
                        "goal_y": env.goal_y,
                        "state_image": state_after_action,
                        "episode_done": result.done,
                    }
                )

                if self.test_step_delay_ms > 0:
                    self._sleep_with_stop(self.test_step_delay_ms)

                if result.done:
                    if result.reached_goal:
                        successes += 1
                    steps_sum += env.step_count
                    break

            current_episode += 1

        episodes_done = min(current_episode - 1, self.total_test_episodes)
        success_rate = 0.0
        avg_steps = 0.0
        if episodes_done > 0:
            success_rate = 100.0 * successes / episodes_done
            avg_steps = steps_sum / episodes_done

        self.finished.emit(
            {
                "mode": "testing_rl",
                "stopped": self._stop_requested,
                "episodes_done": episodes_done,
                "episodes_target": self.total_test_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
            }
        )


class WorldWidget(QWidget):
    def __init__(self, world_size: int, parent=None):
        super().__init__(parent)
        self.world_size = world_size
        self.state_image = np.zeros((3, world_size, world_size), dtype=np.float32)
        self.setMinimumSize(420, 420)

    def set_state_image(self, state_image: np.ndarray) -> None:
        if state_image.shape != (3, self.world_size, self.world_size):
            return
        self.state_image = np.asarray(state_image, dtype=np.float32)
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        painter.fillRect(self.rect(), QColor(25, 25, 25))

        size = min(self.width(), self.height())
        cell = size / self.world_size
        ox = (self.width() - size) / 2.0
        oy = (self.height() - size) / 2.0

        painter.fillRect(int(ox), int(oy), int(size), int(size), QColor(235, 235, 235))

        pixel_size = max(2, int(math.ceil(cell)))

        # Render exactly what the model sees: RGB values from state_image channels.
        for gy in range(self.world_size):
            for gx in range(self.world_size):
                r = int(np.clip(self.state_image[0, gy, gx], 0.0, 1.0) * 255.0)
                g = int(np.clip(self.state_image[1, gy, gx], 0.0, 1.0) * 255.0)
                b = int(np.clip(self.state_image[2, gy, gx], 0.0, 1.0) * 255.0)
                if r == 0 and g == 0 and b == 0:
                    continue

                px = int(ox + gx * cell)
                py = int(oy + gy * cell)
                painter.fillRect(px, py, pixel_size, pixel_size, QColor(r, g, b))


class DistancePlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

        self.steps = []
        self.trends = []

        self.ax.set_title("Distanztrend pro Step")
        self.ax.set_xlabel("Globaler Step")
        self.ax.set_ylabel("Trend (-1/0/1)")
        self.ax.grid(True, alpha=0.3)

    def reset(self) -> None:
        self.steps.clear()
        self.trends.clear()
        self.ax.clear()
        self.ax.set_title("Distanztrend pro Step")
        self.ax.set_xlabel("Globaler Step")
        self.ax.set_ylabel("Trend (-1/0/1)")
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_yticks([-1, 0, 1])
        self.ax.grid(True, alpha=0.3)
        self.draw_idle()

    def add_point(self, step: int, trend: int) -> None:
        self.steps.append(step)
        self.trends.append(trend)
        self.ax.clear()
        self.ax.plot(self.steps, self.trends, color="tab:blue", linewidth=1.2)
        self.ax.set_title("Distanztrend pro Step")
        self.ax.set_xlabel("Globaler Step")
        self.ax.set_ylabel("Trend (-1/0/1)")
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_yticks([-1, 0, 1])
        self.ax.grid(True, alpha=0.3)

        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 + PyTorch: Supervised + RL Navigation Demo")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supervised_agent = SupervisedNavigationAgent(
            world_size=WORLD_SIZE,
            device=self.device,
            lr=1e-3,
        )
        self.rl_agent = RLInteractiveAgent(
            world_size=WORLD_SIZE,
            device=self.device,
            actor_lr=RL_ACTOR_LR,
            critic_lr=RL_CRITIC_LR,
            history_decay=RL_HISTORY_DECAY,
            entropy_coefficient=RL_ENTROPY_COEFFICIENT,
            advantage_clip=RL_ADVANTAGE_CLIP,
        )

        vector_len = self.supervised_agent.get_feature_vector_length()
        self.pre_flatten_shape_text = f"1D Laenge: {vector_len}"

        self.mode = None
        self.worker_thread = None
        self.worker = None

        self.fast_mode_state = False
        self.deterministic_action_selection = True
        self.cycle_blocker_enabled = False
        self.local_sampling_enabled = True
        self.soft_targets_enabled = False
        # Ten recent decisions make the delayed-credit effect visible; choose
        # N=1 in the UI for the simplest immediate-reward baseline.
        self.rl_history_length = 10

        self.total_train_epochs = 0
        self.total_test_episodes = TEST_EPISODES
        self.current_episode = 0
        self.global_step = 0
        self.test_step_delay_ms = 30

        self.build_ui()
        self.reset_world_visual_random()
        self.refresh_info_labels(last_action="-", dist_trend=None)

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        self.world_widget = WorldWidget(WORLD_SIZE)
        root_layout.addWidget(self.world_widget, stretch=3)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        root_layout.addWidget(right_panel, stretch=2)

        form = QFormLayout()

        self.episodes_spinbox = QSpinBox()
        self.episodes_spinbox.setRange(1, 10000)
        self.episodes_spinbox.setValue(50)
        form.addRow("Trainingsepochen / RL-Episoden:", self.episodes_spinbox)

        self.rl_history_spinbox = QSpinBox()
        self.rl_history_spinbox.setRange(1, 100)
        self.rl_history_spinbox.setValue(self.rl_history_length)
        self.rl_history_spinbox.valueChanged.connect(self.on_rl_history_length_changed)
        form.addRow("RL History N:", self.rl_history_spinbox)

        self.fast_mode_checkbox = QCheckBox("Visualisierung aus (Fast Mode)")
        self.fast_mode_checkbox.setChecked(False)
        self.fast_mode_checkbox.stateChanged.connect(self.on_fast_mode_changed)
        form.addRow(self.fast_mode_checkbox)

        self.deterministic_actions_checkbox = QCheckBox("Deterministic Action Selection (argmax)")
        self.deterministic_actions_checkbox.setChecked(True)
        self.deterministic_actions_checkbox.stateChanged.connect(
            self.on_deterministic_actions_changed
        )
        form.addRow(self.deterministic_actions_checkbox)

        self.cycle_blocker_checkbox = QCheckBox("2-Zyklus-Blocker (Testing)")
        self.cycle_blocker_checkbox.setChecked(False)
        self.cycle_blocker_checkbox.stateChanged.connect(self.on_cycle_blocker_changed)
        form.addRow(self.cycle_blocker_checkbox)

        self.local_sampling_checkbox = QCheckBox("Nahbereichs-Sampling (Training)")
        self.local_sampling_checkbox.setChecked(True)
        self.local_sampling_checkbox.stateChanged.connect(self.on_local_sampling_changed)
        form.addRow(self.local_sampling_checkbox)

        self.soft_targets_checkbox = QCheckBox("Soft Targets (Training)")
        self.soft_targets_checkbox.setChecked(False)
        self.soft_targets_checkbox.stateChanged.connect(self.on_soft_targets_changed)
        form.addRow(self.soft_targets_checkbox)

        self.start_training_btn = QPushButton("Train Supervised Agent")
        self.start_training_btn.clicked.connect(self.on_training_button_clicked)
        form.addRow(self.start_training_btn)

        self.start_testing_btn = QPushButton("Test Supervised Agent (100 Episoden)")
        self.start_testing_btn.clicked.connect(self.start_testing_supervised)
        form.addRow(self.start_testing_btn)

        self.start_rl_training_btn = QPushButton("Train RL Agent")
        self.start_rl_training_btn.clicked.connect(self.on_rl_training_button_clicked)
        form.addRow(self.start_rl_training_btn)

        self.start_rl_testing_btn = QPushButton("Test RL Agent (100 Episoden)")
        self.start_rl_testing_btn.clicked.connect(self.start_testing_rl)
        form.addRow(self.start_rl_testing_btn)

        self.mode_label = QLabel("Mode: idle")
        form.addRow("Status:", self.mode_label)

        self.episode_label = QLabel("0")
        form.addRow("Episoden-Nr:", self.episode_label)

        self.step_label = QLabel("0")
        form.addRow("Step-Nr:", self.step_label)

        self.tensor_shape_label = QLabel(self.pre_flatten_shape_text)
        form.addRow("Input ins MLP:", self.tensor_shape_label)

        self.dist_trend_label = QLabel("0")
        form.addRow("Distanztrend (1/0/-1):", self.dist_trend_label)

        self.action_label = QLabel("-")
        form.addRow("Letzte Aktion:", self.action_label)

        self.loss_label = QLabel("-")
        form.addRow("Letzter Loss:", self.loss_label)

        self.train_loss_text = QTextEdit()
        self.train_loss_text.setReadOnly(True)
        self.train_loss_text.setMinimumHeight(140)
        form.addRow("Loss pro Epoche:", self.train_loss_text)

        self.result_label = QLabel("-")
        self.result_label.setWordWrap(True)
        form.addRow("Test-Ergebnis:", self.result_label)

        right_layout.addLayout(form)

        self.plot_canvas = DistancePlotCanvas()
        right_layout.addWidget(self.plot_canvas, stretch=1)

        self.update_controls_state()

    def reset_world_visual_random(self) -> None:
        agent_x = random.randint(0, WORLD_SIZE - 1)
        agent_y = random.randint(0, WORLD_SIZE - 1)

        while True:
            goal_x = random.randint(0, WORLD_SIZE - 1)
            goal_y = random.randint(0, WORLD_SIZE - 1)
            if goal_x != agent_x or goal_y != agent_y:
                break

        state = build_state_image(
            size=WORLD_SIZE,
            agent_x=agent_x,
            agent_y=agent_y,
            goal_x=goal_x,
            goal_y=goal_y,
        )
        self.world_widget.set_state_image(state)

    def refresh_info_labels(
        self,
        last_action: str,
        loss_value: float | None = None,
        dist_trend: int | None = None,
    ) -> None:
        self.episode_label.setText(str(self.current_episode))
        self.action_label.setText(last_action)

        if dist_trend is None:
            self.dist_trend_label.setText("-")
        else:
            self.dist_trend_label.setText(str(dist_trend))

        if loss_value is None:
            self.loss_label.setText("-")
        else:
            self.loss_label.setText(f"{loss_value:.5f}")

    def update_controls_state(self) -> None:
        in_training = self.mode in {"training_supervised", "training_rl"}
        in_testing = self.mode in {"testing_supervised", "testing_rl"}
        is_running = in_training or in_testing

        self.episodes_spinbox.setEnabled(not is_running)
        self.rl_history_spinbox.setEnabled(not is_running)

        # Fast mode can be changed while running.
        self.fast_mode_checkbox.setEnabled(True)
        self.deterministic_actions_checkbox.setEnabled(True)
        self.cycle_blocker_checkbox.setEnabled(True)
        self.local_sampling_checkbox.setEnabled(True)
        self.soft_targets_checkbox.setEnabled(True)

        self.start_training_btn.setEnabled(self.mode in {None, "training_supervised"})
        self.start_training_btn.setText(
            "Stop Supervised Training"
            if self.mode == "training_supervised"
            else "Train Supervised Agent"
        )

        self.start_rl_training_btn.setEnabled(self.mode in {None, "training_rl"})
        self.start_rl_training_btn.setText(
            "Stop RL Training"
            if self.mode == "training_rl"
            else "Train RL Agent"
        )

        # Testing can only be started when idle.
        self.start_testing_btn.setEnabled(not is_running)
        self.start_rl_testing_btn.setEnabled(not is_running)

    def fast_mode(self) -> bool:
        return self.fast_mode_state

    def on_fast_mode_changed(self) -> None:
        self.fast_mode_state = self.fast_mode_checkbox.isChecked()
        self.test_step_delay_ms = 0 if self.fast_mode_state else 30
        if self.worker is not None:
            self.worker.test_step_delay_ms = self.test_step_delay_ms
            self.worker.fast_mode_enabled = self.fast_mode_state

    def on_deterministic_actions_changed(self) -> None:
        self.deterministic_action_selection = self.deterministic_actions_checkbox.isChecked()

    def on_cycle_blocker_changed(self) -> None:
        self.cycle_blocker_enabled = self.cycle_blocker_checkbox.isChecked()

    def on_local_sampling_changed(self) -> None:
        self.local_sampling_enabled = self.local_sampling_checkbox.isChecked()

    def on_soft_targets_changed(self) -> None:
        self.soft_targets_enabled = self.soft_targets_checkbox.isChecked()

    def on_rl_history_length_changed(self) -> None:
        self.rl_history_length = self.rl_history_spinbox.value()

    def on_training_button_clicked(self) -> None:
        if self.mode == "training_supervised":
            self.stop_current_training()
            return

        if self.mode is None:
            self.start_training_supervised()

    def on_rl_training_button_clicked(self) -> None:
        if self.mode == "training_rl":
            self.stop_current_training()
            return

        if self.mode is None:
            self.start_training_rl()

    def stop_current_training(self) -> None:
        if self.mode not in {"training_supervised", "training_rl"}:
            return

        if self.worker is not None:
            self.worker.request_stop()

    def start_training_supervised(self) -> None:
        self.mode = "training_supervised"
        self.total_train_epochs = self.episodes_spinbox.value()
        self.current_episode = 1
        self.global_step = 0
        self.result_label.setText("-")
        self.mode_label.setText("Mode: training supervised")
        self.plot_canvas.reset()
        self.train_loss_text.clear()

        self.update_controls_state()
        self.reset_world_visual_random()
        self.refresh_info_labels(last_action="-", dist_trend=None)
        self.step_label.setText("0")
        self.start_worker(mode="training_supervised")

    def start_testing_supervised(self) -> None:
        self.mode = "testing_supervised"
        self.current_episode = 1
        self.global_step = 0
        self.mode_label.setText("Mode: testing supervised")
        self.plot_canvas.reset()

        self.update_controls_state()
        self.reset_world_visual_random()
        self.refresh_info_labels(last_action="-", dist_trend=None)
        self.step_label.setText("0")
        self.start_worker(mode="testing_supervised")

    def start_training_rl(self) -> None:
        self.mode = "training_rl"
        self.total_train_epochs = self.episodes_spinbox.value()
        self.current_episode = 1
        self.global_step = 0
        self.result_label.setText("-")
        self.mode_label.setText("Mode: training rl")
        self.plot_canvas.reset()
        self.train_loss_text.clear()

        self.update_controls_state()
        self.reset_world_visual_random()
        self.refresh_info_labels(last_action="-", dist_trend=None)
        self.step_label.setText("0")
        self.start_worker(mode="training_rl")

    def start_testing_rl(self) -> None:
        self.mode = "testing_rl"
        self.current_episode = 1
        self.global_step = 0
        self.mode_label.setText("Mode: testing rl")
        self.plot_canvas.reset()

        self.update_controls_state()
        self.reset_world_visual_random()
        self.refresh_info_labels(last_action="-", dist_trend=None)
        self.step_label.setText("0")
        self.start_worker(mode="testing_rl")

    def finish_run(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.worker = None
        self.mode = None
        self.mode_label.setText("Mode: idle")
        self.update_controls_state()

    def start_worker(self, mode: str) -> None:
        self.worker_thread = QThread(self)
        self.worker = TrainingWorker(
            mode=mode,
            total_train_epochs=self.total_train_epochs,
            total_test_episodes=TEST_EPISODES,
            supervised_agent=self.supervised_agent,
            rl_agent=self.rl_agent,
            device=self.device,
            world_size=WORLD_SIZE,
            max_steps=MAX_STEPS_PER_EPISODE,
            test_step_delay_ms=self.test_step_delay_ms,
            deterministic_action_selection=self.deterministic_action_selection,
            cycle_blocker_enabled=self.cycle_blocker_enabled,
            local_sampling_enabled=self.local_sampling_enabled,
            soft_targets_enabled=self.soft_targets_enabled,
            rl_history_length=self.rl_history_length,
            fast_mode_enabled=self.fast_mode_state,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.start()

    @Slot(object)
    def on_worker_progress(self, payload: dict) -> None:
        mode = payload.get("mode")
        if mode == "training_supervised":
            epoch = int(payload["epoch"])
            epochs_total = int(payload["epochs_total"])
            epoch_loss = float(payload["epoch_loss"])

            self.current_episode = epoch
            self.episode_label.setText(str(epoch))
            self.loss_label.setText(f"{epoch_loss:.5f}")
            self.train_loss_text.append(
                f"Epoch {epoch}/{epochs_total}: {epoch_loss:.6f}"
            )
            return

        if mode == "training_rl_episode":
            episode = int(payload["episode"])
            episodes_total = int(payload["episodes_total"])
            avg_actor_loss = float(payload["avg_actor_loss"])
            avg_critic_loss = float(payload["avg_critic_loss"])
            avg_abs_advantage = float(payload["avg_abs_advantage"])
            avg_entropy = float(payload["avg_entropy"])
            reward_sum = float(payload["reward_sum"])
            self.train_loss_text.append(
                (
                    f"RL Episode {episode}/{episodes_total}: "
                    f"actor={avg_actor_loss:.6f}, critic={avg_critic_loss:.6f}, "
                    f"|adv|={avg_abs_advantage:.4f}, entropy={avg_entropy:.4f}, "
                    f"reward={reward_sum:.3f}"
                )
            )
            return

        self.current_episode = int(payload["episode"])
        self.global_step = int(payload["global_step"])
        step_in_episode = int(payload["step"])
        dist_trend = int(payload["distance_trend"])
        action = int(payload["action"])
        loss_value = payload["loss"]

        state_image = payload.get("state_image")
        if state_image is not None:
            self.world_widget.set_state_image(state_image)

        self.step_label.setText(str(step_in_episode))
        self.plot_canvas.add_point(self.global_step, dist_trend)
        self.refresh_info_labels(
            last_action=ACTION_NAMES[action],
            loss_value=loss_value,
            dist_trend=dist_trend,
        )

    @Slot(object)
    def on_worker_finished(self, payload: dict) -> None:
        mode = payload.get("mode")
        stopped = bool(payload.get("stopped", False))

        if mode == "training_supervised":
            epochs_done = int(payload.get("epochs_done", 0))
            epochs_target = int(payload.get("epochs_target", 0))
            if stopped:
                self.result_label.setText(
                    f"Supervised-Training manuell gestoppt bei Epoche {epochs_done}."
                )
            else:
                QMessageBox.information(
                    self,
                    "Training abgeschlossen",
                    f"Supervised-Training mit {epochs_target} Epochen abgeschlossen.",
                )
        elif mode == "training_rl":
            episodes_done = int(payload.get("episodes_done", 0))
            episodes_target = int(payload.get("episodes_target", 0))
            successes = int(payload.get("successes", 0))
            success_rate = float(payload.get("success_rate", 0.0))
            avg_steps = float(payload.get("avg_steps", 0.0))

            self.result_label.setText(
                f"RL Training (Erfolge): {successes}/{episodes_done} ({success_rate:.1f}%), avg Steps: {avg_steps:.1f}"
            )

            if not stopped:
                QMessageBox.information(
                    self,
                    "RL-Training abgeschlossen",
                    (
                        f"RL-Training mit {episodes_target} Episoden abgeschlossen.\n"
                        f"Erfolge: {successes}/{episodes_done} ({success_rate:.1f}%)\n"
                        f"Durchschnittliche Schritte: {avg_steps:.1f}"
                    ),
                )

        elif mode in {"testing_supervised", "testing_rl"}:
            successes = int(payload.get("successes", 0))
            episodes_done = int(payload.get("episodes_done", 0))
            success_rate = float(payload.get("success_rate", 0.0))
            avg_steps = float(payload.get("avg_steps", 0.0))

            label_prefix = "Supervised" if mode == "testing_supervised" else "RL"

            self.result_label.setText(
                f"{label_prefix} Erfolg: {successes}/{episodes_done} "
                f"({success_rate:.1f}%), avg Steps: {avg_steps:.1f}"
            )

            if not stopped:
                QMessageBox.information(
                    self,
                    "Testing abgeschlossen",
                    (
                        f"{label_prefix} Erfolgsrate: {successes}/{episodes_done} "
                        f"({success_rate:.1f}%)\n"
                        f"Durchschnittliche Schritte: {avg_steps:.1f}"
                    ),
                )

        self.finish_run()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()