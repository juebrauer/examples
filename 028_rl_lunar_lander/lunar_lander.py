"""Minimal PySide6 lunar lander demo with a small RL-friendly API.

What this demo does:
- Simulates a 2D lunar lander above procedurally generated terrain with a flat
    landing pad.
- Lets you fly the vehicle manually with the keyboard, run a random agent, or
    train a small DQN agent online inside the GUI.
- Visualizes telemetry, reward shaping, landing outcomes, and learning progress.

Observation state / observation space:
- The agent receives a continuous 8-dimensional observation tuple of floats.
- The state variables are:
    1. vel_x: horizontal velocity
    2. vel_y: vertical velocity
    3. angle_rad: current lander angle in radians, normalized to [-pi, pi]
    4. ang_vel: angular velocity
    5. pad_center_dx: horizontal offset to the landing pad center
    6. pad_center_dy: vertical offset to the landing pad surface
    7. altitude: height above the local ground directly below the lander
    8. in_landing_pad: 1.0 if the lander is horizontally above the pad, else 0.0
- In other words, the observation space is continuous, not discrete.

Action state / action space:
- The action space is discrete with 8 possible actions.
- Internally, each action is a 3-bit mask that controls three thrusters:
    main thruster, left thruster, and right thruster.
- This yields all combinations from 0 to 7:
    no thrust, only main, only left, only right, or any combination of them.

Rewards used in the demo:
- Optional landing-tube shaping reward: reward per step while the lander is
    horizontally inside the pad region.
- Optional distance shaping reward: reward per step when the distance to the pad
    center decreases.
- Optional stability shaping reward: reward per step when the lander is slow,
    level, and not ascending.
- Optional energy-usage shaping reward: usually a penalty per fired thruster.
- Terminal reward for successful landing.
- Terminal reward for crashing.
- If a terminal reward is enabled and a terminal event happens, that terminal
    reward overrides the shaping reward for that step.

Hyperparameters and settings you can adjust in the demo UI:
- Controller mode: Human, Random, or DQN.
- DQN tricks: enable or disable the target network and replay buffer.
- Reward shaping toggles and magnitudes:
    landing tube, distance reward, stability reward, energy usage reward.
- Terminal reward toggles and magnitudes for landing success and crash.
- Learning run limit: stop automatically after a chosen number of learning steps.
- Rendering on/off, which trades visualization for faster training.
- Sound on/off.

Important note:
- The DQN implementation also has additional internal hyperparameters such as
    hidden layer sizes, learning rate, gamma, batch size, replay size, warmup
    steps, train frequency, and epsilon schedule. These exist in code, but in the
    current version of the demo they are fixed defaults and are not exposed as GUI
    controls.

Manual controls:
- Left Arrow: fire left thruster
- Right Arrow: fire right thruster
- Up Arrow: fire bottom thruster

Goal: land softly on the landing pad.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, List, Protocol, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from PySide6 import QtMultimedia
except Exception:  # pragma: no cover
    QtMultimedia = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    FigureCanvas = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __iadd__(self, other: "Vec2") -> "Vec2":
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other: "Vec2") -> "Vec2":
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, scalar: float) -> "Vec2":
        self.x *= scalar
        self.y *= scalar
        return self

    def iadd(self, other: "Vec2") -> None:
        self.x += other.x
        self.y += other.y

    def isub(self, other: "Vec2") -> None:
        self.x -= other.x
        self.y -= other.y

    def imul(self, scalar: float) -> None:
        self.x *= scalar
        self.y *= scalar

    def to_qpointf(self) -> QtCore.QPointF:
        return QtCore.QPointF(self.x, self.y)

    @staticmethod
    def from_qpointf(p: QtCore.QPointF) -> "Vec2":
        return Vec2(p.x(), p.y())


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class Terrain:
    points: List[Tuple[float, float]]  # (x, y)
    width: float

    def height_at(self, x: float) -> float:
        if not self.points:
            return 0.0
        x = clamp(x, 0.0, self.width)
        step = self.points[1][0] - self.points[0][0] if len(self.points) > 1 else self.width
        idx = int(x // step)
        idx = max(0, min(idx, len(self.points) - 2))
        x0, y0 = self.points[idx]
        x1, y1 = self.points[idx + 1]
        if x1 == x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return lerp(y0, y1, t)

    @staticmethod
    def generate(width: float, base_y: float, amplitude: float, step: float, seed: int | None = None) -> "Terrain":
        rng = random.Random(seed)
        n = int(width // step) + 1
        raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]

        smooth = raw[:]
        for _ in range(5):
            nxt = smooth[:]
            for i in range(1, n - 1):
                nxt[i] = (smooth[i - 1] + 2.0 * smooth[i] + smooth[i + 1]) / 4.0
            smooth = nxt

        points: List[Tuple[float, float]] = []
        for i in range(n):
            x = i * step
            y = base_y + amplitude * smooth[i]
            points.append((x, y))

        return Terrain(points=points, width=width)


@dataclass(frozen=True)
class LandingPad:
    x0: float
    x1: float
    y: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Lander:
    pos: Vec2
    vel: Vec2
    angle: float
    ang_vel: float

    radius: float = 14.0
    mass: float = 1.0
    inertia: float = 1.0


# -----------------------------
# RL-friendly API (Env + Agent)
# -----------------------------
#
# We keep the simulation deterministic and independent from rendering.
# A GUI widget can run a standard RL loop:
#   obs = env.reset()
#   while True:
#       action = agent.take_action(obs, info)
#       obs, reward, terminated, info = env.step(action, dt)
#       if terminated: break

ACTION_MAIN = 1 << 0
ACTION_LEFT = 1 << 1
ACTION_RIGHT = 1 << 2
ACTION_SPACE_N = 8  # 3-bit action mask => 0..7


def _action_to_burns(action: int) -> tuple[float, float, float]:
    a = int(action) & 0b111
    burn_main = 1.0 if (a & ACTION_MAIN) else 0.0
    burn_left = 1.0 if (a & ACTION_LEFT) else 0.0
    burn_right = 1.0 if (a & ACTION_RIGHT) else 0.0
    return burn_main, burn_left, burn_right


Observation = tuple[float, ...]


class Agent(Protocol):
    def reset(self) -> None: ...

    def take_action(self, observation: Observation, info: dict) -> int: ...


class HumanKeyboardAgent:
    """Keyboard-controlled agent.

    This makes the current "arrow keys" control a special case of the agent interface.
    """

    def __init__(self) -> None:
        self._left = False
        self._right = False
        self._main = False

    def reset(self) -> None:
        self._left = False
        self._right = False
        self._main = False

    def handle_key_press(self, key: int) -> None:
        if key == QtCore.Qt.Key.Key_Left:
            self._left = True
        elif key == QtCore.Qt.Key.Key_Right:
            self._right = True
        elif key == QtCore.Qt.Key.Key_Up:
            self._main = True

    def handle_key_release(self, key: int) -> None:
        if key == QtCore.Qt.Key.Key_Left:
            self._left = False
        elif key == QtCore.Qt.Key.Key_Right:
            self._right = False
        elif key == QtCore.Qt.Key.Key_Up:
            self._main = False

    def take_action(self, observation: Observation, info: dict) -> int:
        _ = observation, info
        action = 0
        if self._main:
            action |= ACTION_MAIN
        if self._left:
            action |= ACTION_LEFT
        if self._right:
            action |= ACTION_RIGHT
        return action


class RandomAgent:
    """Tiny example RL agent: samples uniformly from the discrete action space."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return

    def take_action(self, observation: Observation, info: dict) -> int:
        _ = observation, info
        return self._rng.randrange(ACTION_SPACE_N)


class DQNAgent:
    """Deep Q-Network (DQN) agent for this environment.

    - Discrete action space (0..7)
    - MLP approximator + replay buffer + target network
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_space_n: int = ACTION_SPACE_N,
        hidden_sizes: tuple[int, int] = (64, 32),
        lr: float = 1e-3,
        gamma: float = 0.995,
        batch_size: int = 64,
        replay_size: int = 50_000,
        warmup_steps: int = 2_000,
        train_every: int = 1,
        target_update_every: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        use_target_network: bool = True,
        use_replay_buffer: bool = True,
        seed: int | None = 0,
        device: str | None = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for DQNAgent. Install torch or use the Random agent.")

        self.obs_dim = int(obs_dim)
        self.action_space_n = int(action_space_n)
        self.hidden_sizes = (int(hidden_sizes[0]), int(hidden_sizes[1]))
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.warmup_steps = int(warmup_steps)
        self.train_every = int(train_every)
        self.target_update_every = int(target_update_every)

        self.use_target_network = bool(use_target_network)
        self.use_replay_buffer = bool(use_replay_buffer)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

        if seed is not None:
            random.seed(int(seed))
            torch.manual_seed(int(seed))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        h1, h2 = self.hidden_sizes

        class _MLP(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.fc1 = nn.Linear(in_dim, int(h1))
                self.fc2 = nn.Linear(int(h1), int(h2))
                self.fc3 = nn.Linear(int(h2), out_dim)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        self.policy_net = _MLP(self.obs_dim, self.action_space_n).to(self.device)
        self.target_net = _MLP(self.obs_dim, self.action_space_n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=float(self.lr))

        # (s, a, r, s2, done)
        self.replay: deque[tuple[Observation, int, float, Observation, bool]] = deque(maxlen=int(self.replay_size))
        self._step_count = 0

        self._prev_obs: Observation | None = None
        self._prev_action: int | None = None

    def reset(self) -> None:
        self._prev_obs = None
        self._prev_action = None

    def set_use_target_network(self, enabled: bool) -> None:
        self.use_target_network = bool(enabled)
        if self.use_target_network:
            # Make the target network sane when (re-)enabled.
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_use_replay_buffer(self, enabled: bool) -> None:
        self.use_replay_buffer = bool(enabled)
        if not self.use_replay_buffer:
            self.replay.clear()

    def _epsilon(self) -> float:
        t = min(1.0, self._step_count / max(1, self.epsilon_decay_steps))
        return (1.0 - t) * self.epsilon_start + t * self.epsilon_end

    def _normalize_obs(self, obs: Observation) -> list[float]:
        # Scale features to roughly [-1, 1] ranges.
        # Order is LunarLanderEnv.state_names.
        scales = [
            250.0,  # vel_x
            250.0,  # vel_y
            math.pi,  # angle_rad
            1.5,  # ang_vel
            450.0,  # pad_center_dx
            520.0,  # pad_center_dy
            520.0,  # altitude
            1.0,  # in_landing_pad
        ]
        out: list[float] = []
        for i, v in enumerate(obs):
            s = scales[i] if i < len(scales) else 1.0
            out.append(clamp(float(v) / s, -1.0, 1.0))
        return out

    def _obs_tensor(self, obs: Observation):
        x = torch.tensor(self._normalize_obs(obs), dtype=torch.float32, device=self.device)
        return x

    def take_action(self, observation: Observation, info: dict) -> int:
        _ = info
        self._step_count += 1
        eps = self._epsilon()
        if random.random() < eps:
            action = random.randrange(self.action_space_n)
        else:
            with torch.no_grad():
                q = self.policy_net(self._obs_tensor(observation).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())

        self._prev_obs = observation
        self._prev_action = int(action)
        return int(action)

    def observe(self, reward: float, terminated: bool, next_observation: Observation, info: dict) -> None:
        _ = info
        if self._prev_obs is None or self._prev_action is None:
            return

        transition = (self._prev_obs, int(self._prev_action), float(reward), next_observation, bool(terminated))

        if self.use_replay_buffer:
            self.replay.append(transition)
        
        if terminated:
            self._prev_obs = None
            self._prev_action = None

        if self.train_every > 1 and (self._step_count % self.train_every) != 0:
            return

        if self.use_replay_buffer:
            if len(self.replay) < max(self.warmup_steps, self.batch_size):
                return
            self._train_step_from_replay()
        else:
            # Online update (no replay buffer): learn from the most recent transition.
            self._train_step_on_batch([transition])

        if self.use_target_network and (self._step_count % max(1, self.target_update_every)) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step_from_replay(self) -> None:
        if torch is None:
            return

        batch = random.sample(self.replay, k=self.batch_size)

        self._train_step_on_batch(batch)

    def _train_step_on_batch(self, batch: list[tuple[Observation, int, float, Observation, bool]]) -> None:
        if torch is None:
            return

        states = torch.stack([self._obs_tensor(s) for (s, _a, _r, _s2, _d) in batch], dim=0)
        actions = torch.tensor([a for (_s, a, _r, _s2, _d) in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([r for (_s, _a, r, _s2, _d) in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([self._obs_tensor(s2) for (_s, _a, _r, s2, _d) in batch], dim=0)
        dones = torch.tensor([1.0 if d else 0.0 for (_s, _a, _r, _s2, d) in batch], dtype=torch.float32, device=self.device)

        q_sa = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_net = self.target_net if self.use_target_network else self.policy_net
            next_q = next_net(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * (self.gamma * next_q)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optim.step()

    # -----------------
    # Checkpointing
    # -----------------
    # The UI uses these helpers to save/restore a learned agent.
    # We intentionally keep checkpoints small by default (no replay buffer).

    def get_checkpoint(self, *, include_replay: bool = False) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required for DQNAgent checkpointing")

        ckpt: dict[str, Any] = {
            "version": 1,
            "agent": "DQNAgent",
            "obs_dim": int(self.obs_dim),
            "action_space_n": int(self.action_space_n),
            "hidden_sizes": tuple(int(x) for x in self.hidden_sizes),
            "lr": float(self.lr),
            "gamma": float(self.gamma),
            "batch_size": int(self.batch_size),
            "replay_size": int(self.replay_size),
            "warmup_steps": int(self.warmup_steps),
            "train_every": int(self.train_every),
            "target_update_every": int(self.target_update_every),
            "use_target_network": bool(self.use_target_network),
            "use_replay_buffer": bool(self.use_replay_buffer),
            "epsilon_start": float(self.epsilon_start),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay_steps": int(self.epsilon_decay_steps),
            "step_count": int(self._step_count),
            "device": str(self.device),
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
        }

        if include_replay:
            ckpt["replay"] = list(self.replay)

        return ckpt

    @staticmethod
    def _validate_checkpoint(ckpt: dict[str, Any]) -> None:
        if ckpt.get("agent") != "DQNAgent":
            raise ValueError("Checkpoint is not a DQNAgent")
        if int(ckpt.get("version", 0)) != 1:
            raise ValueError("Unsupported checkpoint version")

    def save(self, path: str | Path, *, include_replay: bool = False) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for DQNAgent checkpointing")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_checkpoint(include_replay=include_replay), str(p))

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any], *, device: str | None = None) -> "DQNAgent":
        if torch is None:
            raise RuntimeError("PyTorch is required for DQNAgent checkpointing")

        cls._validate_checkpoint(ckpt)

        if device is None:
            # Prefer checkpoint's saved device, but fall back to auto.
            device = ckpt.get("device")

        agent = cls(
            obs_dim=int(ckpt["obs_dim"]),
            action_space_n=int(ckpt["action_space_n"]),
            hidden_sizes=tuple(ckpt["hidden_sizes"]),
            lr=float(ckpt["lr"]),
            gamma=float(ckpt["gamma"]),
            batch_size=int(ckpt["batch_size"]),
            replay_size=int(ckpt.get("replay_size", 50_000)),
            warmup_steps=int(ckpt.get("warmup_steps", 2_000)),
            train_every=int(ckpt.get("train_every", 1)),
            target_update_every=int(ckpt.get("target_update_every", 1_000)),
            use_target_network=bool(ckpt.get("use_target_network", True)),
            use_replay_buffer=bool(ckpt.get("use_replay_buffer", True)),
            epsilon_start=float(ckpt.get("epsilon_start", 1.0)),
            epsilon_end=float(ckpt.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(ckpt.get("epsilon_decay_steps", 50_000)),
            seed=None,
            device=device,
        )

        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
        agent.optim.load_state_dict(ckpt["optim_state_dict"])
        agent._step_count = int(ckpt.get("step_count", 0))

        replay = ckpt.get("replay")
        if isinstance(replay, list):
            agent.replay.clear()
            for item in replay:
                if isinstance(item, tuple) and len(item) == 5:
                    agent.replay.append(item)  # type: ignore[arg-type]

        # Ensure everything is on the agent device.
        agent.policy_net.to(agent.device)
        agent.target_net.to(agent.device)
        agent.target_net.eval()

        return agent

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "DQNAgent":
        if torch is None:
            raise RuntimeError("PyTorch is required for DQNAgent checkpointing")

        ckpt = torch.load(str(path), map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError("Invalid checkpoint format")
        return cls.from_checkpoint(ckpt, device=device)


class LunarLanderEnv:
    """Minimal lunar lander environment.

    - `action` is a 3-bit mask: MAIN/LEFT/RIGHT (0..7)
    - `observation` is a flat tuple of floats (see `state_names`)

    This is intentionally small and Gym-like (reset/step) so adding a Q-learning
    agent later is straightforward.
    """

    action_space_n: int = ACTION_SPACE_N
    # Observation space used by learning agents.
    # Kept intentionally small and purely relative where possible.
    state_names: tuple[str, ...] = (
        "vel_x",
        "vel_y",
        "angle_rad",
        "ang_vel",
        "pad_center_dx",
        "pad_center_dy",
        "altitude",
        "in_landing_pad",
    )

    def __init__(self, *, world_w: float = 900.0, world_h: float = 520.0) -> None:
        self.world_w = float(world_w)
        self.world_h = float(world_h)

        # Global slow-motion factor for easier manual control.
        # 1.0 = real-time, 0.5 = half-speed.
        self.time_scale = 0.55

        # Tune these for "hand-flyable" dynamics in class.
        self.gravity = Vec2(0.0, -140.0)
        self.main_thrust = 650.0
        self.side_thrust = 180.0
        self.side_torque = 420.0

        # Damps angular velocity as: w *= exp(-angular_damping * dt)
        self.angular_damping = 3.2
        self.max_ang_vel = 1.2

        self.max_main_burn = 1.0
        self.max_side_burn = 1.0

        self.success_max_abs_vy = 70.0
        self.success_max_abs_vx = 60.0
        self.success_max_abs_angle = math.radians(15.0)

        self.terrain = Terrain(points=[], width=self.world_w)
        self.landing_pad = LandingPad(0.0, 0.0, 0.0)
        self.lander = self._spawn_lander()

        self.status_text: str | None = None
        self.frozen = False
        self.last_contact: dict | None = None

        self.last_action: int = 0
        self._just_terminated_success: bool | None = None

        # Latest reward breakdown for visualization/debugging.
        self.last_reward_total = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
        self.last_reward_tube = 0.0
        self.last_reward_energy = 0.0
        self.last_reward_terminal = 0.0
        self.last_distance_metric = 0.0
        self.last_stability_metric = 0.0

        # -----------------
        # Rewards (simplified for teaching)
        # -----------------
        # Reward signals can be toggled from the UI. When enabled, each signal
        # contributes its configured magnitude while its condition is satisfied.
        # Terminal reward: +100 if LANDED, otherwise 0.
        self.tube_shaping_enabled = False

        # Shaping reward magnitudes (editable from UI).
        self.tube_shaping_reward = 2.0

        # +1 if the lander reduced its distance to the pad center this step.
        self.distance_shaping_enabled = False

        self.distance_shaping_reward = 1.0

        # +1 if the lander is in a "stable" configuration (slow + level + not ascending).
        self.stability_shaping_enabled = False

        self.stability_shaping_reward = 1.0

        # Energy usage shaping: per-thruster penalty/reward applied each step.
        # Default: -1 for each thruster used (main/left/right).
        self.energy_usage_shaping_enabled = False
        self.energy_usage_reward_per_throttle = -0.05

        # Terminal reward magnitude.
        self.terminal_reward_success = 100.0
        self.terminal_reward_crash = -50.0

        # Terminal reward toggles (editable from UI).
        self.terminal_reward_success_enabled = True
        self.terminal_reward_crash_enabled = True

        self.reset()

    def set_tube_shaping(self, enabled: bool) -> None:
        self.tube_shaping_enabled = bool(enabled)

    def set_tube_shaping_reward(self, reward: float) -> None:
        try:
            self.tube_shaping_reward = float(reward)
        except Exception:
            return

    def set_distance_shaping(self, enabled: bool) -> None:
        self.distance_shaping_enabled = bool(enabled)

    def set_distance_shaping_reward(self, reward: float) -> None:
        try:
            self.distance_shaping_reward = float(reward)
        except Exception:
            return

    def set_stability_shaping(self, enabled: bool) -> None:
        self.stability_shaping_enabled = bool(enabled)

    def set_stability_shaping_reward(self, reward: float) -> None:
        try:
            self.stability_shaping_reward = float(reward)
        except Exception:
            return

    def set_energy_usage_shaping(self, enabled: bool) -> None:
        self.energy_usage_shaping_enabled = bool(enabled)

    def set_energy_usage_reward_per_throttle(self, reward: float) -> None:
        try:
            self.energy_usage_reward_per_throttle = float(reward)
        except Exception:
            return

    def set_terminal_reward_success(self, reward: float) -> None:
        try:
            self.terminal_reward_success = float(reward)
        except Exception:
            return

    def set_terminal_reward_success_enabled(self, enabled: bool) -> None:
        self.terminal_reward_success_enabled = bool(enabled)

    def set_terminal_reward_crash(self, reward: float) -> None:
        try:
            self.terminal_reward_crash = float(reward)
        except Exception:
            return

    def set_terminal_reward_crash_enabled(self, enabled: bool) -> None:
        self.terminal_reward_crash_enabled = bool(enabled)

    def _is_stable_state(self) -> bool:
        """Return True when the lander is slow + level (and not ascending).

        Uses the same thresholds as the successful landing check, but does NOT
        require being over the landing pad. This is used for stability shaping.
        """

        lander = self.lander
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        angle_ok = abs(ang) <= float(self.success_max_abs_angle)
        vx_ok = abs(lander.vel.x) <= float(self.success_max_abs_vx)
        vy_ok = abs(lander.vel.y) <= float(self.success_max_abs_vy)
        descending_or_hovering = lander.vel.y <= 0.0
        return bool(angle_ok and vx_ok and vy_ok and descending_or_hovering)

    def _is_stable_for_landing(self) -> bool:
        """Stable state + over the landing pad."""

        lander = self.lander
        in_pad = self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1
        return bool(in_pad and self._is_stable_state())

    def reset(self, *, seed: int | None = None) -> Observation:
        rng = random.Random(seed)
        if seed is None:
            seed = rng.randrange(0, 10_000_000)
            rng = random.Random(seed)

        self.terrain = Terrain.generate(
            width=self.world_w,
            base_y=95.0,
            amplitude=50.0,
            step=18.0,
            seed=seed,
        )

        pad_width = 140.0
        pad_margin = 40.0
        pad_x0 = rng.uniform(pad_margin, self.world_w - pad_margin - pad_width)
        pad_x1 = pad_x0 + pad_width

        pad_y = self.terrain.height_at(0.5 * (pad_x0 + pad_x1))

        flattened: List[Tuple[float, float]] = []
        for x, y in self.terrain.points:
            if pad_x0 <= x <= pad_x1:
                flattened.append((x, pad_y))
            else:
                flattened.append((x, y))
        self.terrain = Terrain(points=flattened, width=self.world_w)
        self.landing_pad = LandingPad(x0=pad_x0, x1=pad_x1, y=pad_y)

        self.lander = self._spawn_lander()
        self.status_text = None
        self.frozen = False
        self.last_contact = None
        self.last_action = 0
        self._just_terminated_success = None

        self.last_reward_total = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
        self.last_reward_tube = 0.0
        self.last_reward_energy = 0.0
        self.last_reward_terminal = 0.0
        self.last_distance_metric = 0.0
        self.last_stability_metric = 0.0

        return self.observation()

    def observation(self) -> Observation:
        lander = self.lander
        ground = self.terrain.height_at(lander.pos.x)
        altitude = max(0.0, lander.pos.y - (ground + lander.radius))
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        pad_center_dx = lander.pos.x - self.landing_pad.cx
        # Relative vertical position to the pad surface (0 when touching pad at its y-level).
        pad_center_dy = lander.pos.y - (self.landing_pad.y + lander.radius)
        in_pad = 1.0 if (self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1) else 0.0
        return (
            lander.vel.x,
            lander.vel.y,
            ang,
            lander.ang_vel,
            pad_center_dx,
            pad_center_dy,
            altitude,
            in_pad,
        )

    def info(self) -> dict:
        lander = self.lander
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        ground = self.terrain.height_at(lander.pos.x)
        altitude = max(0.0, lander.pos.y - (ground + lander.radius))
        pad_center_dx = lander.pos.x - self.landing_pad.cx
        pad_center_dy = lander.pos.y - (self.landing_pad.y + lander.radius)
        in_pad = self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1
        stable_state = self._is_stable_state()
        stable_for_landing = bool(in_pad and stable_state)
        payload = {
            "pos_x": lander.pos.x,
            "pos_y": lander.pos.y,
            "vel_x": lander.vel.x,
            "vel_y": lander.vel.y,
            "angle_rad": ang,
            "angle_deg": math.degrees(ang),
            "ang_vel": lander.ang_vel,
            "altitude": altitude,
            "pad_center_dx": pad_center_dx,
            "pad_center_dy": pad_center_dy,
            "status": self.status_text or "FLYING",
            "in_landing_pad": in_pad,
            "action": int(self.last_action) & 0b111,
            "terminated": bool(self.frozen),
            "reward_total": float(self.last_reward_total),
            "reward_distance": float(self.last_reward_distance),
            "reward_stability": float(self.last_reward_stability),
            "reward_tube": float(self.last_reward_tube),
            "reward_energy": float(self.last_reward_energy),
            "reward_terminal": float(self.last_reward_terminal),
            "distance_metric": float(self.last_distance_metric),
            "stability_metric": float(self.last_stability_metric),
            "tube_shaping_enabled": bool(self.tube_shaping_enabled),
            "distance_shaping_enabled": bool(self.distance_shaping_enabled),
            "stability_shaping_enabled": bool(self.stability_shaping_enabled),
            "energy_usage_shaping_enabled": bool(self.energy_usage_shaping_enabled),
            "in_reward_tube": bool(self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1),
            "stable_state": bool(stable_state),
            "stable_for_landing": bool(stable_for_landing),
        }
        if self.last_contact is not None:
            payload["last_contact"] = self.last_contact
        if self._just_terminated_success is not None:
            payload["terminal_success"] = bool(self._just_terminated_success)
        return payload

    def step(self, action: int, dt: float, *, include_info: bool = True) -> tuple[Observation, float, bool, dict]:
        self._just_terminated_success = None
        self.last_action = int(action) & 0b111

        # Default reward breakdown (overwritten below).
        self.last_reward_total = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
        self.last_reward_tube = 0.0
        self.last_reward_energy = 0.0
        self.last_reward_terminal = 0.0
        self.last_distance_metric = 0.0
        self.last_stability_metric = 0.0

        if self.frozen:
            obs = self.observation()
            info = self.info() if include_info else {}
            return obs, 0.0, True, info

        dt = clamp(float(dt), 0.0, 1.0 / 20.0)
        dt *= self.time_scale

        reward_distance = 0.0
        reward_stability = 0.0
        reward_tube = 0.0
        reward_energy = 0.0

        distance_metric = 0.0
        stability_metric = 0.0

        # Pre-step distance to pad center (on the pad surface). Used for the
        # "distance reduction" reward.
        d0 = 0.0
        if self.distance_shaping_enabled:
            lander0 = self.lander
            pad0 = self.landing_pad
            dx0 = lander0.pos.x - pad0.cx
            dy0 = lander0.pos.y - (pad0.y + lander0.radius)
            d0 = math.hypot(dx0, dy0)

        was_frozen = self.frozen
        self._step_physics(dt, self.last_action)

        terminated = bool(self.frozen)

        # Landing tube: configured reward per step while within the pad edges.
        if (not terminated) and self.tube_shaping_enabled:
            lander1 = self.lander
            in_tube1 = self.landing_pad.x0 <= lander1.pos.x <= self.landing_pad.x1
            reward_tube = float(self.tube_shaping_reward) if in_tube1 else 0.0

        # Distance reduction: configured reward if the distance to the pad center decreased.
        if (not terminated) and self.distance_shaping_enabled:
            lander1 = self.lander
            pad1 = self.landing_pad
            dx1 = lander1.pos.x - pad1.cx
            dy1 = lander1.pos.y - (pad1.y + lander1.radius)
            d1 = math.hypot(dx1, dy1)
            distance_metric = float(d1)
            reward_distance = float(self.distance_shaping_reward) if (d1 + 1e-9) < float(d0) else 0.0

        # Stability: configured reward when slow + level + not ascending (anywhere).
        if (not terminated) and self.stability_shaping_enabled:
            stable = self._is_stable_state()
            stability_metric = 1.0 if stable else 0.0
            reward_stability = float(self.stability_shaping_reward) if stable else 0.0

        # Energy usage: per-throttle penalty/reward for firing thrusters this step.
        if (not terminated) and self.energy_usage_shaping_enabled:
            burn_main, burn_left, burn_right = _action_to_burns(self.last_action)
            throttles_used = int(burn_main > 0.0) + int(burn_left > 0.0) + int(burn_right > 0.0)
            reward_energy = float(self.energy_usage_reward_per_throttle) * float(throttles_used)

        reward = reward_distance + reward_stability + reward_tube + reward_energy

        terminal_success: bool | None = None
        terminal_reward = 0.0
        # Terminal reward: +100 if LANDED, otherwise -50.
        if (not was_frozen) and terminated:
            if self.status_text == "LANDED":
                terminal_reward = float(self.terminal_reward_success) if self.terminal_reward_success_enabled else 0.0
                terminal_success = True
            else:
                terminal_reward = float(self.terminal_reward_crash) if self.terminal_reward_crash_enabled else 0.0
                terminal_success = False
            self._just_terminated_success = terminal_success

            # Terminal reward overrides shaping reward only if enabled.
            if abs(terminal_reward) > 1e-12:
                reward = terminal_reward

        self.last_reward_total = float(reward)
        self.last_reward_distance = float(reward_distance)
        self.last_reward_stability = float(reward_stability)
        self.last_reward_tube = float(reward_tube)
        self.last_reward_energy = float(reward_energy)
        self.last_reward_terminal = float(terminal_reward)
        # Expose current (post-step) metrics for UI overlays.
        self.last_distance_metric = float(distance_metric)
        self.last_stability_metric = float(stability_metric)

        obs = self.observation()
        if include_info:
            info = self.info()
            if terminal_success is not None:
                info["terminal_success"] = terminal_success
        else:
            info = {}

        return obs, reward, terminated, info

    def _spawn_lander(self) -> Lander:
        return Lander(
            pos=Vec2(self.world_w * 0.25, self.world_h * 0.78),
            vel=Vec2(0.0, 0.0),
            angle=0.25,
            ang_vel=0.0,
            radius=14.0,
            mass=1.0,
            inertia=0.9,
        )

    def _step_physics(self, dt: float, action: int) -> None:
        lander = self.lander

        force = Vec2(self.gravity.x * lander.mass, self.gravity.y * lander.mass)
        torque = 0.0

        burn_main, burn_left, burn_right = _action_to_burns(action)
        burn_main = self.max_main_burn * burn_main
        burn_left = self.max_side_burn * burn_left
        burn_right = self.max_side_burn * burn_right

        c = math.cos(lander.angle)
        s = math.sin(lander.angle)
        up_dir = Vec2(-s, c)
        right_dir = Vec2(c, s)

        if burn_main > 0.0:
            force += up_dir * (self.main_thrust * burn_main)

        if burn_left > 0.0:
            force += right_dir * (self.side_thrust * burn_left)
            torque -= self.side_torque * burn_left

        if burn_right > 0.0:
            force -= right_dir * (self.side_thrust * burn_right)
            torque += self.side_torque * burn_right

        acc = force / lander.mass
        ang_acc = torque / lander.inertia

        lander.vel += acc * dt
        lander.ang_vel += ang_acc * dt

        lander.ang_vel *= math.exp(-self.angular_damping * dt)
        lander.ang_vel = clamp(lander.ang_vel, -self.max_ang_vel, self.max_ang_vel)

        lander.pos += lander.vel * dt
        lander.angle += lander.ang_vel * dt

        if lander.pos.x < lander.radius:
            lander.pos.x = lander.radius
            lander.vel.x = -0.3 * lander.vel.x
        elif lander.pos.x > self.world_w - lander.radius:
            lander.pos.x = self.world_w - lander.radius
            lander.vel.x = -0.3 * lander.vel.x

        if lander.pos.y > self.world_h - lander.radius:
            lander.pos.y = self.world_h - lander.radius
            lander.vel.y = -0.2 * lander.vel.y

        self._handle_ground_contact()

    def _lander_local_point_to_world(self, lx: float, ly_down: float) -> Vec2:
        """Convert a point from lander-local drawing coords to world coords.

        Local coords:
        - +x is to the right
        - +y is down (as used in QPainter drawing)

        World coords:
        - +x is to the right
        - +y is up
        """
        lander = self.lander
        c = math.cos(lander.angle)
        s = math.sin(lander.angle)
        # offset = right_dir*lx + up_dir*(-ly_down)
        ox = c * lx + s * ly_down
        oy = s * lx - c * ly_down
        return Vec2(lander.pos.x + ox, lander.pos.y + oy)

    def _ground_penetration_for_point(self, p: Vec2) -> float:
        ground_y = self.terrain.height_at(p.x)
        return ground_y - p.y

    def _contact_points_world(self) -> list[Vec2]:
        # Use leg "feet" (line endpoints) and a body-bottom point.
        # These match the points used in the GUI renderer.
        return [
            self._lander_local_point_to_world(-20.0, 18.0),
            self._lander_local_point_to_world(20.0, 18.0),
            self._lander_local_point_to_world(0.0, 10.0),
        ]

    def _handle_ground_contact(self) -> None:
        lander = self.lander
        points = self._contact_points_world()
        penetrations = [self._ground_penetration_for_point(p) for p in points]
        max_pen = max(penetrations) if penetrations else 0.0

        if max_pen <= 0.0:
            return

        vx = lander.vel.x
        vy = lander.vel.y
        speed_x = abs(vx)
        speed_y = abs(vy)
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        angle_ok = abs(ang) < self.success_max_abs_angle

        in_pad = self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1
        vy_ok = speed_y <= self.success_max_abs_vy
        vx_ok = speed_x <= self.success_max_abs_vx
        soft = vy_ok and vx_ok and angle_ok and in_pad

        angle_deg = math.degrees(ang)
        max_angle_deg = math.degrees(self.success_max_abs_angle)

        def pf(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        checks = [
            f"on landing pad: {pf(in_pad)}",
            f"|vy| <= {self.success_max_abs_vy:.1f}: {pf(vy_ok)}   (|vy|={speed_y:.1f})",
            f"|vx| <= {self.success_max_abs_vx:.1f}: {pf(vx_ok)}   (|vx|={speed_x:.1f})",
            f"|angle| <= {max_angle_deg:.1f}°: {pf(angle_ok)}   (|angle|={abs(angle_deg):.1f}°)",
        ]

        self.last_contact = {
            "impact_vx": vx,
            "impact_vy": vy,
            "impact_speed_x": speed_x,
            "impact_speed_y": speed_y,
            "impact_angle_deg": angle_deg,
            "impact_in_pad": in_pad,
            "success": soft,
            "checks": checks,
        }

        # Resolve penetration by shifting the lander up in world space.
        lander.pos.y = lander.pos.y + max_pen

        if soft:
            lander.vel = Vec2(0.0, 0.0)
            lander.ang_vel = 0.0
            lander.angle = 0.0
            self.status_text = "LANDED"
            self.frozen = True

            # After snapping angle, resolve any remaining penetration.
            points2 = self._contact_points_world()
            pens2 = [self._ground_penetration_for_point(p) for p in points2]
            max_pen2 = max(pens2) if pens2 else 0.0
            if max_pen2 > 0.0:
                lander.pos.y = lander.pos.y + max_pen2
        else:
            lander.vel = Vec2(0.0, 0.0)
            lander.ang_vel = 0.0
            self.status_text = "CRASHED"
            self.frozen = True


class LanderWidget(QtWidgets.QWidget):
    telemetryUpdated = QtCore.Signal(dict)
    learningStatsUpdated = QtCore.Signal(dict)
    runningChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(900, 520)

        self.env = LunarLanderEnv(world_w=900.0, world_h=520.0)
        self.agent: Agent = HumanKeyboardAgent()

        # Simulation is paused until the user presses Start.
        self.running = False
        self.stop_at_learning_step: int | None = None

        # When fast-forwarding learning, avoid spamming UI updates.
        self._last_ui_emit_learning_step = 0
        self._ui_emit_every_learning_steps = 250

        # When rendering is disabled, we fast-forward the simulation by stepping
        # multiple times per UI tick.
        self.rendering_enabled = True
        self.fast_forward_steps_per_tick = 250

        # Learning stats
        self.episode_nr = 0
        self.total_learning_steps = 0
        self.successful_landings = 0
        # Wallclock time spent inside the learning loop while a DQNAgent is active.
        # Measured in seconds using perf_counter().
        self.learning_wallclock_s = 0.0
        self._reward_window_sum = 0.0
        self._reward_window_count = 0
        self.avg_reward_every_100: list[float] = []

        self._t_last = time.perf_counter()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / 60))

        self.key_left = False
        self.key_right = False
        self.key_up = False

        self.status_text: str | None = None

        self._sound_enabled = False
        self._sound_available = QtMultimedia is not None
        self._sfx_left = None
        self._sfx_right = None
        self._sfx_main = None
        self._sfx_land_pass = None
        self._sfx_land_fail = None
        if self._sound_available:
            self._init_sounds()

        self.reset()

    def set_agent(self, agent: Agent) -> None:
        self.agent = agent
        try:
            self.agent.reset()
        except Exception:
            pass
        self.reset()

    def set_running(self, running: bool) -> None:
        self.running = bool(running)
        if not self.running:
            self._stop_thruster_sounds()
            # While paused, show thrusters as off.
            self.key_left = False
            self.key_right = False
            self.key_up = False
        self.runningChanged.emit(self.running)

    def set_stop_at_learning_step(self, step_limit: int | None) -> None:
        if step_limit is None:
            self.stop_at_learning_step = None
            return
        try:
            v = int(step_limit)
        except Exception:
            self.stop_at_learning_step = None
            return
        self.stop_at_learning_step = v if v > 0 else None

    def reset_learning_stats(self) -> None:
        self.episode_nr = 0
        self.total_learning_steps = 0
        self.successful_landings = 0
        self.learning_wallclock_s = 0.0
        self._reward_window_sum = 0.0
        self._reward_window_count = 0
        self.avg_reward_every_100 = []
        self._emit_learning_stats(latest_avg=None)

    def get_learning_state(self) -> dict[str, Any]:
        return {
            "episode_nr": int(self.episode_nr),
            "total_learning_steps": int(self.total_learning_steps),
            "successful_landings": int(self.successful_landings),
            "learning_wallclock_s": float(self.learning_wallclock_s),
            "reward_window_sum": float(self._reward_window_sum),
            "reward_window_count": int(self._reward_window_count),
            "avg_reward_every_100": list(self.avg_reward_every_100),
            "last_ui_emit_learning_step": int(self._last_ui_emit_learning_step),
        }

    def set_learning_state(self, state: dict[str, Any]) -> None:
        def geti(key: str, default: int) -> int:
            try:
                return int(state.get(key, default))
            except Exception:
                return int(default)

        def getf(key: str, default: float) -> float:
            try:
                return float(state.get(key, default))
            except Exception:
                return float(default)

        self.episode_nr = max(0, geti("episode_nr", 0))
        self.total_learning_steps = max(0, geti("total_learning_steps", 0))
        self.successful_landings = max(0, geti("successful_landings", 0))
        self.learning_wallclock_s = max(0.0, getf("learning_wallclock_s", 0.0))
        self._reward_window_sum = getf("reward_window_sum", 0.0)
        self._reward_window_count = max(0, geti("reward_window_count", 0))

        curve = state.get("avg_reward_every_100")
        if isinstance(curve, list):
            out: list[float] = []
            for x in curve:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            self.avg_reward_every_100 = out
        else:
            self.avg_reward_every_100 = []

        self._last_ui_emit_learning_step = max(0, geti("last_ui_emit_learning_step", 0))

        latest_avg = self.avg_reward_every_100[-1] if self.avg_reward_every_100 else None
        self._emit_learning_stats(latest_avg=latest_avg)

    def _emit_learning_stats(self, *, latest_avg: float | None) -> None:
        self.learningStatsUpdated.emit(
            {
                "episode": int(self.episode_nr),
                "steps": int(self.total_learning_steps),
                "successes": int(self.successful_landings),
                "learning_wallclock_s": float(self.learning_wallclock_s),
                "latest_avg_reward_100": latest_avg,
                "avg_reward_curve": list(self.avg_reward_every_100),
            }
        )

    def set_rendering_enabled(self, enabled: bool) -> None:
        self.rendering_enabled = bool(enabled)
        if not self.rendering_enabled:
            # Avoid wasting cycles on sound when fast-forwarding.
            self._stop_thruster_sounds()

    def set_sound_enabled(self, enabled: bool) -> None:
        if not self._sound_available:
            self._sound_enabled = False
            return
        self._sound_enabled = bool(enabled)
        if not self._sound_enabled:
            self._stop_thruster_sounds()

    def sound_available(self) -> bool:
        return self._sound_available

    def _sound_dir(self) -> str:
        d = f"{tempfile.gettempdir()}/pyside6_lunar_lander_sfx"
        QtCore.QDir().mkpath(d)
        return d

    def _write_wav(self, path: str, samples: list[int], sample_rate: int = 44100) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))

    def _gen_tone(self, freq_hz: float, duration_s: float, *, sample_rate: int = 44100, amp: float = 0.35) -> list[int]:
        n = max(1, int(duration_s * sample_rate))
        out: list[int] = []
        for i in range(n):
            t = i / sample_rate
            # small fade to avoid clicks
            fade = 1.0
            fade_len = int(0.01 * sample_rate)
            if i < fade_len:
                fade = i / fade_len
            elif n - i <= fade_len:
                fade = (n - i) / fade_len
            v = math.sin(2.0 * math.pi * freq_hz * t)
            s = int(clamp(v * amp * fade, -1.0, 1.0) * 32767)
            out.append(s)
        return out

    def _gen_bass_thrust_loop(self, duration_s: float, *, sample_rate: int = 44100) -> list[int]:
        # Main (bottom) thruster: similar to side thruster character (noise + tone),
        # but bassier and smoother.
        rng = random.Random(987)
        n = max(1, int(duration_s * sample_rate))
        out: list[int] = []
        lp = 0.0
        alpha = 0.035
        for i in range(n):
            t = i / sample_rate
            noise = (rng.random() * 2.0 - 1.0)
            lp = (1.0 - alpha) * lp + alpha * noise

            # Lower fundamentals than side thruster.
            tone = 0.70 * math.sin(2.0 * math.pi * 78.0 * t) + 0.30 * math.sin(2.0 * math.pi * 117.0 * t)
            am = 0.72 + 0.28 * math.sin(2.0 * math.pi * 4.0 * t)

            v = (0.60 * lp + 0.40 * tone) * am
            s = int(clamp(v * 0.38, -1.0, 1.0) * 32767)
            out.append(s)
        return out

    def _gen_side_thrust_loop(self, duration_s: float, *, sample_rate: int = 44100) -> list[int]:
        # Less "beepy" side thruster: airy noise + low mid tone.
        rng = random.Random(321)
        n = max(1, int(duration_s * sample_rate))
        out: list[int] = []
        # Simple 1-pole lowpass on noise for a softer hiss.
        lp = 0.0
        alpha = 0.06
        for i in range(n):
            t = i / sample_rate
            noise = (rng.random() * 2.0 - 1.0)
            lp = (1.0 - alpha) * lp + alpha * noise
            tone = 0.6 * math.sin(2.0 * math.pi * 160.0 * t) + 0.4 * math.sin(2.0 * math.pi * 240.0 * t)
            am = 0.75 + 0.25 * math.sin(2.0 * math.pi * 10.0 * t)
            v = (0.55 * lp + 0.45 * tone) * am
            s = int(clamp(v * 0.30, -1.0, 1.0) * 32767)
            out.append(s)
        return out

    def _gen_fail_buzz(self, duration_s: float, *, sample_rate: int = 44100) -> list[int]:
        # Low "bzz" with a bit of noise.
        rng = random.Random(123)
        n = max(1, int(duration_s * sample_rate))
        out: list[int] = []
        for i in range(n):
            t = i / sample_rate
            v = 0.75 * math.sin(2.0 * math.pi * 95.0 * t) + 0.25 * (rng.random() * 2.0 - 1.0)
            # fade out
            fade = 1.0 - (i / max(1, n - 1))
            s = int(clamp(v * 0.45 * fade, -1.0, 1.0) * 32767)
            out.append(s)
        return out

    def _ensure_sound_files(self) -> dict[str, str]:
        d = self._sound_dir()
        files = {
            "thruster_side": f"{d}/thruster_side.wav",
            "thruster_main": f"{d}/thruster_main.wav",
            "land_pass": f"{d}/land_pass.wav",
            "land_fail": f"{d}/land_fail.wav",
        }

        # Always regenerate this file so code tweaks immediately change the sound.
        self._write_wav(files["thruster_side"], self._gen_side_thrust_loop(0.12))
        # Always regenerate this file so code tweaks immediately change the sound.
        # Slightly longer loop than side thruster.
        self._write_wav(files["thruster_main"], self._gen_bass_thrust_loop(0.18))
        if not QtCore.QFileInfo(files["land_pass"]).exists():
            # short "success" chirp: two quick tones
            s1 = self._gen_tone(523.25, 0.10, amp=0.35)
            s2 = self._gen_tone(659.25, 0.14, amp=0.35)
            self._write_wav(files["land_pass"], s1 + s2)
        if not QtCore.QFileInfo(files["land_fail"]).exists():
            self._write_wav(files["land_fail"], self._gen_fail_buzz(0.35))

        return files

    def _mk_sfx(self, file_path: str, *, volume: float) -> object:
        sfx = QtMultimedia.QSoundEffect(self)
        sfx.setSource(QtCore.QUrl.fromLocalFile(file_path))
        # Keep loopCount at 1.
        # Some Linux multimedia backends log noisy warnings when using infinite loop
        # (seek() on sequential device). We instead retrigger one-shots while a key is held.
        sfx.setLoopCount(1)
        sfx.setVolume(clamp(volume, 0.0, 1.0))
        return sfx

    def _init_sounds(self) -> None:
        files = self._ensure_sound_files()
        # Separate effects for left/right so both can play.
        self._sfx_left = self._mk_sfx(files["thruster_side"], volume=0.35)
        self._sfx_right = self._mk_sfx(files["thruster_side"], volume=0.35)
        self._sfx_main = self._mk_sfx(files["thruster_main"], volume=0.55)
        self._sfx_land_pass = self._mk_sfx(files["land_pass"], volume=0.65)
        self._sfx_land_fail = self._mk_sfx(files["land_fail"], volume=0.70)

    def _stop_thruster_sounds(self) -> None:
        for sfx in [self._sfx_left, self._sfx_right, self._sfx_main]:
            if sfx is not None:
                sfx.stop()

    def _update_thruster_sounds(self) -> None:
        if not self._sound_enabled or self.env.frozen:
            self._stop_thruster_sounds()
            return

        def want(sfx: object | None, play: bool) -> None:
            if sfx is None:
                return
            if play:
                # Retrigger one-shots while the key is held.
                if not sfx.isPlaying():
                    sfx.play()
            else:
                if sfx.isPlaying():
                    sfx.stop()

        want(self._sfx_left, self.key_left)
        want(self._sfx_right, self.key_right)
        want(self._sfx_main, self.key_up)

    def _play_landing_sound(self, success: bool) -> None:
        if not self._sound_enabled:
            return
        self._stop_thruster_sounds()
        sfx = self._sfx_land_pass if success else self._sfx_land_fail
        if sfx is not None:
            sfx.stop()
            sfx.play()

    def reset(self) -> None:
        self.env.reset()
        self.agent.reset()
        self.status_text = None
        self._stop_thruster_sounds()
        self.key_left = False
        self.key_right = False
        self.key_up = False
        self.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _world_to_screen(self, p: Vec2) -> QtCore.QPointF:
        return QtCore.QPointF(p.x, self.height() - p.y)

    def _screen_to_world_y(self, y_screen: float) -> float:
        return self.height() - y_screen

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Keep physical top boundary aligned with the visible top edge.
        self.env.world_h = float(self.height())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(900, 520)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Escape:
            QtWidgets.QApplication.quit()
        else:
            # Forward to the active agent if it supports keyboard control.
            if hasattr(self.agent, "handle_key_press"):
                try:
                    getattr(self.agent, "handle_key_press")(k)
                except Exception:
                    pass
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        k = event.key()
        if hasattr(self.agent, "handle_key_release"):
            try:
                getattr(self.agent, "handle_key_release")(k)
            except Exception:
                pass
        super().keyReleaseEvent(event)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now

        if not self.running:
            # Keep the UI responsive without advancing the simulation.
            self.telemetryUpdated.emit(self.env.info())
            if self.rendering_enabled:
                self.update()
            return

        learning_mode = not isinstance(self.agent, HumanKeyboardAgent)
        dqn_learning = isinstance(self.agent, DQNAgent)

        # Rendered mode: use wall-clock dt. Headless mode: fixed dt and many steps.
        if self.rendering_enabled:
            steps = 1
            dt_step = dt
        else:
            steps = self.fast_forward_steps_per_tick
            dt_step = 1.0 / 60.0

        info_last: dict | None = None
        terminated = False
        latest_avg: float | None = None
        stopped_by_step_limit = False
        # In headless learning mode, avoid constructing large info dicts every step.
        fast_learning = learning_mode and (not self.rendering_enabled)

        t0_learning = time.perf_counter() if dqn_learning else 0.0
        for _i in range(steps):
            if learning_mode and self.stop_at_learning_step is not None:
                if self.total_learning_steps >= int(self.stop_at_learning_step):
                    self.set_running(False)
                    stopped_by_step_limit = True
                    break

            obs = self.env.observation()
            info_for_action = {} if fast_learning else self.env.info()
            action = int(self.agent.take_action(obs, info_for_action))
            # Update GUI-visible thruster state from the agent action.
            self.key_up = bool(action & ACTION_MAIN)
            self.key_left = bool(action & ACTION_LEFT)
            self.key_right = bool(action & ACTION_RIGHT)

            obs2, reward, terminated, info_last = self.env.step(action, dt_step, include_info=not fast_learning)
            if fast_learning:
                # Avoid carrying around empty dicts; build info only when emitting UI.
                info_last = None
            self.status_text = self.env.status_text

            if learning_mode:
                self.total_learning_steps += 1
                self._reward_window_sum += float(reward)
                self._reward_window_count += 1
                if self._reward_window_count >= 100:
                    latest_avg = self._reward_window_sum / max(1, self._reward_window_count)
                    self.avg_reward_every_100.append(float(latest_avg))
                    self._reward_window_sum = 0.0
                    self._reward_window_count = 0

            if hasattr(self.agent, "observe"):
                try:
                    getattr(self.agent, "observe")(reward, terminated, obs2, (info_last or {}))
                except Exception:
                    pass

            if terminated:
                if learning_mode:
                    self.episode_nr += 1
                    if self.env.status_text == "LANDED":
                        self.successful_landings += 1

                    # Auto-reset for learning so we get many episodes.
                    self.env.reset()
                    try:
                        self.agent.reset()
                    except Exception:
                        pass
                    self.key_left = False
                    self.key_right = False
                    self.key_up = False
                    terminated = False
                    continue

                # Human: stop on terminal and wait for Start.
                self.set_running(False)
                break

        if dqn_learning:
            self.learning_wallclock_s += max(0.0, time.perf_counter() - t0_learning)

        # Keep audio state in sync even if keys are held.
        if self.rendering_enabled:
            self._update_thruster_sounds()
        else:
            self._stop_thruster_sounds()

        # Only build telemetry dict when it will actually be used.
        if info_last is None and (self.rendering_enabled or (not fast_learning)):
            info_last = self.env.info()

        # Play landing/collision sound on the transition to terminal state.
        # In fast headless learning mode, info_last may be None by design.
        if isinstance(info_last, dict):
            if info_last.get("terminal_success") is True:
                self._play_landing_sound(True)
            elif info_last.get("terminal_success") is False:
                self._play_landing_sound(False)

        emit_ui = True
        if learning_mode and (not self.rendering_enabled):
            steps_since = self.total_learning_steps - self._last_ui_emit_learning_step
            emit_ui = (latest_avg is not None) or (steps_since >= self._ui_emit_every_learning_steps)
            if stopped_by_step_limit:
                emit_ui = True
            if emit_ui:
                self._last_ui_emit_learning_step = self.total_learning_steps

        if emit_ui:
            if info_last is None:
                info_last = self.env.info()
            self.telemetryUpdated.emit(info_last)
            self._emit_learning_stats(latest_avg=latest_avg)

        if self.rendering_enabled:
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _ = event
        painter = QtGui.QPainter()
        painter.begin(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

            painter.fillRect(self.rect(), QtGui.QColor(10, 10, 14))

            self._draw_stars(painter)
            self._draw_terrain(painter)
            self._draw_lander(painter)
            self._draw_shaping_overlay(painter)
        finally:
            painter.end()

    def _draw_shaping_overlay(self, painter: QtGui.QPainter) -> None:
        if not self.rendering_enabled:
            return

        # Option A: only visualize rewards while a learning agent is active.
        if not isinstance(self.agent, DQNAgent):
            return

        if not (
            self.env.tube_shaping_enabled
            or self.env.distance_shaping_enabled
            or self.env.stability_shaping_enabled
        ):
            return

        info = self.env.info()
        lander = self.env.lander
        pad = self.env.landing_pad
        p_lander = self._world_to_screen(lander.pos)
        p_pad_center = self._world_to_screen(Vec2(pad.cx, pad.y + lander.radius))

        painter.save()

        # Landing tube visualization.
        if self.env.tube_shaping_enabled:
            top_y_world = self._screen_to_world_y(0.0)
            y0_screen = self.height() - top_y_world
            y1_screen = self.height() - pad.y
            tube_rect = QtCore.QRectF(pad.x0, y0_screen, pad.x1 - pad.x0, y1_screen - y0_screen)

            fill = QtGui.QColor(240, 220, 60, 40)
            outline = QtGui.QColor(240, 220, 60, 160)
            painter.setPen(QtGui.QPen(outline, 2))
            painter.setBrush(fill)
            painter.drawRect(tube_rect)

            painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 245), 1))
            p_text = QtCore.QPointF(pad.x0 + 6.0, y0_screen + 18.0)
            painter.drawText(p_text, "landing tube")

            in_tube = pad.x0 <= lander.pos.x <= pad.x1
            if in_tube:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 2))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(tube_rect.adjusted(2, 2, -2, -2))

        # Distance-to-pad visualization and reward label.
        if self.env.distance_shaping_enabled:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(p_lander, p_pad_center)

            mid = QtCore.QPointF(0.5 * (p_lander.x() + p_pad_center.x()), 0.5 * (p_lander.y() + p_pad_center.y()))
            r_dist = float(info.get("reward_distance", 0.0))
            painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 245), 1))
            painter.drawText(mid + QtCore.QPointF(6.0, -6.0), f"dist r: {r_dist:+.1f}")

        # When inside the tube, also show the tube shaping reward.
        if self.env.tube_shaping_enabled and (pad.x0 <= lander.pos.x <= pad.x1):
            r_tube = float(info.get("reward_tube", 0.0))
            painter.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60), 1))
            painter.drawText(p_lander + QtCore.QPointF(18.0, -22.0), f"tube r: {r_tube:+.1f}")

        # If stability shaping is enabled, show its reward near the lander.
        if self.env.stability_shaping_enabled:
            r_stab = float(info.get("reward_stability", 0.0))
            if abs(r_stab) > 1e-12:
                painter.setPen(QtGui.QPen(QtGui.QColor(180, 220, 255), 1))
                painter.drawText(p_lander + QtCore.QPointF(18.0, -36.0), f"stab r: {r_stab:+.1f}")

        painter.restore()

    def _draw_stars(self, painter: QtGui.QPainter) -> None:
        painter.save()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(220, 220, 230))
        rng = random.Random(2)
        for _ in range(90):
            x = rng.randrange(0, self.width())
            y = rng.randrange(0, int(self.height() * 0.7))
            r = 1 if rng.random() < 0.85 else 2
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)
        painter.restore()

    def _draw_terrain(self, painter: QtGui.QPainter) -> None:
        painter.save()
        path = QtGui.QPainterPath()

        x0, y0 = self.env.terrain.points[0]
        p0 = self._world_to_screen(Vec2(x0, y0))
        path.moveTo(p0)
        for x, y in self.env.terrain.points[1:]:
            path.lineTo(self._world_to_screen(Vec2(x, y)))

        path.lineTo(self._world_to_screen(Vec2(self.env.world_w, 0.0)))
        path.lineTo(self._world_to_screen(Vec2(0.0, 0.0)))
        path.closeSubpath()

        painter.setPen(QtGui.QPen(QtGui.QColor(210, 210, 225), 2))
        painter.setBrush(QtGui.QColor(135, 135, 150))
        painter.drawPath(path)

        pad = self.env.landing_pad
        pad_screen_y = self.height() - pad.y
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60), 3))
        painter.drawLine(QtCore.QPointF(pad.x0, pad_screen_y), QtCore.QPointF(pad.x1, pad_screen_y))

        painter.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60), 2))
        painter.drawLine(QtCore.QPointF(pad.x0, pad_screen_y), QtCore.QPointF(pad.x0, pad_screen_y - 22))
        painter.drawLine(QtCore.QPointF(pad.x1, pad_screen_y), QtCore.QPointF(pad.x1, pad_screen_y - 22))
        painter.setBrush(QtGui.QColor(240, 220, 60))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(pad.x0, pad_screen_y - 22),
                    QtCore.QPointF(pad.x0 + 14, pad_screen_y - 18),
                    QtCore.QPointF(pad.x0, pad_screen_y - 14),
                ]
            )
        )
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(pad.x1, pad_screen_y - 22),
                    QtCore.QPointF(pad.x1 - 14, pad_screen_y - 18),
                    QtCore.QPointF(pad.x1, pad_screen_y - 14),
                ]
            )
        )
        painter.restore()

    def _draw_lander(self, painter: QtGui.QPainter) -> None:
        lander = self.env.lander
        center = self._world_to_screen(lander.pos)

        painter.save()
        painter.translate(center)
        painter.rotate(-math.degrees(lander.angle))

        body = QtCore.QRectF(-14, -10, 28, 20)
        painter.setPen(QtGui.QPen(QtGui.QColor(235, 245, 255), 2))
        painter.setBrush(QtGui.QColor(80, 140, 210))
        painter.drawRoundedRect(body, 4, 4)

        painter.setPen(QtGui.QPen(QtGui.QColor(225, 235, 245), 2))
        painter.drawLine(QtCore.QPointF(-12, 10), QtCore.QPointF(-20, 18))
        painter.drawLine(QtCore.QPointF(12, 10), QtCore.QPointF(20, 18))

        painter.setBrush(QtGui.QColor(255, 160, 60))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        if self.key_up and not self.env.frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(-5, 11),
                    QtCore.QPointF(5, 11),
                    QtCore.QPointF(0, 28),
                ]
            )
            painter.drawPolygon(flame)

        if self.key_left and not self.env.frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(-16, -4),
                    QtCore.QPointF(-16, 4),
                    QtCore.QPointF(-30, 0),
                ]
            )
            painter.drawPolygon(flame)

        if self.key_right and not self.env.frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(16, -4),
                    QtCore.QPointF(16, 4),
                    QtCore.QPointF(30, 0),
                ]
            )
            painter.drawPolygon(flame)

        painter.restore()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lunar Lander 2D (PySide6 demo)")

        self._contact_text_last: str | None = None
        self._last_agent_dialog_dir: Path | None = None

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.sim = LanderWidget(root)
        layout.addWidget(self.sim, 1)

        panel = QtWidgets.QWidget(root)
        panel.setFixedWidth(520)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)

        top_title = QtWidgets.QLabel("Controls")
        top_title.setStyleSheet("font-weight: 600;")
        panel_layout.addWidget(top_title)

        columns = QtWidgets.QWidget(panel)
        cols = QtWidgets.QHBoxLayout(columns)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(12)

        left_col = QtWidgets.QWidget(columns)
        left_v = QtWidgets.QVBoxLayout(left_col)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(8)

        right_col = QtWidgets.QWidget(columns)
        right_v = QtWidgets.QVBoxLayout(right_col)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(8)

        cols.addWidget(left_col, 0)
        cols.addWidget(right_col, 0)
        panel_layout.addWidget(columns, 1)

        # --------
        # Left column: input + switches
        # --------
        left_v.addWidget(QtWidgets.QLabel("Arrow Left: left thruster"))
        left_v.addWidget(QtWidgets.QLabel("Arrow Right: right thruster"))
        left_v.addWidget(QtWidgets.QLabel("Arrow Up: bottom thruster"))
        left_v.addSpacing(6)

        controller_title = QtWidgets.QLabel("Controller")
        controller_title.setStyleSheet("font-weight: 600;")
        left_v.addWidget(controller_title)

        self.cmb_controller = QtWidgets.QComboBox()
        self.cmb_controller.addItem("Human (keyboard)")
        self.cmb_controller.addItem("Random")
        self.cmb_controller.addItem("DQN (learning)")
        self.cmb_controller.currentIndexChanged.connect(self._on_controller_changed)
        left_v.addWidget(self.cmb_controller)

        left_v.addSpacing(6)
        tricks_title = QtWidgets.QLabel("DQN learning tricks")
        tricks_title.setStyleSheet("font-weight: 600;")
        left_v.addWidget(tricks_title)

        self.chk_dqn_target = QtWidgets.QCheckBox("Target network")
        self.chk_dqn_target.setToolTip("Use a separate target network for the bootstrap term")
        self.chk_dqn_target.setChecked(True)
        left_v.addWidget(self.chk_dqn_target)

        self.chk_dqn_replay = QtWidgets.QCheckBox("Replay buffer")
        self.chk_dqn_replay.setToolTip("Train from a replay buffer instead of online single-step updates")
        self.chk_dqn_replay.setChecked(True)
        left_v.addWidget(self.chk_dqn_replay)

        dqn_buttons = QtWidgets.QWidget()
        dqn_h = QtWidgets.QHBoxLayout(dqn_buttons)
        dqn_h.setContentsMargins(0, 0, 0, 0)
        dqn_h.setSpacing(8)

        self.btn_save_agent = QtWidgets.QPushButton("Save DQN")
        self.btn_save_agent.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_save_agent.setFixedHeight(26)
        self.btn_save_agent.clicked.connect(self._on_save_dqn)

        self.btn_load_agent = QtWidgets.QPushButton("Load DQN")
        self.btn_load_agent.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_load_agent.setFixedHeight(26)
        self.btn_load_agent.clicked.connect(self._on_load_dqn)

        dqn_h.addWidget(self.btn_save_agent, 1)
        dqn_h.addWidget(self.btn_load_agent, 1)
        left_v.addWidget(dqn_buttons)

        self.btn_reset_agent = QtWidgets.QPushButton("Reset DQN Agent")
        self.btn_reset_agent.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_reset_agent.setFixedHeight(26)
        self.btn_reset_agent.clicked.connect(self._on_reset_dqn)
        left_v.addWidget(self.btn_reset_agent)

        left_v.addSpacing(6)
        shaping_title = QtWidgets.QLabel("Reward shaping")
        shaping_title.setStyleSheet("font-weight: 600;")
        left_v.addWidget(shaping_title)

        def add_shaping_row(
            label: str,
            tooltip: str,
            *,
            on_toggle,
            on_value,
            initial_value: float,
        ) -> tuple[QtWidgets.QCheckBox, QtWidgets.QDoubleSpinBox]:
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)

            chk = QtWidgets.QCheckBox(label)
            chk.setToolTip(tooltip)
            chk.toggled.connect(on_toggle)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(90)
            spin.setValue(float(initial_value))
            spin.setToolTip("Reward magnitude (edit me)")
            spin.valueChanged.connect(on_value)

            h.addWidget(chk, 1)
            h.addWidget(spin, 0)
            left_v.addWidget(row)
            return chk, spin

        self.chk_shape_tube, self.spin_shape_tube = add_shaping_row(
            "Landing tube",
            "Per-step reward while x is within the pad's left/right edges",
            on_toggle=self.sim.env.set_tube_shaping,
            on_value=self.sim.env.set_tube_shaping_reward,
            initial_value=float(self.sim.env.tube_shaping_reward),
        )

        self.chk_shape_distance, self.spin_shape_distance = add_shaping_row(
            "Distance reward",
            "Per-step reward if the distance to the pad center decreased this step",
            on_toggle=self.sim.env.set_distance_shaping,
            on_value=self.sim.env.set_distance_shaping_reward,
            initial_value=float(self.sim.env.distance_shaping_reward),
        )

        self.chk_shape_stability, self.spin_shape_stability = add_shaping_row(
            "Stability reward",
            "Per-step reward when vy<=0 AND |angle|, |vx|, |vy| are within the landing success thresholds (anywhere, not only over pad)",
            on_toggle=self.sim.env.set_stability_shaping,
            on_value=self.sim.env.set_stability_shaping_reward,
            initial_value=float(self.sim.env.stability_shaping_reward),
        )

        self.chk_shape_energy, self.spin_shape_energy = add_shaping_row(
            "Energy usage",
            "Per-step reward: (value) × (number of thrusters fired this step: main/left/right)",
            on_toggle=self.sim.env.set_energy_usage_shaping,
            on_value=self.sim.env.set_energy_usage_reward_per_throttle,
            initial_value=float(self.sim.env.energy_usage_reward_per_throttle),
        )

        left_v.addSpacing(6)
        terminal_title = QtWidgets.QLabel("Terminal rewards")
        terminal_title.setStyleSheet("font-weight: 600;")
        left_v.addWidget(terminal_title)

        def add_terminal_row(
            label: str,
            tooltip: str,
            *,
            on_toggle,
            on_value,
            initial_value: float,
            initial_enabled: bool,
        ) -> tuple[QtWidgets.QCheckBox, QtWidgets.QDoubleSpinBox]:
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)

            chk = QtWidgets.QCheckBox(label)
            chk.setToolTip(tooltip)
            chk.setChecked(bool(initial_enabled))
            chk.toggled.connect(on_toggle)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(2)
            spin.setSingleStep(5.0)
            spin.setKeyboardTracking(False)
            spin.setFixedWidth(90)
            spin.setValue(float(initial_value))
            spin.setToolTip("Reward magnitude (edit me)")
            spin.valueChanged.connect(on_value)

            h.addWidget(chk, 1)
            h.addWidget(spin, 0)
            left_v.addWidget(row)
            return chk, spin

        self.chk_terminal_success, self.spin_terminal_success = add_terminal_row(
            "Landing success",
            "Terminal reward applied when status becomes LANDED (overrides shaping rewards)",
            on_toggle=self.sim.env.set_terminal_reward_success_enabled,
            on_value=self.sim.env.set_terminal_reward_success,
            initial_value=float(self.sim.env.terminal_reward_success),
            initial_enabled=bool(self.sim.env.terminal_reward_success_enabled),
        )

        self.chk_terminal_crash, self.spin_terminal_crash = add_terminal_row(
            "Crash",
            "Terminal reward applied when status becomes CRASHED (overrides shaping rewards)",
            on_toggle=self.sim.env.set_terminal_reward_crash_enabled,
            on_value=self.sim.env.set_terminal_reward_crash,
            initial_value=float(self.sim.env.terminal_reward_crash),
            initial_enabled=bool(self.sim.env.terminal_reward_crash_enabled),
        )

        self.chk_sound = QtWidgets.QCheckBox("Sound")
        if not self.sim.sound_available():
            self.chk_sound.setEnabled(False)
            self.chk_sound.setToolTip("QtMultimedia not available in this environment")
        self.chk_sound.toggled.connect(self.sim.set_sound_enabled)
        left_v.addWidget(self.chk_sound)
        if self.sim.sound_available():
            # Sound on by default.
            self.chk_sound.setChecked(True)

        self.chk_render = QtWidgets.QCheckBox("Rendering")
        self.chk_render.setToolTip("Disable to fast-forward the simulation without drawing")
        self.chk_render.setChecked(True)
        self.chk_render.toggled.connect(self.sim.set_rendering_enabled)
        left_v.addWidget(self.chk_render)

        stop_row = QtWidgets.QWidget()
        stop_h = QtWidgets.QHBoxLayout(stop_row)
        stop_h.setContentsMargins(0, 0, 0, 0)
        stop_h.setSpacing(8)

        self.chk_stop_at_step = QtWidgets.QCheckBox("Stop at step:")
        self.chk_stop_at_step.setToolTip("Auto-stop DQN training after the configured number of learning steps")
        self.chk_stop_at_step.setChecked(True)

        self.edit_stop_at_step = QtWidgets.QLineEdit("100000")
        self.edit_stop_at_step.setFixedWidth(110)
        self.edit_stop_at_step.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.edit_stop_at_step.setValidator(QtGui.QIntValidator(1, 1_000_000_000, self.edit_stop_at_step))
        self.edit_stop_at_step.setToolTip("Maximum number of learning steps for DQN runs")

        stop_h.addWidget(self.chk_stop_at_step, 1)
        stop_h.addWidget(self.edit_stop_at_step, 0)
        left_v.addWidget(stop_row)

        self.btn_start = QtWidgets.QPushButton("Start")
        # Don't steal keyboard focus from the simulation widget.
        self.btn_start.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_start.clicked.connect(self._on_start_stop)
        left_v.addWidget(self.btn_start)

        self.btn_run_experiments = QtWidgets.QPushButton("Run experiments")
        self.btn_run_experiments.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_run_experiments.setToolTip(
            "Run all 20 DQN experiment combinations and write results.md + saved agent checkpoints"
        )
        self.btn_run_experiments.clicked.connect(self._on_run_experiments)
        left_v.addWidget(self.btn_run_experiments)

        left_v.addStretch(1)

        # --------
        # Right column: telemetry + learning
        # --------
        stats_title = QtWidgets.QLabel("Telemetry")
        stats_title.setStyleSheet("font-weight: 600;")
        right_v.addWidget(stats_title)

        self.lbl_status = QtWidgets.QLabel("status: FLYING")
        self.lbl_pos = QtWidgets.QLabel("pos: (0.0, 0.0)")
        self.lbl_vel = QtWidgets.QLabel("vel: (0.0, 0.0)")
        self.lbl_angle = QtWidgets.QLabel("angle: 0.0°")
        self.lbl_alt = QtWidgets.QLabel("altitude: 0.0")
        self.lbl_pad = QtWidgets.QLabel("landing pad: no")
        contact_title = QtWidgets.QLabel("Contact")
        contact_title.setStyleSheet("font-weight: 600;")

        self.txt_contact = QtWidgets.QPlainTextEdit()
        self.txt_contact.setReadOnly(True)
        self.txt_contact.setMinimumHeight(90)
        self.txt_contact.setMaximumHeight(130)

        for lbl in [
            self.lbl_status,
            self.lbl_pos,
            self.lbl_vel,
            self.lbl_angle,
            self.lbl_alt,
            self.lbl_pad,
        ]:
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            right_v.addWidget(lbl)

        right_v.addSpacing(6)
        right_v.addWidget(contact_title)
        right_v.addWidget(self.txt_contact)

        right_v.addSpacing(8)
        learn_title = QtWidgets.QLabel("Learning")
        learn_title.setStyleSheet("font-weight: 600;")
        right_v.addWidget(learn_title)

        self.txt_learning = QtWidgets.QPlainTextEdit()
        self.txt_learning.setReadOnly(True)
        self.txt_learning.setMinimumHeight(60)
        self.txt_learning.setMaximumHeight(95)
        right_v.addWidget(self.txt_learning)

        plot_title = QtWidgets.QLabel("Avg reward / 100 steps")
        plot_title.setStyleSheet("font-weight: 600;")
        right_v.addWidget(plot_title)

        self._fig = None
        self._ax = None
        self._canvas = None
        if FigureCanvas is not None and Figure is not None:
            self._fig = Figure(figsize=(2.2, 1.4), dpi=100)
            self._ax = self._fig.add_subplot(111)
            self._ax.grid(True, alpha=0.25)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(120)
            self._canvas.setMaximumHeight(150)
            right_v.addWidget(self._canvas)
        else:
            right_v.addWidget(QtWidgets.QLabel("Matplotlib not available"))

        right_v.addStretch(1)

        layout.addWidget(panel, 0)
        self.setCentralWidget(root)

        self.sim.telemetryUpdated.connect(self._on_telemetry)
        self.sim.learningStatsUpdated.connect(self._on_learning_stats)
        self.sim.runningChanged.connect(self._on_running_changed)

        # Tube shaping off by default.
        self.chk_shape_tube.setChecked(False)
        self.chk_shape_distance.setChecked(False)
        self.chk_shape_stability.setChecked(False)
        self.chk_shape_energy.setChecked(False)

        # No automatic start.
        self.sim.set_running(False)
        self.sim.reset_learning_stats()

        self._update_agent_io_buttons()

        def apply_tricks() -> None:
            if isinstance(self.sim.agent, DQNAgent):
                try:
                    self.sim.agent.set_use_target_network(self.chk_dqn_target.isChecked())
                    self.sim.agent.set_use_replay_buffer(self.chk_dqn_replay.isChecked())
                except Exception:
                    pass

        self.chk_dqn_target.toggled.connect(lambda _v: apply_tricks())
        self.chk_dqn_replay.toggled.connect(lambda _v: apply_tricks())

    def _update_agent_io_buttons(self) -> None:
        is_dqn_active = isinstance(self.sim.agent, DQNAgent)
        torch_available = torch is not None
        self.btn_save_agent.setEnabled(torch_available and is_dqn_active)
        self.btn_load_agent.setEnabled(torch_available)
        self.btn_reset_agent.setEnabled(torch_available and is_dqn_active)
        if not torch_available:
            tip = "PyTorch not available; DQN save/load disabled"
            self.btn_save_agent.setToolTip(tip)
            self.btn_load_agent.setToolTip(tip)
            self.btn_reset_agent.setToolTip(tip)
        else:
            self.btn_save_agent.setToolTip("Save the current DQN agent checkpoint")
            self.btn_load_agent.setToolTip("Load a previously saved DQN agent checkpoint")
            self.btn_reset_agent.setToolTip("Re-initialize the DQN with fresh random network weights")

    def _get_stop_at_step(self) -> int | None:
        text = self.edit_stop_at_step.text().strip()
        if not text:
            return None
        try:
            value = int(text)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    def _build_dqn_agent(self, *, seed: int | None = 0) -> DQNAgent:
        obs_dim = len(self.sim.env.observation())
        return DQNAgent(
            obs_dim=obs_dim,
            seed=seed,
            use_target_network=self.chk_dqn_target.isChecked(),
            use_replay_buffer=self.chk_dqn_replay.isChecked(),
        )

    def _run_fixed_step_dqn_training(
        self,
        *,
        steps: int,
        process_events: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(self.sim.agent, DQNAgent):
            raise RuntimeError("Active agent is not DQN")

        agent = self.sim.agent
        env = self.sim.env

        self.sim.reset_learning_stats()
        env.reset()
        agent.reset()

        total_steps = 0
        episode_nr = 0
        successful_landings = 0
        reward_window_sum = 0.0
        reward_window_count = 0
        avg_reward_every_100: list[float] = []

        t0 = time.perf_counter()
        dt_step = 1.0 / 60.0
        ui_chunk = 2000
        canceled = False

        if progress_cb is not None:
            progress_cb(0, int(steps))

        while total_steps < int(steps):
            if should_stop is not None and should_stop():
                canceled = True
                break

            obs = env.observation()
            action = int(agent.take_action(obs, {}))
            obs2, reward, terminated, _info = env.step(action, dt_step, include_info=False)
            agent.observe(reward, terminated, obs2, {})

            total_steps += 1
            reward_window_sum += float(reward)
            reward_window_count += 1

            if reward_window_count >= 100:
                avg_reward_every_100.append(reward_window_sum / float(reward_window_count))
                reward_window_sum = 0.0
                reward_window_count = 0

            if terminated:
                episode_nr += 1
                if env.status_text == "LANDED":
                    successful_landings += 1
                env.reset()
                agent.reset()

            if process_events and (total_steps % ui_chunk) == 0:
                if progress_cb is not None:
                    progress_cb(total_steps, int(steps))
                QtWidgets.QApplication.processEvents()

        if progress_cb is not None:
            progress_cb(total_steps, int(steps))

        wallclock_s = max(0.0, time.perf_counter() - t0)

        self.sim.episode_nr = int(episode_nr)
        self.sim.total_learning_steps = int(total_steps)
        self.sim.successful_landings = int(successful_landings)
        self.sim.learning_wallclock_s = float(wallclock_s)
        self.sim._reward_window_sum = float(reward_window_sum)
        self.sim._reward_window_count = int(reward_window_count)
        self.sim.avg_reward_every_100 = list(avg_reward_every_100)
        self.sim._last_ui_emit_learning_step = int(total_steps)

        latest_avg = avg_reward_every_100[-1] if avg_reward_every_100 else None
        self.sim.telemetryUpdated.emit(self.sim.env.info())
        self.sim._emit_learning_stats(latest_avg=latest_avg)

        return {
            "steps": int(total_steps),
            "episodes": int(episode_nr),
            "successes": int(successful_landings),
            "learning_wallclock_s": float(wallclock_s),
            "canceled": bool(canceled),
            "completed": bool((not canceled) and (total_steps >= int(steps))),
        }

    def _set_experiment_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.btn_start,
            self.btn_run_experiments,
            self.btn_reset_agent,
            self.cmb_controller,
            self.chk_dqn_target,
            self.chk_dqn_replay,
            self.chk_stop_at_step,
            self.edit_stop_at_step,
        ]
        for w in widgets:
            w.setEnabled(bool(enabled))

    def _apply_reward_profile(self, name: str) -> None:
        # Profiles for automated experiment sweeps.
        profiles: dict[str, tuple[bool, bool, bool]] = {
            "landing_tube_only": (True, False, False),
            "distance_only": (False, True, False),
            "stability_only": (False, False, True),
            "tube_plus_distance": (True, True, False),
            "tube_plus_distance_plus_stability": (True, True, True),
        }
        tube, distance, stability = profiles[name]
        self.chk_shape_tube.setChecked(tube)
        self.chk_shape_distance.setChecked(distance)
        self.chk_shape_stability.setChecked(stability)
        self.chk_shape_energy.setChecked(False)

    def _format_duration(self, seconds: float) -> str:
        total = max(0, int(seconds))
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _on_controller_changed(self, idx: int) -> None:
        # 0 = Human, 1 = Random, 2 = DQN
        self.sim.set_running(False)
        self.btn_start.setText("Start")
        if idx == 0:
            self.sim.set_agent(HumanKeyboardAgent())
            # Human play should be visual by default.
            self.chk_render.setChecked(True)
            self._update_agent_io_buttons()
            return

        if idx == 1:
            self.sim.set_agent(RandomAgent(seed=0))
            self._update_agent_io_buttons()
            return

        # DQN agent learns online while controlling.
        try:
            self.sim.set_agent(self._build_dqn_agent())
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "DQN initialization failed",
                f"Could not create DQN agent:\n{e}",
            )
            # Revert to human control to avoid mismatch between UI and active agent.
            self.cmb_controller.blockSignals(True)
            self.cmb_controller.setCurrentIndex(0)
            self.cmb_controller.blockSignals(False)
            self.sim.set_agent(HumanKeyboardAgent())
            self._update_agent_io_buttons()
            return

        # Default shaping setup for DQN learning.
        self.chk_shape_tube.setChecked(True)
        self.chk_shape_distance.setChecked(True)
        self.chk_shape_stability.setChecked(True)

        self._update_agent_io_buttons()

        # New learning session for this agent.
        self.sim.reset_learning_stats()

        # Disable UI-heavy stuff by default for learning speed.
        self.chk_render.setChecked(False)
        if self.sim.sound_available():
            self.chk_sound.setChecked(False)

    def _get_reward_config(self) -> dict[str, float | bool]:
        return {
            "tube_enabled": bool(self.chk_shape_tube.isChecked()),
            "tube_reward": float(self.spin_shape_tube.value()),
            "distance_enabled": bool(self.chk_shape_distance.isChecked()),
            "distance_reward": float(self.spin_shape_distance.value()),
            "stability_enabled": bool(self.chk_shape_stability.isChecked()),
            "stability_reward": float(self.spin_shape_stability.value()),
            "energy_enabled": bool(self.chk_shape_energy.isChecked()),
            "energy_reward_per_throttle": float(self.spin_shape_energy.value()),
            "terminal_success_enabled": bool(self.chk_terminal_success.isChecked()),
            "terminal_reward_success": float(self.spin_terminal_success.value()),
            "terminal_crash_enabled": bool(self.chk_terminal_crash.isChecked()),
            "terminal_reward_crash": float(self.spin_terminal_crash.value()),
        }

    def _apply_reward_config(self, cfg: dict[str, object]) -> None:
        env = self.sim.env

        def getb(key: str, default: bool) -> bool:
            v = cfg.get(key, default)
            return bool(v)

        def getf(key: str, default: float) -> float:
            v = cfg.get(key, default)
            try:
                return float(v)  # type: ignore[arg-type]
            except Exception:
                return float(default)

        tube_enabled = getb("tube_enabled", bool(env.tube_shaping_enabled))
        distance_enabled = getb("distance_enabled", bool(env.distance_shaping_enabled))
        stability_enabled = getb("stability_enabled", bool(env.stability_shaping_enabled))
        energy_enabled = getb("energy_enabled", bool(getattr(env, "energy_usage_shaping_enabled", False)))

        tube_reward = getf("tube_reward", float(env.tube_shaping_reward))
        distance_reward = getf("distance_reward", float(env.distance_shaping_reward))
        stability_reward = getf("stability_reward", float(env.stability_shaping_reward))
        energy_reward = getf("energy_reward_per_throttle", float(env.energy_usage_reward_per_throttle))

        term_succ_enabled = getb(
            "terminal_success_enabled",
            bool(getattr(env, "terminal_reward_success_enabled", True)),
        )
        term_succ = getf("terminal_reward_success", float(env.terminal_reward_success))
        term_crash_enabled = getb(
            "terminal_crash_enabled",
            bool(getattr(env, "terminal_reward_crash_enabled", True)),
        )
        term_crash = getf("terminal_reward_crash", float(env.terminal_reward_crash))

        # Apply to the environment first.
        env.set_tube_shaping_reward(tube_reward)
        env.set_distance_shaping_reward(distance_reward)
        env.set_stability_shaping_reward(stability_reward)
        env.set_energy_usage_reward_per_throttle(energy_reward)
        env.set_terminal_reward_success(term_succ)
        env.set_terminal_reward_crash(term_crash)

        env.set_terminal_reward_success_enabled(term_succ_enabled)
        env.set_terminal_reward_crash_enabled(term_crash_enabled)

        env.set_tube_shaping(tube_enabled)
        env.set_distance_shaping(distance_enabled)
        env.set_stability_shaping(stability_enabled)
        env.set_energy_usage_shaping(energy_enabled)

        # Reflect in UI without re-triggering callbacks.
        widgets: list[QtCore.QObject] = [
            self.spin_shape_tube,
            self.spin_shape_distance,
            self.spin_shape_stability,
            self.spin_shape_energy,
            self.spin_terminal_success,
            self.spin_terminal_crash,
            self.chk_shape_tube,
            self.chk_shape_distance,
            self.chk_shape_stability,
            self.chk_shape_energy,
            self.chk_terminal_success,
            self.chk_terminal_crash,
        ]

        old = [w.blockSignals(True) for w in widgets]
        try:
            self.spin_shape_tube.setValue(tube_reward)
            self.spin_shape_distance.setValue(distance_reward)
            self.spin_shape_stability.setValue(stability_reward)
            self.spin_shape_energy.setValue(energy_reward)
            self.spin_terminal_success.setValue(term_succ)
            self.spin_terminal_crash.setValue(term_crash)

            self.chk_shape_tube.setChecked(tube_enabled)
            self.chk_shape_distance.setChecked(distance_enabled)
            self.chk_shape_stability.setChecked(stability_enabled)
            self.chk_shape_energy.setChecked(energy_enabled)

            self.chk_terminal_success.setChecked(term_succ_enabled)
            self.chk_terminal_crash.setChecked(term_crash_enabled)
        finally:
            for w, prev in zip(widgets, old):
                w.blockSignals(prev)

    def _on_save_dqn(self) -> None:
        if not isinstance(self.sim.agent, DQNAgent):
            QtWidgets.QMessageBox.information(self, "Save DQN", "Switch the controller to 'DQN (learning)' first.")
            self._update_agent_io_buttons()
            return

        default_name = "dqn_lunar_lander.pt"
        initial_dir = self._get_agent_dialog_initial_dir()
        path, _flt = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save DQN agent",
            str(initial_dir / default_name),
            "PyTorch checkpoint (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        self._remember_agent_dialog_dir(path)

        try:
            # Store both agent weights and the UI learning stats so a specific
            # checkpoint restores the plotted curve and counters.
            ckpt = self.sim.agent.get_checkpoint(include_replay=False)
            ckpt["ui_learning_state"] = self.sim.get_learning_state()
            ckpt["reward_config"] = self._get_reward_config()
            ckpt["dqn_tricks"] = {
                "use_target_network": bool(self.chk_dqn_target.isChecked()),
                "use_replay_buffer": bool(self.chk_dqn_replay.isChecked()),
            }
            torch.save(ckpt, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save DQN", f"Failed to save checkpoint:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Save DQN", f"Saved checkpoint to:\n{path}")

    def _on_load_dqn(self) -> None:
        initial_dir = self._get_agent_dialog_initial_dir()
        path, _flt = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load DQN agent",
            str(initial_dir),
            "PyTorch checkpoint (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        self._remember_agent_dialog_dir(path)

        try:
            ckpt = torch.load(path, map_location="cpu")
            if not isinstance(ckpt, dict):
                raise ValueError("Invalid checkpoint format")
            agent = DQNAgent.from_checkpoint(ckpt)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load DQN", f"Failed to load checkpoint:\n{e}")
            return

        # Activate DQN controller without overwriting the loaded agent.
        self.sim.set_running(False)
        self.btn_start.setText("Start")
        self.cmb_controller.blockSignals(True)
        self.cmb_controller.setCurrentIndex(2)
        self.cmb_controller.blockSignals(False)

        self.sim.set_agent(agent)

        # Restore DQN learning tricks (if present) and reflect in UI.
        dqn_tricks = ckpt.get("dqn_tricks") if isinstance(ckpt, dict) else None
        if isinstance(dqn_tricks, dict):
            use_target = bool(dqn_tricks.get("use_target_network", getattr(agent, "use_target_network", True)))
            use_replay = bool(dqn_tricks.get("use_replay_buffer", getattr(agent, "use_replay_buffer", True)))
        else:
            use_target = bool(getattr(agent, "use_target_network", True))
            use_replay = bool(getattr(agent, "use_replay_buffer", True))

        self.chk_dqn_target.blockSignals(True)
        self.chk_dqn_replay.blockSignals(True)
        try:
            self.chk_dqn_target.setChecked(use_target)
            self.chk_dqn_replay.setChecked(use_replay)
        finally:
            self.chk_dqn_target.blockSignals(False)
            self.chk_dqn_replay.blockSignals(False)

        try:
            agent.set_use_target_network(use_target)
            agent.set_use_replay_buffer(use_replay)
        except Exception:
            pass

        st = ckpt.get("ui_learning_state") if isinstance(ckpt, dict) else None
        if isinstance(st, dict):
            self.sim.set_learning_state(st)
        else:
            self.sim.reset_learning_stats()

        rcfg = ckpt.get("reward_config") if isinstance(ckpt, dict) else None
        if isinstance(rcfg, dict):
            self._apply_reward_config(rcfg)
        self._update_agent_io_buttons()

        QtWidgets.QMessageBox.information(self, "Load DQN", f"Loaded DQN checkpoint from:\n{path}")

    def _on_reset_dqn(self) -> None:
        if torch is None:
            QtWidgets.QMessageBox.warning(self, "Reset DQN", "PyTorch is not available.")
            return

        if not isinstance(self.sim.agent, DQNAgent):
            QtWidgets.QMessageBox.information(self, "Reset DQN", "Switch the controller to 'DQN (learning)' first.")
            return

        ans = QtWidgets.QMessageBox.question(
            self,
            "Reset DQN",
            "Reset DQN agent to freshly randomized network weights?\n"
            "This clears current learning progress.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.sim.set_running(False)
        self.sim.set_stop_at_learning_step(None)
        self.sim.set_agent(self._build_dqn_agent(seed=None))
        self.sim.reset_learning_stats()
        self._update_agent_io_buttons()
        self.statusBar().showMessage("DQN agent reset with fresh random initialization.", 5000)

    def _get_agent_dialog_initial_dir(self) -> Path:
        d = self._last_agent_dialog_dir
        if d is not None and d.is_dir():
            return d
        return Path.cwd()

    def _remember_agent_dialog_dir(self, selected_path: str) -> None:
        p = Path(selected_path).expanduser()
        d = p if p.is_dir() else p.parent
        if d.is_dir():
            self._last_agent_dialog_dir = d

    def _on_telemetry(self, t: dict) -> None:
        # Defensive: during headless learning we may intentionally skip info creation
        # for most steps; ensure we always render a full info dict.
        if "status" not in t:
            t = self.sim.env.info()
        self.lbl_status.setText(f"status: {t['status']}")
        self.lbl_pos.setText(f"pos: ({t['pos_x']:.1f}, {t['pos_y']:.1f})")
        self.lbl_vel.setText(f"vel: ({t['vel_x']:.1f}, {t['vel_y']:.1f})")
        self.lbl_angle.setText(f"angle: {t['angle_deg']:.1f}°")
        self.lbl_alt.setText(f"altitude: {t['altitude']:.1f}")
        self.lbl_pad.setText(f"landing pad: {'yes' if t['in_landing_pad'] else 'no'}")

        contact = t.get("last_contact")
        if isinstance(contact, dict):
            lines = [
                "impact:",
                f"  vx={contact['impact_vx']:.1f}",
                f"  vy={contact['impact_vy']:.1f}",
                f"  angle={contact['impact_angle_deg']:.1f}°",
                "",
                "checks:",
            ]
            checks = contact.get("checks")
            if isinstance(checks, list):
                lines.extend([f"  {str(x)}" for x in checks])
            else:
                lines.append("(no checks)")
            text = "\n".join(lines)
            if text != self._contact_text_last:
                self._contact_text_last = text
                self.txt_contact.setPlainText(text)
        else:
            text = "impact: -"
            if text != self._contact_text_last:
                self._contact_text_last = text
                self.txt_contact.setPlainText(text)

    def _on_learning_stats(self, s: dict) -> None:
        ep = int(s.get("episode", 0))
        steps = int(s.get("steps", 0))
        succ = int(s.get("successes", 0))
        latest = s.get("latest_avg_reward_100")
        wall_s = 0.0
        try:
            wall_s = float(s.get("learning_wallclock_s", 0.0))
        except Exception:
            wall_s = 0.0

        # Format as "xxh yym zzsec".
        total = max(0, int(wall_s))
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        wall_str = f"{hh:02d}h {mm:02d}m {ss:02d}sec"

        lines = [
            f"episode: {ep}",
            f"learning steps: {steps}",
            f"successful landings: {succ}",
            f"learning wallclock: {wall_str}",
        ]
        if latest is not None:
            lines.append(f"avg reward (last 100): {float(latest):.1f}")
        self.txt_learning.setPlainText("\n".join(lines))

        curve = s.get("avg_reward_curve")
        if self._ax is not None and self._fig is not None and self._canvas is not None and isinstance(curve, list):
            self._ax.clear()
            self._ax.grid(True, alpha=0.25)
            if curve:
                xs = list(range(1, len(curve) + 1))
                self._ax.plot(xs, curve)
            self._ax.set_xlabel("chunk (100 steps)")
            self._ax.set_ylabel("avg reward")
            self._fig.tight_layout()
            self._canvas.draw_idle()

        # Optional fixed-length DQN experiment stop.
        if self.sim.running and isinstance(self.sim.agent, DQNAgent) and self.chk_stop_at_step.isChecked():
            stop_at = self._get_stop_at_step()
            if stop_at is not None and steps >= int(stop_at):
                self.sim.set_running(False)

    def _on_start_stop(self) -> None:
        if self.sim.running:
            self.sim.set_running(False)
            self.sim.set_stop_at_learning_step(None)
            return

        # Ensure active agent matches the selected controller.
        selected_idx = int(self.cmb_controller.currentIndex())
        try:
            if selected_idx == 0 and not isinstance(self.sim.agent, HumanKeyboardAgent):
                self.sim.set_agent(HumanKeyboardAgent())
            elif selected_idx == 1 and not isinstance(self.sim.agent, RandomAgent):
                self.sim.set_agent(RandomAgent(seed=0))
            elif selected_idx == 2 and not isinstance(self.sim.agent, DQNAgent):
                self.sim.set_agent(self._build_dqn_agent())
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Start failed",
                f"Could not prepare selected controller:\n{e}",
            )
            return

        self._update_agent_io_buttons()

        # Starting/resuming.
        is_human = isinstance(self.sim.agent, HumanKeyboardAgent)
        is_dqn = isinstance(self.sim.agent, DQNAgent)
        if is_human and self.sim.env.frozen:
            # After crash/landing in human mode, Start begins a fresh episode.
            self.sim.reset()

        if is_dqn and self.chk_stop_at_step.isChecked():
            stop_at = self._get_stop_at_step()
            if stop_at is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid stop step",
                    "Please enter a positive integer in 'Stop at step'.",
                )
                return
            # Absolute step target: stop when total learning steps reaches this value.
            if self.sim.total_learning_steps >= int(stop_at):
                QtWidgets.QMessageBox.information(
                    self,
                    "Stop at step reached",
                    f"Current learning steps are already {self.sim.total_learning_steps}.\n"
                    f"Set a larger 'Stop at step' value, or reset the DQN agent.",
                )
                return
            self.sim.set_stop_at_learning_step(int(stop_at))
        else:
            self.sim.set_stop_at_learning_step(None)

        self.sim.set_running(True)
        if is_human:
            # Ensure cursor keys work immediately (Start button would otherwise keep focus).
            self.sim.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _on_run_experiments(self) -> None:
        if torch is None:
            QtWidgets.QMessageBox.warning(self, "Run experiments", "PyTorch is not available.")
            return

        stop_at = self._get_stop_at_step()
        if stop_at is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Run experiments",
                "Please enter a positive integer in 'Stop at step'.",
            )
            return

        self.sim.set_running(False)

        root_dir = Path(__file__).resolve().parent
        agents_dir = root_dir / "agents"
        results_path = root_dir / "results.md"
        agents_dir.mkdir(parents=True, exist_ok=True)

        combos_tricks = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        reward_profiles = [
            ("landing_tube_only", "Landing Tube only"),
            ("distance_only", "Distance reward only"),
            ("stability_only", "Stability reward only"),
            ("tube_plus_distance", "Landing Tube + Distance reward"),
            ("tube_plus_distance_plus_stability", "Landing Tube + Distance + Stability reward"),
        ]

        rows: list[dict[str, Any]] = []
        self._set_experiment_controls_enabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        progress = QtWidgets.QProgressDialog("Preparing experiments...", "Cancel", 0, 20, self)
        progress.setWindowTitle("Run experiments")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        try:
            exp_nr = 0
            total = len(combos_tricks) * len(reward_profiles)
            progress_scale = 1000
            progress.setMaximum(total * progress_scale)
            t_all_start = time.perf_counter()
            canceled = False
            for use_target_network, use_replay_buffer in combos_tricks:
                self.chk_dqn_target.setChecked(use_target_network)
                self.chk_dqn_replay.setChecked(use_replay_buffer)

                for reward_key, reward_label in reward_profiles:
                    if progress.wasCanceled():
                        canceled = True
                        break

                    exp_nr += 1

                    done_before = exp_nr - 1
                    elapsed_before = max(0.0, time.perf_counter() - t_all_start)
                    if done_before > 0:
                        avg_per_exp = elapsed_before / float(done_before)
                        eta_before = avg_per_exp * float(total - done_before)
                        eta_text = self._format_duration(eta_before)
                    else:
                        eta_text = "calculating..."

                    status_msg = f"Running experiment {exp_nr}/{total} (ETA {eta_text})"
                    self.statusBar().showMessage(status_msg)
                    progress.setLabelText(status_msg)
                    progress.setValue(done_before * progress_scale)
                    QtWidgets.QApplication.processEvents()

                    self._apply_reward_profile(reward_key)
                    self.sim.set_agent(self._build_dqn_agent())

                    t_exp_start = time.perf_counter()

                    def on_exp_progress(current_steps: int, target_steps: int) -> None:
                        target = max(1, int(target_steps))
                        cur = max(0, min(int(current_steps), target))
                        frac = float(cur) / float(target)
                        elapsed_exp = max(0.0, time.perf_counter() - t_exp_start)
                        if cur > 0:
                            eta_exp = elapsed_exp * float(target - cur) / float(cur)
                            exp_eta_text = self._format_duration(eta_exp)
                        else:
                            exp_eta_text = "calculating..."

                        virtual_done = float(done_before) + frac
                        elapsed_global = max(0.0, time.perf_counter() - t_all_start)
                        if virtual_done > 1e-9:
                            eta_global = elapsed_global * float(total - virtual_done) / float(virtual_done)
                            global_eta_text = self._format_duration(eta_global)
                        else:
                            global_eta_text = "calculating..."

                        msg = (
                            f"Running experiment {exp_nr}/{total} | "
                            f"steps {cur}/{target} ({100.0 * frac:.1f}%) | "
                            f"exp ETA {exp_eta_text} | total ETA {global_eta_text}"
                        )
                        self.statusBar().showMessage(msg)
                        progress.setLabelText(msg)
                        progress.setValue(int((float(done_before) + frac) * float(progress_scale)))

                    metrics = self._run_fixed_step_dqn_training(
                        steps=int(stop_at),
                        process_events=True,
                        progress_cb=on_exp_progress,
                        should_stop=progress.wasCanceled,
                    )

                    if bool(metrics.get("canceled", False)):
                        canceled = True
                        break

                    target_tag = "target_on" if use_target_network else "target_off"
                    replay_tag = "replay_on" if use_replay_buffer else "replay_off"
                    file_name = f"exp_{exp_nr:02d}_{target_tag}_{replay_tag}_{reward_key}_steps_{int(stop_at)}.pt"
                    agent_path = agents_dir / file_name

                    if isinstance(self.sim.agent, DQNAgent):
                        ckpt = self.sim.agent.get_checkpoint(include_replay=False)
                        ckpt["ui_learning_state"] = self.sim.get_learning_state()
                        ckpt["reward_config"] = self._get_reward_config()
                        ckpt["dqn_tricks"] = {
                            "use_target_network": bool(use_target_network),
                            "use_replay_buffer": bool(use_replay_buffer),
                        }
                        ckpt["experiment"] = {
                            "id": int(exp_nr),
                            "total_experiments": int(total),
                            "reward_profile": reward_key,
                            "reward_profile_label": reward_label,
                            "steps": int(stop_at),
                            "successes": int(metrics["successes"]),
                        }
                        torch.save(ckpt, str(agent_path))

                    rows.append(
                        {
                            "id": int(exp_nr),
                            "target": bool(use_target_network),
                            "replay": bool(use_replay_buffer),
                            "reward_label": str(reward_label),
                            "steps": int(metrics["steps"]),
                            "successes": int(metrics["successes"]),
                            "agent_file": str(Path("agents") / file_name),
                        }
                    )

                    elapsed_after = max(0.0, time.perf_counter() - t_all_start)
                    avg_after = elapsed_after / float(exp_nr)
                    remaining = max(0, total - exp_nr)
                    eta_after = avg_after * float(remaining)
                    done_msg = (
                        f"Finished experiment {exp_nr}/{total} | "
                        f"elapsed {self._format_duration(elapsed_after)} | "
                        f"ETA {self._format_duration(eta_after)}"
                    )
                    self.statusBar().showMessage(done_msg)
                    progress.setLabelText(done_msg)
                    progress.setValue(exp_nr * progress_scale)
                    QtWidgets.QApplication.processEvents()

                if canceled:
                    break

            lines: list[str] = [
                "# DQN Lunar Lander Experiment Results",
                "",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "| # | TargetNetwork | ReplayBuffer | Reward Setup | Steps | Successful Landings | Agent File |",
                "|---:|:-------------:|:------------:|:-------------|------:|--------------------:|:-----------|",
            ]
            for row in rows:
                lines.append(
                    "| "
                    f"{row['id']} | "
                    f"{'yes' if row['target'] else 'no'} | "
                    f"{'yes' if row['replay'] else 'no'} | "
                    f"{row['reward_label']} | "
                    f"{row['steps']} | "
                    f"{row['successes']} | "
                    f"{row['agent_file']} |"
                )

            results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            total_elapsed = max(0.0, time.perf_counter() - t_all_start)
            if canceled:
                self.statusBar().showMessage(
                    f"Canceled after {len(rows)} experiments. Wrote {results_path.name}. Elapsed {self._format_duration(total_elapsed)}.",
                    10000,
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "Run experiments",
                    f"Canceled after {len(rows)} experiments.\n\n"
                    f"Elapsed: {self._format_duration(total_elapsed)}\n"
                    f"Partial results: {results_path}\n"
                    f"Saved agents:    {agents_dir}",
                )
                return

            self.statusBar().showMessage(
                f"Finished {len(rows)} experiments in {self._format_duration(total_elapsed)}. Wrote {results_path.name} and saved agents.",
                10000,
            )
            QtWidgets.QMessageBox.information(
                self,
                "Run experiments",
                f"Finished {len(rows)} experiments.\n\n"
                f"Elapsed: {self._format_duration(total_elapsed)}\n"
                f"Results: {results_path}\n"
                f"Agents:  {agents_dir}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Run experiments", f"Experiment run failed:\n{e}")
        finally:
            progress.close()
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_experiment_controls_enabled(True)
            self._update_agent_io_buttons()

    def _on_running_changed(self, running: bool) -> None:
        self.btn_start.setText("Stop" if running else "Start")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
