"""Minimal PySide6 lunar lander (no Gymnasium).

Controls:
- Left Arrow: fire left thruster
- Right Arrow: fire right thruster
- Up Arrow: fire bottom thruster

Goal: land softly on the uneven moon surface.
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import List, Protocol, Tuple

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
        hidden_sizes: tuple[int, int] = (128, 128),
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
        seed: int | None = 0,
        device: str | None = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for DQNAgent. Install torch or use the Random agent.")

        self.obs_dim = int(obs_dim)
        self.action_space_n = int(action_space_n)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.train_every = int(train_every)
        self.target_update_every = int(target_update_every)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

        if seed is not None:
            random.seed(int(seed))
            torch.manual_seed(int(seed))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        h1, h2 = hidden_sizes

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

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=float(lr))

        # (s, a, r, s2, done)
        self.replay: deque[tuple[Observation, int, float, Observation, bool]] = deque(maxlen=int(replay_size))
        self._step_count = 0

        self._prev_obs: Observation | None = None
        self._prev_action: int | None = None

    def reset(self) -> None:
        self._prev_obs = None
        self._prev_action = None

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

        self.replay.append((self._prev_obs, int(self._prev_action), float(reward), next_observation, bool(terminated)))

        if terminated:
            self._prev_obs = None
            self._prev_action = None

        if len(self.replay) < max(self.warmup_steps, self.batch_size):
            return
        if self.train_every > 1 and (self._step_count % self.train_every) != 0:
            return

        self._train_step()

        if (self._step_count % max(1, self.target_update_every)) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self) -> None:
        if torch is None:
            return

        batch = random.sample(self.replay, k=self.batch_size)

        states = torch.stack([self._obs_tensor(s) for (s, _a, _r, _s2, _d) in batch], dim=0)
        actions = torch.tensor([a for (_s, a, _r, _s2, _d) in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([r for (_s, _a, r, _s2, _d) in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([self._obs_tensor(s2) for (_s, _a, _r, s2, _d) in batch], dim=0)
        dones = torch.tensor([1.0 if d else 0.0 for (_s, _a, _r, _s2, d) in batch], dtype=torch.float32, device=self.device)

        q_sa = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + (1.0 - dones) * (self.gamma * next_q)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optim.step()


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

        self.success_max_abs_vy = 60.0
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
        self.last_reward_time = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
        self.last_reward_terminal = 0.0
        self.last_distance_metric = 0.0
        self.last_stability_metric = 0.0

        # -----------------
        # Reward shaping
        # -----------------
        # Terminal reward is always applied:
        #   +1 for LANDED, -1 for CRASHED
        # Optional shaping rewards can be toggled independently.
        self.shaping_time_penalty_enabled = True
        self.shaping_distance_enabled = True
        self.shaping_stability_enabled = True

        # Weights are multiplied by dt, so they are roughly time-consistent.
        self.shaping_time_weight = -0.05
        # Potential-based shaping uses the absolute value of these coefficients.
        # (Sign is ignored; potentials are defined so that progress yields +reward.)
        self.shaping_distance_weight = -0.20
        self.shaping_stability_weight = -0.15

        # Potential-based shaping discount. Keep in sync with the agent discount.
        self.shaping_gamma = 0.995

        # Terminal reward magnitude (large to accelerate learning).
        self.terminal_reward_success = 10.0
        self.terminal_reward_failure = -10.0

        self.reset()

    def set_reward_shaping(
        self,
        *,
        time_penalty: bool | None = None,
        distance: bool | None = None,
        stability: bool | None = None,
    ) -> None:
        if time_penalty is not None:
            self.shaping_time_penalty_enabled = bool(time_penalty)
        if distance is not None:
            self.shaping_distance_enabled = bool(distance)
        if stability is not None:
            self.shaping_stability_enabled = bool(stability)

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
        self.last_reward_time = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
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
            "reward_time": float(self.last_reward_time),
            "reward_distance": float(self.last_reward_distance),
            "reward_stability": float(self.last_reward_stability),
            "reward_terminal": float(self.last_reward_terminal),
            "distance_metric": float(self.last_distance_metric),
            "stability_metric": float(self.last_stability_metric),
        }
        if self.last_contact is not None:
            payload["last_contact"] = self.last_contact
        if self._just_terminated_success is not None:
            payload["terminal_success"] = bool(self._just_terminated_success)
        return payload

    def step(self, action: int, dt: float) -> tuple[Observation, float, bool, dict]:
        self._just_terminated_success = None
        self.last_action = int(action) & 0b111

        # Default reward breakdown (overwritten below).
        self.last_reward_total = 0.0
        self.last_reward_time = 0.0
        self.last_reward_distance = 0.0
        self.last_reward_stability = 0.0
        self.last_reward_terminal = 0.0
        self.last_distance_metric = 0.0
        self.last_stability_metric = 0.0

        if self.frozen:
            obs = self.observation()
            info = self.info()
            return obs, 0.0, True, info

        dt = clamp(float(dt), 0.0, 1.0 / 20.0)
        dt *= self.time_scale

        # Shaping reward is computed as:
        # - time penalty (dense): w_time * dt
        # - potential-based shaping for distance/stability:
        #     r = gamma * Phi(s_{t+1}) - Phi(s_t)
        #   with potentials defined so that moving toward the goal yields +reward.
        reward_time = 0.0
        reward_distance = 0.0
        reward_stability = 0.0
        distance_metric_curr = 0.0
        stability_metric_curr = 0.0
        distance_metric_next = 0.0
        stability_metric_next = 0.0

        if self.shaping_time_penalty_enabled:
            reward_time = self.shaping_time_weight * dt

        def distance_metric_for(lander: Lander) -> float:
            dx = lander.pos.x - self.landing_pad.cx
            dy = lander.pos.y - (self.landing_pad.y + lander.radius)
            ndx = abs(dx) / 450.0
            ndy = abs(dy) / 520.0
            return clamp(ndx + ndy, 0.0, 4.0)

        def stability_metric_for(lander: Lander) -> float:
            ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
            nvx = abs(lander.vel.x) / 250.0
            nvy = abs(lander.vel.y) / 250.0
            nang = abs(ang) / math.pi
            nw = abs(lander.ang_vel) / 1.5
            return clamp(nvx + nvy + nang + nw, 0.0, 6.0)

        lander = self.lander
        if self.shaping_distance_enabled:
            distance_metric_curr = distance_metric_for(lander)
        if self.shaping_stability_enabled:
            stability_metric_curr = stability_metric_for(lander)

        was_frozen = self.frozen
        self._step_physics(dt, self.last_action)

        terminated = bool(self.frozen)

        # Potential-based shaping uses the post-step state.
        lander2 = self.lander
        if self.shaping_distance_enabled:
            distance_metric_next = distance_metric_for(lander2)
            phi_curr = -distance_metric_curr
            phi_next = -distance_metric_next
            coef = abs(float(self.shaping_distance_weight))
            reward_distance = coef * (self.shaping_gamma * phi_next - phi_curr)

        if self.shaping_stability_enabled:
            stability_metric_next = stability_metric_for(lander2)
            phi_curr = -stability_metric_curr
            phi_next = -stability_metric_next
            coef = abs(float(self.shaping_stability_weight))
            reward_stability = coef * (self.shaping_gamma * phi_next - phi_curr)

        reward = reward_time + reward_distance + reward_stability

        terminal_success: bool | None = None
        terminal_reward = 0.0
        # Terminal reward overrides shaping reward.
        if (not was_frozen) and terminated:
            if self.status_text == "LANDED":
                terminal_reward = float(self.terminal_reward_success)
                terminal_success = True
            else:
                terminal_reward = float(self.terminal_reward_failure)
                terminal_success = False
            self._just_terminated_success = terminal_success

        if terminal_reward != 0.0:
            reward = terminal_reward

        self.last_reward_total = float(reward)
        self.last_reward_time = float(reward_time)
        self.last_reward_distance = float(reward_distance)
        self.last_reward_stability = float(reward_stability)
        self.last_reward_terminal = float(terminal_reward)
        # Expose current (post-step) metrics for UI overlays.
        self.last_distance_metric = float(distance_metric_next if self.shaping_distance_enabled else 0.0)
        self.last_stability_metric = float(stability_metric_next if self.shaping_stability_enabled else 0.0)

        obs = self.observation()
        info = self.info()
        if terminal_success is not None:
            info["terminal_success"] = terminal_success

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

    def reset_learning_stats(self) -> None:
        self.episode_nr = 0
        self.total_learning_steps = 0
        self.successful_landings = 0
        self._reward_window_sum = 0.0
        self._reward_window_count = 0
        self.avg_reward_every_100 = []
        self._emit_learning_stats(latest_avg=None)

    def _emit_learning_stats(self, *, latest_avg: float | None) -> None:
        self.learningStatsUpdated.emit(
            {
                "episode": int(self.episode_nr),
                "steps": int(self.total_learning_steps),
                "successes": int(self.successful_landings),
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
        for _i in range(steps):
            obs = self.env.observation()
            action = int(self.agent.take_action(obs, self.env.info()))
            # Update GUI-visible thruster state from the agent action.
            self.key_up = bool(action & ACTION_MAIN)
            self.key_left = bool(action & ACTION_LEFT)
            self.key_right = bool(action & ACTION_RIGHT)

            obs2, reward, terminated, info_last = self.env.step(action, dt_step)
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
                    getattr(self.agent, "observe")(reward, terminated, obs2, info_last)
                except Exception:
                    pass

            if terminated:
                if learning_mode:
                    self.episode_nr += 1
                    if info_last.get("terminal_success") is True:
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

        # Keep audio state in sync even if keys are held.
        if self.rendering_enabled:
            self._update_thruster_sounds()
        else:
            self._stop_thruster_sounds()

        if info_last is None:
            info_last = self.env.info()

        # Play landing/collision sound on the transition to terminal state.
        if info_last.get("terminal_success") is True:
            self._play_landing_sound(True)
        elif info_last.get("terminal_success") is False:
            self._play_landing_sound(False)

        emit_ui = True
        if learning_mode and (not self.rendering_enabled):
            steps_since = self.total_learning_steps - self._last_ui_emit_learning_step
            emit_ui = (latest_avg is not None) or (steps_since >= self._ui_emit_every_learning_steps)
            if emit_ui:
                self._last_ui_emit_learning_step = self.total_learning_steps

        if emit_ui:
            self.telemetryUpdated.emit(info_last)
            self._emit_learning_stats(latest_avg=latest_avg)

        if self.rendering_enabled:
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _ = event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QtGui.QColor(10, 10, 14))

        self._draw_stars(painter)
        self._draw_terrain(painter)
        self._draw_lander(painter)
        self._draw_shaping_overlay(painter)

    def _draw_shaping_overlay(self, painter: QtGui.QPainter) -> None:
        if not self.rendering_enabled:
            return
        if isinstance(self.agent, HumanKeyboardAgent):
            return

        info = self.env.info()
        lander = self.env.lander
        pad = self.env.landing_pad

        p_lander = self._world_to_screen(lander.pos)
        p_pad = self._world_to_screen(Vec2(pad.cx, pad.y + lander.radius))

        painter.save()
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setWidthF(1.0)
        painter.setPen(pen)

        # Distance shaping: line from lander to pad center.
        painter.drawLine(p_lander, p_pad)
        mid = QtCore.QPointF(0.5 * (p_lander.x() + p_pad.x()), 0.5 * (p_lander.y() + p_pad.y()))
        dist_r = float(info.get("reward_distance", 0.0))
        dist_m = float(info.get("distance_metric", 0.0))
        painter.drawText(mid + QtCore.QPointF(6.0, -6.0), f"dist: {dist_r:+.3f} (m={dist_m:.2f})")

        # Stability shaping beside the lander.
        vx = float(info.get("vel_x", 0.0))
        vy = float(info.get("vel_y", 0.0))
        ang_deg = float(info.get("angle_deg", 0.0))
        stab_r = float(info.get("reward_stability", 0.0))
        stab_m = float(info.get("stability_metric", 0.0))
        painter.drawText(
            p_lander + QtCore.QPointF(18.0, -18.0),
            f"stab: {stab_r:+.3f} (m={stab_m:.2f}) |v|=({abs(vx):.1f},{abs(vy):.1f}) |a|={abs(ang_deg):.1f}°",
        )

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
        shaping_title = QtWidgets.QLabel("Reward shaping")
        shaping_title.setStyleSheet("font-weight: 600;")
        left_v.addWidget(shaping_title)

        self.chk_shape_time = QtWidgets.QCheckBox("Time penalty")
        self.chk_shape_dist = QtWidgets.QCheckBox("Distance to pad")
        self.chk_shape_stab = QtWidgets.QCheckBox("Stability (vel/angle)")

        self.chk_shape_time.toggled.connect(lambda on: self.sim.env.set_reward_shaping(time_penalty=on))
        self.chk_shape_dist.toggled.connect(lambda on: self.sim.env.set_reward_shaping(distance=on))
        self.chk_shape_stab.toggled.connect(lambda on: self.sim.env.set_reward_shaping(stability=on))

        left_v.addWidget(self.chk_shape_time)
        left_v.addWidget(self.chk_shape_dist)
        left_v.addWidget(self.chk_shape_stab)

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

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start_stop)
        left_v.addWidget(self.btn_start)

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

        # Shaping rewards enabled by default.
        self.chk_shape_time.setChecked(True)
        self.chk_shape_dist.setChecked(True)
        self.chk_shape_stab.setChecked(True)

        # No automatic start.
        self.sim.set_running(False)
        self.sim.reset_learning_stats()

    def _on_controller_changed(self, idx: int) -> None:
        # 0 = Human, 1 = Random, 2 = DQN
        self.sim.set_running(False)
        self.btn_start.setText("Start")
        if idx == 0:
            self.sim.set_agent(HumanKeyboardAgent())
            return

        if idx == 1:
            self.sim.set_agent(RandomAgent(seed=0))
            return

        # DQN agent learns online while controlling.
        obs_dim = len(self.sim.env.observation())
        self.sim.set_agent(DQNAgent(obs_dim=obs_dim, seed=0))

        # New learning session for this agent.
        self.sim.reset_learning_stats()

        # Disable UI-heavy stuff by default for learning speed.
        self.chk_render.setChecked(False)
        if self.sim.sound_available():
            self.chk_sound.setChecked(False)

    def _on_telemetry(self, t: dict) -> None:
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

        lines = [
            f"episode: {ep}",
            f"learning steps: {steps}",
            f"successful landings: {succ}",
        ]
        if latest is not None:
            lines.append(f"avg reward (last 100): {float(latest):.4f}")
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

    def _on_start_stop(self) -> None:
        if self.sim.running:
            self.sim.set_running(False)
            return

        # Starting/resuming.
        is_human = isinstance(self.sim.agent, HumanKeyboardAgent)
        if is_human and self.sim.env.frozen:
            # After crash/landing in human mode, Start begins a fresh episode.
            self.sim.reset()
        self.sim.set_running(True)

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
