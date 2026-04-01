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
from dataclasses import dataclass
from typing import List, Protocol, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6 import QtMultimedia
except Exception:  # pragma: no cover
    QtMultimedia = None


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


class LunarLanderEnv:
    """Minimal lunar lander environment.

    - `action` is a 3-bit mask: MAIN/LEFT/RIGHT (0..7)
    - `observation` is a flat tuple of floats (see `state_names`)

    This is intentionally small and Gym-like (reset/step) so adding a Q-learning
    agent later is straightforward.
    """

    action_space_n: int = ACTION_SPACE_N
    state_names: tuple[str, ...] = (
        "pos_x",
        "pos_y",
        "vel_x",
        "vel_y",
        "angle_rad",
        "ang_vel",
        "altitude",
        "pad_center_dx",
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

        self.reset()

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

        return self.observation()

    def observation(self) -> Observation:
        lander = self.lander
        ground = self.terrain.height_at(lander.pos.x)
        altitude = max(0.0, lander.pos.y - (ground + lander.radius))
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        pad_center_dx = lander.pos.x - self.landing_pad.cx
        in_pad = 1.0 if (self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1) else 0.0
        return (
            lander.pos.x,
            lander.pos.y,
            lander.vel.x,
            lander.vel.y,
            ang,
            lander.ang_vel,
            altitude,
            pad_center_dx,
            in_pad,
        )

    def info(self) -> dict:
        lander = self.lander
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        ground = self.terrain.height_at(lander.pos.x)
        altitude = max(0.0, lander.pos.y - (ground + lander.radius))
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
            "status": self.status_text or "FLYING",
            "in_landing_pad": in_pad,
            "action": int(self.last_action) & 0b111,
            "terminated": bool(self.frozen),
        }
        if self.last_contact is not None:
            payload["last_contact"] = self.last_contact
        if self._just_terminated_success is not None:
            payload["terminal_success"] = bool(self._just_terminated_success)
        return payload

    def step(self, action: int, dt: float) -> tuple[Observation, float, bool, dict]:
        self._just_terminated_success = None
        self.last_action = int(action) & 0b111

        if self.frozen:
            obs = self.observation()
            info = self.info()
            return obs, 0.0, True, info

        dt = clamp(float(dt), 0.0, 1.0 / 20.0)
        dt *= self.time_scale

        was_frozen = self.frozen
        self._step_physics(dt, self.last_action)

        terminated = bool(self.frozen)

        reward = 0.0
        terminal_success: bool | None = None
        # Minimal reward shaping: only terminal events.
        if (not was_frozen) and terminated:
            if self.status_text == "LANDED":
                reward = 1.0
                terminal_success = True
            else:
                reward = -1.0
                terminal_success = False
            self._just_terminated_success = terminal_success

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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(900, 520)

        self.env = LunarLanderEnv(world_w=900.0, world_h=520.0)
        self.agent: Agent = HumanKeyboardAgent()

        # When rendering is disabled, we fast-forward the simulation by stepping
        # multiple times per UI tick.
        self.rendering_enabled = True
        self.fast_forward_steps_per_tick = 250

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

        # Rendered mode: use wall-clock dt. Headless mode: fixed dt and many steps.
        if self.rendering_enabled:
            steps = 1
            dt_step = dt
        else:
            steps = self.fast_forward_steps_per_tick
            dt_step = 1.0 / 60.0

        info_last: dict | None = None
        terminated = False
        for _i in range(steps):
            obs = self.env.observation()
            action = int(self.agent.take_action(obs, self.env.info()))
            # Update GUI-visible thruster state from the agent action.
            self.key_up = bool(action & ACTION_MAIN)
            self.key_left = bool(action & ACTION_LEFT)
            self.key_right = bool(action & ACTION_RIGHT)

            _obs2, _reward, terminated, info_last = self.env.step(action, dt_step)
            self.status_text = self.env.status_text
            if terminated:
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

        self.telemetryUpdated.emit(info_last)

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
        panel.setFixedWidth(260)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        title = QtWidgets.QLabel("Controls")
        title.setStyleSheet("font-weight: 600;")
        v.addWidget(title)

        v.addWidget(QtWidgets.QLabel("Arrow Left: left thruster"))
        v.addWidget(QtWidgets.QLabel("Arrow Right: right thruster"))
        v.addWidget(QtWidgets.QLabel("Arrow Up: bottom thruster"))
        v.addSpacing(6)

        self.chk_sound = QtWidgets.QCheckBox("Sound")
        if not self.sim.sound_available():
            self.chk_sound.setEnabled(False)
            self.chk_sound.setToolTip("QtMultimedia not available in this environment")
        self.chk_sound.toggled.connect(self.sim.set_sound_enabled)
        v.addWidget(self.chk_sound)
        if self.sim.sound_available():
            # Sound on by default.
            self.chk_sound.setChecked(True)

        self.chk_render = QtWidgets.QCheckBox("Rendering")
        self.chk_render.setToolTip("Disable to fast-forward the simulation without drawing")
        self.chk_render.setChecked(True)
        self.chk_render.toggled.connect(self.sim.set_rendering_enabled)
        v.addWidget(self.chk_render)

        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.restart_btn.clicked.connect(self._on_restart)
        v.addWidget(self.restart_btn)

        v.addSpacing(10)
        stats_title = QtWidgets.QLabel("Telemetry")
        stats_title.setStyleSheet("font-weight: 600;")
        v.addWidget(stats_title)

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
        self.txt_contact.setMinimumHeight(110)
        self.txt_contact.setMaximumHeight(170)

        for lbl in [
            self.lbl_status,
            self.lbl_pos,
            self.lbl_vel,
            self.lbl_angle,
            self.lbl_alt,
            self.lbl_pad,
        ]:
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            v.addWidget(lbl)

        v.addSpacing(6)
        v.addWidget(contact_title)
        v.addWidget(self.txt_contact)

        v.addStretch(1)

        layout.addWidget(panel, 0)
        self.setCentralWidget(root)

        self.sim.telemetryUpdated.connect(self._on_telemetry)

    def _on_restart(self) -> None:
        self.sim.reset()

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


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 520)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
