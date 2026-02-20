"""
Simple Q-Learning demo (PySide6) — clean world view (no text overlay)
--------------------------------------------------------------------
A circle "robot" lives in a 2D torus world (wrap-around edges) with:
- green pills  (+reward)
- red pills    (-reward)

State (DISCRETIZED):
    (angle_to_nearest_green, dist_to_nearest_green,
     angle_to_nearest_red,   dist_to_nearest_red)
All angles/distances are RELATIVE to the robot's current orientation.

Actions:
    A: turn left (TURN_DEG)
    B: move forward (MOVE_PX)
    C: turn right (TURN_DEG)

Goal: trial-and-error learning to collect green and avoid red.

Requirements:
    pip install PySide6
Run:
    python qlearning_pills_pyside6.py
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, QPointF, QEvent
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFrame
)

# =========================
# Teaching-friendly knobs
# =========================

WORLD_W, WORLD_H = 800, 600
N_GREEN = 18
N_RED = 18
PILL_RADIUS = 6

ROBOT_RADIUS = 14
TURN_DEG = 8.0          # A / C  (bigger turns = faster credit assignment)
MOVE_PX = 4.0           # B      (bigger steps = more frequent pill hits)

# Simulation speed (RL updates per rendered frame)
STEPS_PER_TICK = 40

# Hold-to-step speed when Space is held down (ms per step)
SPACE_STEP_INTERVAL_MS = 10

# Reward curve (for learning curve)
# We compute ONE average point per MA_WINDOW environment steps.
MA_WINDOW = 1000
MA_HISTORY_MAX = 5000    # max stored points; older points get compressed to keep an overview

# Rewards
R_GREEN = +10.0
R_RED = -10.0
R_STEP = -0.01          # small living cost (keep it small so turning isn't too "expensive")

# Dense reward shaping (potential-based) to speed up learning.
# r' = r + shaping_scale * (gamma * Phi(s') - Phi(s))
USE_REWARD_SHAPING = True
SHAPING_SCALE = 2.0
PHI_GREEN_W = 1.0       # encourage getting closer to green
PHI_RED_W = 0.6         # encourage staying away from red

# Q-learning parameters
ALPHA = 0.20            # learning rate
GAMMA = 0.95            # discount factor
EPS_START = 1.00
EPS_MIN = 0.05
EPS_DECAY = 0.9999      # per step (slower decay keeps exploration longer)

# Optional: occasionally randomize the robot + pills (keeps training diverse)
# Disabled by default.
SOFT_RESET_EVERY_STEPS = 0

# State discretization (compact Q-table)
ANGLE_BINS = 24         # 360/24 = 15° bins
DIST_BINS = 10
MAX_DIST = math.hypot(WORLD_W / 2, WORLD_H / 2)  # max shortest distance on torus

# Actions: indices 0,1,2 map to A,B,C
ACTIONS = ["A: turn left", "B: forward", "C: turn right"]


# =========================
# Data classes
# =========================

@dataclass
class Pill:
    x: float
    y: float
    is_green: bool


@dataclass
class Robot:
    x: float
    y: float
    heading_rad: float  # orientation


# =========================
# Helper math (torus world)
# =========================

def wrap_pos(x: float, y: float) -> tuple[float, float]:
    """Wrap-around edges (torus)."""
    return x % WORLD_W, y % WORLD_H


def torus_delta(rx: float, ry: float, tx: float, ty: float) -> tuple[float, float]:
    """
    Shortest vector from robot (r) to target (t) in a wrap-around world.
    Returns (dx, dy) where robot + (dx,dy) is the closest image of the target.
    """
    dx = ((tx - rx + WORLD_W / 2) % WORLD_W) - WORLD_W / 2
    dy = ((ty - ry + WORLD_H / 2) % WORLD_H) - WORLD_H / 2
    return dx, dy


def angle_normalize(rad: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while rad <= -math.pi:
        rad += 2 * math.pi
    while rad > math.pi:
        rad -= 2 * math.pi
    return rad


def discretize_angle(rad: float) -> int:
    """Map [-pi, pi] -> {0..ANGLE_BINS-1}."""
    rad = angle_normalize(rad)
    shifted = rad + math.pi  # -> [0, 2pi)
    idx = int(shifted / (2 * math.pi) * ANGLE_BINS)
    return max(0, min(ANGLE_BINS - 1, idx))


def discretize_dist(d: float) -> int:
    """Map [0, MAX_DIST] -> {0..DIST_BINS-1}."""
    d = max(0.0, min(MAX_DIST, d))
    idx = int(d / MAX_DIST * DIST_BINS)
    return max(0, min(DIST_BINS - 1, idx))


# =========================
# Q-Learning agent
# =========================

class QLearningAgent:
    def __init__(self):
        # Q-table: dict[state_tuple] -> list[Q(s,a)] for a in {0,1,2}
        # Slightly optimistic init helps exploration in sparse-reward worlds.
        self.Q = defaultdict(lambda: [1.0, 1.0, 1.0])
        self.eps = EPS_START

    def choose_action(self, state: tuple[int, int, int, int]) -> int:
        """Epsilon-greedy policy."""
        if random.random() < self.eps:
            return random.randint(0, 2)
        qs = self.Q[state]
        # argmax with RANDOM tie-breaking (avoids bias toward lower indices)
        max_q = max(qs)
        best_actions = [a for a, q in enumerate(qs) if q == max_q]
        return random.choice(best_actions)

    def decay_epsilon(self):
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

    # ==========================================================
    # >>>>>>>>>>>>   Q-LEARNING UPDATE RULE (core!)   <<<<<<<<<<
    # ==========================================================
    def update(self, s, a, r, s_next):
        """
        Q(s,a) <- Q(s,a) + alpha * ( r + gamma*max_a' Q(s_next,a') - Q(s,a) )
        """
        q_sa = self.Q[s][a]
        max_next = max(self.Q[s_next])
        td_target = r + GAMMA * max_next
        td_error = td_target - q_sa
        self.Q[s][a] = q_sa + ALPHA * td_error
    # ==========================================================


# =========================
# Environment + Simulation
# =========================

class World:
    def __init__(self):
        self.robot = Robot(WORLD_W * 0.35, WORLD_H * 0.35, heading_rad=0.0)
        self.pills: list[Pill] = []
        self.reset_pills()

        self.agent = QLearningAgent()
        self.steps = 0
        self.score = 0.0
        self.last_reward = 0.0
        self.last_reward_train = 0.0
        self.last_action: int | None = None
        self.collected_green = 0
        self.hit_red = 0

        # Reward curve tracking (environment reward, not shaped reward)
        # Accumulate MA_WINDOW rewards, then emit one averaged point.
        self._reward_block_sum = 0.0
        self._reward_block_count = 0
        self.reward_ma_1000 = 0.0  # last completed block average
        self.reward_ma_history: list[float] = []

    def _append_ma_history(self, value: float):
        """Append a new MA point; compress history if it grows too long.

        Compression keeps a full-run overview without unbounded memory growth.
        """
        self.reward_ma_history.append(value)

        # If history gets too long, downsample by averaging pairs and doubling stride.
        if len(self.reward_ma_history) > MA_HISTORY_MAX:
            h = self.reward_ma_history
            compressed: list[float] = []
            for i in range(0, len(h) - 1, 2):
                compressed.append((h[i] + h[i + 1]) * 0.5)
            if len(h) % 2 == 1:
                compressed.append(h[-1])
            self.reward_ma_history = compressed

    def reset_pills(self):
        self.pills = []
        for _ in range(N_GREEN):
            self.pills.append(Pill(random.random() * WORLD_W, random.random() * WORLD_H, True))
        for _ in range(N_RED):
            self.pills.append(Pill(random.random() * WORLD_W, random.random() * WORLD_H, False))

    def respawn_pill(self, pill: Pill):
        pill.x = random.random() * WORLD_W
        pill.y = random.random() * WORLD_H

    def soft_reset(self):
        """Reset robot pose and pill positions but keep learned Q-table."""
        self.robot.x = random.random() * WORLD_W
        self.robot.y = random.random() * WORLD_H
        self.robot.heading_rad = random.uniform(-math.pi, math.pi)
        self.reset_pills()

    def nearest_pill(self, want_green: bool) -> tuple[Pill, float, float, float]:
        """Returns (pill, dist, dx, dy) for nearest pill of requested type using torus distance."""
        rx, ry = self.robot.x, self.robot.y
        best_pill = None
        best_d = 1e9
        best_dx = 0.0
        best_dy = 0.0

        for p in self.pills:
            if p.is_green != want_green:
                continue
            dx, dy = torus_delta(rx, ry, p.x, p.y)
            d = math.hypot(dx, dy)
            if d < best_d:
                best_pill = p
                best_d = d
                best_dx, best_dy = dx, dy

        assert best_pill is not None
        return best_pill, best_d, best_dx, best_dy

    def get_state(self) -> tuple[int, int, int, int]:
        """
        State = (rel_angle_to_green, rel_dist_to_green, rel_angle_to_red, rel_dist_to_red),
        discretized to small integers for a compact Q-table.
        """
        rob = self.robot

        _, gd, gdx, gdy = self.nearest_pill(True)
        _, rd, rdx, rdy = self.nearest_pill(False)

        ang_g = math.atan2(gdy, gdx)
        ang_r = math.atan2(rdy, rdx)

        rel_g = angle_normalize(ang_g - rob.heading_rad)
        rel_r = angle_normalize(ang_r - rob.heading_rad)

        return (
            discretize_angle(rel_g),
            discretize_dist(gd),
            discretize_angle(rel_r),
            discretize_dist(rd),
        )

    def apply_action(self, a: int):
        """Execute one of the three actions."""
        if a == 0:  # A: turn left
            self.robot.heading_rad -= math.radians(TURN_DEG)
        elif a == 2:  # C: turn right
            self.robot.heading_rad += math.radians(TURN_DEG)
        elif a == 1:  # B: forward
            self.robot.x += math.cos(self.robot.heading_rad) * MOVE_PX
            self.robot.y += math.sin(self.robot.heading_rad) * MOVE_PX
            self.robot.x, self.robot.y = wrap_pos(self.robot.x, self.robot.y)

        self.robot.heading_rad = angle_normalize(self.robot.heading_rad)

    def check_collisions_and_reward(self) -> float:
        """Return reward from collisions this step."""
        reward = R_STEP
        rx, ry = self.robot.x, self.robot.y

        for p in self.pills:
            dx, dy = torus_delta(rx, ry, p.x, p.y)
            d = math.hypot(dx, dy)
            if d <= (ROBOT_RADIUS + PILL_RADIUS):
                if p.is_green:
                    reward += R_GREEN
                    self.collected_green += 1
                else:
                    reward += R_RED
                    self.hit_red += 1
                self.respawn_pill(p)
                break  # keep it simple: at most one pill per step

        return reward

    def step(self):
        """One RL interaction step."""
        s = self.get_state()

        # Potential before acting (for potential-based shaping)
        if USE_REWARD_SHAPING:
            _, gd0, _, _ = self.nearest_pill(True)
            _, rd0, _, _ = self.nearest_pill(False)
            phi0 = (-PHI_GREEN_W * (gd0 / MAX_DIST)) + (PHI_RED_W * (rd0 / MAX_DIST))
        a = self.agent.choose_action(s)

        self.apply_action(a)
        r_env = self.check_collisions_and_reward()
        s_next = self.get_state()

        r_train = r_env

        # Potential-based reward shaping: r' = r + scale*(gamma*Phi(s') - Phi(s))
        if USE_REWARD_SHAPING:
            _, gd1, _, _ = self.nearest_pill(True)
            _, rd1, _, _ = self.nearest_pill(False)
            phi1 = (-PHI_GREEN_W * (gd1 / MAX_DIST)) + (PHI_RED_W * (rd1 / MAX_DIST))
            r_train += SHAPING_SCALE * (GAMMA * phi1 - phi0)

        self.agent.update(s, a, r_train, s_next)
        self.agent.decay_epsilon()

        self.steps += 1
        self.score += r_env
        self.last_reward = r_env
        self.last_reward_train = r_train
        self.last_action = a

        # Block average: compute one point per MA_WINDOW steps.
        self._reward_block_sum += r_env
        self._reward_block_count += 1
        if self._reward_block_count >= MA_WINDOW:
            self.reward_ma_1000 = self._reward_block_sum / self._reward_block_count
            self._append_ma_history(self.reward_ma_1000)
            self._reward_block_sum = 0.0
            self._reward_block_count = 0

        if SOFT_RESET_EVERY_STEPS and (self.steps % SOFT_RESET_EVERY_STEPS == 0):
            self.soft_reset()


# =========================
# Visualization (PySide6)
# =========================

class WorldView(QWidget):
    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.setFixedSize(WORLD_W, WORLD_H)
        self.setFocusPolicy(Qt.StrongFocus)

    def _draw_torus_line(self, painter: QPainter, sx: float, sy: float, dx: float, dy: float):
        """Draw the shortest torus vector (sx,sy)->(sx+dx,sy+dy), split at borders if needed."""
        remaining_dx, remaining_dy = dx, dy
        cur_x, cur_y = sx, sy

        # At most 3 segments are ever needed with dx,dy limited to half-world.
        for _ in range(3):
            end_x = cur_x + remaining_dx
            end_y = cur_y + remaining_dy

            if 0.0 <= end_x <= WORLD_W and 0.0 <= end_y <= WORLD_H:
                painter.drawLine(int(cur_x), int(cur_y), int(end_x), int(end_y))
                return

            t_candidates: list[float] = []
            hit_x = False
            hit_y = False

            if remaining_dx != 0.0:
                if end_x < 0.0:
                    t_candidates.append((0.0 - cur_x) / remaining_dx)
                elif end_x > WORLD_W:
                    t_candidates.append((WORLD_W - cur_x) / remaining_dx)

            if remaining_dy != 0.0:
                if end_y < 0.0:
                    t_candidates.append((0.0 - cur_y) / remaining_dy)
                elif end_y > WORLD_H:
                    t_candidates.append((WORLD_H - cur_y) / remaining_dy)

            # Fallback: if we can't find a sensible boundary intersection, draw clipped.
            valid_ts = [t for t in t_candidates if 0.0 < t <= 1.0]
            if not valid_ts:
                painter.drawLine(int(cur_x), int(cur_y), int(end_x), int(end_y))
                return

            t = min(valid_ts)
            ix = cur_x + remaining_dx * t
            iy = cur_y + remaining_dy * t
            painter.drawLine(int(cur_x), int(cur_y), int(ix), int(iy))

            # Determine which boundary(ies) we hit.
            if abs(ix - 0.0) < 1e-6 or abs(ix - WORLD_W) < 1e-6:
                hit_x = True
            if abs(iy - 0.0) < 1e-6 or abs(iy - WORLD_H) < 1e-6:
                hit_y = True

            # Wrap only along hit axes.
            if hit_x:
                cur_x = WORLD_W if ix <= 0.0 else 0.0
            else:
                cur_x = ix
            if hit_y:
                cur_y = WORLD_H if iy <= 0.0 else 0.0
            else:
                cur_y = iy

            remaining_dx *= (1.0 - t)
            remaining_dy *= (1.0 - t)

    def paintEvent(self, _):
        w = self.world
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background
        p.fillRect(0, 0, WORLD_W, WORLD_H, QColor(250, 250, 250))

        # pills
        for pill in w.pills:
            color = QColor(40, 160, 60) if pill.is_green else QColor(200, 50, 50)
            p.setPen(QPen(Qt.NoPen))
            p.setBrush(QBrush(color))
            p.drawEllipse(
                int(pill.x - PILL_RADIUS), int(pill.y - PILL_RADIUS),
                2 * PILL_RADIUS, 2 * PILL_RADIUS
            )

        # robot body
        rx, ry = w.robot.x, w.robot.y
        p.setPen(QPen(QColor(30, 30, 30), 2))
        p.setBrush(QBrush(QColor(235, 235, 235)))
        p.drawEllipse(
            int(rx - ROBOT_RADIUS), int(ry - ROBOT_RADIUS),
            2 * ROBOT_RADIUS, 2 * ROBOT_RADIUS
        )

        # "Sensors": dotted lines to nearest green + nearest red (torus shortest direction)
        try:
            _, _, gdx, gdy = w.nearest_pill(True)
            _, _, rdx, rdy = w.nearest_pill(False)

            green_line = QColor(40, 160, 60, 170)
            red_line = QColor(200, 50, 50, 170)

            p.setBrush(QBrush(Qt.NoBrush))
            p.setPen(QPen(green_line, 2, Qt.DotLine))
            self._draw_torus_line(p, rx, ry, gdx, gdy)

            p.setPen(QPen(red_line, 2, Qt.DotLine))
            self._draw_torus_line(p, rx, ry, rdx, rdy)
        except AssertionError:
            # In the unlikely case pills list is empty, just skip drawing the sensor lines.
            pass

        # robot heading line
        hx = rx + math.cos(w.robot.heading_rad) * (ROBOT_RADIUS + 10)
        hy = ry + math.sin(w.robot.heading_rad) * (ROBOT_RADIUS + 10)
        p.setPen(QPen(QColor(0, 0, 0), 3))
        p.drawLine(int(rx), int(ry), int(hx), int(hy))

        p.end()


class RewardPlot(QWidget):
    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.setFixedSize(260, 120)

    def paintEvent(self, _):
        w = self.world
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background + border
        p.fillRect(0, 0, self.width(), self.height(), QColor(250, 250, 250))
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.setBrush(QBrush(Qt.NoBrush))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)

        values_full = w.reward_ma_history
        if len(values_full) < 2:
            p.end()
            return

        # margins for numeric axis ticks
        pad_left = 28
        pad_right = 8
        pad_top = 8
        pad_bottom = 22
        left = pad_left
        top = pad_top
        right = self.width() - pad_right
        bottom = self.height() - pad_bottom

        def _fmt_steps(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.0f}k"
            return str(n)

        # Numeric tick labels
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.setFont(QFont("Arial", 8))

        # x ticks: 0 .. current steps
        x0 = 0
        x1 = int(w.steps)
        xm = int((x0 + x1) / 2)
        p.drawText(int(left), int(bottom + 6), 1, 14, int(Qt.AlignLeft | Qt.AlignTop), _fmt_steps(x0))
        p.drawText(int((left + right) / 2), int(bottom + 6), 1, 14, int(Qt.AlignHCenter | Qt.AlignTop), _fmt_steps(xm))
        p.drawText(int(right), int(bottom + 6), 1, 14, int(Qt.AlignRight | Qt.AlignTop), _fmt_steps(x1))

        # Downsample for drawing so the whole history always fits.
        # We map the entire stored overview to the available pixel width.
        max_points = max(2, int(right - left))
        if len(values_full) <= max_points:
            values = values_full
        else:
            stride = math.ceil(len(values_full) / max_points)
            values = [
                sum(values_full[i:i + stride]) / len(values_full[i:i + stride])
                for i in range(0, len(values_full), stride)
            ]

        vmin = min(values_full)
        vmax = max(values_full)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        # y ticks: vmin, mid, vmax
        vmid = 0.5 * (vmin + vmax)

        def _y_to_px(v: float) -> float:
            t = (v - vmin) / (vmax - vmin)
            return bottom - t * (bottom - top)

        tick_color = QColor(120, 120, 120)
        p.setPen(QPen(tick_color, 1))
        for val in (vmax, vmid, vmin):
            y = _y_to_px(val)
            # tick mark
            p.drawLine(int(left - 4), int(y), int(left), int(y))
            # label
            p.drawText(0, int(y - 7), int(left - 6), 14, int(Qt.AlignRight | Qt.AlignVCenter), f"{val:.2f}")

        n = len(values)
        xs = right - left
        ys = bottom - top

        pts: list[QPointF] = []
        for i, v in enumerate(values):
            x = left + (i / (n - 1)) * xs
            t = (v - vmin) / (vmax - vmin)
            y = bottom - t * ys
            pts.append(QPointF(x, y))

        p.setPen(QPen(QColor(30, 30, 30), 2))
        p.drawPolyline(QPolygonF(pts))
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Learning: Green vs Red Pills (PySide6)")

        self.world = World()
        self.view = WorldView(self.world)
        self.reward_plot = RewardPlot(self.world)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.setInterval(10)  # ms

        # Space-hold stepping timer (press Space = start, release = stop)
        self.space_timer = QTimer(self)
        self.space_timer.timeout.connect(self.single_step)
        self.space_timer.setInterval(SPACE_STEP_INTERVAL_MS)

        # controls
        self.btn_run = QPushButton("Run")
        self.btn_step = QPushButton("Step")
        self.btn_reset = QPushButton("Reset")
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_step.clicked.connect(self.single_step)
        self.btn_reset.clicked.connect(self.reset_world)

        # status panel (moved from overlay to right side)
        self.lbl_status = QLabel()
        self.lbl_status.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_status.setFont(QFont("Arial", 10))
        self.lbl_status.setMinimumHeight(140)

        self.lbl_plot_title = QLabel("Avg reward per 1000 steps")
        self.lbl_plot_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # teaching notes
        self.lbl_hint = QLabel(
            "Teaching notes:\n"
            "- Q-table is a dict: state -> [Q(s,A), Q(s,B), Q(s,C)]\n"
            "- Find 'Q-LEARNING UPDATE RULE' in the code.\n"
            "- World wraps around (torus), no borders."
        )
        self.lbl_hint.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lbl_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # layout
        controls = QVBoxLayout()
        controls.addWidget(self.btn_run)
        controls.addWidget(self.btn_step)
        controls.addWidget(self.btn_reset)
        controls.addSpacing(10)
        controls.addWidget(self.lbl_status)
        controls.addSpacing(6)
        controls.addWidget(self.lbl_plot_title)
        controls.addWidget(self.reward_plot)
        controls.addSpacing(10)
        controls.addWidget(self.lbl_hint, 1)

        root = QHBoxLayout()
        root.addWidget(self.view)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # Capture Space press/release even if focus is on a button/label.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.update_status_panel()

    def update_status_panel(self):
        w = self.world
        action_name = ACTIONS[w.last_action] if w.last_action is not None else "-"

        self.lbl_status.setText(
            f"steps: {w.steps}\n"
            f"epsilon: {w.agent.eps:.3f}\n"
            f"score (sum r): {w.score:.2f}\n"
            f"ma1000(r): {w.reward_ma_1000:.3f}\n\n"
            f"last action: {action_name}\n"
            f"last reward: {w.last_reward:.2f}\n\n"
            f"green collected: {w.collected_green}\n"
            f"red hit: {w.hit_red}"
        )

    def on_tick(self):
        # do multiple steps per visual frame for faster learning
        for _ in range(STEPS_PER_TICK):
            self.world.step()
        self.update_status_panel()
        self.view.update()
        self.reward_plot.update()

    def toggle_run(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Run")
        else:
            # If the user is holding Space, stop hold-to-step when switching to Run.
            self.space_timer.stop()
            self.timer.start()
            self.btn_run.setText("Pause")

    def eventFilter(self, obj, event):
        # Hold-to-step with Space: press = start stepping, release = stop.
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat() and (not self.timer.isActive()):
                # Do one immediate step for responsiveness, then continue on timer.
                self.single_step()
                if not self.space_timer.isActive():
                    self.space_timer.start()
            return True

        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Space:
            if not event.isAutoRepeat():
                self.space_timer.stop()
            return True

        return super().eventFilter(obj, event)

    def single_step(self):
        if self.timer.isActive():
            return
        self.world.step()
        self.update_status_panel()
        self.view.update()
        self.reward_plot.update()

    def reset_world(self):
        was_running = self.timer.isActive()
        self.timer.stop()
        self.space_timer.stop()

        self.world = World()
        self.view.world = self.world
        self.reward_plot.world = self.world

        self.view.update()
        self.update_status_panel()
        self.reward_plot.update()

        self.btn_run.setText("Run")
        if was_running:
            self.timer.start()
            self.btn_run.setText("Pause")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
