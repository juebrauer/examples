#!/usr/bin/env python3
"""Model-based RL demo: Value Iteration for a simple 2-DOF robot arm.

State space:
- s = (bin_dof1, bin_dof2)
- each DOF uses 10-degree bins -> 36 bins per DOF -> 1296 states total

Action space:
- For each DOF: Left or Right (next bin with wrap-around)
- Combined actions: (L,L), (L,R), (R,L), (R,R)

Transition model:
- Deterministic: T(s, a, s') = 1 for exactly one successor state, else 0.

Reward model:
- r(s) is larger when end-effector is closer to the target X.
- target is sampled from a reachable end-effector position.

Visualization:
- PySide6 canvas with arm, target (large orange X), and end effector (black circle).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

N_BINS = 36
BIN_DEG = 10.0
ACTIONS: Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
ACTION_NAMES = {
    (-1, -1): "L, L",
    (-1, +1): "L, R",
    (+1, -1): "R, L",
    (+1, +1): "R, R",
}

State = Tuple[int, int]
Action = Tuple[int, int]


@dataclass
class ArmKinematics:
    link1: float = 130.0
    link2: float = 110.0

    def forward(self, s: State) -> Tuple[float, float]:
        b1, b2 = s
        theta1 = math.radians(b1 * BIN_DEG)
        theta2 = math.radians(b2 * BIN_DEG)

        x1 = self.link1 * math.cos(theta1)
        y1 = self.link1 * math.sin(theta1)

        x2 = x1 + self.link2 * math.cos(theta1 + theta2)
        y2 = y1 + self.link2 * math.sin(theta1 + theta2)
        return x2, y2

    def joint1(self, s: State) -> Tuple[float, float]:
        b1, _ = s
        theta1 = math.radians(b1 * BIN_DEG)
        return self.link1 * math.cos(theta1), self.link1 * math.sin(theta1)

    @property
    def max_reach(self) -> float:
        return self.link1 + self.link2


class RobotArmMDP:
    """Deterministic model for the 2-DOF arm in discretized angle space."""

    def __init__(self, kinematics: ArmKinematics) -> None:
        self.kin = kinematics
        self.states: List[State] = [(i, j) for i in range(N_BINS) for j in range(N_BINS)]
        self.target_state: State = (0, 0)
        self.target_xy: Tuple[float, float] = (0.0, 0.0)
        self.goal_radius: float = 14.0

        self._ee_cache: Dict[State, Tuple[float, float]] = {
            s: self.kin.forward(s) for s in self.states
        }
        self._reward_cache: Dict[State, float] = {}
        self.goal_states: set[State] = set()

        self.sample_reachable_target()

    def sample_reachable_target(self) -> None:
        # Pick a reachable target by selecting a random arm state and using its EE point.
        self.target_state = random.choice(self.states)
        self.target_xy = self._ee_cache[self.target_state]
        self._rebuild_reward_model()

    def _rebuild_reward_model(self) -> None:
        tx, ty = self.target_xy
        scale = self.kin.max_reach

        self._reward_cache.clear()
        self.goal_states.clear()

        for s in self.states:
            ex, ey = self._ee_cache[s]
            dist = math.hypot(ex - tx, ey - ty)

            # Smoothly increasing reward as distance gets smaller.
            shaped_reward = math.exp(-dist / max(1e-6, scale))

            if dist <= self.goal_radius:
                self.goal_states.add(s)
                self._reward_cache[s] = 3.0 + shaped_reward
            else:
                self._reward_cache[s] = shaped_reward - 0.05

    def reward(self, s: State) -> float:
        return self._reward_cache[s]

    def next_state(self, s: State, a: Action) -> State:
        if s in self.goal_states:
            # Absorbing goal for proper episodic planning.
            return s

        b1, b2 = s
        d1, d2 = a
        return (b1 + d1) % N_BINS, (b2 + d2) % N_BINS

    def transition_prob(self, s: State, a: Action, s_next: State) -> float:
        return 1.0 if self.next_state(s, a) == s_next else 0.0


class ValueIterationPlanner:
    def __init__(self, mdp: RobotArmMDP, gamma: float = 0.96) -> None:
        self.mdp = mdp
        self.gamma = gamma
        self.values: Dict[State, float] = {s: 0.0 for s in self.mdp.states}
        self.policy: Dict[State, Action] = {s: ACTIONS[0] for s in self.mdp.states}

    def reset(self) -> None:
        self.values = {s: 0.0 for s in self.mdp.states}
        self.policy = {s: ACTIONS[0] for s in self.mdp.states}

    def solve(self, theta: float = 1e-5, max_iterations: int = 400) -> Tuple[int, float]:
        last_delta = 0.0
        for i in range(max_iterations):
            delta = 0.0
            new_values = dict(self.values)

            for s in self.mdp.states:
                r = self.mdp.reward(s)
                if s in self.mdp.goal_states:
                    new_values[s] = r
                    continue

                best_q = -float("inf")
                best_a = ACTIONS[0]

                for a in ACTIONS:
                    s_next = self.mdp.next_state(s, a)
                    q = r + self.gamma * self.values[s_next]
                    if q > best_q:
                        best_q = q
                        best_a = a

                new_values[s] = best_q
                self.policy[s] = best_a
                delta = max(delta, abs(new_values[s] - self.values[s]))

            self.values = new_values
            last_delta = delta
            if delta < theta:
                return i + 1, last_delta

        return max_iterations, last_delta


class RobotArmCanvas(QWidget):
    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.setMinimumSize(640, 640)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.fillRect(self.rect(), QColor("#f6f8fb"))

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        self._draw_workspace(p, cx, cy)
        self._draw_target(p, cx, cy)
        self._draw_arm(p, cx, cy)

    def _to_screen(self, x: float, y: float, cx: float, cy: float) -> Tuple[float, float]:
        # World +y is up, screen +y is down.
        return cx + x, cy - y

    def _draw_workspace(self, p: QPainter, cx: float, cy: float) -> None:
        radius = self.parent_window.kin.max_reach

        p.setPen(QPen(QColor("#d7dde8"), 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius))

        p.setPen(QPen(QColor("#777777"), 2))
        p.drawLine(int(cx - 15), int(cy), int(cx + 15), int(cy))
        p.drawLine(int(cx), int(cy - 15), int(cx), int(cy + 15))

    def _draw_target(self, p: QPainter, cx: float, cy: float) -> None:
        tx, ty = self.parent_window.mdp.target_xy
        sx, sy = self._to_screen(tx, ty, cx, cy)

        p.setPen(QPen(QColor("#ff8c00"), 7))
        size = 16
        p.drawLine(int(sx - size), int(sy - size), int(sx + size), int(sy + size))
        p.drawLine(int(sx - size), int(sy + size), int(sx + size), int(sy - size))

        p.setPen(QPen(QColor("#ffad42"), 1, Qt.PenStyle.DashLine))
        goal_r = self.parent_window.mdp.goal_radius
        p.drawEllipse(int(sx - goal_r), int(sy - goal_r), int(2 * goal_r), int(2 * goal_r))

    def _draw_arm(self, p: QPainter, cx: float, cy: float) -> None:
        s = self.parent_window.current_state
        jx, jy = self.parent_window.kin.joint1(s)
        ex, ey = self.parent_window.kin.forward(s)

        bx, by = cx, cy
        jxs, jys = self._to_screen(jx, jy, cx, cy)
        exs, eys = self._to_screen(ex, ey, cx, cy)

        p.setPen(QPen(QColor("#4b6bfb"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(int(bx), int(by), int(jxs), int(jys))

        p.setPen(QPen(QColor("#06a77d"), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(int(jxs), int(jys), int(exs), int(eys))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1f1f1f"))
        p.drawEllipse(int(bx - 8), int(by - 8), 16, 16)
        p.drawEllipse(int(jxs - 6), int(jys - 6), 12, 12)

        p.setBrush(QColor("#000000"))
        p.drawEllipse(int(exs - 7), int(eys - 7), 14, 14)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Value Iteration Demo - 2-DOF Robot Arm")

        self.kin = ArmKinematics()
        self.mdp = RobotArmMDP(self.kin)
        self.planner = ValueIterationPlanner(self.mdp)

        self.current_state: State = self._sample_non_goal_state()
        self.step_count = 0

        self.timer = QTimer(self)
        self.timer.setInterval(220)
        self.timer.timeout.connect(self._policy_step)

        self.canvas = RobotArmCanvas(self)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.50, 0.999)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setValue(0.96)

        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setDecimals(7)
        self.theta_spin.setRange(1e-8, 1e-2)
        self.theta_spin.setSingleStep(1e-5)
        self.theta_spin.setValue(1e-5)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 2000)
        self.max_iter_spin.setValue(400)

        self.solve_btn = QPushButton("Run Value Iteration")
        self.solve_btn.clicked.connect(self.run_value_iteration)

        self.run_btn = QPushButton("Run Policy")
        self.run_btn.clicked.connect(self.toggle_run_policy)

        self.reset_btn = QPushButton("New Random Start + Target")
        self.reset_btn.clicked.connect(self.reset_problem)

        self.status_lbl = QLabel()
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setFont(QFont("DejaVu Sans", 10))

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("gamma:"))
        controls_row.addWidget(self.gamma_spin)
        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("theta:"))
        controls_row.addWidget(self.theta_spin)
        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("max iter:"))
        controls_row.addWidget(self.max_iter_spin)
        controls_row.addSpacing(20)
        controls_row.addWidget(self.solve_btn)
        controls_row.addWidget(self.run_btn)
        controls_row.addWidget(self.reset_btn)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(controls_row)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.status_lbl)

        self.setCentralWidget(root)
        self.resize(860, 920)

        self.run_value_iteration()
        self._update_status("Initialized.")

    def _sample_non_goal_state(self) -> State:
        while True:
            s = (random.randrange(N_BINS), random.randrange(N_BINS))
            if s not in self.mdp.goal_states:
                return s

    def _distance_to_target(self, s: State) -> float:
        ex, ey = self.kin.forward(s)
        tx, ty = self.mdp.target_xy
        return math.hypot(ex - tx, ey - ty)

    def _update_status(self, message: str) -> None:
        b1, b2 = self.current_state
        action = self.planner.policy.get(self.current_state, ACTIONS[0])
        d = self._distance_to_target(self.current_state)

        txt = (
            f"{message}\n"
            f"Current state s=(DOF1_bin={b1}, DOF2_bin={b2}) | "
            f"suggested action a=({ACTION_NAMES[action]}) | "
            f"distance to X={d:.2f} px | steps={self.step_count}\n"
            f"Target state (one exact reachable solution) = {self.mdp.target_state}, "
            f"goal states within radius={len(self.mdp.goal_states)}"
        )
        self.status_lbl.setText(txt)

    def run_value_iteration(self) -> None:
        self.planner.gamma = float(self.gamma_spin.value())
        iterations, delta = self.planner.solve(
            theta=float(self.theta_spin.value()),
            max_iterations=int(self.max_iter_spin.value()),
        )
        self.canvas.update()
        self._update_status(
            f"Value Iteration done: iterations={iterations}, final_delta={delta:.6e}."
        )

    def toggle_run_policy(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("Run Policy")
            self._update_status("Policy execution paused.")
            return

        self.timer.start()
        self.run_btn.setText("Pause Policy")
        self._update_status("Policy execution started.")

    def _policy_step(self) -> None:
        if self.current_state in self.mdp.goal_states:
            self.timer.stop()
            self.run_btn.setText("Run Policy")
            self._update_status("Goal reached. End-effector is at the target region.")
            self.canvas.update()
            return

        action = self.planner.policy.get(self.current_state, ACTIONS[0])
        self.current_state = self.mdp.next_state(self.current_state, action)
        self.step_count += 1

        if self.step_count >= 300:
            self.timer.stop()
            self.run_btn.setText("Run Policy")
            self._update_status("Stopped after 300 steps (safety limit).")
        else:
            self._update_status("Policy step executed.")

        self.canvas.update()

    def reset_problem(self) -> None:
        self.timer.stop()
        self.run_btn.setText("Run Policy")

        self.mdp.sample_reachable_target()
        self.planner.reset()
        self.current_state = self._sample_non_goal_state()
        self.step_count = 0

        self.run_value_iteration()
        self.canvas.update()
        self._update_status("New random reachable target and random start sampled.")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
