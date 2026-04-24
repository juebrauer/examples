#!/usr/bin/env python3
"""Value and Policy Iteration demo (GridWorld) with a simple PySide6 UI.

Teaching-oriented goals:
- Small, editable GridWorld with obstacles, per-cell rewards, and two terminal cells.
- Deterministic actions (up/down/left/right) with wall/obstacle collisions causing no movement.
- Value iteration computes optimal state values V*(s) and overlays them on the grid.

Reward semantics used here:
- Each move yields the reward of the *resulting* cell.
- Entering a terminal cell yields its terminal reward and then the episode ends (no discounted future).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

Cell = Tuple[int, int]  # (x, y)

ACTIONS: Tuple[str, ...] = ("U", "D", "L", "R")


@dataclass
class GridWorld:
    width: int
    height: int
    default_reward: float = -0.04
    rewards: List[List[float]] = field(init=False)
    obstacles: Set[Cell] = field(default_factory=set)
    terminals: Dict[Cell, float] = field(default_factory=dict)  # terminal cell -> terminal reward

    def __post_init__(self) -> None:
        self.rewards = [
            [float(self.default_reward) for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, cell: Cell) -> bool:
        return cell in self.obstacles

    def is_terminal(self, cell: Cell) -> bool:
        return cell in self.terminals

    def states(self) -> Iterable[Cell]:
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if cell in self.obstacles:
                    continue
                yield cell

    def get_reward(self, cell: Cell) -> float:
        x, y = cell
        if not self.in_bounds(cell):
            return 0.0
        return float(self.rewards[y][x])

    def set_reward(self, cell: Cell, reward: float) -> None:
        if not self.in_bounds(cell):
            return
        if self.is_obstacle(cell):
            return
        x, y = cell
        self.rewards[y][x] = float(reward)
        if self.is_terminal(cell):
            self.terminals[cell] = float(reward)

    def set_obstacle(self, cell: Cell, is_obstacle: bool) -> None:
        if not self.in_bounds(cell):
            return
        if is_obstacle:
            self.terminals.pop(cell, None)
            self.obstacles.add(cell)
        else:
            self.obstacles.discard(cell)

    def clear_cell(self, cell: Cell) -> None:
        if not self.in_bounds(cell):
            return
        self.obstacles.discard(cell)
        self.terminals.pop(cell, None)
        self.set_reward(cell, self.default_reward)

    def set_terminal(self, cell: Cell, reward: float, *, kind: str) -> None:
        """Set a terminal cell.

        kind is used to enforce at most one "+" and one "-" terminal for the teaching demo.
        """
        if not self.in_bounds(cell):
            return
        if self.is_obstacle(cell):
            return

        if kind not in {"pos", "neg"}:
            raise ValueError("kind must be 'pos' or 'neg'")

        # Enforce at most one terminal per kind by removing the previous one.
        to_remove: List[Cell] = []
        for t_cell, t_reward in self.terminals.items():
            if kind == "pos" and t_reward > 0:
                to_remove.append(t_cell)
            if kind == "neg" and t_reward < 0:
                to_remove.append(t_cell)
        for t_cell in to_remove:
            self.terminals.pop(t_cell, None)
            self.set_reward(t_cell, self.default_reward)

        self.terminals[cell] = float(reward)
        self.set_reward(cell, reward)

    def next_cell(self, cell: Cell, action: str) -> Cell:
        x, y = cell
        if action == "U":
            candidate = (x, y - 1)
        elif action == "D":
            candidate = (x, y + 1)
        elif action == "L":
            candidate = (x - 1, y)
        elif action == "R":
            candidate = (x + 1, y)
        else:
            raise ValueError(f"Unknown action: {action}")

        if not self.in_bounds(candidate) or self.is_obstacle(candidate):
            return cell
        return candidate

    def step(self, state: Cell, action: str) -> Tuple[Cell, float, bool]:
        if self.is_terminal(state):
            return state, 0.0, True

        next_state = self.next_cell(state, action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done


def _bellman_optimality_sweep(
    world: GridWorld,
    V: Dict[Cell, float],
    *,
    gamma: float,
) -> Tuple[Dict[Cell, float], float]:
    """Run one Bellman optimality sweep over all states.

    Returns (V_new, delta) where delta is the max absolute change.
    """
    V_new = dict(V)
    delta = 0.0

    for s in world.states():
        if world.is_terminal(s):
            V_new[s] = float(world.terminals[s])
            continue

        _, best_q = _optimal_action_and_q(world, V, s, gamma=gamma)
        V_new[s] = best_q
        delta = max(delta, abs(V_new[s] - V[s]))

    return V_new, delta


def _policy_evaluation_sweep(
    world: GridWorld,
    V: Dict[Cell, float],
    policy: Dict[Cell, str],
    *,
    gamma: float,
) -> Tuple[Dict[Cell, float], float]:
    """Run one Bellman expectation sweep for a fixed deterministic policy."""
    V_new = dict(V)
    delta = 0.0

    for s in world.states():
        if world.is_terminal(s):
            V_new[s] = float(world.terminals[s])
            continue

        action = policy.get(s, ACTIONS[0])
        s2, r, done = world.step(s, action)
        V_new[s] = float(r) + (0.0 if done else gamma * float(V[s2]))
        delta = max(delta, abs(V_new[s] - V[s]))

    return V_new, delta


def _optimal_action_and_q(
    world: GridWorld,
    V: Dict[Cell, float],
    state: Cell,
    *,
    gamma: float,
) -> Tuple[str, float]:
    """Return (best_action, best_q) for a given state under greedy control.

    Uses the same reward/terminal semantics as the value iteration sweep.
    """
    best_action = ACTIONS[0]
    best_q = -math.inf

    for action in ACTIONS:
        s2, r, done = world.step(state, action)
        q = float(r) + (0.0 if done else gamma * float(V[s2]))
        if q > best_q:
            best_q = q
            best_action = action

    return best_action, best_q


def greedy_policy_from_values(
    world: GridWorld,
    values: Dict[Cell, float],
    *,
    gamma: float,
) -> Dict[Cell, str]:
    """Compute a greedy deterministic policy π(s) from state values V(s)."""
    policy: Dict[Cell, str] = {}
    for s in world.states():
        if world.is_terminal(s):
            continue
        a, _ = _optimal_action_and_q(world, values, s, gamma=gamma)
        policy[s] = a
    return policy


def value_iteration(
    world: GridWorld,
    *,
    gamma: float,
    theta: float = 1e-4,
    max_iterations: int = 200,
) -> Tuple[Dict[Cell, float], int, float]:
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")

    V: Dict[Cell, float] = {s: 0.0 for s in world.states()}
    for t, r in world.terminals.items():
        # Note: initializing terminal values to 0 is fine; the sweep will clamp
        # terminal states to their terminal reward anyway.
        V[t] = 0.0

    last_delta = 0.0
    for i in range(max_iterations):
        V, delta = _bellman_optimality_sweep(world, V, gamma=gamma)
        last_delta = delta
        if delta < theta:
            return V, i + 1, last_delta

    return V, max_iterations, last_delta


def policy_iteration(
    world: GridWorld,
    *,
    gamma: float,
    theta: float = 1e-4,
    max_iterations: int = 200,
) -> Tuple[Dict[Cell, float], Dict[Cell, str], int, int, float, bool]:
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1)")

    V: Dict[Cell, float] = {s: 0.0 for s in world.states()}
    for t, r in world.terminals.items():
        V[t] = float(r)

    policy: Dict[Cell, str] = {
        s: ACTIONS[0] for s in world.states() if not world.is_terminal(s)
    }

    total_eval_sweeps = 0
    last_delta = math.inf

    for i in range(max_iterations):
        # Full policy evaluation: sweep until V converges for the current policy.
        eval_sweeps = 0
        while eval_sweeps < max_iterations:
            V, delta = _policy_evaluation_sweep(world, V, policy, gamma=gamma)
            eval_sweeps += 1
            total_eval_sweeps += 1
            last_delta = delta
            if delta < theta:
                break

        policy_stable = True
        improved_policy: Dict[Cell, str] = {}
        for s in world.states():
            if world.is_terminal(s):
                continue
            best_action, _ = _optimal_action_and_q(world, V, s, gamma=gamma)
            improved_policy[s] = best_action
            if best_action != policy.get(s, ACTIONS[0]):
                policy_stable = False

        policy = improved_policy
        if policy_stable:
            return V, policy, i + 1, total_eval_sweeps, last_delta, True

    return V, policy, max_iterations, total_eval_sweeps, last_delta, False


class ValueIterationSession:
    """Incremental value iteration (one full sweep per step).

    This is used for teaching: pressing Space runs exactly one sweep
    over all states and updates the currently displayed V(s).
    """

    def __init__(
        self,
        world: GridWorld,
        *,
        gamma: float,
    ) -> None:
        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        self.world = world
        self.gamma = float(gamma)

        self.V: Dict[Cell, float] = {s: 0.0 for s in world.states()}
        for t, r in world.terminals.items():
            self.V[t] = float(r)

        self.iteration = 0
        self.last_delta = math.inf

    def step(self) -> Tuple[Dict[Cell, float], float, int]:
        """Perform one sweep and return (V, delta, iteration_count)."""
        self.V, delta = _bellman_optimality_sweep(self.world, self.V, gamma=self.gamma)
        self.iteration += 1
        self.last_delta = delta
        return dict(self.V), delta, self.iteration


class PolicyIterationSession:
    """Incremental policy iteration (one evaluate/improve round per step)."""

    def __init__(
        self,
        world: GridWorld,
        *,
        gamma: float,
        theta: float,
    ) -> None:
        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be in [0, 1)")
        self.world = world
        self.gamma = float(gamma)
        self.theta = float(theta)

        self.V: Dict[Cell, float] = {s: 0.0 for s in world.states()}
        for t, r in world.terminals.items():
            self.V[t] = float(r)

        self.policy: Dict[Cell, str] = {
            s: ACTIONS[0] for s in world.states() if not world.is_terminal(s)
        }
        self.iteration = 0
        self.last_eval_delta = math.inf
        self.last_eval_sweeps = 0
        self.policy_stable = False

    def step(
        self,
        *,
        max_eval_iterations: int,
    ) -> Tuple[Dict[Cell, float], Dict[Cell, str], float, int, int, bool]:
        """Perform full policy evaluation (until Δ < theta) then policy improvement."""
        # Full policy evaluation: sweep until V converges for the current policy.
        delta = math.inf
        eval_sweeps = 0
        while eval_sweeps < max_eval_iterations:
            self.V, delta = _policy_evaluation_sweep(
                self.world,
                self.V,
                self.policy,
                gamma=self.gamma,
            )
            eval_sweeps += 1
            if delta < self.theta:
                break

        improved_policy: Dict[Cell, str] = {}
        policy_stable = True
        for s in self.world.states():
            if self.world.is_terminal(s):
                continue
            best_action, _ = _optimal_action_and_q(self.world, self.V, s, gamma=self.gamma)
            improved_policy[s] = best_action
            if best_action != self.policy.get(s, ACTIONS[0]):
                policy_stable = False

        self.policy = improved_policy
        self.iteration += 1
        self.last_eval_delta = delta
        self.last_eval_sweeps = eval_sweeps
        self.policy_stable = policy_stable
        return (
            dict(self.V),
            dict(self.policy),
            delta,
            eval_sweeps,
            self.iteration,
            policy_stable,
        )


class GridWorldWidget(QWidget):
    worldEdited = Signal()

    def __init__(self, world: GridWorld, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._world = world
        self._mode: str = "reward"  # reward|obstacle|terminal_pos|terminal_neg|clear
        self._paint_reward: float = 0.0
        self._terminal_pos_reward: float = 10.0
        self._terminal_neg_reward: float = -10.0

        self._vi_values: Dict[Cell, float] = {}
        self._pi_values: Dict[Cell, float] = {}
        self._vi_policy: Dict[Cell, str] = {}
        self._pi_policy: Dict[Cell, str] = {}
        self._show_comparison = False

        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)

    def set_world(self, world: GridWorld) -> None:
        self._world = world
        self._vi_values = {}
        self._pi_values = {}
        self._vi_policy = {}
        self._pi_policy = {}
        self._show_comparison = False
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_paint_reward(self, reward: float) -> None:
        self._paint_reward = float(reward)

    def set_terminal_pos_reward(self, reward: float) -> None:
        self._terminal_pos_reward = float(reward)

    def set_terminal_neg_reward(self, reward: float) -> None:
        self._terminal_neg_reward = float(reward)

    def set_value_comparison(
        self,
        vi_values: Dict[Cell, float],
        pi_values: Dict[Cell, float],
    ) -> None:
        self._vi_values = dict(vi_values)
        self._pi_values = dict(pi_values)
        self._show_comparison = True
        self.update()

    def set_policy_comparison(
        self,
        vi_policy: Dict[Cell, str],
        pi_policy: Dict[Cell, str],
    ) -> None:
        self._vi_policy = dict(vi_policy)
        self._pi_policy = dict(pi_policy)
        self._show_comparison = True
        self.update()

    def clear_values(self) -> None:
        self._vi_values = {}
        self._pi_values = {}
        self._vi_policy = {}
        self._pi_policy = {}
        self._show_comparison = False
        self.update()

    def _draw_action_arrow(self, painter: QPainter, rect: QRectF, action: str) -> None:
        cx = rect.center().x()
        cy = rect.center().y()

        if action == "U":
            dx, dy = 0.0, -1.0
        elif action == "D":
            dx, dy = 0.0, 1.0
        elif action == "L":
            dx, dy = -1.0, 0.0
        elif action == "R":
            dx, dy = 1.0, 0.0
        else:
            return

        length = min(rect.width(), rect.height()) * 0.66
        start_x = cx - dx * (length * 0.5)
        start_y = cy - dy * (length * 0.5)
        end_x = cx + dx * (length * 0.5)
        end_y = cy + dy * (length * 0.5)

        painter.drawLine(start_x, start_y, end_x, end_y)

        # Arrow head
        head_len = length * 0.36
        angle = math.radians(28.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Rotate (-dx, -dy) by ±angle
        bx, by = -dx, -dy
        lx = bx * cos_a - by * sin_a
        ly = bx * sin_a + by * cos_a
        rx = bx * cos_a + by * sin_a
        ry = -bx * sin_a + by * cos_a

        painter.drawLine(end_x, end_y, end_x + lx * head_len, end_y + ly * head_len)
        painter.drawLine(end_x, end_y, end_x + rx * head_len, end_y + ry * head_len)

    def _grid_geometry(self) -> Tuple[QRectF, float, float]:
        margin = 12.0
        w = max(1, self._world.width)
        h = max(1, self._world.height)

        available = QRectF(
            margin,
            margin,
            max(1.0, float(self.width()) - 2 * margin),
            max(1.0, float(self.height()) - 2 * margin),
        )
        cell_size = min(available.width() / w, available.height() / h)
        grid_w = cell_size * w
        grid_h = cell_size * h
        grid_rect = QRectF(
            available.left() + (available.width() - grid_w) / 2.0,
            available.top() + (available.height() - grid_h) / 2.0,
            grid_w,
            grid_h,
        )
        return grid_rect, cell_size, margin

    def _cell_at(self, pos: QPoint) -> Optional[Cell]:
        grid_rect, cell_size, _ = self._grid_geometry()
        if not grid_rect.contains(pos):
            return None
        x = int((pos.x() - grid_rect.left()) / cell_size)
        y = int((pos.y() - grid_rect.top()) / cell_size)
        cell = (x, y)
        if not self._world.in_bounds(cell):
            return None
        return cell

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton:
            return
        cell = self._cell_at(event.position().toPoint())
        if cell is None:
            return

        if self._mode == "obstacle":
            self._world.set_obstacle(cell, is_obstacle=not self._world.is_obstacle(cell))
            self.clear_values()
        elif self._mode == "reward":
            self._world.set_reward(cell, self._paint_reward)
            self.clear_values()
        elif self._mode == "terminal_pos":
            self._world.set_terminal(cell, self._terminal_pos_reward, kind="pos")
            self.clear_values()
        elif self._mode == "terminal_neg":
            self._world.set_terminal(cell, self._terminal_neg_reward, kind="neg")
            self.clear_values()
        elif self._mode == "clear":
            self._world.clear_cell(cell)
            self.clear_values()

        self.worldEdited.emit()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        grid_rect, cell_size, _ = self._grid_geometry()

        # Draw cells
        for y in range(self._world.height):
            for x in range(self._world.width):
                cell = (x, y)
                rect = QRectF(
                    grid_rect.left() + x * cell_size,
                    grid_rect.top() + y * cell_size,
                    cell_size,
                    cell_size,
                )

                if self._world.is_obstacle(cell):
                    fill = QColor(80, 80, 80)
                elif self._world.is_terminal(cell):
                    r = self._world.terminals[cell]
                    fill = QColor(60, 160, 60) if r >= 0 else QColor(180, 60, 60)
                else:
                    fill = QColor(245, 245, 245)

                painter.fillRect(rect, fill)

                # Text overlays: reward and value
                if not self._world.is_obstacle(cell):
                    reward = self._world.get_reward(cell)
                    vi_value = self._vi_values.get(cell)
                    pi_value = self._pi_values.get(cell)

                    painter.setPen(QPen(QColor(20, 20, 20)))
                    font = QFont(self.font())

                    # Reward small, top-left
                    font.setPointSize(max(7, int(cell_size * 0.12)))
                    painter.setFont(font)
                    painter.drawText(
                        rect.adjusted(4, 2, -2, -2),
                        Qt.AlignTop | Qt.AlignLeft,
                        f"R: {reward:g}",
                    )

                    # Center shows V(s) from VI / Vpi(s) from PI.
                    if self._show_comparison and vi_value is not None and pi_value is not None:
                        font.setPointSize(max(8, int(cell_size * 0.14)))
                        font.setBold(True)
                        painter.setFont(font)
                        painter.drawText(
                            rect,
                            Qt.AlignCenter,
                            f"{vi_value:.2f} / {pi_value:.2f}",
                        )

                    if self._show_comparison and (not self._world.is_terminal(cell)):
                        vi_action = self._vi_policy.get(cell)
                        if vi_action is not None:
                            pen_arrow = QPen(QColor(60, 160, 60))
                            pen_arrow.setWidth(max(2, int(cell_size * 0.045)))
                            painter.setPen(pen_arrow)
                            vi_arrow_rect = rect.adjusted(
                                rect.width() * 0.08,
                                rect.height() * 0.64,
                                -rect.width() * 0.54,
                                -rect.height() * 0.06,
                            )
                            self._draw_action_arrow(painter, vi_arrow_rect, vi_action)

                        pi_action = self._pi_policy.get(cell)
                        if pi_action is not None:
                            pen_arrow = QPen(QColor(60, 90, 190))
                            pen_arrow.setWidth(max(2, int(cell_size * 0.045)))
                            painter.setPen(pen_arrow)
                            pi_arrow_rect = rect.adjusted(
                                rect.width() * 0.54,
                                rect.height() * 0.64,
                                -rect.width() * 0.08,
                                -rect.height() * 0.06,
                            )
                            self._draw_action_arrow(painter, pi_arrow_rect, pi_action)

        # Draw grid lines
        pen = QPen(QColor(30, 30, 30))
        pen.setWidth(1)
        painter.setPen(pen)
        for x in range(self._world.width + 1):
            x_pos = grid_rect.left() + x * cell_size
            painter.drawLine(x_pos, grid_rect.top(), x_pos, grid_rect.bottom())
        for y in range(self._world.height + 1):
            y_pos = grid_rect.top() + y * cell_size
            painter.drawLine(grid_rect.left(), y_pos, grid_rect.right(), y_pos)


class ControlsPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(2, 20)
        self.width_spin.setValue(6)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(2, 20)
        self.height_spin.setValue(6)

        self.apply_size_btn = QPushButton("Apply size")
        self.reset_demo_btn = QPushButton("Reset to demo")

        # Painting tools
        self.mode_group = QButtonGroup(self)
        self.rb_reward = QRadioButton("Paint reward")
        self.rb_obstacle = QRadioButton("Toggle obstacle")
        self.rb_terminal_pos = QRadioButton("Set terminal +")
        self.rb_terminal_neg = QRadioButton("Set terminal -")
        self.rb_clear = QRadioButton("Clear cell")
        self.rb_reward.setChecked(True)

        for rb in (
            self.rb_reward,
            self.rb_obstacle,
            self.rb_terminal_pos,
            self.rb_terminal_neg,
            self.rb_clear,
        ):
            self.mode_group.addButton(rb)

        self.reward_spin = QDoubleSpinBox()
        self.reward_spin.setDecimals(3)
        self.reward_spin.setRange(-1000.0, 1000.0)
        self.reward_spin.setSingleStep(0.1)
        self.reward_spin.setValue(-0.04)

        self.terminal_pos_spin = QDoubleSpinBox()
        self.terminal_pos_spin.setDecimals(3)
        self.terminal_pos_spin.setRange(0.0, 1000.0)
        self.terminal_pos_spin.setSingleStep(1.0)
        self.terminal_pos_spin.setValue(10.0)

        self.terminal_neg_spin = QDoubleSpinBox()
        self.terminal_neg_spin.setDecimals(3)
        self.terminal_neg_spin.setRange(-1000.0, 0.0)
        self.terminal_neg_spin.setSingleStep(1.0)
        self.terminal_neg_spin.setValue(-10.0)

        # Value / policy iteration params
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setDecimals(3)
        self.gamma_spin.setRange(0.0, 0.999)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.95)

        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setDecimals(6)
        self.theta_spin.setRange(1e-8, 1.0)
        self.theta_spin.setSingleStep(1e-4)
        self.theta_spin.setValue(1e-4)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 5000)
        self.max_iter_spin.setValue(200)

        self.run_vi_btn = QPushButton("Run VI and PI")
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        size_box = QGroupBox("Grid size")
        size_form = QFormLayout(size_box)
        size_form.addRow("Width", self.width_spin)
        size_form.addRow("Height", self.height_spin)
        size_form.addRow(self.apply_size_btn)
        size_form.addRow(self.reset_demo_btn)

        paint_box = QGroupBox("Edit world")
        paint_layout = QVBoxLayout(paint_box)
        paint_layout.addWidget(self.rb_reward)
        paint_layout.addWidget(self.rb_obstacle)
        paint_layout.addWidget(self.rb_terminal_pos)
        paint_layout.addWidget(self.rb_terminal_neg)
        paint_layout.addWidget(self.rb_clear)

        paint_form = QFormLayout()
        paint_form.addRow("Reward", self.reward_spin)
        paint_form.addRow("Terminal +", self.terminal_pos_spin)
        paint_form.addRow("Terminal -", self.terminal_neg_spin)
        paint_layout.addLayout(paint_form)

        vi_box = QGroupBox("Value iteration and policy iteration")
        vi_form = QFormLayout(vi_box)
        vi_form.addRow("Gamma (discount)", self.gamma_spin)
        vi_form.addRow("Theta (stop)", self.theta_spin)
        vi_form.addRow("Max iterations", self.max_iter_spin)
        vi_form.addRow(self.run_vi_btn)
        vi_form.addRow("Status", self.status)

        root.addWidget(size_box)
        root.addWidget(paint_box)
        root.addWidget(vi_box)
        root.addStretch(1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        hint = QLabel(
            "Click cells in the grid to edit. "
            "Obstacles are unreachable. Terminal cells end the episode." 
            "\nCenter text shows V(s) / Vpi(s). Green arrow = VI greedy action, blue arrow = PI current action."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Value Iteration and Policy Iteration Demo (GridWorld)")

        self.world = build_demo_world()
        self.grid = GridWorldWidget(self.world)
        self.controls = ControlsPanel()
        self._vi_session: Optional[ValueIterationSession] = None
        self._pi_session: Optional[PolicyIterationSession] = None

        self._wire_events()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(16)
        layout.addWidget(self.grid, 1)
        layout.addWidget(self.controls, 0)
        self.setCentralWidget(central)

        self.resize(1050, 650)

        self.setFocusPolicy(Qt.StrongFocus)

    def _wire_events(self) -> None:
        # Mode selection
        self.controls.rb_reward.toggled.connect(lambda checked: checked and self.grid.set_mode("reward"))
        self.controls.rb_obstacle.toggled.connect(lambda checked: checked and self.grid.set_mode("obstacle"))
        self.controls.rb_terminal_pos.toggled.connect(
            lambda checked: checked and self.grid.set_mode("terminal_pos")
        )
        self.controls.rb_terminal_neg.toggled.connect(
            lambda checked: checked and self.grid.set_mode("terminal_neg")
        )
        self.controls.rb_clear.toggled.connect(lambda checked: checked and self.grid.set_mode("clear"))

        # Paint parameters
        self.controls.reward_spin.valueChanged.connect(self.grid.set_paint_reward)
        self.controls.terminal_pos_spin.valueChanged.connect(self.grid.set_terminal_pos_reward)
        self.controls.terminal_neg_spin.valueChanged.connect(self.grid.set_terminal_neg_reward)

        # Size and reset
        self.controls.apply_size_btn.clicked.connect(self._apply_size)
        self.controls.reset_demo_btn.clicked.connect(self._reset_demo)

        # Run VI
        self.controls.run_vi_btn.clicked.connect(self._run_value_iteration)

        # Reset stepping session when user edits the world
        self.grid.worldEdited.connect(self._reset_vi_session)

        # Initialize paint values
        self.grid.set_paint_reward(self.controls.reward_spin.value())
        self.grid.set_terminal_pos_reward(self.controls.terminal_pos_spin.value())
        self.grid.set_terminal_neg_reward(self.controls.terminal_neg_spin.value())

    def _apply_size(self) -> None:
        w = int(self.controls.width_spin.value())
        h = int(self.controls.height_spin.value())
        self.world = GridWorld(w, h, default_reward=float(self.controls.reward_spin.value()))
        self.grid.set_world(self.world)
        self._reset_vi_session()
        self.controls.status.setText("Resized grid. (World cleared)")

    def _reset_demo(self) -> None:
        self.world = build_demo_world()
        self.controls.width_spin.setValue(self.world.width)
        self.controls.height_spin.setValue(self.world.height)
        self.controls.reward_spin.setValue(self.world.default_reward)
        self.grid.set_world(self.world)
        self._reset_vi_session()
        self.controls.status.setText("Reset to demo world.")

    def _reset_vi_session(self) -> None:
        self._vi_session = None
        self._pi_session = None

    def _update_grid_comparison(
        self,
        vi_values: Dict[Cell, float],
        pi_values: Dict[Cell, float],
        *,
        gamma: float,
        pi_policy: Dict[Cell, str],
    ) -> None:
        self.grid.set_value_comparison(vi_values, pi_values)
        self.grid.set_policy_comparison(
            greedy_policy_from_values(self.world, vi_values, gamma=gamma),
            pi_policy,
        )

    def _run_value_iteration(self) -> None:
        if not any(True for _ in self.world.states()):
            self.controls.status.setText("No reachable states.")
            return
        if len(self.world.terminals) < 2:
            self.controls.status.setText("Tip: set two terminal cells (one +, one -).")

        gamma = float(self.controls.gamma_spin.value())
        theta = float(self.controls.theta_spin.value())
        max_iter = int(self.controls.max_iter_spin.value())

        try:
            vi_values, vi_iterations, vi_delta = value_iteration(
                self.world,
                gamma=gamma,
                theta=theta,
                max_iterations=max_iter,
            )
            pi_values, pi_policy, pi_iterations, pi_eval_sweeps, pi_delta, pi_stable = policy_iteration(
                self.world,
                gamma=gamma,
                theta=theta,
                max_iterations=max_iter,
            )
        except Exception as e:  # pragma: no cover
            self.controls.status.setText(f"Error: {e}")
            return

        self._update_grid_comparison(
            vi_values,
            pi_values,
            gamma=gamma,
            pi_policy=pi_policy,
        )
        self.controls.status.setText(
            "VI: "
            f"{vi_iterations} sweeps, last Δ={vi_delta:.6g}"
            "\nPI: "
            f"{pi_iterations} improvement steps, {pi_eval_sweeps} eval sweeps, "
            f"last eval Δ={pi_delta:.6g}, stable={pi_stable}"
        )

    def _step_value_iteration(self) -> None:
        if not any(True for _ in self.world.states()):
            self.controls.status.setText("No reachable states.")
            return

        gamma = float(self.controls.gamma_spin.value())
        theta = float(self.controls.theta_spin.value())
        max_iter = int(self.controls.max_iter_spin.value())

        if self._vi_session is None or self._vi_session.world is not self.world or self._vi_session.gamma != gamma:
            try:
                self._vi_session = ValueIterationSession(self.world, gamma=gamma)
            except Exception as e:  # pragma: no cover
                self.controls.status.setText(f"Error: {e}")
                return

        if self._pi_session is None or self._pi_session.world is not self.world or self._pi_session.gamma != gamma:
            try:
                self._pi_session = PolicyIterationSession(self.world, gamma=gamma, theta=theta)
            except Exception as e:  # pragma: no cover
                self.controls.status.setText(f"Error: {e}")
                return

        self._pi_session.theta = theta

        if self._vi_session.iteration >= max_iter:
            vi_values = dict(self._vi_session.V)
            vi_delta = self._vi_session.last_delta
            vi_message = "VI reached max sweeps"
        elif self._vi_session.last_delta < theta:
            vi_values = dict(self._vi_session.V)
            vi_delta = self._vi_session.last_delta
            vi_message = f"VI already converged, Δ={vi_delta:.6g}"
        else:
            vi_values, vi_delta, k = self._vi_session.step()
            vi_message = f"VI sweep {k}, Δ={vi_delta:.6g}"

        if self._pi_session.iteration >= max_iter:
            pi_values = dict(self._pi_session.V)
            pi_policy = dict(self._pi_session.policy)
            pi_delta = self._pi_session.last_eval_delta
            pi_message = "PI reached max policy steps"
        elif self._pi_session.policy_stable:
            pi_values = dict(self._pi_session.V)
            pi_policy = dict(self._pi_session.policy)
            pi_delta = self._pi_session.last_eval_delta
            pi_message = f"PI already stable, eval Δ={pi_delta:.6g}"
        else:
            pi_values, pi_policy, pi_delta, pi_eval_sweeps, pi_k, pi_stable = self._pi_session.step(
                max_eval_iterations=max_iter,
            )
            stable_suffix = ", stable" if pi_stable else ""
            pi_message = (
                f"PI step {pi_k}, {pi_eval_sweeps} eval sweeps, "
                f"eval Δ={pi_delta:.6g}{stable_suffix}"
            )

        self._update_grid_comparison(
            vi_values,
            pi_values,
            gamma=gamma,
            pi_policy=pi_policy,
        )
        self.controls.status.setText(f"{vi_message}\n{pi_message}")

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Space:
            self._step_value_iteration()
            event.accept()
            return
        super().keyPressEvent(event)


def build_demo_world() -> GridWorld:
    # A small, readable default world
    world = GridWorld(width=6, height=6, default_reward=-0.04)

    # Obstacles (a small wall)
    for cell in [(2, 1), (2, 2), (2, 3)]:
        world.set_obstacle(cell, True)

    # Some custom rewards (non-terminals)
    # Keep these non-positive by default to avoid confusing reward-cycles
    # dominating small terminal rewards (e.g., when terminals are set to ±1).
    world.set_reward((4, 1), -0.6)
    world.set_reward((1, 4), -0.3)
    world.set_reward((3, 4), -0.2)

    # Two terminal cells with large positive/negative reward
    world.set_terminal((5, 0), 10.0, kind="pos")
    world.set_terminal((5, 1), -10.0, kind="neg")

    return world


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
