#!/usr/bin/env python3
"""Value Iteration demo (GridWorld) with a simple PySide6 UI.

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
        V[t] = float(r)

    last_delta = 0.0
    for i in range(max_iterations):
        delta = 0.0
        V_new = dict(V)

        for s in world.states():
            if world.is_terminal(s):
                V_new[s] = float(world.terminals[s])
                continue

            best_q = -math.inf
            for a in ACTIONS:
                s2, r, done = world.step(s, a)
                q = float(r) + (0.0 if done else gamma * float(V[s2]))
                if q > best_q:
                    best_q = q

            V_new[s] = best_q
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        last_delta = delta
        if delta < theta:
            return V, i + 1, last_delta

    return V, max_iterations, last_delta


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
        V_new = dict(self.V)
        delta = 0.0

        for s in self.world.states():
            if self.world.is_terminal(s):
                V_new[s] = float(self.world.terminals[s])
                continue

            best_q = -math.inf
            for a in ACTIONS:
                s2, r, done = self.world.step(s, a)
                q = float(r) + (0.0 if done else self.gamma * float(self.V[s2]))
                if q > best_q:
                    best_q = q

            V_new[s] = best_q
            delta = max(delta, abs(V_new[s] - self.V[s]))

        self.V = V_new
        self.iteration += 1
        self.last_delta = delta
        return dict(self.V), delta, self.iteration


class GridWorldWidget(QWidget):
    worldEdited = Signal()

    def __init__(self, world: GridWorld, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._world = world
        self._mode: str = "reward"  # reward|obstacle|terminal_pos|terminal_neg|clear
        self._paint_reward: float = 0.0
        self._terminal_pos_reward: float = 10.0
        self._terminal_neg_reward: float = -10.0

        self._values: Dict[Cell, float] = {}
        self._show_values = False

        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)

    def set_world(self, world: GridWorld) -> None:
        self._world = world
        self._values = {}
        self._show_values = False
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_paint_reward(self, reward: float) -> None:
        self._paint_reward = float(reward)

    def set_terminal_pos_reward(self, reward: float) -> None:
        self._terminal_pos_reward = float(reward)

    def set_terminal_neg_reward(self, reward: float) -> None:
        self._terminal_neg_reward = float(reward)

    def set_values(self, values: Dict[Cell, float]) -> None:
        self._values = dict(values)
        self._show_values = True
        self.update()

    def clear_values(self) -> None:
        self._values = {}
        self._show_values = False
        self.update()

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
                    value = self._values.get(cell)

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

                    # Value larger, centered (only after running VI)
                    if self._show_values and value is not None:
                        font.setPointSize(max(9, int(cell_size * 0.18)))
                        font.setBold(True)
                        painter.setFont(font)
                        painter.drawText(rect, Qt.AlignCenter, f"{value:.2f}")

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

        # Value iteration params
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

        self.run_vi_btn = QPushButton("Run value iteration")
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

        vi_box = QGroupBox("Value iteration")
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
            "\nValues appear after running value iteration."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Value Iteration Demo (GridWorld)")

        self.world = build_demo_world()
        self.grid = GridWorldWidget(self.world)
        self.controls = ControlsPanel()
        self._vi_session: Optional[ValueIterationSession] = None

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
            values, iterations, delta = value_iteration(
                self.world,
                gamma=gamma,
                theta=theta,
                max_iterations=max_iter,
            )
        except Exception as e:  # pragma: no cover
            self.controls.status.setText(f"Error: {e}")
            return

        self.grid.set_values(values)
        self.controls.status.setText(
            f"Done: {iterations} iterations, last Δ={delta:.6g}"
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

        if self._vi_session.iteration >= max_iter:
            self.controls.status.setText("Reached max iterations.")
            self.grid.set_values(self._vi_session.V)
            return
        if self._vi_session.last_delta < theta:
            self.controls.status.setText("Already converged (Δ < theta).")
            self.grid.set_values(self._vi_session.V)
            return

        values, delta, k = self._vi_session.step()
        self.grid.set_values(values)
        self.controls.status.setText(f"Step: iteration {k}, Δ={delta:.6g}")

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
