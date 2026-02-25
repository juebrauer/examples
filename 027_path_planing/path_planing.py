# bfs_vs_wavefront_qt.py
# PySide6 demo: BFS from START vs BFS from GOAL on a grid world (interactive)
#
# Controls:
#   - Generate world: new random obstacles (tries to keep it solvable)
#   - Select start: click a free cell to set start
#   - Select goal : click a free cell to set goal
#   - BFS from start: BFS expansion from START (explored + shortest path)
#   - BFS from goal : BFS expansion from GOAL (distance field + path via decreasing distances)
#
# Run:
#   python bfs_vs_wavefront_qt.py
#
# Requires: PySide6

from __future__ import annotations

import random
from dataclasses import dataclass
from collections import deque

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout
)


START_CELL_RGB = (140, 170, 220)
GOAL_CELL_RGB = (140, 210, 140)


def _rgb_to_css_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

ACTIONS = ["U", "D", "L", "R"]
DELTA = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


@dataclass
class World:
    w: int
    h: int
    p: float
    obstacles: set[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]


def in_bounds(w: int, h: int, x: int, y: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def neighbors4(w: int, h: int, x: int, y: int):
    for a in ACTIONS:
        dx, dy = DELTA[a]
        nx, ny = x + dx, y + dy
        if in_bounds(w, h, nx, ny):
            yield nx, ny


def bfs_from_start_path_and_explored(w: int, h: int, obstacles: set[tuple[int, int]],
                                     start: tuple[int, int], goal: tuple[int, int]):
    """
    BFS from start to goal.
    Returns: (path or None, explored_order list, explored_set, dist_map)
    """
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    explored_order: list[tuple[int, int]] = []
    explored_set: set[tuple[int, int]] = set()
    dist: dict[tuple[int, int], int] = {start: 0}

    while q:
        x, y = q.popleft()
        explored_order.append((x, y))
        explored_set.add((x, y))

        if (x, y) == goal:
            break

        for nx, ny in neighbors4(w, h, x, y):
            if (nx, ny) in obstacles:
                continue
            if (nx, ny) in parent:
                continue
            parent[(nx, ny)] = (x, y)
            dist[(nx, ny)] = dist[(x, y)] + 1
            q.append((nx, ny))

    if goal not in parent:
        return None, explored_order, explored_set, dist

    # reconstruct path
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, explored_order, explored_set, dist


def bfs_from_goal_dist_and_path(w: int, h: int, obstacles: set[tuple[int, int]],
                                start: tuple[int, int], goal: tuple[int, int]):
    """
    BFS expansion starting at GOAL to compute distance-to-goal for each reachable cell.

    Path reconstruction does NOT need parent pointers: starting at START, repeatedly step
    to a neighbor whose distance is exactly one smaller until reaching GOAL.

    Returns: (path or None, dist_map, expansion_order)
    """
    dist: dict[tuple[int, int], int] = {goal: 0}
    order: list[tuple[int, int]] = [goal]
    q = deque([goal])

    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for nx, ny in neighbors4(w, h, x, y):
            if (nx, ny) in obstacles:
                continue
            if (nx, ny) in dist:
                continue
            dist[(nx, ny)] = d + 1
            order.append((nx, ny))
            q.append((nx, ny))

    if start not in dist:
        return None, dist, order

    # Reconstruct a shortest path START->GOAL by following strictly decreasing distances.
    cur = start
    path = [cur]
    while cur != goal:
        x, y = cur
        cur_d = dist[cur]
        nxt: tuple[int, int] | None = None
        for nx, ny in neighbors4(w, h, x, y):
            if (nx, ny) in obstacles:
                continue
            if (nx, ny) not in dist:
                continue
            if dist[(nx, ny)] == cur_d - 1:
                nxt = (nx, ny)
                break
        if nxt is None:
            # Shouldn't happen for a proper BFS distance field, but keep safe.
            return None, dist, order
        cur = nxt
        path.append(cur)

    return path, dist, order


# --- Rendering (reuses the style from your original script) ---

def render_world_image(
    w: int,
    h: int,
    cell_px: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    explored: set[tuple[int, int]] | None = None,
    path: list[tuple[int, int]] | None = None,
    goal_dist: dict[tuple[int, int], int] | None = None,
    bfs_dist: dict[tuple[int, int], int] | None = None,
    show_goal_numbers: bool = False,
) -> QImage:
    img = QImage(w * cell_px, h * cell_px, QImage.Format_ARGB32)
    img.fill(QColor(245, 245, 245))
    p = QPainter(img)

    # Optional: BFS-from-goal distance shading (very light, keeps original style dominant)
    if goal_dist:
        max_d = max(goal_dist.values()) if goal_dist else 1
        for (x, y), d in goal_dist.items():
            if (x, y) in obstacles:
                continue
            # map distance to a gentle brightness; goal (0) stays near background
            # We avoid heavy heatmap styling; just subtle blue-ish tint.
            t = 0.0 if max_d == 0 else (d / max_d)
            # from near-white to a soft tint
            col = QColor(
                int(245 - 40 * t),
                int(245 - 25 * t),
                int(245 - 10 * t),
            )
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            p.drawRect(x * cell_px, y * cell_px, cell_px, cell_px)

    # Optional: explored cells (BFS) overlay (soft yellow)
    if explored:
        p.setBrush(QColor(245, 235, 180))
        p.setPen(Qt.NoPen)
        for (x, y) in explored:
            if (x, y) in obstacles or (x, y) in (start, goal):
                continue
            p.drawRect(x * cell_px, y * cell_px, cell_px, cell_px)

    # obstacles (same as original)
    p.setBrush(QColor(60, 60, 60))
    p.setPen(QColor(60, 60, 60))
    for (x, y) in obstacles:
        p.drawRect(x * cell_px, y * cell_px, cell_px, cell_px)

    # goal (same as original)
    gx, gy = goal
    p.setBrush(QColor(*GOAL_CELL_RGB))
    p.setPen(QColor(*GOAL_CELL_RGB))
    p.drawRect(gx * cell_px, gy * cell_px, cell_px, cell_px)

    # start (same as original)
    sx, sy = start
    p.setBrush(QColor(*START_CELL_RGB))
    p.setPen(QColor(*START_CELL_RGB))
    p.drawRect(sx * cell_px, sy * cell_px, cell_px, cell_px)

    # path overlay (draw on top; soft red cells + line)
    if path and len(path) >= 2:
        # cell tint
        p.setBrush(QColor(235, 170, 170))
        p.setPen(Qt.NoPen)
        for (x, y) in path:
            if (x, y) in (start, goal):
                continue
            m = max(2, cell_px // 12)
            p.drawRect(x * cell_px + m, y * cell_px + m, cell_px - 2 * m, cell_px - 2 * m)

        # connecting line
        p.setPen(QColor(200, 80, 80))
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            cx0, cy0 = x0 * cell_px + cell_px // 2, y0 * cell_px + cell_px // 2
            cx1, cy1 = x1 * cell_px + cell_px // 2, y1 * cell_px + cell_px // 2
            p.drawLine(cx0, cy0, cx1, cy1)

    # BFS-from-goal distance numbers (optional)
    if goal_dist and show_goal_numbers and cell_px >= 20:
        f = QFont()
        f.setPointSize(max(6, int(cell_px * 0.25)))
        p.setFont(f)
        p.setPen(QColor(90, 90, 90))
        for (x, y), d in goal_dist.items():
            if (x, y) in obstacles:
                continue
            r = QRect(x * cell_px, y * cell_px, cell_px, cell_px)
            p.drawText(r, Qt.AlignCenter, str(d))

    # BFS distance numbers (distance from START)
    if bfs_dist and cell_px >= 20:
        f = QFont()
        f.setPointSize(max(6, int(cell_px * 0.25)))
        p.setFont(f)
        p.setPen(QColor(90, 90, 90))
        for (x, y), d in bfs_dist.items():
            if (x, y) in obstacles:
                continue
            r = QRect(x * cell_px, y * cell_px, cell_px, cell_px)
            p.drawText(r, Qt.AlignCenter, str(d))

    # grid lines (same as original)
    p.setPen(QColor(200, 200, 200))
    for x in range(w + 1):
        xx = x * cell_px
        p.drawLine(xx, 0, xx, h * cell_px)
    for y in range(h + 1):
        yy = y * cell_px
        p.drawLine(0, yy, w * cell_px, yy)

    p.end()
    return img


class GridView(QWidget):
    cellClicked = Signal(int, int)

    def __init__(self, cell_px: int):
        super().__init__()
        self.img: QImage | None = None
        self.cell_px = cell_px

    def set_cell_px(self, cell_px: int):
        self.cell_px = cell_px
        if self.img is not None:
            self.setFixedSize(self.img.width(), self.img.height())
        self.update()

    def set_image(self, img: QImage):
        self.img = img
        self.setFixedSize(img.width(), img.height())
        self.update()

    def paintEvent(self, event):
        if self.img is None:
            return
        p = QPainter(self)
        p.drawImage(0, 0, self.img)
        p.end()

    def mousePressEvent(self, event):
        if self.img is None:
            return
        if event.button() != Qt.LeftButton:
            return
        x = int(event.position().x()) // self.cell_px
        y = int(event.position().y()) // self.cell_px
        self.cellClicked.emit(x, y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFS from Start vs BFS from Goal (Grid World Demo)")

        # world params
        self.rng = random.Random(None)
        self.w = 16
        self.h = 12
        self.p = 0.22
        self.cell_px = 32

        # UI state
        self.select_mode: str | None = None  # None | "start" | "goal"
        self.explored: set[tuple[int, int]] | None = None
        self.path: list[tuple[int, int]] | None = None
        self.goal_dist: dict[tuple[int, int], int] | None = None
        self.bfs_dist: dict[tuple[int, int], int] | None = None
        self.show_goal_numbers = True

        # init world
        self.world = self._generate_solvable_world()

        # widgets
        self.view = GridView(self.cell_px)
        self.view.cellClicked.connect(self.on_cell_clicked)

        self.status = QLabel()
        self.status.setWordWrap(True)

        btn_generate = QPushButton("Generate world")
        btn_start = QPushButton("Select start")
        btn_goal = QPushButton("Select goal")
        btn_bfs_start = QPushButton("BFS from start")
        btn_bfs_goal = QPushButton("BFS from goal")

        # Match button colors to the start/goal cell colors.
        btn_start.setStyleSheet(f"QPushButton {{ background-color: {_rgb_to_css_hex(START_CELL_RGB)}; }}")
        btn_goal.setStyleSheet(f"QPushButton {{ background-color: {_rgb_to_css_hex(GOAL_CELL_RGB)}; }}")

        btn_generate.clicked.connect(self.on_generate_world)
        btn_start.clicked.connect(lambda: self.set_select_mode("start"))
        btn_goal.clicked.connect(lambda: self.set_select_mode("goal"))
        btn_bfs_start.clicked.connect(self.on_run_bfs_from_start)
        btn_bfs_goal.clicked.connect(self.on_run_bfs_from_goal)

        # parameter controls (kept small + optional, useful for demos)
        gb = QGroupBox("World parameters")
        grid = QGridLayout(gb)
        self.sb_w = QSpinBox(); self.sb_w.setRange(4, 80); self.sb_w.setValue(self.w)
        self.sb_h = QSpinBox(); self.sb_h.setRange(4, 60); self.sb_h.setValue(self.h)
        self.sb_cell = QSpinBox(); self.sb_cell.setRange(12, 64); self.sb_cell.setValue(self.cell_px)
        self.sb_p = QDoubleSpinBox(); self.sb_p.setRange(0.0, 0.55); self.sb_p.setSingleStep(0.01); self.sb_p.setDecimals(2); self.sb_p.setValue(self.p)

        grid.addWidget(QLabel("Width"), 0, 0); grid.addWidget(self.sb_w, 0, 1)
        grid.addWidget(QLabel("Height"), 1, 0); grid.addWidget(self.sb_h, 1, 1)
        grid.addWidget(QLabel("Obstacle p"), 2, 0); grid.addWidget(self.sb_p, 2, 1)
        grid.addWidget(QLabel("Cell px"), 3, 0); grid.addWidget(self.sb_cell, 3, 1)

        apply_params = QPushButton("Apply params")
        apply_params.clicked.connect(self.on_apply_params)
        grid.addWidget(apply_params, 4, 0, 1, 2)

        # Keep controls compact in the right-side panel.
        control_max_w = 180
        self.sb_w.setMaximumWidth(control_max_w)
        self.sb_h.setMaximumWidth(control_max_w)
        self.sb_p.setMaximumWidth(control_max_w)
        self.sb_cell.setMaximumWidth(control_max_w)
        apply_params.setMaximumWidth(control_max_w)

        # layout: grid view on the left, controls on the right
        root = QWidget()
        main = QHBoxLayout(root)

        main.addWidget(self.view)

        right = QVBoxLayout()

        buttons = QVBoxLayout()
        buttons.addWidget(btn_generate)
        buttons.addWidget(btn_start)
        buttons.addWidget(btn_goal)
        buttons.addWidget(btn_bfs_start)
        buttons.addWidget(btn_bfs_goal)

        right.addLayout(buttons)
        right.addWidget(gb)
        right.addWidget(self.status)
        right.addStretch(1)

        main.addLayout(right)

        self.setCentralWidget(root)
        self._refresh_view(initial=True)

    # --- World generation ---

    def _random_free_cell(self, obstacles: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            x = self.rng.randrange(self.w)
            y = self.rng.randrange(self.h)
            if (x, y) not in obstacles:
                return (x, y)

    def _generate_solvable_world(self, max_tries: int = 5000) -> World:
        for _ in range(max_tries):
            obstacles = set()
            for y in range(self.h):
                for x in range(self.w):
                    if self.rng.random() < self.p:
                        obstacles.add((x, y))

            start = self._random_free_cell(obstacles)
            goal = self._random_free_cell(obstacles)
            while goal == start:
                goal = self._random_free_cell(obstacles)

            # ensure solvable
            path, _, _, _ = bfs_from_start_path_and_explored(self.w, self.h, obstacles, start, goal)
            if path is None:
                continue

            return World(self.w, self.h, self.p, obstacles, start, goal)

        # fallback: no solvable found -> empty obstacles
        obstacles = set()
        start = (0, 0)
        goal = (self.w - 1, self.h - 1)
        return World(self.w, self.h, self.p, obstacles, start, goal)

    # --- UI actions ---

    def set_select_mode(self, mode: str | None):
        self.select_mode = mode
        if mode == "start":
            self.status.setText("Select start: click a free cell.")
        elif mode == "goal":
            self.status.setText("Select goal: click a free cell.")
        else:
            self.status.setText("")

    def clear_algorithm_overlays(self):
        self.explored = None
        self.path = None
        self.goal_dist = None
        self.bfs_dist = None

    def on_apply_params(self):
        self.w = int(self.sb_w.value())
        self.h = int(self.sb_h.value())
        self.p = float(self.sb_p.value())
        self.cell_px = int(self.sb_cell.value())
        self.view.set_cell_px(self.cell_px)
        self.on_generate_world()

    def on_generate_world(self):
        self.clear_algorithm_overlays()
        self.set_select_mode(None)
        self.world = self._generate_solvable_world()
        self._refresh_view()
        self.status.setText(
            "Generated a new random world.\n"
            "Use Select start / Select goal, then run BFS from start or BFS from goal."
        )

    def on_cell_clicked(self, x: int, y: int):
        if not in_bounds(self.world.w, self.world.h, x, y):
            return
        if (x, y) in self.world.obstacles:
            self.status.setText("That cell is an obstacle. Pick a free cell.")
            return

        if self.select_mode == "start":
            if (x, y) == self.world.goal:
                self.status.setText("Start cannot equal goal.")
                return
            self.world.start = (x, y)
            self.clear_algorithm_overlays()
            self._refresh_view()
            self.status.setText(f"Start set to {self.world.start}. Now pick a goal or run an algorithm.")
            return

        if self.select_mode == "goal":
            if (x, y) == self.world.start:
                self.status.setText("Goal cannot equal start.")
                return
            self.world.goal = (x, y)
            self.clear_algorithm_overlays()
            self._refresh_view()
            self.status.setText(f"Goal set to {self.world.goal}. Now run BFS from start or BFS from goal.")
            return


    def on_run_bfs_from_start(self):
        self.set_select_mode(None)
        path, _, explored, dist = bfs_from_start_path_and_explored(
            self.world.w, self.world.h, self.world.obstacles, self.world.start, self.world.goal
        )
        self.goal_dist = None
        self.bfs_dist = dist
        self.explored = explored
        self.path = path
        self._refresh_view()

        if path is None:
            self.status.setText(
                "BFS from start: No path found.\n"
                "Expands outward from START level-by-level until it reaches the GOAL."
            )
        else:
            self.status.setText(
                f"BFS from start: Found a shortest path with length {len(path) - 1}.\n"
                "Expands outward from START level-by-level, storing parents to reconstruct a shortest path."
            )

    def on_run_bfs_from_goal(self):
        self.set_select_mode(None)
        path, dist, _ = bfs_from_goal_dist_and_path(
            self.world.w, self.world.h, self.world.obstacles, self.world.start, self.world.goal
        )
        self.explored = None
        self.bfs_dist = None
        self.goal_dist = dist
        self.path = path
        self._refresh_view()

        if path is None:
            self.status.setText(
                "BFS from goal: No path found (start not reachable).\n"
                "Expands outward from GOAL (BFS), building a distance-to-goal field."
            )
        else:
            self.status.setText(
                f"BFS from goal: Found a shortest path with length {len(path) - 1}.\n"
                "Expands outward from GOAL (BFS) to compute distances; then reconstructs a shortest path by stepping from START to a neighbor with distance −1 until reaching GOAL."
            )

    # --- Rendering ---

    def _refresh_view(self, initial: bool = False):
        img = render_world_image(
            self.world.w, self.world.h, self.cell_px,
            self.world.obstacles, self.world.start, self.world.goal,
            explored=self.explored,
            path=self.path,
            goal_dist=self.goal_dist,
            bfs_dist=self.bfs_dist,
            show_goal_numbers=self.show_goal_numbers,
        )
        self.view.set_image(img)

        if initial:
            self.status.setText(
                "Pick Select start / Select goal, then run BFS from start or BFS from goal.\n"
                "BFS from start grows from START; BFS from goal grows from GOAL (distance field)."
            )


def main():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()