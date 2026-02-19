import argparse
import csv
import random
from collections import deque
from pathlib import Path

from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtGui import QGuiApplication  # needed on some platforms for image plugins

ACTIONS = ["U", "D", "L", "R"]
DELTA = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def bfs_path(w, h, obstacles, start, goal):
    q = deque([start])
    parent = {start: None}

    def nbrs(x, y):
        for a in ACTIONS:
            dx, dy = DELTA[a]
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles:
                yield (nx, ny)

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for nb in nbrs(x, y):
            if nb not in parent:
                parent[nb] = (x, y)
                q.append(nb)

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def path_to_actions(path):
    out = []
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx, dy = x1 - x0, y1 - y0
        if dx == 1 and dy == 0:
            out.append("R")
        elif dx == -1 and dy == 0:
            out.append("L")
        elif dx == 0 and dy == 1:
            out.append("D")
        elif dx == 0 and dy == -1:
            out.append("U")
        else:
            raise ValueError("Non-4-neighborhood step in path.")
    return out


class GridWorld:
    def __init__(self, w, h, p, rng):
        self.w, self.h, self.p = int(w), int(h), float(p)
        self.rng = rng
        self.start = self.goal = self.robot = (0, 0)
        self.obstacles = set()
        self.collisions = 0
        self.steps = 0

    def reset_solvable(self, max_tries=5000):
        for _ in range(max_tries):
            self.collisions = 0
            self.steps = 0
            self.obstacles = set()

            self.start = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            self.goal = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            while self.goal == self.start:
                self.goal = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            self.robot = self.start

            for y in range(self.h):
                for x in range(self.w):
                    if (x, y) in (self.start, self.goal):
                        continue
                    if self.rng.random() < self.p:
                        self.obstacles.add((x, y))

            if bfs_path(self.w, self.h, self.obstacles, self.start, self.goal) is not None:
                return
        raise RuntimeError("Could not generate a solvable episode. Lower --p.")

    def step(self, action):
        dx, dy = DELTA[action]
        x, y = self.robot
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.w and 0 <= ny < self.h) or (nx, ny) in self.obstacles:
            self.collisions += 1  # robot stays
        else:
            self.robot = (nx, ny)
        self.steps += 1
        return self.robot == self.goal


def render_image(w, h, cell_px, obstacles, start, goal, robot):
    img = QImage(w * cell_px, h * cell_px, QImage.Format_ARGB32)
    img.fill(QColor(245, 245, 245))
    p = QPainter(img)

    # obstacles
    p.setBrush(QColor(60, 60, 60))
    p.setPen(QColor(60, 60, 60))
    for (x, y) in obstacles:
        p.drawRect(x * cell_px, y * cell_px, cell_px, cell_px)

    # goal
    gx, gy = goal
    p.setBrush(QColor(140, 210, 140))
    p.setPen(QColor(140, 210, 140))
    p.drawRect(gx * cell_px, gy * cell_px, cell_px, cell_px)

    # start
    sx, sy = start
    p.setBrush(QColor(140, 170, 220))
    p.setPen(QColor(140, 170, 220))
    p.drawRect(sx * cell_px, sy * cell_px, cell_px, cell_px)

    # robot (inner square)
    rx, ry = robot
    m = max(2, cell_px // 10)
    p.setBrush(QColor(220, 120, 120))
    p.setPen(QColor(220, 120, 120))
    p.drawRect(rx * cell_px + m, ry * cell_px + m, cell_px - 2 * m, cell_px - 2 * m)

    # grid lines (optional but helpful)
    p.setPen(QColor(200, 200, 200))
    for x in range(w + 1):
        xx = x * cell_px
        p.drawLine(xx, 0, xx, h * cell_px)
    for y in range(h + 1):
        yy = y * cell_px
        p.drawLine(0, yy, w * cell_px, yy)

    p.end()
    return img


def ensure_csv_headers(path, header):
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)


def next_episode_index(episodes_csv):
    if not episodes_csv.exists():
        return 1
    last = 0
    with episodes_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                last = max(last, int(row["episode"]))
            except Exception:
                pass
    return last + 1


def main():
    ap = argparse.ArgumentParser(description="UI-free 2D grid robot demo dataset generator (PySide6 offscreen render).")
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--h", type=int, default=12)
    ap.add_argument("--p", type=float, default=0.22, help="Obstacle probability per cell (except start/goal).")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default="./dataset")
    ap.add_argument("--cell", type=int, default=32, help="Cell size in pixels for rendered images.")
    ap.add_argument("--ep-digits", type=int, default=6)
    ap.add_argument("--img-digits", type=int, default=6)
    args = ap.parse_args()

    # QGuiApplication helps ensure PNG plugin availability on some systems.
    _app = QGuiApplication.instance() or QGuiApplication([])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    actions_csv = out / "actions.csv"
    episodes_csv = out / "episodes.csv"
    ensure_csv_headers(actions_csv, ["episode", "image", "action"])
    ensure_csv_headers(episodes_csv, ["episode", "N"])

    rng = random.Random(args.seed)
    env = GridWorld(args.w, args.h, args.p, rng)

    ep_idx = next_episode_index(episodes_csv)

    total_collisions = 0
    for _ in range(args.episodes):
        env.reset_solvable()

        path = bfs_path(env.w, env.h, env.obstacles, env.robot, env.goal)
        actions = path_to_actions(path)
        N = len(actions)

        # record images and actions: image is BEFORE applying action (image->action pairing)
        with actions_csv.open("a", newline="") as fa:
            wa = csv.writer(fa)
            for img_i, a in enumerate(actions, start=1):
                img = render_image(env.w, env.h, args.cell, env.obstacles, env.start, env.goal, env.robot)
                ep_s = str(ep_idx).zfill(args.ep_digits)
                im_s = str(img_i).zfill(args.img_digits)
                img_path = out / f"episode{ep_s}-{im_s}.png"
                img.save(str(img_path))

                wa.writerow([ep_idx, img_i, a])
                env.step(a)

        with episodes_csv.open("a", newline="") as fe:
            csv.writer(fe).writerow([ep_idx, N])

        total_collisions += env.collisions
        print(
            f"episode {ep_idx}: N={N}, collisions={env.collisions}, "
            f"start={env.start}, goal={env.goal}, obstacles={len(env.obstacles)}"
        )

        ep_idx += 1

    print(f"Done. Total collisions (all episodes): {total_collisions}")
    print(f"Dataset written to: {out.resolve()}")


if __name__ == "__main__":
    main()
