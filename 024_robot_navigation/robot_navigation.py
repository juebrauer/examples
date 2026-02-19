"""
robot_navigation.py

ONE single script with 4 modes:
  - record       : generate expert (BFS) demonstrations into a dataset folder
  - train        : train a CNN classifier on first 80% of episodes (by episode order)
  - test-offline : evaluate accuracy on last 20% of episodes (no visualization)
  - test-online  : run N new episodes online, CNN controls robot, live visualization, budget=5*X

Dataset format (same as before):
  images: episode<ep>-<img>.png  (leading zeros)
  actions.csv : episode,image,action
  episodes.csv: episode,N

# 1) record demonstrations
QT_QPA_PLATFORM=offscreen python robot_navigation.py record --dataset ./dataset --episodes 500 --w 16 --h 12 --p 0.22

# 2) train CNN
python robot_navigation.py train --dataset ./dataset --out-model cnn.pt --epochs 10 --device auto

# 3) offline test accuracy (last 20% episodes)
python robot_navigation.py test-offline --model cnn.pt --dataset ./dataset --device auto

# 4) online test (new episodes, live)
python robot_navigation.py test-online --model cnn.pt --episodes 30 --w 16 --h 12 --p 0.22 --device auto

"""

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QColor, QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel


# ----------------------------
# Constants
# ----------------------------

ACTIONS = ["U", "D", "L", "R"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
I2A = {i: a for a, i in A2I.items()}
DELTA = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


# ----------------------------
# Grid world + expert
# ----------------------------

@dataclass
class Episode:
    w: int
    h: int
    start: tuple[int, int]
    goal: tuple[int, int]
    robot: tuple[int, int]
    obstacles: set[tuple[int, int]]
    steps: int = 0
    collisions: int = 0


def bfs_path(w, h, obstacles, start, goal):
    q = deque([start])
    parent = {start: None}

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for a in ACTIONS:
            dx, dy = DELTA[a]
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in obstacles and (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def path_to_actions(path_positions):
    out = []
    for (x0, y0), (x1, y1) in zip(path_positions[:-1], path_positions[1:]):
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
            raise ValueError("Non-4-neighborhood step in BFS path.")
    return out


class GridWorld:
    def __init__(self, w: int, h: int, p: float, rng: random.Random):
        self.w, self.h, self.p = int(w), int(h), float(p)
        self.rng = rng
        self.ep: Episode | None = None

    def reset_solvable(self, max_tries=5000) -> Episode:
        for _ in range(max_tries):
            start = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            goal = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            while goal == start:
                goal = (self.rng.randrange(self.w), self.rng.randrange(self.h))

            obstacles = set()
            for y in range(self.h):
                for x in range(self.w):
                    if (x, y) in (start, goal):
                        continue
                    if self.rng.random() < self.p:
                        obstacles.add((x, y))

            if bfs_path(self.w, self.h, obstacles, start, goal) is None:
                continue

            self.ep = Episode(self.w, self.h, start, goal, start, obstacles, steps=0, collisions=0)
            return self.ep

        raise RuntimeError("Could not generate solvable episode. Lower --p or increase grid size.")

    def step(self, action: str) -> bool:
        ep = self.ep
        dx, dy = DELTA[action]
        x, y = ep.robot
        nx, ny = x + dx, y + dy
        if not (0 <= nx < ep.w and 0 <= ny < ep.h) or (nx, ny) in ep.obstacles:
            ep.collisions += 1  # stay
        else:
            ep.robot = (nx, ny)
        ep.steps += 1
        return ep.robot == ep.goal


# ----------------------------
# Rendering + QImage -> torch
# ----------------------------

def render_image(w, h, cell_px, obstacles, start, goal, robot) -> QImage:
    img = QImage(w * cell_px, h * cell_px, QImage.Format_ARGB32)
    img.fill(QColor(245, 245, 245))
    p = QPainter(img)

    # obstacles
    p.setBrush(QColor(60, 60, 60)); p.setPen(QColor(60, 60, 60))
    for (x, y) in obstacles:
        p.drawRect(x * cell_px, y * cell_px, cell_px, cell_px)

    # goal
    gx, gy = goal
    p.setBrush(QColor(140, 210, 140)); p.setPen(QColor(140, 210, 140))
    p.drawRect(gx * cell_px, gy * cell_px, cell_px, cell_px)

    # start
    sx, sy = start
    p.setBrush(QColor(140, 170, 220)); p.setPen(QColor(140, 170, 220))
    p.drawRect(sx * cell_px, sy * cell_px, cell_px, cell_px)

    # robot (inner square)
    rx, ry = robot
    m = max(2, cell_px // 10)
    p.setBrush(QColor(220, 120, 120)); p.setPen(QColor(220, 120, 120))
    p.drawRect(rx * cell_px + m, ry * cell_px + m, cell_px - 2*m, cell_px - 2*m)

    # grid lines (helpful for learning signal; keep)
    p.setPen(QColor(200, 200, 200))
    for x in range(w + 1):
        xx = x * cell_px
        p.drawLine(xx, 0, xx, h * cell_px)
    for y in range(h + 1):
        yy = y * cell_px
        p.drawLine(0, yy, w * cell_px, yy)

    p.end()
    return img


def qimage_to_tensor(img: QImage) -> torch.Tensor:
    """float tensor (3,H,W) in [0,1]"""
    img = img.convertToFormat(QImage.Format_RGB888)
    w = img.width()
    h = img.height()

    bpl = img.bytesPerLine()  # includes possible row padding
    ptr = img.bits()          # memoryview in PySide6/Py3.12
    buf = ptr.tobytes()       # length = h * bpl

    x = torch.frombuffer(buf, dtype=torch.uint8).clone()
    x = x.view(h, bpl)[:, : w * 3]          # drop padding
    x = x.view(h, w, 3).permute(2, 0, 1)    # (3,H,W)
    return x.float() / 255.0


# ----------------------------
# Dataset I/O + splits
# ----------------------------

def ensure_csv_headers(path: Path, header: list[str]):
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)


def read_episodes_csv(dataset_dir: Path):
    path = dataset_dir / "episodes.csv"
    out = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((int(row["episode"]), int(row["N"])))
    out.sort(key=lambda x: x[0])
    return out


def read_actions_csv(dataset_dir: Path):
    path = dataset_dir / "actions.csv"
    out = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((int(row["episode"]), int(row["image"]), row["action"].strip()))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def next_episode_index(episodes_csv: Path) -> int:
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


def split_episodes_80_20(dataset_dir: Path):
    eps = read_episodes_csv(dataset_dir)
    ep_ids = [e for (e, _) in eps]
    if len(ep_ids) < 2:
        raise RuntimeError("Need at least 2 episodes in episodes.csv for an 80/20 split.")
    cut = max(1, int(math.floor(0.8 * len(ep_ids))))
    train_ids = set(ep_ids[:cut])
    test_ids = set(ep_ids[cut:])
    return train_ids, test_ids


class DemoDataset(Dataset):
    def __init__(self, dataset_dir: Path, episodes_set: set[int], ep_digits=6, img_digits=6):
        self.dataset_dir = Path(dataset_dir)
        self.ep_digits = ep_digits
        self.img_digits = img_digits
        self.samples = [(ep, im, a) for (ep, im, a) in read_actions_csv(self.dataset_dir) if ep in episodes_set]

    def __len__(self):
        return len(self.samples)

    def _img_path(self, ep, im):
        ep_s = str(ep).zfill(self.ep_digits)
        im_s = str(im).zfill(self.img_digits)
        return self.dataset_dir / f"episode{ep_s}-{im_s}.png"

    def __getitem__(self, idx):
        ep, im, a = self.samples[idx]
        img = QImage(str(self._img_path(ep, im)))
        x = qimage_to_tensor(img)
        y = torch.tensor(A2I[a], dtype=torch.long)
        return x, y


# ----------------------------
# CNN
# ----------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x)


def save_model(path: Path, model: nn.Module, meta: dict):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(Path(path)))


def load_model(path: Path, device: str):
    payload = torch.load(str(Path(path)), map_location=device)
    model = SmallCNN(num_actions=4).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def resolve_device(device_str: str) -> str:
    d = device_str.strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        # macOS MPS optional
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


# ----------------------------
# Mode: record
# ----------------------------

def mode_record(dataset_dir: Path, episodes: int, w: int, h: int, p: float, cell: int,
                seed: int | None, ep_digits: int, img_digits: int):
    # ensure Qt image plugins available on some platforms
    _app = QGuiApplication.instance() or QGuiApplication([])

    dataset_dir.mkdir(parents=True, exist_ok=True)
    actions_csv = dataset_dir / "actions.csv"
    episodes_csv = dataset_dir / "episodes.csv"
    ensure_csv_headers(actions_csv, ["episode", "image", "action"])
    ensure_csv_headers(episodes_csv, ["episode", "N"])

    ep_idx = next_episode_index(episodes_csv)
    rng = random.Random(seed)
    env = GridWorld(w, h, p, rng)

    for _ in range(episodes):
        env.reset_solvable()
        ep = env.ep
        path = bfs_path(ep.w, ep.h, ep.obstacles, ep.start, ep.goal)
        actions = path_to_actions(path)
        N = len(actions)

        with actions_csv.open("a", newline="") as fa:
            wa = csv.writer(fa)
            for img_i, a in enumerate(actions, start=1):
                # record image BEFORE applying action: (image -> action)
                img = render_image(ep.w, ep.h, cell, ep.obstacles, ep.start, ep.goal, ep.robot)
                ep_s = str(ep_idx).zfill(ep_digits)
                im_s = str(img_i).zfill(img_digits)
                img_path = dataset_dir / f"episode{ep_s}-{im_s}.png"
                img.save(str(img_path))
                wa.writerow([ep_idx, img_i, a])
                env.step(a)

        with episodes_csv.open("a", newline="") as fe:
            csv.writer(fe).writerow([ep_idx, N])

        print(f"Recorded episode {ep_idx}: N={N}, obstacles={len(ep.obstacles)}")
        ep_idx += 1


# ----------------------------
# Mode: train
# ----------------------------

def mode_train(dataset_dir: Path, out_model: Path, epochs: int, batch: int, lr: float, device_str: str):
    device = resolve_device(device_str)
    train_eps, test_eps = split_episodes_80_20(dataset_dir)
    ds_train = DemoDataset(dataset_dir, train_eps)
    ds_test = DemoDataset(dataset_dir, test_eps)

    if len(ds_train) == 0 or len(ds_test) == 0:
        raise RuntimeError("Train/test split produced empty set. Add more episodes.")

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=max(64, batch), shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def eval_acc():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / max(1, total)

    for e in range(1, epochs + 1):
        running = 0.0
        seen = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
        acc = eval_acc()
        print(f"epoch {e}/{epochs} | train loss {running/max(1,seen):.4f} | val acc {acc*100:.2f}%")

    meta = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "actions": ACTIONS,
        "split": "first80_train_last20_test_by_episode_order",
        "device_trained": device,
    }
    save_model(out_model, model, meta)
    print(f"Saved model to {Path(out_model).resolve()}")


# ----------------------------
# Mode: test-offline
# ----------------------------

def mode_test_offline(model_path: Path, dataset_dir: Path, batch: int, device_str: str):
    device = resolve_device(device_str)
    _, test_eps = split_episodes_80_20(dataset_dir)
    ds = DemoDataset(dataset_dir, test_eps)
    if len(ds) == 0:
        raise RuntimeError("Offline test set empty. Add more episodes.")
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    model, meta = load_model(model_path, device=device)

    correct = total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / max(1, total)
    print(json.dumps({
        "mode": "test-offline",
        "model": str(Path(model_path).resolve()),
        "dataset": str(Path(dataset_dir).resolve()),
        "num_samples": total,
        "accuracy": acc,
        "meta": meta,
    }, indent=2))


# ----------------------------
# Mode: test-online (live)
# ----------------------------

class GridView(QWidget):
    def __init__(self):
        super().__init__()
        self.img = None

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


class OnlineWindow(QMainWindow):
    def __init__(self, model_path: Path, num_episodes: int, w: int, h: int, p: float,
                 cell_px: int, seed: int | None, device: str, fps: int):
        super().__init__()
        self.setWindowTitle("test-online: CNN controls the robot (live)")

        self.device = device
        self.model, _ = load_model(model_path, device=device)

        self.rng = random.Random(seed)
        self.env = GridWorld(w, h, p, self.rng)

        self.num_episodes = num_episodes
        self.cell_px = cell_px
        self.fps = max(1, fps)
        self.ms = int(round(1000 / self.fps))

        self.cur_episode = 0
        self.expert_len = 0
        self.max_steps = 0
        self.successes = 0
        self.total_steps = 0
        self.total_collisions = 0

        self.view = GridView()
        self.lbl = QLabel()

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(self.view)
        lay.addWidget(self.lbl)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        self._start_next_episode()

    def _policy_action(self) -> str:
        ep = self.env.ep
        img = render_image(ep.w, ep.h, self.cell_px, ep.obstacles, ep.start, ep.goal, ep.robot)
        x = qimage_to_tensor(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a_i = int(self.model(x).argmax(dim=1).item())
        return I2A[a_i]

    def _update_view(self):
        ep = self.env.ep
        img = render_image(ep.w, ep.h, self.cell_px, ep.obstacles, ep.start, ep.goal, ep.robot)
        self.view.set_image(img)

    def _start_next_episode(self):
        self.cur_episode += 1
        if self.cur_episode > self.num_episodes:
            sr = self.successes / max(1, self.num_episodes)
            self.lbl.setText(
                f"Done. success_rate={sr*100:.1f}% | "
                f"avg_steps={self.total_steps/max(1,self.num_episodes):.1f} | "
                f"avg_collisions={self.total_collisions/max(1,self.num_episodes):.1f}"
            )
            self.timer.stop()
            return

        ep = self.env.reset_solvable()
        path = bfs_path(ep.w, ep.h, ep.obstacles, ep.start, ep.goal)
        self.expert_len = max(1, len(path) - 1)
        self.max_steps = 5 * self.expert_len
        ep.steps = 0
        ep.collisions = 0

        self._update_view()
        self.lbl.setText(
            f"Episode {self.cur_episode}/{self.num_episodes} | expert X={self.expert_len} | "
            f"budget={self.max_steps} | steps=0 | collisions=0"
        )
        self.timer.start(self.ms)

    def _finish_episode(self, success: bool, reason: str):
        ep = self.env.ep
        self.total_steps += ep.steps
        self.total_collisions += ep.collisions
        if success:
            self.successes += 1

        self.lbl.setText(
            f"Episode {self.cur_episode}/{self.num_episodes} {('SUCCESS' if success else 'FAIL')} ({reason}) | "
            f"steps={ep.steps} collisions={ep.collisions} | (expert X={self.expert_len}, budget={self.max_steps})"
        )
        self.timer.stop()
        QTimer.singleShot(350, self._start_next_episode)

    def tick(self):
        ep = self.env.ep

        if ep.robot == ep.goal:
            self._finish_episode(True, "reached_goal")
            return

        if ep.steps >= self.max_steps:
            self._finish_episode(False, "budget_exhausted")
            return

        a = self._policy_action()
        self.env.step(a)
        self._update_view()
        self.lbl.setText(
            f"Episode {self.cur_episode}/{self.num_episodes} | a={a} | "
            f"steps={ep.steps}/{self.max_steps} | collisions={ep.collisions} | expert X={self.expert_len}"
        )


def mode_test_online(model_path: Path, episodes: int, w: int, h: int, p: float, cell: int,
                     seed: int | None, device_str: str, fps: int):
    device = resolve_device(device_str)
    app = QApplication.instance() or QApplication([])
    win = OnlineWindow(model_path, episodes, w, h, p, cell, seed, device, fps)
    win.show()
    app.exec()


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser("Robot imitation learning (all-in-one): record/train/test-offline/test-online.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # record
    ap_rec = sub.add_parser("record", help="Generate expert demonstrations into a dataset folder.")
    ap_rec.add_argument("--dataset", type=str, required=True)
    ap_rec.add_argument("--episodes", type=int, required=True)
    ap_rec.add_argument("--w", type=int, default=16)
    ap_rec.add_argument("--h", type=int, default=12)
    ap_rec.add_argument("--p", type=float, default=0.22, help="Obstacle probability per cell (except start/goal).")
    ap_rec.add_argument("--cell", type=int, default=32)
    ap_rec.add_argument("--seed", type=int, default=None)
    ap_rec.add_argument("--ep-digits", type=int, default=6)
    ap_rec.add_argument("--img-digits", type=int, default=6)

    # train
    ap_tr = sub.add_parser("train", help="Train CNN on first 80% episodes; validate on last 20%.")
    ap_tr.add_argument("--dataset", type=str, required=True)
    ap_tr.add_argument("--out-model", type=str, default="./cnn.pt")
    ap_tr.add_argument("--epochs", type=int, default=5)
    ap_tr.add_argument("--batch", type=int, default=64)
    ap_tr.add_argument("--lr", type=float, default=1e-3)
    ap_tr.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")

    # test-offline
    ap_to = sub.add_parser("test-offline", help="Accuracy on last 20% episodes from dataset (no visualization).")
    ap_to.add_argument("--model", type=str, required=True)
    ap_to.add_argument("--dataset", type=str, required=True)
    ap_to.add_argument("--batch", type=int, default=128)
    ap_to.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")

    # test-online
    ap_tn = sub.add_parser("test-online", help="Run N new episodes online; visualize CNN controlling the robot.")
    ap_tn.add_argument("--model", type=str, required=True)
    ap_tn.add_argument("--episodes", type=int, required=True)
    ap_tn.add_argument("--w", type=int, default=16)
    ap_tn.add_argument("--h", type=int, default=12)
    ap_tn.add_argument("--p", type=float, default=0.22)
    ap_tn.add_argument("--cell", type=int, default=32)
    ap_tn.add_argument("--seed", type=int, default=None)
    ap_tn.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap_tn.add_argument("--fps", type=int, default=8, help="Visualization update rate.")

    args = ap.parse_args()

    if args.cmd == "record":
        mode_record(Path(args.dataset), args.episodes, args.w, args.h, args.p, args.cell,
                    args.seed, args.ep_digits, args.img_digits)
    elif args.cmd == "train":
        mode_train(Path(args.dataset), Path(args.out_model), args.epochs, args.batch, args.lr, args.device)
    elif args.cmd == "test-offline":
        mode_test_offline(Path(args.model), Path(args.dataset), args.batch, args.device)
    elif args.cmd == "test-online":
        mode_test_online(Path(args.model), args.episodes, args.w, args.h, args.p, args.cell,
                         args.seed, args.device, args.fps)


if __name__ == "__main__":
    main()
