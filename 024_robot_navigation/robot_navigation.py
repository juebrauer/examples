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
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel


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


def path_to_actions(path):
    out = []
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx, dy = x1 - x0, y1 - y0
        if dx == 1 and dy == 0: out.append("R")
        elif dx == -1 and dy == 0: out.append("L")
        elif dx == 0 and dy == 1: out.append("D")
        elif dx == 0 and dy == -1: out.append("U")
        else: raise ValueError("Non-4-neighborhood step.")
    return out


class GridWorld:
    def __init__(self, w, h, p, rng):
        self.w, self.h, self.p = int(w), int(h), float(p)
        self.rng = rng
        self.ep: Episode | None = None

    def reset_solvable(self, max_tries=5000):
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

            self.ep = Episode(self.w, self.h, start, goal, start, obstacles)
            return self.ep

        raise RuntimeError("Could not generate solvable episode. Lower --p.")

    def step(self, action):
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
# Rendering (dataset + online)
# ----------------------------

def render_image(w, h, cell_px, obstacles, start, goal, robot):
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

    # robot
    rx, ry = robot
    m = max(2, cell_px // 10)
    p.setBrush(QColor(220, 120, 120)); p.setPen(QColor(220, 120, 120))
    p.drawRect(rx * cell_px + m, ry * cell_px + m, cell_px - 2*m, cell_px - 2*m)

    # grid lines
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
    """Returns float tensor (3,H,W) in [0,1]."""
    img = img.convertToFormat(QImage.Format_RGB888)
    w = img.width()
    h = img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 3)
    data = torch.frombuffer(ptr, dtype=torch.uint8).clone()  # clone -> own memory
    data = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return data


# ----------------------------
# Dataset reading (episode split)
# ----------------------------

def read_episodes_csv(dataset_dir: Path):
    path = dataset_dir / "episodes.csv"
    episodes = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            episodes.append((int(row["episode"]), int(row["N"])))
    episodes.sort(key=lambda x: x[0])
    return episodes


def read_actions_csv(dataset_dir: Path):
    path = dataset_dir / "actions.csv"
    rows = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ep = int(row["episode"])
            im = int(row["image"])
            a = row["action"].strip()
            rows.append((ep, im, a))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


class DemoDataset(Dataset):
    def __init__(self, dataset_dir: Path, episodes_set: set[int], ep_digits=6, img_digits=6):
        self.dataset_dir = Path(dataset_dir)
        self.ep_digits = ep_digits
        self.img_digits = img_digits

        all_actions = read_actions_csv(self.dataset_dir)
        self.samples = [(ep, im, a) for (ep, im, a) in all_actions if ep in episodes_set]

    def __len__(self):
        return len(self.samples)

    def _img_path(self, ep, im):
        ep_s = str(ep).zfill(self.ep_digits)
        im_s = str(im).zfill(self.img_digits)
        return self.dataset_dir / f"episode{ep_s}-{im_s}.png"

    def __getitem__(self, idx):
        ep, im, a = self.samples[idx]
        path = self._img_path(ep, im)
        img = QImage(str(path))
        x = qimage_to_tensor(img)
        y = torch.tensor(A2I[a], dtype=torch.long)
        return x, y


# ----------------------------
# CNN model
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
        x = self.head(x)
        return x


def save_model(path: Path, model: nn.Module, meta: dict):
    path = Path(path)
    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, str(path))


def load_model(path: Path, device):
    payload = torch.load(str(path), map_location=device)
    meta = payload.get("meta", {})
    model = SmallCNN(num_actions=4).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


# ----------------------------
# Train / offline test
# ----------------------------

def split_episodes_80_20(dataset_dir: Path):
    eps = read_episodes_csv(dataset_dir)
    ep_ids = [e for (e, _) in eps]
    if len(ep_ids) < 2:
        raise RuntimeError("Need at least 2 episodes to do an 80/20 split.")
    cut = max(1, int(math.floor(0.8 * len(ep_ids))))
    train_ids = set(ep_ids[:cut])
    test_ids = set(ep_ids[cut:])
    return train_ids, test_ids


def train_cnn(dataset_dir: Path, out_model: Path, epochs=5, batch_size=64, lr=1e-3, device="cpu"):
    train_eps, test_eps = split_episodes_80_20(dataset_dir)
    ds_train = DemoDataset(dataset_dir, train_eps)
    ds_test = DemoDataset(dataset_dir, test_eps)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def eval_acc():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / max(1, total)

    for ep in range(1, epochs + 1):
        running = 0.0
        seen = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        acc = eval_acc()
        print(f"epoch {ep}/{epochs} | train loss {running/max(1,seen):.4f} | val acc {acc*100:.2f}%")

    meta = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "actions": ACTIONS,
        "split": "first80_train_last20_test_by_episode_order",
    }
    save_model(out_model, model, meta)
    print(f"Saved model to {Path(out_model).resolve()}")


def test_offline(model_path: Path, dataset_dir: Path, batch_size=128, device="cpu"):
    _, test_eps = split_episodes_80_20(dataset_dir)
    ds = DemoDataset(dataset_dir, test_eps)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

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
        "model": str(Path(model_path).resolve()),
        "dataset": str(Path(dataset_dir).resolve()),
        "num_samples": total,
        "accuracy": acc
    }, indent=2))


# ----------------------------
# Online test with visualization
# ----------------------------

class GridView(QWidget):
    def __init__(self, cell_px=32):
        super().__init__()
        self.cell_px = cell_px
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
    def __init__(self, model_path, num_episodes, w, h, p, cell_px, seed, device):
        super().__init__()
        self.setWindowTitle("test-online: CNN controls the robot (PySide6 live)")

        self.device = device
        self.model, _ = load_model(model_path, device=device)

        self.rng = random.Random(seed)
        self.env = GridWorld(w, h, p, self.rng)

        self.num_episodes = num_episodes
        self.cell_px = cell_px

        self.cur_ep = 0
        self.max_steps = 0
        self.expert_len = 0

        self.view = GridView(cell_px=cell_px)
        self.lbl = QLabel()

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(self.view)
        lay.addWidget(self.lbl)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        self._start_next_episode()

    def _start_next_episode(self):
        self.cur_ep += 1
        if self.cur_ep > self.num_episodes:
            self.lbl.setText("Done.")
            self.timer.stop()
            return

        ep = self.env.reset_solvable()
        path = bfs_path(ep.w, ep.h, ep.obstacles, ep.start, ep.goal)
        self.expert_len = len(path) - 1
        self.max_steps = max(1, 5 * self.expert_len)

        ep.steps = 0
        ep.collisions = 0

        self._update_view()
        self.lbl.setText(
            f"Episode {self.cur_ep}/{self.num_episodes} | expert X={self.expert_len} | "
            f"budget={self.max_steps} | steps={ep.steps} | collisions={ep.collisions}"
        )

        self.timer.start(120)

    def _policy_action(self):
        ep = self.env.ep
        img = render_image(ep.w, ep.h, self.cell_px, ep.obstacles, ep.start, ep.goal, ep.robot)
        x = qimage_to_tensor(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            a_i = int(logits.argmax(dim=1).item())
        return I2A[a_i]

    def _update_view(self):
        ep = self.env.ep
        img = render_image(ep.w, ep.h, self.cell_px, ep.obstacles, ep.start, ep.goal, ep.robot)
        self.view.set_image(img)

    def tick(self):
        ep = self.env.ep

        if ep.robot == ep.goal:
            self.lbl.setText(
                f"Episode {self.cur_ep}/{self.num_episodes} SUCCESS | "
                f"steps={ep.steps} collisions={ep.collisions} | (expert X={self.expert_len}, budget={self.max_steps})"
            )
            self.timer.stop()
            QTimer.singleShot(400, self._start_next_episode)
            return

        if ep.steps >= self.max_steps:
            self.lbl.setText(
                f"Episode {self.cur_ep}/{self.num_episodes} FAIL (budget) | "
                f"steps={ep.steps} collisions={ep.collisions} | (expert X={self.expert_len}, budget={self.max_steps})"
            )
            self.timer.stop()
            QTimer.singleShot(400, self._start_next_episode)
            return

        a = self._policy_action()
        self.env.step(a)
        self._update_view()
        self.lbl.setText(
            f"Episode {self.cur_ep}/{self.num_episodes} | a={a} | steps={ep.steps}/{self.max_steps} | "
            f"collisions={ep.collisions} | expert X={self.expert_len}"
        )


def test_online(model_path: Path, num_episodes: int, w: int, h: int, p: float, cell: int, seed: int | None, device: str):
    app = QApplication.instance() or QApplication([])
    win = OnlineWindow(model_path, num_episodes, w, h, p, cell, seed, device)
    win.show()
    app.exec()


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser("Robot imitation learning: train/test offline + online.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train CNN on first 80% of episodes, validate on last 20%.")
    ap_train.add_argument("--dataset", type=str, required=True)
    ap_train.add_argument("--out-model", type=str, default="./cnn.pt")
    ap_train.add_argument("--epochs", type=int, default=5)
    ap_train.add_argument("--batch", type=int, default=64)
    ap_train.add_argument("--lr", type=float, default=1e-3)
    ap_train.add_argument("--device", type=str, default="cpu")

    ap_to = sub.add_parser("test-offline", help="Accuracy on last 20% of episodes from dataset (no visualization).")
    ap_to.add_argument("--model", type=str, required=True)
    ap_to.add_argument("--dataset", type=str, required=True)
    ap_to.add_argument("--batch", type=int, default=128)
    ap_to.add_argument("--device", type=str, default="cpu")

    ap_tn = sub.add_parser("test-online", help="Run N new episodes online; visualize CNN controlling robot.")
    ap_tn.add_argument("--model", type=str, required=True)
    ap_tn.add_argument("--episodes", type=int, required=True)
    ap_tn.add_argument("--w", type=int, default=16)
    ap_tn.add_argument("--h", type=int, default=12)
    ap_tn.add_argument("--p", type=float, default=0.22)
    ap_tn.add_argument("--cell", type=int, default=32)
    ap_tn.add_argument("--seed", type=int, default=None)
    ap_tn.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    if args.cmd == "train":
        train_cnn(Path(args.dataset), Path(args.out_model),
                  epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
    elif args.cmd == "test-offline":
        test_offline(Path(args.model), Path(args.dataset), batch_size=args.batch, device=args.device)
    elif args.cmd == "test-online":
        test_online(Path(args.model), args.episodes, args.w, args.h, args.p, args.cell, args.seed, args.device)


if __name__ == "__main__":
    main()
