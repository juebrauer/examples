"""
robot_navigation.py

ONE single script with 4 modes:
  - record       : generate expert (BFS) demonstrations into a dataset folder
  - train        : train a CNN classifier on first 80% of episodes (by episode order, or shuffled)
  - test-offline : evaluate accuracy on last 20% of episodes (no visualization)
  - test-online  : run N new episodes online, CNN controls robot, live visualization, budget=5*X

Dataset format:
  images: episode<ep>-<img>.png  (leading zeros)
  actions.csv : episode,image,action
  episodes.csv: episode,N

Examples
--------
# 1) record demonstrations
QT_QPA_PLATFORM=offscreen python robot_navigation.py record --dataset ./dataset500 --episodes 500 --w 16 --h 12 --p 0.22

# 2) train CNN (AdamW weight decay + best checkpoint + scheduler + plots)
python robot_navigation.py train --dataset ./dataset500 --out-model cnn.pt --epochs 30 --device auto \
  --lr 5e-4 --weight-decay 1e-4 --sched plateau --patience 8 --split-seed 0

# 3) offline test accuracy (last 20% episodes)
python robot_navigation.py test-offline --model cnn.pt.best.pt --dataset ./dataset500 --device auto

# 4) online test (new episodes, live)
python robot_navigation.py test-online --model cnn.pt.best.pt --episodes 30 --w 16 --h 12 --p 0.22 --device auto

What this patch adds
--------------------
- Training now computes BOTH val_loss and val_acc each epoch.
- Saves the best checkpoint automatically:
    <out-model>.best.pt   (chosen by lowest val_loss)
- Optional early stopping: --patience N (based on val_loss)
- Optional LR scheduling: --sched {none,plateau,cosine}
- Optional shuffled split: --split-seed SEED (deterministic shuffle before 80/20 split)
- Logs to CSV: <out-model>.train_log.csv
- Saves training curves plot: <out-model>.train_curves.png
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

from PySide6.QtCore import QTimer
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

    # grid lines (keep)
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

    # clone() makes it safe; warning about non-writable buffer can be ignored here
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


def split_episodes_80_20(dataset_dir: Path, split_seed: int | None):
    """
    If split_seed is None: keep original episode order (your original behavior).
    If split_seed is set: shuffle deterministically before splitting (reduces order bias).
    """
    eps = read_episodes_csv(dataset_dir)
    ep_ids = [e for (e, _) in eps]
    if len(ep_ids) < 2:
        raise RuntimeError("Need at least 2 episodes in episodes.csv for an 80/20 split.")

    if split_seed is not None:
        rng = random.Random(int(split_seed))
        rng.shuffle(ep_ids)

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
# CNN (deeper + spatial head)
# ----------------------------

class DeeperSpatialCNN(nn.Module):
    """
    Deeper CNN while keeping spatial info:
      backbone -> AdaptiveAvgPool2d((4,4)) -> MLP head
    """
    def __init__(self, num_actions=4):
        super().__init__()

        def block(cin, cout, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            # stage 1 (/2)
            block(3, 32, k=5, s=2, p=2),
            block(32, 32),
            block(32, 32),

            # stage 2 (/4)
            block(32, 64, s=2),
            block(64, 64),
            block(64, 64),

            # stage 3 (/8)
            block(64, 128, s=2),
            block(128, 128),
            block(128, 128),

            # stage 4 (/16)
            block(128, 192, s=2),
            block(192, 192),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),                # 192*4*4 = 3072
            nn.Linear(192 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


def save_model(path: Path, model: nn.Module, meta: dict):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(Path(path)))


def load_model(path: Path, device: str):
    payload = torch.load(str(Path(path)), map_location=device)
    model = DeeperSpatialCNN(num_actions=4).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})


def resolve_device(device_str: str) -> str:
    d = device_str.strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


# ----------------------------
# Mode: record
# ----------------------------

def mode_record(dataset_dir: Path, episodes: int, w: int, h: int, p: float, cell: int,
                seed: int | None, ep_digits: int, img_digits: int):
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

def _write_train_log_csv(path: Path, rows: list[dict]):
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "is_best"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _save_train_curves_png(path: Path, rows: list[dict], title: str):
    # Import here so non-train modes don't require matplotlib at runtime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    fig = plt.figure(figsize=(8.5, 4.8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(epochs, train_loss, marker="o", label="train loss")
    ax1.plot(epochs, val_loss, marker="o", label="val loss")
    ax2.plot(epochs, val_acc, marker="o", label="val acc")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("accuracy")

    ax1.set_title(title)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def mode_train(
    dataset_dir: Path,
    out_model: Path,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    device_str: str,
    split_seed: int | None,
    sched: str,
    patience: int,
):
    device = resolve_device(device_str)
    train_eps, test_eps = split_episodes_80_20(dataset_dir, split_seed=split_seed)

    ds_train = DemoDataset(dataset_dir, train_eps)
    ds_test = DemoDataset(dataset_dir, test_eps)

    if len(ds_train) == 0 or len(ds_test) == 0:
        raise RuntimeError("Train/test split produced empty set. Add more episodes.")

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=max(64, batch), shuffle=False, num_workers=0)

    model = DeeperSpatialCNN().to(device)

    # AdamW = Adam + decoupled weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if sched == "plateau":
        # PyTorch versions differ: some accept `verbose`, some don't.
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6, verbose=True
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6
            )
    elif sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    def eval_val():
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                loss_sum += float(loss.item())
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        val_loss = loss_sum / max(1, total)
        val_acc = correct / max(1, total)
        return val_loss, val_acc

    out_model = Path(out_model)
    best_path = out_model.with_suffix(out_model.suffix + ".best.pt")
    log_csv = out_model.with_suffix(out_model.suffix + ".train_log.csv")
    plot_png = out_model.with_suffix(out_model.suffix + ".train_curves.png")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    bad_epochs = 0

    log_rows: list[dict] = []

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

        train_loss = running / max(1, seen)
        val_loss, val_acc = eval_val()

        cur_lr = float(opt.param_groups[0]["lr"])
        is_best = val_loss < best_val_loss - 1e-10  # tiny epsilon to avoid float tie noise

        if is_best:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            bad_epochs = 0
            meta = {
                "dataset_dir": str(Path(dataset_dir).resolve()),
                "actions": ACTIONS,
                "split": ("shuffled80_20_by_seed" if split_seed is not None else "first80_train_last20_test_by_episode_order"),
                "split_seed": split_seed,
                "device_trained": device,
                "arch": "DeeperSpatialCNN(pool=4x4,mlp_head)",
                "optimizer": "AdamW",
                "lr": lr,
                "weight_decay": weight_decay,
                "batch": batch,
                "sched": sched,
                "best_epoch": e,
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
            }
            save_model(best_path, model, meta)
        else:
            bad_epochs += 1

        print(
            f"epoch {e}/{epochs} | lr {cur_lr:.2e} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc*100:.2f}%"
            f"{'  [BEST]' if is_best else ''}"
        )

        log_rows.append(
            {
                "epoch": e,
                "lr": cur_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "is_best": int(is_best),
            }
        )

        # scheduler step
        if scheduler is not None:
            if sched == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # early stopping
        if patience > 0 and bad_epochs >= patience:
            print(f"Early stopping: no val_loss improvement for {bad_epochs} epochs (patience={patience}).")
            break

    # Save final model too (useful for debugging); best is in *.best.pt
    final_meta = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "actions": ACTIONS,
        "split": ("shuffled80_20_by_seed" if split_seed is not None else "first80_train_last20_test_by_episode_order"),
        "split_seed": split_seed,
        "device_trained": device,
        "arch": "DeeperSpatialCNN(pool=4x4,mlp_head)",
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "batch": batch,
        "sched": sched,
        "epochs_ran": int(log_rows[-1]["epoch"]) if log_rows else 0,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    save_model(out_model, model, final_meta)

    _write_train_log_csv(log_csv, log_rows)
    _save_train_curves_png(plot_png, log_rows, title=f"Training curves ({out_model.name})")

    print(f"Saved final model to: {out_model.resolve()}")
    print(f"Saved best model to : {best_path.resolve()} (by lowest val_loss)")
    print(f"Wrote train log     : {log_csv.resolve()}")
    print(f"Wrote train plot    : {plot_png.resolve()}")


# ----------------------------
# Mode: test-offline
# ----------------------------

def mode_test_offline(model_path: Path, dataset_dir: Path, batch: int, device_str: str, split_seed: int | None):
    device = resolve_device(device_str)
    _, test_eps = split_episodes_80_20(dataset_dir, split_seed=split_seed)
    ds = DemoDataset(dataset_dir, test_eps)
    if len(ds) == 0:
        raise RuntimeError("Offline test set empty. Add more episodes.")
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    model, meta = load_model(model_path, device=device)

    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / max(1, total)
    loss = loss_sum / max(1, total)
    print(json.dumps(
        {
            "mode": "test-offline",
            "model": str(Path(model_path).resolve()),
            "dataset": str(Path(dataset_dir).resolve()),
            "split_seed": split_seed,
            "num_samples": total,
            "loss": loss,
            "accuracy": acc,
            "meta": meta,
        },
        indent=2,
    ))


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
    ap_tr = sub.add_parser("train", help="Train CNN on 80% episodes; validate on 20%.")
    ap_tr.add_argument("--dataset", type=str, required=True)
    ap_tr.add_argument("--out-model", type=str, default="./cnn.pt")
    ap_tr.add_argument("--epochs", type=int, default=20)
    ap_tr.add_argument("--batch", type=int, default=64)
    ap_tr.add_argument("--lr", type=float, default=5e-4)
    ap_tr.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    ap_tr.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap_tr.add_argument("--split-seed", type=int, default=None, help="If set, shuffle episodes before split (deterministic).")
    ap_tr.add_argument("--sched", type=str, default="plateau", choices=["none", "plateau", "cosine"],
                      help="Learning-rate scheduler.")
    ap_tr.add_argument("--patience", type=int, default=0,
                      help="Early stopping patience (epochs without val_loss improvement). 0 disables.")

    # test-offline
    ap_to = sub.add_parser("test-offline", help="Evaluate on last 20% episodes from dataset (no visualization).")
    ap_to.add_argument("--model", type=str, required=True)
    ap_to.add_argument("--dataset", type=str, required=True)
    ap_to.add_argument("--batch", type=int, default=128)
    ap_to.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap_to.add_argument("--split-seed", type=int, default=None, help="Must match training split if you used shuffling.")

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
        mode_train(
            dataset_dir=Path(args.dataset),
            out_model=Path(args.out_model),
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device_str=args.device,
            split_seed=args.split_seed,
            sched=args.sched,
            patience=args.patience,
        )

    elif args.cmd == "test-offline":
        mode_test_offline(
            model_path=Path(args.model),
            dataset_dir=Path(args.dataset),
            batch=args.batch,
            device_str=args.device,
            split_seed=args.split_seed,
        )

    elif args.cmd == "test-online":
        mode_test_online(Path(args.model), args.episodes, args.w, args.h, args.p, args.cell,
                         args.seed, args.device, args.fps)


if __name__ == "__main__":
    main()