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

# 2) train CNN into its own run directory (model + log + plots)
python robot_navigation.py train --dataset ./dataset500 --run-dir ./runs/cnn_try1 --epochs 30 --device auto \
  --lr 5e-4 --weight-decay 1e-4 --sched plateau --patience 8 --split-seed 0

# 2b) train a pretrained ResNet18 instead
python robot_navigation.py train --dataset ./dataset500 --run-dir ./runs/resnet_try1 --epochs 30 --device auto --arch resnet18

# 2c) train a pretrained ViT-B/16 instead
python robot_navigation.py train --dataset ./dataset500 --run-dir ./runs/vit_try1 --epochs 30 --device auto --arch vit

# 3) offline test accuracy (last 20% episodes)
python robot_navigation.py test-offline --model ./runs/cnn_try1/cnn.pt --dataset ./dataset500 --device auto

# 4) online test (new episodes, live)
python robot_navigation.py test-online --model ./runs/cnn_try1/cnn.pt --episodes 30 --w 16 --h 12 --p 0.22 --device auto

What this patch adds
--------------------
- `train` writes all artifacts to a per-run directory (`--run-dir`):
    - best model: <arch>.pt (chosen by lowest val_loss)
    - training log: training.log
    - curves plot: train_curves.png (train/val loss + train/val acc)
    - optional CSV log: train_log.csv
- Optional early stopping: --patience N (based on val_loss)
- Optional LR scheduling: --sched {none,plateau,cosine}
- Optional shuffled split: --split-seed SEED (deterministic shuffle before 80/20 split)
- `test-online` appends a Markdown table row to overall_results.md (and also overall_results.txt) in this script folder.
"""

import argparse
import csv
import json
import math
import logging
import random
from datetime import datetime
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


def load_image_as_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image as float tensor (3,H,W) in [0,1].

    Training/test-offline can be bottlenecked by image decode + Qt conversion.
    If torchvision is available, use torchvision.io.read_image (fast, worker-friendly).
    Otherwise fall back to QImage.
    """
    p = str(Path(path))
    try:
        from torchvision.io import read_image
        x = read_image(p)  # uint8 (C,H,W), typically RGB
        if x.dim() != 3 or x.size(0) not in (1, 3, 4):
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.size(0) == 4:
            x = x[:3]
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x.float() / 255.0
    except Exception:
        img = QImage(p)
        return qimage_to_tensor(img)


def pad_to_square(x: torch.Tensor, *, fill: float = 0.5) -> torch.Tensor:
    """Pad (3,H,W) tensor to square with constant fill."""
    if x.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")
    _, h, w = x.shape
    if h == w:
        return x
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # F.pad pads last dims: (left, right, top, bottom)
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=float(fill))


def resize_chw(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (3,H,W) tensor to (3,size,size) using bilinear."""
    if x.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")
    size_i = int(size)
    if x.shape[-2] == size_i and x.shape[-1] == size_i:
        return x
    return F.interpolate(x.unsqueeze(0), size=(size_i, size_i), mode="bilinear", align_corners=False).squeeze(0)


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
    def __init__(
        self,
        dataset_dir: Path,
        episodes_set: set[int],
        ep_digits=6,
        img_digits=6,
        transform=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.ep_digits = ep_digits
        self.img_digits = img_digits
        self.transform = transform
        self.samples = [(ep, im, a) for (ep, im, a) in read_actions_csv(self.dataset_dir) if ep in episodes_set]

    def __len__(self):
        return len(self.samples)

    def _img_path(self, ep, im):
        ep_s = str(ep).zfill(self.ep_digits)
        im_s = str(im).zfill(self.img_digits)
        return self.dataset_dir / f"episode{ep_s}-{im_s}.png"

    def __getitem__(self, idx):
        ep, im, a = self.samples[idx]
        x = load_image_as_tensor(self._img_path(ep, im))
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(A2I[a], dtype=torch.long)
        return x, y


# ----------------------------
# CNN (deeper + spatial head)
# ----------------------------

class _Normalize(nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std", std_t, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ResNet18Classifier(nn.Module):
    def __init__(self, num_actions: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:
            raise RuntimeError(
                "torchvision is required for --arch resnet18. Install it via pip/conda (matching your torch version)."
            ) from e

        # Newer torchvision uses Weights enums; older versions use `pretrained=True`.
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=bool(pretrained))

        if pretrained:
            # ImageNet normalization expected for pretrained weights
            self.normalize = _Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = nn.Identity()

        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Linear(in_features, num_actions)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.backbone(x)


class ViTB16Classifier(nn.Module):
    def __init__(
        self,
        num_actions: int = 4,
        pretrained: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = int(image_size)
        try:
            from torchvision import models
        except Exception as e:
            raise RuntimeError(
                "torchvision is required for --arch vit (ViT). Install it via pip/conda (matching your torch version)."
            ) from e

        # Newer torchvision uses Weights enums; older versions use `pretrained=True`.
        try:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            if pretrained:
                # Use the transform's normalization constants when available.
                mean = list(weights.transforms().mean)
                std = list(weights.transforms().std)
                self.normalize = _Normalize(mean=mean, std=std)
            else:
                self.normalize = nn.Identity()
        except Exception:
            backbone = models.vit_b_16(pretrained=bool(pretrained))
            if pretrained:
                # Fallback to the common ImageNet normalization.
                self.normalize = _Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else:
                self.normalize = nn.Identity()

        # Replace classification head.
        # torchvision ViT uses `heads.head` (Linear) in common versions.
        if hasattr(backbone, "heads") and hasattr(backbone.heads, "head"):
            in_features = int(backbone.heads.head.in_features)
            backbone.heads.head = nn.Linear(in_features, num_actions)
        else:
            # Defensive fallback if torchvision changes internals.
            raise RuntimeError("Unsupported torchvision ViT structure: expected backbone.heads.head")

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT in torchvision expects a fixed input resolution (typically 224x224).
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B,3,H,W), got {tuple(x.shape)}")
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self.normalize(x)
        return self.backbone(x)

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


def build_model(arch: str, *, num_actions: int = 4) -> nn.Module:
    arch_n = (arch or "").strip().lower()
    if arch_n in ("cnn", "deepercnn", "deeperspatialcnn"):
        return DeeperSpatialCNN(num_actions=num_actions)
    if arch_n in ("resnet18", "resnet"):
        return ResNet18Classifier(num_actions=num_actions, pretrained=True)
    if arch_n in ("vit", "vit_b_16", "vitb16", "vit-b-16"):
        return ViTB16Classifier(num_actions=num_actions, pretrained=True, image_size=224)
    raise ValueError(f"Unknown --arch '{arch}'. Expected: cnn, resnet18, vit")


def save_model(path: Path, model: nn.Module, meta: dict, *, arch: str):
    torch.save({"state_dict": model.state_dict(), "meta": meta, "arch": str(arch)}, str(Path(path)))


def load_model(path: Path, device: str, *, arch_override: str | None = None):
    payload = torch.load(str(Path(path)), map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        meta = payload.get("meta", {})
        arch_in_ckpt = payload.get("arch")
    else:
        # Backward compatibility: allow loading plain state_dict checkpoints
        state_dict = payload
        meta = {}
        arch_in_ckpt = None

    arch = (arch_override or arch_in_ckpt or meta.get("arch_id") or "cnn")
    model = build_model(arch, num_actions=4).to(device)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint '{Path(path)}' with --arch {arch}. "
            "If this is an older CNN checkpoint, try --arch cnn; for ResNet checkpoints, use --arch resnet18."
        ) from e

    model.eval()
    return model, meta


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
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "is_best"]
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
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True)

    ax_loss.plot(epochs, train_loss, marker="o", label="train loss")
    ax_loss.plot(epochs, val_loss, marker="o", label="val loss")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_loss.legend(loc="best")

    ax_acc.plot(epochs, train_acc, marker="o", label="train acc")
    ax_acc.plot(epochs, val_acc, marker="o", label="val acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_acc.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _setup_train_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"train:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if mode_train is called multiple times in-process.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(run_dir / "training.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _default_run_dir(arch: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_arch = (arch or "cnn").strip().lower()
    return Path("./runs") / f"{ts}_{safe_arch}"


def mode_train(
    dataset_dir: Path,
    out_model: Path,
    run_dir: Path | None,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    device_str: str,
    split_seed: int | None,
    sched: str,
    patience: int,
    arch: str,
    amp: str,
    num_workers: int,
    pin_memory: str,
    vit_image_size: int,
    vit_pad_square: bool,
    freeze_backbone: bool,
):
    device = resolve_device(device_str)
    train_eps, test_eps = split_episodes_80_20(dataset_dir, split_seed=split_seed)

    run_dir = Path(run_dir) if run_dir is not None else None
    if run_dir is None:
        run_dir = _default_run_dir(arch)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_train_logger(run_dir)
    logger.info(f"run_dir={run_dir.resolve()}")
    logger.info(f"dataset_dir={Path(dataset_dir).resolve()}")
    logger.info(f"arch={arch} device={device} epochs={epochs} batch={batch} lr={lr} weight_decay={weight_decay}")
    logger.info(f"split_seed={split_seed} sched={sched} patience={patience}")

    if device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    arch_n = (arch or "").strip().lower()

    amp_enabled = False
    if (amp or "auto").lower() == "on":
        amp_enabled = (device == "cuda")
    elif (amp or "auto").lower() == "off":
        amp_enabled = False
    else:
        # auto
        amp_enabled = (device == "cuda" and arch_n in ("resnet18", "resnet", "vit", "vit_b_16", "vitb16", "vit-b-16"))

    if (pin_memory or "auto").lower() == "on":
        pin_memory_enabled = True
    elif (pin_memory or "auto").lower() == "off":
        pin_memory_enabled = False
    else:
        pin_memory_enabled = (device == "cuda")

    logger.info(
        f"amp={amp} (enabled={amp_enabled}) num_workers={int(num_workers)} pin_memory={pin_memory} (enabled={pin_memory_enabled}) "
        f"vit_image_size={int(vit_image_size)} vit_pad_square={bool(vit_pad_square)} freeze_backbone={bool(freeze_backbone)}"
    )

    def _transform(x: torch.Tensor) -> torch.Tensor:
        if arch_n in ("vit", "vit_b_16", "vitb16", "vit-b-16"):
            if vit_pad_square:
                x = pad_to_square(x, fill=0.5)
            x = resize_chw(x, vit_image_size)
        return x

    ds_train = DemoDataset(dataset_dir, train_eps, transform=_transform)
    ds_test = DemoDataset(dataset_dir, test_eps, transform=_transform)

    if len(ds_train) == 0 or len(ds_test) == 0:
        raise RuntimeError("Train/test split produced empty set. Add more episodes.")

    dl_train = DataLoader(
        ds_train,
        batch_size=batch,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory_enabled),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=max(64, batch),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory_enabled),
    )

    model = build_model(arch).to(device)

    if freeze_backbone:
        if arch_n in ("resnet18", "resnet"):
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.backbone.fc.parameters():
                p.requires_grad = True
        elif arch_n in ("vit", "vit_b_16", "vitb16", "vit-b-16"):
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.backbone.heads.head.parameters():
                p.requires_grad = True
        # CNN is trained end-to-end by default.

    # AdamW = Adam + decoupled weight decay
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters (did you freeze everything?)")
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

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
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = F.cross_entropy(logits, y, reduction="sum")
                else:
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

    # Save all artifacts into the run_dir.
    arch_name = (arch or "cnn").strip().lower()
    best_path = run_dir / f"{arch_name}.pt"
    log_csv = run_dir / "train_log.csv"
    plot_png = run_dir / "train_curves.png"

    best_val_loss = float("inf")
    best_val_acc = 0.0
    bad_epochs = 0

    log_rows: list[dict] = []

    scaler = None
    if amp_enabled:
        scaler = torch.cuda.amp.GradScaler()

    for e in range(1, epochs + 1):
        running = 0.0
        seen = 0
        correct = 0

        for x, y in dl_train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        train_loss = running / max(1, seen)
        train_acc = correct / max(1, seen)
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
                "arch_id": str(arch),
                "optimizer": "AdamW",
                "lr": lr,
                "weight_decay": weight_decay,
                "batch": batch,
                "sched": sched,
                "amp": amp,
                "amp_enabled": bool(amp_enabled),
                "freeze_backbone": bool(freeze_backbone),
                "num_workers": int(num_workers),
                "pin_memory": str(pin_memory),
                "vit_image_size": int(vit_image_size),
                "vit_pad_square": bool(vit_pad_square),
                "best_epoch": e,
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
            }
            save_model(best_path, model, meta, arch=str(arch))
        else:
            bad_epochs += 1

        logger.info(
            f"epoch {e}/{epochs} | lr {cur_lr:.2e} | "
            f"train loss {train_loss:.4f} | train acc {train_acc*100:.2f}% | "
            f"val loss {val_loss:.4f} | val acc {val_acc*100:.2f}%"
            f"{'  [BEST]' if is_best else ''}"
        )

        log_rows.append(
            {
                "epoch": e,
                "lr": cur_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
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
            logger.info(f"Early stopping: no val_loss improvement for {bad_epochs} epochs (patience={patience}).")
            break

    _write_train_log_csv(log_csv, log_rows)
    _save_train_curves_png(plot_png, log_rows, title=f"Training curves ({arch_name})")

    logger.info(f"Saved best model to: {best_path.resolve()} (by lowest val_loss)")
    logger.info(f"Wrote train log CSV: {log_csv.resolve()}")
    logger.info(f"Wrote train plot   : {plot_png.resolve()}")


def _append_overall_results(
    *,
    model_path: Path,
    arch: str,
    episodes: int,
    successes: int,
    w: int,
    h: int,
    p: float,
    seed: int | None,
    device: str,
):
    root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = "| timestamp | arch | model | episodes | successes | success_rate | w | h | p | seed | device |\n"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"

    model_s = f"`{Path(model_path).resolve()}`"
    sr = successes / max(1, episodes)
    row = (
        f"| {ts} | {arch} | {model_s} | {int(episodes)} | {int(successes)} | {sr:.4f} | "
        f"{int(w)} | {int(h)} | {float(p):.4f} | {seed} | {device} |\n"
    )

    def _ensure_header(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            with path.open("a", encoding="utf-8") as f:
                f.write(header)
                f.write(sep)
            return

        # Check tail so we don't re-add headers even if the file started as plain text.
        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                if size > 8192:
                    f.seek(max(0, size - 8192))
                tail = f.read()
            if header.strip() in tail:
                return
        except Exception:
            # If we fail to read the file, fall back to just appending.
            return

        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write(header)
            f.write(sep)

    # Write to both .md (easy viewing) and .txt (backward compatibility).
    for out_path in (root / "overall_results.md", root / "overall_results.txt"):
        _ensure_header(out_path)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(row)


# ----------------------------
# Mode: test-offline
# ----------------------------

def mode_test_offline(
    model_path: Path,
    dataset_dir: Path,
    batch: int,
    device_str: str,
    split_seed: int | None,
    arch_override: str | None,
):
    device = resolve_device(device_str)
    _, test_eps = split_episodes_80_20(dataset_dir, split_seed=split_seed)
    ds = DemoDataset(dataset_dir, test_eps)
    if len(ds) == 0:
        raise RuntimeError("Offline test set empty. Add more episodes.")
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    model, meta = load_model(model_path, device=device, arch_override=arch_override)

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
    def __init__(
        self,
        model_path: Path,
        num_episodes: int,
        w: int,
        h: int,
        p: float,
        cell_px: int,
        seed: int | None,
        device: str,
        fps: int,
        arch_override: str | None,
    ):
        super().__init__()

        self.device = device
        self.model_path = Path(model_path)
        self.model, meta = load_model(model_path, device=device, arch_override=arch_override)
        arch_label = arch_override or meta.get("arch_id") or "cnn"
        self.arch_label = str(arch_label)
        self.setWindowTitle(f"test-online: {arch_label} controls the robot (live)")

        self.rng = random.Random(seed)
        self.env = GridWorld(w, h, p, self.rng)

        self.w = int(w)
        self.h = int(h)
        self.p = float(p)
        self.seed = seed

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
            # Append summary line for tracking across runs.
            try:
                _append_overall_results(
                    model_path=self.model_path,
                    arch=self.arch_label,
                    episodes=self.num_episodes,
                    successes=self.successes,
                    w=self.w,
                    h=self.h,
                    p=self.p,
                    seed=self.seed,
                    device=self.device,
                )
            except Exception as e:
                print(f"Warning: failed to append overall_results.md/.txt: {e}")
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


def mode_test_online(
    model_path: Path,
    episodes: int,
    w: int,
    h: int,
    p: float,
    cell: int,
    seed: int | None,
    device_str: str,
    fps: int,
    arch_override: str | None,
    headless: bool,
):
    device = resolve_device(device_str)

    if headless:
        _app = QGuiApplication.instance() or QGuiApplication([])
        rng = random.Random(seed)
        env = GridWorld(w, h, p, rng)
        model, meta = load_model(model_path, device=device, arch_override=arch_override)
        arch_label = arch_override or meta.get("arch_id") or "cnn"

        successes = 0
        total_steps = 0
        total_collisions = 0

        for ep_i in range(1, episodes + 1):
            ep = env.reset_solvable()
            path = bfs_path(ep.w, ep.h, ep.obstacles, ep.start, ep.goal)
            expert_len = max(1, len(path) - 1)
            max_steps = 5 * expert_len
            ep.steps = 0
            ep.collisions = 0

            success = False
            while True:
                if ep.robot == ep.goal:
                    success = True
                    break
                if ep.steps >= max_steps:
                    success = False
                    break

                img = render_image(ep.w, ep.h, cell, ep.obstacles, ep.start, ep.goal, ep.robot)
                x = qimage_to_tensor(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    a_i = int(model(x).argmax(dim=1).item())
                a = I2A[a_i]
                env.step(a)

            total_steps += ep.steps
            total_collisions += ep.collisions
            if success:
                successes += 1

            # keep output minimal; users can compute stats from final JSON
            if episodes <= 10 or ep_i in (1, episodes) or ep_i % 10 == 0:
                print(
                    f"episode {ep_i}/{episodes} | {'SUCCESS' if success else 'FAIL'} | "
                    f"steps={ep.steps}/{max_steps} collisions={ep.collisions} expertX={expert_len}"
                )

        summary = {
            "mode": "test-online-headless",
            "model": str(Path(model_path).resolve()),
            "arch": str(arch_label),
            "episodes": int(episodes),
            "successes": int(successes),
            "success_rate": float(successes / max(1, episodes)),
            "avg_steps": float(total_steps / max(1, episodes)),
            "avg_collisions": float(total_collisions / max(1, episodes)),
            "w": int(w),
            "h": int(h),
            "p": float(p),
            "seed": seed,
            "device": str(device),
        }
        print(json.dumps(summary, indent=2))

        try:
            _append_overall_results(
                model_path=Path(model_path),
                arch=str(arch_label),
                episodes=int(episodes),
                successes=int(successes),
                w=int(w),
                h=int(h),
                p=float(p),
                seed=seed,
                device=str(device),
            )
        except Exception as e:
            print(f"Warning: failed to append overall_results.md/.txt: {e}")
        return

    app = QApplication.instance() or QApplication([])
    win = OnlineWindow(model_path, episodes, w, h, p, cell, seed, device, fps, arch_override)
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
    ap_tr = sub.add_parser("train", help="Train a model on 80% episodes; validate on 20%.")
    ap_tr.add_argument("--dataset", type=str, required=True)
    ap_tr.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory name/path for this training run. Artifacts are written here (model/plot/log).",
    )
    ap_tr.add_argument(
        "--out-model",
        type=str,
        default="./cnn.pt",
        help="(Legacy) Ignored for artifact paths if --run-dir is set. Kept for backward compatibility.",
    )
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
    ap_tr.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Mixed precision on CUDA. 'auto' enables for resnet/vit on cuda.",
    )
    ap_tr.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest; >0 can speed up image loading).")
    ap_tr.add_argument(
        "--pin-memory",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="DataLoader pin_memory. 'auto' enables on cuda.",
    )
    ap_tr.add_argument("--freeze-backbone", action="store_true", help="For resnet/vit: train only the final classification head.")
    ap_tr.add_argument("--vit-image-size", type=int, default=224, help="ViT preprocessing: resize to this square size.")
    ap_tr.add_argument(
        "--vit-pad-square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ViT preprocessing: pad to square before resize (recommended).",
    )
    ap_tr.add_argument(
        "--arch",
        type=str,
        default="cnn",
        choices=["cnn", "resnet18", "vit"],
        help="Model architecture: 'cnn' (train from scratch), 'resnet18' (pretrained ImageNet), or 'vit' (pretrained ViT-B/16).",
    )

    # test-offline
    ap_to = sub.add_parser("test-offline", help="Evaluate on last 20% episodes from dataset (no visualization).")
    ap_to.add_argument("--model", type=str, required=True)
    ap_to.add_argument("--dataset", type=str, required=True)
    ap_to.add_argument("--batch", type=int, default=128)
    ap_to.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap_to.add_argument("--split-seed", type=int, default=None, help="Must match training split if you used shuffling.")
    ap_to.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "cnn", "resnet18", "vit"],
        help="Override model architecture. 'auto' uses the checkpoint's stored arch (or defaults to cnn).",
    )

    # test-online
    ap_tn = sub.add_parser("test-online", help="Run N new episodes online; visualize the model controlling the robot.")
    ap_tn.add_argument("--model", type=str, required=True)
    ap_tn.add_argument("--episodes", type=int, required=True)
    ap_tn.add_argument("--w", type=int, default=16)
    ap_tn.add_argument("--h", type=int, default=12)
    ap_tn.add_argument("--p", type=float, default=0.22)
    ap_tn.add_argument("--cell", type=int, default=32)
    ap_tn.add_argument("--seed", type=int, default=None)
    ap_tn.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap_tn.add_argument("--fps", type=int, default=8, help="Visualization update rate.")
    ap_tn.add_argument("--headless", action="store_true", help="Run online evaluation without opening a window (server-friendly).")
    ap_tn.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "cnn", "resnet18", "vit"],
        help="Override model architecture. 'auto' uses the checkpoint's stored arch (or defaults to cnn).",
    )

    args = ap.parse_args()

    if args.cmd == "record":
        mode_record(Path(args.dataset), args.episodes, args.w, args.h, args.p, args.cell,
                    args.seed, args.ep_digits, args.img_digits)

    elif args.cmd == "train":
        mode_train(
            dataset_dir=Path(args.dataset),
            out_model=Path(args.out_model),
            run_dir=None if args.run_dir is None else Path(args.run_dir),
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device_str=args.device,
            split_seed=args.split_seed,
            sched=args.sched,
            patience=args.patience,
            arch=args.arch,
            amp=args.amp,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            vit_image_size=args.vit_image_size,
            vit_pad_square=args.vit_pad_square,
            freeze_backbone=args.freeze_backbone,
        )

    elif args.cmd == "test-offline":
        mode_test_offline(
            model_path=Path(args.model),
            dataset_dir=Path(args.dataset),
            batch=args.batch,
            device_str=args.device,
            split_seed=args.split_seed,
            arch_override=None if args.arch == "auto" else args.arch,
        )

    elif args.cmd == "test-online":
        mode_test_online(
            model_path=Path(args.model),
            episodes=args.episodes,
            w=args.w,
            h=args.h,
            p=args.p,
            cell=args.cell,
            seed=args.seed,
            device_str=args.device,
            fps=args.fps,
            arch_override=None if args.arch == "auto" else args.arch,
            headless=bool(args.headless),
        )


if __name__ == "__main__":
    main()