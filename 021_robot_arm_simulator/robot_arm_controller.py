#!/usr/bin/env python3
"""
Train/Test a PyTorch CNN, custom ResNet, or torchvision ViT on the recorded robot-arm dataset.

Dataset format (inside --data_dir):
- samples.csv with columns: image_filename, action
    - image_filename: relative path like "images/image_000123.png"
    - action: JSON list string of one-hot vector, e.g. "[0,0,1,0,...]"
- images/ folder containing the PNG files

Modes:
- train:
    - trains on first --train_frac portion of samples.csv
    - tests on the remaining portion
    - writes a log file
    - saves a checkpoint after each epoch: <model_stem>_epoch001.pt, ...
    - prints/logs progress every next 5% of mini-batches per epoch
    - plots and saves learning curves after each epoch

- test:
    - loads a saved checkpoint and evaluates on the test split.

Output management:
- All generated files are saved inside --exp_dir.

Model selection:
- Use --model {cnn,resnet,vit} to choose architecture.

Augmentation:
- Training augmentation is applied ONLY in train mode.
- Enable/disable via --augment / --no_augment and control via --aug_* options.

NEW: faster evaluation
- Use --test_frac and/or --test_max_samples to evaluate only a subset of the test split
  (useful for quick sanity checks / overfitting tests).

NEW: ViT-friendly training options
- --warmup_epochs: linear LR warmup
- --scheduler: none or cosine
- --min_lr: cosine LR floor
- --label_smoothing: CrossEntropy label smoothing
- --amp: automatic mixed precision (CUDA)

Requirements:
  pip install torch torchvision pillow matplotlib
"""

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

try:
    from torchvision import transforms
    import torchvision
except ImportError as e:
    raise ImportError("torchvision is required (pip install torchvision).") from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required (pip install matplotlib).") from e


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("arm_action_train_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_path.resolve()}")
    return logger


# -----------------------------
# Dataset
# -----------------------------
class RobotArmImageActionDataset(Dataset):
    def __init__(self, data_dir: Path, pairs: List[Tuple[str, int]], transform=None):
        self.data_dir = data_dir
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rel_path, class_idx = self.pairs[idx]
        img_path = self.data_dir / rel_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = torch.tensor(class_idx, dtype=torch.long)
        return img, y


def load_samples_csv(data_dir: Path) -> Tuple[List[Tuple[str, int]], int]:
    csv_path = data_dir / "samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing samples.csv in {data_dir}")

    pairs: List[Tuple[str, int]] = []
    action_dim: Optional[int] = None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"image_filename", "action"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"samples.csv must have exactly columns {sorted(expected)} "
                f"but has {reader.fieldnames}"
            )

        for row in reader:
            img_rel = row["image_filename"].strip()
            action_str = row["action"].strip()

            onehot = json.loads(action_str)
            if not isinstance(onehot, list) or not onehot:
                raise ValueError(f"Invalid one-hot action: {action_str[:120]}...")

            if action_dim is None:
                action_dim = len(onehot)
            elif len(onehot) != action_dim:
                raise ValueError("Inconsistent action one-hot length in samples.csv")

            class_idx = int(max(range(len(onehot)), key=lambda i: onehot[i]))
            pairs.append((img_rel, class_idx))

    if action_dim is None:
        raise ValueError("samples.csv appears empty.")

    return pairs, action_dim


def split_pairs(
    pairs: List[Tuple[str, int]],
    train_frac: float
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train_frac must be between 0 and 1 (exclusive).")

    n = len(pairs)
    n_train = int(n * train_frac)
    n_train = max(1, min(n - 1, n_train))
    return pairs[:n_train], pairs[n_train:]


def limit_test_pairs(
    test_pairs: List[Tuple[str, int]],
    test_frac: float,
    test_max_samples: int
) -> List[Tuple[str, int]]:
    """
    Returns a deterministic subset of test_pairs (prefix) for faster evaluation.
    Apply both constraints: fraction first, then max_samples.
    """
    if not (0.0 < test_frac <= 1.0):
        raise ValueError("--test_frac must be in (0, 1].")

    n = len(test_pairs)
    n_frac = max(1, int(round(n * test_frac)))
    subset = test_pairs[:n_frac]

    if test_max_samples > 0:
        subset = subset[: max(1, min(len(subset), test_max_samples))]

    return subset


# -----------------------------
# Models
# -----------------------------
class SimpleCNN(nn.Module):
    """Baseline CNN."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Small ResNet basic block."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    Custom lightweight ResNet for geometric, synthetic images.
    - No early maxpool to preserve spatial precision longer.
    - Global average pooling.
    """
    def __init__(self, num_classes: int, layers=(2, 2, 2, 2), channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.in_ch = c1
        self.layer1 = self._make_layer(c1, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(c2, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(c3, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(c4, blocks=layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)

    def _make_layer(self, out_ch: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_vit_b_16(num_classes: int) -> nn.Module:
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)
    # Replace classification head
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Unexpected ViT head structure in torchvision.")
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower().strip()
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if model_name == "resnet":
        return SmallResNet(num_classes=num_classes)
    if model_name == "vit":
        return build_vit_b_16(num_classes=num_classes)
    raise ValueError(f"Unknown --model '{model_name}'. Use 'cnn', 'resnet', or 'vit'.")


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(
    image_size: int,
    augment: bool,
    aug_rotate_deg: float,
    aug_translate: float,
    aug_scale: float,
    aug_brightness: float,
    aug_contrast: float,
    aug_erasing_p: float,
) -> Tuple[transforms.Compose, transforms.Compose]:
    test_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    if not augment:
        return test_tfm, test_tfm

    train_ops = [
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(max(0.7, 1.0 - aug_scale), 1.0),
            ratio=(0.95, 1.05),
        ),
        transforms.RandomAffine(
            degrees=aug_rotate_deg,
            translate=(aug_translate, aug_translate),
            scale=(max(0.7, 1.0 - aug_scale), 1.0 + aug_scale),
            shear=None,
            fill=255,
        ),
        transforms.ColorJitter(
            brightness=aug_brightness,
            contrast=aug_contrast,
            saturation=0.0,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    if aug_erasing_p > 0.0:
        train_ops.append(
            transforms.RandomErasing(
                p=aug_erasing_p,
                scale=(0.01, 0.06),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            )
        )

    return transforms.Compose(train_ops), test_tfm


# -----------------------------
# Eval / Train
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_smoothing: float) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(1, total), correct / max(1, total)


def _epoch_checkpoint_path(base_model_path: Path, epoch: int) -> Path:
    stem = base_model_path.stem
    suffix = base_model_path.suffix if base_model_path.suffix else ".pt"
    return base_model_path.with_name(f"{stem}_epoch{epoch:03d}{suffix}")


def save_checkpoint(
    model_path: Path,
    model: nn.Module,
    action_dim: int,
    image_size: int,
    epoch: int,
    history: Dict[str, List[float]],
    augment_cfg: Dict[str, float],
    model_name: str,
    optim_cfg: Dict[str, float],
    train_cfg: Dict[str, float],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "action_dim": int(action_dim),
        "image_size": int(image_size),
        "model_name": str(model_name),
        "epoch": int(epoch),
        "history": history,
        "augment_cfg": augment_cfg,
        "optim_cfg": optim_cfg,
        "train_cfg": train_cfg,
        "torchvision_version": getattr(torchvision, "__version__", "unknown"),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)


def load_checkpoint_payload(model_path: Path, device: torch.device) -> Dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, map_location=device)


def plot_learning_curves(out_path: Path, history: Dict[str, List[float]], title: str) -> None:
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    train_acc = history.get("train_acc", [])
    test_acc = history.get("test_acc", [])

    epochs = list(range(1, max(len(train_loss), len(test_loss), len(train_acc), len(test_acc)) + 1))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if train_loss:
        plt.plot(epochs[: len(train_loss)], train_loss, label="train_loss")
    if test_loss:
        plt.plot(epochs[: len(test_loss)], test_loss, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    if train_acc:
        plt.plot(epochs[: len(train_acc)], train_acc, label="train_acc")
    if test_acc:
        plt.plot(epochs[: len(test_acc)], test_acc, label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    min_lr: float,
    weight_decay: float,
    warmup_epochs: int,
    scheduler: str,
    label_smoothing: float,
    use_amp: bool,
    logger: logging.Logger,
    base_model_path: Path,
    action_dim: int,
    image_size: int,
    plot_path: Path,
    history_path: Path,
    augment_cfg: Dict[str, float],
    model_name: str,
    train_cfg: Dict[str, float],
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    scaler = None
    amp_enabled = bool(use_amp and device.type == "cuda")
    if amp_enabled:
        scaler = torch.cuda.amp.GradScaler()

    optim_cfg = {
        "optimizer": "AdamW",
        "base_lr": float(base_lr),
        "min_lr": float(min_lr),
        "weight_decay": float(weight_decay),
        "scheduler": str(scheduler),
        "warmup_epochs": int(warmup_epochs),
        "label_smoothing": float(label_smoothing),
        "amp": bool(amp_enabled),
    }

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    total_steps = epochs * len(train_loader)
    warmup_steps = max(0, warmup_epochs * len(train_loader))
    global_step = 0

    def lr_for_step(step: int) -> float:
        # Linear warmup then cosine decay (if selected)
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps

        if scheduler.lower() != "cosine":
            return base_lr

        # Cosine from base_lr down to min_lr over remaining steps
        if total_steps <= warmup_steps:
            return min_lr

        t = (step - warmup_steps) / float(total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    # Initialize LR to warmup start if warmup is enabled
    _set_lr(optimizer, lr_for_step(0))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(train_loader)
        if total_batches <= 0:
            raise RuntimeError("Training DataLoader has no batches. Check batch_size and dataset size.")

        step_pct = 0.05
        reported = set()
        epoch_start = datetime.now()

        for b_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update LR per step (warmup + optional cosine)
            lr_now = lr_for_step(global_step)
            _set_lr(optimizer, lr_now)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                assert scaler is not None
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            global_step += 1

            running_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))

            progress = b_idx / total_batches
            bucket = int(progress / step_pct)
            if bucket not in reported and progress >= step_pct:
                reported.add(bucket)
                pct = min(100, int(round(progress * 100)))
                logger.info(
                    f"[{model_name}] Epoch {epoch:03d}/{epochs} | progress ~{pct:3d}% "
                    f"({b_idx}/{total_batches} batches) | "
                    f"lr={_get_lr(optimizer):.6g} | "
                    f"running_loss={running_loss / max(1, total):.4f} | "
                    f"running_acc={correct / max(1, total) * 100:.2f}%"
                )

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        test_loss, test_acc = evaluate(model, test_loader, device, label_smoothing=label_smoothing)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))

        elapsed = datetime.now() - epoch_start
        logger.info(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} DONE | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}% | "
            f"epoch_time={elapsed}"
        )

        ckpt_path = _epoch_checkpoint_path(base_model_path, epoch)
        save_checkpoint(
            model_path=ckpt_path,
            model=model,
            action_dim=action_dim,
            image_size=image_size,
            epoch=epoch,
            history=history,
            augment_cfg=augment_cfg,
            model_name=model_name,
            optim_cfg=optim_cfg,
            train_cfg=train_cfg,
        )
        logger.info(f"Saved checkpoint: {ckpt_path.resolve()}")

        plot_title = f"Learning Curves ({model_name})"
        plot_learning_curves(plot_path, history, plot_title)
        plot_epoch_path = plot_path.with_name(f"{plot_path.stem}_epoch{epoch:03d}{plot_path.suffix}")
        plot_learning_curves(plot_epoch_path, history, plot_title)
        logger.info(f"Saved plot: {plot_path.resolve()}")
        logger.info(f"Saved plot: {plot_epoch_path.resolve()}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved history: {history_path.resolve()}")


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/test CNN, ResNet, or ViT on robot-arm image->action data.")
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing samples.csv.")
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="train or test")
    p.add_argument("--model", type=str, required=True, choices=["cnn", "resnet", "vit"], help="Which model architecture to use.")

    p.add_argument("--model_path", type=str, required=True,
                   help="Model filename or path; will be saved/loaded inside --exp_dir.")
    p.add_argument("--exp_dir", type=str, required=True, help="Experiment output directory (all outputs go here).")

    p.add_argument("--train_frac", type=float, default=0.8, help="Fraction of samples used for training (default: 0.8).")

    # NEW: limit evaluation size
    p.add_argument("--test_frac", type=float, default=1.0,
                   help="Fraction of test split to evaluate each epoch (default: 1.0).")
    p.add_argument("--test_max_samples", type=int, default=0,
                   help="Max number of test samples to evaluate (0 = no limit).")

    p.add_argument("--epochs", type=int, default=10, help="Training epochs (train mode only).")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (train mode only).")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine schedule.")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")

    # NEW: ViT-friendly knobs (also fine for CNN/ResNet)
    p.add_argument("--warmup_epochs", type=int, default=0, help="Linear warmup epochs (recommended for ViT).")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"],
                   help="LR scheduler: none or cosine.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing (e.g. 0.1).")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training on CUDA.")

    p.add_argument("--image_size", type=int, default=224, help="Resize images to this square size (default: 224).")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    p.add_argument("--log_path", type=str, default="",
                   help="Optional log filename/path; will be placed inside --exp_dir (only the name is used).")

    aug = p.add_argument_group("augmentation")
    aug.add_argument("--augment", dest="augment", action="store_true", help="Enable training data augmentation (default).")
    aug.add_argument("--no_augment", dest="augment", action="store_false", help="Disable training data augmentation.")
    p.set_defaults(augment=True)

    aug.add_argument("--aug_rotate_deg", type=float, default=10.0, help="Random rotation degrees for RandomAffine.")
    aug.add_argument("--aug_translate", type=float, default=0.05, help="Random translation fraction for RandomAffine.")
    aug.add_argument("--aug_scale", type=float, default=0.10, help="Random scale fraction for crop/affine.")
    aug.add_argument("--aug_brightness", type=float, default=0.15, help="ColorJitter brightness.")
    aug.add_argument("--aug_contrast", type=float, default=0.15, help="ColorJitter contrast.")
    aug.add_argument("--aug_erasing_p", type=float, default=0.0, help="RandomErasing probability (0 disables).")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        return 1

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Force all outputs into exp_dir by using only filenames.
    base_model_name = Path(args.model_path).name
    base_model_path = exp_dir / base_model_name
    if not base_model_path.suffix:
        base_model_path = base_model_path.with_suffix(".pt")

    log_name = Path(args.log_path).name if args.log_path.strip() else f"{base_model_path.stem}.log"
    log_path = exp_dir / log_name
    logger = setup_logger(log_path)

    pairs, action_dim = load_samples_csv(data_dir)
    train_pairs, test_pairs_full = split_pairs(pairs, args.train_frac)

    # Limit test evaluation subset (useful for quick sanity checks)
    test_pairs = limit_test_pairs(test_pairs_full, test_frac=args.test_frac, test_max_samples=args.test_max_samples)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    augment_cfg = {
        "augment": bool(args.augment),
        "aug_rotate_deg": float(args.aug_rotate_deg),
        "aug_translate": float(args.aug_translate),
        "aug_scale": float(args.aug_scale),
        "aug_brightness": float(args.aug_brightness),
        "aug_contrast": float(args.aug_contrast),
        "aug_erasing_p": float(args.aug_erasing_p),
    }

    train_cfg = {
        "train_frac": float(args.train_frac),
        "test_frac": float(args.test_frac),
        "test_max_samples": int(args.test_max_samples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "scheduler": str(args.scheduler),
        "label_smoothing": float(args.label_smoothing),
        "amp": bool(args.amp),
        "image_size": int(args.image_size),
    }

    logger.info("=== Run ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data dir: {data_dir.resolve()}")
    logger.info(f"Experiment dir: {exp_dir.resolve()}")
    logger.info(f"Base model file: {base_model_path.resolve()}")
    logger.info(f"Total samples: {len(pairs)}")
    logger.info(f"Train samples (first {args.train_frac*100:.1f}%): {len(train_pairs)}")
    logger.info(f"Test samples (remaining): {len(test_pairs_full)}")
    logger.info(f"Test samples evaluated per epoch: {len(test_pairs)} (test_frac={args.test_frac}, test_max_samples={args.test_max_samples})")
    logger.info(f"Action dim (num classes): {action_dim}")
    logger.info(f"Device: {device}")
    logger.info(f"Hyperparams: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, min_lr={args.min_lr}, wd={args.weight_decay}")
    logger.info(f"Scheduler: {args.scheduler}, warmup_epochs={args.warmup_epochs}, label_smoothing={args.label_smoothing}, amp={args.amp}")
    logger.info(f"Augment cfg: {augment_cfg}")
    logger.info(f"torchvision: {torchvision.__version__}")

    train_tfm, test_tfm = build_transforms(
        image_size=args.image_size,
        augment=args.augment if args.mode == "train" else False,
        aug_rotate_deg=args.aug_rotate_deg,
        aug_translate=args.aug_translate,
        aug_scale=args.aug_scale,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_erasing_p=args.aug_erasing_p,
    )

    train_ds = RobotArmImageActionDataset(data_dir, train_pairs, transform=train_tfm)
    test_ds = RobotArmImageActionDataset(data_dir, test_pairs, transform=test_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    plot_path = exp_dir / f"{base_model_path.stem}_learning_curve.png"
    history_path = exp_dir / f"{base_model_path.stem}_history.json"

    if args.mode == "train":
        model = build_model(args.model, num_classes=action_dim).to(device)

        logger.info("=== Training ===")
        logger.info(f"Checkpoints: {base_model_path.stem}_epoch###.pt (in exp_dir)")
        logger.info(f"Plot (latest): {plot_path.resolve()}")
        logger.info(f"History JSON: {history_path.resolve()}")

        train_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            scheduler=args.scheduler,
            label_smoothing=args.label_smoothing,
            use_amp=args.amp,
            logger=logger,
            base_model_path=base_model_path,
            action_dim=action_dim,
            image_size=args.image_size,
            plot_path=plot_path,
            history_path=history_path,
            augment_cfg=augment_cfg,
            model_name=args.model,
            train_cfg=train_cfg,
        )
        logger.info("Training finished.")
        return 0

    # test mode
    logger.info("=== Testing ===")
    model_to_load = exp_dir / Path(args.model_path).name
    if not model_to_load.suffix:
        model_to_load = model_to_load.with_suffix(".pt")

    payload = load_checkpoint_payload(model_to_load, device=device)
    saved_model_name = str(payload.get("model_name", "unknown"))
    saved_action_dim = int(payload["action_dim"])
    saved_epoch = int(payload.get("epoch", 0))
    saved_cfg = payload.get("train_cfg", {})

    if saved_action_dim != action_dim:
        logger.warning(
            "Action dim mismatch between dataset and checkpoint.\n"
            f"  dataset action_dim: {action_dim}\n"
            f"  checkpoint action_dim: {saved_action_dim}\n"
            "Results may be invalid."
        )

    if saved_model_name != args.model:
        logger.warning(
            "Model type mismatch between CLI and checkpoint.\n"
            f"  --model: {args.model}\n"
            f"  checkpoint model_name: {saved_model_name}\n"
            "Will instantiate the CLI model type; if this is wrong, loading may fail."
        )

    model = build_model(args.model, num_classes=saved_action_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    logger.info(f"Loaded checkpoint: {model_to_load.resolve()}")
    logger.info(f"Checkpoint epoch: {saved_epoch}")
    if saved_cfg:
        logger.info(f"Checkpoint train_cfg: {saved_cfg}")

    test_loss, test_acc = evaluate(model, test_loader, device, label_smoothing=float(args.label_smoothing))
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_acc*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
