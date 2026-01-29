#!/usr/bin/env python3
"""
Train/Test a PyTorch CNN on the recorded robot-arm dataset.

Dataset format (inside --data_dir):
- samples.csv with columns: image_filename, action
    - image_filename: relative path like "images/image_000123.png"
    - action: JSON list string of one-hot vector, e.g. "[0,0,1,0,...]"
- images/ folder containing the PNG files

Modes:
- train: trains on the first --train_frac portion (default 0.8), tests on the rest, and:
    1) writes a log file
    2) saves a model checkpoint after each epoch: <model_stem>_epoch001.pt, ...
    3) prints/logs progress every next 5% of mini-batches within an epoch
    4) plots and saves learning curves (loss + accuracy) after each epoch

- test: loads a saved model and evaluates on the remaining (1 - train_frac) portion.

Output management:
- All generated files are saved inside --exp_dir:
    - model checkpoints (.pt)
    - log file (.log)
    - history json (.json)
    - learning curve plots (.png)

NEW (augmentation):
- Training data augmentation is applied ONLY in train mode (not in test mode).
- Enable/disable via --augment / --no_augment and control via --aug_* options.

Examples:
  # Train with augmentation (default)
  python robot_arm_controller_cnn.py \
      --data_dir ./data_dof2_100000 \
      --mode train \
      --exp_dir ./experiment1_cnn_aug \
      --model_path cnn.pt \
      --train_frac 0.8 --epochs 50

  # Train without augmentation
  python robot_arm_controller_cnn.py \
      --data_dir ./data_dof2_100000 \
      --mode train \
      --exp_dir ./experiment1_cnn_noaug \
      --model_path cnn.pt \
      --no_augment

  # Test
  python robot_arm_controller_cnn.py \
      --data_dir ./data_dof2_100000 \
      --mode test \
      --exp_dir ./experiment1_cnn_aug \
      --model_path cnn_epoch050.pt \
      --train_frac 0.8

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
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

try:
    from torchvision import transforms
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

    logger = logging.getLogger("cnn_train_test")
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
    action_dim = None

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
                raise ValueError(f"Invalid one-hot action: {action_str[:80]}...")

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


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
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
    """
    Returns (train_transform, test_transform).
    Augmentations are applied ONLY to train_transform if augment=True.
    """

    # Deterministic test transform
    test_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    if not augment:
        return test_tfm, test_tfm

    # Mild but useful augmentations for synthetic images
    # - RandomResizedCrop provides scale/shift variation
    # - RandomAffine adds rotation/translation/scale variation
    # - ColorJitter adds camera-like intensity changes
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
            fill=255,  # white background
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
                value=0.0,  # erase to black (after normalization this is fine)
                inplace=False,
            )
        )

    train_tfm = transforms.Compose(train_ops)
    return train_tfm, test_tfm


# -----------------------------
# Eval / Train
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

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
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "action_dim": int(action_dim),
        "image_size": int(image_size),
        "model_type": "SimpleCNN",
        "epoch": int(epoch),
        "history": history,
        "augment_cfg": augment_cfg,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)


def load_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    payload = torch.load(model_path, map_location=device)
    action_dim = int(payload["action_dim"])
    image_size = int(payload.get("image_size", 224))
    epoch = int(payload.get("epoch", 0))
    history = payload.get("history", {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []})
    augment_cfg = payload.get("augment_cfg", {})

    model = SimpleCNN(num_classes=action_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, action_dim, image_size, epoch, history, augment_cfg


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


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    logger: logging.Logger,
    base_model_path: Path,
    action_dim: int,
    image_size: int,
    plot_path: Path,
    history_path: Path,
    augment_cfg: Dict[str, float],
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

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

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

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
                    f"Epoch {epoch:03d}/{epochs} | progress ~{pct:3d}% "
                    f"({b_idx}/{total_batches} batches) | "
                    f"running_loss={running_loss / max(1, total):.4f} | "
                    f"running_acc={correct / max(1, total) * 100:.2f}%"
                )

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        test_loss, test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))

        elapsed = datetime.now() - epoch_start
        logger.info(
            f"Epoch {epoch:03d}/{epochs} DONE | "
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
        )
        logger.info(f"Saved checkpoint: {ckpt_path.resolve()}")

        plot_title = f"Learning Curves ({base_model_path.stem})"
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
    p = argparse.ArgumentParser(description="Train/test a CNN on robot-arm image->action data.")
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing samples.csv.")
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="train or test")

    p.add_argument("--model_path", type=str, required=True,
                   help="Model filename or path; will be saved/loaded inside --exp_dir.")
    p.add_argument("--exp_dir", type=str, required=True, help="Experiment output directory (all outputs go here).")

    p.add_argument("--train_frac", type=float, default=0.8, help="Fraction of samples used for training (default: 0.8).")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (train mode only).")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (train mode only).")

    p.add_argument("--image_size", type=int, default=224, help="Resize images to this square size (default: 224).")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    p.add_argument("--log_path", type=str, default="",
                   help="Optional log filename/path; will be placed inside --exp_dir (only the name is used).")

    # Augmentation controls
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
    base_model_path = (exp_dir / base_model_name)
    if not base_model_path.suffix:
        base_model_path = base_model_path.with_suffix(".pt")

    log_name = Path(args.log_path).name if args.log_path.strip() else f"{base_model_path.stem}.log"
    log_path = exp_dir / log_name
    logger = setup_logger(log_path)

    pairs, action_dim = load_samples_csv(data_dir)
    train_pairs, test_pairs = split_pairs(pairs, args.train_frac)

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

    logger.info("=== Dataset ===")
    logger.info(f"Data dir: {data_dir.resolve()}")
    logger.info(f"Experiment dir: {exp_dir.resolve()}")
    logger.info(f"Total samples: {len(pairs)}")
    logger.info(f"Train samples (first {args.train_frac*100:.1f}%): {len(train_pairs)}")
    logger.info(f"Test samples (remaining): {len(test_pairs)}")
    logger.info(f"Action dim (num classes): {action_dim}")
    logger.info(f"Device: {device}")
    logger.info(f"Augment cfg: {augment_cfg}")

    train_tfm, test_tfm = build_transforms(
        image_size=args.image_size,
        augment=args.augment if args.mode == "train" else False,  # never augment in test mode
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
        model = SimpleCNN(num_classes=action_dim).to(device)
        logger.info("=== Training ===")
        logger.info(f"Base model file: {base_model_path.resolve()}")
        logger.info(f"Checkpoints: {base_model_path.stem}_epoch###.pt (in exp_dir)")
        logger.info(f"Plot (latest): {plot_path.resolve()}")
        logger.info(f"History JSON: {history_path.resolve()}")

        train_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            logger=logger,
            base_model_path=base_model_path,
            action_dim=action_dim,
            image_size=args.image_size,
            plot_path=plot_path,
            history_path=history_path,
            augment_cfg=augment_cfg,
        )
        logger.info("Training finished.")
        return 0

    # test mode
    logger.info("=== Testing ===")

    model_to_load = exp_dir / Path(args.model_path).name
    if not model_to_load.suffix:
        model_to_load = model_to_load.with_suffix(".pt")

    model, saved_action_dim, saved_image_size, saved_epoch, _saved_history, saved_aug_cfg = load_model(model_to_load, device=device)

    if saved_action_dim != action_dim:
        logger.warning(
            "Action dim mismatch between dataset and saved model.\n"
            f"  dataset action_dim: {action_dim}\n"
            f"  model action_dim:   {saved_action_dim}\n"
            "Results may be invalid."
        )

    if saved_image_size != args.image_size:
        logger.warning(
            "Image size differs from saved model metadata.\n"
            f"  saved image_size: {saved_image_size}\n"
            f"  current --image_size: {args.image_size}\n"
        )

    if saved_aug_cfg:
        logger.info(f"Model was trained with augment cfg: {saved_aug_cfg}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(f"Loaded checkpoint epoch: {saved_epoch}")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_acc*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
