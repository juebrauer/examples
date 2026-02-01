#!/usr/bin/env python3
"""
robot_arm_controller.py

Train/Test:
  - A PyTorch CNN, custom ResNet, or torchvision ViT (supervised classifier)
  - OR a Vision-Language Model (VLM) finetuned with Unsloth (instruction -> action)

Goal:
Given an image of an n-DOF robot arm and a target position (red X),
predict the expert action (a discrete action index in a 3*DOF action space).

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
    - saves checkpoints / adapters into --exp_dir

- test:
    - loads a saved checkpoint / adapter and evaluates on the test split.

Model selection:
- Use --model {cnn,resnet,vit,vlm} to choose architecture.

VLM option (Unsloth):
- Finetunes a VLM with LoRA adapters using:
    - unsloth.FastVisionModel
    - trl.SFTTrainer + unsloth.trainer.UnslothVisionDataCollator
- Supervised target is a short text: the action index (integer).

Notes:
- VLM training is typically heavier than CNN/ViT training.
- VLM checkpoints are saved as a directory (adapter + tokenizer), not a single .pt file.

Requirements (classic models):
  pip install torch torchvision pillow matplotlib

Additional requirements (VLM):
  pip install unsloth trl transformers accelerate bitsandbytes

Example usage (classic):
  python robot_arm_controller.py --data_dir ./data_dof2_5000 --exp_dir ./exp --mode train --model vit --model_path vit.pt
  python robot_arm_controller.py --data_dir ./data_dof2_5000 --exp_dir ./exp --mode test  --model vit --model_path vit_epoch010.pt

Example usage (VLM):
  python robot_arm_controller.py --data_dir ./data_dof2_5000 --exp_dir ./exp --mode train --model vlm --model_path vlm_adapter
  python robot_arm_controller.py --data_dir ./data_dof2_5000 --exp_dir ./exp --mode test  --model vlm --model_path vlm_adapter
"""

import argparse
import csv
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

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


# =============================================================================
# Logging
# =============================================================================
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


# =============================================================================
# Dataset helpers (CSV -> (image_rel_path, class_idx))
# =============================================================================
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
    train_frac: float,
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
    test_max_samples: int,
) -> List[Tuple[str, int]]:
    """Deterministic subset of test_pairs (prefix) for faster evaluation."""
    if not (0.0 < test_frac <= 1.0):
        raise ValueError("--test_frac must be in (0, 1].")

    n = len(test_pairs)
    n_frac = max(1, int(round(n * test_frac)))
    subset = test_pairs[:n_frac]

    if test_max_samples > 0:
        subset = subset[: max(1, min(len(subset), test_max_samples))]

    return subset


# =============================================================================
# Classic models (CNN / ResNet / ViT)
# =============================================================================
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


def build_vit_b_16(num_classes: int, pretrained: bool) -> nn.Module:
    """Build torchvision ViT-B/16 and replace head."""
    from torchvision.models import vit_b_16

    weights = None
    if pretrained:
        try:
            from torchvision.models import ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT
        except Exception:
            weights = None

    model = vit_b_16(weights=weights)

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Unexpected ViT head structure in torchvision.")

    return model


def build_model(model_name: str, num_classes: int, vit_pretrained: bool) -> nn.Module:
    model_name = model_name.lower().strip()
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if model_name == "resnet":
        return SmallResNet(num_classes=num_classes)
    if model_name == "vit":
        return build_vit_b_16(num_classes=num_classes, pretrained=vit_pretrained)
    raise ValueError(f"Unknown --model '{model_name}'. Use 'cnn', 'resnet', 'vit', or 'vlm'.")


# =============================================================================
# Transforms (classic models only)
# =============================================================================
def _vit_pretrained_norm() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return mean/std matching torchvision ViT pretrained weights."""
    try:
        from torchvision.models import ViT_B_16_Weights
        w = ViT_B_16_Weights.DEFAULT
        mean = tuple(float(x) for x in w.meta["mean"])
        std = tuple(float(x) for x in w.meta["std"])
        return mean, std
    except Exception:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int,
    augment: bool,
    aug_rotate_deg: float,
    aug_translate: float,
    aug_scale: float,
    aug_brightness: float,
    aug_contrast: float,
    aug_erasing_p: float,
    model_name: str,
    vit_pretrained: bool,
):
    """
    "Safe augmentation" that preserves the full scene:
        Resize -> RandomAffine -> ColorJitter -> ToTensor -> Normalize
    """
    model_name = model_name.lower().strip()

    if model_name == "vit" and vit_pretrained:
        mean, std = _vit_pretrained_norm()
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    test_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if not augment:
        return test_tfm, test_tfm

    min_scale = max(0.7, 1.0 - float(aug_scale))
    max_scale = 1.0 + float(aug_scale)

    train_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(
            degrees=float(aug_rotate_deg),
            translate=(float(aug_translate), float(aug_translate)),
            scale=(min_scale, max_scale),
            shear=None,
            fill=255,
        ),
        transforms.ColorJitter(
            brightness=float(aug_brightness),
            contrast=float(aug_contrast),
            saturation=0.0,
            hue=0.0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if aug_erasing_p > 0.0:
        train_ops.append(
            transforms.RandomErasing(
                p=float(aug_erasing_p),
                scale=(0.01, 0.06),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            )
        )

    return transforms.Compose(train_ops), test_tfm


# =============================================================================
# Classic eval / train
# =============================================================================
@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> Tuple[float, float]:
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


def save_classifier_checkpoint(
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


def load_classifier_checkpoint_payload(model_path: Path, device: torch.device) -> Dict[str, Any]:
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


def train_classifier_loop(
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

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

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
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps

        if scheduler.lower() != "cosine":
            return base_lr

        if total_steps <= warmup_steps:
            return min_lr

        t = (step - warmup_steps) / float(total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

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

            lr_now = lr_for_step(global_step)
            _set_lr(optimizer, lr_now)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                assert scaler is not None
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

        test_loss, test_acc = evaluate_classifier(model, test_loader, device, label_smoothing=label_smoothing)

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
        save_classifier_checkpoint(
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


# =============================================================================
# VLM (Unsloth) training / evaluation
# =============================================================================
@dataclass
class VLMConfig:
    model_name: str
    load_in_4bit: bool
    max_seq_length: int
    # LoRA
    r: int
    lora_alpha: int
    lora_dropout: float
    finetune_vision_layers: bool
    finetune_language_layers: bool
    finetune_attention_modules: bool
    finetune_mlp_modules: bool
    # Trainer
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    num_train_epochs: int
    logging_steps: int
    max_steps: int
    disable_tqdm: bool
    optim: str
    lr_scheduler_type: str
    seed: int
    max_new_tokens: int
    temperature: float


def _require_vlm_deps() -> None:
    try:
        import unsloth  # noqa: F401
        import trl      # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        raise ImportError(
            "VLM mode requires extra packages.\n"
            "Install with:\n"
            "  pip install unsloth trl transformers accelerate bitsandbytes\n"
        ) from e


def _vlm_instruction(action_dim: int) -> str:
    # Keep it short and unambiguous: output an integer.
    # The expert encoding is defined in robot_arm_simulator.py:
    #   idx = dof_index*3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree
    return (
        "You control a planar robot arm. "
        "Given the image, output the best expert action as a single integer in [0, "
        f"{action_dim-1}]. "
        "Action encoding: idx = dof_index*3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree. "
        "Answer with ONLY the integer."
    )


def make_vlm_conversation_records(
    data_dir: Path,
    pairs: List[Tuple[str, int]],
    action_dim: int,
) -> List[Dict[str, Any]]:
    """
    Converts (image, class_idx) pairs into Unsloth VLM SFT format:
      {"messages": [ {role:'user', content:[{text},{image}]} , {role:'assistant', content:[{text}]} ]}
    """
    instruction = _vlm_instruction(action_dim)
    records: List[Dict[str, Any]] = []
    for rel_path, class_idx in pairs:
        img_path = str((data_dir / rel_path).resolve())
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": img_path},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": str(int(class_idx))},
                        ],
                    },
                ]
            }
        )
    return records


def train_vlm_with_unsloth(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    exp_dir: Path,
    adapter_dir_name: str,
    cfg: VLMConfig,
    logger: logging.Logger,
) -> Path:
    """
    Trains a VLM with Unsloth and saves the adapter+tokenizer to exp_dir/adapter_dir_name.
    Returns the adapter directory path.
    """
    _require_vlm_deps()

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    logger.info("=== VLM (Unsloth) Training ===")
    logger.info(f"VLM base model: {cfg.model_name}")
    logger.info(f"load_in_4bit: {cfg.load_in_4bit}")
    logger.info(f"Train records: {len(train_records)} | Test records: {len(test_records)}")

    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.model_name,
        load_in_4bit=bool(cfg.load_in_4bit),
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(cfg.finetune_vision_layers),
        finetune_language_layers=bool(cfg.finetune_language_layers),
        finetune_attention_modules=bool(cfg.finetune_attention_modules),
        finetune_mlp_modules=bool(cfg.finetune_mlp_modules),
        r=int(cfg.r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        random_state=int(cfg.seed),
    )

    FastVisionModel.for_training(model)

    out_dir = exp_dir / adapter_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(out_dir / "vlm_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Trainer config: closely matches your notebook style.
    sft_args = SFTConfig(
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        warmup_steps=int(cfg.warmup_steps),
        num_train_epochs=int(cfg.num_train_epochs),
        learning_rate=float(cfg.learning_rate),
        logging_steps=int(cfg.logging_steps),
        max_steps=int(cfg.max_steps) if int(cfg.max_steps) > 0 else -1,
        disable_tqdm=bool(cfg.disable_tqdm),
        optim=str(cfg.optim),
        weight_decay=float(cfg.weight_decay),
        lr_scheduler_type=str(cfg.lr_scheduler_type),
        seed=int(cfg.seed),
        output_dir=str(out_dir / "trainer_outputs"),
        report_to="none",
        # REQUIRED for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=int(cfg.max_seq_length),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_records,
        args=sft_args,
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB. Start reserved = {start_gpu_memory} GB.")

    trainer_stats = trainer.train()
    # Save trainer metrics
    try:
        metrics = dict(trainer_stats.metrics)
    except Exception:
        metrics = {}
    with open(out_dir / "trainer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save adapter + tokenizer (same pattern as your notebook)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Saved VLM adapter+tokenizer to: {out_dir.resolve()}")

    # Optional: quick eval on provided test_records (generative)
    try:
        acc = evaluate_vlm_action_accuracy(
            adapter_dir=out_dir,
            test_records=test_records,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            logger=logger,
            limit=min(200, len(test_records)),
        )
        logger.info(f"[VLM] Quick eval accuracy on up to 200 test samples: {acc*100:.2f}%")
    except Exception as e:
        logger.warning(f"VLM quick eval failed (non-fatal): {e}")

    return out_dir


def _parse_action_int(text: str, action_dim: int) -> Optional[int]:
    """
    Extract the first integer from model output and validate range.
    Returns None if parsing fails.
    """
    if text is None:
        return None
    ms = re.findall(r"-?\d+", text)
    if not ms:
        return None
    try:
        val = int(ms[-1])
    except Exception:
        return None
    if 0 <= val < action_dim:
        return val
    return None


@torch.no_grad()
def evaluate_vlm_action_accuracy(
    adapter_dir: Path,
    test_records: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    logger: logging.Logger,
    limit: int = 0,
) -> float:
    """
    Loads a finetuned adapter dir (saved by train_vlm_with_unsloth) and computes accuracy on test_records.
    test_records must be in the same "messages" format created by make_vlm_conversation_records.
    """
    _require_vlm_deps()

    from unsloth import FastVisionModel

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    # Load from adapter dir (as in your notebook)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(adapter_dir),
        load_in_4bit=True,  # adapter dirs are typically 4bit LoRA; safe default
    )
    FastVisionModel.for_inference(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Infer action_dim from the instruction text in the first record
    # (we keep it robust if you change instruction format later)
    action_dim = None
    if test_records:
        first_user = test_records[0]["messages"][0]["content"]
        for c in first_user:
            if c.get("type") == "text":
                m = re.search(r"\[0,\s*(\d+)\]", c.get("text", ""))
                if m:
                    action_dim = int(m.group(1)) + 1
    # If not parsable, fall back to reading config if present
    if action_dim is None:
        cfg_path = adapter_dir / "vlm_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                # action_dim is not stored in cfg; still ok.
            except Exception:
                pass

    # We'll parse valid range from the instruction itself per sample to be safe.
    def get_action_dim_from_instruction(instr: str) -> Optional[int]:
        m = re.search(r"\[0,\s*(\d+)\]", instr)
        if m:
            return int(m.group(1)) + 1
        m = re.search(r"in \[0,\s*(\d+)\]", instr)
        if m:
            return int(m.group(1)) + 1
        return None

    n = len(test_records) if limit <= 0 else min(len(test_records), int(limit))
    correct = 0
    total = 0

    for i in range(n):
        sample = test_records[i]
        user_msg = sample["messages"][0]["content"]
        assistant_gt = sample["messages"][1]["content"][0]["text"]

        img_path = None
        instr = ""
        for c in user_msg:
            if c.get("type") == "image":
                img_path = c.get("image", None)
            elif c.get("type") == "text":
                instr = c.get("text", "")

        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Build inference messages: include image and instruction.
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "image"},
            ]}
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            do_sample=float(temperature) > 0.0,
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(temperature)
        out = model.generate(**inputs, **gen_kwargs)

        # Decode only generated part
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Determine valid range
        ad = get_action_dim_from_instruction(instr) or 1_000_000
        gt = _parse_action_int(str(assistant_gt), ad)
        pred = _parse_action_int(pred_text, ad)

        if gt is None or pred is None:
            total += 1
            continue

        correct += int(pred == gt)
        total += 1

        if (i + 1) % 100 == 0:
            logger.info(f"[VLM eval] {i+1}/{n} | running_acc={(correct/max(1,total))*100:.2f}%")

    acc = correct / max(1, total)
    logger.info(f"[VLM eval] done | evaluated={total} | acc={acc*100:.2f}%")
    return acc


# =============================================================================
# CLI / Main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/test CNN, ResNet, ViT, or a VLM on robot-arm image->action data.")
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing samples.csv.")
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="train or test")
    p.add_argument("--model", type=str, required=True, choices=["cnn", "resnet", "vit", "vlm"],
                   help="Which model architecture to use.")

    p.add_argument("--model_path", type=str, required=True,
                   help="Model filename/path stem; saved/loaded inside --exp_dir. "
                        "Classic: .pt file. VLM: adapter directory name.")
    p.add_argument("--exp_dir", type=str, required=True, help="Experiment output directory (all outputs go here).")

    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of samples used for training (default: 0.8).")

    # Quick-run / verbosity controls (applies to VLM; classic uses epoch logging)
    p.add_argument("--train_subset", type=int, default=0,
                   help="Use only the first N training samples (0 = use all). Useful for quick sanity tests.")
    p.add_argument("--test_subset", type=int, default=0,
                   help="Use only the first N test samples for evaluation (0 = use all).")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="(VLM) Stop training after this many optimizer steps. -1 means 'no cap' (train full epochs).")
    p.add_argument("--logging_steps", type=int, default=50,
                   help="(VLM) Log trainer metrics every N steps (default: 50).")
    p.add_argument("--disable_tqdm", action="store_true",
                   help="(VLM) Disable the progress bar to reduce console spam.")

    # Limit evaluation size
    p.add_argument("--test_frac", type=float, default=1.0,
                   help="Fraction of test split to evaluate each epoch (default: 1.0).")
    p.add_argument("--test_max_samples", type=int, default=0,
                   help="Max number of test samples to evaluate (0 = no limit).")

    # Classic training params (ignored in VLM mode unless they overlap)
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (classic models).")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (classic models).")
    p.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (classic models).")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine schedule (classic models).")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW (classic models).")

    p.add_argument("--warmup_epochs", type=int, default=0, help="Linear warmup epochs (classic models).")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"],
                   help="LR scheduler (classic models): none or cosine.")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing (classic models).")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training on CUDA (classic models).")

    p.add_argument("--image_size", type=int, default=224, help="Resize images (classic models).")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (classic models).")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available (classic models).")

    p.add_argument("--log_path", type=str, default="",
                   help="Optional log filename/path; placed inside --exp_dir (only the name is used).")

    # ViT pretrained toggle (default: on)
    vit = p.add_argument_group("vit")
    vit.add_argument("--vit_pretrained", dest="vit_pretrained", action="store_true",
                     help="Use pretrained ViT weights (default).")
    vit.add_argument("--vit_no_pretrained", dest="vit_pretrained", action="store_false",
                     help="Train ViT from scratch (not recommended).")
    p.set_defaults(vit_pretrained=True)

    # Augmentation (classic models)
    aug = p.add_argument_group("augmentation")
    aug.add_argument("--augment", dest="augment", action="store_true",
                     help="Enable training data augmentation (safe, no cropping).")
    aug.add_argument("--no_augment", dest="augment", action="store_false",
                     help="Disable training data augmentation.")
    p.set_defaults(augment=True)

    aug.add_argument("--aug_rotate_deg", type=float, default=10.0)
    aug.add_argument("--aug_translate", type=float, default=0.05)
    aug.add_argument("--aug_scale", type=float, default=0.10)
    aug.add_argument("--aug_brightness", type=float, default=0.15)
    aug.add_argument("--aug_contrast", type=float, default=0.15)
    aug.add_argument("--aug_erasing_p", type=float, default=0.0)

    # VLM options (Unsloth)
    vlm = p.add_argument_group("vlm (unsloth)")
    vlm.add_argument("--vlm_model_name", type=str,
                     default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                     help="Base VLM model name (HuggingFace/Unsloth).")
    vlm.add_argument("--vlm_load_in_4bit", action="store_true", help="Load VLM in 4bit (recommended).")
    vlm.add_argument("--vlm_no_4bit", dest="vlm_load_in_4bit", action="store_false",
                     help="Disable 4bit loading (uses more VRAM).")
    p.set_defaults(vlm_load_in_4bit=True)

    vlm.add_argument("--vlm_max_seq_length", type=int, default=2048)
    vlm.add_argument("--vlm_r", type=int, default=16)
    vlm.add_argument("--vlm_lora_alpha", type=int, default=16)
    vlm.add_argument("--vlm_lora_dropout", type=float, default=0.0)

    vlm.add_argument("--vlm_finetune_vision_layers", action="store_true")
    vlm.add_argument("--vlm_no_finetune_vision_layers", dest="vlm_finetune_vision_layers", action="store_false")
    p.set_defaults(vlm_finetune_vision_layers=True)

    vlm.add_argument("--vlm_finetune_language_layers", action="store_true")
    vlm.add_argument("--vlm_no_finetune_language_layers", dest="vlm_finetune_language_layers", action="store_false")
    p.set_defaults(vlm_finetune_language_layers=True)

    vlm.add_argument("--vlm_finetune_attention_modules", action="store_true")
    vlm.add_argument("--vlm_no_finetune_attention_modules", dest="vlm_finetune_attention_modules", action="store_false")
    p.set_defaults(vlm_finetune_attention_modules=True)

    vlm.add_argument("--vlm_finetune_mlp_modules", action="store_true")
    vlm.add_argument("--vlm_no_finetune_mlp_modules", dest="vlm_finetune_mlp_modules", action="store_false")
    p.set_defaults(vlm_finetune_mlp_modules=True)

    vlm.add_argument("--vlm_train_batch_size", type=int, default=2)
    vlm.add_argument("--vlm_grad_accum", type=int, default=4)
    vlm.add_argument("--vlm_lr", type=float, default=2e-4)
    vlm.add_argument("--vlm_weight_decay", type=float, default=1e-3)
    vlm.add_argument("--vlm_warmup_steps", type=int, default=5)
    vlm.add_argument("--vlm_epochs", type=int, default=2)
    vlm.add_argument("--vlm_logging_steps", type=int, default=None,
                     help="Override --logging_steps for VLM only (optional).")
    vlm.add_argument("--vlm_optim", type=str, default="adamw_8bit")
    vlm.add_argument("--vlm_lr_scheduler_type", type=str, default="linear")
    vlm.add_argument("--vlm_seed", type=int, default=3407)

    vlm.add_argument("--vlm_max_new_tokens", type=int, default=16)
    vlm.add_argument("--vlm_temperature", type=float, default=0.2)

    # VLM eval limit in train mode quick-eval
    vlm.add_argument("--vlm_quick_eval_max", type=int, default=200,
                     help="Max samples for quick eval after training (0 disables).")

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
    model_path_name = Path(args.model_path).name

    log_name = Path(args.log_path).name if args.log_path.strip() else f"{Path(model_path_name).stem}.log"
    log_path = exp_dir / log_name
    logger = setup_logger(log_path)

    pairs, action_dim = load_samples_csv(data_dir)
    train_pairs, test_pairs_full = split_pairs(pairs, args.train_frac)

    # Optional quick-test subsets (deterministic: keep order as in CSV)
    if int(args.train_subset) > 0:
        train_pairs = train_pairs[: int(args.train_subset)]
    if int(args.test_subset) > 0:
        test_pairs_full = test_pairs_full[: int(args.test_subset)]

    test_pairs = limit_test_pairs(test_pairs_full, test_frac=args.test_frac, test_max_samples=args.test_max_samples)

    logger.info("=== Run ===")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data dir: {data_dir.resolve()}")
    logger.info(f"Experiment dir: {exp_dir.resolve()}")
    logger.info(f"Total samples: {len(pairs)} | Train: {len(train_pairs)} | Test(full): {len(test_pairs_full)} | Test(eval): {len(test_pairs)}")
    logger.info(f"Action dim (num classes): {action_dim}")

    # -------------------------------------------------------------------------
    # VLM PATH
    # -------------------------------------------------------------------------
    if args.model == "vlm":
        adapter_dir = exp_dir / model_path_name  # directory

        if args.mode == "train":
            train_records = make_vlm_conversation_records(data_dir, train_pairs, action_dim=action_dim)
            test_records = make_vlm_conversation_records(data_dir, test_pairs, action_dim=action_dim)

            cfg = VLMConfig(
                model_name=str(args.vlm_model_name),
                load_in_4bit=bool(args.vlm_load_in_4bit),
                max_seq_length=int(args.vlm_max_seq_length),
                r=int(args.vlm_r),
                lora_alpha=int(args.vlm_lora_alpha),
                lora_dropout=float(args.vlm_lora_dropout),
                finetune_vision_layers=bool(args.vlm_finetune_vision_layers),
                finetune_language_layers=bool(args.vlm_finetune_language_layers),
                finetune_attention_modules=bool(args.vlm_finetune_attention_modules),
                finetune_mlp_modules=bool(args.vlm_finetune_mlp_modules),
                per_device_train_batch_size=int(args.vlm_train_batch_size),
                gradient_accumulation_steps=int(args.vlm_grad_accum),
                learning_rate=float(args.vlm_lr),
                weight_decay=float(args.vlm_weight_decay),
                warmup_steps=int(args.vlm_warmup_steps),
                num_train_epochs=int(args.vlm_epochs),
                logging_steps=int(args.vlm_logging_steps if args.vlm_logging_steps is not None else args.logging_steps),
                max_steps=int(args.max_steps),
                disable_tqdm=bool(args.disable_tqdm),
                optim=str(args.vlm_optim),
                lr_scheduler_type=str(args.vlm_lr_scheduler_type),
                seed=int(args.vlm_seed),
                max_new_tokens=int(args.vlm_max_new_tokens),
                temperature=float(args.vlm_temperature),
            )

            train_vlm_with_unsloth(
                train_records=train_records,
                test_records=test_records,
                exp_dir=exp_dir,
                adapter_dir_name=model_path_name,
                cfg=cfg,
                logger=logger,
            )
            logger.info("VLM training finished.")
            return 0

        # test mode
        if not adapter_dir.exists():
            logger.error(f"Adapter directory not found: {adapter_dir.resolve()}")
            return 1

        test_records = make_vlm_conversation_records(data_dir, test_pairs, action_dim=action_dim)
        acc = evaluate_vlm_action_accuracy(
            adapter_dir=adapter_dir,
            test_records=test_records,
            max_new_tokens=int(args.vlm_max_new_tokens),
            temperature=float(args.vlm_temperature),
            logger=logger,
            limit=0,
        )
        logger.info(f"[VLM] Test accuracy: {acc*100:.2f}%")
        return 0

    # -------------------------------------------------------------------------
    # CLASSIC PATH (CNN/ResNet/ViT)
    # -------------------------------------------------------------------------
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info(f"Device: {device}")

    augment_cfg = {
        "augment": bool(args.augment),
        "aug_rotate_deg": float(args.aug_rotate_deg),
        "aug_translate": float(args.aug_translate),
        "aug_scale": float(args.aug_scale),
        "aug_brightness": float(args.aug_brightness),
        "aug_contrast": float(args.aug_contrast),
        "aug_erasing_p": float(args.aug_erasing_p),
        "vit_pretrained": bool(args.vit_pretrained),
        "safe_augmentation_no_crop": True,
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
        "vit_pretrained": bool(args.vit_pretrained),
    }

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
        model_name=args.model,
        vit_pretrained=bool(args.vit_pretrained),
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

    # Classic model path (single file)
    base_model_path = exp_dir / model_path_name
    if not base_model_path.suffix:
        base_model_path = base_model_path.with_suffix(".pt")

    plot_path = exp_dir / f"{base_model_path.stem}_learning_curve.png"
    history_path = exp_dir / f"{base_model_path.stem}_history.json"

    if args.mode == "train":
        model = build_model(args.model, num_classes=action_dim, vit_pretrained=bool(args.vit_pretrained)).to(device)

        logger.info("=== Training (classic) ===")
        logger.info(f"Checkpoints: {base_model_path.stem}_epoch###.pt (in exp_dir)")
        logger.info(f"Plot (latest): {plot_path.resolve()}")
        logger.info(f"History JSON: {history_path.resolve()}")

        train_classifier_loop(
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

    # test mode (classic)
    logger.info("=== Testing (classic) ===")
    model_to_load = exp_dir / model_path_name
    if not model_to_load.suffix:
        model_to_load = model_to_load.with_suffix(".pt")

    payload = load_classifier_checkpoint_payload(model_to_load, device=device)
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

    model = build_model(args.model, num_classes=saved_action_dim, vit_pretrained=bool(args.vit_pretrained)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    logger.info(f"Loaded checkpoint: {model_to_load.resolve()}")
    logger.info(f"Checkpoint epoch: {saved_epoch}")
    if saved_cfg:
        logger.info(f"Checkpoint train_cfg: {saved_cfg}")

    test_loss, test_acc = evaluate_classifier(model, test_loader, device, label_smoothing=float(args.label_smoothing))
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test accuracy: {test_acc*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
