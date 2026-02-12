#!/usr/bin/env python3
"""
robot_arm_controller.py  –  Config-driven experiment runner

Compares CNN, ResNet, ViT, and VLA models on robot-arm image → action prediction.

Usage:
  python robot_arm_controller.py --config experiments.yaml

Toggle experiments on/off via the "enabled" flag in the YAML.

Each experiment produces a result folder:
  <results_dir>/experiment001_<name>_<dataset>/
      ├── training.log
      ├── learning_curves.png
      └── final_model.pt   (or final_model/ for VLA adapters)

A combined results.txt table is written to <results_dir>/results.txt.

Requirements (classic models): pip install torch torchvision pillow matplotlib pyyaml
Requirements (VLA):            pip install unsloth trl transformers accelerate bitsandbytes
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from torchvision import transforms
    import torchvision
except ImportError as e:
    raise ImportError("torchvision is required: pip install torchvision") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required: pip install matplotlib") from e


# =============================================================================
# Logging
# =============================================================================
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"experiment_{log_path.stem}")
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
    return logger


# =============================================================================
# Dataset
# =============================================================================
class RobotArmDataset(Dataset):
    def __init__(self, data_dir: Path, pairs: List[Tuple[str, int]], transform=None):
        self.data_dir = data_dir
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rel_path, class_idx = self.pairs[idx]
        img = Image.open(self.data_dir / rel_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(class_idx, dtype=torch.long)


def load_samples_csv(data_dir: Path) -> Tuple[List[Tuple[str, int]], int]:
    csv_path = data_dir / "samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing samples.csv in {data_dir}")
    pairs: List[Tuple[str, int]] = []
    action_dim: Optional[int] = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            onehot = json.loads(row["action"].strip())
            if action_dim is None:
                action_dim = len(onehot)
            class_idx = int(max(range(len(onehot)), key=lambda i: onehot[i]))
            pairs.append((row["image_filename"].strip(), class_idx))
    if action_dim is None:
        raise ValueError("samples.csv is empty")
    return pairs, action_dim


def split_data(pairs, train_frac):
    n_train = max(1, min(len(pairs) - 1, int(len(pairs) * train_frac)))
    return pairs[:n_train], pairs[n_train:]


# =============================================================================
# Models
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """Custom lightweight ResNet for synthetic robot arm images."""
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True),
        )
        self.in_ch = c1
        self.layer1 = self._make_layer(c1, 2, stride=1)
        self.layer2 = self._make_layer(c2, 2, stride=2)
        self.layer3 = self._make_layer(c3, 2, stride=2)
        self.layer4 = self._make_layer(c4, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def build_resnet18_pretrained(num_classes: int) -> nn.Module:
    from torchvision.models import resnet18
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
    except ImportError:
        weights = "imagenet"
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision.models import vit_b_16
    weights = None
    if pretrained:
        try:
            from torchvision.models import ViT_B_16_Weights
            weights = ViT_B_16_Weights.DEFAULT
        except Exception:
            pass
    model = vit_b_16(weights=weights)
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise RuntimeError("Unexpected ViT head structure")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = model_name.lower().strip()
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "resnet":
        return build_resnet18_pretrained(num_classes) if pretrained else SmallResNet(num_classes)
    if name == "vit":
        return build_vit(num_classes, pretrained)
    raise ValueError(f"Unknown model: '{name}'. Use 'cnn', 'resnet', 'vit', or 'vla'.")


# =============================================================================
# Transforms
# =============================================================================
def _imagenet_norm():
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(cfg: dict, model_name: str, pretrained: bool):
    img_size = cfg["image_size"]
    use_imagenet = pretrained and model_name in ("vit", "resnet")
    mean, std = _imagenet_norm() if use_imagenet else ((0.5,) * 3, (0.5,) * 3)
    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    aug = cfg.get("augmentation", {})
    if not aug.get("enabled", False):
        return test_tfm, test_tfm
    min_scale = max(0.7, 1.0 - aug.get("scale", 0.1))
    max_scale = 1.0 + aug.get("scale", 0.1)
    train_ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=aug.get("rotate_deg", 10),
            translate=(aug.get("translate", 0.05),) * 2,
            scale=(min_scale, max_scale),
            fill=255,
        ),
        transforms.ColorJitter(
            brightness=aug.get("brightness", 0.15),
            contrast=aug.get("contrast", 0.15),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    erasing_p = aug.get("erasing_p", 0.0)
    if erasing_p > 0:
        train_ops.append(transforms.RandomErasing(p=erasing_p, scale=(0.01, 0.06)))
    return transforms.Compose(train_ops), test_tfm


# =============================================================================
# Evaluation & Inference Timing
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def measure_inference_time_ms(model, device, image_size, num_warmup=10, num_runs=100):
    """Measure average inference time per single image in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup
    for _ in range(num_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        for _ in range(num_runs):
            model(dummy)
        elapsed = time.perf_counter() - start

    return (elapsed / num_runs) * 1000.0  # ms per image


# =============================================================================
# Plotting
# =============================================================================
def plot_learning_curves(path: Path, history: dict, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["test_loss"], label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["test_acc"]], label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Training (classic models)
# =============================================================================
def train_classic(exp_cfg: dict, defaults: dict, data_dir: Path, exp_dir: Path, logger: logging.Logger) -> dict:
    """Train a CNN / ResNet / ViT model. Returns metrics dict."""
    cfg = {**defaults}
    for key in ("epochs", "batch_size", "lr", "min_lr", "weight_decay",
                "warmup_epochs", "scheduler", "label_smoothing", "image_size",
                "num_workers", "use_amp"):
        if key in exp_cfg:
            cfg[key] = exp_cfg[key]

    # YAML parses scientific notation like 1e-3 as strings → cast numerics
    for key in ("lr", "min_lr", "weight_decay", "label_smoothing"):
        cfg[key] = float(cfg[key])
    for key in ("epochs", "batch_size", "warmup_epochs", "image_size", "num_workers"):
        cfg[key] = int(cfg[key])

    model_name = exp_cfg["model"]
    pretrained = exp_cfg.get("pretrained", False)
    quicktest = defaults.get("quicktest", False)

    pairs, action_dim = load_samples_csv(data_dir)

    if quicktest:
        # Smoke test: first 5% for train, last 5% for test
        n = len(pairs)
        chunk = max(1, int(n * 0.01))
        train_pairs = pairs[:chunk]
        test_pairs = pairs[-chunk:]
        cfg["epochs"] = 1
        logger.info(f"*** QUICKTEST MODE: 1 epoch, {len(train_pairs)} train, {len(test_pairs)} test ***")
    else:
        train_pairs, test_pairs = split_data(pairs, cfg["train_split"])

    logger.info(f"Dataset: {data_dir} | Samples: {len(pairs)} | Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    logger.info(f"Action dim: {action_dim} | Model: {model_name} | Pretrained: {pretrained}")
    logger.info(f"Epochs: {cfg['epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['lr']} | Scheduler: {cfg['scheduler']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_tfm, test_tfm = build_transforms(cfg, model_name, pretrained)
    train_ds = RobotArmDataset(data_dir, train_pairs, train_tfm)
    test_ds = RobotArmDataset(data_dir, test_pairs, test_tfm)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))

    model = build_model(model_name, action_dim, pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    epochs = cfg["epochs"]
    base_lr = cfg["lr"]
    min_lr = cfg["min_lr"]
    warmup_epochs = cfg["warmup_epochs"]
    scheduler_name = cfg["scheduler"]
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_for_step(step):
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        if scheduler_name != "cosine":
            return base_lr
        if total_steps <= warmup_steps:
            return min_lr
        t = (step - warmup_steps) / float(total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    global_step = 0

    # ---- Training loop with wall-clock timing ----
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_start = datetime.now()

        num_batches = len(train_loader)
        next_log_pct = 10  # log at every 10% milestone

        for b_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr_now = lr_for_step(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
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
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

            # Progress log every 10% of batches
            pct_done = b_idx * 100 // num_batches
            if pct_done >= next_log_pct:
                logger.info(
                    f"  Epoch {epoch:03d}/{epochs} | {pct_done:3d}% "
                    f"({b_idx}/{num_batches}) | "
                    f"lr={lr_now:.6g} | "
                    f"loss={running_loss/total:.4f} | "
                    f"acc={correct/total*100:.2f}%"
                )
                next_log_pct = pct_done + 10

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        test_loss, test_acc = evaluate(model, test_loader, device, cfg["label_smoothing"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        elapsed = datetime.now() - epoch_start
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}% | "
            f"time={elapsed}"
        )

    train_elapsed_sec = time.perf_counter() - train_start
    train_minutes = train_elapsed_sec / 60.0
    logger.info(f"Total training time: {format_time(train_elapsed_sec)}")

    # ---- Inference timing ----
    inference_ms = measure_inference_time_ms(model, device, cfg["image_size"])
    logger.info(f"Inference time per image: {inference_ms:.2f} ms")

    # ---- Save final model only ----
    model_path = exp_dir / "final_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "pretrained": pretrained,
        "action_dim": action_dim,
        "image_size": cfg["image_size"],
        "epochs_trained": epochs,
        "history": history,
        "config": cfg,
        "train_time_sec": train_elapsed_sec,
        "inference_ms_per_image": inference_ms,
    }, model_path)
    logger.info(f"Saved final model: {model_path}")

    # ---- Plot ----
    plot_path = exp_dir / "learning_curves.png"
    title = f"{model_name.upper()}{' (pretrained)' if pretrained else ''}"
    plot_learning_curves(plot_path, history, title)
    logger.info(f"Saved plot: {plot_path}")

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_train_acc = history["train_acc"][-1]
    final_test_acc = history["test_acc"][-1]
    logger.info(f"Final train accuracy: {final_train_acc*100:.2f}%")
    logger.info(f"Final test accuracy:  {final_test_acc*100:.2f}%")

    return {
        "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "train_time_sec": train_elapsed_sec,
        "time_per_epoch_sec": train_elapsed_sec / max(1, epochs),
        "epochs": epochs,
        "inference_ms": inference_ms,
    }


# =============================================================================
# VLA (Unsloth) training
# =============================================================================
def _vlm_instruction(action_dim: int) -> str:
    return (
        "You control a planar robot arm. "
        "Given the image, output the best expert action as a single integer in [0, "
        f"{action_dim-1}]. "
        "Action encoding: idx = dof_index*3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree. "
        "Answer with ONLY the integer."
    )


def make_vlm_records(data_dir, pairs, action_dim):
    instruction = _vlm_instruction(action_dim)
    records = []
    for rel_path, class_idx in pairs:
        img_path = str((data_dir / rel_path).resolve())
        records.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": img_path},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": str(class_idx)},
                ]},
            ]
        })
    return records


def _parse_action_int(text, action_dim):
    if not text:
        return None
    ms = re.findall(r"-?\d+", text)
    if not ms:
        return None
    val = int(ms[-1])
    return val if 0 <= val < action_dim else None


def evaluate_vla(adapter_dir, test_records, max_new_tokens, temperature, logger, action_dim):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(str(adapter_dir), load_in_4bit=True)
    FastVisionModel.for_inference(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    correct, total = 0, 0
    total_inference_ms = 0.0

    for i, sample in enumerate(test_records):
        user_msg = sample["messages"][0]["content"]
        gt_text = sample["messages"][1]["content"][0]["text"]

        img_path, instr = None, ""
        for c in user_msg:
            if c.get("type") == "image":
                img_path = c["image"]
            elif c.get("type") == "text":
                instr = c["text"]
        if img_path is None:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        messages = [{"role": "user", "content": [
            {"type": "text", "text": instr}, {"type": "image"}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": True,
                      "do_sample": temperature > 0.0}
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temperature

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_inference_ms += (time.perf_counter() - t0) * 1000.0

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        gt = _parse_action_int(gt_text, action_dim)
        pred = _parse_action_int(pred_text, action_dim)
        if gt is not None:
            total += 1
            if pred == gt:
                correct += 1

        if (i + 1) % 100 == 0:
            logger.info(f"[VLA eval] {i+1}/{len(test_records)} | running_acc={correct/max(1,total)*100:.2f}%")

    acc = correct / max(1, total)
    avg_ms = total_inference_ms / max(1, total)
    logger.info(f"[VLA eval] done | evaluated={total} | acc={acc*100:.2f}% | avg_inference={avg_ms:.2f} ms/image")
    return acc, avg_ms


def train_vla(exp_cfg: dict, defaults: dict, data_dir: Path, exp_dir: Path, logger: logging.Logger) -> dict:
    """Train a VLA model. Returns metrics dict."""
    try:
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise ImportError("VLA requires: pip install unsloth trl transformers accelerate bitsandbytes") from e

    vla_defaults = defaults.get("vla", {})
    vla_cfg = {**vla_defaults, **exp_cfg.get("vla", {})}

    # YAML parses scientific notation as strings → cast numerics
    for key in ("lr", "weight_decay", "lora_dropout", "temperature"):
        if key in vla_cfg:
            vla_cfg[key] = float(vla_cfg[key])
    for key in ("lora_r", "lora_alpha", "train_batch_size", "gradient_accumulation_steps",
                "warmup_steps", "epochs", "seed", "max_new_tokens", "max_seq_length"):
        if key in vla_cfg:
            vla_cfg[key] = int(vla_cfg[key])

    pairs, action_dim = load_samples_csv(data_dir)
    quicktest = defaults.get("quicktest", False)

    # Experiment-level epochs override vla defaults
    if "epochs" in exp_cfg and not quicktest:
        vla_cfg["epochs"] = int(exp_cfg["epochs"])

    if quicktest:
        n = len(pairs)
        chunk = max(1, int(n * 0.01))
        train_pairs = pairs[:chunk]
        test_pairs = pairs[-chunk:]
        vla_cfg["epochs"] = 1
        logger.info(f"*** QUICKTEST MODE: 1 epoch, {len(train_pairs)} train, {len(test_pairs)} test ***")
    else:
        train_pairs, test_pairs = split_data(pairs, defaults["train_split"])

    logger.info(f"Dataset: {data_dir} | Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    logger.info(f"Action dim: {action_dim} | VLA base: {vla_cfg['base_model']}")

    train_records = make_vlm_records(data_dir, train_pairs, action_dim)
    test_records = make_vlm_records(data_dir, test_pairs, action_dim)

    seed = vla_cfg.get("seed", 3407)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = FastVisionModel.from_pretrained(
        vla_cfg["base_model"],
        load_in_4bit=vla_cfg.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=vla_cfg.get("finetune_vision_layers", True),
        finetune_language_layers=vla_cfg.get("finetune_language_layers", True),
        finetune_attention_modules=vla_cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules=vla_cfg.get("finetune_mlp_modules", True),
        r=vla_cfg.get("lora_r", 16),
        lora_alpha=vla_cfg.get("lora_alpha", 16),
        lora_dropout=vla_cfg.get("lora_dropout", 0.0),
        bias="none",
        random_state=seed,
    )
    FastVisionModel.for_training(model)

    adapter_dir = exp_dir / "final_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    with open(adapter_dir / "vla_config.json", "w") as f:
        json.dump(vla_cfg, f, indent=2)

    sft_args = SFTConfig(
        per_device_train_batch_size=vla_cfg.get("train_batch_size", 2),
        gradient_accumulation_steps=vla_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=vla_cfg.get("warmup_steps", 5),
        num_train_epochs=vla_cfg.get("epochs", 2),
        learning_rate=vla_cfg.get("lr", 2e-4),
        logging_steps=50,
        max_steps=-1,
        optim=vla_cfg.get("optimizer", "adamw_8bit"),
        weight_decay=vla_cfg.get("weight_decay", 1e-3),
        lr_scheduler_type=vla_cfg.get("lr_scheduler", "linear"),
        seed=seed,
        output_dir=str(exp_dir / "trainer_tmp"),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=vla_cfg.get("max_seq_length", 2048),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_records,
        args=sft_args,
    )

    train_start = time.perf_counter()
    trainer_stats = trainer.train()
    train_elapsed_sec = time.perf_counter() - train_start
    logger.info(f"Total VLA training time: {format_time(train_elapsed_sec)}")

    try:
        metrics = dict(trainer_stats.metrics)
        with open(exp_dir / "trainer_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Saved VLA adapter to: {adapter_dir}")

    # Evaluate + measure inference time
    max_new_tokens = vla_cfg.get("max_new_tokens", 16)
    temperature = vla_cfg.get("temperature", 0.2)
    train_acc = 0.0  # VLA: we don't re-evaluate on train set (too expensive)
    test_acc = 0.0
    inference_ms = 0.0

    try:
        test_acc, inference_ms = evaluate_vla(adapter_dir, test_records, max_new_tokens, temperature, logger, action_dim)
        logger.info(f"Final VLA test accuracy: {test_acc*100:.2f}%")
    except Exception as e:
        logger.warning(f"VLA evaluation failed: {e}")

    # Clean up trainer temp files
    import shutil
    trainer_tmp = exp_dir / "trainer_tmp"
    if trainer_tmp.exists():
        shutil.rmtree(trainer_tmp, ignore_errors=True)

    vla_epochs = vla_cfg.get("epochs", 2)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_time_sec": train_elapsed_sec,
        "time_per_epoch_sec": train_elapsed_sec / max(1, vla_epochs),
        "epochs": vla_epochs,
        "inference_ms": inference_ms,
    }


# =============================================================================
# Helpers
# =============================================================================
def format_time(seconds: float) -> str:
    """Format seconds into a human readable string like '12m 34s' or '1h 23m 45s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_experiment_dir_name(index: int, exp_name: str, data_dir: str) -> str:
    dataset_name = Path(data_dir).name
    return f"experiment{index:03d}_{exp_name}_{dataset_name}"


def write_results_table(results_path: Path, rows: list):
    """Write a formatted results summary table."""
    header = (
        f"{'#':>3}  {'Experiment':<25}  {'Epochs':>6}  {'Train Acc':>10}  {'Test Acc':>10}  "
        f"{'Total Time':>12}  {'Time/Epoch':>12}  {'Infer (ms)':>11}"
    )
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "EXPERIMENT RESULTS SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for row in rows:
        if row["status"] == "ok":
            epochs_str = str(row["epochs"])
            train_acc_str = f"{row['train_acc']*100:.2f}%"
            test_acc_str = f"{row['test_acc']*100:.2f}%"
            total_str = format_time(row["train_time_sec"])
            per_epoch_str = format_time(row["time_per_epoch_sec"])
            infer_str = f"{row['inference_ms']:.2f}"
        elif row["status"] == "skipped":
            epochs_str = train_acc_str = test_acc_str = total_str = per_epoch_str = infer_str = "—"
        else:
            epochs_str = train_acc_str = test_acc_str = total_str = per_epoch_str = infer_str = "FAILED"

        lines.append(
            f"{row['index']:>3}  {row['name']:<25}  {epochs_str:>6}  {train_acc_str:>10}  {test_acc_str:>10}  "
            f"{total_str:>12}  {per_epoch_str:>12}  {infer_str:>11}"
        )

    lines.append(sep)
    lines.append("")

    text = "\n".join(lines)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Also print to console
    print("\n" + text)


# =============================================================================
# CLI & Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Run robot arm model comparison experiments.")
    p.add_argument("--config", type=str, required=True, help="Path to experiments YAML config file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])

    if not experiments:
        print("No experiments defined in config.")
        return 1

    results_dir = Path(defaults.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []

    for i, exp_cfg in enumerate(experiments, start=1):
        exp_name = exp_cfg.get("name", exp_cfg["model"])

        # Check enabled flag
        if not exp_cfg.get("enabled", True):
            print(f"[{i}] {exp_name}: SKIPPED (enabled: false)")
            result_rows.append({"index": i, "name": exp_name, "status": "skipped"})
            continue

        data_dir = Path(exp_cfg.get("data_dir", defaults.get("data_dir", "./data")))
        dir_name = make_experiment_dir_name(i, exp_name, str(data_dir))
        exp_dir = results_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        log_path = exp_dir / "training.log"
        logger = setup_logger(log_path)

        logger.info("=" * 70)
        logger.info(f"EXPERIMENT {i}: {exp_name}")
        if defaults.get("quicktest", False):
            logger.info("*** QUICKTEST MODE ***")
        logger.info("=" * 70)

        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump({"index": i, "experiment": exp_cfg, "defaults": defaults}, f, indent=2)

        model_type = exp_cfg["model"].lower()
        try:
            if model_type == "vla":
                metrics = train_vla(exp_cfg, defaults, data_dir, exp_dir, logger)
            else:
                metrics = train_classic(exp_cfg, defaults, data_dir, exp_dir, logger)

            result_rows.append({"index": i, "name": exp_name, "status": "ok", **metrics})
            logger.info(f"Experiment {i} ({exp_name}) completed successfully.")

        except Exception as e:
            logger.error(f"Experiment {i} ({exp_name}) FAILED: {e}", exc_info=True)
            result_rows.append({"index": i, "name": exp_name, "status": "failed"})

        # Close logger handlers
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    # Write combined results table
    results_path = results_dir / "results.txt"
    write_results_table(results_path, result_rows)
    print(f"\nResults written to: {results_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())