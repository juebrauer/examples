#!/usr/bin/env python3
"""
robot_arm_controller.py

Main orchestrator script for robot arm controller experiments.
All controller classes are included in this single file for easy deployment.

Models compared:
- CNN (from scratch only - no pretrained weights available)
- ResNet (random init vs ImageNet pretrained)
- ViT (random init vs ImageNet pretrained)
- VLA (Qwen3-VL-2B with fresh LoRA adapters)

Modes:
- train: Train a single model
- test: Test a single model
- do_all_experiments: Run all experiments (train + test for all models)

Usage:
  # Quick test (1% of data)
  python robot_arm_controller.py --data_dir ./data_2dof_100000 --exp_dir ./experiments --mode do_all_experiments --quick

  # Full training
  python robot_arm_controller.py --data_dir ./data_2dof_100000 --exp_dir ./experiments --mode do_all_experiments

Requirements:
  pip install torch torchvision pillow matplotlib numpy

Additional requirements (VLA):
  pip install unsloth trl transformers accelerate bitsandbytes
"""

import argparse
import csv
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required (pip install matplotlib).") from e


# =============================================================================
# Configuration dataclasses
# =============================================================================
@dataclass
class TrainingConfig:
    """Common training configuration."""
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    use_amp: bool = True
    image_size: int = 224
    num_workers: int = 2
    seed: int = 42
    
    # Augmentation
    augment: bool = True
    aug_rotate_deg: float = 10.0
    aug_translate: float = 0.05
    aug_scale: float = 0.10
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    aug_erasing_p: float = 0.0


@dataclass
class TrainingHistory:
    """Stores training history for plotting."""
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[float]]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "TrainingHistory":
        return cls(**d)


@dataclass
class ExperimentResult:
    """Stores final experiment results."""
    model_name: str          # e.g., "CNN", "ResNet (random)", "ResNet (pretrained)", "VLA"
    final_train_acc: float
    final_test_acc: float
    total_training_time_seconds: float
    inference_time_per_image_seconds: float
    num_epochs: int
    num_train_samples: int
    num_test_samples: int


@dataclass
class VLAConfig:
    """VLA-specific configuration."""
    model_name: str = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 5
    num_train_epochs: int = 2
    logging_steps: int = 50
    max_steps: int = -1
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    seed: int = 42
    max_new_tokens: int = 16
    temperature: float = 0.2


# =============================================================================
# Utility functions
# =============================================================================
def set_global_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_path: Path, name: str = "controller") -> logging.Logger:
    """Setup logger with file and console handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
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


def plot_learning_curves(history: TrainingHistory, out_path: Path, title: str = "Learning Curves") -> None:
    """Plot and save learning curves."""
    if not history.train_loss:
        return
        
    epochs = list(range(1, len(history.train_loss) + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(epochs, history.train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history.test_loss, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, [a * 100 for a in history.train_acc], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history.test_acc], 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Dataset
# =============================================================================
class RobotArmDataset(Dataset):
    """Dataset for robot arm images and actions."""
    
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


# =============================================================================
# Abstract Base Controller
# =============================================================================
class BaseController(ABC):
    """Abstract base class for robot arm controllers."""
    
    def __init__(
        self,
        model_name: str,
        display_name: str,
        num_classes: int,
        config: TrainingConfig,
        exp_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.display_name = display_name
        self.num_classes = num_classes
        self.config = config
        self.exp_dir = Path(exp_dir)
        
        # Create model-specific output directory
        safe_dir_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        self.model_dir = self.exp_dir / safe_dir_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if logger is None:
            log_path = self.model_dir / f"{safe_dir_name}.log"
            self.logger = setup_logger(log_path, name=f"{safe_dir_name}_{id(self)}")
        else:
            self.logger = logger
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (to be built by subclass)
        self.model: Optional[nn.Module] = None
        
        # Training state
        self.history = TrainingHistory()
        self.total_training_time = 0.0
        
        # Set seed
        set_global_seed(config.seed)
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def get_transforms(self, training: bool = False):
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        test_subsample_loader: Optional[DataLoader] = None,
    ) -> TrainingHistory:
        if self.model is None:
            self.model = self.build_model().to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        
        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
        
        total_steps = self.config.epochs * len(train_loader)
        warmup_steps = self.config.warmup_epochs * len(train_loader)
        global_step = 0
        
        self.logger.info(f"Starting training: {self.config.epochs} epochs, {len(train_loader)} batches/epoch")
        self.logger.info(f"Device: {self.device}, AMP: {amp_enabled}")
        
        training_start = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (x, y) in enumerate(train_loader, 1):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                lr = self._compute_lr(global_step, total_steps, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                
                optimizer.zero_grad(set_to_none=True)
                
                if amp_enabled and scaler is not None:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                
                global_step += 1
                running_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
                
                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    self.logger.info(
                        f"[{self.display_name}] Epoch {epoch}/{self.config.epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {running_loss/total:.4f} | Acc: {100*correct/total:.2f}%"
                    )
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            eval_loader = test_subsample_loader if test_subsample_loader else test_loader
            test_loss, test_acc = self._evaluate(eval_loader, criterion)
            
            epoch_time = time.time() - epoch_start
            
            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.test_loss.append(test_loss)
            self.history.test_acc.append(test_acc)
            self.history.epoch_times.append(epoch_time)
            
            self.logger.info(
                f"[{self.display_name}] Epoch {epoch}/{self.config.epochs} DONE | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {100*test_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
            
            plot_path = self.model_dir / f"{self.model_name}_learning_curve.png"
            plot_learning_curves(self.history, plot_path, f"{self.display_name} Learning Curves")
        
        self.total_training_time = time.time() - training_start
        self.logger.info(f"Total training time: {self.total_training_time:.1f}s")
        
        self.save()
        
        history_path = self.model_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)
        
        return self.history
    
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        return self._evaluate(test_loader, criterion)
    
    def measure_inference_time(self, test_loader: DataLoader, num_samples: int = 100) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        times = []
        count = 0
        
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device, non_blocking=True)
                
                if count == 0:
                    _ = self.model(x)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                batch_time = (end - start) / x.size(0)
                times.extend([batch_time] * x.size(0))
                count += x.size(0)
                
                if count >= num_samples:
                    break
        
        return sum(times[:num_samples]) / min(len(times), num_samples) if times else 0.0
    
    def save(self, path: Optional[Path] = None) -> Path:
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        if path is None:
            path = self.model_dir / f"{self.model_name}_final.pt"
        
        checkpoint = {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "state_dict": self.model.state_dict(),
            "num_classes": self.num_classes,
            "config": asdict(self.config),
            "history": self.history.to_dict(),
            "total_training_time": self.total_training_time,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        return path
    
    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self.model_dir / f"{self.model_name}_final.pt"
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if self.model is None:
            self.model = self.build_model().to(self.device)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        
        if "history" in checkpoint:
            self.history = TrainingHistory.from_dict(checkpoint["history"])
        if "total_training_time" in checkpoint:
            self.total_training_time = checkpoint["total_training_time"]
        
        self.logger.info(f"Loaded checkpoint: {path}")
    
    def get_experiment_result(self, test_loader: DataLoader, num_train_samples: int) -> ExperimentResult:
        test_loss, test_acc = self.test(test_loader)
        inference_time = self.measure_inference_time(test_loader)
        
        return ExperimentResult(
            model_name=self.display_name,
            final_train_acc=self.history.train_acc[-1] if self.history.train_acc else 0.0,
            final_test_acc=test_acc,
            total_training_time_seconds=self.total_training_time,
            inference_time_per_image_seconds=inference_time,
            num_epochs=len(self.history.train_loss),
            num_train_samples=num_train_samples,
            num_test_samples=len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0,
        )
    
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            logits = self.model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        
        return total_loss / max(1, total), correct / max(1, total)
    
    def _compute_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return self.config.lr * (step + 1) / warmup_steps
        
        if self.config.scheduler.lower() != "cosine":
            return self.config.lr
        
        if total_steps <= warmup_steps:
            return self.config.min_lr
        
        t = (step - warmup_steps) / float(total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return self.config.min_lr + 0.5 * (self.config.lr - self.config.min_lr) * (1.0 + np.cos(np.pi * t))


# =============================================================================
# CNN Model
# =============================================================================
class SimpleCNN(nn.Module):
    """Baseline CNN architecture."""
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNController(BaseController):
    """CNN-based controller (trained from scratch only)."""
    
    def __init__(self, num_classes: int, config: TrainingConfig, exp_dir: Path,
                 logger: Optional[logging.Logger] = None):
        super().__init__(
            model_name="cnn",
            display_name="CNN",
            num_classes=num_classes,
            config=config,
            exp_dir=exp_dir,
            logger=logger,
        )
    
    def build_model(self) -> nn.Module:
        model = SimpleCNN(num_classes=self.num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"CNN parameters: {total_params:,}")
        return model
    
    def get_transforms(self, training: bool = False):
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        if not training or not self.config.augment:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        min_scale = max(0.7, 1.0 - self.config.aug_scale)
        max_scale = 1.0 + self.config.aug_scale
        
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomAffine(
                degrees=self.config.aug_rotate_deg,
                translate=(self.config.aug_translate, self.config.aug_translate),
                scale=(min_scale, max_scale),
                fill=255,
            ),
            transforms.ColorJitter(brightness=self.config.aug_brightness, contrast=self.config.aug_contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# =============================================================================
# ResNet Model
# =============================================================================
class BasicBlock(nn.Module):
    """ResNet basic block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """Custom lightweight ResNet."""
    
    def __init__(self, num_classes: int, layers=(2, 2, 2, 2), channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        
        self.in_channels = c1
        self.layer1 = self._make_layer(c1, layers[0], 1)
        self.layer2 = self._make_layer(c2, layers[1], 2)
        self.layer3 = self._make_layer(c3, layers[2], 2)
        self.layer4 = self._make_layer(c4, layers[3], 2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


class ResNetController(BaseController):
    """ResNet-based controller."""
    
    def __init__(self, num_classes: int, config: TrainingConfig, exp_dir: Path,
                 logger: Optional[logging.Logger] = None, pretrained: bool = False):
        self.pretrained = pretrained
        
        if pretrained:
            model_name = "resnet_pretrained"
            display_name = "ResNet (pretrained)"
        else:
            model_name = "resnet_random"
            display_name = "ResNet (random)"
        
        super().__init__(
            model_name=model_name,
            display_name=display_name,
            num_classes=num_classes,
            config=config,
            exp_dir=exp_dir,
            logger=logger,
        )
    
    def build_model(self) -> nn.Module:
        if self.pretrained:
            from torchvision.models import resnet18, ResNet18_Weights
            try:
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.logger.info("Loaded pretrained ResNet18 (ImageNet)")
            except Exception:
                model = resnet18(pretrained=True)
                self.logger.info("Loaded pretrained ResNet18 (legacy)")
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            model = SmallResNet(num_classes=self.num_classes)
            self.logger.info("Using SmallResNet (random init)")
        
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"ResNet parameters: {total_params:,}")
        return model
    
    def get_transforms(self, training: bool = False):
        if self.pretrained:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        if not training or not self.config.augment:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        min_scale = max(0.7, 1.0 - self.config.aug_scale)
        max_scale = 1.0 + self.config.aug_scale
        
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomAffine(
                degrees=self.config.aug_rotate_deg,
                translate=(self.config.aug_translate, self.config.aug_translate),
                scale=(min_scale, max_scale),
                fill=255,
            ),
            transforms.ColorJitter(brightness=self.config.aug_brightness, contrast=self.config.aug_contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# =============================================================================
# ViT Model
# =============================================================================
class ViTController(BaseController):
    """Vision Transformer controller."""
    
    def __init__(self, num_classes: int, config: TrainingConfig, exp_dir: Path,
                 logger: Optional[logging.Logger] = None, pretrained: bool = True):
        self.pretrained = pretrained
        
        if pretrained:
            model_name = "vit_pretrained"
            display_name = "ViT (pretrained)"
        else:
            model_name = "vit_random"
            display_name = "ViT (random)"
        
        super().__init__(
            model_name=model_name,
            display_name=display_name,
            num_classes=num_classes,
            config=config,
            exp_dir=exp_dir,
            logger=logger,
        )
        
        if not pretrained:
            self.logger.warning("Training ViT from scratch not recommended for small datasets.")
    
    def build_model(self) -> nn.Module:
        from torchvision.models import vit_b_16
        
        weights = None
        if self.pretrained:
            try:
                from torchvision.models import ViT_B_16_Weights
                weights = ViT_B_16_Weights.DEFAULT
                self.logger.info("Loaded pretrained ViT-B/16 (ImageNet)")
            except ImportError:
                self.logger.warning("Could not load ViT weights.")
        else:
            self.logger.info("ViT-B/16 with random initialization")
        
        model = vit_b_16(weights=weights)
        
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        else:
            raise RuntimeError("Unexpected ViT head structure")
        
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"ViT parameters: {total_params:,}")
        return model
    
    def get_transforms(self, training: bool = False):
        if self.pretrained:
            try:
                from torchvision.models import ViT_B_16_Weights
                weights = ViT_B_16_Weights.DEFAULT
                mean = tuple(float(x) for x in weights.meta["mean"])
                std = tuple(float(x) for x in weights.meta["std"])
            except Exception:
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        
        if not training or not self.config.augment:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        min_scale = max(0.7, 1.0 - self.config.aug_scale)
        max_scale = 1.0 + self.config.aug_scale
        
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomAffine(
                degrees=self.config.aug_rotate_deg,
                translate=(self.config.aug_translate, self.config.aug_translate),
                scale=(min_scale, max_scale),
                fill=255,
            ),
            transforms.ColorJitter(brightness=self.config.aug_brightness, contrast=self.config.aug_contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# =============================================================================
# VLA Model (Qwen3-VL-2B with Unsloth)
# =============================================================================
def _require_vlm_deps():
    try:
        import unsloth
        import trl
        import transformers
    except ImportError as e:
        raise ImportError(
            "VLA mode requires: pip install unsloth trl transformers accelerate bitsandbytes"
        ) from e


def _vlm_instruction(action_dim: int) -> str:
    return (
        f"You control a planar robot arm. Output the best expert action as a single integer in [0, {action_dim-1}]. "
        "Action encoding: idx = dof_index*3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree. "
        "Answer with ONLY the integer."
    )


def _parse_action_int(text: str, action_dim: int) -> Optional[int]:
    if text is None:
        return None
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        val = int(matches[-1])
        return val if 0 <= val < action_dim else None
    except Exception:
        return None


class VLAController(BaseController):
    """Vision-Language-Action controller using Qwen3-VL-2B with fresh LoRA adapters."""
    
    def __init__(self, num_classes: int, config: TrainingConfig, exp_dir: Path,
                 logger: Optional[logging.Logger] = None,
                 vla_config: Optional[VLAConfig] = None):
        self.vla_config = vla_config or VLAConfig(seed=config.seed)
        
        super().__init__(
            model_name="vla",
            display_name="VLA (Qwen3-VL-2B)",
            num_classes=num_classes,
            config=config,
            exp_dir=exp_dir,
            logger=logger,
        )
        
        self._model = None
        self._tokenizer = None
        self.action_dim = num_classes
    
    def build_model(self) -> nn.Module:
        return None  # VLA uses different loading
    
    def get_transforms(self, training: bool = False):
        return None  # VLA handles its own image processing
    
    def _make_conversation_records(self, data_dir: Path, pairs: List[Tuple[str, int]]) -> List[Dict]:
        instruction = _vlm_instruction(self.action_dim)
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
                        {"type": "text", "text": str(int(class_idx))},
                    ]},
                ]
            })
        return records
    
    def train_vla(self, data_dir: Path, train_pairs: List[Tuple[str, int]],
                  test_pairs: List[Tuple[str, int]],
                  test_subsample_pairs: Optional[List[Tuple[str, int]]] = None) -> TrainingHistory:
        _require_vlm_deps()
        
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
        
        set_global_seed(self.vla_config.seed)
        
        self.logger.info("=== VLA (Qwen3-VL-2B + fresh LoRA) Training ===")
        self.logger.info(f"Base model: {self.vla_config.model_name}")
        self.logger.info(f"Train: {len(train_pairs)} | Test: {len(test_pairs)}")
        
        train_records = self._make_conversation_records(data_dir, train_pairs)
        test_records = self._make_conversation_records(data_dir, test_pairs)
        test_subsample_records = self._make_conversation_records(
            data_dir, test_subsample_pairs or test_pairs[:min(200, len(test_pairs))]
        )
        
        model, tokenizer = FastVisionModel.from_pretrained(
            self.vla_config.model_name,
            load_in_4bit=self.vla_config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=self.vla_config.finetune_vision_layers,
            finetune_language_layers=self.vla_config.finetune_language_layers,
            finetune_attention_modules=self.vla_config.finetune_attention_modules,
            finetune_mlp_modules=self.vla_config.finetune_mlp_modules,
            r=self.vla_config.r,
            lora_alpha=self.vla_config.lora_alpha,
            lora_dropout=self.vla_config.lora_dropout,
            bias="none",
            random_state=self.vla_config.seed,
        )
        
        FastVisionModel.for_training(model)
        
        adapter_dir = self.model_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        with open(adapter_dir / "vla_config.json", "w") as f:
            json.dump(asdict(self.vla_config), f, indent=2)
        
        sft_args = SFTConfig(
            per_device_train_batch_size=self.vla_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.vla_config.gradient_accumulation_steps,
            warmup_steps=self.vla_config.warmup_steps,
            num_train_epochs=self.vla_config.num_train_epochs,
            learning_rate=self.vla_config.learning_rate,
            logging_steps=self.vla_config.logging_steps,
            max_steps=self.vla_config.max_steps if self.vla_config.max_steps > 0 else -1,
            optim=self.vla_config.optim,
            weight_decay=self.vla_config.weight_decay,
            lr_scheduler_type=self.vla_config.lr_scheduler_type,
            seed=self.vla_config.seed,
            output_dir=str(adapter_dir / "trainer_outputs"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=self.vla_config.max_seq_length,
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_records,
            args=sft_args,
        )
        
        training_start = time.time()
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu.name}, Memory: {gpu.total_memory / 1e9:.1f} GB")
        
        trainer_stats = trainer.train()
        self.total_training_time = time.time() - training_start
        
        try:
            metrics = dict(trainer_stats.metrics)
        except Exception:
            metrics = {}
        
        with open(adapter_dir / "trainer_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        self.logger.info(f"Saved adapter to: {adapter_dir}")
        
        FastVisionModel.for_inference(model)
        test_acc = self._evaluate_vla(model, tokenizer, test_subsample_records, len(test_subsample_records))
        
        self.history.train_loss.append(metrics.get("train_loss", 0.0))
        self.history.train_acc.append(0.0)
        self.history.test_loss.append(0.0)
        self.history.test_acc.append(test_acc)
        self.history.epoch_times.append(self.total_training_time)
        
        history_path = self.model_dir / "vla_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)
        
        self.logger.info(f"Test accuracy: {test_acc*100:.2f}%")
        self.logger.info(f"Total training time: {self.total_training_time:.1f}s")
        
        self._model = model
        self._tokenizer = tokenizer
        
        return self.history
    
    def test_vla(self, data_dir: Path, test_pairs: List[Tuple[str, int]], limit: int = 0) -> float:
        _require_vlm_deps()
        from unsloth import FastVisionModel
        
        adapter_dir = self.model_dir / "adapter"
        
        if self._model is None or self._tokenizer is None:
            if not adapter_dir.exists():
                raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
            model, tokenizer = FastVisionModel.from_pretrained(str(adapter_dir), load_in_4bit=True)
            FastVisionModel.for_inference(model)
            self._model = model
            self._tokenizer = tokenizer
        
        test_records = self._make_conversation_records(data_dir, test_pairs)
        return self._evaluate_vla(self._model, self._tokenizer, test_records, limit or len(test_records))
    
    def _evaluate_vla(self, model, tokenizer, test_records: List[Dict], limit: int) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        n = min(len(test_records), limit)
        correct = total = 0
        
        for i in range(n):
            sample = test_records[i]
            user_msg = sample["messages"][0]["content"]
            gt_text = sample["messages"][1]["content"][0]["text"]
            
            img_path = instr = None
            for c in user_msg:
                if c.get("type") == "image":
                    img_path = c.get("image")
                elif c.get("type") == "text":
                    instr = c.get("text", "")
            
            if img_path is None:
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            
            messages = [{"role": "user", "content": [{"type": "text", "text": instr}, {"type": "image"}]}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=self.vla_config.max_new_tokens, use_cache=True)
            
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            gt = _parse_action_int(gt_text, self.action_dim)
            pred = _parse_action_int(pred_text, self.action_dim)
            
            if gt is not None and pred is not None:
                correct += int(pred == gt)
            total += 1
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"[VLA eval] {i+1}/{n} | acc={100*correct/max(1,total):.2f}%")
        
        acc = correct / max(1, total)
        self.logger.info(f"[VLA eval] done | n={total} | acc={acc*100:.2f}%")
        return acc
    
    def measure_inference_time_vla(self, data_dir: Path, test_pairs: List[Tuple[str, int]], num_samples: int = 20) -> float:
        _require_vlm_deps()
        from unsloth import FastVisionModel
        
        adapter_dir = self.model_dir / "adapter"
        if self._model is None or self._tokenizer is None:
            if not adapter_dir.exists():
                return 0.0
            model, tokenizer = FastVisionModel.from_pretrained(str(adapter_dir), load_in_4bit=True)
            FastVisionModel.for_inference(model)
            self._model = model
            self._tokenizer = tokenizer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        
        test_records = self._make_conversation_records(data_dir, test_pairs[:num_samples])
        times = []
        
        for i, sample in enumerate(test_records):
            user_msg = sample["messages"][0]["content"]
            img_path = instr = None
            for c in user_msg:
                if c.get("type") == "image":
                    img_path = c.get("image")
                elif c.get("type") == "text":
                    instr = c.get("text", "")
            
            if img_path is None:
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            
            messages = [{"role": "user", "content": [{"type": "text", "text": instr}, {"type": "image"}]}]
            input_text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)
            
            if i == 0:  # Warmup
                with torch.no_grad():
                    _ = self._model.generate(**inputs, max_new_tokens=self.vla_config.max_new_tokens, use_cache=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = self._model.generate(**inputs, max_new_tokens=self.vla_config.max_new_tokens, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        return sum(times) / len(times) if times else 0.0
    
    def save(self, path: Optional[Path] = None) -> Path:
        adapter_dir = self.model_dir / "adapter"
        self.logger.info(f"VLA adapter at: {adapter_dir}")
        return adapter_dir
    
    def load(self, path: Optional[Path] = None) -> None:
        _require_vlm_deps()
        from unsloth import FastVisionModel
        
        adapter_dir = path or (self.model_dir / "adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
        
        model, tokenizer = FastVisionModel.from_pretrained(str(adapter_dir), load_in_4bit=True)
        FastVisionModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer
        self.logger.info(f"Loaded VLA adapter from: {adapter_dir}")
    
    def get_experiment_result_vla(self, data_dir: Path, test_pairs: List[Tuple[str, int]], 
                                   num_train_samples: int) -> ExperimentResult:
        test_acc = self.test_vla(data_dir, test_pairs)
        inference_time = self.measure_inference_time_vla(data_dir, test_pairs, 20)
        
        return ExperimentResult(
            model_name=self.display_name,
            final_train_acc=self.history.train_acc[-1] if self.history.train_acc else 0.0,
            final_test_acc=test_acc,
            total_training_time_seconds=self.total_training_time,
            inference_time_per_image_seconds=inference_time,
            num_epochs=self.vla_config.num_train_epochs,
            num_train_samples=num_train_samples,
            num_test_samples=len(test_pairs),
        )


# =============================================================================
# Data Loading
# =============================================================================
def load_samples_csv(data_dir: Path) -> Tuple[List[Tuple[str, int]], int]:
    csv_path = data_dir / "samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing samples.csv in {data_dir}")
    
    pairs: List[Tuple[str, int]] = []
    action_dim: Optional[int] = None
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row["image_filename"].strip()
            action_str = row["action"].strip()
            onehot = json.loads(action_str)
            
            if action_dim is None:
                action_dim = len(onehot)
            
            class_idx = int(max(range(len(onehot)), key=lambda i: onehot[i]))
            pairs.append((img_rel, class_idx))
    
    if action_dim is None:
        raise ValueError("samples.csv appears empty.")
    
    return pairs, action_dim


def split_data(pairs: List[Tuple[str, int]], train_frac: float = 0.8, quick_mode: bool = False):
    n = len(pairs)
    if quick_mode:
        n_quick = max(1, int(n * 0.01))
        return pairs[:n_quick], pairs[-n_quick:]
    else:
        n_train = int(n * train_frac)
        n_train = max(1, min(n - 1, n_train))
        return pairs[:n_train], pairs[n_train:]


# =============================================================================
# Experiment Runner
# =============================================================================
def run_classic_experiment(
    controller: BaseController,
    data_dir: Path,
    train_pairs: List[Tuple[str, int]],
    test_pairs: List[Tuple[str, int]],
    config: TrainingConfig,
    test_subsample_frac: float = 0.1,
) -> ExperimentResult:
    """Run experiment for classic models (CNN, ResNet, ViT)."""
    
    train_tfm = controller.get_transforms(training=True)
    test_tfm = controller.get_transforms(training=False)
    
    train_ds = RobotArmDataset(data_dir, train_pairs, train_tfm)
    test_ds = RobotArmDataset(data_dir, test_pairs, test_tfm)
    
    n_subsample = max(1, int(len(test_pairs) * test_subsample_frac))
    test_sub_ds = RobotArmDataset(data_dir, test_pairs[:n_subsample], test_tfm)
    
    train_loader = DataLoader(train_ds, config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    test_sub_loader = DataLoader(test_sub_ds, config.batch_size, shuffle=False, 
                                  num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    
    controller.train(train_loader, test_loader, test_sub_loader)
    return controller.get_experiment_result(test_loader, num_train_samples=len(train_pairs))


def run_all_experiments(data_dir: Path, exp_dir: Path, config: TrainingConfig,
                        vla_config: VLAConfig, quick_mode: bool = False, skip_vla: bool = False):
    """
    Run all experiments:
    - CNN (1 variant)
    - ResNet random init + pretrained (2 variants)
    - ViT random init + pretrained (2 variants)
    - VLA (1 variant)
    
    Total: 6 experiments
    """
    
    print(f"Loading data from {data_dir}...")
    pairs, action_dim = load_samples_csv(data_dir)
    print(f"Total samples: {len(pairs)}, Action dim: {action_dim}")
    
    train_pairs, test_pairs = split_data(pairs, quick_mode=quick_mode)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    
    if quick_mode:
        print("\n*** QUICK MODE: Using 1% of data ***\n")
    
    results: List[ExperimentResult] = []
    test_subsample_frac = 1.0 if quick_mode else 0.1
    
    # =========================================================================
    # 1. CNN (from scratch only)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running: CNN")
    print(f"{'='*60}")
    try:
        controller = CNNController(action_dim, config, exp_dir)
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, 
                                        config, test_subsample_frac)
        results.append(result)
    except Exception as e:
        print(f"ERROR: CNN failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 2. ResNet (random init)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running: ResNet (random)")
    print(f"{'='*60}")
    try:
        controller = ResNetController(action_dim, config, exp_dir, pretrained=False)
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, 
                                        config, test_subsample_frac)
        results.append(result)
    except Exception as e:
        print(f"ERROR: ResNet (random) failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 3. ResNet (pretrained)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running: ResNet (pretrained)")
    print(f"{'='*60}")
    try:
        controller = ResNetController(action_dim, config, exp_dir, pretrained=True)
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, 
                                        config, test_subsample_frac)
        results.append(result)
    except Exception as e:
        print(f"ERROR: ResNet (pretrained) failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 4. ViT (random init)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running: ViT (random)")
    print(f"{'='*60}")
    try:
        controller = ViTController(action_dim, config, exp_dir, pretrained=False)
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, 
                                        config, test_subsample_frac)
        results.append(result)
    except Exception as e:
        print(f"ERROR: ViT (random) failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 5. ViT (pretrained)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Running: ViT (pretrained)")
    print(f"{'='*60}")
    try:
        controller = ViTController(action_dim, config, exp_dir, pretrained=True)
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, 
                                        config, test_subsample_frac)
        results.append(result)
    except Exception as e:
        print(f"ERROR: ViT (pretrained) failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 6. VLA (Qwen3-VL-2B + fresh LoRA)
    # =========================================================================
    if not skip_vla:
        print(f"\n{'='*60}")
        print("Running: VLA (Qwen3-VL-2B)")
        print(f"{'='*60}")
        try:
            controller = VLAController(action_dim, config, exp_dir, vla_config=vla_config)
            n_subsample = max(1, int(len(test_pairs) * test_subsample_frac))
            controller.train_vla(data_dir, train_pairs, test_pairs, test_pairs[:n_subsample])
            result = controller.get_experiment_result_vla(data_dir, test_pairs, 
                                                          num_train_samples=len(train_pairs))
            results.append(result)
        except Exception as e:
            print(f"ERROR: VLA failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_results_table(results: List[ExperimentResult], output_path: Path) -> str:
    """Generate final results table for paper."""
    
    header = "| Model                | Accuracy (%) | Training Time | Inference Time/Image |"
    sep =    "|----------------------|--------------|---------------|----------------------|"
    rows = [header, sep]
    
    for r in results:
        acc = f"{r.final_test_acc * 100:.2f}%"
        
        train_mins = r.total_training_time_seconds / 60
        if train_mins >= 60:
            train_str = f"{train_mins/60:.1f} h"
        elif train_mins >= 1:
            train_str = f"{train_mins:.1f} min"
        else:
            train_str = f"{r.total_training_time_seconds:.1f} s"
        
        inf_ms = r.inference_time_per_image_seconds * 1000
        if inf_ms >= 1000:
            inf_str = f"{inf_ms/1000:.2f} s"
        elif inf_ms >= 1:
            inf_str = f"{inf_ms:.2f} ms"
        else:
            inf_str = f"{inf_ms * 1000:.2f} µs"
        
        rows.append(f"| {r.model_name:20} | {acc:12} | {train_str:13} | {inf_str:20} |")
    
    table = "\n".join(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ROBOT ARM CONTROLLER EXPERIMENT RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(table + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            f.write(f"Model: {r.model_name}\n")
            f.write(f"  Final Train Accuracy: {r.final_train_acc * 100:.2f}%\n")
            f.write(f"  Final Test Accuracy: {r.final_test_acc * 100:.2f}%\n")
            f.write(f"  Training Time: {r.total_training_time_seconds:.1f}s ({r.total_training_time_seconds/60:.2f} min)\n")
            f.write(f"  Inference Time/Image: {r.inference_time_per_image_seconds * 1000:.4f} ms\n")
            f.write(f"  Epochs: {r.num_epochs}\n")
            f.write(f"  Train Samples: {r.num_train_samples}\n")
            f.write(f"  Test Samples: {r.num_test_samples}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("NOTES\n")
        f.write("=" * 80 + "\n\n")
        f.write("- CNN: Simple CNN trained from scratch (no pretrained weights available)\n")
        f.write("- ResNet (random): Custom SmallResNet with random initialization\n")
        f.write("- ResNet (pretrained): torchvision ResNet18 with ImageNet weights\n")
        f.write("- ViT (random): ViT-B/16 with random initialization (not recommended)\n")
        f.write("- ViT (pretrained): ViT-B/16 with ImageNet weights\n")
        f.write("- VLA: Qwen3-VL-2B with fresh LoRA adapters fine-tuned on robot arm data\n")
    
    print(f"\nResults saved to: {output_path}")
    return table


def generate_csv_results(results: List[ExperimentResult], output_path: Path):
    """Save results as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy", "train_accuracy",
                         "training_time_seconds", "inference_time_ms", "epochs",
                         "train_samples", "test_samples"])
        for r in results:
            writer.writerow([
                r.model_name,
                f"{r.final_test_acc:.4f}",
                f"{r.final_train_acc:.4f}",
                f"{r.total_training_time_seconds:.1f}",
                f"{r.inference_time_per_image_seconds * 1000:.4f}",
                r.num_epochs,
                r.num_train_samples,
                r.num_test_samples,
            ])
    print(f"CSV saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Robot Arm Controller Experiments")
    p.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    p.add_argument("--exp_dir", type=str, required=True, help="Experiment output directory")
    p.add_argument("--mode", type=str, required=True, choices=["train", "test", "do_all_experiments"])
    p.add_argument("--model", type=str, choices=["cnn", "resnet", "resnet_pretrained", "vit", "vit_pretrained", "vla"])
    p.add_argument("--quick", action="store_true", help="Use 1% of data for quick testing")
    p.add_argument("--skip_vla", action="store_true", help="Skip VLA experiment")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--vla_epochs", type=int, default=2)
    p.add_argument("--vla_batch_size", type=int, default=2)
    p.add_argument("--vla_lr", type=float, default=2e-4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    exp_dir = Path(args.exp_dir)
    
    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        return 1
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    
    config = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, image_size=args.image_size, num_workers=args.num_workers
    )
    
    vla_config = VLAConfig(
        model_name="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        num_train_epochs=args.vla_epochs,
        per_device_train_batch_size=args.vla_batch_size,
        learning_rate=args.vla_lr,
        seed=args.seed,
    )
    
    if args.quick:
        config.epochs = min(2, config.epochs)
        vla_config.num_train_epochs = 1
        vla_config.max_steps = 10
    
    print(f"Data: {data_dir}")
    print(f"Output: {exp_dir}")
    print(f"Mode: {args.mode}, Quick: {args.quick}, Seed: {args.seed}")
    
    if args.mode == "do_all_experiments":
        results = run_all_experiments(data_dir, exp_dir, config, vla_config, args.quick, args.skip_vla)
        
        if results:
            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            table = generate_results_table(results, exp_dir / "final_results.txt")
            print(table)
            generate_csv_results(results, exp_dir / "final_results.csv")
        
        return 0
    
    elif args.mode in ["train", "test"]:
        if args.model is None:
            print("Error: --model required for train/test mode")
            return 1
        
        pairs, action_dim = load_samples_csv(data_dir)
        train_pairs, test_pairs = split_data(pairs, quick_mode=args.quick)
        
        # Create appropriate controller
        if args.model == "cnn":
            controller = CNNController(action_dim, config, exp_dir)
        elif args.model == "resnet":
            controller = ResNetController(action_dim, config, exp_dir, pretrained=False)
        elif args.model == "resnet_pretrained":
            controller = ResNetController(action_dim, config, exp_dir, pretrained=True)
        elif args.model == "vit":
            controller = ViTController(action_dim, config, exp_dir, pretrained=False)
        elif args.model == "vit_pretrained":
            controller = ViTController(action_dim, config, exp_dir, pretrained=True)
        elif args.model == "vla":
            controller = VLAController(action_dim, config, exp_dir, vla_config=vla_config)
            n_subsample = max(1, int(len(test_pairs) * 0.1))
            controller.train_vla(data_dir, train_pairs, test_pairs, test_pairs[:n_subsample])
            result = controller.get_experiment_result_vla(data_dir, test_pairs, len(train_pairs))
            print(f"\nResult: {result}")
            return 0
        else:
            print(f"Unknown model: {args.model}")
            return 1
        
        result = run_classic_experiment(controller, data_dir, train_pairs, test_pairs, config)
        print(f"\nResult: {result}")
        return 0
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
