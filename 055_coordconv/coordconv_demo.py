import math
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from torch.utils.data import DataLoader, TensorDataset


IMAGE_SIZE = 100
CHANNELS_RGB = 3
DEFAULT_BATCH_SIZE = 64


@dataclass
class TrainingConfig:
    model_type: str
    train_images: int
    test_images: int
    epochs: int
    batch_size: int = DEFAULT_BATCH_SIZE
    image_size: int = IMAGE_SIZE
    learning_rate: float = 1e-3


class PlainCNN(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        pooled_size = image_size
        for _ in range(4):
            pooled_size //= 2

        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS_RGB, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * pooled_size * pooled_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.regressor(x)


class CoordConvCNN(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        in_channels = CHANNELS_RGB + 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * image_size * image_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def _coordinate_channels(self, batch_size: int, size: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, steps=size, device=device)
        x = torch.linspace(0.0, 1.0, steps=size, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, _ = x.shape
        coord = self._coordinate_channels(batch_size=b, size=h, device=x.device)
        x = torch.cat([x, coord], dim=1)
        x = self.features(x)
        return self.regressor(x)


def generate_dataset(num_images: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros((num_images, CHANNELS_RGB, image_size, image_size), dtype=torch.float32)
    targets = torch.zeros((num_images, 2), dtype=torch.float32)

    for i in range(num_images):
        x_pos = random.randint(0, image_size - 1)
        y_pos = random.randint(0, image_size - 1)

        images[i, :, y_pos, x_pos] = 1.0
        targets[i, 0] = x_pos / (image_size - 1)
        targets[i, 1] = y_pos / (image_size - 1)

    return images, targets


class TrainingWorker(QThread):
    epoch_log = Signal(str)
    trained_model_ready = Signal(object)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    def _build_model(self) -> nn.Module:
        if self.config.model_type == "Plain CNN":
            return PlainCNN(image_size=self.config.image_size)
        return CoordConvCNN(image_size=self.config.image_size)

    def run(self) -> None:
        try:
            random.seed(42)
            torch.manual_seed(42)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.epoch_log.emit(f"Using device: {device}")

            train_images, train_targets = generate_dataset(self.config.train_images, self.config.image_size)
            test_images, test_targets = generate_dataset(self.config.test_images, self.config.image_size)

            train_loader = DataLoader(
                TensorDataset(train_images, train_targets),
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            model = self._build_model().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

            test_images = test_images.to(device)
            test_targets = test_targets.to(device)

            self.epoch_log.emit(
                f"Model: {self.config.model_type} | Train images: {self.config.train_images} | "
                f"Test images: {self.config.test_images} | Epochs: {self.config.epochs}"
            )

            for epoch in range(1, self.config.epochs + 1):
                model.train()
                running_loss = 0.0

                for batch_images, batch_targets in train_loader:
                    batch_images = batch_images.to(device)
                    batch_targets = batch_targets.to(device)

                    optimizer.zero_grad()
                    prediction = model(batch_images)
                    loss = criterion(prediction, batch_targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * batch_images.size(0)

                mean_loss = running_loss / max(1, self.config.train_images)

                model.eval()
                with torch.no_grad():
                    pred_norm = model(test_images)
                    pred_norm = torch.clamp(pred_norm, 0.0, 1.0)

                    scale = float(self.config.image_size - 1)
                    pred_px = pred_norm * scale
                    gt_px = test_targets * scale
                    distances = torch.linalg.vector_norm(pred_px - gt_px, dim=1)

                    mean_distance = distances.mean().item()
                    median_distance = distances.median().item()
                    max_distance = distances.max().item()

                self.epoch_log.emit(
                    f"Epoch {epoch:03d}/{self.config.epochs:03d} | "
                    f"Train MSE: {mean_loss:.6f} | "
                    f"Mean pixel distance: {mean_distance:.3f} | "
                    f"Median: {median_distance:.3f} | "
                    f"Max: {max_distance:.3f}"
                )

            model_cpu = self._build_model().cpu()
            model_cpu.load_state_dict(model.cpu().state_dict())
            model_cpu.eval()
            self.trained_model_ready.emit(
                {
                    "model": model_cpu,
                    "model_type": self.config.model_type,
                    "image_size": self.config.image_size,
                }
            )

            self.done.emit("Training finished.")
        except Exception as exc:
            self.failed.emit(f"Error during training: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoordConv vs Plain CNN Defect Localization Demo")
        self.worker: TrainingWorker | None = None
        self.trained_model: nn.Module | None = None
        self.trained_model_type: str | None = None
        self.trained_image_size: int = IMAGE_SIZE

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        form = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Plain CNN", "CNN + CoordConv"])

        self.train_spin = QSpinBox()
        self.train_spin.setRange(100, 100000)
        self.train_spin.setValue(5000)

        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 50000)
        self.test_spin.setValue(1000)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(20)

        form.addRow("Model", self.model_combo)
        form.addRow("Number of training images", self.train_spin)
        form.addRow("Number of test images", self.test_spin)
        form.addRow("Number of training epochs", self.epochs_spin)

        root.addLayout(form)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start training")
        self.start_btn.clicked.connect(self.start_training)
        buttons.addWidget(self.start_btn)
        self.test_btn = QPushButton("Test model")
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self.test_model)
        buttons.addWidget(self.test_btn)
        buttons.addWidget(QLabel("After every epoch, evaluation distance is reported."))
        root.addLayout(buttons)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        root.addWidget(self.log_output)

        self.resize(980, 600)

    def start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Training in progress", "A training run is already active.")
            return

        config = TrainingConfig(
            model_type=self.model_combo.currentText(),
            train_images=self.train_spin.value(),
            test_images=self.test_spin.value(),
            epochs=self.epochs_spin.value(),
        )

        self.log_output.clear()
        self.log_output.appendPlainText("Preparing training run...")
        self.start_btn.setEnabled(False)

        self.worker = TrainingWorker(config)
        self.worker.epoch_log.connect(self.log_output.appendPlainText)
        self.worker.trained_model_ready.connect(self.capture_trained_model)
        self.worker.done.connect(self.training_done)
        self.worker.failed.connect(self.training_failed)
        self.worker.start()

    def capture_trained_model(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        model = payload.get("model")
        model_type = payload.get("model_type")
        image_size = payload.get("image_size")
        if isinstance(model, nn.Module) and isinstance(model_type, str) and isinstance(image_size, int):
            self.trained_model = model
            self.trained_model_type = model_type
            self.trained_image_size = image_size
            self.test_btn.setEnabled(True)

    def training_done(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        self.start_btn.setEnabled(True)

    def training_failed(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        self.start_btn.setEnabled(True)
        QMessageBox.critical(self, "Training failed", message)

    def test_model(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Training in progress", "Please wait for training to finish.")
            return

        if self.trained_model is None or self.trained_model_type is None:
            QMessageBox.information(self, "No trained model", "Please train a model first.")
            return

        num_tests = 100
        images, targets = generate_dataset(num_tests, self.trained_image_size)
        model = self.trained_model
        model.eval()

        self.log_output.appendPlainText(
            f"Running test for '{self.trained_model_type}' on {num_tests} fresh test images..."
        )

        with torch.no_grad():
            pred_norm = torch.clamp(model(images), 0.0, 1.0)

        scale = float(self.trained_image_size - 1)
        pred_px = pred_norm * scale
        gt_px = targets * scale

        for i in range(num_tests):
            gt_x = int(round(gt_px[i, 0].item()))
            gt_y = int(round(gt_px[i, 1].item()))
            pred_x = pred_px[i, 0].item()
            pred_y = pred_px[i, 1].item()
            distance = torch.linalg.vector_norm(pred_px[i] - gt_px[i], dim=0).item()

            self.log_output.appendPlainText(
                f"Test {i + 1:03d} | GT: ({gt_x:02d}, {gt_y:02d}) | "
                f"Pred: ({pred_x:6.2f}, {pred_y:6.2f}) | Distance: {distance:7.3f}"
            )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
