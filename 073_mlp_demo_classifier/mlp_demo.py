"""Run with: conda run -n env_teaching python mlp_demo.py"""
import sys
import time

import numpy as np
import torch
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from mlp_core import build_mlp, evaluate, make_optimizer, predict_classes, train_one_epoch

COLORS = ["#2563eb", "#e34850", "#159567", "#a855d8", "#e59418"]
DEFAULT_ARCHITECTURE = "linear 100\nrelu\ndropout 0.1\nlinear 50\nlogistic"


class FeatureCanvas(QWidget):
    """The same coordinate transform is used for points and decision regions."""
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.points = []
        self.labels = []
        self.background = None
        self.show_regions = True
        self.rng = np.random.default_rng(42)
        self.setMinimumSize(460, 460)

    def plot_rect(self):
        side = max(1, min(self.width() - 70, self.height() - 70))
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def mousePressEvent(self, event):
        rect = self.plot_rect()
        if self.window.timer.isActive() or not rect.contains(event.position()):
            return
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        x = 2 * (event.position().x() - rect.left()) / rect.width() - 1
        y = 1 - 2 * (event.position().y() - rect.top()) / rect.height()
        count = 1 if event.button() == Qt.MouseButton.LeftButton else 10
        points = np.array([[x, y]]) if count == 1 else self.rng.normal([x, y], 0.065, (10, 2))
        self.points.extend(np.clip(points, -1, 1).tolist())
        self.labels.extend([self.window.selected_class.currentIndex()] * count)
        self.window.clear_learning_curve()
        self.window.update_metrics()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f1f5f9"))
        rect = self.plot_rect()
        painter.fillRect(rect, QColor("white"))
        if self.background is not None and self.show_regions:
            painter.drawImage(rect, self.background)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for value in (-1, -0.5, 0, 0.5, 1):
            px = rect.left() + (value + 1) * rect.width() / 2
            py = rect.top() + (1 - value) * rect.height() / 2
            painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
            painter.setPen(QColor("#475569"))
            painter.drawText(QRectF(px - 20, rect.bottom() + 5, 40, 20), Qt.AlignmentFlag.AlignCenter, f"{value:g}")
            painter.drawText(QRectF(rect.left() - 36, py - 10, 30, 20), Qt.AlignmentFlag.AlignCenter, f"{value:g}")
        painter.drawText(QRectF(rect.center().x() - 45, rect.bottom() + 25, 90, 20), Qt.AlignmentFlag.AlignCenter, "Feature x")
        painter.drawText(QRectF(rect.left(), rect.top() - 27, 100, 20), "Feature y")
        painter.save()
        painter.setClipRect(rect)
        for (x, y), label in zip(self.points, self.labels):
            center = QPointF(rect.left() + (x + 1) * rect.width() / 2,
                             rect.top() + (1 - y) * rect.height() / 2)
            painter.setPen(QPen(QColor("white"), 1.5))
            painter.setBrush(QColor(COLORS[label]))
            painter.drawEllipse(center, 5, 5)
        painter.restore()


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D MLP Classifier — Draw, Train, Explore")
        self.resize(1550, 850)
        self.history = []
        self.model = None
        self.optimizer = None
        self.epochs_trained = 0
        self.remaining = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.training_step)
        self.canvas = FeatureCanvas(self)
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 5)
        self.num_classes.setValue(3)
        self.selected_class = QComboBox()
        self.architecture = QPlainTextEdit(DEFAULT_ARCHITECTURE)
        self.architecture.setMaximumHeight(230)
        self.architecture.setToolTip("Return: insert a new line. Space: insert a space.")
        self.architecture.setStyleSheet("font-family: monospace; font-size: 14px;")
        self.fixed_output = QLabel()
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 10)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setSingleStep(0.001)
        self.optimizer_choice = QComboBox()
        self.optimizer_choice.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(1)
        self.train_button = QPushButton("Train")
        self.train_button.setToolTip("Click or press Space once: train for the selected number of epochs (outside the architecture editor).")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.reset_button = QPushButton("Reset model")
        self.clear_button = QPushButton("Clear all points")
        self.undo_button = QPushButton("Remove last point")
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setMinimumHeight(65)
        self.status = QLabel("Draw points for at least two classes, then click Train.")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(55)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        self.regions = QCheckBox("Show decision regions")
        self.regions.setChecked(True)
        sidebar = QWidget()
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(440)
        layout = QVBoxLayout(sidebar)
        data_box = QGroupBox("1. Draw your data")
        data_form = QFormLayout(data_box)
        data_form.addRow("Number of classes", self.num_classes)
        data_form.addRow("Active class", self.selected_class)
        hint = QLabel("Left click: add one point\nRight click: add 10 nearby points")
        data_form.addRow(hint)
        data_form.addRow(self.undo_button)
        data_form.addRow(self.clear_button)
        layout.addWidget(data_box)
        model_box = QGroupBox("2. Define the MLP")
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(QLabel("Input: 2 features (x, y)\nHidden layers, one per line:"))
        model_layout.addWidget(self.architecture)
        model_layout.addWidget(self.fixed_output)
        syntax = QLabel("Syntax: linear N · relu · logistic · tanh · dropout P\nlogistic = sigmoid. Blank lines and # comments allowed.")
        syntax.setWordWrap(True)
        model_layout.addWidget(syntax)
        model_layout.addWidget(self.model_info)
        model_layout.addWidget(self.reset_button)
        layout.addWidget(model_box)
        training_box = QGroupBox("3. Train and inspect")
        training_form = QFormLayout(training_box)
        training_form.addRow("Learning rate", self.learning_rate)
        training_form.addRow("Optimizer", self.optimizer_choice)
        training_form.addRow("Epochs per click", self.epochs)
        buttons = QHBoxLayout()
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.stop_button)
        training_form.addRow(buttons)
        training_form.addRow(self.regions)
        training_form.addRow(self.metrics)
        layout.addStretch()
        results = QWidget()
        results.setMinimumWidth(360)
        results_layout = QVBoxLayout(results)
        results_layout.addWidget(training_box)
        curve_box = QGroupBox("4. Learning curve")
        curve_layout = QVBoxLayout(curve_box)
        self.figure = Figure(figsize=(4, 4), layout="constrained")
        self.loss_axes, self.error_axes = self.figure.subplots(2, 1, sharex=True)
        self.curve_canvas = FigureCanvasQTAgg(self.figure)
        self.curve_canvas.setMinimumSize(330, 340)
        curve_layout.addWidget(self.curve_canvas)
        curve_hint = QLabel("Evaluated after each epoch, with dropout disabled.\nChanging data or resetting the model clears the curve.")
        curve_hint.setWordWrap(True)
        curve_layout.addWidget(curve_hint)
        results_layout.addWidget(curve_box, 1)
        results_layout.addWidget(self.status)
        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(460)
        splitter.addWidget(scroll)
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setWidget(results)
        result_scroll.setMinimumWidth(385)
        splitter.addWidget(result_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([760, 370, 420])
        self.setCentralWidget(splitter)
        self.num_classes.valueChanged.connect(self.classes_changed)
        self.architecture.textChanged.connect(self.invalidate_model)
        self.learning_rate.valueChanged.connect(self.change_learning_rate)
        self.optimizer_choice.currentTextChanged.connect(self.change_optimizer)
        self.train_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.reset_button.clicked.connect(self.reset_model)
        self.clear_button.clicked.connect(self.clear_points)
        self.undo_button.clicked.connect(self.undo_point)
        self.regions.toggled.connect(self.toggle_regions)
        # Grid rows run from y=+1 (screen top) to y=-1 (screen bottom).
        self.grid_size = 220
        axis = torch.linspace(-1, 1, self.grid_size)
        yy, xx = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        self.grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        self.classes_changed()
        self.reset_model()
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event):
        # Consume both press and release so focused buttons/checkboxes cannot react.
        # Preserve ordinary typing inside the architecture editor.
        if watched is self.architecture or (
                isinstance(watched, QWidget) and self.architecture.isAncestorOf(watched)):
            return super().eventFilter(watched, event)
        if (isinstance(watched, QWidget) and QWidget.window(watched) is self
                and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and event.key() == Qt.Key.Key_Space
                and event.modifiers() == Qt.KeyboardModifier.NoModifier):
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            if (event.type() == QEvent.Type.KeyPress
                    and not event.isAutoRepeat() and self.train_button.isEnabled()):
                for spin in (self.num_classes, self.learning_rate, self.epochs):
                    spin.interpretText()
                self.train_button.click()
            return True
        return super().eventFilter(watched, event)

    def clear_learning_curve(self):
        self.history.clear()
        self.draw_learning_curve()

    def record_learning_metrics(self):
        loss, accuracy = evaluate(self.model, self.features, self.targets)
        self.history.append((self.epochs_trained, loss, 100 * (1 - accuracy)))

    def draw_learning_curve(self):
        self.loss_axes.clear()
        self.error_axes.clear()
        if self.history:
            epochs, losses, errors = zip(*self.history)
            self.loss_axes.plot(epochs, losses, color=COLORS[0], marker=".", markersize=3)
            self.error_axes.plot(epochs, errors, color=COLORS[1], marker=".", markersize=3)
        self.loss_axes.set_ylabel("Cross-entropy loss")
        self.loss_axes.set_title("Training data")
        self.error_axes.set_ylabel("Classification error (%)")
        self.error_axes.set_xlabel("Completed epochs")
        self.error_axes.set_ylim(-2, 102)
        for axes in (self.loss_axes, self.error_axes):
            axes.grid(True, alpha=0.25)
        self.curve_canvas.draw_idle()

    def classes_changed(self):
        n = self.num_classes.value()
        active = self.selected_class.currentIndex()
        self.selected_class.clear()
        for i in range(n):
            self.selected_class.addItem(f"Class {i + 1}")
            self.selected_class.setItemData(i, QColor(COLORS[i]), Qt.ItemDataRole.ForegroundRole)
        self.selected_class.setCurrentIndex(max(0, min(active, n - 1)))
        # Keep removed classes' points by restoring the class count instead of deleting data.
        if self.canvas.labels and max(self.canvas.labels) >= n:
            self.num_classes.setValue(max(self.canvas.labels) + 1)
            self.status.setText("Clear points before removing a class that contains data.")
            return
        self.fixed_output.setText(f"Fixed output: linear {n} (automatically appended)")
        self.invalidate_model()

    def invalidate_model(self):
        self.model = None
        self.optimizer = None
        self.epochs_trained = 0
        self.clear_learning_curve()
        self.canvas.background = None
        self.model_info.setText("Model will be built on Train or Reset model.")
        self.status.setText("Architecture changed. Next training starts with new weights.")
        self.update_metrics()
        self.canvas.update()

    def reset_model(self):
        try:
            model = build_mlp(self.architecture.toPlainText(), self.num_classes.value())
        except ValueError as error:
            QMessageBox.warning(self, "Invalid architecture", str(error))
            return False
        self.model = model
        self.optimizer = make_optimizer(model, self.optimizer_choice.currentText(), self.learning_rate.value())
        self.epochs_trained = 0
        self.clear_learning_curve()
        count = sum(p.numel() for p in model.parameters())
        self.model_info.setText(f"2 inputs → {self.num_classes.value()} logits | {count:,} parameters")
        self.refresh_regions()
        self.update_metrics()
        self.status.setText("New random weights. Draw data or train the model.")
        return True

    def change_learning_rate(self):
        if self.optimizer:
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate.value()

    def change_optimizer(self):
        if self.model is not None:
            self.optimizer = make_optimizer(self.model, self.optimizer_choice.currentText(), self.learning_rate.value())
            self.status.setText("Optimizer state reset; model weights retained.")

    def tensors(self):
        return (torch.tensor(self.canvas.points, dtype=torch.float32),
                torch.tensor(self.canvas.labels, dtype=torch.long))

    def start_training(self):
        if self.timer.isActive():
            return
        if len(set(self.canvas.labels)) < 2:
            QMessageBox.information(self, "Add training data", "Draw points for at least two classes.")
            return
        if self.model is None and not self.reset_model():
            return
        self.features, self.targets = self.tensors()
        if not self.history:
            self.record_learning_metrics()
        self.remaining = self.epochs.value()
        self.last_refresh = 0.0
        self.set_training_controls(True)
        self.status.setText(f"Training… {self.remaining} epochs remaining.")
        self.timer.start(0)

    def set_training_controls(self, training):
        for widget in (self.num_classes, self.selected_class, self.architecture,
                       self.learning_rate, self.optimizer_choice, self.epochs,
                       self.reset_button, self.clear_button, self.undo_button, self.train_button):
            widget.setEnabled(not training)
        self.stop_button.setEnabled(training)

    def training_step(self):
        # One epoch per Qt event: the UI can repaint and handle Stop between epochs.
        try:
            train_one_epoch(self.model, self.optimizer, self.features, self.targets)
            self.epochs_trained += 1
            self.record_learning_metrics()
            self.remaining -= 1
            if self.remaining == 0:
                self.stop_training()
            elif time.monotonic() - self.last_refresh >= 0.15:
                self.refresh_regions()
                self.update_metrics()
                self.draw_learning_curve()
                self.status.setText(f"Training… {self.remaining} epochs remaining.")
                self.last_refresh = time.monotonic()
        except (ValueError, RuntimeError) as error:
            self.timer.stop()
            self.set_training_controls(False)
            self.status.setText(f"Training stopped: {error}")

    def stop_training(self):
        self.timer.stop()
        self.set_training_controls(False)
        self.refresh_regions()
        self.update_metrics()
        self.draw_learning_curve()
        self.status.setText("Training stopped. Click Train to continue." if self.remaining else
                            "Training complete. Click Train to continue from these weights.")

    def refresh_regions(self):
        # Chunked inference also keeps memory bounded for wider hidden layers.
        prediction = torch.cat([predict_classes(self.model, batch)
                                for batch in self.grid.split(2048)]).numpy()
        palette = np.array([[QColor(c).red(), QColor(c).green(), QColor(c).blue()] for c in COLORS])
        rgb = (0.25 * palette[prediction] + 0.75 * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb.reshape(self.grid_size, self.grid_size, 3))
        self.canvas.background = QImage(rgb.data, self.grid_size, self.grid_size,
                                       3 * self.grid_size, QImage.Format.Format_RGB888).copy()
        self.canvas.update()

    def update_metrics(self):
        counts = [self.canvas.labels.count(i) for i in range(self.num_classes.value())]
        text = "Points: " + " | ".join(f"C{i+1}: {n}" for i, n in enumerate(counts))
        text += f"\nCompleted epochs: {self.epochs_trained}"
        if self.model is not None and self.canvas.points:
            loss, accuracy = evaluate(self.model, *self.tensors())
            text += f"\nTraining loss: {loss:.4f} | Accuracy: {accuracy:.1%}"
        self.metrics.setText(text)

    def clear_points(self):
        self.canvas.points.clear()
        self.canvas.labels.clear()
        self.clear_learning_curve()
        self.update_metrics()
        self.canvas.update()
        self.status.setText("Points cleared; model weights retained. Reset model for a fresh start.")

    def undo_point(self):
        if self.canvas.points:
            self.canvas.points.pop()
            self.canvas.labels.pop()
            self.clear_learning_curve()
            self.update_metrics()
            self.canvas.update()

    def toggle_regions(self, checked):
        self.canvas.show_regions = checked
        self.canvas.update()


def main():
    # Small teaching datasets are faster without a large CPU thread pool.
    torch.set_num_threads(2)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
