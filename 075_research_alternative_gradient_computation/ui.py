"""PySide6 widgets, plotting and worker-thread adapter for the headless core."""
import json
from pathlib import Path
from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from data import DataConfig, TASKS
from experiment import Experiment, TrainConfig
from gradients import ALGORITHMS, GradientConfig, TrainingCancelled, compare_gradients
from network import DEFAULT_ARCHITECTURE


class TrainingWorker(QThread):
    snapshot_ready = Signal(object)
    diagnostic_ready = Signal(object)
    progress = Signal(int)
    message = Signal(str)
    failed = Signal(str)

    def __init__(self, experiment, config, epochs, diagnostic=False, parent=None):
        super().__init__(parent)
        self.experiment, self.config = experiment, config
        self.epochs, self.diagnostic = epochs, diagnostic
        self.stop_event = Event()

    def run(self):
        try:
            if self.diagnostic:
                x, y = self.experiment.train_data
                self.diagnostic_ready.emit(compare_gradients(
                    self.experiment.model, x[:self.config.batch_size], y[:self.config.batch_size],
                    self.experiment.data_config.task, self.config.gradient.epsilon,
                    self.config.gradient.directions, seed=self.experiment.data_config.seed + 3,
                    should_stop=self.stop_event.is_set))
                self.message.emit("Gradientenvergleich beendet; Gewichte unverändert.")
                return
            last_refresh = perf_counter()
            for epoch in range(self.epochs):
                self.experiment.train_epoch(self.config, self.stop_event.is_set)
                self.progress.emit(epoch + 1)
                if perf_counter() - last_refresh >= 0.25:
                    self.snapshot_ready.emit(self.experiment.snapshot())
                    last_refresh = perf_counter()
            self.message.emit(f"Fertig: {self.epochs} weitere Epochen gelernt.")
        except TrainingCancelled:
            self.message.emit("Gestoppt. Bereits ausgeführte Updates bleiben erhalten; nur volle Epochen werden gezählt.")
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            if not self.diagnostic:
                try:
                    self.experiment.record_metrics()
                    self.snapshot_ready.emit(self.experiment.snapshot())
                except Exception as error:
                    self.failed.emit(str(error))


def integer(value, minimum, maximum):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def decimal(value, minimum, maximum, decimals=5, step=0.001):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(minimum, maximum)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


class ExperimentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EarthQuakeWeights — Gradientenlabor")
        self.resize(1450, 950)
        self.worker = None
        self.experiment = None
        self.snapshot_data = None
        self.remembered = []
        self.closing = False
        self.locked_controls = []
        self.build_controls()
        self.build_plots()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(350)
        splitter.addWidget(scroll)
        splitter.addWidget(self.plots)
        splitter.setSizes([375, 1075])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.apply_experiment()

    def group(self, title):
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.control_layout.addWidget(box)
        return form

    def build_controls(self):
        self.controls = QWidget()
        self.control_layout = QVBoxLayout(self.controls)
        intro = QLabel("<b>EarthQuakeWeights</b><br>Zufällige Gewichtsänderungen versus AutoDiff")
        intro.setWordWrap(True)
        self.control_layout.addWidget(intro)
        form = self.group("1 · Daten & Netz")
        self.task = QComboBox()
        self.task.addItems(TASKS)
        self.train_samples = integer(512, 2, 10000)
        self.test_samples = integer(256, 2, 10000)
        self.seed = integer(42, 0, 2**31 - 1)
        self.architecture = QPlainTextEdit(DEFAULT_ARCHITECTURE)
        self.architecture.setMaximumHeight(145)
        self.architecture.setStyleSheet("font-family: monospace;")
        self.architecture.setToolTip("linear N = N Neuronen. Aktivierungen: relu, tanh, sigmoid, gelu, silu. # für Kommentare.")
        for label, widget in (("Aufgabe", self.task), ("Trainingspunkte", self.train_samples),
                              ("Testpunkte", self.test_samples), ("Seed", self.seed),
                              ("Verborgene Schichten", self.architecture)):
            form.addRow(label, widget)
            self.locked_controls.append(widget)
        hint = QLabel("Input: (x, y). Output automatisch: 1 Logit / 1 Regressionswert.\nÄnderungen gelten bei Anwenden oder Lernen.")
        hint.setWordWrap(True)
        form.addRow(hint)
        self.apply_button = QPushButton("Daten & Netz anwenden")
        self.apply_button.clicked.connect(lambda: self.apply_experiment(force=True))
        self.reset_button = QPushButton("Netz zurücksetzen (gleicher Seed)")
        self.reset_button.clicked.connect(self.reset_model)
        form.addRow(self.apply_button)
        form.addRow(self.reset_button)

        form = self.group("2 · Lernen")
        self.algorithm = QComboBox()
        self.algorithm.addItems(ALGORITHMS)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "SGD"])
        self.learning_rate = decimal(0.01, 0.000001, 2, 6)
        self.batch_size = integer(64, 1, 10000)
        self.epochs = integer(50, 1, 100000)
        self.epsilon = decimal(0.01, 0.000001, 1, 6)
        self.directions = integer(32, 1, 4096)
        self.clip_norm = decimal(5, 0, 1000, 2, 0.5)
        self.epsilon.setToolTip("Jedes Gewicht und jeder Bias wird um +ε oder −ε gestört, ohne Normierung nach Parameterzahl.")
        self.directions.setToolTip("K unabhängige Richtungen mitteln. Mehr Richtungen reduzieren Rauschen, benötigen aber mehr Vorwärtsläufe.")
        self.clip_norm.setToolTip("Maximale Norm des gesamten Gradienten für alle Methoden. 0 = deaktiviert.")
        for label, widget in (("Gradienten", self.algorithm), ("Optimizer", self.optimizer),
                              ("Lernrate", self.learning_rate), ("Mini-Batch", self.batch_size),
                              ("Weitere Epochen", self.epochs), ("Störung ε", self.epsilon),
                              ("Zufallsrichtungen K", self.directions), ("Gradient-Clipping", self.clip_norm)):
            form.addRow(label, widget)
            self.locked_controls.append(widget)
        self.cost_label = QLabel()
        self.cost_label.setWordWrap(True)
        form.addRow(self.cost_label)
        self.algorithm.currentTextChanged.connect(self.update_cost)
        self.directions.valueChanged.connect(self.update_cost)
        self.train_button = QPushButton("Lernen")
        self.train_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.train_button.clicked.connect(lambda: self.start_worker(False))
        self.stop_button = QPushButton("Stopp")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        buttons = QHBoxLayout()
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.stop_button)
        form.addRow(buttons)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.epochs.value())
        self.progress_bar.setValue(0)
        form.addRow(self.progress_bar)

        form = self.group("3 · Vergleichen")
        self.compare_button = QPushButton("Gradienten mit AutoDiff vergleichen")
        self.compare_button.clicked.connect(lambda: self.start_worker(True))
        self.remember_button = QPushButton("Aktuelle Testkurve merken")
        self.remember_button.clicked.connect(self.remember_curve)
        self.clear_button = QPushButton("Gemerkte Kurven löschen")
        self.clear_button.clicked.connect(self.clear_curves)
        self.export_button = QPushButton("Messwerte als JSON exportieren")
        self.export_button.clicked.connect(self.export_results)
        for button in (self.compare_button, self.remember_button, self.clear_button, self.export_button):
            form.addRow(button)
        self.diagnostic_label = QLabel("Vergleich am ersten Trainingsbatch, ohne Gewichtsupdate.\nFür faire Läufe: Kurve merken, Netz zurücksetzen, Methode wechseln.\nGleicher Seed → gleiche Daten, Startgewichte und Batch-Reihenfolge.")
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.diagnostic_label)
        self.locked_controls.extend([self.apply_button, self.reset_button, self.train_button,
                                     self.compare_button, self.remember_button,
                                     self.clear_button, self.export_button])
        self.control_layout.addStretch()
        self.update_cost()

    def build_plots(self):
        self.plots = QWidget()
        layout = QVBoxLayout(self.plots)
        choices = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Netzvorhersage", "Wahre Funktion", "Absoluter Fehler"])
        self.show_train = QCheckBox("Trainingspunkte · Kreise")
        self.show_train.setChecked(True)
        self.show_test = QCheckBox("Testpunkte · Dreiecke")
        self.show_test.setChecked(True)
        choices.addWidget(self.view_mode)
        choices.addWidget(self.show_train)
        choices.addWidget(self.show_test)
        choices.addStretch()
        layout.addLayout(choices)
        self.figure = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        grid = self.figure.add_gridspec(1, 2, width_ratios=[1, 0.035])
        self.ax = self.figure.add_subplot(grid[0, 0])
        self.color_ax = self.figure.add_subplot(grid[0, 1])
        self.colorbar = None
        self.image = None
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 4)
        self.probe_label = QLabel("Klick in die Fläche: (x, y), Ziel und Netzausgabe am nächsten Rasterpunkt.")
        layout.addWidget(self.probe_label)
        self.metric_label = QLabel()
        self.metric_label.setWordWrap(True)
        self.metric_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.metric_label)
        self.curve_figure = Figure(figsize=(9, 2), layout="constrained")
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_canvas.setMinimumHeight(180)
        self.curve_ax = self.curve_figure.add_subplot(111)
        layout.addWidget(self.curve_canvas, 1)
        self.status_label = QLabel("Bereit.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.view_mode.currentIndexChanged.connect(self.draw_feature_space)
        self.show_train.toggled.connect(self.draw_feature_space)
        self.show_test.toggled.connect(self.draw_feature_space)
        self.canvas.mpl_connect("button_press_event", self.probe)

    def update_cost(self):
        algorithm = self.algorithm.currentText()
        k = self.directions.value()
        if algorithm == "AutoDiff":
            text = "Pro Batch: 1 Vorwärtslauf + Backpropagation. ε und K gelten nur für Störungsverfahren / Diagnose."
        elif algorithm == "EarthQuakeWeights":
            text = f"Pro Batch: {2*k+1} Vorwärtsläufe; Schätzung über Ausgabedifferenzen, ohne Backpropagation."
        else:
            text = f"Pro Batch: {2*k} Vorwärtsläufe; Schätzung über Loss-Differenzen, ohne Backpropagation."
        self.cost_label.setText(text)

    def data_config(self):
        return DataConfig(self.task.currentText(), self.train_samples.value(),
                          self.test_samples.value(), self.seed.value())

    def training_config(self):
        return TrainConfig(GradientConfig(self.algorithm.currentText(), self.epsilon.value(), self.directions.value()),
                           self.optimizer.currentText(), self.learning_rate.value(),
                           self.batch_size.value(), self.clip_norm.value())

    def apply_experiment(self, force=False):
        if self.worker is not None:
            return False
        try:
            data_config = self.data_config()
            architecture = self.architecture.toPlainText()
            if force or self.experiment is None or data_config != self.experiment.data_config or architecture != self.experiment.architecture:
                experiment = Experiment(data_config, architecture)
                self.experiment = experiment
                self.receive_snapshot(experiment.snapshot())
                self.status_label.setText("Daten und Netz initialisiert. Gleicher Seed erzeugt dasselbe Experiment.")
                self.probe_label.setText("Klick in die Fläche zeigt die Werte am nächsten Rasterpunkt.")
            return True
        except Exception as error:
            QMessageBox.warning(self, "Ungültige Eingabe", str(error))
            return False

    def reset_model(self):
        # Data are deterministic, so applying the same config also retains them.
        self.apply_experiment(force=True)

    def start_worker(self, diagnostic):
        if not self.apply_experiment():
            return
        config = self.training_config()
        self.worker = TrainingWorker(self.experiment, config, self.epochs.value(), diagnostic, self)
        self.worker.snapshot_ready.connect(self.receive_snapshot)
        self.worker.diagnostic_ready.connect(self.show_diagnostic)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.message.connect(self.status_label.setText)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        for control in self.locked_controls:
            control.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setRange(0, 0 if diagnostic else self.epochs.value())
        self.progress_bar.setValue(0)
        self.status_label.setText("Gradientenvergleich läuft …" if diagnostic else "Training läuft … Weitere Klicks auf Lernen setzen das Netz fort.")
        self.worker.start()

    def stop_training(self):
        if self.worker is not None:
            self.worker.stop_event.set()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stoppe nach der aktuellen Störungsmessung / dem aktuellen Batch …")

    def worker_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        for control in self.locked_controls:
            control.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
        if self.closing:
            self.close()

    def show_error(self, message):
        self.status_label.setText(f"Fehler: {message}")
        if not self.closing:
            QMessageBox.warning(self, "Experiment fehlgeschlagen", message)

    def receive_snapshot(self, snapshot):
        # Worker emits only detached copies. The GUI never reads the live model.
        self.snapshot_data = snapshot
        row = snapshot["history"][-1]
        classification = self.experiment.data_config.task == TASKS[0]
        metric = (f"Accuracy Train/Test: {row['train_metric']:.1%} / {row['test_metric']:.1%}" if classification
                  else f"RMSE Train/Test: {row['train_metric']:.4f} / {row['test_metric']:.4f}")
        self.metric_label.setText(
            f"{self.experiment.data_config.task} · {snapshot['parameters']:,} Parameter · {row['algorithm']}\n"
            f"{row['epoch']} volle Epochen · {row['steps']} Updates · Loss Train/Test: {row['train_loss']:.5f} / {row['test_loss']:.5f} · {metric}\n"
            f"Training: {row['seconds']:.2f} s · {row['forwards']:,} Trainings-Vorwärtsläufe (abgeschlossene Batches; ohne Evaluation)")
        self.draw_feature_space()
        self.draw_curves()

    def draw_feature_space(self, *_):
        if self.snapshot_data is None:
            return
        prediction = self.snapshot_data["prediction"]
        truth = self.experiment.truth
        mode = self.view_mode.currentIndex()
        classification = self.experiment.data_config.task == TASKS[0]
        values = (prediction, truth, np.abs(prediction - truth))[mode]
        data_norm = Normalize(0, 1) if classification else Normalize(min(-1.0, float(truth.min())), max(1.0, float(truth.max())))
        cmap = "coolwarm" if classification else "viridis"
        norm = data_norm
        title = "P(Klasse 1)" if classification else "f̂(x, y)"
        if mode == 1:
            title = "Wahre Klasse" if classification else "f(x, y)"
        elif mode == 2:
            cmap = "magma"
            norm = Normalize(0, 1 if classification else max(0.1, float(values.max())))
            title = "|P(Klasse 1) − Ziel|" if classification else "|f̂(x, y) − f(x, y)|"
        self.color_ax.clear()
        self.ax.clear()
        # Pixel centers coincide with the evaluation grid, including its endpoints.
        half_pixel = 1 / (self.experiment.grid_size - 1)
        self.image = self.ax.imshow(values, origin="lower", extent=(-1-half_pixel, 1+half_pixel, -1-half_pixel, 1+half_pixel),
                                   cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        self.colorbar = self.figure.colorbar(self.image, cax=self.color_ax)
        self.colorbar.set_label(title)
        for data, checkbox, marker, label in ((self.experiment.train_data, self.show_train, "o", "Training"),
                                               (self.experiment.test_data, self.show_test, "^", "Test")):
            if checkbox.isChecked():
                x, y = (tensor.numpy() for tensor in data)
                self.ax.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap="coolwarm" if classification else "viridis",
                                norm=data_norm, marker=marker, s=16 if marker == "o" else 25,
                                edgecolors="white" if marker == "o" else "black", linewidths=0.4,
                                alpha=0.85, label=label)
        if classification and mode == 0 and prediction.min() < 0.5 < prediction.max():
            axis = np.linspace(-1, 1, self.experiment.grid_size)
            self.ax.contour(axis, axis, prediction, levels=[0.5], colors="black", linewidths=1)
        if self.show_train.isChecked() or self.show_test.isChecked():
            self.ax.legend(loc="upper right", fontsize=8)
        self.ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Feature x", ylabel="Feature y",
                    title=f"{self.view_mode.currentText()} · Punkte zeigen Zielwerte")
        self.canvas.draw_idle()

    def draw_curves(self):
        ax = self.curve_ax
        ax.clear()
        for saved in self.remembered:
            history = saved["history"]
            ax.plot([r["steps"] for r in history], [r["test_loss"] for r in history],
                    linestyle="--", alpha=0.65, label=saved["label"])
        history = self.snapshot_data["history"]
        for key, label, color in (("train_loss", "Training", "#2563eb"), ("test_loss", "Test", "#d97706")):
            ax.plot([r["steps"] for r in history], [r[key] for r in history], label=label, color=color,
                    marker="o" if len(history) == 1 else None)
        ax.set(xlabel="Gewichtsupdates", ylabel="Loss", ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
        self.curve_canvas.draw_idle()

    def probe(self, event):
        if event.inaxes != self.ax or event.xdata is None or self.snapshot_data is None:
            return
        n = self.experiment.grid_size
        ix, iy = np.clip(np.rint((np.array([event.xdata, event.ydata]) + 1) * (n-1) / 2), 0, n-1).astype(int)
        x, y = -1 + 2 * ix / (n-1), -1 + 2 * iy / (n-1)
        value = self.snapshot_data["prediction"][iy, ix]
        target = self.experiment.truth[iy, ix]
        self.probe_label.setText(f"Rasterpunkt ({x:.3f}, {y:.3f}) · Ziel: {target:.4f} · Netzausgabe: {value:.4f} · |Fehler|: {abs(target-value):.4f}")

    def show_diagnostic(self, rows):
        lines = [f"Erster Trainingsbatch · ε={self.epsilon.value():g} · K={self.directions.value()}"]
        for row in rows:
            lines.append(f"{row['algorithm']}: Kosinus={row['cosine']:.3f}, rel. Fehler={row['relative_error']:.3f}, "
                         f"‖ĝ‖={row['estimate_norm']:.3g}")
        lines.append(f"‖g AutoDiff‖={rows[0]['reference_norm']:.3g}. Ideal: Kosinus 1, relativer Fehler 0. Vor Clipping; NaN bei Nullgradient.")
        self.diagnostic_label.setText("\n".join(lines))

    def remember_curve(self):
        row = self.snapshot_data["history"][-1]
        config = self.experiment.data_config
        self.remembered.append({"label": f"{row['algorithm']} · {config.task.split('-')[0]} · Seed {config.seed} · {len(self.remembered)+1}",
                                "history": [r.copy() for r in self.snapshot_data["history"]],
                                "experiment": self.experiment.export_data()})
        self.remembered = self.remembered[-4:]
        self.draw_curves()
        self.status_label.setText("Testkurve gemerkt (maximal 4). Für den nächsten Vergleich Netz zurücksetzen und Methode wechseln.")

    def clear_curves(self):
        self.remembered.clear()
        self.draw_curves()

    def export_results(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Messwerte exportieren", "earthquake_results.json", "JSON (*.json)")
        if filename:
            try:
                result = self.experiment.export_data()
                result["remembered"] = self.remembered
                Path(filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                self.status_label.setText(f"Messwerte exportiert: {filename}")
            except OSError as error:
                QMessageBox.warning(self, "Export fehlgeschlagen", str(error))

    def closeEvent(self, event):
        if self.worker is not None:
            self.closing = True
            self.stop_training()
            event.ignore()
        else:
            event.accept()
