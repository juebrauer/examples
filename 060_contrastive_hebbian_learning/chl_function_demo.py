"""Interaktive Demonstration von Contrastive Hebbian Learning (CHL).

Die Anwendung approximiert eine unregelmäßige eindimensionale Funktion mit
einem kleinen bidirektionalen Netz. Das Training läuft in einem separaten
QThread, damit das Qt-Hauptfenster während mehrerer Epochen bedienbar bleibt.
Die Visualisierung wird erst nach Abschluss des gesamten Trainings aktualisiert.

Start:
    python chl_function_demo.py

Abhängigkeiten:
    numpy, matplotlib, PySide6
"""

from __future__ import annotations

import sys

import numpy as np
from chl_model import CHLNetwork, X_MAX, X_MIN, target_function
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainingWorker(QThread):
    """Trainiert das Modell, ohne den Qt-GUI-Thread zu blockieren.

    QThread.run() wird von Qt in einem eigenen Betriebssystem-Thread
    ausgeführt. Währenddessen greift die GUI nicht auf das Modell zu. Erst das
    Abschlusssignal gibt die Nutzung des Modells für den Hauptthread wieder
    frei.
    """

    training_completed = Signal(object)
    training_failed = Signal(str)

    def __init__(
        self,
        network: CHLNetwork,
        train_x: np.ndarray,
        train_y: np.ndarray,
        epochs: int,
        learning_rate: float,
        relax_steps: int,
        random_seed: int,
        parent=None,
    ) -> None:
        super().__init__(parent)

        # Es existiert bewusst nur ein Modell. Während run() arbeitet, ist der
        # Worker sein exklusiver Benutzer: Die GUI verändert oder liest es erst
        # wieder nach dem Abschlusssignal. Die Trainingsarrays werden dagegen
        # kopiert, damit auch sie im Worker unveränderlich und unabhängig sind.
        self.network = network
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.relax_steps = relax_steps
        self.random_seed = random_seed

    def run(self) -> None:
        """Führt alle Epochen aus und meldet genau ein Endergebnis zurück."""
        try:
            rng = np.random.default_rng(self.random_seed)

            for _ in range(self.epochs):
                # Online-CHL mit in jeder Epoche neu gemischten Beispielen.
                for sample_index in rng.permutation(self.train_x.size):
                    # closeEvent() kann einen Abbruch anfordern. Geprüft wird
                    # zwischen zwei Beispielen, damit kein Update halb endet.
                    if self.isInterruptionRequested():
                        return

                    self.network.train_sample(
                        float(self.train_x[sample_index]),
                        float(self.train_y[sample_index]),
                        self.learning_rate,
                        self.relax_steps,
                    )

            prediction = self.network.predict(self.train_x, self.relax_steps)
            mse = float(np.mean((prediction - self.train_y) ** 2))

            # Qt stellt die Signalverbindung zum GUI-Objekt automatisch in die
            # Ereigniswarteschlange des Hauptthreads. Der Slot zeichnet daher
            # sicher im GUI-Thread und niemals direkt aus diesem Worker heraus.
            self.training_completed.emit(
                {
                    "epochs": self.epochs,
                    "mse": mse,
                }
            )
        except Exception as error:  # Fehler kontrolliert an die GUI weitergeben
            self.training_failed.emit(f"{type(error).__name__}: {error}")


class CHLDemo(QMainWindow):
    """Qt-Hauptfenster und Steuerung des CHL-Trainings-Threads."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Contrastive Hebbian Learning — Function Approximation")
        self.resize(1180, 760)

        # Ein fester Seed macht die Vorlesungsdemo reproduzierbar. Wiederholte
        # Klicks auf "Generate" liefern dennoch nacheinander neue Datensätze.
        self.rng = np.random.default_rng(23)
        self.network = CHLNetwork()

        # Trainingsdaten und Anzahl der bereits abgeschlossenen Epochen.
        self.train_x = np.array([], dtype=float)
        self.train_y = np.array([], dtype=float)
        self.epoch = 0

        # Solange kein Training läuft, verweist worker auf None. Die Referenz
        # muss während des Trainings erhalten bleiben, damit Qt den QThread
        # nicht vorzeitig zerstört.
        self.worker: TrainingWorker | None = None

        self._build_ui()
        self._generate_training_data()

    def _build_ui(self) -> None:
        """Erzeugt Bedienelemente und das Matplotlib-Diagramm."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("Contrastive Hebbian Learning: freie vs. geklemmte Phase")
        title.setFont(QFont("Sans Serif", 16, QFont.Weight.Bold))
        root.addWidget(title)

        explanation = QLabel(
            "Minus phase: x is clamped, the network predicts freely.   "
            "Plus phase: x and target y are clamped.   "
            "Update: ΔW = η (s⁺ᵀs⁺ − s⁻ᵀs⁻).   "
            "x activates a fixed population of 31 overlapping input units."
        )
        explanation.setWordWrap(True)
        root.addWidget(explanation)

        controls = QHBoxLayout()
        data_box = QGroupBox("Training data")
        data_form = QFormLayout(data_box)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(4, 500)
        self.n_spin.setValue(60)
        self.n_spin.setToolTip("Number of uniformly sampled training examples")
        self.generate_button = QPushButton("Generate Training Data")
        self.generate_button.clicked.connect(self._generate_training_data)
        data_form.addRow("Number of samples N:", self.n_spin)
        data_form.addRow(self.generate_button)
        controls.addWidget(data_box)

        chl_box = QGroupBox("CHL parameters")
        chl_form = QFormLayout(chl_box)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.2)
        self.lr_spin.setSingleStep(0.002)
        self.lr_spin.setValue(0.012)
        self.relax_spin = QSpinBox()
        self.relax_spin.setRange(5, 100)
        self.relax_spin.setValue(30)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10_000)
        self.epochs_spin.setValue(1)
        self.epochs_spin.setToolTip(
            "Number of complete epochs trained before the plots are updated"
        )
        chl_form.addRow("Learning rate η:", self.lr_spin)
        chl_form.addRow("Relaxation steps:", self.relax_spin)
        chl_form.addRow("Number of epochs N:", self.epochs_spin)
        controls.addWidget(chl_box)

        self.train_button = QPushButton("Train with CHL")
        self.train_button.setMinimumHeight(72)
        self.train_button.clicked.connect(self._start_training)
        controls.addWidget(self.train_button, stretch=1)
        root.addLayout(controls)

        # Da es bewusst keine Zwischenvisualisierung mehr gibt, genügt ein
        # großes Diagramm für Ziel- und Approximationsfunktion.
        self.figure = Figure(figsize=(11, 5.5), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.fit_axis = self.figure.add_subplot(111)
        root.addWidget(self.canvas, stretch=1)

        self.status = QLabel("Ready")
        self.status.setStyleSheet("padding: 6px; background: #eeeeee;")
        root.addWidget(self.status)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Verhindert Parameteränderungen mitten in einer laufenden Epoche."""
        self.generate_button.setEnabled(enabled)
        self.n_spin.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        self.relax_spin.setEnabled(enabled)
        self.epochs_spin.setEnabled(enabled)
        self.train_button.setEnabled(enabled)

    def _generate_training_data(self) -> None:
        """Sampelt N Trainingspunkte und setzt das CHL-Netz zurück."""
        if self.worker is not None:
            return
        n = self.n_spin.value()

        # Die x-Werte werden gleichverteilt gezogen und nur für eine übersicht-
        # liche Darstellung sortiert. train_x und train_y gehören paarweise
        # zusammen; die spätere Trainingsreihenfolge wird separat gemischt.
        self.train_x = np.sort(self.rng.uniform(X_MIN, X_MAX, n))
        self.train_y = target_function(self.train_x)

        # Ein neuer Datensatz startet absichtlich auch mit neuen, noch
        # untrainierten Modellzuständen (aber reproduzierbarer Initialisierung).
        self.network = CHLNetwork(seed=7)
        self.epoch = 0
        self.status.setText(
            f"Generated {n} samples. The network was reset; press Train to begin."
        )
        self._draw_fit()
        self.canvas.draw_idle()

    def _start_training(self) -> None:
        """Erzeugt und startet einen Worker für die gewählte Epochendauer."""
        if self.train_x.size == 0:
            QMessageBox.information(self, "No data", "Generate training data first.")
            return

        epochs = self.epochs_spin.value()
        self._set_controls_enabled(False)
        self.status.setText(
            f"Training {epochs} epoch(s) in worker thread — GUI remains responsive …"
        )

        # Der Seed wird im GUI-Thread erzeugt und als einfacher Integer an den
        # Worker übergeben; der GUI-Zufallsgenerator wird dort nicht gemeinsam
        # benutzt.
        random_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        self.worker = TrainingWorker(
            network=self.network,
            train_x=self.train_x,
            train_y=self.train_y,
            epochs=epochs,
            learning_rate=self.lr_spin.value(),
            relax_steps=self.relax_spin.value(),
            random_seed=random_seed,
            parent=self,
        )
        self.worker.training_completed.connect(self._training_completed)
        self.worker.training_failed.connect(self._training_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _training_completed(self, result: object) -> None:
        """Übernimmt das fertige Modell und aktualisiert die Diagramme einmal."""
        if not isinstance(result, dict):
            self._training_failed("Worker returned an invalid result.")
            return

        trained_epochs = int(result["epochs"])
        mse = float(result["mse"])
        self.epoch += trained_epochs

        self._draw_fit()
        self.canvas.draw_idle()
        self.status.setText(
            f"Training complete — {trained_epochs} new epoch(s), "
            f"{self.epoch} total, training MSE: {mse:.5f}"
        )

    def _training_failed(self, message: str) -> None:
        """Zeigt eine im Worker aufgetretene Ausnahme im GUI-Thread an."""
        self.status.setText(f"Training failed: {message}")
        QMessageBox.critical(self, "Training failed", message)

    def _worker_finished(self) -> None:
        """Räumt den beendeten QThread auf und aktiviert die Bedienung."""
        self._set_controls_enabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _draw_fit(self) -> None:
        """Zeichnet Zielfunktion, Trainingspunkte und aktuelle Approximation."""
        axis = self.fit_axis
        axis.clear()
        grid = np.linspace(X_MIN, X_MAX, 500)
        truth = target_function(grid)
        # Auch die Kurve f_hat entsteht durch freie Relaxation. Die 500
        # Gitterpunkte werden dafür effizient als ein Batch ausgewertet.
        estimate = self.network.predict(grid, self.relax_spin.value())
        axis.plot(grid, truth, color="#222222", linewidth=2.0, label="target f(x)")
        axis.plot(grid, estimate, color="#d62728", linewidth=2.2, label="CHL estimate f̂(x)")
        axis.scatter(
            self.train_x,
            self.train_y,
            s=24,
            color="#1f77b4",
            alpha=0.75,
            zorder=4,
            label="training samples",
        )
        axis.set_title("Function approximation (updated after complete training)")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(X_MIN, X_MAX)
        axis.set_ylim(-1.15, 1.15)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        """Beendet einen laufenden Worker kontrolliert vor dem Fensterschluss."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)


def main() -> None:
    """Startet Qt-Ereignisschleife und Hauptfenster."""
    app = QApplication(sys.argv)
    window = CHLDemo()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
