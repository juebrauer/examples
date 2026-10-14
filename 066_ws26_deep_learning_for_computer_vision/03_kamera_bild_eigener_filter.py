import contextlib
import glob
import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


@contextlib.contextmanager
def stderr_unterdrueckt():
    """Leitet stderr (auch von nativen Bibliotheken) vorübergehend nach /dev/null um."""
    sys.stderr.flush()
    alt = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(alt, 2)
        os.close(alt)
        os.close(null)


def kandidaten():
    """Liefert (index, backend) der Geräte, die als Videoquelle in Frage kommen."""
    if sys.platform.startswith("linux"):
        liste = []
        for pfad in glob.glob("/dev/video*"):
            name = os.path.basename(pfad)
            if not name[5:].isdigit():
                continue
            # Metadaten-Nodes haben in sysfs index != 0 und liefern keine Bilder
            try:
                with open(f"/sys/class/video4linux/{name}/index") as f:
                    if f.read().strip() != "0":
                        continue
            except OSError:
                pass
            liste.append((int(name[5:]), cv2.CAP_V4L2))
        return sorted(liste)
    return [(i, cv2.CAP_ANY) for i in range(10)]


def finde_kameras():
    """Liefert die Indizes aller tatsächlich öffnbaren Capture-Devices."""
    gefunden = []
    with stderr_unterdrueckt():
        for index, backend in kandidaten():
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                gefunden.append(index)
            cap.release()
    return gefunden


STANDARD_FILTER = """-1 0 1
-2 0 2
-1 0 1"""


def parse_filter(text):
    """Wandelt den Text in eine NxN-Matrix um; wirft ValueError bei ungültiger Eingabe."""
    zeilen = [z for z in text.replace(";", "\n").splitlines() if z.strip()]
    if not zeilen:
        raise ValueError("Filter ist leer")
    try:
        werte = [[float(x) for x in z.replace(",", " ").split()] for z in zeilen]
    except ValueError:
        raise ValueError("Ungültige Zahl im Filter")
    n = len(werte)
    if any(len(z) != n for z in werte):
        raise ValueError(f"Filter muss quadratisch sein (NxN, {n} Zeilen)")
    return np.array(werte, dtype=np.float32)


class KameraFenster(QWidget):
    def __init__(self, kameras):
        super().__init__()
        self.setWindowTitle("Kamerabild mit eigenem Filter")
        self.cap = None
        self.filter = parse_filter(STANDARD_FILTER)

        self.grau_label = QLabel("Keine Kamera")
        self.grau_label.setMinimumSize(480, 360)
        self.grau_label.setAlignment(Qt.AlignCenter)

        self.heat_label = QLabel("Filterantwort")
        self.heat_label.setMinimumSize(480, 360)
        self.heat_label.setAlignment(Qt.AlignCenter)

        self.combo = QComboBox()
        for index in kameras:
            self.combo.addItem(f"Kamera {index}", index)
        self.combo.currentIndexChanged.connect(self.kamera_wechseln)

        self.filter_edit = QPlainTextEdit(STANDARD_FILTER)
        self.filter_edit.setFixedWidth(260)
        self.filter_edit.textChanged.connect(self.filter_geaendert)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        rechts = QVBoxLayout()
        rechts.addWidget(QLabel("Capture-Device:"))
        rechts.addWidget(self.combo)
        rechts.addWidget(QLabel("NxN-Filter (Zeilen untereinander,\nWerte durch Leerzeichen/Komma):"))
        rechts.addWidget(self.filter_edit)
        rechts.addWidget(self.status)
        rechts.addStretch()

        layout = QHBoxLayout(self)
        layout.addWidget(self.grau_label, 1)
        layout.addWidget(self.heat_label, 1)
        layout.addLayout(rechts)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.bild_holen)
        self.timer.start(30)

        if kameras:
            self.kamera_wechseln()

    def filter_geaendert(self):
        try:
            self.filter = parse_filter(self.filter_edit.toPlainText())
            n = self.filter.shape[0]
            self.status.setText(f"Filter {n}x{n} aktiv")
        except ValueError as e:
            self.status.setText(f"Fehler: {e} (letzter gültiger Filter bleibt aktiv)")

    def kamera_wechseln(self):
        if self.cap is not None:
            self.cap.release()
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        with stderr_unterdrueckt():
            self.cap = cv2.VideoCapture(self.combo.currentData(), backend)

    def bild_holen(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ok, bild = self.cap.read()
        if not ok:
            return
        grau = np.ascontiguousarray(cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY))
        self.zeige(self.grau_label, grau, QImage.Format_Grayscale8)

        antwort = cv2.filter2D(grau.astype(np.float32), -1, self.filter)
        lo, hi = float(antwort.min()), float(antwort.max())
        norm = (antwort - lo) / (hi - lo) if hi > lo else np.zeros_like(antwort)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = np.ascontiguousarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        self.zeige(self.heat_label, heat, QImage.Format_RGB888)

    @staticmethod
    def zeige(label, array, format):
        h, w = array.shape[:2]
        qimg = QImage(array.data, w, h, array.strides[0], format)
        pixmap = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenster = KameraFenster(finde_kameras())
    fenster.show()
    sys.exit(app.exec())
