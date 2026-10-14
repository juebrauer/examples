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


class KameraFenster(QWidget):
    def __init__(self, kameras):
        super().__init__()
        self.setWindowTitle("Kamerabild")
        self.cap = None

        self.bild_label = QLabel("Keine Kamera")
        self.bild_label.setMinimumSize(640, 480)
        self.bild_label.setAlignment(Qt.AlignCenter)

        self.maske_label = QLabel("Segmentierung (Gelb)")
        self.maske_label.setMinimumSize(640, 480)
        self.maske_label.setAlignment(Qt.AlignCenter)

        self.combo = QComboBox()
        for index in kameras:
            self.combo.addItem(f"Kamera {index}", index)
        self.combo.currentIndexChanged.connect(self.kamera_wechseln)

        rechts = QVBoxLayout()
        rechts.addWidget(QLabel("Capture-Device:"))
        rechts.addWidget(self.combo)
        rechts.addStretch()

        layout = QHBoxLayout(self)
        layout.addWidget(self.bild_label, 1)
        layout.addWidget(self.maske_label, 1)
        layout.addLayout(rechts)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.bild_holen)
        self.timer.start(30)

        if kameras:
            self.kamera_wechseln()

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
        rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
        self.zeige(self.bild_label, rgb, QImage.Format_RGB888)

        # Gelb: Rot und Grün hoch, Blau niedrig
        maske = (rgb[..., 0] > 125) & (rgb[..., 1] > 125) & (rgb[..., 2] < 125)
        maske = np.ascontiguousarray(maske.astype(np.uint8) * 255)
        self.zeige(self.maske_label, maske, QImage.Format_Grayscale8)

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
