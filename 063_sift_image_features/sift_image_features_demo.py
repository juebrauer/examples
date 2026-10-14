"""Interactive SIFT teaching demo. Run with Python from env_teaching."""

from dataclasses import dataclass
import math
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QObject, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)


@dataclass(frozen=True)
class SiftSettings:
    features: int = 1000
    layers: int = 3
    contrast: float = 0.04
    edge: float = 10.0
    sigma: float = 1.6


def create_sift(settings):
    """Use the standard OpenCV SIFT implementation with 128D float descriptors."""
    try:
        return cv2.SIFT_create(
            nfeatures=settings.features, nOctaveLayers=settings.layers,
            contrastThreshold=settings.contrast, edgeThreshold=settings.edge,
            sigma=settings.sigma,
        )
    except (AttributeError, cv2.error) as error:
        raise RuntimeError(
            "Could not initialize SIFT. Use a recent standard OpenCV package "
            "(opencv-python) and valid detector settings.\n"
            f"OpenCV details: {error}"
        ) from error



def match_descriptors(reference, current, ratio):
    """L2 nearest neighbors, Lowe's ratio test, then unique target keypoints."""
    if reference is None or current is None or len(reference) == 0 or len(current) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference, current, k=2)
    candidates = [a for pair in pairs if len(pair) == 2
                  for a, b in [pair] if a.distance < ratio * b.distance]
    matches, used = [], set()
    for match in sorted(candidates, key=lambda item: item.distance):
        if match.trainIdx not in used:
            matches.append(match)
            used.add(match.trainIdx)
    return matches


def as_qimage(rgb):
    """Display grayscale pixels while keeping colored overlays separate."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0],
                  QImage.Format.Format_Grayscale8).copy()


class FeatureWorker(QObject):
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.settings = None
        self.detector = None
        self.reference_key = None
        self.reference_points = []
        self.reference_descriptors = None

    @Slot(object)
    def process(self, job):
        start = time.perf_counter()
        try:
            if self.settings != job["settings"]:
                self.detector = create_sift(job["settings"])
                self.settings = job["settings"]
                self.reference_key = None
            key = (job["reference_id"], self.settings)
            if self.reference_key != key:
                reference = job["reference"]
                self.reference_points, self.reference_descriptors = ([], None)
                if reference is not None:
                    self.reference_points, self.reference_descriptors = self.detector.detectAndCompute(
                        cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None,
                    )
                self.reference_key = key
            points, descriptors = self.detector.detectAndCompute(
                cv2.cvtColor(job["rgb"], cv2.COLOR_RGB2GRAY), None,
            )
            matches = match_descriptors(self.reference_descriptors, descriptors, job["ratio"])
            self.finished.emit(dict(
                job, reference_points=self.reference_points, points=points,
                matches=matches, elapsed=time.perf_counter() - start,
            ))
        except Exception as error:
            # Always release the GUI's busy flag, including when a backend fails.
            self.finished.emit(dict(job, error=str(error)))


class MatchView(QWidget):
    """Paint both image panes in one coordinate system for connecting lines."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.reference = QImage()
        self.current = QImage()
        self.reference_points = []
        self.points = []
        self.matches = []
        self.line_limit = 30
        self.feature_color = QColor("#66ffff")

    @staticmethod
    def image_rect(image, pane):
        if image.isNull():
            return QRectF()
        scale = min(pane.width() / image.width(), pane.height() / image.height())
        width, height = image.width() * scale, image.height() * scale
        return QRectF(pane.center().x() - width / 2, pane.center().y() - height / 2, width, height)

    @staticmethod
    def location(point, rect, image):
        scale = rect.width() / image.width()
        return QPointF(rect.left() + point.pt[0] * scale, rect.top() + point.pt[1] * scale)

    def draw_points(self, painter, image, rect, points, matched):
        if image.isNull():
            return
        painter.save()
        painter.setClipRect(rect)
        scale = rect.width() / image.width()
        for index, point in enumerate(points):
            if index not in matched:
                continue
            center = self.location(point, rect, image)
            radius = max(1.5, point.size * scale / 2)
            painter.setPen(QPen(self.feature_color, 1.1))
            painter.drawEllipse(center, radius, radius)
            if point.angle >= 0:
                angle = math.radians(point.angle)
                painter.drawLine(center, QPointF(
                    center.x() + radius * math.cos(angle),
                    center.y() + radius * math.sin(angle),
                ))
        painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#141920"))
        half = self.width() / 2
        panes = [QRectF(10, 46, half - 20, self.height() - 60),
                 QRectF(half + 10, 46, half - 20, self.height() - 60)]
        images = [self.reference, self.current]
        rects = [self.image_rect(image, pane) for image, pane in zip(images, panes)]
        for index, (image, pane, rect) in enumerate(zip(images, panes, rects)):
            painter.setPen(QColor("#e7edf5"))
            painter.drawText(QRectF(pane.x(), 8, pane.width(), 30), Qt.AlignmentFlag.AlignCenter,
                             "1 · Reference (frozen)" if index == 0 else "2 · Live camera + SIFT")
            if image.isNull():
                painter.setPen(QColor("#a4afbd"))
                painter.drawText(pane, Qt.AlignmentFlag.AlignCenter,
                                 "Click Capture reference" if index == 0 else "Waiting for camera…")
            else:
                painter.drawImage(rect, image)
        visible = self.matches[:self.line_limit]
        left_ids = {match.queryIdx for match in visible}
        right_ids = {match.trainIdx for match in visible}
        if not any(image.isNull() for image in images):
            painter.setPen(QPen(self.feature_color, 0.8))
            for match in visible:
                painter.drawLine(self.location(self.reference_points[match.queryIdx], rects[0], images[0]),
                                 self.location(self.points[match.trainIdx], rects[1], images[1]))
        self.draw_points(painter, images[0], rects[0], self.reference_points, left_ids)
        self.draw_points(painter, images[1], rects[1], self.points, right_ids)


class SiftDemo(QMainWindow):
    process_requested = Signal(object)

    def __init__(self, discover_cameras=True):
        super().__init__()
        self.setWindowTitle("SIFT Image Features — Live Matching Demo")
        self.resize(1500, 820)
        self.camera = None
        self.devices = []
        self.latest_rgb = None
        self.displayed_rgb = None
        self.reference_rgb = None
        self.reference_id = 0
        self.revision = 0
        self.frame_id = 0
        self.submitted = None
        self.busy = False
        self.closing = False
        self.sift_available = True

        self.media_devices = QMediaDevices(self)
        self.session = QMediaCaptureSession(self)
        self.sink = QVideoSink(self)
        self.session.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.receive_frame)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        self.view = MatchView()
        layout.addWidget(self.view, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        layout.addWidget(scroll)
        controls = QWidget()
        scroll.setWidget(controls)
        column = QVBoxLayout(controls)

        camera_group = QGroupBox("Camera and reference")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_choice = QComboBox()
        self.camera_choice.setMinimumContentsLength(15)
        self.camera_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        camera_layout.addWidget(self.camera_choice)
        refresh = QPushButton("Refresh cameras")
        camera_layout.addWidget(refresh)
        self.capture = QPushButton("Capture reference")
        self.capture.setEnabled(False)
        camera_layout.addWidget(self.capture)
        self.clear = QPushButton("Clear reference")
        self.clear.setEnabled(False)
        camera_layout.addWidget(self.clear)
        column.addWidget(camera_group)

        self.feature_group = QGroupBox("SIFT detector and descriptor")
        form = QFormLayout(self.feature_group)
        self.features = QSpinBox()
        self.features.setRange(0, 10000)
        self.features.setSingleStep(100)
        self.features.setSpecialValueText("Unlimited")
        self.features.setValue(1000)
        self.features.setToolTip("Retain the strongest features. Zero means unlimited; ties may exceed the target.")
        form.addRow("Target features", self.features)
        self.layers = QSpinBox()
        self.layers.setRange(1, 6)
        self.layers.setValue(3)
        self.layers.setToolTip("Scale samples per octave. The number of octaves is chosen automatically.")
        form.addRow("Layers per octave", self.layers)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.001, 1.0)
        self.contrast.setDecimals(3)
        self.contrast.setSingleStep(0.005)
        self.contrast.setValue(0.04)
        self.contrast.setToolTip("Higher values reject more weak features. OpenCV divides this value by layers per octave.")
        form.addRow("Contrast threshold", self.contrast)
        self.edge = QDoubleSpinBox()
        self.edge.setRange(1.0, 100.0)
        self.edge.setSingleStep(1.0)
        self.edge.setValue(10.0)
        self.edge.setToolTip("Higher values retain MORE edge-like features (less filtering).")
        form.addRow("Edge threshold", self.edge)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.8, 3.0)
        self.sigma.setDecimals(2)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.6)
        self.sigma.setToolTip("Gaussian blur sigma at the first octave, in pixels.")
        form.addRow("Initial sigma", self.sigma)
        column.addWidget(self.feature_group)

        matching_group = QGroupBox("Matching and display")
        matching_form = QFormLayout(matching_group)
        self.ratio = QDoubleSpinBox()
        self.ratio.setRange(0.1, 0.99)
        self.ratio.setSingleStep(0.05)
        self.ratio.setValue(0.75)
        self.ratio.setToolTip("Accept when best L2 distance < ratio × second-best distance. Lower is stricter.")
        matching_form.addRow("Lowe ratio", self.ratio)
        self.max_lines = QSpinBox()
        self.max_lines.setRange(0, 1000)
        self.max_lines.setValue(30)
        self.max_lines.setToolTip("Show only the best N accepted matches and their keypoints, ranked by lowest L2 descriptor distance. Zero hides all overlays.")
        matching_form.addRow("Best matches to show", self.max_lines)
        column.addWidget(matching_group)

        explanation = QLabel(
            "1. Point the camera at a textured object and capture a reference.\n"
            "2. Move or rotate the object to compare it with the frozen image.\n\n"
            "Circles show keypoint size; radial lines show orientation. "
            "Images are grayscale; all feature overlays are light cyan. "
            "Only the best matches by descriptor distance and their keypoints are shown. "
            "SIFT uses oriented, 128D descriptors.\n\n"
            "Thin lines show candidate correspondences (L2 + ratio test, unique targets). "
            "These are not geometrically verified and can be incorrect.\n\n"
            "Frames are reduced to at most 800 pixels on the longest side for live processing."
        )
        explanation.setWordWrap(True)
        column.addWidget(explanation)
        self.statistics = QLabel("Capture a reference to start matching.")
        self.statistics.setWordWrap(True)
        column.addWidget(self.statistics)
        self.processing_status = QLabel()
        self.processing_status.setWordWrap(True)
        self.processing_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.processing_status)
        self.camera_status = QLabel("No camera selected.")
        self.camera_status.setWordWrap(True)
        column.addWidget(self.camera_status)
        column.addStretch()

        self.worker_thread = QThread(self)
        self.worker = FeatureWorker()
        self.worker.moveToThread(self.worker_thread)
        self.process_requested.connect(self.worker.process)
        self.worker.finished.connect(self.processing_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.submit_frame)
        self.timer.start()

        self.camera_choice.currentIndexChanged.connect(self.select_camera)
        refresh.clicked.connect(self.refresh_cameras)
        self.media_devices.videoInputsChanged.connect(self.refresh_cameras)
        self.capture.clicked.connect(self.capture_reference)
        self.clear.clicked.connect(self.clear_reference)
        for widget in (self.features, self.layers, self.contrast, self.edge, self.sigma, self.ratio):
            widget.valueChanged.connect(self.settings_changed)
        self.max_lines.valueChanged.connect(self.display_changed)
        try:
            create_sift(self.settings())
        except RuntimeError as error:
            self.sift_available = False
            self.feature_group.setEnabled(False)
            self.processing_status.setText(str(error))
        if discover_cameras:
            self.refresh_cameras()

    def settings(self):
        return SiftSettings(self.features.value(), self.layers.value(), self.contrast.value(),
                            self.edge.value(), self.sigma.value())

    def settings_changed(self, *_):
        self.revision += 1
        self.view.reference_points = []
        self.view.points = []
        self.view.matches = []
        self.view.update()

    def display_changed(self, *_):
        self.view.line_limit = self.max_lines.value()
        self.view.update()

    def capture_reference(self):
        rgb = self.displayed_rgb if self.displayed_rgb is not None else self.latest_rgb
        if rgb is None:
            return
        self.reference_rgb = rgb.copy()
        self.reference_id += 1
        self.view.reference = as_qimage(self.reference_rgb)
        self.capture.setText("Recapture reference")
        self.clear.setEnabled(True)
        self.settings_changed()

    def clear_reference(self):
        self.reference_rgb = None
        self.reference_id += 1
        self.view.reference = QImage()
        self.capture.setText("Capture reference")
        self.clear.setEnabled(False)
        self.settings_changed()

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.session.setCamera(None)
            self.camera.deleteLater()
            self.camera = None
        self.latest_rgb = None
        self.displayed_rgb = None
        self.capture.setEnabled(False)
        self.revision += 1
        self.view.current = QImage()
        self.view.points = []
        self.view.matches = []
        self.view.update()

    def refresh_cameras(self):
        previous_id = self.camera.cameraDevice().id() if self.camera else None
        self.stop_camera()
        # Qt enumerates actual devices, including hot-plugged cameras.
        self.devices = QMediaDevices.videoInputs()
        self.camera_choice.blockSignals(True)
        self.camera_choice.clear()
        selected = 0
        for index, device in enumerate(self.devices):
            self.camera_choice.addItem(f"{index + 1}: {device.description()}")
            self.camera_choice.setItemData(index, bytes(device.id()).decode(errors="replace"),
                                          Qt.ItemDataRole.ToolTipRole)
            if device.id() == previous_id:
                selected = index
        self.camera_choice.setEnabled(bool(self.devices))
        self.camera_choice.setCurrentIndex(selected if self.devices else -1)
        self.camera_choice.blockSignals(False)
        if self.devices:
            self.select_camera(selected)
        else:
            self.camera_status.setText("No cameras found. Connect a camera and click Refresh cameras.")

    def select_camera(self, index):
        self.stop_camera()
        if not 0 <= index < len(self.devices):
            return
        camera = QCamera(self.devices[index], self)
        self.camera = camera
        camera.errorOccurred.connect(
            lambda error, message: self.camera_error(camera, message))
        self.session.setCamera(camera)
        self.camera_status.setText("Starting camera…")
        camera.start()
        QTimer.singleShot(5000, lambda: self.check_camera(camera))

    def check_camera(self, camera):
        if self.camera is camera and self.latest_rgb is None:
            self.camera_status.setText("No frames received. Check camera permissions or close other camera apps.")

    def camera_error(self, camera, message):
        if self.camera is camera:
            self.stop_camera()
            self.camera_status.setText(f"Camera error: {message}\nSelect a camera or refresh to retry.")

    def receive_frame(self, frame):
        if self.camera is None or not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        if max(image.width(), image.height()) > 800:
            image = image.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        image = image.convertToFormat(QImage.Format.Format_RGB888)
        # RGB888 scanlines may have padding; copy only the actual pixel columns.
        rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        self.latest_rgb = rows[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()
        self.frame_id += 1
        self.capture.setEnabled(True)
        self.camera_status.setText(f"Live input · {image.width()} × {image.height()}")
        if not self.sift_available:
            self.displayed_rgb = self.latest_rgb
            self.view.current = as_qimage(self.latest_rgb)
            self.view.update()

    def submit_frame(self):
        token = (self.frame_id, self.revision)
        if (self.closing or self.busy or not self.sift_available or self.latest_rgb is None
                or token == self.submitted):
            return
        self.busy = True
        self.submitted = token
        # At most one job is queued. New camera frames replace the pending image.
        self.process_requested.emit(dict(
            rgb=self.latest_rgb, reference=self.reference_rgb,
            reference_id=self.reference_id, settings=self.settings(),
            ratio=self.ratio.value(), revision=self.revision,
        ))

    @Slot(object)
    def processing_finished(self, result):
        self.busy = False
        if self.closing or result["revision"] != self.revision:
            return
        if "error" in result:
            self.processing_status.setText(f"Processing error: {result['error']}")
            return
        self.displayed_rgb = result["rgb"]
        self.view.current = as_qimage(result["rgb"])
        self.view.reference_points = result["reference_points"]
        self.view.points = result["points"]
        self.view.matches = result["matches"]
        self.view.update()
        self.statistics.setText(
            f"Reference keypoints: {len(result['reference_points'])}\n"
            f"Live keypoints: {len(result['points'])}\n"
            f"Accepted matches: {len(result['matches'])}\n"
            "Descriptor: 128D"
        )
        self.processing_status.setText(f"SIFT + matching: {result['elapsed'] * 1000:.0f} ms")

    def closeEvent(self, event):
        self.closing = True
        self.timer.stop()
        self.stop_camera()
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    cv2.setNumThreads(2)
    window = SiftDemo()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
