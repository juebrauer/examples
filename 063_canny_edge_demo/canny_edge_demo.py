"""Live Canny edge detection demo. Run with Python in env_teaching."""

import sys

import cv2
import matplotlib as mpl
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import (
    QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)


# Build an RGB lookup table from Matplotlib's coolwarm so the UI can colorize fast.
COOLWARM_LUT_RGB = np.rint(
    mpl.colormaps["coolwarm"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0,
).astype(np.uint8)


def processing_stages(rgb, kernel_size, lower_threshold, upper_threshold):
    """Return the blurred image, raw Sobel magnitude, and Canny edges."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # A kernel size of 1 leaves the grayscale image unchanged.
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Match Canny's 3 x 3 derivatives and replicated image borders.
    dx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    magnitude = cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32))
    edges = cv2.Canny(dx, dy, lower_threshold, upper_threshold, L2gradient=True)
    return blurred, magnitude, edges


def gradient_heatmap(magnitude):
    """Map each frame's minimum to blue and maximum to red using Matplotlib coolwarm."""
    minimum = float(magnitude.min())
    maximum = float(magnitude.max())
    if maximum > minimum:
        normalized = np.rint((magnitude - minimum) * (255.0 / (maximum - minimum)))
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    else:
        # A constant image has no range; display it entirely in the minimum color.
        normalized = np.zeros(magnitude.shape, dtype=np.uint8)
    return COOLWARM_LUT_RGB[normalized]


class ImageView(QLabel):
    """Display an image with its aspect ratio preserved during resizing."""

    def __init__(self):
        super().__init__("No camera image")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 180)
        # Ignore pixmap size hints so every grid cell stays the same size.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #181818; color: #dddddd;")
        self.image = QImage()

    def set_image(self, image):
        self.image = image.copy()
        self.update_pixmap()

    def update_pixmap(self):
        if not self.image.isNull():
            self.setPixmap(QPixmap.fromImage(self.image).scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def reset(self):
        self.image = QImage()
        self.clear()
        self.setText("No camera image")


class CannyDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canny Edge Detection — Live Camera Demo")
        self.resize(1280, 800)
        self.camera = None
        self.last_rgb = None
        self.devices = []
        self.media_devices = QMediaDevices(self)
        self.session = QMediaCaptureSession(self)
        self.sink = QVideoSink(self)
        self.session.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.receive_frame)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        grid = QGridLayout()
        layout.addLayout(grid, 1)
        self.original = ImageView()
        self.blurred = ImageView()
        self.gradient = ImageView()
        self.edges = ImageView()
        self.views = (self.original, self.blurred, self.gradient, self.edges)
        self.statistics = []
        titles = ("Camera image (RGB)", "Gaussian blur (grayscale)",
                  "Sobel gradient magnitude (L2 heatmap)", "Canny edges")
        for index, (title, view) in enumerate(zip(titles, self.views)):
            group = QGroupBox(title)
            column = QVBoxLayout(group)
            column.addWidget(view, 1)
            statistics = QLabel("Min: —    Max: —")
            statistics.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(statistics)
            self.statistics.append(statistics)
            grid.addWidget(group, index // 2, index % 2)
        for index in range(2):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)

        controls = QGroupBox("Controls")
        controls.setFixedWidth(285)
        layout.addWidget(controls)
        column = QVBoxLayout(controls)
        form = QFormLayout()
        column.addLayout(form)
        self.camera_choice = QComboBox()
        self.camera_choice.setMinimumContentsLength(12)
        self.camera_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addRow("Camera", self.camera_choice)
        refresh = QPushButton("Refresh cameras")
        column.addWidget(refresh)

        self.kernel = QComboBox()
        for size in range(1, 32, 2):
            self.kernel.addItem(f"{size} × {size}", size)
        self.kernel.setCurrentIndex(2)
        form.addRow("Gaussian kernel", self.kernel)
        self.lower = QSpinBox()
        self.upper = QSpinBox()
        for spin, value in ((self.lower, 50), (self.upper, 150)):
            spin.setRange(0, 2040)
            spin.setValue(value)
        form.addRow("Lower threshold", self.lower)
        form.addRow("Upper threshold", self.upper)

        explanation = QLabel(
            "Pipeline: grayscale → Gaussian blur → Sobel gradients → Canny\n\n"
            "Larger kernels reduce noise and fine detail. Sigma is chosen automatically.\n\n"
            "Gradients above the upper threshold seed strong edges. "
            "Gradients between the thresholds survive only when connected to strong edges.\n\n"
            "Canny uses a 3 × 3 Sobel operator and the L2 gradient magnitude. "
            "Thresholds measure gradient strength, so they can exceed 255.\n\n"
            "Sobel heatmap (Matplotlib coolwarm): blue = frame minimum, red = frame maximum. "
            "Colors are rescaled for each frame. Min/Max values show raw gradient "
            "magnitudes. Camera Min/Max spans all RGB channels."
        )
        explanation.setWordWrap(True)
        column.addWidget(explanation)
        column.addStretch()
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)

        self.camera_choice.currentIndexChanged.connect(self.select_camera)
        refresh.clicked.connect(self.refresh_cameras)
        self.media_devices.videoInputsChanged.connect(self.refresh_cameras)
        self.kernel.currentIndexChanged.connect(self.render_images)
        self.lower.valueChanged.connect(self.lower_changed)
        self.upper.valueChanged.connect(self.upper_changed)
        self.refresh_cameras()

    def lower_changed(self, value):
        if value > self.upper.value():
            self.upper.setValue(value)
        self.render_images()

    def upper_changed(self, value):
        if value < self.lower.value():
            self.lower.setValue(value)
        self.render_images()

    def stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.session.setCamera(None)
            self.camera.deleteLater()
            self.camera = None
        self.last_rgb = None
        for view, statistics in zip(self.views, self.statistics):
            view.reset()
            statistics.setText("Min: —    Max: —")

    def refresh_cameras(self):
        previous_id = self.camera.cameraDevice().id() if self.camera else None
        self.stop_camera()
        # Enumerate actual devices through Qt rather than guessing camera indices.
        self.devices = QMediaDevices.videoInputs()
        self.camera_choice.blockSignals(True)
        self.camera_choice.clear()
        selected = 0
        for index, device in enumerate(self.devices):
            self.camera_choice.addItem(f"{index + 1}: {device.description()}")
            self.camera_choice.setItemData(index, bytes(device.id()).decode(errors="replace"), Qt.ItemDataRole.ToolTipRole)
            if device.id() == previous_id:
                selected = index
        self.camera_choice.setEnabled(bool(self.devices))
        if self.devices:
            self.camera_choice.setCurrentIndex(selected)
        self.camera_choice.blockSignals(False)
        if self.devices:
            self.select_camera(selected)
        else:
            self.status.setText("No cameras found. Connect a camera and click Refresh cameras.")

    def select_camera(self, index):
        self.stop_camera()
        if not 0 <= index < len(self.devices):
            return
        self.camera = QCamera(self.devices[index], self)
        self.camera.errorOccurred.connect(self.camera_error)
        self.session.setCamera(self.camera)
        self.status.setText("Starting camera…")
        self.camera.start()

    def camera_error(self, error, message):
        self.status.setText(f"Camera error: {message}\nCheck camera permissions or close other camera applications.")

    def receive_frame(self, frame):
        if self.camera is None or not frame.isValid():
            return
        image = frame.toImage().convertToFormat(QImage.Format.Format_RGB888)
        if image.isNull():
            return
        # Respect Qt's row padding, then own the pixels beyond the QImage lifetime.
        rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine(),
        )
        self.last_rgb = rows[:, :image.width() * 3].reshape(
            image.height(), image.width(), 3,
        ).copy()
        self.render_images()
        self.status.setText(f"Live · {image.width()} × {image.height()}")

    def render_images(self, *_):
        if self.last_rgb is None:
            return
        rgb = self.last_rgb
        blurred, magnitude, edges = processing_stages(
            rgb, self.kernel.currentData(), self.lower.value(), self.upper.value(),
        )
        gradient_display = gradient_heatmap(magnitude)
        raw_images = (rgb, blurred, magnitude, edges)
        display_images = (rgb, blurred, gradient_display, edges)
        for view, statistics, raw, display in zip(
            self.views, self.statistics, raw_images, display_images,
        ):
            height, width = display.shape[:2]
            image_format = (QImage.Format.Format_RGB888 if display.ndim == 3
                            else QImage.Format.Format_Grayscale8)
            view.set_image(QImage(display.data, width, height, display.strides[0], image_format))
            if np.issubdtype(raw.dtype, np.floating):
                statistics.setText(f"Min: {raw.min():.2f}    Max: {raw.max():.2f}")
            else:
                statistics.setText(f"Min: {raw.min()}    Max: {raw.max()}")

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = CannyDemo()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
