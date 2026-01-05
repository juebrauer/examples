"""
DBSCAN demo (single-file version)

Controls:
- Left click: add one point
- Right click: add a small random cloud of points near the mouse
- R: run DBSCAN (scikit-learn)
- E: toggle epsilon neighborhood visualization around the nearest point
- C: clear all points

Dependencies:
    pip install pyside6 scikit-learn numpy
"""

import sys
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from PySide6 import QtCore, QtGui, QtWidgets


# --------
# Settings
# --------
RIGHT_CLICK_POINTS_TO_GENERATE = 10
POINT_RADIUS = 5

DBSCAN_EPSILON = 60
DBSCAN_MIN_SAMPLES = 12


Point = Tuple[int, int]


class DBSCANDemo(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setMouseTracking(True)

        self.points: List[Point] = []
        self.labels_by_point: Optional[Dict[Point, Dict[str, object]]] = None

        self.show_epsilon_neighborhood = False
        self.nearest_point_to_mouse: Optional[Point] = None

        # A small fixed palette + many random colors for clusters
        self.cluster_colors: List[QtGui.QColor] = [
            QtGui.QColor(0, 0, 0),       # used for noise / fallback
            QtGui.QColor(255, 0, 0),
            QtGui.QColor(0, 255, 0),
            QtGui.QColor(0, 0, 255),
            QtGui.QColor(0, 255, 255),
            QtGui.QColor(128, 128, 128),
        ]
        for _ in range(1000 - len(self.cluster_colors)):
            r = int(np.random.randint(0, 256))
            g = int(np.random.randint(0, 256))
            b = int(np.random.randint(0, 256))
            self.cluster_colors.append(QtGui.QColor(r, g, b))

    # ----------------------------
    # Input handling
    # ----------------------------
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._update_mouse_state(event)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._update_mouse_state(event)

        if event.button() == QtCore.Qt.LeftButton:
            self._add_point(self._mouse_x, self._mouse_y)

        elif event.button() == QtCore.Qt.RightButton:
            self._add_random_cloud(self._mouse_x, self._mouse_y, radius=30)

        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        char = chr(key) if 0 <= key <= 0x10FFFF else ""

        if char in ("R", "r"):
            self._run_dbscan()

        elif char in ("E", "e"):
            self.show_epsilon_neighborhood = not self.show_epsilon_neighborhood

        elif char in ("C", "c"):
            self._clear()

        self.update()

    # ----------------------------
    # Core logic
    # ----------------------------
    def _add_point(self, x: int, y: int) -> None:
        # Adding new points invalidates previous clustering
        self.labels_by_point = None
        self.points.append((int(x), int(y)))

    def _add_random_cloud(self, x: int, y: int, radius: int) -> None:
        height = self.size().height()
        width = self.size().width()

        for _ in range(RIGHT_CLICK_POINTS_TO_GENERATE):
            rx = x - radius + random.randint(0, 2 * radius)
            ry = y - radius + random.randint(0, 2 * radius)
            rx = int(np.clip(rx, 0, width - 1))
            ry = int(np.clip(ry, 0, height - 1))
            self._add_point(rx, ry)

    def _clear(self) -> None:
        self.points = []
        self.labels_by_point = None
        self.nearest_point_to_mouse = None
        self.show_epsilon_neighborhood = False

    def _run_dbscan(self) -> None:
        if not self.points:
            self.labels_by_point = None
            return

        data = np.array(self.points, dtype=float)

        model = DBSCAN(eps=DBSCAN_EPSILON, min_samples=DBSCAN_MIN_SAMPLES)
        model.fit(data)

        labels = model.labels_  # -1 is noise, 0..k-1 are clusters
        core_indices = set(getattr(model, "core_sample_indices_", []))

        # Create labels by point (mirrors your original dictionary approach)
        labels_by_point: Dict[Point, Dict[str, object]] = {}

        # Note: if you add duplicate points, dict keys collide.
        # Your original code also assumed unique (x, y) locations.
        for i, p in enumerate(self.points):
            cluster_label = int(labels[i])

            if cluster_label == -1:
                point_type = "noise"
                cluster_id_for_color = 0  # use black
                cluster_id_for_display = -1
            else:
                if i in core_indices:
                    point_type = "core"
                else:
                    point_type = "border"

                # Shift by +1 so cluster 0 doesn't overwrite "noise color" at index 0
                cluster_id_for_color = cluster_label + 1
                cluster_id_for_display = cluster_label

            labels_by_point[p] = {
                "data_point_type": point_type,
                "cluster_id": cluster_id_for_display,
                "color_index": cluster_id_for_color,
            }

        self.labels_by_point = labels_by_point

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Clustering {len(self.points)} data points with sklearn DBSCAN ...")
        print("Clustering finished.")
        print(f"Found {num_clusters} cluster(s).")

    # ----------------------------
    # Mouse state + nearest point
    # ----------------------------
    def _update_mouse_state(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position().toPoint()
        self._mouse_x = int(pos.x())
        self._mouse_y = int(pos.y())

        self._update_nearest_point()
        self._update_window_title()

    def _update_nearest_point(self) -> None:
        if not self.points:
            self.nearest_point_to_mouse = None
            return

        mx, my = self._mouse_x, self._mouse_y
        best_point = None
        best_dist2 = None

        for (px, py) in self.points:
            dx = mx - px
            dy = my - py
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_point = (px, py)

        self.nearest_point_to_mouse = best_point

    def _update_window_title(self) -> None:
        n_points = len(self.points)

        point_type = ""
        cluster_id = ""
        if self.labels_by_point and self.nearest_point_to_mouse in self.labels_by_point:
            info = self.labels_by_point[self.nearest_point_to_mouse]
            point_type = str(info["data_point_type"])
            cluster_id = str(info["cluster_id"])

        self.setWindowTitle(
            f"mouse: ({self._mouse_x},{self._mouse_y}), points: {n_points}    "
            f"type: {point_type}, cluster: {cluster_id}"
        )

    # ----------------------------
    # Painting
    # ----------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)        
        self._draw_all(painter)        

    def _draw_all(self, painter: QtGui.QPainter) -> None:
        # Font setup (kept simple)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPixelSize(17)
        font.setBold(False)
        painter.setFont(font)

        # Draw points
        for p in self.points:
            color = QtGui.QColor(0, 0, 0)

            if self.labels_by_point and p in self.labels_by_point:
                color_index = int(self.labels_by_point[p]["color_index"])
                color = self.cluster_colors[color_index % len(self.cluster_colors)]

            pen = QtGui.QPen(color)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPoint(*p), POINT_RADIUS, POINT_RADIUS)

        # Draw epsilon neighborhood around nearest point
        if self.show_epsilon_neighborhood and self.nearest_point_to_mouse is not None:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)

            painter.drawEllipse(
                QtCore.QPoint(*self.nearest_point_to_mouse),
                DBSCAN_EPSILON,
                DBSCAN_EPSILON,
            )


def main() -> None:
    manual = f"""
DBSCAN demo (single-file, scikit-learn)

Click:
  left mouse button  -> generate one point
  right mouse button -> generate {RIGHT_CLICK_POINTS_TO_GENERATE} points near the mouse

Press:
  C -> clear points
  R -> run DBSCAN
  E -> toggle epsilon-neighborhood of the point nearest to the mouse
"""
    print(manual)

    app = QtWidgets.QApplication([])
    widget = DBSCANDemo()
    widget.resize(800, 800)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
