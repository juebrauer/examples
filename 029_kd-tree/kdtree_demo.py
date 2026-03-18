import sys
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# Datenmodell
# ============================================================

@dataclass
class Point:
    name: str
    x: float
    y: float


@dataclass
class KDNode:
    point: Point
    axis: int  # 0=x, 1=y
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


# ============================================================
# Parsing
# ============================================================

def parse_points(text: str) -> List[Point]:
    points: List[Point] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            if "=" not in line:
                raise ValueError("Es fehlt '='.")
            left, right = line.split("=", 1)
            name = left.strip()
            if not name:
                raise ValueError("Punktname fehlt.")

            right = right.strip()
            if not (right.startswith("(") and right.endswith(")")):
                raise ValueError("Koordinaten müssen in Klammern stehen, z. B. (2,3).")

            coords = right[1:-1]
            if "," not in coords:
                raise ValueError("Koordinaten müssen durch Komma getrennt sein.")

            xs, ys = coords.split(",", 1)
            x = float(xs.strip())
            y = float(ys.strip())

            points.append(Point(name=name, x=x, y=y))
        except Exception as exc:
            raise ValueError(f"Fehler in Zeile {line_no}: '{raw_line}' -> {exc}") from exc

    if not points:
        raise ValueError("Keine Punkte gefunden.")

    names = [p.name for p in points]
    if len(names) != len(set(names)):
        raise ValueError("Punktnamen müssen eindeutig sein.")

    return points


# ============================================================
# KD-Tree mit Untermedian
# ============================================================

def build_kdtree(points: List[Point], depth: int = 0) -> Optional[KDNode]:
    if not points:
        return None

    axis = depth % 2

    if axis == 0:
        points_sorted = sorted(points, key=lambda p: (p.x, p.y, p.name))
    else:
        points_sorted = sorted(points, key=lambda p: (p.y, p.x, p.name))

    # Untermedian:
    # bei 2 Elementen: Index 0
    # bei 4 Elementen: Index 1
    median = (len(points_sorted) - 1) // 2

    return KDNode(
        point=points_sorted[median],
        axis=axis,
        left=build_kdtree(points_sorted[:median], depth + 1),
        right=build_kdtree(points_sorted[median + 1 :], depth + 1),
    )


# ============================================================
# Matplotlib-Plot: Punkte + Trennebenen
# ============================================================

class KDPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 5))
        super().__init__(self.figure)
        self.setParent(parent)

    def draw_kdtree(self, points: List[Point], root: Optional[KDNode]) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not points:
            ax.set_title("Keine Punkte")
            self.draw()
            return

        xs = [p.x for p in points]
        ys = [p.y for p in points]

        margin = 1.0
        xmin, xmax = min(xs) - margin, max(xs) + margin
        ymin, ymax = min(ys) - margin, max(ys) + margin

        def draw_splits(node: Optional[KDNode], x0: float, x1: float, y0: float, y1: float) -> None:
            if node is None:
                return

            px, py = node.point.x, node.point.y

            if node.axis == 0:
                ax.plot([px, px], [y0, y1], color="red", linewidth=2)
                draw_splits(node.left, x0, px, y0, y1)
                draw_splits(node.right, px, x1, y0, y1)
            else:
                ax.plot([x0, x1], [py, py], color="blue", linewidth=2)
                draw_splits(node.left, x0, x1, y0, py)
                draw_splits(node.right, x0, x1, py, y1)

        draw_splits(root, xmin, xmax, ymin, ymax)

        ax.scatter(xs, ys, s=180, facecolors="white", edgecolors="black", linewidths=1.6, zorder=3)

        for p in points:
            ax.text(p.x + 0.08, p.y + 0.08, f"{p.name}", fontsize=11, zorder=4)
            ax.text(p.x + 0.08, p.y - 0.28, f"({p.x:g},{p.y:g})", fontsize=9, zorder=4)

        ax.set_title("Punkte und Trennebenen")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.text(
            0.02, 0.98,
            "Rot = x-Split\nBlau = y-Split",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        self.figure.tight_layout()
        self.draw()


# ============================================================
# Graph-Ansicht ähnlich deiner Skizze
# ============================================================

class KDTreeGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setMinimumHeight(420)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom_min = 0.05
        self._zoom_max = 50.0

    def wheelEvent(self, event) -> None:
        # Zoom mit Mausrad
        angle = event.angleDelta().y()
        if angle == 0:
            event.ignore()
            return

        zoom_in_factor = 1.25
        factor = zoom_in_factor if angle > 0 else 1 / zoom_in_factor

        current = self.transform().m11()
        new_scale = current * factor
        if new_scale < self._zoom_min or new_scale > self._zoom_max:
            event.accept()
            return

        self.scale(factor, factor)
        event.accept()

    def draw_tree(self, root: Optional[KDNode]) -> None:
        self._scene.clear()

        if root is None:
            return

        node_radius = 26
        edge_pen = QPen(QColor("#d08a00"), 2)

        def axis_color(axis: int) -> QColor:
            return QColor("red") if axis == 0 else QColor("blue")

        def edge_label(axis: int, is_left: bool) -> str:
            if axis == 0:
                return "x kleiner" if is_left else "x größer"
            return "y kleiner" if is_left else "y größer"

        def height(node: Optional[KDNode]) -> int:
            if node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))

        max_depth = height(root) - 1
        base_step = 55.0

        def add_node(node: KDNode, depth: int, x: float, y: float) -> None:
            # Knoten zeichnen
            color = axis_color(node.axis)

            circle = QGraphicsEllipseItem(
                x - node_radius,
                y - node_radius,
                2 * node_radius,
                2 * node_radius,
            )
            pen = QPen(color)
            pen.setWidth(2)
            circle.setPen(pen)
            circle.setBrush(QBrush(Qt.transparent))
            self._scene.addItem(circle)

            label = f"({node.point.x:g},{node.point.y:g})"
            text_item = QGraphicsTextItem(label)
            text_item.setDefaultTextColor(color)
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            text_item.setFont(font)
            rect = text_item.boundingRect()
            text_item.setPos(x - rect.width() / 2, y - rect.height() / 2)
            self._scene.addItem(text_item)

            if depth >= max_depth:
                return

            # Für 45°: |dx| == |dy| pro Kante
            step = base_step * (2 ** (max_depth - depth - 1))

            def connect_child(child: KDNode, is_left: bool) -> None:
                child_x = x - step if is_left else x + step
                child_y = y + step

                # Start/Ende an Kreisrand setzen (bei 45° ist die Normierung konstant)
                inv_sqrt2 = 0.7071067811865476
                ux = (-inv_sqrt2) if is_left else inv_sqrt2
                uy = inv_sqrt2
                start_x = x + ux * node_radius
                start_y = y + uy * node_radius
                end_x = child_x - ux * node_radius
                end_y = child_y - uy * node_radius

                line = QGraphicsLineItem(start_x, start_y, end_x, end_y)
                line.setPen(edge_pen)
                self._scene.addItem(line)

                rel_text = QGraphicsTextItem(edge_label(node.axis, is_left))
                rel_text.setDefaultTextColor(color)
                rel_text.setFont(QFont("", 8))
                mx = (start_x + end_x) / 2
                my = (start_y + end_y) / 2
                rel_text.setPos(mx - 28, my - 18)
                self._scene.addItem(rel_text)

                add_node(child, depth + 1, child_x, child_y)

            if node.left is not None:
                connect_child(node.left, True)
            if node.right is not None:
                connect_child(node.right, False)

        add_node(root, 0, 0.0, 0.0)

        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-40, -40, 40, 40))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


# ============================================================
# GUI
# ============================================================

DEFAULT_POINTS = """A = (2,3)
B = (5,4)
C = (9,6)
D = (4,7)
E = (8,1)
F = (7,2)
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("kd-Tree Demo")
        self.resize(1400, 850)

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        # Linke Seite
        left_layout = QVBoxLayout()
        input_label = QLabel("Punkteingabe")
        input_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        info = QLabel(
            "Format pro Zeile: Name = (x,y)\n"
            "Beispiel: A = (2,3)\n\n"
            "Der Baum wird mit Untermedian aufgebaut."
        )
        info.setWordWrap(True)

        self.input_edit = QTextEdit()
        self.input_edit.setPlainText(DEFAULT_POINTS)

        self.build_button = QPushButton("kd-Tree berechnen")
        self.build_button.clicked.connect(self.rebuild)

        left_layout.addWidget(input_label)
        left_layout.addWidget(info)
        left_layout.addWidget(self.input_edit, 1)
        left_layout.addWidget(self.build_button)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Rechte Seite: Splitter, damit Plot vs. Baum frei skalierbar sind
        plot_label = QLabel("Visualisierung im Raum")
        plot_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.plot_canvas = KDPlotCanvas()

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(plot_label)
        plot_layout.addWidget(self.plot_canvas)

        tree_label = QLabel("Baumstruktur")
        tree_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.tree_view = KDTreeGraphicsView()

        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.addWidget(tree_label)
        tree_layout.addWidget(self.tree_view)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(plot_widget)
        right_splitter.addWidget(tree_widget)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 3)

        # Gesamt-Splitter: links Eingabe, rechts Visualisierung
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        root_layout.addWidget(main_splitter)

        self.rebuild()

    def rebuild(self) -> None:
        try:
            points = parse_points(self.input_edit.toPlainText())
            root = build_kdtree(points)

            self.plot_canvas.draw_kdtree(points, root)
            self.tree_view.draw_tree(root)

        except Exception as exc:
            QMessageBox.critical(self, "Fehler", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()