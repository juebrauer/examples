#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interaktive PySide6-Demo für Forward- und Reverse-Mode AutoDiff.

    f(x, y) = (x*y) * (x**2 + y**2)

Start: pip install PySide6
       python autodiff_pyside6_demo.py

Keine weiteren Abhängigkeiten. Pfeiltasten = Schritt vor/zurück,
Leertaste = automatische Wiedergabe starten/stoppen.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFont, QKeySequence, QLinearGradient, QPainter, QPainterPath,
    QPen, QPolygonF, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QDoubleSpinBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPushButton, QProgressBar, QScrollArea,
    QSlider, QStyle, QStyleOptionSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# ----------------------------------- Modell ----------------------------------

SUBS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def w(name: str) -> str:
    return "w" + name[1:].translate(SUBS)


def bar(name: str) -> str:
    return "w\u0304" + name[1:].translate(SUBS)


def fmt(number: float | None) -> str:
    if number is None:
        return "—"
    if abs(number) < 1e-12:
        return "0"
    return f"{number:.6g}"


@dataclass
class Node:
    """Rechenknoten: Vorgänger, Wert und Adjungierte (Reverse Mode)."""
    name: str
    operation: str
    inputs: tuple[str, ...]
    expression: str
    position: tuple[float, float]
    value: float | None = None
    bar: float = 0.0
    forward_done: bool = False
    backward_done: bool = False
    received_bar: bool = False


@dataclass
class StepInfo:
    phase: str = "start"
    name: str = ""
    title: str = "Bereit für die Berechnung"
    description: str = (
        "Starte mit „Weiter“. Zuerst werden die Knotenwerte von oben nach unten "
        "berechnet. Anschließend läuft Reverse Mode von unten nach oben."
    )
    formula: str = "f(x, y) = (x · y) · (x² + y²)"
    details: list[str] = field(default_factory=list)
    active_edges: set[tuple[str, str]] = field(default_factory=set)


class AutoDiffGraph:
    """Kleiner, absichtlich expliziter Rechengraph für den Unterricht.

    Die Anzeige einer beliebigen Schrittzahl wird durch deterministisches
    Wiederholen der vorherigen Schritte erzeugt. Dadurch funktioniert auch
    'Zurück' korrekt, ohne Zwischenstände manuell zurückzurollen.
    """

    order = ("w1", "w2", "w3", "w4", "w5", "w6", "w7")
    backward_order = tuple(reversed(order))
    count = 14

    def __init__(self, x: float = 4.0, y: float = 5.0):
        self.x = x
        self.y = y
        self.nodes: dict[str, Node] = {}
        self.step = 0
        self.current = StepInfo()
        self.replay(0)

    def replay(self, step: int) -> None:
        self.step = max(0, min(step, self.count))
        self.nodes = {
            # Kopfbereich mit Funktionsformel oberhalb der Eingabeknoten freihalten.
            "w1": Node("w1", "input", (), "x", (185, 150), value=self.x),
            "w2": Node("w2", "input", (), "y", (695, 150), value=self.y),
            "w3": Node("w3", "square", ("w1",), "x²", (145, 289)),
            "w4": Node("w4", "square", ("w2",), "y²", (720, 289)),
            "w5": Node("w5", "mult", ("w1", "w2"), "x · y", (455, 309)),
            "w6": Node("w6", "add", ("w3", "w4"), "x² + y²", (274, 492)),
            "w7": Node("w7", "mult", ("w6", "w5"), "f(x, y)", (472, 644)),
        }
        self.current = StepInfo()
        for k in range(1, self.step + 1):
            self.current = (
                self._forward(self.order[k - 1])
                if k <= 7 else self._backward(self.backward_order[k - 8])
            )

    @staticmethod
    def local(name: str, values: list[float]) -> tuple[float, list[float]]:
        """Liefert (Funktionswert, lokale partielle Ableitungen)."""
        if name == "add":
            a, b = values
            return a + b, [1.0, 1.0]
        if name == "mult":
            a, b = values
            return a * b, [b, a]
        if name == "square":
            (a,) = values
            return a * a, [2.0 * a]
        raise ValueError(f"Unbekannte Operation: {name}")

    def _forward(self, key: str) -> StepInfo:
        node = self.nodes[key]
        node.forward_done = True
        edges = {(a, key) for a in node.inputs}
        if node.operation == "input":
            value = fmt(node.value)
            return StepInfo(
                phase="forward", name=key, title=f"Eingabe {w(key)} setzen",
                description="Eingabeknoten besitzen einen vorgegebenen Wert. Hier ist keine Rechenoperation nötig.",
                formula=f"{w(key)} = {node.expression} = {value}",
                details=["Noch keine Ableitungen: Zuerst wird der normale Funktionswert berechnet."],
                active_edges=edges,
            )

        values = [self.nodes[a].value for a in node.inputs]
        assert all(v is not None for v in values)
        value, _ = self.local(node.operation, values)  # type: ignore[arg-type]
        node.value = value
        ins = [fmt(v) for v in values]
        if node.operation == "square":
            formula = f"{w(key)} = {w(node.inputs[0])}² = {ins[0]}² = {fmt(value)}"
            note = "Quadrieren: Der Knoten liest den Wert seines einzigen Vorgängers."
        elif node.operation == "add":
            formula = (f"{w(key)} = {w(node.inputs[0])} + {w(node.inputs[1])} "
                       f"= {ins[0]} + {ins[1]} = {fmt(value)}")
            note = "Addition: Die beiden Eingangswerte werden addiert."
        else:
            formula = (f"{w(key)} = {w(node.inputs[0])} · {w(node.inputs[1])} "
                       f"= {ins[0]} · {ins[1]} = {fmt(value)}")
            note = "Multiplikation: Die beiden Eingangswerte werden multipliziert."
        return StepInfo(
            phase="forward", name=key, title=f"{w(key)} berechnen",
            description=note, formula=formula,
            details=[f"Vorgängerwerte: {', '.join(w(a) + ' = ' + fmt(self.nodes[a].value) for a in node.inputs)}",
                     f"Neuer Wert: {w(key)} = {fmt(value)}"],
            active_edges=edges,
        )

    def _backward(self, key: str) -> StepInfo:
        node = self.nodes[key]
        if key == "w7":
            node.bar = 1.0  # Saatvariable d(output)/d(output) = 1
            node.received_bar = True

        current_bar = node.bar
        node.backward_done = True
        if node.operation == "input":
            var = "x" if key == "w1" else "y"
            return StepInfo(
                phase="reverse", name=key, title=f"Gradient nach {var} ist fertig",
                description=("Der Eingabeknoten hat keine Vorgänger. Alle Beiträge "
                             "seiner Nachfolger sind bereits aufsummiert."),
                formula=f"∂f/∂{var} = {bar(key)} = {fmt(node.bar)}",
                details=["Die Adjungierte des Eingabeknotens ist die gesuchte partielle Ableitung."],
            )

        values = [self.nodes[a].value for a in node.inputs]
        assert all(v is not None for v in values)
        _, partials = self.local(node.operation, values)  # type: ignore[arg-type]
        lines: list[str] = []
        for input_name, local_deriv in zip(node.inputs, partials):
            target = self.nodes[input_name]
            before = target.bar
            contribution = current_bar * local_deriv
            target.bar += contribution  # Kern des Reverse Mode: Beiträge ADDIEREN!
            target.received_bar = True
            lines.append(
                f"{bar(input_name)} += {bar(key)} · ∂{w(key)}/∂{w(input_name)}"
                f" = {fmt(current_bar)} · {fmt(local_deriv)} = {fmt(contribution)}"
                f"     ({fmt(before)} → {fmt(target.bar)})"
            )
        if key == "w7":
            title = "Rückwärtspass starten: Saat = 1"
            description = ("Am Ausgang wird die Adjungierte auf 1 gesetzt, denn "
                           "∂w₇/∂w₇ = 1. Danach wird sie per Kettenregel auf "
                           "die direkten Vorgänger verteilt.")
        else:
            title = f"Adjungierte von {w(key)} weitergeben"
            description = (
                f"Die bereits angesammelte Adjungierte {bar(key)} = {fmt(current_bar)} "
                "wird mit jeder lokalen partiellen Ableitung multipliziert. "
                "Die Beiträge werden bei den Vorgängern addiert (+=)."
            )
        if node.operation == "square":
            rule = f"Lokal: ∂{w(key)}/∂{w(node.inputs[0])} = 2 · {fmt(values[0])} = {fmt(partials[0])}"
        elif node.operation == "add":
            rule = "Lokal: Ableitungen der Addition = (1, 1)"
        else:
            rule = f"Lokal: Multiplikation ⇒ ({fmt(partials[0])}, {fmt(partials[1])})"
        return StepInfo(
            phase="reverse", name=key, title=title, description=description,
            formula=f"{bar(key)} = {fmt(current_bar)}     |     {rule}",
            details=lines, active_edges={(a, key) for a in node.inputs},
        )

    def gradient(self) -> tuple[float | None, float | None]:
        return (
            self.nodes["w1"].bar if self.nodes["w1"].backward_done else None,
            self.nodes["w2"].bar if self.nodes["w2"].backward_done else None,
        )


# --------------------------------- Oberfläche --------------------------------
BG = "#080F1C"
CARD = "#121E30"
TEXT = "#F0F6FF"
MUTED = "#98ACC8"
CYAN = "#4BD7E8"
AMBER = "#F8BB55"
VIOLET = "#B59CF8"


class GraphCanvas(QWidget):
    node_clicked = Signal(str)
    WIDTH = 870.0
    HEIGHT = 744.0
    NW = 174.0
    NH = 100.0

    def __init__(self, engine: AutoDiffGraph):
        super().__init__()
        self.engine = engine
        self.selected: str | None = None
        self.setMinimumSize(540, 490)
        self.setMouseTracking(True)
        self.setToolTip("Knoten anklicken, um seine Bedeutung im rechten Bereich zu sehen")
        self._transform: tuple[float, float, float] = (1, 0, 0)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QLinearGradient(0, 0, self.width(), self.height())
        bg.setColorAt(0, QColor("#101F33"))
        bg.setColorAt(1, QColor("#0B1627"))
        p.fillRect(self.rect(), bg)
        s = min((self.width() - 18) / self.WIDTH,
                (self.height() - 18) / self.HEIGHT)
        tx = (self.width() - self.WIDTH * s) / 2
        ty = (self.height() - self.HEIGHT * s) / 2
        self._transform = (s, tx, ty)
        p.translate(tx, ty)
        p.scale(s, s)

        # Dezentes Punktraster
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#253650"))
        for x in range(24, 870, 26):
            for y in range(28, 735, 26):
                p.drawEllipse(QPointF(float(x), float(y)), 1.0, 1.0)

        self._text(p, 35, 34, "BERECHNUNGSGRAPH", 12, MUTED, bold=True)
        self._text(p, 35, 61, "f(x,y) = (x·y) · (x²+y²)", 17, TEXT, bold=True)
        phase = self.engine.current.phase
        tag = "FORWARD" if phase == "forward" else "REVERSE" if phase == "reverse" else "START"
        accent = CYAN if phase == "forward" else AMBER if phase == "reverse" else MUTED
        self._text(p, 735, 52, tag, 12, accent, bold=True)

        # Graphkanten hinter die Knoten zeichnen
        for a, b in (("w1", "w3"), ("w2", "w4"), ("w1", "w5"),
                     ("w2", "w5"), ("w3", "w6"), ("w4", "w6"),
                     ("w6", "w7"), ("w5", "w7")):
            self._edge(p, a, b)

        for name in self.engine.order:
            self._node(p, self.engine.nodes[name])

        self._text(p, 36, 720,
                   "Knoten anklicken   •   Türkis: Werte ↓   •   Gold: Adjungierte ↑",
                   12, MUTED)
        p.end()

    def _text(self, p: QPainter, x: float, y: float, content: str,
              size: int, color: str, bold: bool = False) -> None:
        p.setFont(QFont("DejaVu Sans", size, QFont.Weight.DemiBold if bold else QFont.Weight.Normal))
        p.setPen(QColor(color))
        p.drawText(QPointF(x, y), content)

    def _edge(self, p: QPainter, source: str, target: str) -> None:
        src = self.engine.nodes[source]
        dst = self.engine.nodes[target]
        sx, sy = src.position
        dx, dy = dst.position
        sy += self.NH / 2 - 3
        dy -= self.NH / 2 - 3
        # Zwei eigene Anschlussports am Ausgabeknoten
        if target == "w7":
            dx += -40 if source == "w6" else 40
        if target == "w5":
            dx += -29 if source == "w1" else 29
        if target == "w6":
            dx += -30 if source == "w3" else 30
        path = QPainterPath(QPointF(sx, sy))
        dist = max(28, abs(dy - sy) * 0.45)
        path.cubicTo(QPointF(sx, sy + dist), QPointF(dx, dy - dist), QPointF(dx, dy))

        info = self.engine.current
        highlighted = (source, target) in info.active_edges
        if highlighted:
            color = QColor(CYAN if info.phase == "forward" else AMBER)
        elif info.phase == "reverse" and src.backward_done:
            color = QColor("#7A6241")
        elif src.forward_done and dst.forward_done:
            color = QColor("#356E7B")
        else:
            color = QColor("#344359")
        p.setPen(QPen(color, 4.5 if highlighted else 2.5,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Ein hervorgehobener Reverse-Schritt zeigt den Pfeil absichtlich
        # GEGEN die Rechengraph-Richtung: vom Knoten zu seinen Vorgängern.
        if highlighted and info.phase == "reverse":
            tip = path.pointAtPercent(0)
            near = path.pointAtPercent(0.025)
        else:
            tip = path.pointAtPercent(1)
            near = path.pointAtPercent(0.975)
        angle = math.atan2(tip.y() - near.y(), tip.x() - near.x())
        length = 12.5 if highlighted else 9.0
        left = QPointF(tip.x() - length * math.cos(angle - 0.44),
                       tip.y() - length * math.sin(angle - 0.44))
        right = QPointF(tip.x() - length * math.cos(angle + 0.44),
                        tip.y() - length * math.sin(angle + 0.44))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(QPolygonF([tip, left, right]))

    def _node(self, p: QPainter, node: Node) -> None:
        x, y = node.position
        rect = QRectF(x - self.NW / 2, y - self.NH / 2, self.NW, self.NH)
        active = self.engine.current.name == node.name
        picked = self.selected == node.name
        reverse = self.engine.current.phase == "reverse"
        accent = AMBER if reverse else CYAN
        border = accent if active else VIOLET if picked else "#354961"
        fill = QColor("#26364A" if active else "#18273C")

        if active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(248, 187, 85, 33) if reverse else QColor(75, 215, 232, 36))
            p.drawRoundedRect(rect.adjusted(-7, -7, 7, 7), 19, 19)
        p.setBrush(fill)
        p.setPen(QPen(QColor(border), 2.6 if active else 1.5))
        p.drawRoundedRect(rect, 14, 14)

        if node.operation == "input":
            kind = "EINGABE"
        elif node.operation == "square":
            kind = "QUADRAT"
        elif node.operation == "add":
            kind = "ADDITION"
        else:
            kind = "PRODUKT"
        self._text(p, rect.left() + 12, rect.top() + 19, kind, 9, MUTED, bold=True)
        self._text(p, rect.left() + 12, rect.top() + 44,
                   f"{w(node.name)}  {node.expression}", 15, TEXT, bold=True)
        val_color = CYAN if node.forward_done else MUTED
        self._text(p, rect.left() + 12, rect.top() + 68,
                   f"Wert  {fmt(node.value)}", 11, val_color)
        if reverse:
            bar_repr = fmt(node.bar) if node.received_bar else "0"
            suffix = " ✓" if node.backward_done else " +" if node.received_bar else ""
            self._text(p, rect.left() + 12, rect.top() + 88,
                       f"Adj.  {bar_repr}{suffix}", 11, AMBER if node.received_bar else MUTED)
        else:
            self._text(p, rect.left() + 12, rect.top() + 88,
                       "Adj.  —", 11, MUTED)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            s, tx, ty = self._transform
            x = (event.position().x() - tx) / s
            y = (event.position().y() - ty) / s
            for key, node in self.engine.nodes.items():
                nx, ny = node.position
                if abs(x - nx) <= self.NW / 2 and abs(y - ny) <= self.NH / 2:
                    self.selected = key
                    self.node_clicked.emit(key)
                    self.update()
                    return
        super().mousePressEvent(event)


class InputSpinBox(QDoubleSpinBox):
    """Kontrastreiche Pfeile unabhängig vom nativen Desktop-Theme."""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for control, direction, step in (
            (QStyle.SubControl.SC_SpinBoxUp, -1, self.StepEnabledFlag.StepUpEnabled),
            (QStyle.SubControl.SC_SpinBoxDown, 1, self.StepEnabledFlag.StepDownEnabled),
        ):
            rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, option, control, self)
            center = QRectF(rect).center()
            painter.setBrush(QColor(TEXT if self.isEnabled() and self.stepEnabled() & step else MUTED))
            painter.drawPolygon(QPolygonF([
                QPointF(center.x() - 4, center.y() - direction * 2),
                QPointF(center.x() + 4, center.y() - direction * 2),
                QPointF(center.x(), center.y() + direction * 3),
            ]))
        painter.end()


STYLESHEET = """
QMainWindow, QWidget#root { background: #080F1C; color: #F0F6FF; }
QLabel { color: #F0F6FF; }
QFrame#panel { background: #121E30; border: 1px solid #263B52; border-radius: 17px; }
QFrame#subpanel { background: #18273B; border: 1px solid #2B4059; border-radius: 12px; }
QPushButton { background: #203249; color: #F0F6FF; border: 1px solid #3A526D;
              padding: 11px 17px; border-radius: 10px; font-weight: 600; }
QPushButton:hover { background: #2A4260; border-color: #6D92B2; }
QPushButton:disabled { background: #172334; color: #5B6E85; border-color: #213247; }
QPushButton#primary { background: #19AEC4; border-color: #46DDEA; color: #051621; }
QPushButton#primary:hover { background: #50D7E7; }
QPushButton#play { background: #745A2E; border-color: #C79C4D; }
QDoubleSpinBox { background: #17273B; border: 1px solid #3D536C; color: #F0F6FF;
                 border-radius: 8px; padding: 6px 26px 6px 8px; font-size: 13px; }
QDoubleSpinBox:focus { border: 1px solid #4BD7E8; }
QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
                          width: 22px; background: #2D415B; border-top-right-radius: 8px; }
QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
                            width: 22px; background: #2D415B; border-bottom-right-radius: 8px; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #3D5875; }
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed { background: #167E92; }
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { image: none; }
QTableWidget { background: #142237; alternate-background-color: #17283D;
               border: 1px solid #2B425C; color: #DFEDFF; gridline-color: #253851;
               selection-background-color: #24485C; font-size: 12px; }
QHeaderView::section { background: #21324B; color: #B7D0E8;
                       padding: 7px 5px; border: none; }
QProgressBar { border: 0; border-radius: 5px; background: #253851; height: 9px; }
QProgressBar::chunk { background: #4BD7E8; border-radius: 5px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #142236; width: 8px; }
QScrollBar::handle:vertical { background: #3C5471; border-radius: 4px; }
QSlider::groove:horizontal { height: 5px; background: #2D415B; border-radius: 2px; }
QSlider::handle:horizontal { background: #F8BB55; width: 14px; border-radius: 7px; margin: -5px 0; }
"""


def label(text: str, font_size: int = 12, color: str = TEXT,
          bold: bool = False, wrap: bool = False) -> QLabel:
    result = QLabel(text)
    result.setWordWrap(wrap)
    result.setStyleSheet(f"color: {color}; font-size: {font_size}px;"
                         + ("font-weight: 700;" if bold else ""))
    return result


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AutoDiffGraph()
        self.selected: str | None = None
        self.setWindowTitle("AutoDiff Studio · Forward & Reverse Mode")
        self.resize(1530, 900)
        self.setMinimumSize(1120, 740)
        self.setStyleSheet(STYLESHEET)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self._setup_ui()
        self._setup_shortcuts()
        self.refresh()

    def _setup_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(label("LEHRDEMO  /  AUTOMATIC DIFFERENTIATION", 11, CYAN, True))
        names.addWidget(label("Forward & Reverse Mode", 27, TEXT, True))
        header.addLayout(names, 1)
        formula = label("f(x, y) = (x·y) · (x² + y²)", 20, "#BFD5F0", True)
        header.addWidget(formula)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(15)
        self.canvas = GraphCanvas(self.engine)
        self.canvas.node_clicked.connect(self.select_node)
        graph_frame = QFrame()
        graph_frame.setObjectName("panel")
        gl = QVBoxLayout(graph_frame)
        gl.setContentsMargins(3, 3, 3, 3)
        gl.addWidget(self.canvas)
        body.addWidget(graph_frame, 6)

        side = QFrame()
        side.setObjectName("panel")
        side.setMinimumWidth(480)
        side.setMaximumWidth(760)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 18, 16, 16)
        side_layout.setSpacing(11)
        self.stage = label("START", 16, CYAN, True)
        side_layout.addWidget(self.stage)
        self.title = label("Bereit für die Berechnung", 28, TEXT, True, True)
        side_layout.addWidget(self.title)
        self.desc = label("", 21, "#C3D3E8", wrap=True)
        side_layout.addWidget(self.desc)

        formula_box = QFrame()
        formula_box.setObjectName("subpanel")
        fb = QVBoxLayout(formula_box)
        fb.setContentsMargins(12, 12, 12, 12)
        fb.setSpacing(7)
        fb.addWidget(label("LOKALE BERECHNUNG", 15, MUTED, True))
        self.calculation = label("", 22, TEXT, True, True)
        self.calculation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fb.addWidget(self.calculation)
        side_layout.addWidget(formula_box)

        side_layout.addWidget(label("WAS PASSIERT IN DIESEM SCHRITT?", 15, MUTED, True))
        self.details = label("", 21, "#D0E0F3", wrap=True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        side_layout.addWidget(self.details)
        self.focus_note = label("", 17, VIOLET, wrap=True)
        side_layout.addWidget(self.focus_note)

        side_layout.addWidget(label("WERTE UND ADJUNGIERTE", 15, MUTED, True))
        self.table = QTableWidget(7, 4)
        self.table.setHorizontalHeaderLabels(["Knoten", "Operation", "Wert", "Adj. bar"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget, QHeaderView::section { font-size: 20px; }")
        self.table.setFixedHeight(320)
        self.table.verticalHeader().setDefaultSectionSize(37)
        side_layout.addWidget(self.table)

        self.summary = label("", 21, CYAN, True, True)
        side_layout.addWidget(self.summary)
        side_layout.addStretch(1)
        # Die Erklärung bleibt auch auf kleineren Notebook-/Beamer-Displays
        # zugänglich: Der rechte Bereich wird bei Bedarf vertikal scrollbar.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(485)
        scroll.setMaximumWidth(765)
        scroll.setWidget(side)
        body.addWidget(scroll, 5)
        layout.addLayout(body, 1)

        # Bedienleiste
        controls = QFrame()
        controls.setObjectName("panel")
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(11)
        cl.addWidget(label("x", 14, MUTED, True))
        self.xspin = InputSpinBox()
        self.xspin.setRange(-100000.0, 100000.0)
        self.xspin.setDecimals(2)
        self.xspin.setValue(4)
        self.xspin.setFixedWidth(96)
        cl.addWidget(self.xspin)
        cl.addWidget(label("y", 14, MUTED, True))
        self.yspin = InputSpinBox()
        self.yspin.setRange(-100000.0, 100000.0)
        self.yspin.setDecimals(2)
        self.yspin.setValue(5)
        self.yspin.setFixedWidth(96)
        cl.addWidget(self.yspin)
        self.xspin.valueChanged.connect(self.change_inputs)
        self.yspin.valueChanged.connect(self.change_inputs)

        self.reset_btn = QPushButton("↺  Neustart")
        self.reset_btn.clicked.connect(self.reset)
        cl.addWidget(self.reset_btn)
        self.prev_btn = QPushButton("←  Zurück")
        self.prev_btn.clicked.connect(self.prev_step)
        cl.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Weiter  →")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self.next_step)
        cl.addWidget(self.next_btn)
        self.play_btn = QPushButton("▶  Auto")
        self.play_btn.setObjectName("play")
        self.play_btn.clicked.connect(self.toggle_play)
        cl.addWidget(self.play_btn)
        cl.addWidget(label("Tempo", 11, MUTED))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 5)
        self.speed.setValue(3)
        self.speed.setFixedWidth(80)
        self.speed.valueChanged.connect(self.change_speed)
        cl.addWidget(self.speed)
        self.progress_label = label("0 / 14", 11, MUTED, True)
        cl.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, self.engine.count)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(88)
        cl.addWidget(self.progress)
        layout.addWidget(controls)

    def _setup_shortcuts(self) -> None:
        for key, callback in (("Right", self.next_step), ("Left", self.prev_step),
                              ("Space", self.toggle_play), ("Home", self.reset),
                              ("End", self.finish)):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(callback)

    def stop_play(self) -> None:
        self.timer.stop()
        self.play_btn.setText("▶  Auto")

    def change_inputs(self) -> None:
        self.stop_play()
        self.engine.x = self.xspin.value()
        self.engine.y = self.yspin.value()
        self.engine.replay(0)
        self.refresh()

    def reset(self) -> None:
        self.stop_play()
        self.engine.replay(0)
        self.refresh()

    def prev_step(self) -> None:
        self.stop_play()
        self.engine.replay(self.engine.step - 1)
        self.refresh()

    def next_step(self) -> None:
        if self.engine.step >= self.engine.count:
            self.stop_play()
            return
        self.engine.replay(self.engine.step + 1)
        self.refresh()
        if self.engine.step == self.engine.count:
            self.stop_play()

    def finish(self) -> None:
        self.stop_play()
        self.engine.replay(self.engine.count)
        self.refresh()

    def change_speed(self) -> None:
        if self.timer.isActive():
            self.timer.setInterval(self._interval())

    def _interval(self) -> int:
        return {1: 2200, 2: 1600, 3: 1100, 4: 720, 5: 400}[self.speed.value()]

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.stop_play()
        else:
            if self.engine.step >= self.engine.count:
                self.engine.replay(0)
                self.refresh()
            self.play_btn.setText("Ⅱ  Pause")
            self.timer.start(self._interval())

    def select_node(self, key: str) -> None:
        self.selected = key
        self.refresh()

    def refresh(self) -> None:
        current = self.engine.current
        step = self.engine.step
        if step == 0:
            phase_name, accent = "START · BEREIT", CYAN
        elif step <= 7:
            phase_name, accent = f"VORWÄRTSPASS  ·  SCHRITT {step} VON 7", CYAN
        else:
            phase_name, accent = f"RÜCKWÄRTSPASS  ·  SCHRITT {step - 7} VON 7", AMBER
        self.stage.setText(phase_name)
        self.stage.setStyleSheet(f"color: {accent}; font-size: 16px; font-weight: 700;")
        self.title.setText(current.title)
        self.desc.setText(current.description)
        self.calculation.setText(current.formula)
        details = current.details or ["Ein Schritt nach dem anderen: Beginne mit dem ersten Eingabeknoten."]
        self.details.setText("\n".join("•  " + part for part in details))
        if self.selected:
            node = self.engine.nodes[self.selected]
            refs = ", ".join(w(a) for a in node.inputs) or "keine"
            adj = fmt(node.bar) if step >= 8 else "noch nicht begonnen"
            self.focus_note.setText(
                f"Ausgewählt: {w(self.selected)}  ·  Operation: {node.operation}  ·  "
                f"Vorgänger: {refs}  ·  Adjungierte: {adj}"
            )
        else:
            self.focus_note.setText("Tipp: Du kannst jeden Knoten im Graphen anklicken.")
        for row, key in enumerate(self.engine.order):
            node = self.engine.nodes[key]
            op_name = {"input": "Eingabe", "square": "Quadrat",
                       "mult": "Produkt", "add": "Summe"}[node.operation]
            vals = [w(key), op_name,
                    fmt(node.value) if (node.forward_done or node.operation == "input") else "—",
                    fmt(node.bar) if step >= 8 else "—"]
            for col, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter |
                                      (Qt.AlignmentFlag.AlignRight if col >= 2 else Qt.AlignmentFlag.AlignLeft))
                if current.name == key:
                    item.setBackground(QColor("#21455B" if step <= 7 else "#4D3A27"))
                    item.setForeground(QColor(CYAN if step <= 7 else AMBER))
                elif node.backward_done:
                    item.setForeground(QColor("#F2CC88"))
                self.table.setItem(row, col, item)
        fval = self.engine.nodes["w7"].value
        dx, dy = self.engine.gradient()
        if step == 14:
            x, y = self.engine.x, self.engine.y
            expected_dx = 3 * x * x * y + y ** 3
            expected_dy = x ** 3 + 3 * x * y * y
            ok = (math.isclose(dx, expected_dx, rel_tol=1e-9, abs_tol=1e-8)
                  and math.isclose(dy, expected_dy, rel_tol=1e-9, abs_tol=1e-8))
            self.summary.setText(
                f"ERGEBNIS  ·  f = {fmt(fval)}\n"
                f"∇f = ({fmt(dx)}, {fmt(dy)})   {'✓ geprüft' if ok else '⚠ prüfen'}"
            )
        elif step >= 7:
            self.summary.setText(
                f"f = {fmt(fval)}     ∂f/∂x = {fmt(dx)}     ∂f/∂y = {fmt(dy)}"
            )
        else:
            self.summary.setText("f = " + fmt(fval) if fval is not None else
                                 "Ziel: f(4,5)=820, ∇f(4,5)=(365,364)")
        self.progress.setValue(step)
        self.progress_label.setText(f"{step} / {self.engine.count}")
        self.prev_btn.setEnabled(step > 0)
        self.next_btn.setEnabled(step < self.engine.count)
        self.canvas.update()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AutoDiff Studio")
    app.setStyle("Fusion")
    app.setFont(QFont("DejaVu Sans", 10))
    window = DemoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
