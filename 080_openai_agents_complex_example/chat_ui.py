"""PySide6-Ansicht. Kennt keine Tools, Prompts oder OpenAI-Ereignistypen.

Ein QThread übersetzt lediglich zwischen Qt-Signalen und dem asynchronen Backend.
Alle Widgets und Zeichenoperationen bleiben im Hauptthread.
"""

import asyncio
import json
from pathlib import Path
import sys
from threading import Event

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from agent_backend import MODEL, RealtimeService
from audio_io import audio_devices
from bestelldaten import ROOT

STYLE = """
QWidget { background: #000; color: #fff; font-family: 'Courier New', 'Liberation Mono', monospace; }
QMainWindow, QScrollArea { border: none; }
QLabel { background: transparent; }
QPushButton, QComboBox, QLineEdit {
    background: #111; border: 1px solid #444; border-radius: 6px; padding: 9px 12px;
}
QPushButton:hover { border-color: #ddd; background: #222; }
QPushButton:checked { background: #eee; color: #000; }
QPushButton:disabled, QLineEdit:disabled { color: #666; border-color: #222; }
QLineEdit:focus { border-color: #fff; }
QScrollBar:vertical { background: #080808; width: 10px; }
QScrollBar::handle:vertical { background: #444; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#assistant { background: #111; border: 1px solid #333; border-radius: 12px; }
QFrame#user { background: #202020; border: 1px solid #555; border-radius: 12px; }
QFrame#tool { background: #171400; border: 1px solid #6b5916; border-radius: 8px; }
QFrame#tool QLabel { color: #ffdc62; }
QFrame#chart { background: #111; border: 1px solid #333; border-radius: 12px; }
"""


class AgentWorker(QThread):
    """Technische Brücke: keine Agentenentscheidungen und keine Widget-Zugriffe."""

    event = Signal(dict)

    def __init__(self, input_device=None, output_device=None, parent=None):
        super().__init__(parent)
        self.input_device = input_device
        self.output_device = output_device
        self.loop = None
        self.task = None
        self.service = None
        self.stop_requested = Event()

    def run(self):
        try:
            asyncio.run(self._run())
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.event.emit({"type": "error", "text": str(error)})
        finally:
            self.event.emit({"type": "disconnected"})

    async def _run(self):
        self.loop = asyncio.get_running_loop()
        self.service = RealtimeService(self.event.emit, self.input_device, self.output_device)
        self.task = asyncio.current_task()
        if self.stop_requested.is_set():
            return
        await self.service.run()

    def command(self, command, value):
        if self.loop is not None and self.service is not None:
            try:
                self.loop.call_soon_threadsafe(self.service.commands.put_nowait, (command, value))
            except RuntimeError:
                pass  # Sitzung wurde inzwischen geschlossen.

    def stop(self):
        if self.stop_requested.is_set():
            return  # Laufende Aufräumarbeiten nicht durch erneutes Cancel abbrechen.
        self.stop_requested.set()
        if self.loop is not None and self.task is not None:
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:
                pass


class TypewriterLabel(QLabel):
    """Auch komplette Spracherkennungsresultate werden Zeichen für Zeichen sichtbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.target = ""
        self.visible_count = 0
        self.final = False
        self.blink = 0
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.tick)

    def update_text(self, text, final=False):
        # Spracherkennung kann frühere Wörter korrigieren.
        common = 0
        for old, new in zip(self.target, text):
            if old != new:
                break
            common += 1
        self.visible_count = min(self.visible_count, common)
        self.target = text
        self.final = final
        self.timer.start()
        self.tick()

    def tick(self):
        self.blink += 1
        pending = len(self.target) - self.visible_count
        if pending:
            self.visible_count += min(pending, max(2, min(12, pending // 15)))
        done = self.final and self.visible_count == len(self.target)
        cursor = "" if done else ("█" if (self.blink // 20) % 2 == 0 else " ")
        self.setText(self.target[:self.visible_count] + cursor)
        if done:
            self.timer.stop()


class MessageBubble(QFrame):
    def __init__(self, role, parent=None):
        super().__init__(parent)
        self.setObjectName(role)
        self.role = role
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 16)
        self.caption = QLabel({"user": "DU", "assistant": "KI", "tool": "TOOL / FUNCTION"}[role])
        self.caption.setStyleSheet("font-size: 22px; color: " + ("#ffdc62" if role == "tool" else "#aaa"))
        layout.addWidget(self.caption)
        self.text = TypewriterLabel()
        self.text.setStyleSheet("font-size: 32px;")
        layout.addWidget(self.text)


class ClickableImage(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class ChartCard(QFrame):
    def __init__(self, png, filename, title, parent=None):
        super().__init__(parent)
        self.setObjectName("chart")
        self.png = png
        self.filename = Path(filename).name
        self.original = QPixmap()
        if not self.original.loadFromData(png):
            raise ValueError("Das Diagrammbild konnte nicht geladen werden.")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("KI · " + title))
        body = QHBoxLayout()
        self.image = ClickableImage()
        self.image.setMinimumSize(100, 80)
        self.image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image.setToolTip("Klicken zum Vergrößern und Zoomen")
        self.image.clicked.connect(self.enlarge)
        body.addWidget(self.image, 1)
        buttons = QVBoxLayout()
        self.save_button = QPushButton("In diagramme\nspeichern")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.save_button)
        zoom = QPushButton("Vergrößern")
        zoom.clicked.connect(self.enlarge)
        buttons.addWidget(zoom)
        buttons.addStretch()
        body.addLayout(buttons)
        layout.addLayout(body)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_image()

    def render_image(self):
        width = max(100, self.width() - 190)
        pixmap = self.original.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        self.image.setPixmap(pixmap)
        self.image.setFixedHeight(pixmap.height())

    def save(self):
        try:
            folder = ROOT / "diagramme"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / self.filename
            path.write_bytes(self.png)
        except OSError as error:
            self.save_button.setText("Speichern fehlgeschlagen")
            self.save_button.setToolTip(str(error))
        else:
            self.save_button.setText("Gespeichert ✓")
            self.save_button.setToolTip(str(path))

    def enlarge(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Diagramm · Zoomen und Scrollen")
        dialog.resize(1100, 760)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        image = QLabel()
        scroll.setWidget(image)
        layout.addWidget(scroll, 1)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(25, 200)
        slider.setValue(75)
        def zoom(percent):
            pixmap = self.original.scaledToWidth(
                self.original.width() * percent // 100, Qt.TransformationMode.SmoothTransformation,
            )
            image.setPixmap(pixmap)
            image.resize(pixmap.size())
        slider.valueChanged.connect(zoom)
        layout.addWidget(QLabel("Zoom 25–200 %"))
        layout.addWidget(slider)
        zoom(slider.value())
        dialog.exec()


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bestellagent · Sprache, Daten & Diagramme")
        self.resize(1180, 860)
        self.setMinimumSize(760, 540)
        self.worker = None
        self.closing = False
        self.messages = {}
        self.rows = []
        self.follow_bottom = True
        self.prefix = ""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("BESTELLAGENT")
        title.setStyleSheet("font-size: 44px; font-weight: bold;")
        header.addWidget(title)
        header.addWidget(QLabel(MODEL))
        header.addStretch()
        self.mic_button = QPushButton("Mikrofon an")
        self.mic_button.setCheckable(True)
        self.mic_button.setEnabled(False)
        self.mic_button.toggled.connect(self.toggle_mic)
        header.addWidget(self.mic_button)
        self.connect_button = QPushButton("Gespräch starten")
        self.connect_button.clicked.connect(self.toggle_connection)
        header.addWidget(self.connect_button)
        layout.addLayout(header)
        self.stop_speaking_button = QPushButton("Hör auf zu quatschen!")
        self.stop_speaking_button.setEnabled(False)
        self.stop_speaking_button.clicked.connect(self.stop_speaking)
        layout.addWidget(self.stop_speaking_button)
        devices = QHBoxLayout()
        devices.addWidget(QLabel("Mikrofon"))
        self.input_device = QComboBox()
        devices.addWidget(self.input_device, 1)
        devices.addWidget(QLabel("Lautsprecher"))
        self.output_device = QComboBox()
        devices.addWidget(self.output_device, 1)
        for combo in (self.input_device, self.output_device):
            combo.addItem("Systemstandard", None)
            combo.setMinimumWidth(100)
        try:
            inputs, outputs = audio_devices()
            for combo, entries in ((self.input_device, inputs), (self.output_device, outputs)):
                for index, name in entries:
                    combo.addItem(name, index)
        except Exception:
            pass  # Ein Startversuch zeigt die konkrete Audiofehlermeldung im Verlauf.
        layout.addLayout(devices)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.canvas = QWidget()
        self.conversation = QVBoxLayout(self.canvas)
        self.conversation.setContentsMargins(4, 16, 4, 16)
        self.conversation.setSpacing(18)
        self.conversation.addStretch()
        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll, 1)
        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(self._scroll_changed)
        bar.rangeChanged.connect(self._scroll_range_changed)

        entry = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Alternativ tippen … z. B. Bestellungen pro Bundesland als Diagramm")
        self.input.setEnabled(False)
        self.input.returnPressed.connect(self.send_text)
        entry.addWidget(self.input, 1)
        self.send_button = QPushButton("Senden ↵")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_text)
        entry.addWidget(self.send_button)
        layout.addLayout(entry)
        self.status = QLabel("Bereit · Start aktiviert das Mikrofon. Audio und Werkzeugdaten gehen an OpenAI.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 22px; color: #aaa;")
        layout.addWidget(self.status)
        self.add_message("welcome", "assistant", (
            "Sprich mit deinen Daten.\n\n"
            "Starte das Gespräch und frage zum Beispiel:\n"
            "„Wie viele Bestellungen gibt es pro Bundesland?“\n"
            "„Erstelle dafür ein Balkendiagramm.“\n\n"
            "Ich wähle die passenden Werkzeuge selbst. Ihre Aufrufe erscheinen gelb; "
            "Diagramme kannst du hier ansehen, vergrößern und speichern.\n\n"
            "Während meiner Sprachausgabe pausiert das Mikrofon automatisch. "
            "Stelle deine nächste Frage nach meiner Antwort."
        ), True)

    def _scroll_changed(self, value):
        bar = self.scroll.verticalScrollBar()
        self.follow_bottom = value >= bar.maximum() - 36

    def _scroll_range_changed(self, minimum, maximum):
        if self.follow_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)

    def add_row(self, widget, role):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row_layout.addStretch(1)
        row_layout.addWidget(widget, 4)
        if role != "user":
            row_layout.addStretch(1)
        self.conversation.insertWidget(self.conversation.count() - 1, row)
        self.rows.append((row, widget))

    def add_message(self, item_id, role, text, final=False):
        if item_id not in self.messages:
            bubble = MessageBubble(role)
            self.messages[item_id] = bubble
            self.add_row(bubble, role)
        self.messages[item_id].text.update_text(text, final)

    def toggle_connection(self):
        if self.worker is not None:
            self.connect_button.setEnabled(False)
            self.status.setText("Gespräch wird beendet …")
            self.worker.stop()
            return
        self.prefix = str(len(self.rows)) + ":"
        self.worker = AgentWorker(self.input_device.currentData(), self.output_device.currentData(), self)
        self.worker.event.connect(self.handle_event)
        self.worker.finished.connect(self.worker_finished)
        self.connect_button.setText("Gespräch beenden")
        self.input_device.setEnabled(False)
        self.output_device.setEnabled(False)
        self.status.setText("Verbindung wird aufgebaut …")
        self.worker.start()

    def toggle_mic(self, enabled):
        self.mic_button.setText("Mikrofon an" if enabled else "Mikrofon aus")
        if self.worker:
            self.worker.command("mute", not enabled)

    def send_text(self):
        text = self.input.text().strip()
        if text and self.worker:
            self.worker.command("text", text)
            self.input.clear()

    def stop_speaking(self):
        if self.worker:
            self.worker.command("interrupt", None)

    def handle_event(self, event):
        kind = event["type"]
        if kind == "message":
            self.add_message(self.prefix + event["id"], event["role"], event["text"], event["final"])
        elif kind == "listening":
            self.add_message(self.prefix + event["id"], "user", "")
            self.status.setText("Ich höre zu …")
        elif kind == "interrupted":
            bubble = self.messages.get(self.prefix + event["id"])
            if bubble:
                bubble.caption.setText("KI · UNTERBROCHEN")
                bubble.text.update_text(bubble.text.target, True)
        elif kind == "tool":
            try:
                args = json.loads(event["arguments"])
                args = ", ".join(f"{k}={v!r}" for k, v in args.items())
            except (ValueError, AttributeError):
                args = event["arguments"]
            self.add_message(f"tool-{len(self.rows)}", "tool", f"{event['name']}({args})", True)
            self.status.setText("Werkzeug wird ausgeführt: " + event["name"])
        elif kind == "chart":
            self.add_row(ChartCard(event["image"], event["filename"], event["title"]), "assistant")
        elif kind == "connected":
            self.stop_speaking_button.setEnabled(True)
            self.mic_button.setEnabled(True)
            self.mic_button.setChecked(True)
            self.input.setEnabled(True)
            self.send_button.setEnabled(True)
            self.status.setText("Verbunden · Sprich natürlich. Die KI-Stimme wird künstlich erzeugt.")
        elif kind == "status":
            self.status.setText(event["text"])
        elif kind == "error":
            self.add_message(f"error-{len(self.rows)}", "assistant", "Hinweis: " + event["text"], True)
            self.status.setText("Fehler · Details stehen im Gesprächsverlauf.")
        elif kind == "disconnected":
            self.stop_speaking_button.setEnabled(False)
            self.input.setEnabled(False)
            self.send_button.setEnabled(False)
            self.mic_button.setEnabled(False)

    def worker_finished(self):
        self.stop_speaking_button.setEnabled(False)
        worker = self.worker
        self.worker = None
        if worker:
            worker.deleteLater()
        for bubble in self.messages.values():
            if not bubble.text.final:
                bubble.text.update_text(bubble.text.target, True)
        self.connect_button.setText("Gespräch starten")
        self.connect_button.setEnabled(True)
        self.mic_button.setChecked(False)
        self.input_device.setEnabled(True)
        self.output_device.setEnabled(True)
        self.status.setText("Gespräch beendet · Der Verlauf und die Diagramme bleiben sichtbar.")
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.closing = True
            self.status.setText("Audio und Verbindung werden geschlossen …")
            self.worker.stop()
            event.ignore()
        else:
            event.accept()


def run_app():
    app = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 22))
    app.setStyleSheet(STYLE)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
