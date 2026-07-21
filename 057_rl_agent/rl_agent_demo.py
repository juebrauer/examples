"""Minimalistische Demo fuer einen Agenten, der durch Versuch und Irrtum lernt."""

import math
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from agent_rl import AgentRL, LearningReport


WORLD_SIZE = 12
MAX_STEPS = 100
ACTION_NAMES = ("hoch", "runter", "links", "rechts")
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class StepResult:
    reward: float
    reached_goal: bool
    done: bool


class ImageWorld:
    """Die Welt liefert dem Agenten nur Bilder, niemals ihre Koordinaten."""

    def __init__(self, size: int = WORLD_SIZE):
        self.size = size
        self.agent_x = 0
        self.agent_y = 0
        self.goal_x = 0
        self.goal_y = 0
        self.steps = 0
        self.reset()

    def reset(self) -> None:
        self.agent_x = random.randrange(self.size)
        self.agent_y = random.randrange(self.size)
        while True:
            self.goal_x = random.randrange(self.size)
            self.goal_y = random.randrange(self.size)
            distance = abs(self.goal_x - self.agent_x) + abs(self.goal_y - self.agent_y)
            if distance >= self.size // 2:
                break
        self.steps = 0

    def image(self) -> np.ndarray:
        image = np.zeros((3, self.size, self.size), dtype=np.float32)
        image[0, self.goal_y, self.goal_x] = 1.0    # Ziel: rot
        image[2, self.agent_y, self.agent_x] = 1.0  # Agent: blau
        return image

    def distance(self) -> int:
        return abs(self.goal_x - self.agent_x) + abs(self.goal_y - self.agent_y)

    def step(self, action: int) -> StepResult:
        old_distance = self.distance()
        dx, dy = ACTION_DELTAS[action]
        self.agent_x = max(0, min(self.size - 1, self.agent_x + dx))
        self.agent_y = max(0, min(self.size - 1, self.agent_y + dy))
        self.steps += 1

        reached_goal = self.distance() == 0
        timed_out = self.steps >= MAX_STEPS
        if reached_goal:
            reward = 1.0
        elif timed_out:
            reward = -0.5
        elif self.distance() < old_distance:
            reward = 0.10
        elif self.distance() > old_distance:
            reward = -0.10
        else:
            reward = -0.15  # Gegen den Rand gelaufen.
        return StepResult(reward, reached_goal, reached_goal or timed_out)


class WorldView(QWidget):
    def __init__(self, world: ImageWorld):
        super().__init__()
        self.world = world
        self.setMinimumSize(460, 460)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#20242b"))
        side = min(self.width(), self.height()) - 24
        cell = side / self.world.size
        left = (self.width() - side) / 2
        top = (self.height() - side) / 2

        painter.fillRect(int(left), int(top), int(side), int(side), QColor("#f4f2ed"))
        painter.setPen(QColor("#d8d5ce"))
        for index in range(self.world.size + 1):
            x = int(left + index * cell)
            y = int(top + index * cell)
            painter.drawLine(x, int(top), x, int(top + side))
            painter.drawLine(int(left), y, int(left + side), y)

        def cell_rect(x: int, y: int, margin: float) -> tuple[int, int, int, int]:
            return (
                int(left + x * cell + margin),
                int(top + y * cell + margin),
                max(1, int(cell - 2 * margin)),
                max(1, int(cell - 2 * margin)),
            )

        painter.setBrush(QColor("#e34b4b"))
        painter.setPen(QColor("#a51f2c"))
        painter.drawEllipse(*cell_rect(self.world.goal_x, self.world.goal_y, cell * 0.18))
        painter.setBrush(QColor("#3478d4"))
        painter.setPen(QColor("#174c91"))
        painter.drawEllipse(*cell_rect(self.world.agent_x, self.world.agent_y, cell * 0.12))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("057 – Lernen durch Versuch und Irrtum")
        self.world = ImageWorld()
        self.agent = self._new_agent(history_length=8)
        self.mode: str | None = None
        self.target_episodes = 0
        self.finished_episodes = 0
        self.successes = 0
        self.recent_successes: deque[int] = deque(maxlen=100)
        self.last_report: LearningReport | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._run_tick)
        self._build_ui()
        self._refresh_labels()

    def _new_agent(self, history_length: int) -> AgentRL:
        return AgentRL(
            image_size=WORLD_SIZE,
            history_length=history_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        self.world_view = WorldView(self.world)
        layout.addWidget(self.world_view, 3)

        panel = QVBoxLayout()
        layout.addLayout(panel, 2)
        title = QLabel("agent_rl")
        title.setStyleSheet("font-size: 24px; font-weight: bold")
        explanation = QLabel(
            "Blau sieht nur das Bild der Welt. Gute Ueberraschungen verstaerken "
            "die letzten N Entscheidungen, schlechte schwaechen sie ab."
        )
        explanation.setWordWrap(True)
        panel.addWidget(title)
        panel.addWidget(explanation)

        form = QFormLayout()
        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100_000)
        self.episodes_input.setValue(1000)
        form.addRow("Lern-Episoden", self.episodes_input)
        self.history_input = QSpinBox()
        self.history_input.setRange(1, 100)
        self.history_input.setValue(8)
        form.addRow("Letzte N Aktionen", self.history_input)
        panel.addLayout(form)

        self.learn_button = QPushButton("Lernen starten")
        self.learn_button.clicked.connect(self._toggle_learning)
        self.test_button = QPushButton("20 Episoden testen")
        self.test_button.clicked.connect(self._start_test)
        self.reset_button = QPushButton("Neues, ungelerntes Modell")
        self.reset_button.clicked.connect(self._reset_agent)
        panel.addWidget(self.learn_button)
        panel.addWidget(self.test_button)
        panel.addWidget(self.reset_button)

        self.status_label = QLabel()
        self.episode_label = QLabel()
        self.success_label = QLabel()
        self.action_label = QLabel()
        self.reward_label = QLabel()
        self.expected_label = QLabel()
        self.surprise_label = QLabel()
        values = QFormLayout()
        values.addRow("Status", self.status_label)
        values.addRow("Episode", self.episode_label)
        values.addRow("Erfolge", self.success_label)
        values.addRow("Letzte Aktion", self.action_label)
        values.addRow("Belohnung", self.reward_label)
        values.addRow("Erwartet", self.expected_label)
        values.addRow("Ueberraschung", self.surprise_label)
        panel.addSpacing(16)
        panel.addLayout(values)
        panel.addStretch()

    def _set_controls_running(self, running: bool) -> None:
        self.episodes_input.setEnabled(not running)
        self.history_input.setEnabled(not running)
        self.test_button.setEnabled(not running)
        self.reset_button.setEnabled(not running)

    def _toggle_learning(self) -> None:
        if self.mode == "learning":
            self._stop("Lernen angehalten")
            return
        if self.mode is not None:
            return
        self._start("learning", self.episodes_input.value())

    def _start_test(self) -> None:
        if self.mode is None:
            self._start("test", 20)

    def _start(self, mode: str, episodes: int) -> None:
        self.mode = mode
        self.target_episodes = episodes
        self.finished_episodes = 0
        self.successes = 0
        self.recent_successes.clear()
        self.last_report = None
        self.agent.set_history_length(self.history_input.value())
        self.world.reset()
        self.agent.begin_episode()
        self._set_controls_running(True)
        self.learn_button.setEnabled(mode == "learning")
        self.learn_button.setText("Lernen stoppen" if mode == "learning" else "Lernen starten")
        self.timer.start(0 if mode == "learning" else 60)
        self._refresh_labels()

    def _stop(self, message: str) -> None:
        self.timer.stop()
        self.mode = None
        self._set_controls_running(False)
        self.learn_button.setEnabled(True)
        self.learn_button.setText("Lernen starten")
        self.status_label.setText(message)

    def _reset_agent(self) -> None:
        self.agent = self._new_agent(self.history_input.value())
        self.world.reset()
        self.finished_episodes = 0
        self.successes = 0
        self.recent_successes.clear()
        self.last_report = None
        self.action_label.setText("-")
        self._refresh_labels("Neues Modell")
        self.world_view.update()

    def _run_tick(self) -> None:
        # Beim Lernen werden mehrere Schritte zwischen zwei UI-Zeichnungen
        # ausgefuehrt. Beim Test bleibt jeder einzelne Schritt sichtbar.
        steps_this_tick = 12 if self.mode == "learning" else 1
        for _ in range(steps_this_tick):
            if self.mode is None:
                return
            image_before_action = self.world.image()
            learning = self.mode == "learning"
            action = self.agent.choose_action(image_before_action, try_new_actions=learning)
            result = self.world.step(action)
            self.action_label.setText(ACTION_NAMES[action])

            if learning:
                self.last_report = self.agent.observe(
                    image_before_action, action, result.reward
                )
            else:
                self.last_report = LearningReport(
                    reward=result.reward,
                    expected_reward=float("nan"),
                    surprise=float("nan"),
                    history_items=0,
                )

            if result.done:
                self.finished_episodes += 1
                self.successes += int(result.reached_goal)
                self.recent_successes.append(int(result.reached_goal))
                if self.finished_episodes >= self.target_episodes:
                    rate = 100.0 * self.successes / self.finished_episodes
                    label = "Lernen" if learning else "Test"
                    self._stop(f"{label} fertig: {rate:.1f} % Erfolg")
                    break
                self.world.reset()
                self.agent.begin_episode()

        self._refresh_labels()
        self.world_view.update()

    def _refresh_labels(self, status: str | None = None) -> None:
        if status is not None:
            self.status_label.setText(status)
        elif self.mode == "learning":
            self.status_label.setText("lernt durch Ausprobieren")
        elif self.mode == "test":
            self.status_label.setText("testet ohne zu lernen")
        elif not self.status_label.text():
            self.status_label.setText("bereit")

        target = self.target_episodes if self.mode else max(self.finished_episodes, 0)
        self.episode_label.setText(f"{self.finished_episodes} / {target}")
        recent_rate = (
            100.0 * sum(self.recent_successes) / len(self.recent_successes)
            if self.recent_successes else 0.0
        )
        self.success_label.setText(
            f"{self.successes} (letzte 100: {recent_rate:.0f} %)"
        )
        if self.last_report is None:
            self.reward_label.setText("-")
            self.expected_label.setText("-")
            self.surprise_label.setText("-")
        else:
            self.reward_label.setText(f"{self.last_report.reward:+.2f}")
            if math.isnan(self.last_report.expected_reward):
                self.expected_label.setText("- (Test)")
                self.surprise_label.setText("-")
            else:
                self.expected_label.setText(f"{self.last_report.expected_reward:+.2f}")
                self.surprise_label.setText(f"{self.last_report.surprise:+.2f}")


def main() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    # Fuer die sehr kleinen Einzelbilder ist ein CPU-Thread schneller und
    # laesst der Qt-Ereignisschleife mehr Luft als ein grosser Thread-Pool.
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 540)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
