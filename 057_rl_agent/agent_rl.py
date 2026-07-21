"""Ein bewusst einfacher Agent, der nur aus seinen Erlebnissen lernt.

Der Agent bekommt als Zustand ausschliesslich ein RGB-Bild der Welt. Er kennt
weder Koordinaten noch die Richtung zum Ziel. Ein Netz erzeugt vier
Aktionswahrscheinlichkeiten, ein zweites schaetzt die in einem Bild zu
erwartende Belohnung.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


NUMBER_OF_ACTIONS = 4


class ImageNetwork(nn.Module):
    """Kleines CNN mit einem frei waehlbaren Ausgabekopf."""

    def __init__(self, image_size: int, outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature_count = int(self.features(dummy).numel())

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, outputs),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


@dataclass(frozen=True)
class LearningReport:
    reward: float
    expected_reward: float
    surprise: float
    history_items: int


class AgentRL:
    """Lernt durch Verstaerken oder Abschwaechen der letzten N Entscheidungen.

    Eine positive Ueberraschung vergroessert die Wahrscheinlichkeit jeder in
    der Historie gespeicherten (Bild, Aktion)-Entscheidung. Eine negative
    Ueberraschung verkleinert sie. Das Erwartungsnetz sieht die Aktion bewusst
    nicht: Es lernt, welche Belohnung vor der Auswahl einer Aktion in diesem
    Zustand normalerweise zu erwarten ist.
    """

    def __init__(
        self,
        image_size: int,
        history_length: int = 8,
        exploration: float = 0.10,
        learning_rate: float = 3e-4,
        expectation_learning_rate: float = 1e-3,
        surprise_limit: float = 1.0,
        device: torch.device | None = None,
    ):
        if history_length < 1:
            raise ValueError("history_length muss mindestens 1 sein")
        if not 0.0 <= exploration < 1.0:
            raise ValueError("exploration muss zwischen 0 und 1 liegen")

        self.image_size = image_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.exploration = exploration
        self.surprise_limit = surprise_limit

        self.action_network = ImageNetwork(image_size, NUMBER_OF_ACTIONS).to(self.device)
        self.expectation_network = ImageNetwork(image_size, 1).to(self.device)
        self.action_optimizer = torch.optim.Adam(
            self.action_network.parameters(), lr=learning_rate
        )
        self.expectation_optimizer = torch.optim.Adam(
            self.expectation_network.parameters(), lr=expectation_learning_rate
        )
        self.expectation_loss = nn.SmoothL1Loss()
        self._history: deque[tuple[np.ndarray, int]] = deque(maxlen=history_length)

    @property
    def history_length(self) -> int:
        return int(self._history.maxlen or 1)

    def set_history_length(self, length: int) -> None:
        if length < 1:
            raise ValueError("length muss mindestens 1 sein")
        old_items = list(self._history)[-length:]
        self._history = deque(old_items, maxlen=length)

    def begin_episode(self) -> None:
        self._history.clear()

    def _image_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.shape != (3, self.image_size, self.image_size):
            raise ValueError(
                f"Erwartetes Bildformat: (3, {self.image_size}, {self.image_size})"
            )
        return torch.as_tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)

    def action_probabilities(self, image: np.ndarray) -> np.ndarray:
        self.action_network.eval()
        with torch.no_grad():
            logits = self.action_network(self._image_tensor(image))[0]
            probabilities = torch.softmax(logits, dim=0)
        return probabilities.cpu().numpy()

    def choose_action(self, image: np.ndarray, try_new_actions: bool = True) -> int:
        probabilities = self.action_probabilities(image)
        if not try_new_actions:
            return int(np.argmax(probabilities))

        # Ein kleiner gleichverteilter Anteil sorgt dafuer, dass der Agent auch
        # nach vielen Erfahrungen weiterhin Alternativen ausprobiert.
        probabilities = (
            (1.0 - self.exploration) * probabilities
            + self.exploration / NUMBER_OF_ACTIONS
        )
        return int(np.random.choice(NUMBER_OF_ACTIONS, p=probabilities))

    def observe(self, image: np.ndarray, action: int, reward: float) -> LearningReport:
        """Beobachtet eine Belohnung und lernt sofort aus der letzten Historie."""
        if not 0 <= action < NUMBER_OF_ACTIONS:
            raise ValueError("Unbekannte Aktion")

        image_tensor = self._image_tensor(image)

        # Zuerst wird die alte Erwartung gelesen. Erst danach darf das
        # Erwartungsnetz die gerade beobachtete Belohnung kennenlernen.
        self.expectation_network.train()
        predicted = self.expectation_network(image_tensor)
        expected_reward = float(predicted.detach().item())
        target = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        expectation_loss = self.expectation_loss(predicted, target)
        self.expectation_optimizer.zero_grad()
        expectation_loss.backward()
        self.expectation_optimizer.step()

        raw_surprise = reward - expected_reward
        surprise = float(
            np.clip(raw_surprise, -self.surprise_limit, self.surprise_limit)
        )
        self._history.append((np.array(image, copy=True), action))

        history_images = np.stack([item[0] for item in self._history])
        history_actions = torch.tensor(
            [item[1] for item in self._history],
            dtype=torch.long,
            device=self.device,
        )
        images_tensor = torch.as_tensor(
            history_images, dtype=torch.float32, device=self.device
        )

        self.action_network.train()
        logits = self.action_network(images_tensor)
        chosen_log_probabilities = torch.log_softmax(logits, dim=1).gather(
            1, history_actions[:, None]
        ).squeeze(1)

        # surprise > 0: Loss-Minimierung erhoeht log(p) der gewaehlten Aktionen.
        # surprise < 0: Sie senkt log(p). Alle N Eintraege zaehlen gleich stark.
        action_loss = -surprise * chosen_log_probabilities.mean()
        self.action_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_network.parameters(), 1.0)
        self.action_optimizer.step()

        return LearningReport(
            reward=float(reward),
            expected_reward=expected_reward,
            surprise=raw_surprise,
            history_items=len(self._history),
        )
