"""
DQN Lehr-Demo für die Vorlesung "Reinforcement Learning"
=========================================================

Dieses Skript demonstriert, wie ein Deep Q-Network (DQN) ein einfaches Navigationsproblem
direkt aus Bildpixeln (RGB-Bild der Größe 48x48) löst.

Didaktische Highlights für die Vorlesung:
1. **Pixelgenaue Visualisierung**: Links wird das 48x48 Bild, das das CNN als Input erhält,
   pixelgenau vergrößert angezeigt. So wird klar, dass der Agent keine Koordinaten sieht.
2. **Live Q-Werte-Kreuz**: In der Mitte zeigt ein 2D-Layout in Echtzeit die berechneten Q-Werte
   für Oben, Unten, Links und Rechts. Die maximale Aktion (greedy) wird grün hervorgehoben.
   Dies zeigt live, wie die Value-Function konvergiert!
3. **Exploration vs. Exploitation**: Die tatsächlich gewählte Aktion wird markiert. Wenn der
   Agent aufgrund von Epsilon-Exploration eine nicht-optimale Aktion wählt, wird das farblich
   sichtbar.
4. **Matplotlib-Lernkurve**: Unten wird der gleitende Mittelwert der Rewards pro Episode live geplottet.
   (Die Schrittanzahl wurde bewusst entfernt, um die Visualisierung übersichtlich zu halten.)
5. **Zwei Trainingsmodi**:
   - *Schritt-für-Schritt beobachten*: Der Agent läuft mit anpassbarer Geschwindigkeit, und jeder
     Schritt inklusive Q-Werte und Bild-Input wird live im UI visualisiert (ideal zum Erklären).
   - *Maximal schnell trainieren*: Die Visualisierung wird übersprungen und die Grafik nur alle
     10 Episoden aktualisiert, was ein extrem schnelles Training ermöglicht.
6. **Modell-Verwaltung**: Zwischenmodelle werden automatisch im Ordner `models/` gespeichert und
   können jederzeit über das UI geladen werden, um das Training fortzusetzen oder zu evaluieren.

Voraussetzungen:
    pip install PySide6 torch numpy matplotlib

Autor: Antigravity (Google DeepMind Advanced Agentic Coding Team)
"""

import os
import sys
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Matplotlib Integration in PySide6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# 1. Das Environment (Einfache Grid-Welt mit Bild-Output)
# ==============================================================================

class SimpleGridEnv:
    """
    Eine einfache 24x24 Grid-Welt. (Didaktisch vergrößert für anspruchsvollere Pfade)
    Der Zustand (Observation) ist ein 48x48 RGB-Bild (float32, [0, 1]).
    Der Agent (grünes Quadrat) versucht das Ziel (roter Kreis) zu erreichen.
    """
    def __init__(self, grid_size: int = 24, obs_size: int = 48):
        self.grid_size = grid_size
        self.obs_size = obs_size
        self.cell_size = obs_size / grid_size  # Pixel pro Grid-Zelle (hier: 48 / 24 = 2)
        
        # Aktionen: 0: Oben, 1: Unten, 2: Links, 3: Rechts
        self.action_space = [0, 1, 2, 3]
        self.action_names = {0: "Oben", 1: "Unten", 2: "Links", 3: "Rechts"}
        self.action_deltas = {
            0: (0, -1),  # Oben (y-1)
            1: (0, 1),   # Unten (y+1)
            2: (-1, 0),  # Links (x-1)
            3: (1, 0),   # Rechts (x+1)
        }
        
        self.agent_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.steps = 0
        self.max_steps = 150  # Erhöht für die größere Welt
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Setzt die Umgebung zurück. Platziert Agent und Ziel zufällig."""
        self.steps = 0
        
        # Agent zufällig platzieren
        self.agent_pos = (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )
        
        # Ziel zufällig platzieren (darf nicht auf Agenten liegen)
        while True:
            self.goal_pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if self.goal_pos != self.agent_pos:
                break
                
        return self.render_observation()
        
    def get_l1_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Berechnet den Manhattan-Abstand zwischen zwei Punkten."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Führt eine Aktion aus und gibt (nächster_zustand, reward, done, info) zurück."""
        self.steps += 1
        dx, dy = self.action_deltas[action]
        
        old_dist = self.get_l1_distance(self.agent_pos, self.goal_pos)
        
        # Neue Position berechnen und an die Grenzen clippen
        new_x = max(0, min(self.grid_size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.agent_pos[1] + dy))
        
        hit_wall = (new_x == self.agent_pos[0] and new_y == self.agent_pos[1])
        self.agent_pos = (new_x, new_y)
        
        new_dist = self.get_l1_distance(self.agent_pos, self.goal_pos)
        reached_goal = (self.agent_pos == self.goal_pos)
        
        # --- Reward Shaping für schnelles Lernen im Hörsaal ---
        # 1. Zeitschmerz (animiert den Agenten, schnell zu sein)
        reward = -0.05
        
        # 2. Wand-Kollision leicht bestrafen
        if hit_wall:
            reward -= 0.1
            
        # 3. Annäherung belohnen / Entfernung bestrafen
        # (Dies gibt dem Agenten dichte, graduelle Rückmeldung)
        dist_change = old_dist - new_dist
        reward += dist_change * 0.5
        
        # 4. Zielerreichung belohnen
        if reached_goal:
            reward += 10.0
            
        done = reached_goal or (self.steps >= self.max_steps)
        
        info = {
            "reached_goal": reached_goal,
            "steps": self.steps,
            "hit_wall": hit_wall
        }
        
        return self.render_observation(), reward, done, info

    def render_observation(self) -> np.ndarray:
        """
        Rendert die Umgebung als 48x48 RGB-Bild im Wertebereich [0, 1].
        Dies ist das exakte Bild, das das neuronale Netz als Eingabe erhält.
        """
        # Weißer Hintergrund
        img = np.ones((self.obs_size, self.obs_size, 3), dtype=np.float32)
        
        cs = int(self.cell_size)
        
        # 1. Ziel zeichnen (Roter Bereich exakt in seiner Zelle)
        gx, gy = self.goal_pos
        img[gy*cs:(gy+1)*cs, gx*cs:(gx+1)*cs] = [1.0, 0.0, 0.0]  # Rot
        
        # 2. Agent zeichnen (Grüner Bereich exakt in seiner Zelle)
        ax, ay = self.agent_pos
        img[ay*cs:(ay+1)*cs, ax*cs:(ax+1)*cs] = [0.0, 0.8, 0.0]  # Grün
        
        return img


# ==============================================================================
# 2. Das DQN Modell (CNN mit PyTorch)
# ==============================================================================

class DQN(nn.Module):
    """
    Einfaches, verständliches Convolutional Neural Network (CNN).
    Eingabe: (Batch_Size, Channels=3, Height=48, Width=48)
    Ausgabe: (Batch_Size, Aktionen=4)
    """
    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        super().__init__()
        
        # 1. Convolutional Block: Extrahiert einfache geometrische Muster
        # Input: 3 x 48 x 48
        # Conv: 3x3 Filter, Padding 1 -> Output: 16 Kanäle, 48 x 48
        # Max-Pooling: 2x2 -> Halbiert die Dimension -> Output: 16 Kanäle, 24 x 24
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Convolutional Block: Kombiniert Muster zu komplexeren Objekten
        # Input: 16 Kanäle, 24 x 24
        # Conv: 3x3 Filter, Padding 1 -> Output: 32 Kanäle, 24 x 24
        # Max-Pooling: 2x2 -> Halbiert die Dimension -> Output: 32 Kanäle, 12 x 12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Nach dem Pooling haben wir ein Tensor-Volumen von 32 x 12 x 12.
        # Um dies in ein fully connected Layer zu leiten, müssen wir es glätten (Flatten).
        # Die flache Feature-Größe ist: 32 Kanäle * 12 Pixel * 12 Pixel = 4608 Features.
        self.flat_features = 32 * 12 * 12
        
        # Fully Connected (Dichte) Schichten zur Entscheidungsfindung
        # FC1: Reduziert 4608 Features auf 256 repräsentative Neuronen (Erhöht für 24x24 Grid)
        self.fc1 = nn.Linear(self.flat_features, 256)
        # FC2: Mappt die 256 Neuronen auf die 4 Q-Werte (einer pro Richtung)
        self.fc2 = nn.Linear(256, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eingabe-Aktivierung: Conv1 -> ReLU -> Pool1
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Zweite Aktivierung: Conv2 -> ReLU -> Pool2
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten: (Batch_Size, 32, 12, 12) -> (Batch_Size, 4608)
        x = x.reshape(-1, self.flat_features)
        
        # Fully Connected Schichten mit ReLU Aktivierung im Hidden-Layer
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


# ==============================================================================
# 3. Der DQN Agent (RL-Algorithmus & Experience Replay)
# ==============================================================================

class DQNAgent:
    """
    Klassischer DQN Agent mit Experience Replay und Target-Netzwerk.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (48, 48, 3),
        num_actions: int = 4,
        lr: float = 5e-4,  # Leicht reduziert für stabilere Updates
        gamma: float = 0.99, # Erhöht, damit der Agent weitsichtiger wird
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay_steps: int = 150000,  # Deutlich erhöht für die größere 24x24 Welt!
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_frequency: int = 1000, # Etwas seltener updaten für stabilere Q-Targets
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Epsilon-Greedy Parameter
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon_start - epsilon_min) / epsilon_decay_steps
        
        # Device-Auswahl (Nutze GPU falls verfügbar)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Replay Buffer: Speichert (s, a, r, s', done)
        self.memory = deque(maxlen=buffer_capacity)
        
        # Policy-Netzwerk (wird trainiert) und Target-Netzwerk (berechnet stabile Zielwerte)
        self.policy_net = DQN(in_channels=obs_shape[2], num_actions=num_actions).to(self.device)
        self.target_net = DQN(in_channels=obs_shape[2], num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.total_steps = 0
        self.training_steps = 0
        
    def preprocess(self, state: np.ndarray) -> torch.Tensor:
        """Konvertiert ein NumPy Bild (H, W, C) in einen PyTorch Tensor (1, C, H, W)."""
        # HWC -> CHW und normalisieren
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state_t = state_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return state_t

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> Tuple[int, np.ndarray, bool]:
        """
        Wählt eine Aktion nach Epsilon-Greedy aus.
        Gibt (Aktion, alle_Q_Werte, war_explorativ) zurück.
        """
        state_t = self.preprocess(state)
        
        # Q-Werte vom Policy-Netzwerk berechnen lassen
        with torch.no_grad():
            q_values_tensor = self.policy_net(state_t)
            q_values = q_values_tensor.cpu().numpy()[0]
            
        # Epsilon-Greedy Entscheidung
        if not evaluation and random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            is_exploration = True
        else:
            action = int(np.argmax(q_values))
            is_exploration = False
            
        return action, q_values, is_exploration
        
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Speichert einen Übergang im Experience Replay Buffer."""
        self.memory.append((state, action, reward, next_state, done))
        self.total_steps += 1
        
        # Epsilon linear dekrementieren
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
    def train_step(self) -> Optional[float]:
        """Führt einen Trainingsschritt auf einem zufälligen Batch aus. Gibt den Loss zurück."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Zufälliges Sample aus dem Replay Buffer ziehen
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Arrays in PyTorch Tensoren konvertieren
        # NumPy Stacken für effiziente Konvertierung
        states_np = np.stack(states)
        next_states_np = np.stack(next_states)
        
        # HWC -> CHW für den gesamten Batch: (Batch, H, W, C) -> (Batch, C, H, W)
        states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2).contiguous()
        next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2).contiguous()
        
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 1. Aktuelle Q(s, a) Werte berechnen (über das Policy Netzwerk)
        # gather() wählt die Q-Werte der tatsächlich gewählten Aktionen aus.
        q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # 2. Maximale Q(s', a') Werte für den nächsten Zustand berechnen (über das Target Netzwerk)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            # Bellman-Gleichung für das Target: Q_target = r + gamma * max_a' Q_target(s', a')
            target_q_values = rewards_t + (self.gamma * next_q_values * (1.0 - dones_t))
            
        # 3. Loss berechnen (MSE zwischen Policy-Vorhersage und Bellman-Zielwert)
        loss = self.loss_fn(q_values, target_q_values)
        
        # 4. Gradientenabstieg durchführen
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping zur Stabilisierung des Trainings
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Target-Netzwerk periodisch aktualisieren (Kopieren der Gewichte)
        if self.training_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save_model(self, filepath: str):
        """Speichert die Netzwerk-Gewichte."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_steps': self.training_steps
        }, filepath)

    def load_model(self, filepath: str):
        """Lädt die Netzwerk-Gewichte."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_steps = checkpoint.get('total_steps', self.total_steps)
        self.training_steps = checkpoint.get('training_steps', self.training_steps)


# ==============================================================================
# 4. Die UI Widgets (CnnInputView, QValueWidget, LearningCurvePlot)
# ==============================================================================

class CnnInputView(QWidget):
    """
    Zeigt das 48x48 Bild des Environments pixelgenau hochskaliert an.
    Dadurch sehen Studierende genau, was das neuronale Netz "sieht".
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.obs: Optional[np.ndarray] = None
        self.setMinimumSize(256, 256)
        
    def set_observation(self, obs: np.ndarray):
        self.obs = obs
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Hintergrund zeichnen falls kein Bild geladen ist
        if self.obs is None:
            painter.fillRect(self.rect(), QColor(240, 240, 240))
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "Warte auf Observation...")
            return
            
        # Konvertiere NumPy-Bild [0,1] float32 in QImage
        arr = np.clip(self.obs * 255.0, 0, 255).astype(np.uint8)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, c * w, QImage.Format_RGB888)
        
        # Hochskalieren mittels FastTransformation (Nearest Neighbor) für den Retro-Pixel-Look
        scaled_pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(),
            self.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        
        # Zentriert zeichnen
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.fillRect(self.rect(), QColor(30, 30, 30))  # Dunkler Hintergrund für Kontrast
        painter.drawPixmap(x, y, scaled_pixmap)


class QValueWidget(QGroupBox):
    """
    Ein didaktisches 2D-Kreuz-Widget zur Visualisierung der Q-Werte.
    Zeigt Oben, Unten, Links und Rechts mit Werten an.
    Hebt das Maximum (Greedy Aktion) grün hervor.
    Zeigt die tatsächlich ausgeführte Aktion gelb umrandet an (Exploration Visualisierung!).
    """
    def __init__(self, title: str = "Live Q(s,a) Werte des CNNs", parent=None):
        super().__init__(title, parent)
        self.q_values = np.zeros(4)
        self.max_idx = 0
        self.actual_action: Optional[int] = None
        self.is_exploration = False
        
        # 3x3 Grid für das Q-Werte-Kreuz
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(6)
        
        # Erstelle die Labels für die Richtungen
        self.labels = {
            0: QLabel("↑ Oben\n-"),
            1: QLabel("↓ Unten\n-"),
            2: QLabel("← Links\n-"),
            3: QLabel("→ Rechts\n-")
        }
        
        # Style für die Labels definieren
        for key, lbl in self.labels.items():
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "border: 2px solid #bdc3c7;"
                "border-radius: 6px;"
                "background-color: #ecf0f1;"
                "font-weight: bold;"
                "font-size: 11px;"
                "padding: 6px;"
            )
            
        # Im 3x3 Gitter anordnen
        #   [ ] [O] [ ]
        #   [L] [C] [R]
        #   [ ] [U] [ ]
        self.grid_layout.addWidget(self.labels[0], 0, 1)  # Oben
        self.grid_layout.addWidget(self.labels[2], 1, 0)  # Links
        
        # Center-Label als visueller Platzhalter für den Agenten
        self.center_lbl = QLabel("Agent\ns")
        self.center_lbl.setAlignment(Qt.AlignCenter)
        self.center_lbl.setStyleSheet(
            "border: 2px solid #7f8c8d;"
            "border-radius: 6px;"
            "background-color: #95a5a6;"
            "color: white;"
            "font-weight: bold;"
        )
        self.grid_layout.addWidget(self.center_lbl, 1, 1)
        
        self.grid_layout.addWidget(self.labels[3], 1, 2)  # Rechts
        self.grid_layout.addWidget(self.labels[1], 2, 1)  # Unten
        
    def update_q_values(self, q_values: np.ndarray, actual_action: int, is_exploration: bool):
        """Aktualisiert das Widget mit neuen Q-Werten und hebt Aktionen hervor."""
        self.q_values = q_values
        self.max_idx = int(np.argmax(q_values))
        self.actual_action = actual_action
        self.is_exploration = is_exploration
        
        for i in range(4):
            val = q_values[i]
            direction = ["↑ Oben", "↓ Unten", "← Links", "→ Rechts"][i]
            self.labels[i].setText(f"{direction}\n{val:.3f}")
            
            # Basis-Style
            style = "border-radius: 6px; font-weight: bold; font-size: 11px; padding: 6px; "
            
            # Farbliche Kennzeichnung
            if i == self.max_idx:
                # Greedy Aktion (Maximaler Q-Wert) -> Grün
                bg_color = "background-color: #2ecc71; color: white;"
                border_color = "border: 2px solid #27ae60;"
            else:
                # Andere Aktionen -> Grau
                bg_color = "background-color: #ecf0f1; color: black;"
                border_color = "border: 2px solid #bdc3c7;"
                
            # Wenn dies die tatsächlich ausgeführte Aktion war (evtl. Exploration!)
            if i == self.actual_action:
                if self.is_exploration:
                    # Zufälliger Explorationsschritt -> Orange hervorgehoben
                    border_color = "border: 3px solid #e67e22;"
                    bg_color = "background-color: #f39c12; color: white;" if i != self.max_idx else "background-color: #2ecc71; color: white;"
                else:
                    # Regulärer Greedy-Schritt -> Starker grüner Rand
                    border_color = "border: 3px solid #1abc9c;"
                    
            self.labels[i].setStyleSheet(style + bg_color + border_color)
            
        # Text im Center-Label anpassen
        if self.is_exploration:
            self.center_lbl.setText("Exploration!\n(Zufall)")
            self.center_lbl.setStyleSheet("border: 2px solid #d35400; border-radius: 6px; background-color: #e67e22; color: white; font-weight: bold;")
        else:
            self.center_lbl.setText("Exploitation\n(Greedy)")
            self.center_lbl.setStyleSheet("border: 2px solid #2980b9; border-radius: 6px; background-color: #3498db; color: white; font-weight: bold;")

    def clear(self):
        """Setzt die Anzeige zurück."""
        for i in range(4):
            direction = ["↑ Oben", "↓ Unten", "← Links", "→ Rechts"][i]
            self.labels[i].setText(f"{direction}\n-")
            self.labels[i].setStyleSheet(
                "border: 2px solid #bdc3c7;"
                "border-radius: 6px;"
                "background-color: #ecf0f1;"
                "font-weight: bold;"
                "font-size: 11px;"
                "padding: 6px;"
            )
        self.center_lbl.setText("Agent\ns")
        self.center_lbl.setStyleSheet("border: 2px solid #7f8c8d; border-radius: 6px; background-color: #95a5a6; color: white; font-weight: bold;")


class LearningCurvePlot(FigureCanvas):
    """
    Ein integrierter Matplotlib-Plot, der den gleitenden Mittelwert der Rewards
    anzeigt, um den Lernfortschritt zu visualisieren. (Schritt-Kurve entfernt für Übersichtlichkeit).
    """
    def __init__(self, parent=None):
        # Erstelle Matplotlib Figure
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.fig.patch.set_facecolor('#f9f9f9')
        super().__init__(self.fig)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        
        # Daten-Listen
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.clear()
        self.ax.set_title("Lernfortschritt (Episoden-Reward)", fontsize=10, fontweight='bold')
        self.ax.set_xlabel("Episode", fontsize=8)
        self.ax.set_ylabel("Reward", fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.draw_idle()
        
    def add_episode_data(self, episode: int, reward: float, redraw: bool = True):
        """Fügt Daten einer abgeschlossen Episode hinzu und aktualisiert optional die Grafik."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        
        if redraw:
            self.update_plot()
            
    def update_plot(self):
        """Zeichnet die Matplotlib-Grafik neu."""
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_title("Lernfortschritt", fontsize=10, fontweight='bold')
        self.ax.set_xlabel("Episode", fontsize=8)
        self.ax.set_ylabel("Reward", color='#1f77b4', fontsize=8)
        self.ax.tick_params(axis='y', labelcolor='#1f77b4')
        
        # Plot der rohen Rewards (hellblau, transparent)
        self.ax.plot(self.episodes, self.rewards, color='#1f77b4', alpha=0.3, label="Reward")
        
        # Plot des gleitenden Mittelwerts (dunkelblau, fett)
        if len(self.rewards) >= 5:
            window = min(10, len(self.rewards))
            moving_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            self.ax.plot(self.episodes[window-1:], moving_avg, color='#0f3a5a', linewidth=2, label=f"Mittelwert ({window} Ep.)")
            
        self.ax.legend(loc='upper left', fontsize=7)
        self.fig.tight_layout()
        self.draw_idle()
        
    def clear(self):
        """Setzt den Plot zurück."""
        self.episodes.clear()
        self.rewards.clear()
        self.setup_plot()


# ==============================================================================
# 5. Das Hauptfenster (MainWindow)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DQN Bild-Input Lehr-Demo (PySide6 + PyTorch)")
        self.resize(1150, 720)
        
        # RL Komponenten initialisieren
        self.env = SimpleGridEnv()
        self.agent = DQNAgent(
            obs_shape=(self.env.obs_size, self.env.obs_size, 3),
            num_actions=4
        )
        
        # State & Trainingsvariablen
        self.state = self.env.reset()
        self.episode_count = 1
        self.episode_reward = 0.0
        self.episode_steps = 0
        
        self.running = False
        self.observe_mode = True  # Standardmäßig im didaktischen Beobachtungsmodus
        self.last_loss: Optional[float] = None
        
        # Ordner für Modelle
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # GUI aufbauen
        self.setup_ui()
        
        # Timer für den Trainings-Loop (gesteuert durch PySide)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.train_loop_step)
        self.update_timer_speed()  # Timer auf Standardgeschwindigkeit einstellen
        self.timer.start()
        
        # Erstes Rendering
        self.refresh_visuals()
        self.update_stats_labels()
        
    def setup_ui(self):
        # Haupt-Widget und Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(12)
        
        # ================= LINKS: Visualisierung des Inputs & Q-Werte =================
        left_panel = QVBoxLayout()
        
        # Gruppenbox für das Bild
        img_box = QGroupBox("Was das CNN sieht (Eingabebild 48x48 pixelgenau vergrößert)")
        img_layout = QVBoxLayout(img_box)
        self.cnn_view = CnnInputView()
        img_layout.addWidget(self.cnn_view)
        left_panel.addWidget(img_box, stretch=2)
        
        # Live Q-Werte Kreuz
        self.q_value_widget = QValueWidget()
        left_panel.addWidget(self.q_value_widget, stretch=1)
        
        main_layout.addLayout(left_panel, stretch=1)
        
        # ================= RECHTS: Steuerung, Status und Plot =================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # 1. Status & Parameter Anzeige
        stats_box = QGroupBox("RL Status & Parameter")
        stats_grid = QGridLayout(stats_box)
        stats_grid.setSpacing(8)
        
        # Statische Labels
        stats_grid.addWidget(QLabel("Aktuelle Episode:"), 0, 0)
        self.lbl_episode = QLabel("1")
        self.lbl_episode.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        stats_grid.addWidget(self.lbl_episode, 0, 1)
        
        stats_grid.addWidget(QLabel("Schritte in Episode:"), 1, 0)
        self.lbl_steps = QLabel("0")
        stats_grid.addWidget(self.lbl_steps, 1, 1)
        
        stats_grid.addWidget(QLabel("Episoden-Reward (aktuell):"), 2, 0)
        self.lbl_reward = QLabel("0.00")
        stats_grid.addWidget(self.lbl_reward, 2, 1)
        
        stats_grid.addWidget(QLabel("Explorationsrate (Epsilon ε):"), 3, 0)
        self.lbl_epsilon = QLabel("1.000")
        self.lbl_epsilon.setStyleSheet("font-weight: bold; color: #d35400;")
        stats_grid.addWidget(self.lbl_epsilon, 3, 1)
        
        stats_grid.addWidget(QLabel("Replay Buffer Größe:"), 4, 0)
        self.lbl_buffer = QLabel("0 / 10000")
        stats_grid.addWidget(self.lbl_buffer, 4, 1)
        
        stats_grid.addWidget(QLabel("Letzter Trainings-Loss:"), 5, 0)
        self.lbl_loss = QLabel("N/A")
        stats_grid.addWidget(self.lbl_loss, 5, 1)
        
        stats_grid.addWidget(QLabel("Berechnungs-Device:"), 6, 0)
        lbl_device = QLabel(f"{self.agent.device}".upper())
        lbl_device.setStyleSheet("font-weight: bold; color: #27ae60;" if "cuda" in str(self.agent.device) else "font-weight: bold; color: #7f8c8d;")
        stats_grid.addWidget(lbl_device, 6, 1)
        
        right_panel.addWidget(stats_box)
        
        # 2. Anzeige- & Trainingsmodus (Neue didaktische Modusauswahl)
        mode_box = QGroupBox("Anzeige- & Trainingsmodus")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(10)
        
        self.rad_observe = QRadioButton("Schritt-für-Schritt beobachten (jeden Schritt visualisieren)")
        self.rad_observe.setChecked(True)
        self.rad_observe.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.rad_observe)
        
        self.rad_fast = QRadioButton("Maximal schnell trainieren (UI-Update alle 10 Episoden)")
        self.rad_fast.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.rad_fast)
        
        # Geschwindigkeits-Regler für den Beobachtungsmodus
        speed_layout = QHBoxLayout()
        self.lbl_speed = QLabel("Visualisierungs-Tempo (Schritte pro Sekunde):")
        speed_layout.addWidget(self.lbl_speed)
        
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(1, 50)
        self.spin_speed.setValue(10)  # Standard: 10 Schritte pro Sekunde
        self.spin_speed.valueChanged.connect(self.change_speed)
        speed_layout.addWidget(self.spin_speed)
        
        mode_layout.addLayout(speed_layout)
        right_panel.addWidget(mode_box)
        
        # 3. Trainingssteuerung
        control_box = QGroupBox("Trainings-Aktionen")
        control_layout = QVBoxLayout(control_box)
        control_layout.setSpacing(8)
        
        btn_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("Training Starten")
        self.btn_toggle.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 13px; height: 30px;")
        self.btn_toggle.clicked.connect(self.toggle_training)
        btn_layout.addWidget(self.btn_toggle)
        
        self.btn_step = QPushButton("Einzelschritt")
        self.btn_step.setStyleSheet("height: 30px;")
        self.btn_step.clicked.connect(self.single_step)
        btn_layout.addWidget(self.btn_step)
        
        self.btn_reset = QPushButton("Zurücksetzen")
        self.btn_reset.setStyleSheet("height: 30px;")
        self.btn_reset.clicked.connect(self.reset_agent_and_env)
        btn_layout.addWidget(self.btn_reset)
        
        control_layout.addLayout(btn_layout)
        right_panel.addWidget(control_box)
        
        # 4. Modell-Verwaltung (Speichern / Laden)
        model_box = QGroupBox("Modell-Verwaltung")
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(8)
        
        model_btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Modell laden (.pt)...")
        self.btn_load.clicked.connect(self.load_model_dialog)
        model_btn_layout.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("Modell manuell speichern")
        self.btn_save.clicked.connect(self.save_model_manually)
        model_btn_layout.addWidget(self.btn_save)
        
        model_layout.addLayout(model_btn_layout)
        
        self.lbl_model_status = QLabel("Aktuelles Modell: [Neu trainiert / Nicht gespeichert]")
        self.lbl_model_status.setStyleSheet("font-style: italic; color: #7f8c8d;")
        self.lbl_model_status.setWordWrap(True)
        model_layout.addWidget(self.lbl_model_status)
        
        right_panel.addWidget(model_box)
        
        # 5. Lernkurve (Plot)
        self.plot = LearningCurvePlot()
        right_panel.addWidget(self.plot, stretch=1)
        
        main_layout.addLayout(right_panel, stretch=1)

    # ==============================================================================
    # UI Event-Handler und Logik
    # ==============================================================================
    
    def on_mode_changed(self):
        """Wird aufgerufen, wenn der Benutzer zwischen Beobachtungs- und Schnellmodus umschaltet."""
        if self.rad_observe.isChecked():
            self.observe_mode = True
            self.spin_speed.setEnabled(True)
            self.lbl_speed.setEnabled(True)
            self.update_timer_speed()
            
            # Direkt das UI auf den aktuellen Stand bringen
            self.refresh_visuals()
            self.update_stats_labels()
        else:
            self.observe_mode = False
            self.spin_speed.setEnabled(False)
            self.lbl_speed.setEnabled(False)
            
            # Q-Werte-Kreuz im Schnell-Modus leeren, um Verwirrung zu vermeiden
            self.q_value_widget.clear()
            
            # Im Schnellmodus tickt der Timer sehr schnell (~5ms) und macht dort viele Schritte
            self.timer.setInterval(5)
            
    def change_speed(self):
        """Ändert das Timer-Intervall basierend auf den Schritten pro Sekunde im Beobachtungsmodus."""
        self.update_timer_speed()
        
    def update_timer_speed(self):
        """Setzt das Intervall des QTimers passend zur eingestellten Geschwindigkeit."""
        if self.observe_mode:
            steps_per_sec = self.spin_speed.value()
            interval = int(1000 / steps_per_sec)
            self.timer.setInterval(interval)

    def toggle_training(self):
        """Startet oder pausiert das Training."""
        self.running = not self.running
        if self.running:
            self.btn_toggle.setText("Training Pausieren")
            self.btn_toggle.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; font-size: 13px; height: 30px;")
            self.btn_step.setEnabled(False)
        else:
            self.btn_toggle.setText("Training Fortsetzen")
            self.btn_toggle.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 13px; height: 30px;")
            self.btn_step.setEnabled(True)

    def single_step(self):
        """Führt genau einen Trainingsschritt (Aktion + Lernen) aus."""
        if not self.running:
            self.step_agent_and_learn(force_visuals=True)
            self.refresh_visuals()
            self.update_stats_labels()

    def reset_agent_and_env(self):
        """Setzt das gesamte Experiment zurück."""
        self.running = False
        self.btn_toggle.setText("Training Starten")
        self.btn_toggle.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 13px; height: 30px;")
        self.btn_step.setEnabled(True)
        
        # Environment und Agent neu instanziieren
        self.env = SimpleGridEnv()
        self.agent = DQNAgent(
            obs_shape=(self.env.obs_size, self.env.obs_size, 3),
            num_actions=4
        )
        
        self.state = self.env.reset()
        self.episode_count = 1
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.last_loss = None
        
        # Modus-Zustand re-synchronisieren
        self.observe_mode = self.rad_observe.isChecked()
        self.spin_speed.setEnabled(self.observe_mode)
        self.lbl_speed.setEnabled(self.observe_mode)
        self.update_timer_speed()
        
        # UI zurücksetzen
        self.plot.clear()
        self.q_value_widget.clear()
        self.lbl_model_status.setText("Aktuelles Modell: [Neu trainiert / Nicht gespeichert]")
        
        self.refresh_visuals()
        self.update_stats_labels()
        
    def train_loop_step(self):
        """Timer-Callback: Wird periodisch aufgerufen, um das Training voranzutreiben."""
        if not self.running:
            return
            
        if self.observe_mode:
            # Didaktischer Modus: Ein Schritt pro Timer-Tick, alles sofort rendern
            self.step_agent_and_learn(force_visuals=True)
            self.refresh_visuals()
            self.update_stats_labels()
        else:
            # Schnell-Modus: Viele Schritte pro Timer-Tick im Hintergrund ausführen
            episode_ended = False
            # Führe 50 Schritte pro Tick aus (ausgewogen für maximale Geschwindigkeit bei responsivem UI)
            for _ in range(50):
                ended = self.step_agent_and_learn(force_visuals=False)
                if ended:
                    episode_ended = True
            
            # Im Schnell-Modus aktualisieren wir das UI und die Grafik nur alle 10 Episoden
            if episode_ended and (self.episode_count % 10 == 0):
                self.refresh_visuals()
                self.update_stats_labels()
                self.plot.update_plot()

    def step_agent_and_learn(self, force_visuals: bool = False) -> bool:
        """
        Führt einen einzelnen Schritt aus:
        1. Aktion wählen (Epsilon-Greedy)
        2. Schritt im Environment machen
        3. Transition im Buffer speichern
        4. Netzwerk trainieren
        Gibt True zurück, wenn die Episode beendet ist.
        """
        # 1. Aktion wählen
        action, q_values, is_exploration = self.agent.select_action(self.state, evaluation=False)
        
        # Q-Werte im UI aktualisieren (nur im Beobachtungsmodus)
        if self.observe_mode or force_visuals:
            self.q_value_widget.update_q_values(q_values, action, is_exploration)
            
        # 2. Aktion ausführen
        next_state, reward, done, info = self.env.step(action)
        
        # 3. Transition speichern
        self.agent.store_transition(self.state, action, reward, next_state, done)
        
        # 4. Trainingsschritt ausführen
        loss = self.agent.train_step()
        if loss is not None:
            self.last_loss = loss
            
        # Akkumulieren
        self.state = next_state
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Episode abgeschlossen?
        if done:
            # Daten für Plot hinzufügen (wird im Schnellmodus nur alle 10 Episoden gezeichnet)
            redraw_plot = self.observe_mode or (self.episode_count % 10 == 0)
            self.plot.add_episode_data(self.episode_count, self.episode_reward, redraw=redraw_plot)
            
            # Didaktisches Auto-Save alle 50 Episoden
            if self.episode_count % 50 == 0:
                auto_save_path = os.path.join(self.models_dir, f"dqn_auto_ep_{self.episode_count}.pt")
                self.agent.save_model(auto_save_path)
                self.lbl_model_status.setText(f"Automatisch gesichert in: dqn_auto_ep_{self.episode_count}.pt")
                
            # Zurücksetzen für neue Episode
            self.state = self.env.reset()
            self.episode_count += 1
            self.episode_reward = 0.0
            self.episode_steps = 0
            return True
            
        return False

    def refresh_visuals(self):
        """Aktualisiert die grafische Anzeige des Environments."""
        self.cnn_view.set_observation(self.state)

    def update_stats_labels(self):
        """Aktualisiert die numerischen Status-Anzeigen im UI."""
        self.lbl_episode.setText(str(self.episode_count))
        self.lbl_steps.setText(f"{self.episode_steps} / {self.env.max_steps}")
        self.lbl_reward.setText(f"{self.episode_reward:.2f}")
        self.lbl_epsilon.setText(f"{self.agent.epsilon:.3f}")
        self.lbl_buffer.setText(f"{len(self.agent.memory)} / {self.agent.memory.maxlen}")
        
        if self.last_loss is not None:
            self.lbl_loss.setText(f"{self.last_loss:.5f}")
        else:
            self.lbl_loss.setText("Warte auf Buffer...")

    # ==============================================================================
    # Modell-Speichern und Laden über QFileDialog
    # ==============================================================================

    def save_model_manually(self):
        """Öffnet einen Dialog zum manuellen Speichern des Modells."""
        default_filename = f"dqn_model_ep_{self.episode_count}.pt"
        default_path = os.path.join(self.models_dir, default_filename)
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Modell speichern",
            default_path,
            "PyTorch Model Files (*.pt *.pth)"
        )
        
        if filepath:
            try:
                self.agent.save_model(filepath)
                filename = os.path.basename(filepath)
                self.lbl_model_status.setText(f"Erfolgreich manuell gespeichert: {filename}")
            except Exception as e:
                self.lbl_model_status.setText(f"Fehler beim Speichern: {str(e)}")

    def load_model_dialog(self):
        """Öffnet einen Dialog zum Laden eines gespeicherten Modells."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Modell laden",
            self.models_dir,
            "PyTorch Model Files (*.pt *.pth)"
        )
        
        if filepath:
            try:
                self.running = False
                self.btn_toggle.setText("Training Starten")
                self.btn_toggle.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 13px; height: 30px;")
                self.btn_step.setEnabled(True)
                
                self.agent.load_model(filepath)
                filename = os.path.basename(filepath)
                
                self.lbl_model_status.setText(f"Erfolgreich geladen: {filename} (ε wurde auf {self.agent.epsilon:.3f} gesetzt)")
                
                self.state = self.env.reset()
                self.episode_reward = 0.0
                self.episode_steps = 0
                
                self.refresh_visuals()
                self.update_stats_labels()
                
            except Exception as e:
                self.lbl_model_status.setText(f"Fehler beim Laden: {str(e)}")


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================

def main():
    app = QApplication(sys.argv)
    
    # App-weites, ansprechendes Flat-Design Styling (Fusion Style)
    app.setStyle("Fusion")
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
