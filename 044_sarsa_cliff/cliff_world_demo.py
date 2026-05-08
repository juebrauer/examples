import sys
import time
import numpy as np
from enum import Enum
from typing import Tuple
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QKeyEvent, QShortcut
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Action(Enum):
    """Possible actions in the grid world"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CliffEnvironment:
    """4x12 Cliff Gridworld environment"""
    
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_pos = (self.rows - 1, 0)  # Bottom-left (0,0)
        self.goal_pos = (self.rows - 1, self.cols - 1)  # Bottom-right
        self.cliff_positions = set(
            (self.rows - 1, c) for c in range(1, self.cols - 1)
        )
        self.current_pos = self.start_pos
        
    def reset(self):
        """Reset agent to start position"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute one step in the environment
        Returns: (new_position, reward, done)
        """
        row, col = self.current_pos
        
        # Calculate next position
        if action == Action.UP:
            row = max(0, row - 1)
        elif action == Action.DOWN:
            row = min(self.rows - 1, row + 1)
        elif action == Action.LEFT:
            col = max(0, col - 1)
        elif action == Action.RIGHT:
            col = min(self.cols - 1, col + 1)
        
        next_pos = (row, col)
        
        # Check rewards
        done = False
        if next_pos == self.goal_pos:
            reward = 1.0
            done = True
        elif next_pos in self.cliff_positions:
            reward = -1.0
            done = True
        else:
            reward = -0.01  # Small step penalty
        
        self.current_pos = next_pos
        return next_pos, reward, done


class QLearningAgent:
    """Q-Learning agent"""
    
    def __init__(self, env: CliffEnvironment, learning_rate=0.1, 
                 discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros(
            (env.rows, env.cols, len(Action))
        )
        
    def get_action(self, pos: Tuple[int, int], training=True) -> Action:
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return Action(np.random.randint(0, len(Action)))
        
        row, col = pos
        best_action = np.argmax(self.Q[row, col, :])
        return Action(best_action)
    
    def learn_step(self, training=True) -> Tuple[float, bool]:
        """Execute one learning step. Returns (reward, done)"""
        pos = self.env.current_pos
        action = self.get_action(pos, training=training)
        
        row, col = pos
        next_pos, reward, done = self.env.step(action)
        next_row, next_col = next_pos
        
        # Q-Learning update
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(
                self.Q[next_row, next_col, :]
            )
        
        self.Q[row, col, action.value] += self.learning_rate * (
            target - self.Q[row, col, action.value]
        )
        
        if done:
            self.env.reset()
        
        return reward, done
    
    def get_policy(self) -> np.ndarray:
        """Get current policy as array of action indices"""
        return np.argmax(self.Q, axis=2)


class SARSAAgent:
    """SARSA (State-Action-Reward-State-Action) agent"""
    
    def __init__(self, env: CliffEnvironment, learning_rate=0.1,
                 discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros(
            (env.rows, env.cols, len(Action))
        )
        
    def get_action(self, pos: Tuple[int, int], training=True) -> Action:
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return Action(np.random.randint(0, len(Action)))
        
        row, col = pos
        best_action = np.argmax(self.Q[row, col, :])
        return Action(best_action)
    
    def learn_step(self, training=True) -> Tuple[float, bool]:
        """Execute one learning step. Returns (reward, done)"""
        pos = self.env.current_pos
        action = self.get_action(pos, training=training)
        
        row, col = pos
        next_pos, reward, done = self.env.step(action)
        next_row, next_col = next_pos
        
        # SARSA update (uses next action, not max)
        if done:
            target = reward
        else:
            next_action = self.get_action(next_pos, training=training)
            target = reward + self.discount_factor * self.Q[
                next_row, next_col, next_action.value
            ]
        
        self.Q[row, col, action.value] += self.learning_rate * (
            target - self.Q[row, col, action.value]
        )
        
        if done:
            self.env.reset()
        
        return reward, done
    
    def get_policy(self) -> np.ndarray:
        """Get current policy as array of action indices"""
        return np.argmax(self.Q, axis=2)


class RewardPlot(FigureCanvas):
    """Matplotlib canvas for plotting episode rewards"""
    
    def __init__(self):
        self.figure = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.episode_rewards = []
        self.init_plot()
    
    def init_plot(self):
        """Initialize the plot"""
        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Reward per Episode")
        self.figure.tight_layout()
    
    def update_plot(self, episode_rewards: list):
        """Update the plot with new episode rewards"""
        self.episode_rewards = episode_rewards
        self.ax.clear()
        
        if self.episode_rewards:
            episodes = list(range(1, len(self.episode_rewards) + 1))
            colors = ['green' if r > 0 else 'red' for r in self.episode_rewards]
            self.ax.bar(episodes, self.episode_rewards, color=colors, alpha=0.7)
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.ax.set_title("Reward per Episode")
        else:
            self.ax.text(0.5, 0.5, "No episodes yet", 
                        ha='center', va='center', transform=self.ax.transAxes)
        
        self.figure.tight_layout()
        self.draw()


class GridCanvas(QWidget):
    """Canvas for drawing the grid world"""
    
    # Direction symbols
    ARROWS = {
        0: "↑",  # UP
        1: "↓",  # DOWN
        2: "←",  # LEFT
        3: "→",  # RIGHT
    }
    
    def __init__(self, env: CliffEnvironment):
        super().__init__()
        self.env = env
        self.policy = np.zeros((env.rows, env.cols), dtype=int)
        self.agent_pos = env.start_pos
        
        self.cell_size = 60
        self.setMinimumSize(
            self.cell_size * env.cols + 20,
            self.cell_size * env.rows + 20
        )
    
    def update_policy(self, policy: np.ndarray):
        """Update the policy display"""
        self.policy = policy
        self.update()
    
    def update_agent_pos(self, pos: Tuple[int, int]):
        """Update agent position"""
        self.agent_pos = pos
        self.update()
    
    def paintEvent(self, event):
        """Paint the grid world"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw grid cells
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                x = col * self.cell_size + 10
                y = row * self.cell_size + 10
                
                pos = (row, col)
                
                # Determine cell color
                if pos == self.env.start_pos:
                    color = QColor(100, 200, 100)  # Green
                    label = "S"
                elif pos == self.env.goal_pos:
                    color = QColor(100, 150, 255)  # Blue
                    label = "G"
                elif pos in self.env.cliff_positions:
                    color = QColor(200, 50, 50)  # Red
                    label = "C"
                else:
                    color = QColor(220, 220, 220)  # Light gray
                    label = ""
                
                # Draw cell
                painter.fillRect(x, y, self.cell_size, self.cell_size, color)
                painter.drawRect(x, y, self.cell_size, self.cell_size)
                
                # Draw label or arrow
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                if label:
                    painter.drawText(
                        x, y, self.cell_size, self.cell_size,
                        Qt.AlignCenter, label
                    )
                else:
                    action_idx = self.policy[row, col]
                    arrow = self.ARROWS.get(action_idx, "?")
                    painter.drawText(
                        x, y, self.cell_size, self.cell_size,
                        Qt.AlignCenter, arrow
                    )
                
                # Draw agent if at this position
                if pos == self.agent_pos:
                    painter.fillRect(
                        x + 5, y + 5, self.cell_size - 10,
                        self.cell_size - 10, QColor(255, 200, 0)
                    )


class CliffWorldDemo(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cliff Gridworld: Q-Learning vs SARSA")
        self.setGeometry(100, 100, 1400, 700)
        
        # Initialize environment and agents
        self.env = CliffEnvironment()
        self.current_agent = None
        self.agent_type = "QLearning"
        self.is_learning = False
        self.exploration_rate = 0.10
        # Keep each timer callback short so UI interactions stay responsive.
        self.max_compute_ms_per_tick = 6
        self.max_steps_per_tick = 5000
        self.step_count = 0
        self.episode_count = 0
        self.cliff_fall_count = 0
        self.current_episode_reward = 0.0
        self.episode_rewards = []
        self.last_plotted_episode_count = -1
        
        # Create UI
        self.init_ui()
        
        # Timer for automatic learning
        self.learn_timer = QTimer()
        self.learn_timer.timeout.connect(self.learn_step)

        # Window-level shortcut so SPACE works regardless of focused child widget.
        self.space_shortcut = QShortcut(Qt.Key_Space, self)
        self.space_shortcut.setContext(Qt.WindowShortcut)
        self.space_shortcut.activated.connect(self.perform_single_step)
        
        # Initialize with Q-Learning
        self.switch_agent("QLearning")
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left side: Canvas + Reward Plot
        left_layout = QVBoxLayout()
        self.canvas = GridCanvas(self.env)
        left_layout.addWidget(self.canvas, 1)
        
        self.reward_plot = RewardPlot()
        left_layout.addWidget(self.reward_plot, 1)
        
        # Right side: Controls
        right_layout = QVBoxLayout()
        
        # Agent type selection
        agent_group = QGroupBox("Agent Type")
        agent_layout = QVBoxLayout()
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(["Q-Learning", "SARSA"])
        self.agent_combo.currentTextChanged.connect(self.on_agent_changed)
        agent_layout.addWidget(self.agent_combo)
        agent_group.setLayout(agent_layout)
        right_layout.addWidget(agent_group)
        
        # Learning controls
        control_group = QGroupBox("Learning Controls")
        control_layout = QVBoxLayout()
        
        self.toggle_button = QPushButton("Start Learning")
        self.toggle_button.clicked.connect(self.toggle_learning)
        control_layout.addWidget(self.toggle_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_learning)
        control_layout.addWidget(self.reset_button)

        # Constant exploration rate (epsilon)
        epsilon_label = QLabel("Exploration Rate epsilon:")
        control_layout.addWidget(epsilon_label)

        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 1.0)
        self.epsilon_spinbox.setSingleStep(0.01)
        self.epsilon_spinbox.setDecimals(2)
        self.epsilon_spinbox.setValue(self.exploration_rate)
        self.epsilon_spinbox.valueChanged.connect(self.on_exploration_rate_changed)
        control_layout.addWidget(self.epsilon_spinbox)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0")
        stats_layout.addWidget(self.episode_label)
        
        self.step_label = QLabel("Learning Steps: 0")
        stats_layout.addWidget(self.step_label)
        
        self.episode_reward_label = QLabel("Episode Reward: 0.00")
        stats_layout.addWidget(self.episode_reward_label)

        self.cliff_fall_count_label = QLabel("Cliff Falls: 0")
        self.cliff_fall_count_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        stats_layout.addWidget(self.cliff_fall_count_label)

        self.cliff_fall_rate_label = QLabel("Cliff Fall Rate: 0.0%")
        self.cliff_fall_rate_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        stats_layout.addWidget(self.cliff_fall_rate_label)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Info
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout()
        info_label = QLabel(
            "Press SPACE to perform one learning step\n\n"
            "Green: Start (S)\n"
            "Blue: Goal (G)\n"
            "Red: Cliff (C)\n"
            "Yellow: Agent\n\n"
            "Rewards:\n"
            "+1: Reach goal\n"
            "-1: Fall in cliff\n"
            "-0.01: Each step"
        )
        info_label.setFont(QFont("Courier", 9))
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        right_layout.addStretch()
        
        # Combine layouts
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        central_widget.setLayout(main_layout)
    
    def on_agent_changed(self, text: str):
        """Handle agent type change"""
        if text == "Q-Learning":
            self.switch_agent("QLearning")
        else:
            self.switch_agent("SARSA")
    
    def on_exploration_rate_changed(self, value: float):
        """Handle epsilon changes; keep it constant unless user changes it."""
        self.exploration_rate = float(value)
        if self.current_agent is not None:
            self.current_agent.epsilon = self.exploration_rate
    
    def switch_agent(self, agent_type: str):
        """Switch to a different agent"""
        if self.is_learning:
            self.stop_learning()
        
        self.agent_type = agent_type
        self.env.reset()
        
        if agent_type == "QLearning":
            self.current_agent = QLearningAgent(self.env, epsilon=self.exploration_rate)
        else:
            self.current_agent = SARSAAgent(self.env, epsilon=self.exploration_rate)
        
        self.step_count = 0
        self.episode_count = 0
        self.cliff_fall_count = 0
        self.current_episode_reward = 0.0
        self.episode_rewards = []
        self.last_plotted_episode_count = -1
        self.update_display(force_plot_update=True)
    
    def toggle_learning(self):
        """Toggle learning on/off"""
        if self.is_learning:
            self.stop_learning()
        else:
            self.start_learning()
    
    def start_learning(self):
        """Start automatic learning"""
        self.is_learning = True
        self.toggle_button.setText("Stop Learning")
        self.agent_combo.setEnabled(False)
        self.learn_timer.start(0)
    
    def stop_learning(self):
        """Stop automatic learning"""
        self.is_learning = False
        self.learn_timer.stop()
        self.toggle_button.setText("Start Learning")
        self.agent_combo.setEnabled(True)
    
    def reset_learning(self):
        """Reset the learning process"""
        self.switch_agent(self.agent_type)
    
    def learn_step(self):
        """Perform a high-speed batch of learning steps."""
        if self.current_agent is None:
            return

        tick_start = time.perf_counter()
        steps_done = 0

        while steps_done < self.max_steps_per_tick:
            reward, done = self.current_agent.learn_step(training=True)
            self.step_count += 1
            self.current_episode_reward += reward
            steps_done += 1

            if done:
                self.episode_count += 1
                if reward < 0.0:
                    self.cliff_fall_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0.0
                # Always refresh UI at episode boundaries.
                self.update_display(force_plot_update=True)
                # Return control to Qt after each finished episode.
                break

            elapsed_ms = (time.perf_counter() - tick_start) * 1000.0
            if elapsed_ms >= self.max_compute_ms_per_tick:
                break

        # Always visualize current state once per timer callback.
        if steps_done > 0:
            self.update_display()
    
    def update_display(self, force_plot_update: bool = False):
        """Update the display"""
        if self.current_agent is None:
            return
        
        # Update policy
        policy = self.current_agent.get_policy()
        self.canvas.update_policy(policy)
        
        # Update agent position
        self.canvas.update_agent_pos(self.env.current_pos)
        
        # Update statistics
        self.episode_label.setText(f"Episode: {self.episode_count}")
        self.step_label.setText(f"Learning Steps: {self.step_count}")
        self.episode_reward_label.setText(f"Episode Reward: {self.current_episode_reward:.2f}")
        cliff_fall_rate = 0.0
        if self.episode_count > 0:
            cliff_fall_rate = 100.0 * self.cliff_fall_count / self.episode_count
        self.cliff_fall_count_label.setText(f"Cliff Falls: {self.cliff_fall_count}")
        self.cliff_fall_rate_label.setText(f"Cliff Fall Rate: {cliff_fall_rate:.1f}%")
        
        # Redrawing matplotlib every step is expensive; only update on new episodes
        # or when explicitly requested.
        needs_plot_update = force_plot_update or (
            self.last_plotted_episode_count != len(self.episode_rewards)
        )
        if needs_plot_update:
            self.reward_plot.update_plot(self.episode_rewards)
            self.last_plotted_episode_count = len(self.episode_rewards)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.perform_single_step()
        else:
            super().keyPressEvent(event)

    def perform_single_step(self):
        """Execute exactly one manual learning step."""
        if self.current_agent is None:
            return

        reward, done = self.current_agent.learn_step(training=True)
        self.step_count += 1
        self.current_episode_reward += reward

        if done:
            self.episode_count += 1
            if reward < 0.0:
                self.cliff_fall_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        self.update_display(force_plot_update=done)


def main():
    app = QApplication(sys.argv)
    window = CliffWorldDemo()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
