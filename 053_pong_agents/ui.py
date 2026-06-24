import os
import collections
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QGroupBox, QSlider, QCheckBox,
    QHBoxLayout as QHBox, QComboBox, QFileDialog, QSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from worker import RLWorker
from experiment import ExperimentWorker

class GameView(QWidget):
    """Custom widget to render the game frame perfectly scaled and centered."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self._current_frame = None

    def set_frame(self, frame):
        self._current_frame = np.ascontiguousarray(frame)
        self.update()

    def paintEvent(self, event):
        if self._current_frame is None:
            painter = QPainter(self)
            painter.drawText(self.rect(), Qt.AlignCenter, "Game will be rendered here")
            return

        h, w, ch = self._current_frame.shape
        bytes_per_line = ch * w
        
        q_img = QImage(self._current_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.width(), 
            self.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        painter = QPainter(self)
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)


class LearningCurvePlot(FigureCanvas):
    """Matplotlib plot to visualize the raw and average reward."""
    def __init__(self, parent=None, max_steps=1000):
        self.fig = Figure(figsize=(4, 2.5), dpi=100)
        self.fig.patch.set_facecolor('#f9f9f9')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.max_steps = max_steps
        
        self.step_history = []
        self.raw_history = []
        self.avg_history = []
        self.reward_buffer = collections.deque(maxlen=max_steps)
        
        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()
        self.ax.set_title(f"Reward (Raw vs Avg {self.max_steps} steps)", fontsize=10)
        self.ax.set_xlabel("Steps", fontsize=8)
        self.ax.set_ylabel("Reward", fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.draw_idle()

    def add_step_reward(self, step: int, reward: float):
        self.reward_buffer.append(reward)
        avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
        
        self.step_history.append(step)
        self.raw_history.append(reward)
        self.avg_history.append(avg_reward)
        
        if len(self.step_history) > 5000:
            self.step_history.pop(0)
            self.raw_history.pop(0)
            self.avg_history.pop(0)

    def update_plot(self):
        if not self.step_history:
            return
        self.ax.clear()
        self.ax.set_title(f"Reward (Raw vs Avg {self.max_steps} steps)", fontsize=10)
        self.ax.set_xlabel("Steps", fontsize=8)
        self.ax.set_ylabel("Reward", fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.ax.plot(self.step_history, self.raw_history, color='black', linewidth=1, alpha=0.7, label='Raw')
        self.ax.plot(self.step_history, self.avg_history, color='lightgray', linewidth=2, label='Avg')
        
        self.ax.legend(loc='upper left', fontsize=6)
        self.fig.tight_layout()
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, env, agent_factories):
        super().__init__()
        self.setWindowTitle("Pong Agent Framework (Threaded)")
        self.resize(1000, 700)

        self.env = env
        self.agent_factories = agent_factories

        initial_agent_name = list(agent_factories.keys())[0]
        self.current_agent = agent_factories[initial_agent_name]()

        self.worker = RLWorker(self.env, self.current_agent)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.speed_updated.connect(self.on_speed_updated)
        
        # State for manual environment
        self.last_ep = 0
        self.last_ep_reward = 0.0

        self.init_ui()
        
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.learning_plot.update_plot)
        self.plot_timer.start(500)
        
        self.worker.reset_env()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        self.image_label = GameView()
        main_layout.addWidget(self.image_label, stretch=2)

        ui_panel = QVBoxLayout()
        main_layout.addLayout(ui_panel, stretch=1)

        # Agent Selection
        agent_group = QGroupBox("Agent Selection")
        agent_layout = QVBoxLayout()
        agent_group.setLayout(agent_layout)
        
        self.combo_agent = QComboBox()
        self.combo_agent.addItems(list(self.agent_factories.keys()))
        self.combo_agent.currentTextChanged.connect(self.on_agent_changed)
        agent_layout.addWidget(self.combo_agent)
        
        self.btn_load_model = QPushButton("Load Model (.pth)")
        self.btn_load_model.clicked.connect(self.load_model)
        agent_layout.addWidget(self.btn_load_model)
        
        self.lbl_loaded_model = QLabel("No custom model loaded")
        self.lbl_loaded_model.setStyleSheet("color: gray; font-style: italic;")
        agent_layout.addWidget(self.lbl_loaded_model)
        
        ui_panel.addWidget(agent_group)

        # Controls Group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.toggle_start)
        controls_layout.addWidget(self.btn_start)

        self.btn_step = QPushButton("Next Step")
        self.btn_step.clicked.connect(self.single_step)
        controls_layout.addWidget(self.btn_step)

        self.btn_reset = QPushButton("Reset Episode")
        self.btn_reset.clicked.connect(self.reset_env)
        controls_layout.addWidget(self.btn_reset)

        options_layout = QVBoxLayout()
        self.chk_skip_render = QCheckBox("Skip Rendering (Max Speed)")
        self.chk_skip_render.stateChanged.connect(self.on_skip_render_changed)
        options_layout.addWidget(self.chk_skip_render)

        speed_layout = QHBox()
        speed_label = QLabel("Delay (ms):")
        speed_layout.addWidget(speed_label)
        
        self.slider_delay = QSlider(Qt.Horizontal)
        self.slider_delay.setRange(0, 500)
        self.slider_delay.setValue(16)
        self.slider_delay.setTickPosition(QSlider.TicksBelow)
        self.slider_delay.setTickInterval(50)
        self.slider_delay.valueChanged.connect(self.on_delay_changed)
        speed_layout.addWidget(self.slider_delay)
        
        self.lbl_delay_val = QLabel("16")
        speed_layout.addWidget(self.lbl_delay_val)
        
        options_layout.addLayout(speed_layout)
        controls_layout.addLayout(options_layout)

        ui_panel.addWidget(controls_group)
        
        # Experiments
        exp_group = QGroupBox("Automated Experiments")
        exp_layout = QVBoxLayout()
        exp_group.setLayout(exp_layout)
        
        box_layout = QHBox()
        box_layout.addWidget(QLabel("Max Steps per Agent:"))
        self.spin_max_steps = QSpinBox()
        self.spin_max_steps.setRange(1000, 10000000)
        self.spin_max_steps.setSingleStep(50000)
        self.spin_max_steps.setValue(500000)
        box_layout.addWidget(self.spin_max_steps)
        exp_layout.addLayout(box_layout)
        
        self.btn_run_exp = QPushButton("Run Experiments (DQN -> PPO)")
        self.btn_run_exp.clicked.connect(self.start_experiments)
        exp_layout.addWidget(self.btn_run_exp)
        
        self.lbl_exp_progress = QLabel("Progress: Not running")
        self.lbl_exp_progress.setStyleSheet("color: darkgreen;")
        exp_layout.addWidget(self.lbl_exp_progress)
        
        ui_panel.addWidget(exp_group)

        # Stats Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)

        self.lbl_episode = QLabel("Episode: 0")
        stats_layout.addWidget(self.lbl_episode)

        self.lbl_reward = QLabel("Current Reward: 0.0")
        stats_layout.addWidget(self.lbl_reward)

        self.lbl_steps = QLabel("Total Steps: 0")
        stats_layout.addWidget(self.lbl_steps)
        
        self.lbl_epsilon = QLabel("Epsilon: 1.0 (Random)")
        stats_layout.addWidget(self.lbl_epsilon)
        
        self.lbl_speed = QLabel("Speed: 0 steps/h")
        self.lbl_speed.setStyleSheet("color: blue;")
        stats_layout.addWidget(self.lbl_speed)

        ui_panel.addWidget(stats_group)
        
        self.learning_plot = LearningCurvePlot(max_steps=1000)
        ui_panel.addWidget(self.learning_plot)

        ui_panel.addStretch()

    def load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Model", "models/", "PyTorch Models (*.pth)")
        if filepath:
            try:
                self.current_agent.load(filepath)
                filename = os.path.basename(filepath)
                self.lbl_loaded_model.setText(f"Loaded: {filename}")
                self.lbl_loaded_model.setStyleSheet("color: green; font-weight: bold;")
                self.reset_env()
                QMessageBox.information(self, "Success", f"Model '{filename}' loaded into {self.current_agent.name}!\nSimulation has been reset.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def on_agent_changed(self, agent_name):
        was_running = self.worker._is_running
        if was_running:
            self.toggle_start()
            
        # Memory cleanup before instantiating the new agent to prevent OOM
        if hasattr(self, 'current_agent'):
            del self.current_agent
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self.current_agent = self.agent_factories[agent_name]()
        self.worker.set_agent(self.current_agent)
        
        # Reset loaded model label
        self.lbl_loaded_model.setText("No custom model loaded")
        self.lbl_loaded_model.setStyleSheet("color: gray; font-style: italic;")
        
        self.update_epsilon_label()
        self.reset_env()

    def update_epsilon_label(self):
        if hasattr(self.current_agent, 'epsilon'):
            self.lbl_epsilon.setText(f"Epsilon: {self.current_agent.epsilon:.3f}")
        else:
            self.lbl_epsilon.setText(f"Epsilon: N/A ({self.current_agent.name})")

    def on_speed_updated(self, steps_per_hour):
        self.lbl_speed.setText(f"Speed: ~{int(steps_per_hour):,} steps/h")

    def on_frame_ready(self, frame):
        self.image_label.set_frame(frame)

    def on_stats_updated(self, agent_name, episode, steps, current_ep_reward, step_reward):
        # Ignore stale signals from the previous agent still in the event queue
        if agent_name != self.current_agent.name:
            return

        self.lbl_episode.setText(f"Episode: {episode}")
        self.lbl_steps.setText(f"Total Steps: {steps}")
        self.lbl_reward.setText(f"Current Reward: {current_ep_reward}")
        
        self.learning_plot.add_step_reward(steps, step_reward)
        self.update_epsilon_label()
        
        self.last_ep_reward = current_ep_reward

    def start_experiments(self):
        max_steps = self.spin_max_steps.value()
        
        # Lock UI
        self.btn_start.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.combo_agent.setEnabled(False)
        self.btn_run_exp.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        
        # Pause main worker
        if self.worker._is_running:
            self.toggle_start()
            
        self.lbl_exp_progress.setText(f"Progress: Initializing...")
        
        # Start dedicated background worker
        self.exp_worker = ExperimentWorker(self.agent_factories, max_steps)
        self.exp_worker.progress_updated.connect(self.on_exp_progress)
        self.exp_worker.finished_experiment.connect(self.on_exp_finished)
        self.exp_worker.start()

    def on_exp_progress(self, agent_name, step, max_steps, speed):
        self.lbl_exp_progress.setText(f"Progress: {agent_name} ({step}/{max_steps}) @ ~{int(speed):,} steps/h")

    def on_exp_finished(self, results):
        self.lbl_exp_progress.setText("Progress: Finished!")
        
        # Unlock UI
        self.btn_start.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.combo_agent.setEnabled(True)
        self.btn_run_exp.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        
        self.write_experiment_results(results)
        QMessageBox.information(self, "Experiments Finished", "Results have been written to results.md!")

    def write_experiment_results(self, results):
        max_steps = self.spin_max_steps.value()
        num_chunks = (max_steps - 1) // 100000 + 1
        
        with open("results.md", "w") as f:
            f.write("# Pong Experiment Results\n\n")
            f.write(f"**Training Steps:** {max_steps} per Agent\n\n")
            
            # Header
            f.write("| Agent | Episodes Played | Won | Lost | Total Rewards |")
            for c in range(num_chunks):
                f.write(f" Wins ({c*100}k-{(c+1)*100}k) |")
            f.write("\n")
            
            # Separator
            f.write("| --- | --- | --- | --- | --- |")
            for c in range(num_chunks):
                f.write(" --- |")
            f.write("\n")
            
            # Data
            for agent_name, stats in results.items():
                f.write(f"| {agent_name} | {stats['episodes_played']} | {stats['episodes_won']} | {stats['episodes_lost']} | {stats['total_reward']:.1f} |")
                for c in range(num_chunks):
                    chunk_wins = stats['chunk_stats'].get(c, {}).get('won', 0)
                    f.write(f" {chunk_wins} |")
                f.write("\n")

    def on_skip_render_changed(self, state):
        self.worker.set_skip_rendering(bool(state))

    def on_delay_changed(self, value):
        self.lbl_delay_val.setText(str(value))
        self.worker.set_delay(value)

    def reset_env(self):
        if self.worker._is_running:
            self.toggle_start()
        self.last_ep = 0
        self.last_ep_reward = 0.0
        self.worker.reset_env()

    def toggle_start(self):
        if self.worker._is_running:
            self.worker.pause_loop()
            self.btn_start.setText("Start")
            self.btn_step.setEnabled(True)
        else:
            self.worker.start_loop()
            self.btn_start.setText("Pause")
            self.btn_step.setEnabled(False)

    def single_step(self):
        self.worker.request_single_step()

    def closeEvent(self, event):
        self.worker.pause_loop()
        self.worker.wait()
        super().closeEvent(event)
