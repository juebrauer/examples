import time
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

class RLWorker(QThread):
    # Signals to communicate with the main thread
    frame_ready = Signal(np.ndarray)
    # agent_name, episode_count, total_steps, current_episode_reward, step_reward
    stats_updated = Signal(str, int, int, float, float)
    speed_updated = Signal(float) # steps per hour

    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent

        self.mutex = QMutex()
        self._is_running = False
        self._single_step = False
        
        # User configurations
        self.delay_ms = 16  # default roughly 60 FPS
        self.skip_rendering = False

        # State tracking
        self.current_state = None
        self.episode_count = 0
        self.total_steps = 0
        self.current_reward = 0.0

    def set_agent(self, agent):
        with QMutexLocker(self.mutex):
            self.agent = agent

    def set_delay(self, ms):
        with QMutexLocker(self.mutex):
            self.delay_ms = ms

    def set_skip_rendering(self, skip):
        with QMutexLocker(self.mutex):
            self.skip_rendering = skip

    def start_loop(self):
        with QMutexLocker(self.mutex):
            self._is_running = True
        self.start()

    def pause_loop(self):
        with QMutexLocker(self.mutex):
            self._is_running = False

    def request_single_step(self):
        with QMutexLocker(self.mutex):
            self._single_step = True
        if not self.isRunning():
            self.start()

    def reset_env(self):
        with QMutexLocker(self.mutex):
            self.current_state, _ = self.env.reset()
            self.current_reward = 0.0
            
            # Emit initial state if not skipping
            if not self.skip_rendering:
                frame = self.env.get_frame()
                if frame is not None:
                    self.frame_ready.emit(frame)
            if hasattr(self, 'agent') and self.agent is not None:
                agent_name = self.agent.name
            else:
                agent_name = "Unknown"
            self.stats_updated.emit(agent_name, self.episode_count, self.total_steps, self.current_reward, 0.0)

    def run(self):
        # Initial reset if not initialized
        if self.current_state is None:
            self.reset_env()

        last_time = time.time()
        last_calc_steps = self.total_steps

        while True:
            # Check loop conditions safely
            self.mutex.lock()
            is_running = self._is_running
            single_step = self._single_step
            self.mutex.unlock()

            if not is_running and not single_step:
                break # Exit thread loop if paused and no single step requested

            if single_step:
                with QMutexLocker(self.mutex):
                    self._single_step = False

            # --- RL Step ---
            action = self.agent.get_action(self.current_state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Allow the agent to store the transition and train
            if hasattr(self.agent, 'step_end'):
                self.agent.step_end(reward, next_state, done)
            
            self.agent.train_step()
            
            with QMutexLocker(self.mutex):
                self.current_state = next_state
                self.current_reward += reward
                self.total_steps += 1
                
                # Check episode end
                if done:
                    self.episode_count += 1
                    self.current_state, _ = self.env.reset()
                    self.current_reward = 0.0

                delay = self.delay_ms
                skip = self.skip_rendering

            # --- Rendering and Stats ---
            with QMutexLocker(self.mutex):
                self.stats_updated.emit(self.agent.name, self.episode_count, self.total_steps, self.current_reward, reward)
            
            if not skip:
                frame = self.env.get_frame()
                if frame is not None:
                    self.frame_ready.emit(frame)

            # --- Speed Calculation ---
            if self.total_steps - last_calc_steps >= 1000:
                current_time = time.time()
                delta = current_time - last_time
                if delta > 0:
                    steps_per_sec = (self.total_steps - last_calc_steps) / delta
                    steps_per_hour = steps_per_sec * 3600
                    self.speed_updated.emit(steps_per_hour)
                last_time = current_time
                last_calc_steps = self.total_steps

            # --- Delay ---
            if delay > 0 and is_running:
                # We use msleep from QThread for simple sleeping without blocking main thread
                QThread.msleep(delay)
