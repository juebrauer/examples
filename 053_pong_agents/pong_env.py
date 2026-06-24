import gymnasium as gym
import ale_py
import numpy as np

# Für neuere Gymnasium-Versionen (>= 1.0) müssen die Atari-Umgebungen
# explizit registriert werden:
gym.register_envs(ale_py)

class PongEnv:
    def __init__(self):
        # We use rgb_array to be able to render the frames in our GUI
        # We also pass frameskip=1 to disable the built-in frame skipping, 
        # so AtariPreprocessing can handle it without throwing a ValueError.
        self._base_env = gym.make('ALE/Pong-v5', render_mode='rgb_array', frameskip=1)
        
        # Standard Atari Wrappers for DQN:
        # 1. AtariPreprocessing: Grayscale, resize to 84x84, max pooling over last 2 frames, scale to [0,1], frame skip=4
        self.env = gym.wrappers.AtariPreprocessing(self._base_env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=True)
        # 2. FrameStack: Stack the last 4 frames so the agent can see motion
        self.env = gym.wrappers.FrameStackObservation(self.env, stack_size=4)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.current_state = None
        self.current_frame = None

    def reset(self):
        # Reset the wrapped environment (returns stacked 84x84 frames)
        self.current_state, info = self.env.reset()
        # Render the base environment to get the raw RGB 210x160x3 frame for the GUI
        self.current_frame = self._base_env.render()
        return self.current_state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.current_state = next_state
        self.current_frame = self._base_env.render()
        return next_state, reward, terminated, truncated, info

    def get_frame(self):
        """Returns the current raw RGB frame (210x160x3) as a numpy array."""
        return self.current_frame

    def get_action_space(self):
        return self.action_space

    def close(self):
        self.env.close()
