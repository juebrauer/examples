from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

State = np.ndarray


@dataclass
class Transition:
    state: State
    action: int
    reward: float
    next_state: State
    done: bool


class UniformReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = 4
        self.gamma = 0.98
        self.batch_size = 64
        self.learning_starts = 800
        self.target_update_interval = 500
        self.replay = UniformReplayBuffer(capacity=30_000)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 25_000
        self.epsilon = self.epsilon_start

        self.policy_net = MLPQNetwork(4, self.num_actions).to(self.device)
        self.target_net = MLPQNetwork(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.env_steps = 0
        self.training_steps = 0

    def register_env_step(self) -> None:
        self.env_steps += 1
        progress = min(1.0, self.env_steps / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, state: State, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(x)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        self.replay.add(
            Transition(
                state=state.copy(),
                action=int(action),
                reward=float(reward),
                next_state=next_state.copy(),
                done=bool(done),
            )
        )

    def learn(self, episode_done: bool = False) -> float | None:
        del episode_done

        if len(self.replay) < max(self.batch_size, self.learning_starts):
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())
