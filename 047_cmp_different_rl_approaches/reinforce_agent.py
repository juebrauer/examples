from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

State = np.ndarray


class PolicyNetwork(nn.Module):
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


class ReinforceAgent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = 4
        self.alpha = 3e-4
        self.gamma = 0.98
        self.epsilon = 0.0

        self.policy_net = PolicyNetwork(4, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

        self.episode_states: List[State] = []
        self.episode_actions: List[int] = []
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []

        self.env_steps = 0
        self.training_steps = 0
        self._pending_log_prob: torch.Tensor | None = None

    def register_env_step(self) -> None:
        self.env_steps += 1

    def act(self, state: State, greedy: bool = False) -> int:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(x)
        dist = Categorical(logits=logits)

        if greedy:
            action_tensor = torch.argmax(logits, dim=1).squeeze(0)
        else:
            action_tensor = dist.sample().squeeze(0)

        self._pending_log_prob = dist.log_prob(action_tensor)
        return int(action_tensor.item())

    def store(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        del next_state, done
        if self._pending_log_prob is None:
            return

        self.episode_states.append(state.copy())
        self.episode_actions.append(int(action))
        self.episode_log_probs.append(self._pending_log_prob)
        self.episode_rewards.append(float(reward))
        self._pending_log_prob = None

    def learn(self, episode_done: bool = False) -> float | None:
        if not episode_done or not self.episode_rewards:
            return None

        # REINFORCE pseudocode (mirrors the lecture structure):
        # 1) Sample one complete trajectory tau.
        # 2) For every t, compute R_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        # 3) Build J(theta) = sum_t R_t * log pi_theta(a_t | s_t).
        # 4) Gradient ascent on J(theta).
        returns: List[float] = []
        horizon = len(self.episode_rewards)
        for t in range(horizon):
            return_t = 0.0
            for tp in range(t, horizon):
                return_t += (self.gamma ** (tp - t)) * self.episode_rewards[tp]
            returns.append(return_t)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        log_probs_t = torch.stack(self.episode_log_probs)

        # Variance reduction for REINFORCE:
        # A_t = R_t - mean(R), then normalize A_t.
        advantages_t = returns_t - returns_t.mean()
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        objective = (advantages_t.detach() * log_probs_t).mean()
        loss = -objective

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_log_probs.clear()
        self.episode_rewards.clear()

        return float(loss.item())
