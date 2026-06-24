from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

State = np.ndarray


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


@dataclass
class PPOStep:
    state: State
    action: int
    reward: float
    next_state: State
    done: bool
    old_log_prob: float
    old_value: float


class PPOAgent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = 4
        self.alpha = 3e-4
        self.gamma = 0.95
        self.gae_lambda = 0.95
        self.ppo_clip = 0.2
        self.ppo_epochs = 6
        self.rollout_steps = 256
        self.minibatch_size = 64
        self.value_loss_weight = 0.5
        self.entropy_coef = 0.01
        self.epsilon = 0.0

        self.net = ActorCriticNetwork(4, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.alpha)

        self.trajectory: List[PPOStep] = []
        self.env_steps = 0
        self.training_steps = 0

        self.last_entropy: float | None = None
        self.last_ratio_mean: float | None = None
        self.last_clip_fraction: float | None = None

        self._pending_log_prob: torch.Tensor | None = None
        self._pending_value: torch.Tensor | None = None

    def register_env_step(self) -> None:
        self.env_steps += 1

    def _encode_state_features(self, state: State) -> State:
        # Raw state is [dx_{t-1}, dy_{t-1}, dx_t, dy_t].
        # We feed [dx_t, dy_t, d_dx, d_dy] to make target motion explicit.
        prev_dx, prev_dy, curr_dx, curr_dy = [float(x) for x in state]
        d_dx = curr_dx - prev_dx
        d_dy = curr_dy - prev_dy
        return np.array([curr_dx, curr_dy, d_dx, d_dy], dtype=np.float32)

    def act(self, state: State, greedy: bool = False) -> int:
        encoded_state = self._encode_state_features(state)
        x = torch.as_tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        dist = Categorical(logits=logits)

        if greedy:
            action_tensor = torch.argmax(logits, dim=1).squeeze(0)
        else:
            action_tensor = dist.sample().squeeze(0)

        self._pending_log_prob = dist.log_prob(action_tensor)
        self._pending_value = value.squeeze(0).squeeze(0)
        return int(action_tensor.item())

    def store(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        if self._pending_log_prob is None or self._pending_value is None:
            return

        encoded_state = self._encode_state_features(state)
        encoded_next_state = self._encode_state_features(next_state)

        self.trajectory.append(
            PPOStep(
                state=encoded_state,
                action=int(action),
                reward=float(reward),
                next_state=encoded_next_state,
                done=bool(done),
                old_log_prob=float(self._pending_log_prob.detach().cpu().item()),
                old_value=float(self._pending_value.detach().cpu().item()),
            )
        )

        self._pending_log_prob = None
        self._pending_value = None

    def learn(self, episode_done: bool = False) -> float | None:
        if not self.trajectory:
            return None
        if len(self.trajectory) < self.rollout_steps:
            return None

        # PPO pseudocode (mirrors the lecture structure):
        # 1) Data collection D already stored as (s_t, a_t, r_t, s_{t+1}, log pi_old(a_t|s_t), V_phi(s_t)).
        # 2) Compute returns R_t and advantages A_t = R_t - V_phi(s_t).
        # 3) Multiple optimization epochs over mini-batches using clipped ratio objective.
        # 4) total_loss = actor_loss + c * critic_loss and update both theta and phi.

        bootstrap_value = 0.0
        last_step = self.trajectory[-1]
        if not last_step.done:
            next_state_t = torch.as_tensor(last_step.next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.net(next_state_t)
                bootstrap_value = float(next_value.squeeze(0).squeeze(0).item())

        rewards = np.array([step.reward for step in self.trajectory], dtype=np.float32)
        dones = np.array([step.done for step in self.trajectory], dtype=np.float32)
        values = np.array([step.old_value for step in self.trajectory], dtype=np.float32)
        values_next = np.concatenate([values[1:], np.array([bootstrap_value], dtype=np.float32)])

        deltas = rewards + self.gamma * (1.0 - dones) * values_next - values

        advantages = np.zeros_like(deltas, dtype=np.float32)
        gae = 0.0
        for t in range(len(deltas) - 1, -1, -1):
            gae = float(deltas[t]) + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae

        returns = advantages + values

        states = np.stack([step.state for step in self.trajectory])
        actions = np.array([step.action for step in self.trajectory], dtype=np.int64)
        old_log_probs = np.array([step.old_log_prob for step in self.trajectory], dtype=np.float32)
        old_values = values

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(old_values, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        dataset_size = states_t.shape[0]
        losses: List[float] = []
        entropy_values: List[float] = []
        ratio_means: List[float] = []
        clip_fractions: List[float] = []

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_returns = returns_t[idx]
                mb_advantages = advantages_t[idx]

                logits, values = self.net(mb_states)
                values = values.squeeze(1)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                clip_fraction = ((ratio < (1.0 - self.ppo_clip)) | (ratio > (1.0 + self.ppo_clip))).float().mean()

                normal_actor_term = ratio * mb_advantages
                clipped_actor_term = clipped_ratio * mb_advantages
                actor_objective = torch.min(normal_actor_term, clipped_actor_term)
                actor_loss = -actor_objective.mean()

                critic_loss = (values - mb_returns).pow(2).mean()
                total_loss = actor_loss + self.value_loss_weight * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.optimizer.step()

                losses.append(float(total_loss.item()))
                entropy_values.append(float(entropy.detach().cpu().item()))
                ratio_means.append(float(ratio.detach().mean().cpu().item()))
                clip_fractions.append(float(clip_fraction.detach().cpu().item()))

        self.training_steps += 1
        self.last_entropy = float(np.mean(entropy_values)) if entropy_values else None
        self.last_ratio_mean = float(np.mean(ratio_means)) if ratio_means else None
        self.last_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else None
        self.trajectory.clear()
        return float(np.mean(losses)) if losses else None
