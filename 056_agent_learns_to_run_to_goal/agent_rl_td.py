"""Sparse-reward navigation agent trained with temporal-difference targets."""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent_rl import ACTION_COUNT, ActionProbabilityCNN


class TDTargetNavigationAgent:
    """Predicts the probability of eventually reaching the visible goal."""

    def __init__(
        self,
        world_size: int,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        warmup_samples: int = 1000,
        batch_size: int = 32,
        replay_capacity: int = 10_000,
        exploration_epsilon: float = 0.10,
        target_update_interval: int = 250,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if warmup_samples < 1:
            raise ValueError("warmup_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if replay_capacity < warmup_samples:
            raise ValueError("replay_capacity must be at least warmup_samples")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be between 0 and 1")
        if target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")

        self.world_size = world_size
        self.device = device
        self.gamma = gamma
        self.warmup_samples = warmup_samples
        self.batch_size = batch_size
        self.exploration_epsilon = exploration_epsilon
        self.target_update_interval = target_update_interval

        self.model = ActionProbabilityCNN(input_size=world_size).to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.replay_buffer: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque(maxlen=replay_capacity)
        self.examples_seen = 0
        self.training_steps = 0
        self.target_updates = 0

    def begin_episode(self) -> None:
        """Episode hook; replay memory deliberately spans episodes."""

    def _validate_state_image(self, state_image: np.ndarray) -> None:
        expected_shape = (3, self.world_size, self.world_size)
        if state_image.shape != expected_shape:
            raise ValueError(
                f"state_image has shape {state_image.shape}, expected {expected_shape}"
            )

    @staticmethod
    def _encode_state(state_image: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(state_image, 0.0, 1.0) * 255.0).astype(
            np.uint8,
            copy=False,
        )

    def _state_to_tensor(self, state_image: np.ndarray) -> torch.Tensor:
        self._validate_state_image(state_image)
        return torch.as_tensor(
            state_image,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def predict_action_probabilities(self, state_image: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(self._state_to_tensor(state_image))[0]
        return probabilities.cpu().numpy()

    def act(self, state_image: np.ndarray, deterministic: bool) -> tuple[int, np.ndarray]:
        probabilities = self.predict_action_probabilities(state_image)

        if deterministic:
            action = int(np.argmax(probabilities))
        elif random.random() < self.exploration_epsilon:
            action = random.randrange(ACTION_COUNT)
        else:
            sampling_probabilities = probabilities.astype(np.float64)
            probability_sum = float(sampling_probabilities.sum())
            if probability_sum <= 1e-12:
                action = random.randrange(ACTION_COUNT)
            else:
                sampling_probabilities /= probability_sum
                action = int(np.random.choice(ACTION_COUNT, p=sampling_probabilities))

        return action, probabilities

    def _update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_updates += 1

    def _train_random_batch(self) -> tuple[float, float]:
        batch = random.sample(
            self.replay_buffer,
            k=min(self.batch_size, len(self.replay_buffer)),
        )

        states = np.stack([transition[0] for transition in batch]).astype(np.float32)
        next_states = np.stack(
            [transition[3] for transition in batch]
        ).astype(np.float32)
        states /= 255.0
        next_states /= 255.0

        actions = torch.tensor(
            [transition[1] for transition in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [transition[2] for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [transition[4] for transition in batch],
            dtype=torch.bool,
            device=self.device,
        )
        states_tensor = torch.from_numpy(states).to(self.device)
        next_states_tensor = torch.from_numpy(next_states).to(self.device)

        with torch.no_grad():
            next_probabilities = self.target_model(next_states_tensor)
            best_next_probabilities = next_probabilities.max(dim=1).values
            td_targets = torch.where(
                dones,
                rewards,
                rewards + self.gamma * best_next_probabilities,
            )
            td_targets = torch.clamp(td_targets, 0.0, 1.0)

        self.model.train()
        logits = self.model.logits(states_tensor)
        selected_logits = logits.gather(1, actions[:, None]).squeeze(1)
        loss = self.loss_fn(selected_logits, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1

        if self.training_steps % self.target_update_interval == 0:
            self._update_target_model()

        return float(loss.item()), float(td_targets.mean().item())

    def observe_and_learn(
        self,
        state_image: np.ndarray,
        action: int,
        reward: float,
        next_state_image: np.ndarray,
        done: bool,
    ) -> dict:
        """Stores one transition and trains one random replay mini-batch."""

        self._validate_state_image(state_image)
        self._validate_state_image(next_state_image)
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("sparse reward must be between 0 and 1")

        encoded_state = self._encode_state(state_image)
        encoded_next_state = self._encode_state(next_state_image)
        self.replay_buffer.append(
            (
                encoded_state.copy(),
                action,
                float(reward),
                encoded_next_state.copy(),
                bool(done),
            )
        )
        self.examples_seen += 1

        loss = 0.0
        mean_td_target = 0.0
        trained = False
        if len(self.replay_buffer) >= self.warmup_samples:
            loss, mean_td_target = self._train_random_batch()
            trained = True

        return {
            "loss": loss,
            "trained": trained,
            "replay_size": len(self.replay_buffer),
            "examples_seen": self.examples_seen,
            "training_steps": self.training_steps,
            "target_updates": self.target_updates,
            "mean_td_target": mean_td_target,
        }
