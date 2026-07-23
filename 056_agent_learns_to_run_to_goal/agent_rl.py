"""RL agent trained as a supervised classifier on its own experience.

The environment supplies the supervision:

* positive reward -> target 1 for the selected action
* negative reward -> target 0 for the selected action

Only the output belonging to the action that was actually tried contributes to
the loss.  The other three actions have no known target for that state.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ACTION_COUNT = 4


class ActionProbabilityCNN(nn.Module):
    """Maps a three-channel state image to four independent probabilities."""

    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        in_features = self.get_feature_vector_length(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )

    def extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return torch.flatten(x, start_dim=1)

    def get_feature_vector_length(self, input_size: int) -> int:
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy = torch.zeros(1, 3, input_size, input_size, device=device)
            features = self.extract_feature_vector(dummy)
        return int(features.shape[1])

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_feature_vector(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Independent sigmoid outputs are intentional: an unsuccessful action
        # can be trained towards 0 without declaring another action correct.
        return torch.sigmoid(self.logits(x))


class RLInteractiveAgent:
    """Builds a replay dataset by trial and error and trains from random batches."""

    def __init__(
        self,
        world_size: int,
        device: torch.device,
        lr: float = 1e-3,
        warmup_samples: int = 1000,
        batch_size: int = 32,
        replay_capacity: int = 10_000,
        exploration_epsilon: float = 0.10,
    ):
        if warmup_samples < 1:
            raise ValueError("warmup_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if replay_capacity < warmup_samples:
            raise ValueError("replay_capacity must be at least warmup_samples")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be between 0 and 1")

        self.world_size = world_size
        self.device = device
        self.warmup_samples = warmup_samples
        self.batch_size = batch_size
        self.exploration_epsilon = exploration_epsilon

        self.model = ActionProbabilityCNN(input_size=world_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Images contain values in [0, 1].  Keeping uint8 copies makes a
        # 10,000-item replay memory about four times smaller than float32.
        self.replay_buffer: deque[tuple[np.ndarray, int, float]] = deque(
            maxlen=replay_capacity
        )
        self.examples_seen = 0
        self.training_steps = 0

    def begin_episode(self) -> None:
        """Episode hook; replay memory deliberately spans episodes."""

    def _validate_state_image(self, state_image: np.ndarray) -> None:
        expected_shape = (3, self.world_size, self.world_size)
        if state_image.shape != expected_shape:
            raise ValueError(
                f"state_image has shape {state_image.shape}, expected {expected_shape}"
            )

    def _state_to_tensor(self, state_image: np.ndarray) -> torch.Tensor:
        self._validate_state_image(state_image)
        return torch.as_tensor(
            state_image,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    @staticmethod
    def _encode_state(state_image: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(state_image, 0.0, 1.0) * 255.0).astype(
            np.uint8,
            copy=False,
        )

    def predict_action_probabilities(self, state_image: np.ndarray) -> np.ndarray:
        """Returns the network's four independent values in the interval [0, 1]."""

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
            # The sigmoid outputs are independent confidence values.  They are
            # normalized only for sampling an action; the network outputs
            # themselves remain independent probabilities.
            sampling_probabilities = probabilities.astype(np.float64)
            probability_sum = float(sampling_probabilities.sum())
            if probability_sum <= 1e-12:
                action = random.randrange(ACTION_COUNT)
            else:
                sampling_probabilities /= probability_sum
                action = int(np.random.choice(ACTION_COUNT, p=sampling_probabilities))

        return action, probabilities

    def _train_random_batch(self) -> tuple[float, float]:
        batch = random.sample(
            self.replay_buffer,
            k=min(self.batch_size, len(self.replay_buffer)),
        )
        states = np.stack([example[0] for example in batch]).astype(np.float32)
        states /= 255.0
        actions = torch.tensor(
            [example[1] for example in batch],
            dtype=torch.long,
            device=self.device,
        )
        targets = torch.tensor(
            [example[2] for example in batch],
            dtype=torch.float32,
            device=self.device,
        )
        states_tensor = torch.from_numpy(states).to(self.device)

        self.model.train()
        all_logits = self.model.logits(states_tensor)
        selected_logits = all_logits.gather(1, actions[:, None]).squeeze(1)
        loss = self.loss_fn(selected_logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        selected_probabilities = torch.sigmoid(selected_logits.detach())
        return float(loss.item()), float(selected_probabilities.mean().item())

    def observe_and_learn(self, state_image: np.ndarray, action: int, reward: float) -> dict:
        """Adds one labeled experience and, after warm-up, trains one mini-batch."""

        self._validate_state_image(state_image)
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")

        # Exactly zero supplies no positive/negative label and is therefore not
        # inserted.  The current environment produces non-zero step rewards.
        target = None
        if reward > 0.0:
            target = 1.0
        elif reward < 0.0:
            target = 0.0

        if target is not None:
            encoded_state = self._encode_state(state_image)
            self.replay_buffer.append((encoded_state.copy(), action, target))
            self.examples_seen += 1

        loss = 0.0
        mean_selected_probability = 0.0
        trained = False
        if len(self.replay_buffer) >= self.warmup_samples:
            loss, mean_selected_probability = self._train_random_batch()
            trained = True

        return {
            "loss": loss,
            "trained": trained,
            "target": target,
            "replay_size": len(self.replay_buffer),
            "examples_seen": self.examples_seen,
            "training_steps": self.training_steps,
            "mean_selected_probability": mean_selected_probability,
        }
