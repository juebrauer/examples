import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_SAMPLES = 10000
TRAIN_BATCH_SIZE = 128

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class NavigationCNN(nn.Module):
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
            nn.Linear(128, 4),
        )

    def extract_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return torch.flatten(x, start_dim=1)

    def get_feature_vector_length(self, input_size: int) -> int:
        with torch.no_grad():
            model_device = next(self.parameters()).device
            dummy = torch.zeros(1, 3, input_size, input_size, device=model_device)
            out = self.extract_feature_vector(dummy)
        return int(out.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_feature_vector(x)
        return self.classifier(x)


class SupervisedNavigationAgent:
    def __init__(self, world_size: int, device: torch.device, lr: float = 1e-3):
        self.world_size = world_size
        self.device = device
        self.model = NavigationCNN(input_size=world_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_feature_vector_length(self) -> int:
        return self.model.get_feature_vector_length(self.world_size)

    def predict_scores_from_state_image(self, state_image: np.ndarray) -> torch.Tensor:
        state_tensor = torch.from_numpy(state_image).unsqueeze(0).to(self.device)
        return self.model(state_tensor)[0]

    def train(
        self,
        total_epochs: int,
        should_stop: Callable[[], bool],
        progress_callback: Callable[[dict], None],
        local_sampling_enabled: bool,
        soft_targets_enabled: bool,
        sample_training_positions_fn: Callable[[int, bool], tuple[int, int, int, int]],
        build_state_image_fn: Callable[[int, int, int, int, int], np.ndarray],
        expert_action_from_positions_fn: Callable[[int, int, int, int], int],
        expert_action_distribution_fn: Callable[[int, int, int, int], np.ndarray],
    ) -> int:
        self.model.train()

        states = np.zeros(
            (TRAIN_SAMPLES, 3, self.world_size, self.world_size),
            dtype=np.float32,
        )
        if soft_targets_enabled:
            targets = np.zeros((TRAIN_SAMPLES, 4), dtype=np.float32)
        else:
            targets = np.zeros((TRAIN_SAMPLES,), dtype=np.int64)

        for i in range(TRAIN_SAMPLES):
            if should_stop():
                return 0

            agent_x, agent_y, goal_x, goal_y = sample_training_positions_fn(
                self.world_size,
                local_sampling_enabled,
            )

            states[i] = build_state_image_fn(
                self.world_size,
                agent_x,
                agent_y,
                goal_x,
                goal_y,
            )
            if soft_targets_enabled:
                targets[i] = expert_action_distribution_fn(agent_x, agent_y, goal_x, goal_y)
            else:
                targets[i] = expert_action_from_positions_fn(agent_x, agent_y, goal_x, goal_y)

        states_tensor = torch.from_numpy(states)
        targets_tensor = torch.from_numpy(targets)

        epoch = 1
        while epoch <= total_epochs and not should_stop():
            perm = torch.randperm(TRAIN_SAMPLES)
            total_loss = 0.0
            total_seen = 0

            for start in range(0, TRAIN_SAMPLES, TRAIN_BATCH_SIZE):
                if should_stop():
                    break

                end = min(start + TRAIN_BATCH_SIZE, TRAIN_SAMPLES)
                idx = perm[start:end]

                batch_x = states_tensor[idx].to(self.device)
                batch_y = targets_tensor[idx].to(self.device)

                logits = self.model(batch_x)
                if soft_targets_enabled:
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = -(batch_y * log_probs).sum(dim=1).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(logits, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = int(batch_y.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_seen += batch_size

            epoch_loss = total_loss / max(total_seen, 1)
            progress_callback(
                {
                    "mode": "training_supervised",
                    "epoch": epoch,
                    "epochs_total": total_epochs,
                    "epoch_loss": epoch_loss,
                }
            )
            epoch += 1

        return min(epoch - 1, total_epochs)

    def select_action(self, state_image: np.ndarray, deterministic: bool) -> tuple[int, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            scores = self.predict_scores_from_state_image(state_image)
            if deterministic:
                action = int(torch.argmax(scores).item())
            else:
                probs = torch.softmax(scores, dim=0)
                action = int(torch.multinomial(probs, num_samples=1).item())
        return action, scores
