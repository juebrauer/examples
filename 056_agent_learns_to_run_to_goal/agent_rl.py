from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        in_features = self.get_feature_vector_length(input_size)
        self.head = nn.Sequential(
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
        feats = self.extract_feature_vector(x)
        return self.head(feats)


class StateValueCriticCNN(nn.Module):
    """Predicts the reward expected *before* an action is selected.

    The critic deliberately does not receive the selected action.  Therefore
    ``reward - value(state)`` tells the actor whether the sampled action was
    better or worse than what is normally expected in this state.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        in_features = self.get_feature_vector_length(input_size)
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feature_vector(states)
        return self.head(feats)


class RLInteractiveAgent:
    def __init__(
        self,
        world_size: int,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        history_decay: float = 0.90,
        entropy_coefficient: float = 0.01,
        advantage_clip: float = 2.0,
    ):
        self.world_size = world_size
        self.device = device
        self.actor = ActorCNN(input_size=world_size).to(device)
        self.critic = StateValueCriticCNN(input_size=world_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # SmoothL1 is less sensitive than MSE to the relatively large terminal
        # reward while the critic is still inaccurate.
        self.critic_loss_fn = nn.SmoothL1Loss()

        if not 0.0 <= history_decay <= 1.0:
            raise ValueError("history_decay must be between 0 and 1")
        if entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative")
        if advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive")

        self.history_decay = history_decay
        self.entropy_coefficient = entropy_coefficient
        self.advantage_clip = advantage_clip
        self.history = deque()

    def begin_episode(self, history_length: int) -> None:
        self.history = deque(maxlen=max(1, history_length))

    def _state_to_tensor(self, state_image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state_image).unsqueeze(0).to(self.device)

    def act(self, state_image: np.ndarray, deterministic: bool) -> tuple[int, np.ndarray]:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state_image)
            logits = self.actor(state_tensor)[0]
            probs = torch.softmax(logits, dim=0)
            if deterministic:
                action = int(torch.argmax(probs).item())
            else:
                action = int(torch.multinomial(probs, num_samples=1).item())

        return action, probs.detach().cpu().numpy()

    def observe_and_learn(self, state_image: np.ndarray, action: int, reward: float) -> dict:
        self.actor.train()
        self.critic.train()

        state_tensor = self._state_to_tensor(state_image)
        predicted_reward_before = float(self.critic(state_tensor).detach().item())

        reward_target = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

        predicted_reward = self.critic(state_tensor)
        critic_loss = self.critic_loss_fn(predicted_reward, reward_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        raw_advantage = reward - predicted_reward_before
        advantage = max(
            -self.advantage_clip,
            min(self.advantage_clip, raw_advantage),
        )

        self.history.append((state_image.copy(), action))

        # Newest item has age 0 and weight 1.  Older decisions receive less
        # credit/blame for the reward observed now.  Evaluate the whole history
        # as one batch so large N values remain reasonably fast.
        newest_first = list(reversed(self.history))
        history_states = torch.from_numpy(
            np.stack([item[0] for item in newest_first])
        ).to(self.device)
        history_actions = torch.tensor(
            [item[1] for item in newest_first],
            dtype=torch.long,
            device=self.device,
        )
        weights = torch.tensor(
            [self.history_decay ** age for age in range(len(newest_first))],
            dtype=torch.float32,
            device=self.device,
        )

        logits = self.actor(history_states)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        chosen_log_probs = log_probs.gather(1, history_actions[:, None]).squeeze(1)
        entropies = -(probs * log_probs).sum(dim=1)

        actor_loss_value = 0.0
        policy_loss_value = 0.0
        entropy_value = 0.0
        if newest_first:
            weight_sum = torch.clamp(weights.sum(), min=1e-12)
            # Positive advantage increases log pi(action|state); negative
            # advantage decreases it.  Softmax keeps all four probabilities
            # normalized automatically.
            policy_loss = -(weights * advantage * chosen_log_probs).sum() / weight_sum
            mean_entropy = (weights * entropies).sum() / weight_sum
            actor_loss = policy_loss - self.entropy_coefficient * mean_entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())
            policy_loss_value = float(policy_loss.item())
            entropy_value = float(mean_entropy.item())

        return {
            "predicted_reward": predicted_reward_before,
            "reward": reward,
            "advantage": advantage,
            "raw_advantage": raw_advantage,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "policy_loss": policy_loss_value,
            "entropy": entropy_value,
        }