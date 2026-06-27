import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions.categorical import Categorical

class BaseAgent:
    def __init__(self, action_space, name):
        self.action_space = action_space
        self.name = name
        self.steps_done = 0

    def get_action(self, state, greedy=False):
        raise NotImplementedError("Subclasses must implement get_action")

    def step_end(self, reward, next_state, done):
        self.steps_done += 1
        self.check_auto_save()

    def train_step(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def check_auto_save(self):
        if self.steps_done > 0 and self.steps_done % 100000 == 0:
            os.makedirs("models", exist_ok=True)
            filepath = f"models/{self.name}_{self.steps_done}.pth"
            self.save(filepath)
            print(f"[{self.name}] Automatically saved model at {self.steps_done} steps to {filepath}")


class RandomAgent(BaseAgent):
    def __init__(self, action_space, name="Random"):
        super().__init__(action_space, name)

    def get_action(self, state, greedy=False):
        return self.action_space.sample()


class DQNNetwork(nn.Module):
    """Standard Nature-DQN CNN Architecture for Atari (84x84)."""
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        conv_out = self.conv_net(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        return self.fc_net(flattened)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.array(state, copy=False), action, reward, np.array(next_state, copy=False), done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    def __init__(self, action_space, name="DQN", lr=1e-4, gamma=0.99, batch_size=32, buffer_capacity=10000, 
                 epsilon_start=1.0, epsilon_min=0.02, epsilon_decay_steps=100000, target_update_freq=1000):
        super().__init__(action_space, name)
        self.num_actions = action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent is using device: {self.device}")

        self.policy_net = DQNNetwork(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = DQNNetwork(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.target_update_freq = target_update_freq
        
        self.last_state = None
        self.last_action = None

    def get_action(self, state, greedy=False):
        state_np = np.array(state, copy=False)
        self.last_state = state_np
        
        epsilon_to_use = 0.0 if greedy else self.epsilon
        if random.random() < epsilon_to_use:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_t)
                action = q_values.argmax(dim=1).item()
        
        self.last_action = action
        return action
        
    def step_end(self, reward, next_state, done):
        if self.last_state is not None and self.last_action is not None:
            self.memory.push(self.last_state, self.last_action, reward, next_state, done)
        super().step_end(reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q_values = rewards_t + (self.gamma * max_next_q_values * (1 - dones_t))
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_rate

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class PPONetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        conv_out = self.conv_net(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        shared = self.fc_shared(flattened)
        return self.actor(shared), self.critic(shared)


class PPOBuffer:
    def __init__(self, capacity=2048):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.capacity = capacity
        
    def push(self, state, action, logprob, reward, value, done):
        self.states.append(np.array(state, copy=False))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def is_full(self):
        return len(self.states) >= self.capacity


class PPOAgent(BaseAgent):
    def __init__(self, action_space, name="PPO", lr=2.5e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.1, entropy_coef=0.01, vloss_coef=0.5, rollout_steps=2048, batch_size=64, epochs=4):
        super().__init__(action_space, name)
        self.num_actions = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPOAgent is using device: {self.device}")

        self.network = PPONetwork(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.vloss_coef = vloss_coef
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs = epochs

        self.memory = PPOBuffer(rollout_steps)

        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        self.last_value = None
        
    def get_action(self, state, greedy=False):
        state_np = np.array(state, copy=False)
        self.last_state = state_np
        
        with torch.no_grad():
            state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.network(state_t)
            if greedy:
                action = logits.argmax(dim=1)
                self.last_logprob = 0.0
            else:
                probs = Categorical(logits=logits)
                action = probs.sample()
                self.last_logprob = probs.log_prob(action).item()
            
        self.last_action = action.item()
        self.last_value = value.item()
        
        return self.last_action

    def step_end(self, reward, next_state, done):
        if self.last_state is not None and self.last_action is not None:
            self.memory.push(self.last_state, self.last_action, self.last_logprob, reward, self.last_value, done)
        super().step_end(reward, next_state, done)
        
    def train_step(self):
        if not self.memory.is_full():
            return
            
        states = torch.tensor(np.array(self.memory.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.memory.logprobs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.memory.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = 0.0
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_inds = np.arange(self.rollout_steps)
        for epoch in range(self.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.rollout_steps, self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]

                logits, new_values = self.network(states[mb_inds])
                probs = Categorical(logits=logits)
                new_logprobs = probs.log_prob(actions[mb_inds])
                entropy = probs.entropy().mean()

                logratio = new_logprobs - old_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_values = new_values.view(-1)
                v_loss = 0.5 * ((new_values - returns[mb_inds]) ** 2).mean()

                loss = pg_loss - self.entropy_coef * entropy + self.vloss_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        self.memory.clear()

    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath):
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
