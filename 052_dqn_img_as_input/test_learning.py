import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from dqn_image_demo import SimpleGridEnv, DQNAgent

def test():
    env = SimpleGridEnv()
    agent = DQNAgent(
        obs_shape=(48, 48, 3),
        num_actions=4,
        lr=5e-4, # lower LR
        gamma=0.99, # higher gamma
        epsilon_decay_steps=100000, # slower decay
        target_update_frequency=1000
    )
    
    # Increase FC size just in case, but let's test with existing agent first to see if just HParams fix it.
    
    episodes = 1000
    rewards = []
    
    state = env.reset()
    for ep in range(episodes):
        ep_reward = 0
        done = False
        while not done:
            action, _, _ = agent.select_action(state, evaluation=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            ep_reward += reward
        
        state = env.reset()
        rewards.append(ep_reward)
        if ep % 50 == 0:
            avg_rew = np.mean(rewards[-50:]) if len(rewards) > 0 else 0
            print(f"Episode {ep}, Avg Reward: {avg_rew:.2f}, Epsilon: {agent.epsilon:.3f}")

test()
