import time
import os
import matplotlib.pyplot as plt
from collections import deque
from PySide6.QtCore import QThread, Signal
from pong_env import PongEnv

class ExperimentWorker(QThread):
    # agent_name, current_step, max_steps, speed (steps/h)
    progress_updated = Signal(str, int, int, float) 
    # results dictionary
    finished_experiment = Signal(dict)
    
    def __init__(self, agent_factories, max_steps):
        super().__init__()
        self.agent_factories = agent_factories
        self.max_steps = max_steps
        
    def run(self):
        env = PongEnv()
        results = {}
        
        for agent_name in ["DQN Agent", "PPO Agent"]:
            agent = self.agent_factories[agent_name]()
            state, _ = env.reset()
            
            episodes_played = 0
            episodes_won = 0
            episodes_lost = 0
            total_reward = 0.0
            ep_reward = 0.0
            
            chunk_stats = {}
            num_chunks = (self.max_steps - 1) // 100000 + 1
            for i in range(num_chunks):
                chunk_stats[i] = {'won': 0, 'lost': 0}
                
            step_history = []
            raw_rewards = []
            avg_rewards = []
            reward_buffer = deque(maxlen=1000)
            
            last_time = time.time()
            
            for step in range(1, self.max_steps + 1):
                action = agent.get_action(state)
                # Use env.step which returns (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.step_end(reward, next_state, done)
                agent.train_step()
                
                ep_reward += reward
                total_reward += reward
                
                reward_buffer.append(reward)
                raw_rewards.append(reward)
                step_history.append(step)
                avg_rewards.append(sum(reward_buffer) / len(reward_buffer))
                
                if done:
                    episodes_played += 1
                    chunk_idx = (step - 1) // 100000
                    if ep_reward > 0:
                        episodes_won += 1
                        if chunk_idx in chunk_stats:
                            chunk_stats[chunk_idx]['won'] += 1
                    else:
                        episodes_lost += 1
                        if chunk_idx in chunk_stats:
                            chunk_stats[chunk_idx]['lost'] += 1
                            
                    ep_reward = 0.0
                    state, _ = env.reset()
                else:
                    state = next_state
                    
                if step % 1000 == 0:
                    current_time = time.time()
                    delta = current_time - last_time
                    if delta > 0:
                        speed = (1000 / delta) * 3600
                    else:
                        speed = 0.0
                    last_time = current_time
                    self.progress_updated.emit(agent_name, step, self.max_steps, speed)
                    
            results[agent_name] = {
                'episodes_played': episodes_played,
                'episodes_won': episodes_won,
                'episodes_lost': episodes_lost,
                'total_reward': total_reward,
                'chunk_stats': chunk_stats
            }
            
            # Save plot
            os.makedirs("models", exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(step_history, raw_rewards, color='black', alpha=0.3, label='Raw Reward')
            plt.plot(step_history, avg_rewards, color='lightgray', label='Avg Reward (1000 steps)')
            plt.title(f"Learning Curve: {agent_name}")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True, linestyle='--')
            safe_name = agent_name.replace(" ", "_")
            plt.savefig(f"models/{safe_name}_learning_curve.png")
            plt.close()
            
            # Clean up memory explicitly before next agent
            del agent
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self.finished_experiment.emit(results)
