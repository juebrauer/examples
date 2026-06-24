import sys
from PySide6.QtWidgets import QApplication
from pong_env import PongEnv
from agent import RandomAgent, DQNAgent, PPOAgent
from ui import MainWindow

def main():
    app = QApplication(sys.argv)

    # 1. Initialize environment
    env = PongEnv()
    action_space = env.get_action_space()
    
    # 2. Define available agents
    agent_factories = {
        "Random Agent": lambda: RandomAgent(action_space),
        "DQN Agent": lambda: DQNAgent(action_space),
        "PPO Agent": lambda: PPOAgent(action_space)
    }

    # 3. Initialize and show UI
    window = MainWindow(env, agent_factories)
    window.show()

    # 4. Start event loop
    exit_code = app.exec()
    
    # Clean up
    env.close()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
