import unittest

import numpy as np
import torch

from agent_rl import AgentRL


class AgentRLTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        np.random.seed(3)
        self.image = np.zeros((3, 8, 8), dtype=np.float32)
        self.image[0, 1, 6] = 1.0
        self.image[2, 5, 2] = 1.0

    def make_agent(self) -> AgentRL:
        return AgentRL(
            image_size=8,
            history_length=2,
            exploration=0.0,
            learning_rate=1e-2,
            device=torch.device("cpu"),
        )

    def test_positive_surprise_makes_chosen_action_more_likely(self) -> None:
        agent = self.make_agent()
        action = 3
        probability_before = agent.action_probabilities(self.image)[action]
        agent.observe(self.image, action, reward=10.0)
        probability_after = agent.action_probabilities(self.image)[action]
        self.assertGreater(probability_after, probability_before)

    def test_negative_surprise_makes_chosen_action_less_likely(self) -> None:
        agent = self.make_agent()
        action = 3
        probability_before = agent.action_probabilities(self.image)[action]
        agent.observe(self.image, action, reward=-10.0)
        probability_after = agent.action_probabilities(self.image)[action]
        self.assertLess(probability_after, probability_before)

    def test_history_keeps_only_last_n_decisions(self) -> None:
        agent = self.make_agent()
        reports = [agent.observe(self.image, i % 4, reward=0.0) for i in range(4)]
        self.assertEqual([report.history_items for report in reports], [1, 2, 2, 2])
        agent.begin_episode()
        report = agent.observe(self.image, 0, reward=0.0)
        self.assertEqual(report.history_items, 1)


if __name__ == "__main__":
    unittest.main()
