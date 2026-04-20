"""Simple greedy agents for the grid world environment."""

import numpy as np


class HeuristicAgent:
    """Move toward the nearest visible resource using Manhattan distance."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        try:
            self.agent_value = 2 + int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            self.agent_value = 2

    def get_action(self, observation: np.ndarray) -> int:
        """
        Return an action: 0=stay, 1=up, 2=down, 3=left, 4=right.
        """
        agent_positions = np.argwhere(observation == self.agent_value)
        if len(agent_positions) == 0:
            return np.random.randint(1, 5)
        agent_row, agent_col = agent_positions[0]

        resource_positions = np.argwhere(observation == 1)
        if len(resource_positions) == 0:
            return np.random.randint(1, 5)

        distances = np.abs(resource_positions[:, 0] - agent_row) + np.abs(resource_positions[:, 1] - agent_col)
        nearest_resource = resource_positions[int(np.argmin(distances))]

        row_diff = nearest_resource[0] - agent_row
        col_diff = nearest_resource[1] - agent_col

        if row_diff > 0:
            return 2
        if row_diff < 0:
            return 1
        if col_diff > 0:
            return 4
        if col_diff < 0:
            return 3
        return 0

    def reset(self):
        """Reset agent state (no state to reset for heuristic agent)."""
        pass
