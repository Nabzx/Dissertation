"""
Heuristic agents for the grid world environment.

These agents use simple greedy strategies to collect resources:
- Scan for nearest resource
- Move greedily toward it (Manhattan distance)
- If no resources remain, move randomly
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class HeuristicAgent:
    """
    A simple heuristic agent that greedily moves toward the nearest resource.

    The agent uses Manhattan distance to find the closest resource and
    moves one step toward it each turn.
    """

    def __init__(self, agent_id: str):
        """
        Initialise the heuristic agent.

        Args:
            agent_id: Unique identifier for this agent (e.g., "agent_0")
        """
        self.agent_id = agent_id
        try:
            self.agent_value = 2 + int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            self.agent_value = 2

    def get_action(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.

        Args:
            observation: grid observation (0=empty, 1=resource, 2+=agents, final high value=obstacle)

        Returns:
            Action value (0=stay, 1=up, 2=down, 3=left, 4=right)
        """
        # Find current agent position
        agent_pos = None
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if observation[i, j] == self.agent_value:
                    agent_pos = (i, j)
                    break
            if agent_pos is not None:
                break

        if agent_pos is None:
            # Agent position not found, return random action
            return np.random.randint(1, 5)

        # Find all resource positions
        resource_positions = []
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if observation[i, j] == 1:  # Resource
                    resource_positions.append((i, j))

        # If no resources, move randomly
        if len(resource_positions) == 0:
            return np.random.randint(1, 5)

        # Find nearest resource using Manhattan distance
        nearest_resource = None
        min_distance = float("inf")

        for resource_pos in resource_positions:
            distance = abs(agent_pos[0] - resource_pos[0]) + abs(agent_pos[1] - resource_pos[1])
            if distance < min_distance:
                min_distance = distance
                nearest_resource = resource_pos

        # Move greedily toward nearest resource
        row_diff = nearest_resource[0] - agent_pos[0]
        col_diff = nearest_resource[1] - agent_pos[1]

        # Prioritise row movement if there's a difference
        if row_diff > 0:
            return 2  # Down
        elif row_diff < 0:
            return 1  # Up
        elif col_diff > 0:
            return 4  # Right
        elif col_diff < 0:
            return 3  # Left
        else:
            # Already at resource (shouldn't happen, but handle it)
            return 0  # Stay

    def reset(self):
        """Reset agent state (no state to reset for heuristic agent)."""
        pass
