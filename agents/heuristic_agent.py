import numpy as np


class HeuristicAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        try:
            self.agent_value = 2 + int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            self.agent_value = 2

    def get_action(self, obs: np.ndarray) -> int:
        agent_spots = np.argwhere(obs == self.agent_value)
        if len(agent_spots) == 0:
            return np.random.randint(1, 5)
        agent_row, agent_col = agent_spots[0]

        food_spots = np.argwhere(obs == 1)
        if len(food_spots) == 0:
            return np.random.randint(1, 5)

        dists = np.abs(food_spots[:, 0] - agent_row) + np.abs(food_spots[:, 1] - agent_col)
        nearest_food = food_spots[int(np.argmin(dists))]

        row_diff = nearest_food[0] - agent_row
        col_diff = nearest_food[1] - agent_col

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
        pass
