import numpy as np


class HeuristicAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id  # store agent id
        self.just_communicated = False  # flag (not really used here but kept for consistency)

        try:
            self.agent_value = 2 + int(agent_id.split("_")[-1])  
            # map agent id to grid value (e.g. agent_0 -> 2, agent_1 -> 3)
        except (ValueError, IndexError):
            self.agent_value = 2  
            # fallback if id format is unexpected

    def get_action(self, obs: np.ndarray) -> int:
        agent_spots = np.argwhere(obs == self.agent_value)  
        # find where this agent is on the grid

        if len(agent_spots) == 0:
            return np.random.randint(1, 5)  
            # if agent not found just move randomly

        agent_row, agent_col = agent_spots[0]  
        # take first match (should only be one)

        food_spots = np.argwhere(obs == 1)  
        # find all food locations (assumes food = 1)

        if len(food_spots) == 0:
            return np.random.randint(1, 5)  
            # no food left so just wander randomly

        dists = np.abs(food_spots[:, 0] - agent_row) + np.abs(food_spots[:, 1] - agent_col)  
        # manhattan distance to each food

        nearest_food = food_spots[int(np.argmin(dists))]  
        # pick closest food

        row_diff = nearest_food[0] - agent_row
        col_diff = nearest_food[1] - agent_col  
        # direction from agent to food

        if row_diff > 0:
            return 2  # move down
        if row_diff < 0:
            return 1  # move up
        if col_diff > 0:
            return 4  # move right
        if col_diff < 0:
            return 3  # move left

        return 0  # already on food (no move)

    def reset(self):
        pass  # nothing to reset for this simple agent