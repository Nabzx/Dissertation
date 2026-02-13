
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from typing import Dict, List, Tuple, Optional


class GridWorldEnv(ParallelEnv):

    
    metadata = {
        "name": "gridworld_v0",
        "render_mode": ["human", "rgb_array"],
    }
    
    def __init__(self, grid_size: int = 15, num_resources: int = 10, max_steps: int = 200, seed: Optional[int] = None):
       
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.max_steps = max_steps
        self.seed = seed
        

        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()
        

        self.grid = None
        self.agent_positions = {}
        self.resource_positions = []
        self.step_count = 0
        

        self.heatmaps = {
            "agent_0": np.zeros((grid_size, grid_size), dtype=np.int32),
            "agent_1": np.zeros((grid_size, grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }
        self.initial_resource_positions = []
        

        self.action_spaces = {
            agent: Discrete(5) for agent in self.agents
        }
        

        self.observation_spaces = {
            agent: Box(
                low=0,
                high=3,
                shape=(grid_size, grid_size),
                dtype=np.int32
            ) for agent in self.agents
        }

        if seed is not None:
            np.random.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)
        

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        

        self.step_count = 0
        

        self.heatmaps = {
            "agent_0": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
            "agent_1": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }
        

        self.agent_positions = {}
        for i, agent in enumerate(self.agents):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )

                if pos not in self.agent_positions.values():
                    self.agent_positions[agent] = pos
                    self.grid[pos[0], pos[1]] = 2 + i  
                    break
        

        self.resource_positions = []
        self.initial_resource_positions = []
        resource_count = 0
        
        while resource_count < self.num_resources:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )

            if pos not in self.agent_positions.values():
                if self.grid[pos[0], pos[1]] == 0:  
                    self.grid[pos[0], pos[1]] = 1  
                    self.resource_positions.append(pos)
                    self.initial_resource_positions.append(pos)
                    resource_count += 1
        

        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        
        for agent in self.agents:
            pos = self.agent_positions[agent]
          
            if self.grid[pos[0], pos[1]] in [2, 3]:  
                if pos in self.resource_positions:
                    self.grid[pos[0], pos[1]] = 1  
                else:
                    self.grid[pos[0], pos[1]] = 0  
        

        rewards = {}
        for agent in self.agents:
            action = actions[agent]
            pos = self.agent_positions[agent]
            new_pos = self._move_agent(pos, action)
            self.agent_positions[agent] = new_pos
            

            if new_pos in self.resource_positions:
                rewards[agent] = 1.0
                self.resources_collected[agent] += 1
                self.resource_positions.remove(new_pos)

                if self.grid[new_pos[0], new_pos[1]] == 1:
                    self.grid[new_pos[0], new_pos[1]] = 0
            else:
                rewards[agent] = 0.0
        

        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]

            if self.grid[pos[0], pos[1]] == 0:
                self.grid[pos[0], pos[1]] = 2 + i
            elif self.grid[pos[0], pos[1]] == 1:

                self.grid[pos[0], pos[1]] = 2 + i

        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1
        

        self.step_count += 1
        

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        

        if len(self.resource_positions) == 0:
            terminations = {agent: True for agent in self.agents}
        

        if self.step_count >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:

        row, col = pos
        
        if action == 0:  # Stay
            return pos
        elif action == 1:  # Up
            row = max(0, row - 1)
        elif action == 2:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
        elif action == 4:  # Right
            col = min(self.grid_size - 1, col + 1)
        
        return (row, col)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:

        observations = {}
        for agent in self.agents:
            # Create a copy of the grid as observation
            obs = self.grid.copy()
            observations[agent] = obs
        return observations
    
    def get_heatmaps(self) -> Dict[str, np.ndarray]:

        return self.heatmaps.copy()
    
    def get_resources_collected(self) -> Dict[str, int]:
 
        return self.resources_collected.copy()
    
    def get_final_grid(self) -> np.ndarray:

        return self.grid.copy()
    
    def get_initial_resource_positions(self) -> List[Tuple[int, int]]:

        return self.initial_resource_positions.copy()

