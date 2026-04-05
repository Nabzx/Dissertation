<<<<<<< HEAD
"""
2D Grid World Environment for Multi-Agent Resource Scarcity Simulation.

This module implements a PettingZoo ParallelEnv with a 15x15 grid where
two agents compete to collect non-renewable resources.
"""
=======
>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from typing import Dict, List, Tuple, Optional


class GridWorldEnv(ParallelEnv):
<<<<<<< HEAD
    """
    A 15x15 grid world environment for multi-agent resource collection.
    
    Agents can move in 4 directions and collect resources. The episode ends
    when all resources are collected or max_steps is reached.
    """
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
    
    metadata = {
        "name": "gridworld_v0",
        "render_mode": ["human", "rgb_array"],
    }
    
    def __init__(self, grid_size: int = 15, num_resources: int = 10, max_steps: int = 200, seed: Optional[int] = None):
<<<<<<< HEAD
        """
        Initialise the grid world environment.
        
        Args:
            grid_size: Size of the square grid (default: 15)
            num_resources: Number of resources to spawn (default: 10)
            max_steps: Maximum steps per episode (default: 200)
            seed: Random seed for reproducibility
        """
=======
       
>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.max_steps = max_steps
        self.seed = seed
        
<<<<<<< HEAD
        # Agent IDs
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()
        
        # Grid encoding: 0=empty, 1=resource, 2=agent_0, 3=agent_1
=======

        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()
        

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.grid = None
        self.agent_positions = {}
        self.resource_positions = []
        self.step_count = 0
        
<<<<<<< HEAD
        # Logging data
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.heatmaps = {
            "agent_0": np.zeros((grid_size, grid_size), dtype=np.int32),
            "agent_1": np.zeros((grid_size, grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }
        self.initial_resource_positions = []
        
<<<<<<< HEAD
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.action_spaces = {
            agent: Discrete(5) for agent in self.agents
        }
        
<<<<<<< HEAD
        # Observation space: 15x15 grid with values 0-3
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.observation_spaces = {
            agent: Box(
                low=0,
                high=3,
                shape=(grid_size, grid_size),
                dtype=np.int32
            ) for agent in self.agents
        }
<<<<<<< HEAD
        
        # Set random seed
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        if seed is not None:
            np.random.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
<<<<<<< HEAD
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Dict of agent observations
            infos: Dict of agent info
        """
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)
        
<<<<<<< HEAD
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Reset step count
        self.step_count = 0
        
        # Reset logging data
=======

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        

        self.step_count = 0
        

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.heatmaps = {
            "agent_0": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
            "agent_1": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }
        
<<<<<<< HEAD
        # Place agents at random positions
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.agent_positions = {}
        for i, agent in enumerate(self.agents):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
<<<<<<< HEAD
                # Ensure agents don't start on same cell (optional, but cleaner)
                if pos not in self.agent_positions.values():
                    self.agent_positions[agent] = pos
                    self.grid[pos[0], pos[1]] = 2 + i  # 2 for agent_0, 3 for agent_1
                    break
        
        # Place resources randomly
=======

                if pos not in self.agent_positions.values():
                    self.agent_positions[agent] = pos
                    self.grid[pos[0], pos[1]] = 2 + i  
                    break
        

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        self.resource_positions = []
        self.initial_resource_positions = []
        resource_count = 0
        
        while resource_count < self.num_resources:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
<<<<<<< HEAD
            # Don't place resources on agent starting positions
            if pos not in self.agent_positions.values():
                if self.grid[pos[0], pos[1]] == 0:  # Empty cell
                    self.grid[pos[0], pos[1]] = 1  # Resource
=======

            if pos not in self.agent_positions.values():
                if self.grid[pos[0], pos[1]] == 0:  
                    self.grid[pos[0], pos[1]] = 1  
>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
                    self.resource_positions.append(pos)
                    self.initial_resource_positions.append(pos)
                    resource_count += 1
        
<<<<<<< HEAD
        # Update heatmaps with initial positions
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
<<<<<<< HEAD
        """
        Execute one step in the environment.
        
        Args:
            actions: Dict mapping agent IDs to action values (0-4)
        
        Returns:
            observations: Dict of agent observations
            rewards: Dict of agent rewards
            terminations: Dict of termination flags
            truncations: Dict of truncation flags
            infos: Dict of agent info
        """
        # Clear agent positions from grid (but keep resources)
        for agent in self.agents:
            pos = self.agent_positions[agent]
            # Only clear if it's an agent marker, preserve resources
            if self.grid[pos[0], pos[1]] in [2, 3]:  # Agent marker
                # Check if there's a resource at this position (shouldn't happen, but be safe)
                if pos in self.resource_positions:
                    self.grid[pos[0], pos[1]] = 1  # Restore resource marker
                else:
                    self.grid[pos[0], pos[1]] = 0  # Clear to empty
        
        # Move agents and handle resource collection
=======
        
        for agent in self.agents:
            pos = self.agent_positions[agent]
          
            if self.grid[pos[0], pos[1]] in [2, 3]:  
                if pos in self.resource_positions:
                    self.grid[pos[0], pos[1]] = 1  
                else:
                    self.grid[pos[0], pos[1]] = 0  
        

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        rewards = {}
        for agent in self.agents:
            action = actions[agent]
            pos = self.agent_positions[agent]
            new_pos = self._move_agent(pos, action)
            self.agent_positions[agent] = new_pos
            
<<<<<<< HEAD
            # Check if agent collected a resource
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
            if new_pos in self.resource_positions:
                rewards[agent] = 1.0
                self.resources_collected[agent] += 1
                self.resource_positions.remove(new_pos)
<<<<<<< HEAD
                # Remove resource from grid
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
                if self.grid[new_pos[0], new_pos[1]] == 1:
                    self.grid[new_pos[0], new_pos[1]] = 0
            else:
                rewards[agent] = 0.0
        
<<<<<<< HEAD
        # Update grid with new agent positions
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            # Mark agent position (cell should be empty or already cleared resource)
            if self.grid[pos[0], pos[1]] == 0:
                self.grid[pos[0], pos[1]] = 2 + i
            elif self.grid[pos[0], pos[1]] == 1:
                # Resource still there (shouldn't happen after collection, but handle it)
                self.grid[pos[0], pos[1]] = 2 + i
        
        # Update heatmaps
=======

        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]

            if self.grid[pos[0], pos[1]] == 0:
                self.grid[pos[0], pos[1]] = 2 + i
            elif self.grid[pos[0], pos[1]] == 1:

                self.grid[pos[0], pos[1]] = 2 + i

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1
        
<<<<<<< HEAD
        # Increment step count
        self.step_count += 1
        
        # Check termination conditions
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        # Episode ends if all resources collected
        if len(self.resource_positions) == 0:
            terminations = {agent: True for agent in self.agents}
        
        # Episode truncates if max_steps reached
=======

        self.step_count += 1
        

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        

        if len(self.resource_positions) == 0:
            terminations = {agent: True for agent in self.agents}
        

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        if self.step_count >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
<<<<<<< HEAD
        """
        Calculate new position after action.
        
        Args:
            pos: Current (row, col) position
            action: Action value (0=stay, 1=up, 2=down, 3=left, 4=right)
        
        Returns:
            New (row, col) position
        """
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
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
<<<<<<< HEAD
        """
        Get observations for all agents.
        
        Returns:
            Dict mapping agent IDs to observation arrays
        """
=======

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        observations = {}
        for agent in self.agents:
            # Create a copy of the grid as observation
            obs = self.grid.copy()
            observations[agent] = obs
        return observations
    
    def get_heatmaps(self) -> Dict[str, np.ndarray]:
<<<<<<< HEAD
        """
        Get heatmap data for all agents.
        
        Returns:
            Dict mapping agent IDs to heatmap arrays
        """
        return self.heatmaps.copy()
    
    def get_resources_collected(self) -> Dict[str, int]:
        """
        Get resource collection counts for all agents.
        
        Returns:
            Dict mapping agent IDs to resource counts
        """
        return self.resources_collected.copy()
    
    def get_final_grid(self) -> np.ndarray:
        """
        Get the final grid state.
        
        Returns:
            Final grid array
        """
        return self.grid.copy()
    
    def get_initial_resource_positions(self) -> List[Tuple[int, int]]:
        """
        Get initial resource positions for this episode.
        
        Returns:
            List of (row, col) tuples
        """
=======

        return self.heatmaps.copy()
    
    def get_resources_collected(self) -> Dict[str, int]:
 
        return self.resources_collected.copy()
    
    def get_final_grid(self) -> np.ndarray:

        return self.grid.copy()
    
    def get_initial_resource_positions(self) -> List[Tuple[int, int]]:

>>>>>>> f5e8926c77e2e7d5530adc6c79cab9bb3b9f021b
        return self.initial_resource_positions.copy()

