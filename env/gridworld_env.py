"""
2D Grid World Environment for Multi-Agent Resource Scarcity Simulation.

This module implements a PettingZoo ParallelEnv with a 15x15 grid where
two agents compete to collect non-renewable resources.
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from typing import Dict, List, Tuple, Optional

from env.arena import compute_octagon_mask, is_inside_arena


class GridWorldEnv(ParallelEnv):
    """
    A 15x15 grid world environment for multi-agent resource collection.

    Agents can move in 4 directions and collect resources. The episode ends
    when all resources are collected or max_steps is reached.
    """

    metadata = {
        "name": "gridworld_v0",
        "render_mode": ["human", "rgb_array"],
    }

    def __init__(
        self,
        grid_size: int = 15,
        num_resources: int = 10,
        num_obstacles: int = 12,
        resource_respawn_prob: float = 0.02,
        max_resources: Optional[int] = None,
        max_steps: int = 200,
        view_size: int = 5,
        partial_observability: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialise the grid world environment.

        Args:
            grid_size: Size of the square grid (default: 15)
            num_resources: Number of resources to spawn (default: 10)
            num_obstacles: Number of obstacle cells to place (default: 12)
            resource_respawn_prob: Per-step probability of spawning one resource
            max_resources: Maximum active resources allowed at once
            max_steps: Maximum steps per episode (default: 200)
            view_size: Side length of the local observation window (default: 5)
            partial_observability: If True, return a local view per agent; if False,
                return the full grid (original behaviour).
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.num_obstacles = num_obstacles
        self.resource_respawn_prob = resource_respawn_prob
        self.max_resources = max_resources if max_resources is not None else max(num_resources, 15)
        self.max_steps = max_steps
        self.view_size = view_size
        self.partial_observability = partial_observability
        self.seed = seed
        self.arena_mask = compute_octagon_mask(grid_size, grid_size)

        # Agent IDs
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()

        # Grid encoding: 0=empty, 1=resource, 2=agent_0, 3=agent_1, 4=obstacle
        self.grid = None
        self.agent_positions = {}
        self.resource_positions = []
        self.newly_spawned_resources = []
        self.obstacle_positions = []
        self.step_count = 0

        # Logging data
        self.heatmaps = {
            "agent_0": np.zeros((grid_size, grid_size), dtype=np.int32),
            "agent_1": np.zeros((grid_size, grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }
        self.initial_resource_positions = []

        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_spaces = {agent: Discrete(5) for agent in self.agents}

        # Observation space:
        # - full observability: (grid_size, grid_size)
        # - partial observability: (view_size, view_size)
        obs_shape = (view_size, view_size) if partial_observability else (grid_size, grid_size)
        self.observation_spaces = {
            agent: Box(low=0, high=4, shape=obs_shape, dtype=np.int32)
            for agent in self.agents
        }

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initial state.

        Returns:
            observations: Dict of agent observations
            infos: Dict of agent info
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)

        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Reset step count
        self.step_count = 0

        # Reset logging data
        self.heatmaps = {
            "agent_0": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
            "agent_1": np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
        }
        self.resources_collected = {
            "agent_0": 0,
            "agent_1": 0,
        }

        # Place obstacles first so agents/resources avoid blocked cells.
        self.obstacle_positions = []
        obstacle_count = 0
        while obstacle_count < self.num_obstacles:
            pos = self._sample_empty_arena_cell()
            self.grid[pos[0], pos[1]] = 4
            self.obstacle_positions.append(pos)
            obstacle_count += 1

        # Place agents at random positions
        self.agent_positions = {}
        for i, agent in enumerate(self.agents):
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
            self.agent_positions[agent] = pos
            self.grid[pos[0], pos[1]] = 2 + i  # 2 for agent_0, 3 for agent_1

        # Place resources randomly
        self.resource_positions = []
        self.initial_resource_positions = []
        self.newly_spawned_resources = []
        resource_count = 0

        while resource_count < self.num_resources:
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
            self.grid[pos[0], pos[1]] = 1  # Resource
            self.resource_positions.append(pos)
            self.initial_resource_positions.append(pos)
            resource_count += 1

        # Update heatmaps with initial positions
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):
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
        rewards = {}
        for agent in self.agents:
            action = actions[agent]
            pos = self.agent_positions[agent]
            new_pos = self._move_agent(pos, action)
            self.agent_positions[agent] = new_pos

            # Check if agent collected a resource
            if new_pos in self.resource_positions:
                rewards[agent] = 1.0
                self.resources_collected[agent] += 1
                self.resource_positions.remove(new_pos)
                # Remove resource from grid
                if self.grid[new_pos[0], new_pos[1]] == 1:
                    self.grid[new_pos[0], new_pos[1]] = 0
            else:
                rewards[agent] = 0.0

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
        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1

        # Increment step count
        self.step_count += 1

        self._maybe_respawn_resource()

        # Check termination conditions
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Episode ends if all resources collected
        if len(self.resource_positions) == 0:
            terminations = {agent: True for agent in self.agents}

        # Episode truncates if max_steps reached
        if self.step_count >= self.max_steps:
            truncations = {agent: True for agent in self.agents}

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _maybe_respawn_resource(self) -> None:
        """Spawn at most one new resource with a small per-step probability."""
        self.newly_spawned_resources = []

        if len(self.resource_positions) >= self.max_resources:
            return
        if np.random.random() >= self.resource_respawn_prob:
            return

        try:
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
        except RuntimeError:
            return

        self.grid[pos[0], pos[1]] = 1
        self.resource_positions.append(pos)
        self.newly_spawned_resources.append(pos)

    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Calculate new position after action.

        Args:
            pos: Current (row, col) position
            action: Action value (0=stay, 1=up, 2=down, 3=left, 4=right)

        Returns:
            New (row, col) position
        """
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

        new_pos = (row, col)
        if not self.is_inside_arena(new_pos[0], new_pos[1]):
            return pos
        if new_pos in self.obstacle_positions:
            return pos
        return new_pos

    def is_inside_arena(self, x: int, y: int) -> bool:
        """
        Return True if the grid cell is inside the physical octagon arena.

        Args:
            x: Row index
            y: Column index
        """
        return is_inside_arena(self.arena_mask, x, y)

    def _sample_empty_arena_cell(self, exclude: Optional[set] = None) -> Tuple[int, int]:
        """Sample an empty cell that is inside the octagon arena."""
        exclude = exclude or set()
        valid_cells = np.argwhere((self.grid == 0) & self.arena_mask)
        if len(valid_cells) == 0:
            raise RuntimeError("No empty cells available inside the octagon arena.")

        while True:
            row, col = valid_cells[np.random.randint(0, len(valid_cells))]
            pos = (int(row), int(col))
            if pos not in exclude:
                return pos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all agents.

        Returns:
            Dict mapping agent IDs to observation arrays
        """
        observations = {}
        for agent in self.agents:
            if not self.partial_observability:
                # Original behaviour: each agent sees the full grid.
                observations[agent] = self.grid.copy()
                continue

            # Partial observability: local window centered on the agent.
            # We pad with zeros near edges.
            size = int(self.view_size)
            radius = size // 2

            row, col = self.agent_positions[agent]
            padded = np.pad(self.grid, pad_width=radius, mode="constant", constant_values=0)

            # Convert to padded coordinates
            pr = row + radius
            pc = col + radius

            obs = padded[pr - radius : pr + radius + 1, pc - radius : pc + radius + 1]

            # In case an even view_size is accidentally passed, enforce the requested shape
            # by slicing to (size, size).
            observations[agent] = obs[:size, :size].copy()
        return observations

    def get_heatmaps(self) -> Dict[str, np.ndarray]:
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
        return self.initial_resource_positions.copy()
