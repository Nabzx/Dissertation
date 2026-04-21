from typing import Dict, List, Optional, Tuple

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

from env.arena import compute_octagon_mask, is_inside_arena


class GridWorldEnv(ParallelEnv):
    metadata = {
        "name": "gridworld_v0",
        "render_mode": ["human", "rgb_array"],
    }

    def __init__(
        self,
        grid_size: int = 25,
        num_resources: int = 25,
        num_obstacles: int = 45,
        resource_respawn_prob: float = 0.02,
        max_resources: Optional[int] = None,
        max_steps: int = 250,
        view_size: int = 5,
        partial_observability: bool = True,
        num_agents: int = 4,
        seed: Optional[int] = None,
    ):
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.num_resources = num_resources
        self.num_obstacles = num_obstacles
        self.resource_respawn_prob = resource_respawn_prob
        self.max_resources = max_resources if max_resources is not None else max(num_resources, 15)
        self.max_steps = max_steps
        self.view_size = view_size
        self.partial_observability = partial_observability
        self.seed = seed
        self.arena_mask = compute_octagon_mask(grid_size, grid_size)

        self.resource_value = 1
        self.agent_value_start = 2
        self.obstacle_value = self.agent_value_start + self.n_agents

        self.agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.possible_agents = self.agents.copy()

        self.grid = None
        self.agent_positions = {}
        self.just_communicated = {agent: False for agent in self.agents}
        self.resource_positions = []
        self.newly_spawned_resources = []
        self.obstacle_positions = []
        self.step_count = 0

        self.heatmaps = {
            agent: np.zeros((grid_size, grid_size), dtype=np.int32)
            for agent in self.agents
        }
        self.resources_collected = {agent: 0 for agent in self.agents}
        self.initial_resource_positions = []

        self.action_spaces = {agent: Discrete(5) for agent in self.agents}

        obs_size = (view_size, view_size) if partial_observability else (grid_size, grid_size)
        self.observation_spaces = {
            agent: Box(low=0, high=self.obstacle_value, shape=obs_size, dtype=np.int32)
            for agent in self.agents
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
            agent: np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            for agent in self.agents
        }
        self.resources_collected = {agent: 0 for agent in self.agents}

        self.obstacle_positions = []
        obstacle_count = 0
        while obstacle_count < self.num_obstacles:
            pos = self._sample_empty_arena_cell()
            self.grid[pos[0], pos[1]] = self.obstacle_value
            self.obstacle_positions.append(pos)
            obstacle_count += 1

        self.agent_positions = {}
        self.just_communicated = {agent: False for agent in self.agents}
        for agent in self.agents:
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
            self.agent_positions[agent] = pos
            self.grid[pos[0], pos[1]] = self.agent_value(agent)

        self.resource_positions = []
        self.initial_resource_positions = []
        self.newly_spawned_resources = []
        resource_count = 0

        while resource_count < self.num_resources:
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
            self.grid[pos[0], pos[1]] = self.resource_value
            self.resource_positions.append(pos)
            self.initial_resource_positions.append(pos)
            resource_count += 1

        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1

        return self._get_obs(), {agent: {} for agent in self.agents}

    def step(self, actions: Dict[str, int]):
        self.reset_communication_flags()

        for agent in self.agents:
            pos = self.agent_positions[agent]
            if self.is_agent_value(self.grid[pos[0], pos[1]]):
                if pos in self.resource_positions:
                    self.grid[pos[0], pos[1]] = self.resource_value
                else:
                    self.grid[pos[0], pos[1]] = 0

        rewards = {}
        occupied_positions = set(self.agent_positions.values())
        for agent in self.agents:
            action = actions[agent]
            pos = self.agent_positions[agent]
            occupied_positions.discard(pos)
            new_pos = self._move_agent(pos, action)
            if new_pos in occupied_positions:
                new_pos = pos
            self.agent_positions[agent] = new_pos
            occupied_positions.add(new_pos)

            if new_pos in self.resource_positions:
                rewards[agent] = 1.0
                self.resources_collected[agent] += 1
                self.just_communicated[agent] = True
                self.resource_positions.remove(new_pos)
                if self.grid[new_pos[0], new_pos[1]] == self.resource_value:
                    self.grid[new_pos[0], new_pos[1]] = 0
            else:
                rewards[agent] = 0.0

        self.mark_proximity_communication()

        for agent in self.agents:
            pos = self.agent_positions[agent]
            if self.grid[pos[0], pos[1]] == 0:
                self.grid[pos[0], pos[1]] = self.agent_value(agent)
            elif self.grid[pos[0], pos[1]] == self.resource_value:
                self.grid[pos[0], pos[1]] = self.agent_value(agent)

        for agent in self.agents:
            pos = self.agent_positions[agent]
            self.heatmaps[agent][pos[0], pos[1]] += 1

        self.step_count += 1
        self._maybe_add_resource()

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        if len(self.resource_positions) == 0:
            terminations = {agent: True for agent in self.agents}

        if self.step_count >= self.max_steps:
            truncations = {agent: True for agent in self.agents}

        return self._get_obs(), rewards, terminations, truncations, {agent: {} for agent in self.agents}

    def _maybe_add_resource(self) -> None:
        self.newly_spawned_resources = []

        if len(self.resource_positions) >= self.max_resources:
            return
        if np.random.random() >= self.resource_respawn_prob:
            return

        try:
            pos = self._sample_empty_arena_cell(exclude=set(self.agent_positions.values()))
        except RuntimeError:
            return

        self.grid[pos[0], pos[1]] = self.resource_value
        self.resource_positions.append(pos)
        self.newly_spawned_resources.append(pos)

    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        row, col = pos

        if action == 0:
            return pos
        elif action == 1:
            row = max(0, row - 1)
        elif action == 2:
            row = min(self.grid_size - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)
        elif action == 4:
            col = min(self.grid_size - 1, col + 1)

        new_pos = (row, col)
        if not self.is_inside_arena(new_pos[0], new_pos[1]):
            return pos
        if new_pos in self.obstacle_positions:
            return pos
        return new_pos

    def agent_value(self, agent_id: str) -> int:
        return self.agent_value_start + self.agents.index(agent_id)

    def agent_id_from_value(self, value: int) -> Optional[str]:
        if not self.is_agent_value(value):
            return None
        return self.agents[int(value) - self.agent_value_start]

    def is_agent_value(self, value: int) -> bool:
        return self.agent_value_start <= int(value) < self.obstacle_value

    def is_inside_arena(self, x: int, y: int) -> bool:
        return is_inside_arena(self.arena_mask, x, y)

    def _sample_empty_arena_cell(self, exclude: Optional[set] = None) -> Tuple[int, int]:
        exclude = exclude or set()
        valid_cells = np.argwhere((self.grid == 0) & self.arena_mask)
        if len(valid_cells) == 0:
            raise RuntimeError("No empty cells available inside the octagon arena.")

        while True:
            row, col = valid_cells[np.random.randint(0, len(valid_cells))]
            pos = (int(row), int(col))
            if pos not in exclude:
                return pos

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs_all = {}
        for agent in self.agents:
            if not self.partial_observability:
                obs_all[agent] = self.grid.copy()
                continue

            size = int(self.view_size)
            radius = size // 2

            row, col = self.agent_positions[agent]
            padded = np.pad(self.grid, pad_width=radius, mode="constant", constant_values=0)

            pr = row + radius
            pc = col + radius

            obs = padded[pr - radius : pr + radius + 1, pc - radius : pc + radius + 1]
            obs_all[agent] = obs[:size, :size].copy()
        return obs_all

    def get_heatmaps(self) -> Dict[str, np.ndarray]:
        return self.heatmaps.copy()

    def get_resources_collected(self) -> Dict[str, int]:
        return self.resources_collected.copy()

    def get_final_grid(self) -> np.ndarray:
        return self.grid.copy()

    def get_initial_resource_positions(self) -> List[Tuple[int, int]]:
        return self.initial_resource_positions.copy()

    def reset_communication_flags(self) -> None:
        self.just_communicated = {agent: False for agent in self.agents}

    def mark_proximity_communication(self, max_distance: int = 3) -> None:
        for agent in self.agents:
            agent_pos = self.agent_positions.get(agent)
            if agent_pos is None:
                continue
            for other in self.agents:
                if other == agent:
                    continue
                other_pos = self.agent_positions.get(other)
                if other_pos is None:
                    continue
                distance = abs(agent_pos[0] - other_pos[0]) + abs(agent_pos[1] - other_pos[1])
                if distance < max_distance:
                    self.just_communicated[agent] = True
                    self.just_communicated[other] = True
