from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from minigames.base_game import ArenaScenario


Position = Tuple[int, int]


class CaptureFlagGame(ArenaScenario):
    def __init__(
        self,
        env,
        win_reward: float = 10.0,
        distance_delta_reward: float = 0.01,
        step_penalty: float = -0.01,
    ):
        super().__init__(env)
        self.win_reward = win_reward
        self.distance_delta_reward = distance_delta_reward
        self.step_penalty = step_penalty
        self.teams = self._build_teams()
        self.flag_position: Optional[Position] = None
        self.winner: Optional[str] = None
        self.done = False
        self.previous_distances: Dict[str, int] = {}
        self.current_distances: Dict[str, int] = {}
        self.last_rewards: Dict[str, float] = {}

    def reset(self):
        self._spawn_teams_near_bottom()
        self.flag_position = self._pick_flag_pos()
        self.winner = None
        self.done = False
        self.current_distances = self._get_dists()
        self.previous_distances = self.current_distances.copy()
        self.last_rewards = {agent: 0.0 for agent in self.env.agents}
        return None

    def step(self, actions):
        if self.flag_position is None:
            self.flag_position = self._pick_flag_pos()

        self.current_distances = self._get_dists()
        self.winner = self._find_winner()
        self.done = self.winner is not None
        self.last_rewards = self._get_rewards()
        self.previous_distances = self.current_distances.copy()
        if not self.done:
            return {
                "terminations": {agent: False for agent in self.env.agents},
            }
        return None

    def compute_arena_rewards(self):
        return self.last_rewards.copy()

    def is_round_done(self):
        return self.done

    def get_metrics(self):
        return {
            "mode": "capture_flag",
            "flag_position": self.flag_position,
            "winner": self.winner,
            "teams": self.teams.copy(),
            "distances_to_flag": self.current_distances.copy(),
            "done": self.done,
        }

    def get_arena_render_info(self):
        return {
            "mode": "capture_flag",
            "flag_position": self.flag_position,
            "winner": self.winner,
            "teams": self.teams.copy(),
            "distances_to_flag": self.current_distances.copy(),
            "flag_color": "#facc15",
            "winner_highlight": self.winner is not None,
        }

    def _pick_flag_pos(self) -> Position:
        arena_mask = getattr(self.env, "arena_mask", np.ones_like(self.env.grid, dtype=bool))
        top_limit = max(1, int(self.env.grid_size * 0.30))
        top_mask = np.zeros_like(arena_mask, dtype=bool)
        top_mask[:top_limit, :] = True
        occupied = set(self.env.agent_positions.values())
        valid_cells = [
            tuple(map(int, cell))
            for cell in np.argwhere((self.env.grid == 0) & arena_mask & top_mask)
            if tuple(map(int, cell)) not in occupied
        ]
        if len(valid_cells) == 0:
            raise RuntimeError("No valid empty arena cell available for Capture the Flag.")

        return valid_cells[np.random.randint(0, len(valid_cells))]

    def _get_dists(self) -> Dict[str, int]:
        if self.flag_position is None:
            return {agent: 0 for agent in self.env.agents}

        flag_row, flag_col = self.flag_position
        dists = {}
        for agent, position in self.env.agent_positions.items():
            row, col = position
            dists[agent] = abs(row - flag_row) + abs(col - flag_col)
        return dists

    def _find_winner(self) -> Optional[str]:
        if self.flag_position is None:
            return None

        for agent, position in self.env.agent_positions.items():
            if position == self.flag_position:
                return agent
        return None

    def _get_rewards(self) -> Dict[str, float]:
        rewards = {}
        for agent in self.env.agents:
            reward = self.step_penalty
            previous = self.previous_distances.get(agent)
            current = self.current_distances.get(agent)

            if previous is not None and current is not None:
                reward += (previous - current) * self.distance_delta_reward

            if agent == self.winner:
                reward += self.win_reward

            rewards[agent] = float(reward)
        return rewards

    def _build_teams(self) -> Dict[str, str]:
        return {
            agent: "Team A" if idx < 2 else "Team B"
            for idx, agent in enumerate(self.env.agents)
        }

    def _spawn_teams_near_bottom(self) -> None:
        for agent, position in self.env.agent_positions.items():
            if self.env.grid[position[0], position[1]] == self.env.agent_value(agent):
                self.env.grid[position[0], position[1]] = 0

        team_groups = [
            self.env.agents[0:2],
            self.env.agents[2:4],
        ]
        if len(self.env.agents) > 4:
            team_groups.append(self.env.agents[4:])

        used: set[Position] = set()
        anchor: Optional[Position] = None
        for group in team_groups:
            if not group:
                continue
            positions = self._sample_adjacent_spawn_group(len(group), used, anchor)
            if anchor is None and positions:
                anchor = positions[0]
            for agent, position in zip(group, positions):
                self.env.agent_positions[agent] = position
                self.env.grid[position[0], position[1]] = self.env.agent_value(agent)
                used.add(position)

        if not hasattr(self.env, "heatmaps"):
            return

        for heatmap in self.env.heatmaps.values():
            heatmap[:, :] = 0
        for agent, position in self.env.agent_positions.items():
            self.env.heatmaps[agent][position[0], position[1]] += 1

    def _sample_adjacent_spawn_group(
        self,
        group_size: int,
        used: set[Position],
        anchor: Optional[Position] = None,
    ) -> List[Position]:
        bottom_start = max(0, int(self.env.grid_size * 0.70))
        candidates: List[List[Position]] = []

        for row in range(bottom_start, self.env.grid_size):
            for col in range(0, self.env.grid_size - group_size + 1):
                group = [(row, col + offset) for offset in range(group_size)]
                if self._is_valid_spawn_group(group, used):
                    candidates.append(group)

        if not candidates:
            for row in range(self.env.grid_size - 1, -1, -1):
                for col in range(0, self.env.grid_size - group_size + 1):
                    group = [(row, col + offset) for offset in range(group_size)]
                    if self._is_valid_spawn_group(group, used):
                        candidates.append(group)

        if not candidates:
            raise RuntimeError("No valid adjacent team spawn group available.")

        if anchor is not None:
            nearby = [
                group for group in candidates
                if abs(group[0][0] - anchor[0]) <= 2
                and min(abs(col - anchor[1]) for _, col in group) <= 6
            ]
            if nearby:
                candidates = nearby

        return candidates[np.random.randint(0, len(candidates))]

    def _is_valid_spawn_group(self, group: List[Position], used: set[Position]) -> bool:
        for row, col in group:
            if (row, col) in used:
                return False
            if not self.env.is_inside_arena(row, col):
                return False
            if self.env.grid[row, col] == self.env.obstacle_value:
                return False
        return True
