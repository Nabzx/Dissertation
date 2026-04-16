"""Capture the Flag minigame for the GridWorld arena."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from minigames.base_game import GameMode


Position = Tuple[int, int]


class CaptureFlagGame(GameMode):
    """
    Competitive race mode where the first agent to reach a flag wins.

    The wrapped environment still handles normal movement, obstacles, arena
    bounds, and observations. This game mode adds a separate flag target,
    custom rewards, and a winner-based termination condition.
    """

    def __init__(
        self,
        env,
        win_reward: float = 10.0,
        closer_reward: float = 0.05,
        farther_penalty: float = -0.02,
        step_penalty: float = -0.01,
    ):
        super().__init__(env)
        self.win_reward = win_reward
        self.closer_reward = closer_reward
        self.farther_penalty = farther_penalty
        self.step_penalty = step_penalty
        self.flag_position: Optional[Position] = None
        self.winner: Optional[str] = None
        self.done = False
        self.previous_distances: Dict[str, int] = {}
        self.current_distances: Dict[str, int] = {}
        self.last_rewards: Dict[str, float] = {}

    def reset(self):
        self.flag_position = self._sample_flag_position()
        self.winner = None
        self.done = False
        self.current_distances = self._compute_distances()
        self.previous_distances = self.current_distances.copy()
        self.last_rewards = {agent: 0.0 for agent in self.env.agents}
        return None

    def step(self, actions):
        if self.flag_position is None:
            self.flag_position = self._sample_flag_position()

        self.current_distances = self._compute_distances()
        self.winner = self._find_winner()
        self.done = self.winner is not None
        self.last_rewards = self._build_rewards()
        self.previous_distances = self.current_distances.copy()
        return None

    def compute_rewards(self):
        return self.last_rewards.copy()

    def is_done(self):
        return self.done

    def get_metrics(self):
        return {
            "mode": "capture_flag",
            "flag_position": self.flag_position,
            "winner": self.winner,
            "distances_to_flag": self.current_distances.copy(),
            "done": self.done,
        }

    def get_render_info(self):
        return {
            "mode": "capture_flag",
            "flag_position": self.flag_position,
            "winner": self.winner,
            "distances_to_flag": self.current_distances.copy(),
            "flag_color": "#facc15",
            "winner_highlight": self.winner is not None,
        }

    def _sample_flag_position(self) -> Position:
        """Pick an empty in-arena tile so the flag does not overlap entities."""
        arena_mask = getattr(self.env, "arena_mask", np.ones_like(self.env.grid, dtype=bool))
        valid_cells = np.argwhere((self.env.grid == 0) & arena_mask)
        if len(valid_cells) == 0:
            raise RuntimeError("No valid empty arena cell available for Capture the Flag.")

        row, col = valid_cells[np.random.randint(0, len(valid_cells))]
        return int(row), int(col)

    def _compute_distances(self) -> Dict[str, int]:
        if self.flag_position is None:
            return {agent: 0 for agent in self.env.agents}

        flag_row, flag_col = self.flag_position
        distances = {}
        for agent, position in self.env.agent_positions.items():
            row, col = position
            distances[agent] = abs(row - flag_row) + abs(col - flag_col)
        return distances

    def _find_winner(self) -> Optional[str]:
        if self.flag_position is None:
            return None

        for agent, position in self.env.agent_positions.items():
            if position == self.flag_position:
                return agent
        return None

    def _build_rewards(self) -> Dict[str, float]:
        rewards = {}
        for agent in self.env.agents:
            reward = self.step_penalty
            previous = self.previous_distances.get(agent)
            current = self.current_distances.get(agent)

            if previous is not None and current is not None:
                if current < previous:
                    reward += self.closer_reward
                elif current > previous:
                    reward += self.farther_penalty

            if agent == self.winner:
                reward += self.win_reward

            rewards[agent] = float(reward)
        return rewards
