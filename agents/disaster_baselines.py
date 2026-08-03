"""Non-learning baselines for the disaster environment.

These exist to prove that learning actually happened. Without them, a trained policy's
"lives saved" number is uninterpretable - we cannot tell competence from luck.

  RandomPolicy  - uniform over actions; the floor.
  GreedyPolicy  - move toward the nearest victim in view, rescue when standing on one,
                  otherwise explore with persistent heading. A strong, sensible heuristic.

Note GreedyPolicy is deliberately *selfish and uncoordinated*: each agent independently
chases its own nearest victim. It therefore rescues severe victims only by accident (when two
agents happen to converge), which makes it a useful reference point for the severe-victim
save rate - the quantity H1 predicts an individual mandate will damage.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from env.disaster_env import (
    STAY, UP, DOWN, LEFT, RIGHT, RESCUE,
    N_ACTIONS, CH_VICTIM, CH_OBSTACLE,
)


class RandomPolicy:
    is_learning = False
    name = "random"

    def __init__(self, n_actions: int = N_ACTIONS, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, agent_id: str) -> int:
        return int(self.rng.integers(0, self.n_actions))


class GreedyPolicy:
    is_learning = False
    name = "greedy"

    def __init__(self, view_size: int = 7, seed: Optional[int] = None):
        self.view_size = view_size
        self.centre = view_size // 2
        self.rng = np.random.default_rng(seed)
        self._heading: Dict[str, int] = {}

    def reset(self) -> None:
        self._heading = {}

    def _explore(self, agent_id: str, obs: np.ndarray) -> int:
        # persistent heading, re-rolled when blocked - covers ground far better than
        # uniform random, which mostly jitters in place.
        h = self._heading.get(agent_id)
        if h is None:
            h = int(self.rng.integers(1, 5))
        dr, dc = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}[h]
        nr, nc = self.centre + dr, self.centre + dc
        blocked = (
            not (0 <= nr < self.view_size and 0 <= nc < self.view_size)
            or obs[CH_OBSTACLE, nr, nc] > 0.5
        )
        if blocked or self.rng.random() < 0.05:
            h = int(self.rng.integers(1, 5))
        self._heading[agent_id] = h
        return h

    def act(self, obs: np.ndarray, agent_id: str) -> int:
        victims = np.argwhere(obs[CH_VICTIM] > 0.5)
        if victims.size == 0:
            return self._explore(agent_id, obs)

        # nearest victim by Manhattan distance from the agent (always at the window centre)
        d = np.abs(victims[:, 0] - self.centre) + np.abs(victims[:, 1] - self.centre)
        vr, vc = victims[int(np.argmin(d))]

        if vr == self.centre and vc == self.centre:
            return RESCUE  # standing on a victim

        # step along the larger axis first, preferring an unblocked move
        opts = []
        if vr < self.centre:
            opts.append((UP, -1, 0))
        elif vr > self.centre:
            opts.append((DOWN, 1, 0))
        if vc < self.centre:
            opts.append((LEFT, 0, -1))
        elif vc > self.centre:
            opts.append((RIGHT, 0, 1))

        for action, dr, dc in opts:
            nr, nc = self.centre + dr, self.centre + dc
            if 0 <= nr < self.view_size and 0 <= nc < self.view_size and obs[CH_OBSTACLE, nr, nc] <= 0.5:
                return action
        return self._explore(agent_id, obs)
