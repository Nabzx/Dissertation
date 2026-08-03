"""Coordinated greedy baseline - the coordination *ceiling*.

This policy is deliberately **privileged**: it sees the full map and all victims, plans with
BFS, and assigns responders centrally. It is not a decentralised policy and we never claim it
is. Its purpose is to answer "how many lives are achievable here if coordination were solved?"
- without it we cannot tell whether a gap between PPO and greedy is real headroom or a limit
imposed by the task itself.

Contrast with GreedyPolicy, which is decentralised and uncoordinated: it chases whatever
victim it can see and rescues severe ones only when two agents happen to collide.

Assignment rule: victims are prioritised by urgency (soonest death first), then each is given
the nearest available responder(s) - two for severe, one for minor. Assignees path to the
victim and issue RESCUE on arrival, waiting there until their partner arrives.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from env.disaster_env import STAY, UP, DOWN, LEFT, RIGHT, RESCUE

Pos = Tuple[int, int]


class CoordinatedGreedyPolicy:
    is_learning = False
    name = "coordinated"
    privileged = True          # sees global state; a ceiling reference, not a fair policy

    def __init__(self, env, seed: Optional[int] = None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self._assign: Dict[str, Pos] = {}

    def reset(self) -> None:
        self._assign = {}

    # ---------- pathfinding ----------
    def _bfs_step(self, start: Pos, goal: Pos) -> int:
        """First move along a shortest path from start to goal, or STAY if unreachable."""
        if start == goal:
            return STAY
        env = self.env
        g = env.grid_size
        prev: Dict[Pos, Pos] = {start: start}
        q = deque([start])
        found = False
        while q:
            cur = q.popleft()
            if cur == goal:
                found = True
                break
            r, c = cur
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + dr, c + dc)
                if (0 <= n[0] < g and 0 <= n[1] < g and env.passable[n] and n not in prev):
                    prev[n] = cur
                    q.append(n)
        if not found:
            return STAY
        # walk back to the first step from start
        cur = goal
        while prev[cur] != start:
            cur = prev[cur]
        dr, dc = cur[0] - start[0], cur[1] - start[1]
        return {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}.get((dr, dc), STAY)

    @staticmethod
    def _manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ---------- assignment ----------
    def _plan(self) -> None:
        env = self.env
        self._assign = {}
        free: Set[str] = set(env.agents)

        # soonest-to-die first: triage by urgency, which is what a coordinator would do
        victims = sorted(env.victims, key=lambda v: v.ttl)
        for v in victims:
            need = v.severity
            if len(free) < need:
                continue
            nearest = sorted(free, key=lambda a: self._manhattan(env.agent_positions[a], v.pos))
            crew = nearest[:need]
            # skip victims nobody can plausibly reach in time
            if self._manhattan(env.agent_positions[crew[-1]], v.pos) > v.ttl:
                continue
            for a in crew:
                self._assign[a] = v.pos
                free.discard(a)

    def act(self, obs: np.ndarray, agent_id: str) -> int:
        env = self.env
        # replan whenever this agent has no target or its target is gone (rescued/died)
        target = self._assign.get(agent_id)
        live = {v.pos for v in env.victims}
        if target is None or target not in live:
            self._plan()
            target = self._assign.get(agent_id)
        if target is None:
            return STAY

        pos = env.agent_positions[agent_id]
        if pos == target:
            # issue RESCUE and hold: for a severe victim this fails harmlessly until the
            # assigned partner arrives, which is exactly the rendezvous behaviour we want.
            return RESCUE
        return self._bfs_step(pos, target)
