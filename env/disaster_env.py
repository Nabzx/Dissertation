"""Multi-agency disaster response environment (provision dilemma).

Companion to GridWorldEnv, which is an *appropriation* dilemma (agents take from a shared
pool). Here agents must *expend effort* for collective benefit, and inaction destroys value:
victims carry a countdown and die if unreached.

Two design choices distinguish this from the gridworld (see disaster-response/DESIGN.md):
  1. Severe victims need TWO responders acting simultaneously, so cooperation is genuinely
     required rather than merely being about staying out of each other's way.
  2. Observations are multi-channel, so severity, urgency and agency membership can be
     expressed (the gridworld's single integer per cell could not represent these).

The original gridworld env is untouched; all workshop-paper results remain reproducible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

# actions
STAY, UP, DOWN, LEFT, RIGHT, RESCUE = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

# observation channels
CH_OBSTACLE, CH_VICTIM, CH_SEVERITY, CH_URGENCY, CH_OWN_AGENCY, CH_OTHER_AGENCY = range(6)
N_CHANNELS = 6


class Victim:
    __slots__ = ("row", "col", "severity", "ttl", "max_ttl")

    def __init__(self, row: int, col: int, severity: int, ttl: int):
        self.row = row
        self.col = col
        self.severity = severity      # 1 = minor (1 responder), 2 = severe (2 responders)
        self.ttl = ttl                # timesteps until death
        self.max_ttl = ttl

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)


class DisasterEnv(ParallelEnv):
    metadata = {"name": "disaster_v0", "render_mode": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 40,
        num_agents: int = 8,
        num_agencies: int = 2,
        num_victims: int = 30,
        num_obstacles: int = 120,
        max_steps: int = 400,
        view_size: int = 7,
        severe_fraction: float = 0.4,      # share of victims needing two responders
        ttl_range: Tuple[int, int] = (120, 320),
        severe_value: float = 2.0,         # a severe rescue is worth more than a minor one
        seed: Optional[int] = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")
        if num_agencies < 1 or num_agencies > num_agents:
            raise ValueError("num_agencies must be between 1 and num_agents.")

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.num_agencies = num_agencies
        self.num_victims = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.view_size = view_size
        self.severe_fraction = severe_fraction
        self.ttl_range = ttl_range
        self.severe_value = severe_value
        self.seed = seed

        self.agents: List[str] = [f"responder_{i}" for i in range(num_agents)]
        self.possible_agents = list(self.agents)

        # agency membership: contiguous blocks, e.g. 8 agents / 2 agencies -> 0,0,0,0,1,1,1,1
        per = int(np.ceil(num_agents / num_agencies))
        self.agency_of: Dict[str, int] = {
            a: min(i // per, num_agencies - 1) for i, a in enumerate(self.agents)
        }

        self.action_spaces = {a: Discrete(N_ACTIONS) for a in self.agents}
        obs_shape = (N_CHANNELS, view_size, view_size)
        self.observation_spaces = {
            a: Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32) for a in self.agents
        }

        # state
        self.victims: List[Victim] = []
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self.obstacles: set = set()
        self.step_count = 0

        # metrics
        self.rescues: Dict[str, float] = {}      # value credited per agent (participation)
        self.rescue_counts: Dict[str, int] = {}  # number of rescues participated in
        self.idle_steps: Dict[str, int] = {}
        self.lives_saved = 0
        self.lives_lost = 0
        self.severe_saved = 0

        if seed is not None:
            np.random.seed(seed)

    # ---------- setup ----------
    def _free_cell(self, exclude: Optional[set] = None) -> Tuple[int, int]:
        exclude = exclude or set()
        for _ in range(10000):
            r = int(np.random.randint(0, self.grid_size))
            c = int(np.random.randint(0, self.grid_size))
            if (r, c) in self.obstacles or (r, c) in exclude:
                continue
            return (r, c)
        raise RuntimeError("no free cell available")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)

        self.step_count = 0
        self.obstacles = set()
        self.victims = []
        self.agent_positions = {}

        self.rescues = {a: 0.0 for a in self.agents}
        self.rescue_counts = {a: 0 for a in self.agents}
        self.idle_steps = {a: 0 for a in self.agents}
        self.lives_saved = 0
        self.lives_lost = 0
        self.severe_saved = 0

        # rubble
        while len(self.obstacles) < self.num_obstacles:
            r = int(np.random.randint(0, self.grid_size))
            c = int(np.random.randint(0, self.grid_size))
            self.obstacles.add((r, c))

        # victims
        occupied = set()
        for _ in range(self.num_victims):
            pos = self._free_cell(exclude=occupied)
            occupied.add(pos)
            severity = 2 if np.random.random() < self.severe_fraction else 1
            ttl = int(np.random.randint(self.ttl_range[0], self.ttl_range[1] + 1))
            self.victims.append(Victim(pos[0], pos[1], severity, ttl))

        # responders
        for a in self.agents:
            pos = self._free_cell(exclude=occupied | set(self.agent_positions.values()))
            self.agent_positions[a] = pos

        return self._get_obs(), {a: {} for a in self.agents}

    # ---------- dynamics ----------
    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        r, c = pos
        if action == UP:
            r -= 1
        elif action == DOWN:
            r += 1
        elif action == LEFT:
            c -= 1
        elif action == RIGHT:
            c += 1
        # out of bounds or rubble -> stay put (agents must learn to route around)
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return pos
        if (r, c) in self.obstacles:
            return pos
        return (r, c)

    def step(self, actions: Dict[str, int]):
        raw = {a: 0.0 for a in self.agents}

        # --- movement ---
        for a in self.agents:
            act = int(actions[a])
            if act in (UP, DOWN, LEFT, RIGHT):
                self.agent_positions[a] = self._move(self.agent_positions[a], act)
            elif act == STAY:
                self.idle_steps[a] += 1

        # --- rescues: group rescuers by the cell they are standing on ---
        # a victim is saved when the number of responders issuing RESCUE on its cell in this
        # timestep meets its severity requirement (1 for minor, 2 for severe).
        rescuers_at: Dict[Tuple[int, int], List[str]] = {}
        for a in self.agents:
            if int(actions[a]) == RESCUE:
                rescuers_at.setdefault(self.agent_positions[a], []).append(a)

        saved: List[Victim] = []
        for v in self.victims:
            crew = rescuers_at.get(v.pos, [])
            if len(crew) >= v.severity:
                value = self.severe_value if v.severity == 2 else 1.0
                # every participant is credited: participation is the unit of effort
                for a in crew:
                    raw[a] += value
                    self.rescue_counts[a] += 1
                    self.rescues[a] += value
                saved.append(v)
                self.lives_saved += 1
                if v.severity == 2:
                    self.severe_saved += 1

        if saved:
            saved_ids = {id(v) for v in saved}
            self.victims = [v for v in self.victims if id(v) not in saved_ids]

        # --- countdown: unreached victims die ---
        still_alive: List[Victim] = []
        for v in self.victims:
            v.ttl -= 1
            if v.ttl <= 0:
                self.lives_lost += 1
            else:
                still_alive.append(v)
        self.victims = still_alive

        self.step_count += 1

        # --- termination ---
        done = len(self.victims) == 0
        truncated = self.step_count >= self.max_steps
        terminations = {a: done for a in self.agents}
        truncations = {a: (truncated and not done) for a in self.agents}

        infos = {a: {"raw_reward": raw[a]} for a in self.agents}
        return self._get_obs(), raw, terminations, truncations, infos

    # ---------- observation ----------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        half = self.view_size // 2
        # build full-grid channels once per step, then slice per agent
        full = np.zeros((N_CHANNELS, self.grid_size, self.grid_size), dtype=np.float32)

        for (r, c) in self.obstacles:
            full[CH_OBSTACLE, r, c] = 1.0

        for v in self.victims:
            full[CH_VICTIM, v.row, v.col] = 1.0
            full[CH_SEVERITY, v.row, v.col] = v.severity / 2.0
            # urgency rises to 1.0 as the victim approaches death
            full[CH_URGENCY, v.row, v.col] = 1.0 - (v.ttl / max(1, v.max_ttl))

        obs: Dict[str, np.ndarray] = {}
        for a in self.agents:
            ar, ac = self.agent_positions[a]
            mine = self.agency_of[a]
            # agency channels are viewer-relative, so they are filled per agent
            own = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            other = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            for b in self.agents:
                if b == a:
                    continue
                br, bc = self.agent_positions[b]
                if self.agency_of[b] == mine:
                    own[br, bc] = 1.0
                else:
                    other[br, bc] = 1.0

            window = np.zeros((N_CHANNELS, self.view_size, self.view_size), dtype=np.float32)
            for i in range(self.view_size):
                for j in range(self.view_size):
                    rr, cc = ar - half + i, ac - half + j
                    if not (0 <= rr < self.grid_size and 0 <= cc < self.grid_size):
                        window[CH_OBSTACLE, i, j] = 1.0  # outside the zone reads as blocked
                        continue
                    window[CH_OBSTACLE, i, j] = full[CH_OBSTACLE, rr, cc]
                    window[CH_VICTIM, i, j] = full[CH_VICTIM, rr, cc]
                    window[CH_SEVERITY, i, j] = full[CH_SEVERITY, rr, cc]
                    window[CH_URGENCY, i, j] = full[CH_URGENCY, rr, cc]
                    window[CH_OWN_AGENCY, i, j] = own[rr, cc]
                    window[CH_OTHER_AGENCY, i, j] = other[rr, cc]
            obs[a] = window
        return obs

    # ---------- reporting ----------
    def get_metrics(self) -> Dict:
        total = self.lives_saved + self.lives_lost + len(self.victims)
        return {
            "lives_saved": self.lives_saved,
            "lives_lost": self.lives_lost,
            "victims_remaining": len(self.victims),
            "save_rate": self.lives_saved / max(1, total),
            "severe_saved": self.severe_saved,
            "rescues_per_agent": dict(self.rescue_counts),
            "value_per_agent": dict(self.rescues),
            "idle_rate": {
                a: self.idle_steps[a] / max(1, self.step_count) for a in self.agents
            },
            "steps": self.step_count,
        }

    def agency_totals(self) -> Dict[int, float]:
        out: Dict[int, float] = {g: 0.0 for g in range(self.num_agencies)}
        for a in self.agents:
            out[self.agency_of[a]] += self.rescues[a]
        return out
