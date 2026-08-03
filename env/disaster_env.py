"""Multi-agency disaster response environment (provision dilemma).

Companion to GridWorldEnv, which is an *appropriation* dilemma (agents take from a shared
pool). Here agents must *expend effort* for collective benefit, and inaction destroys value:
victims carry a countdown and die if unreached.

Four design choices distinguish this from the gridworld (see disaster-response/DESIGN.md):
  1. Severe victims need TWO responders acting simultaneously, so cooperation is genuinely
     required rather than merely being about staying out of each other's way.
  2. A village layout of irregular buildings, roads, trees and water, with victims mostly
     indoors.
  3. **Line of sight**: walls and trees block vision, so a responder cannot see who is inside
     a building from the street. Without this, putting victims indoors would be cosmetic -
     search would be unnecessary.
  4. Multi-channel observations, including an explicit *visibility* channel so an agent can
     distinguish "nothing there" from "cannot see there".

The original gridworld env is untouched; all workshop-paper results remain reproducible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

from env.disaster_terrain import (
    generate_village, passable_mask, opaque_mask,
    GRASS, ROAD, FLOOR, WALL, RUBBLE, WATER, TREE, DOOR,
)
from env.disaster_los import build_ray_template, visible_offsets

# actions
STAY, UP, DOWN, LEFT, RIGHT, RESCUE = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

# observation channels
(CH_OBSTACLE, CH_VICTIM, CH_SEVERITY, CH_URGENCY,
 CH_OWN_AGENCY, CH_OTHER_AGENCY, CH_VISIBLE, CH_INDOOR) = range(8)
N_CHANNELS = 8

Pos = Tuple[int, int]


class Victim:
    __slots__ = ("row", "col", "severity", "ttl", "max_ttl", "indoors")

    def __init__(self, row: int, col: int, severity: int, ttl: int, indoors: bool = False):
        self.row, self.col = row, col
        self.severity = severity          # 1 = minor (1 responder), 2 = severe (2 responders)
        self.ttl = ttl
        self.max_ttl = ttl
        self.indoors = indoors

    @property
    def pos(self) -> Pos:
        return (self.row, self.col)


class DisasterEnv(ParallelEnv):
    metadata = {"name": "disaster_v1", "render_mode": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 80,
        num_agents: int = 12,
        num_agencies: int = 3,
        num_victims: int = 60,
        max_steps: int = 600,
        view_size: int = 9,
        severe_fraction: float = 0.4,
        ttl_range: Tuple[int, int] = (250, 560),
        severe_value: float = 2.0,
        layout: str = "village",
        plot: int = 16,
        rubble_fraction: float = 0.035,
        indoor_victim_fraction: float = 0.75,
        line_of_sight: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1.")
        if not 1 <= num_agencies <= num_agents:
            raise ValueError("num_agencies must be between 1 and num_agents.")

        self.grid_size = grid_size
        self.n_agents = num_agents
        self.num_agencies = num_agencies
        self.num_victims = num_victims
        self.max_steps = max_steps
        self.view_size = view_size
        self.severe_fraction = severe_fraction
        self.ttl_range = ttl_range
        self.severe_value = severe_value
        self.layout = layout
        self.plot = plot
        self.rubble_fraction = rubble_fraction
        self.indoor_victim_fraction = indoor_victim_fraction
        self.line_of_sight = line_of_sight
        self.seed = seed

        self.agents: List[str] = [f"responder_{i}" for i in range(num_agents)]
        self.possible_agents = list(self.agents)

        per = int(np.ceil(num_agents / num_agencies))
        self.agency_of: Dict[str, int] = {
            a: min(i // per, num_agencies - 1) for i, a in enumerate(self.agents)
        }

        self.action_spaces = {a: Discrete(N_ACTIONS) for a in self.agents}
        self.observation_spaces = {
            a: Box(low=0.0, high=1.0, shape=(N_CHANNELS, view_size, view_size), dtype=np.float32)
            for a in self.agents
        }

        self._rays = build_ray_template(view_size)
        self._vis_cache: Dict[Pos, Dict[Tuple[int, int], bool]] = {}

        # state
        self.terrain = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.passable = np.zeros((grid_size, grid_size), dtype=bool)
        self.opaque = np.zeros((grid_size, grid_size), dtype=bool)
        self.buildings: List = []
        self.interior: Set[Pos] = set()
        self.doors: Set[Pos] = set()
        self.victims: List[Victim] = []
        self.agent_positions: Dict[str, Pos] = {}
        self.step_count = 0

        # metrics
        self.rescues: Dict[str, float] = {}
        self.rescue_counts: Dict[str, int] = {}
        self.idle_steps: Dict[str, int] = {}
        self.lives_saved = self.lives_lost = 0
        self.severe_saved = self.minor_saved = 0
        self.severe_lost = self.minor_lost = 0
        self.joint_rescues = 0

        if seed is not None:
            np.random.seed(seed)

    # ---------- layout ----------
    def _reachable(self, start: Pos) -> Set[Pos]:
        seen = {start}
        stack = [start]
        H = self.grid_size
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < H and 0 <= n[1] < H and self.passable[n] and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return seen

    def _largest_region(self) -> Set[Pos]:
        free = set(map(tuple, np.argwhere(self.passable)))
        best: Set[Pos] = set()
        while free:
            comp = self._reachable(next(iter(free)))
            free -= comp
            if len(comp) > len(best):
                best = comp
        return best

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed if seed is None else seed)

        self.step_count = 0
        self.victims = []
        self.agent_positions = {}
        self._vis_cache = {}
        self.rescues = {a: 0.0 for a in self.agents}
        self.rescue_counts = {a: 0 for a in self.agents}
        self.idle_steps = {a: 0 for a in self.agents}
        self.lives_saved = self.lives_lost = 0
        self.severe_saved = self.minor_saved = 0
        self.severe_lost = self.minor_lost = 0
        self.joint_rescues = 0

        if self.layout == "village":
            self.terrain, self.buildings, self.interior, self.doors = generate_village(
                self.grid_size, plot=self.plot, rubble_fraction=self.rubble_fraction, rng=rng
            )
        else:
            self.terrain = np.full((self.grid_size, self.grid_size), GRASS, dtype=np.int8)
            self.buildings, self.interior, self.doors = [], set(), set()

        self.passable = passable_mask(self.terrain)
        self.opaque = opaque_mask(self.terrain)

        # place only within the largest connected region, so every victim is savable
        region = sorted(self._largest_region())
        indoor = [p for p in region if p in self.interior]
        outdoor = [p for p in region if p not in self.interior]
        occupied: Set[Pos] = set()

        for _ in range(self.num_victims):
            want_in = indoor and rng.random() < self.indoor_victim_fraction
            pool = indoor if want_in else (outdoor or indoor)
            choices = [p for p in pool if p not in occupied] or [p for p in region if p not in occupied]
            if not choices:
                break
            pos = choices[int(rng.integers(len(choices)))]
            occupied.add(pos)
            sev = 2 if rng.random() < self.severe_fraction else 1
            ttl = int(rng.integers(self.ttl_range[0], self.ttl_range[1] + 1))
            self.victims.append(Victim(pos[0], pos[1], sev, ttl, indoors=pos in self.interior))

        start_pool = [p for p in (outdoor or region) if p not in occupied]
        for a in self.agents:
            if not start_pool:
                start_pool = [p for p in region if p not in occupied]
            pos = start_pool[int(rng.integers(len(start_pool)))]
            occupied.add(pos)
            start_pool.remove(pos)
            self.agent_positions[a] = pos

        return self._get_obs(), {a: {} for a in self.agents}

    # ---------- dynamics ----------
    def _move(self, pos: Pos, action: int) -> Pos:
        r, c = pos
        if action == UP:
            r -= 1
        elif action == DOWN:
            r += 1
        elif action == LEFT:
            c -= 1
        elif action == RIGHT:
            c += 1
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return pos
        return (r, c) if self.passable[r, c] else pos

    def step(self, actions: Dict[str, int]):
        raw = {a: 0.0 for a in self.agents}

        for a in self.agents:
            act = int(actions[a])
            if act in (UP, DOWN, LEFT, RIGHT):
                self.agent_positions[a] = self._move(self.agent_positions[a], act)
            elif act == STAY:
                self.idle_steps[a] += 1

        rescuers_at: Dict[Pos, List[str]] = {}
        for a in self.agents:
            if int(actions[a]) == RESCUE:
                rescuers_at.setdefault(self.agent_positions[a], []).append(a)

        saved: List[Victim] = []
        for v in self.victims:
            crew = rescuers_at.get(v.pos, [])
            if len(crew) >= v.severity:
                value = self.severe_value if v.severity == 2 else 1.0
                for a in crew:                       # participation is the unit of effort
                    raw[a] += value
                    self.rescue_counts[a] += 1
                    self.rescues[a] += value
                saved.append(v)
                self.lives_saved += 1
                if v.severity == 2:
                    self.severe_saved += 1
                    self.joint_rescues += 1
                else:
                    self.minor_saved += 1
        if saved:
            ids = {id(v) for v in saved}
            self.victims = [v for v in self.victims if id(v) not in ids]

        alive: List[Victim] = []
        for v in self.victims:
            v.ttl -= 1
            if v.ttl <= 0:
                self.lives_lost += 1
                if v.severity == 2:
                    self.severe_lost += 1
                else:
                    self.minor_lost += 1
            else:
                alive.append(v)
        self.victims = alive

        self.step_count += 1
        done = len(self.victims) == 0
        truncated = self.step_count >= self.max_steps
        terminations = {a: done for a in self.agents}
        truncations = {a: (truncated and not done) for a in self.agents}
        infos = {a: {"raw_reward": raw[a]} for a in self.agents}
        return self._get_obs(), raw, terminations, truncations, infos

    # ---------- observation ----------
    def _visible(self, pos: Pos) -> Dict[Tuple[int, int], bool]:
        # walls are static within an episode, so visibility depends only on position:
        # cache it and the per-step cost collapses after the first visit.
        v = self._vis_cache.get(pos)
        if v is None:
            v = visible_offsets(pos, self.opaque, self._rays)
            self._vis_cache[pos] = v
        return v

    def _get_obs(self) -> Dict[str, np.ndarray]:
        half = self.view_size // 2
        g = self.grid_size

        victim_at: Dict[Pos, Victim] = {v.pos: v for v in self.victims}
        agents_at: Dict[Pos, List[str]] = {}
        for b in self.agents:
            agents_at.setdefault(self.agent_positions[b], []).append(b)

        obs: Dict[str, np.ndarray] = {}
        for a in self.agents:
            ar, ac = self.agent_positions[a]
            mine = self.agency_of[a]
            w = np.zeros((N_CHANNELS, self.view_size, self.view_size), dtype=np.float32)
            w[CH_OBSTACLE] = 1.0   # off-map reads as blocked

            vis = self._visible((ar, ac)) if self.line_of_sight else None
            for i in range(self.view_size):
                for j in range(self.view_size):
                    dr, dc = i - half, j - half
                    rr, cc = ar + dr, ac + dc
                    if not (0 <= rr < g and 0 <= cc < g):
                        continue
                    seen = True if vis is None else vis[(dr, dc)]
                    w[CH_VISIBLE, i, j] = 1.0 if seen else 0.0
                    if not seen:
                        # unseen cells are reported as blocked-and-empty; the visibility
                        # channel is what tells the agent this is ignorance, not absence.
                        continue
                    w[CH_OBSTACLE, i, j] = 0.0 if self.passable[rr, cc] else 1.0
                    w[CH_INDOOR, i, j] = 1.0 if (rr, cc) in self.interior else 0.0
                    v = victim_at.get((rr, cc))
                    if v is not None:
                        w[CH_VICTIM, i, j] = 1.0
                        w[CH_SEVERITY, i, j] = v.severity / 2.0
                        w[CH_URGENCY, i, j] = 1.0 - (v.ttl / max(1, v.max_ttl))
                    for b in agents_at.get((rr, cc), ()):
                        if b == a:
                            continue
                        ch = CH_OWN_AGENCY if self.agency_of[b] == mine else CH_OTHER_AGENCY
                        w[ch, i, j] = 1.0
            obs[a] = w
        return obs

    # ---------- reporting ----------
    def get_metrics(self) -> Dict:
        total = self.lives_saved + self.lives_lost + len(self.victims)
        sev_total = self.severe_saved + self.severe_lost
        min_total = self.minor_saved + self.minor_lost
        return {
            "lives_saved": self.lives_saved,
            "lives_lost": self.lives_lost,
            "victims_remaining": len(self.victims),
            "save_rate": self.lives_saved / max(1, total),
            # H1: does an individual mandate skew effort away from victims needing cooperation?
            "severe_save_rate": self.severe_saved / max(1, sev_total),
            "minor_save_rate": self.minor_saved / max(1, min_total),
            "severe_saved": self.severe_saved,
            "minor_saved": self.minor_saved,
            "joint_rescues": self.joint_rescues,
            "rescues_per_agent": dict(self.rescue_counts),
            "value_per_agent": dict(self.rescues),
            "idle_rate": {a: self.idle_steps[a] / max(1, self.step_count) for a in self.agents},
            "steps": self.step_count,
        }

    def agency_totals(self) -> Dict[int, float]:
        out = {g: 0.0 for g in range(self.num_agencies)}
        for a in self.agents:
            out[self.agency_of[a]] += self.rescues[a]
        return out

    # convenience for the renderer
    @property
    def walls(self) -> Set[Pos]:
        return set(map(tuple, np.argwhere(self.terrain == WALL)))

    @property
    def rubble(self) -> Set[Pos]:
        return set(map(tuple, np.argwhere(self.terrain == RUBBLE)))
