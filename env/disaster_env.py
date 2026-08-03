"""Multi-agency disaster response environment (provision dilemma).

Companion to GridWorldEnv, which is an *appropriation* dilemma (agents take from a shared
pool). Here agents must *expend effort* for collective benefit, and inaction destroys value:
victims carry a countdown and die if unreached.

Three design choices distinguish this from the gridworld (see disaster-response/DESIGN.md):
  1. Severe victims need TWO responders acting simultaneously, so cooperation is genuinely
     required rather than merely being about staying out of each other's way.
  2. A structured village layout - buildings with walls and doorways, streets, collapsed
     rubble - so search is a real problem: victims are mostly *inside* buildings and cannot
     be seen from the street. Doorways create chokepoints where congestion matters.
  3. Multi-channel observations, so severity, urgency and agency membership can be expressed
     (the gridworld's single integer per cell could not represent these).

The original gridworld env is untouched; all workshop-paper results remain reproducible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

# actions
STAY, UP, DOWN, LEFT, RIGHT, RESCUE = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

# observation channels
CH_OBSTACLE, CH_VICTIM, CH_SEVERITY, CH_URGENCY, CH_OWN_AGENCY, CH_OTHER_AGENCY = range(6)
N_CHANNELS = 6

Pos = Tuple[int, int]


class Victim:
    __slots__ = ("row", "col", "severity", "ttl", "max_ttl", "indoors")

    def __init__(self, row: int, col: int, severity: int, ttl: int, indoors: bool = False):
        self.row = row
        self.col = col
        self.severity = severity      # 1 = minor (1 responder), 2 = severe (2 responders)
        self.ttl = ttl                # timesteps until death
        self.max_ttl = ttl
        self.indoors = indoors

    @property
    def pos(self) -> Pos:
        return (self.row, self.col)


class DisasterEnv(ParallelEnv):
    metadata = {"name": "disaster_v0", "render_mode": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 48,
        num_agents: int = 8,
        num_agencies: int = 2,
        num_victims: int = 30,
        max_steps: int = 400,
        view_size: int = 7,
        severe_fraction: float = 0.4,
        ttl_range: Tuple[int, int] = (150, 380),
        severe_value: float = 2.0,
        layout: str = "village",          # "village" | "open"
        block_size: int = 10,             # building block pitch (village layout)
        street_width: int = 2,
        rubble_fraction: float = 0.04,    # share of walkable cells blocked by debris
        indoor_victim_fraction: float = 0.75,
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
        self.max_steps = max_steps
        self.view_size = view_size
        self.severe_fraction = severe_fraction
        self.ttl_range = ttl_range
        self.severe_value = severe_value
        self.layout = layout
        self.block_size = block_size
        self.street_width = street_width
        self.rubble_fraction = rubble_fraction
        self.indoor_victim_fraction = indoor_victim_fraction
        self.seed = seed

        self.agents: List[str] = [f"responder_{i}" for i in range(num_agents)]
        self.possible_agents = list(self.agents)

        # agency membership: contiguous blocks (8 agents / 2 agencies -> 0,0,0,0,1,1,1,1)
        per = int(np.ceil(num_agents / num_agencies))
        self.agency_of: Dict[str, int] = {
            a: min(i // per, num_agencies - 1) for i, a in enumerate(self.agents)
        }

        self.action_spaces = {a: Discrete(N_ACTIONS) for a in self.agents}
        self.observation_spaces = {
            a: Box(low=0.0, high=1.0, shape=(N_CHANNELS, view_size, view_size), dtype=np.float32)
            for a in self.agents
        }

        # state
        self.walls: Set[Pos] = set()
        self.rubble: Set[Pos] = set()
        self.interior: Set[Pos] = set()   # cells inside buildings (for placement + rendering)
        self.doors: Set[Pos] = set()
        self.victims: List[Victim] = []
        self.agent_positions: Dict[str, Pos] = {}
        self.step_count = 0

        # metrics
        self.rescues: Dict[str, float] = {}
        self.rescue_counts: Dict[str, int] = {}
        self.idle_steps: Dict[str, int] = {}
        self.lives_saved = 0
        self.lives_lost = 0
        self.severe_saved = 0
        self.minor_saved = 0
        self.severe_lost = 0
        self.minor_lost = 0
        self.joint_rescues = 0

        if seed is not None:
            np.random.seed(seed)

    # ---------- layout generation ----------
    @property
    def blocked(self) -> Set[Pos]:
        return self.walls | self.rubble

    def _generate_village(self) -> None:
        """Lay out city blocks separated by streets; each block is a building with a
        perimeter wall, one or two doorways, and an interior that may be subdivided."""
        self.walls, self.interior, self.doors = set(), set(), set()
        pitch = self.block_size + self.street_width

        for br in range(0, self.grid_size, pitch):
            for bc in range(0, self.grid_size, pitch):
                r0, c0 = br + self.street_width, bc + self.street_width
                r1 = min(r0 + self.block_size - 1, self.grid_size - 1)
                c1 = min(c0 + self.block_size - 1, self.grid_size - 1)
                if r1 - r0 < 3 or c1 - c0 < 3:
                    continue  # too small to be a building

                # perimeter wall
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        on_edge = r in (r0, r1) or c in (c0, c1)
                        if on_edge:
                            self.walls.add((r, c))
                        else:
                            self.interior.add((r, c))

                # doorways: punch 1-2 openings on random walls so the building is enterable
                for _ in range(int(np.random.randint(1, 3))):
                    side = int(np.random.randint(0, 4))
                    if side == 0:
                        d = (r0, int(np.random.randint(c0 + 1, c1)))
                    elif side == 1:
                        d = (r1, int(np.random.randint(c0 + 1, c1)))
                    elif side == 2:
                        d = (int(np.random.randint(r0 + 1, r1)), c0)
                    else:
                        d = (int(np.random.randint(r0 + 1, r1)), c1)
                    self.walls.discard(d)
                    self.doors.add(d)

                # optional interior partition with a gap, creating rooms
                if r1 - r0 >= 6 and np.random.random() < 0.6:
                    mid = int(np.random.randint(r0 + 2, r1 - 1))
                    gap = int(np.random.randint(c0 + 1, c1))
                    for c in range(c0 + 1, c1):
                        if c != gap:
                            self.walls.add((mid, c))
                            self.interior.discard((mid, c))

    def _generate_open(self) -> None:
        self.walls, self.interior, self.doors = set(), set(), set()

    def _scatter_rubble(self) -> None:
        walkable = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.walls
        ]
        n = int(len(walkable) * self.rubble_fraction)
        self.rubble = set()
        if n <= 0 or not walkable:
            return
        idx = np.random.choice(len(walkable), size=min(n, len(walkable)), replace=False)
        for i in idx:
            cell = walkable[int(i)]
            if cell in self.doors:
                continue  # never block a doorway entirely
            self.rubble.add(cell)

    def _free_cells(self) -> List[Pos]:
        blocked = self.blocked
        return [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in blocked
        ]

    def _reachable_cells(self) -> Set[Pos]:
        """Largest connected walkable region. Victims and responders are placed only here, so
        every victim is in principle savable - otherwise rubble can seal a room and create
        guaranteed deaths that no policy could prevent, which would just add noise."""
        blocked = self.blocked
        free = set(self._free_cells())
        best: Set[Pos] = set()
        unvisited = set(free)
        while unvisited:
            start = next(iter(unvisited))
            comp: Set[Pos] = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if (
                        0 <= n[0] < self.grid_size
                        and 0 <= n[1] < self.grid_size
                        and n not in blocked
                        and n not in comp
                    ):
                        comp.add(n)
                        stack.append(n)
            unvisited -= comp
            if len(comp) > len(best):
                best = comp
        return best

    # ---------- api ----------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        elif self.seed is not None:
            np.random.seed(self.seed)

        self.step_count = 0
        self.victims = []
        self.agent_positions = {}
        self.rescues = {a: 0.0 for a in self.agents}
        self.rescue_counts = {a: 0 for a in self.agents}
        self.idle_steps = {a: 0 for a in self.agents}
        self.lives_saved = self.lives_lost = 0
        self.severe_saved = self.minor_saved = 0
        self.severe_lost = self.minor_lost = 0
        self.joint_rescues = 0

        if self.layout == "village":
            self._generate_village()
        else:
            self._generate_open()
        self._scatter_rubble()

        # restrict placement to the connected region so every victim is savable
        free = sorted(self._reachable_cells())
        indoor = [p for p in free if p in self.interior]
        outdoor = [p for p in free if p not in self.interior]
        occupied: Set[Pos] = set()

        # victims: mostly indoors, so they cannot be seen from the street
        for _ in range(self.num_victims):
            want_indoor = indoor and np.random.random() < self.indoor_victim_fraction
            pool = indoor if want_indoor else (outdoor or indoor)
            choices = [p for p in pool if p not in occupied]
            if not choices:
                choices = [p for p in free if p not in occupied]
            if not choices:
                break
            pos = choices[int(np.random.randint(len(choices)))]
            occupied.add(pos)
            severity = 2 if np.random.random() < self.severe_fraction else 1
            ttl = int(np.random.randint(self.ttl_range[0], self.ttl_range[1] + 1))
            self.victims.append(Victim(pos[0], pos[1], severity, ttl, indoors=pos in self.interior))

        # responders start outside, on the street, as arriving units would
        start_pool = [p for p in (outdoor or free) if p not in occupied]
        for a in self.agents:
            if not start_pool:
                start_pool = [p for p in free if p not in occupied]
            pos = start_pool[int(np.random.randint(len(start_pool)))]
            occupied.add(pos)
            start_pool = [p for p in start_pool if p != pos]
            self.agent_positions[a] = pos

        return self._get_obs(), {a: {} for a in self.agents}

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
        if (r, c) in self.walls or (r, c) in self.rubble:
            return pos
        return (r, c)

    def step(self, actions: Dict[str, int]):
        raw = {a: 0.0 for a in self.agents}

        for a in self.agents:
            act = int(actions[a])
            if act in (UP, DOWN, LEFT, RIGHT):
                self.agent_positions[a] = self._move(self.agent_positions[a], act)
            elif act == STAY:
                self.idle_steps[a] += 1

        # rescues: group rescuers by the cell they occupy. A victim is saved when the number
        # of responders issuing RESCUE on its cell meets its severity requirement.
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
    def _get_obs(self) -> Dict[str, np.ndarray]:
        half = self.view_size // 2
        g = self.grid_size
        full = np.zeros((4, g, g), dtype=np.float32)  # obstacle, victim, severity, urgency
        for (r, c) in self.walls:
            full[0, r, c] = 1.0
        for (r, c) in self.rubble:
            full[0, r, c] = 1.0
        for v in self.victims:
            full[1, v.row, v.col] = 1.0
            full[2, v.row, v.col] = v.severity / 2.0
            full[3, v.row, v.col] = 1.0 - (v.ttl / max(1, v.max_ttl))

        pos_by_agency: Dict[int, List[Pos]] = {}
        for b in self.agents:
            pos_by_agency.setdefault(self.agency_of[b], []).append(self.agent_positions[b])

        obs: Dict[str, np.ndarray] = {}
        for a in self.agents:
            ar, ac = self.agent_positions[a]
            mine = self.agency_of[a]
            w = np.zeros((N_CHANNELS, self.view_size, self.view_size), dtype=np.float32)

            r0, r1 = ar - half, ar + half + 1
            c0, c1 = ac - half, ac + half + 1
            sr0, sc0 = max(0, r0), max(0, c0)
            sr1, sc1 = min(g, r1), min(g, c1)
            dr0, dc0 = sr0 - r0, sc0 - c0
            dr1, dc1 = dr0 + (sr1 - sr0), dc0 + (sc1 - sc0)

            w[CH_OBSTACLE] = 1.0  # outside the map reads as blocked
            if sr1 > sr0 and sc1 > sc0:
                w[CH_OBSTACLE, dr0:dr1, dc0:dc1] = full[0, sr0:sr1, sc0:sc1]
                w[CH_VICTIM, dr0:dr1, dc0:dc1] = full[1, sr0:sr1, sc0:sc1]
                w[CH_SEVERITY, dr0:dr1, dc0:dc1] = full[2, sr0:sr1, sc0:sc1]
                w[CH_URGENCY, dr0:dr1, dc0:dc1] = full[3, sr0:sr1, sc0:sc1]

            for g_id, positions in pos_by_agency.items():
                ch = CH_OWN_AGENCY if g_id == mine else CH_OTHER_AGENCY
                for (br, bc) in positions:
                    if (br, bc) == (ar, ac) and g_id == mine:
                        continue  # don't mark self
                    i, j = br - r0, bc - c0
                    if 0 <= i < self.view_size and 0 <= j < self.view_size:
                        w[ch, i, j] = 1.0
            obs[a] = w
        return obs

    # ---------- reporting ----------
    def get_metrics(self) -> Dict:
        total = self.lives_saved + self.lives_lost + len(self.victims)
        severe_total = self.severe_saved + self.severe_lost
        minor_total = self.minor_saved + self.minor_lost
        return {
            "lives_saved": self.lives_saved,
            "lives_lost": self.lives_lost,
            "victims_remaining": len(self.victims),
            "save_rate": self.lives_saved / max(1, total),
            # H1: does an individual mandate skew effort away from victims needing cooperation?
            "severe_save_rate": self.severe_saved / max(1, severe_total),
            "minor_save_rate": self.minor_saved / max(1, minor_total),
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
