"""Terrain types and village generation for the disaster environment.

Split out from the env so layout generation can be developed and tested on its own.

Terrain is ordinal only in the sense that some types block movement and some block sight;
nothing else is implied by the integer values.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Pos = Tuple[int, int]

# terrain codes
GRASS, ROAD, FLOOR, WALL, RUBBLE, WATER, TREE, DOOR = range(8)

# movement: can an agent enter?
PASSABLE = {GRASS: True, ROAD: True, FLOOR: True, DOOR: True,
            WALL: False, RUBBLE: False, WATER: False, TREE: False}

# sight: does it block line of sight?
OPAQUE = {GRASS: False, ROAD: False, FLOOR: False, DOOR: False,
          WALL: True, RUBBLE: False, WATER: False, TREE: True}


class Building:
    __slots__ = ("r0", "c0", "r1", "c1")

    def __init__(self, r0: int, c0: int, r1: int, c1: int):
        self.r0, self.c0, self.r1, self.c1 = r0, c0, r1, c1

    def interior_cells(self):
        for r in range(self.r0 + 1, self.r1):
            for c in range(self.c0 + 1, self.c1):
                yield (r, c)


def generate_village(
    grid_size: int,
    plot: int = 16,
    min_building: int = 6,
    max_building: int = 13,
    build_prob: float = 0.82,
    rubble_fraction: float = 0.035,
    tree_fraction: float = 0.03,
    water_patches: int = 2,
    rng: Optional[np.random.Generator] = None,
):
    """Village of irregular buildings on jittered plots, separated by roads, with grass,
    trees, and a little water. Returns (terrain, buildings, interior, doors).

    Buildings vary in size and position rather than sitting on a rigid grid, which makes the
    search problem less uniform and the map read like a settlement rather than a lattice.
    """
    rng = rng or np.random.default_rng()
    t = np.full((grid_size, grid_size), GRASS, dtype=np.int8)

    # road network along plot boundaries
    for x in range(0, grid_size, plot):
        t[x:min(x + 2, grid_size), :] = ROAD
        t[:, x:min(x + 2, grid_size)] = ROAD

    buildings: List[Building] = []
    interior: Set[Pos] = set()
    doors: Set[Pos] = set()

    for pr in range(0, grid_size, plot):
        for pc in range(0, grid_size, plot):
            if rng.random() > build_prob:
                continue  # leave a park / open square
            avail = plot - 3
            if avail < min_building:
                continue
            h = int(rng.integers(min_building, min(max_building, avail) + 1))
            w = int(rng.integers(min_building, min(max_building, avail) + 1))
            r0 = pr + 2 + int(rng.integers(0, max(1, avail - h + 1)))
            c0 = pc + 2 + int(rng.integers(0, max(1, avail - w + 1)))
            r1, c1 = r0 + h - 1, c0 + w - 1
            if r1 >= grid_size - 1 or c1 >= grid_size - 1:
                continue

            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r in (r0, r1) or c in (c0, c1):
                        t[r, c] = WALL
                    else:
                        t[r, c] = FLOOR
                        interior.add((r, c))

            b = Building(r0, c0, r1, c1)
            buildings.append(b)

            # doorways
            for _ in range(int(rng.integers(1, 3))):
                side = int(rng.integers(0, 4))
                if side == 0:
                    d = (r0, int(rng.integers(c0 + 1, c1)))
                elif side == 1:
                    d = (r1, int(rng.integers(c0 + 1, c1)))
                elif side == 2:
                    d = (int(rng.integers(r0 + 1, r1)), c0)
                else:
                    d = (int(rng.integers(r0 + 1, r1)), c1)
                t[d] = DOOR
                doors.add(d)

            # interior partitions -> rooms, each with a gap so they stay reachable
            if h >= 8 and rng.random() < 0.65:
                mid = int(rng.integers(r0 + 3, r1 - 2))
                gap = int(rng.integers(c0 + 1, c1))
                for c in range(c0 + 1, c1):
                    if c != gap:
                        t[mid, c] = WALL
                        interior.discard((mid, c))
            if w >= 8 and rng.random() < 0.5:
                mid = int(rng.integers(c0 + 3, c1 - 2))
                gap = int(rng.integers(r0 + 1, r1))
                for r in range(r0 + 1, r1):
                    if r != gap and t[r, mid] != WALL:
                        t[r, mid] = WALL
                        interior.discard((r, mid))

    # water features on open ground
    for _ in range(water_patches):
        cr = int(rng.integers(0, grid_size))
        cc = int(rng.integers(0, grid_size))
        rad = int(rng.integers(2, 5))
        for r in range(max(0, cr - rad), min(grid_size, cr + rad + 1)):
            for c in range(max(0, cc - rad), min(grid_size, cc + rad + 1)):
                if (r - cr) ** 2 + (c - cc) ** 2 <= rad * rad and t[r, c] == GRASS:
                    t[r, c] = WATER

    # trees on grass
    grass = np.argwhere(t == GRASS)
    if len(grass):
        n = int(len(grass) * tree_fraction)
        for i in rng.choice(len(grass), size=min(n, len(grass)), replace=False):
            r, c = grass[int(i)]
            t[r, c] = TREE

    # rubble: debris on passable ground, never on a doorway
    passable = np.argwhere(np.isin(t, [GRASS, ROAD, FLOOR]))
    if len(passable):
        n = int(len(passable) * rubble_fraction)
        for i in rng.choice(len(passable), size=min(n, len(passable)), replace=False):
            r, c = passable[int(i)]
            if (int(r), int(c)) in doors:
                continue
            t[r, c] = RUBBLE
            interior.discard((int(r), int(c)))

    return t, buildings, interior, doors


def passable_mask(terrain: np.ndarray) -> np.ndarray:
    m = np.zeros(terrain.shape, dtype=bool)
    for code, ok in PASSABLE.items():
        if ok:
            m |= terrain == code
    return m


def opaque_mask(terrain: np.ndarray) -> np.ndarray:
    m = np.zeros(terrain.shape, dtype=bool)
    for code, blocks in OPAQUE.items():
        if blocks:
            m |= terrain == code
    return m
