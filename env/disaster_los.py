"""Line-of-sight for the disaster environment.

Without this, an agent standing in the street can see victims inside a building, which
defeats the point of putting victims indoors: searching would be unnecessary. With it,
walls and trees block sight, so a responder must actually enter a structure to find who is
inside - which is what makes coordination and area allocation matter.

Rays are precomputed once per (view_size) and reused, so the per-step cost is just walking
short integer paths.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Offset = Tuple[int, int]


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Offset]:
    """Integer line from (r0,c0) to (r1,c1), excluding the origin."""
    cells: List[Offset] = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        cells.append((r, c))
    return cells


def build_ray_template(view_size: int) -> Dict[Offset, List[Offset]]:
    """For each offset in the window, the cells crossed on the way there (origin-relative)."""
    half = view_size // 2
    rays: Dict[Offset, List[Offset]] = {}
    for i in range(view_size):
        for j in range(view_size):
            dr, dc = i - half, j - half
            if (dr, dc) == (0, 0):
                rays[(dr, dc)] = []
            else:
                rays[(dr, dc)] = _bresenham(0, 0, dr, dc)
    return rays


def visible_offsets(
    agent_rc: Tuple[int, int],
    opaque: np.ndarray,
    rays: Dict[Offset, List[Offset]],
) -> Dict[Offset, bool]:
    """Which window offsets the agent can actually see. An opaque cell is itself visible
    (you can see the wall) but blocks everything beyond it."""
    ar, ac = agent_rc
    H, W = opaque.shape
    out: Dict[Offset, bool] = {}
    for off, path in rays.items():
        vis = True
        for (dr, dc) in path[:-1]:          # cells strictly between origin and target
            rr, cc = ar + dr, ac + dc
            if not (0 <= rr < H and 0 <= cc < W) or opaque[rr, cc]:
                vis = False
                break
        out[off] = vis
    return out
