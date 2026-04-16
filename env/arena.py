"""
Shared arena geometry helpers.

The environment remains grid-based, but these helpers define the octagonal
arena mask used by both movement constraints and live rendering.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def build_octagon_vertices(rows: int, cols: int) -> np.ndarray:
    """Return octagon vertices in plot/grid coordinate space."""
    inset = 1.5
    return np.array(
        [
            [inset + 1.2, 0.4],
            [cols - inset - 1.2, 0.4],
            [cols - 0.4, inset + 1.2],
            [cols - 0.4, rows - inset - 1.2],
            [cols - inset - 1.2, rows - 0.4],
            [inset + 1.2, rows - 0.4],
            [0.4, rows - inset - 1.2],
            [0.4, inset + 1.2],
        ],
        dtype=float,
    )


def compute_octagon_mask(rows: int, cols: int) -> np.ndarray:
    """Return a boolean mask where True means the cell is inside the arena."""
    octagon_path = Path(build_octagon_vertices(rows, cols))
    mask = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            center = (float(col) + 0.5, float(row) + 0.5)
            mask[row, col] = bool(octagon_path.contains_point(center))
    return mask


def is_inside_arena(mask: np.ndarray, x: int, y: int) -> bool:
    """
    Check whether a grid cell is inside the arena.

    Args:
        mask: Boolean arena mask
        x: Row index
        y: Column index
    """
    return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and bool(mask[x, y])
