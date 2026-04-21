from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def build_octagon_vertices(rows: int, cols: int) -> np.ndarray:
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
    octagon_path = Path(build_octagon_vertices(rows, cols))
    mask = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            centre = (float(col) + 0.5, float(row) + 0.5)
            mask[row, col] = bool(octagon_path.contains_point(centre))
    return mask


def is_inside_arena(mask: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and bool(mask[x, y])
