"""Tile-based renderer for the disaster environment.

Draws the village as pixel-art tiles (grass, cobbled road, plank floors, brick walls, water,
trees, rubble) with responders and victims on top. Purely for communication - demos, talks,
interviews, figures. It has no bearing on the science, but it makes the environment legible.

  render_rgb(env)             -> H x W x 3 uint8 array
  save_frame(env, path)       -> PNG (with a HUD strip)
  save_episode_gif(paths, out)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.disaster_terrain import GRASS, ROAD, FLOOR, WALL, RUBBLE, WATER, TREE, DOOR

TILE = 14  # pixels per grid cell

# base palette (RGB)
PALETTE: Dict[int, Tuple[int, int, int]] = {
    GRASS:  (106, 153, 78),
    ROAD:   (150, 143, 132),
    FLOOR:  (196, 164, 121),
    WALL:   (110, 98, 90),
    RUBBLE: (128, 106, 84),
    WATER:  (70, 130, 180),
    TREE:   (46, 92, 54),
    DOOR:   (156, 104, 52),
}

AGENCY_RGB = [
    (44, 123, 186), (44, 162, 95), (136, 86, 167),
    (230, 129, 42), (217, 79, 112), (23, 162, 184),
]
MINOR_RGB = (242, 193, 78)
SEVERE_RGB = (224, 82, 82)


def _tile_texture(code: int, r: int, c: int) -> np.ndarray:
    """A TILE x TILE patch for one cell. Deterministic per-cell jitter gives texture without
    randomness changing between frames."""
    base = np.array(PALETTE.get(int(code), (128, 128, 128)), dtype=np.int16)
    t = np.tile(base, (TILE, TILE, 1))

    rng = np.random.default_rng((int(r) * 73856093) ^ (int(c) * 19349663) ^ int(code))
    noise = rng.integers(-10, 11, size=(TILE, TILE, 1))
    t = t + noise

    if code == WALL:                     # brick courses
        t[0, :, :] -= 22
        row = (int(r) % 2) * (TILE // 2)
        t[:, row:row + 1, :] -= 18
    elif code == FLOOR:                  # plank lines
        t[::4, :, :] -= 16
    elif code == ROAD:                   # cobble speckle
        spots = rng.random((TILE, TILE)) < 0.18
        t[spots] -= 18
    elif code == WATER:                  # gentle ripples
        t[::3, :, :] += 14
    elif code == TREE:                   # canopy blob on grass
        t = np.tile(np.array(PALETTE[GRASS], dtype=np.int16), (TILE, TILE, 1))
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        d = (yy - TILE / 2) ** 2 + (xx - TILE / 2) ** 2
        canopy = d <= (TILE * 0.44) ** 2
        t[canopy] = np.array(PALETTE[TREE], dtype=np.int16) + rng.integers(-12, 13)
    elif code == DOOR:                   # door panel with frame
        t[1:-1, 2:-2, :] = np.array((120, 78, 38), dtype=np.int16)
    elif code == RUBBLE:                 # chunky debris
        chunks = rng.random((TILE, TILE)) < 0.45
        t[chunks] -= 26

    return np.clip(t, 0, 255).astype(np.uint8)


def _disk(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - (size - 1) / 2) ** 2 + (xx - (size - 1) / 2) ** 2) <= ((size - 1) / 2) ** 2


def render_rgb(env) -> np.ndarray:
    g = env.grid_size
    img = np.zeros((g * TILE, g * TILE, 3), dtype=np.uint8)

    terrain = env.terrain
    for r in range(g):
        for c in range(g):
            img[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = _tile_texture(terrain[r, c], r, c)

    def stamp(r: int, c: int, colour, frac: float, ring=(255, 255, 255)):
        """Filled disk with a light ring, so markers read clearly against textured ground."""
        size = max(4, int(TILE * frac))
        off = (TILE - size) // 2
        y0, x0 = r * TILE + off, c * TILE + off
        patch = img[y0:y0 + size, x0:x0 + size]
        if patch.shape[:2] != (size, size):
            return
        outer = _disk(size)
        patch[outer] = np.array(ring, dtype=np.uint8)          # ring
        inner_size = size - 2
        if inner_size >= 2:
            inner = _disk(inner_size)
            sub = patch[1:1 + inner_size, 1:1 + inner_size]
            sub[inner] = np.array(colour, dtype=np.uint8)      # fill
        else:
            patch[outer] = np.array(colour, dtype=np.uint8)

    # victims: fade as time runs out
    for v in env.victims:
        urgency = 1.0 - (v.ttl / max(1, v.max_ttl))
        base = np.array(SEVERE_RGB if v.severity == 2 else MINOR_RGB, dtype=float)
        colour = tuple(int(x) for x in (base * (1.0 - 0.45 * urgency) + 40 * urgency))
        stamp(v.row, v.col, colour, 0.90 if v.severity == 2 else 0.70)

    # responders, coloured by agency
    for a in env.agents:
        r, c = env.agent_positions[a]
        stamp(r, c, AGENCY_RGB[env.agency_of[a] % len(AGENCY_RGB)], 0.95)

    return img


def save_frame(env, path: str, title: Optional[str] = None, hud: bool = True) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = render_rgb(env)
    m = env.get_metrics()
    fig, ax = plt.subplots(figsize=(8, 8.4), dpi=110)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if hud:
        head = title or (
            f"step {m['steps']}    saved {m['lives_saved']}    lost {m['lives_lost']}    "
            f"remaining {m['victims_remaining']}    joint rescues {m['joint_rescues']}"
        )
        ax.set_title(head, fontsize=11, family="monospace")
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=np.array(SEVERE_RGB) / 255,
                       label="severe victim (needs 2)"),
            plt.Line2D([], [], marker="o", ls="", color=np.array(MINOR_RGB) / 255,
                       label="minor victim"),
        ] + [
            plt.Line2D([], [], marker="o", ls="", color=np.array(AGENCY_RGB[i % len(AGENCY_RGB)]) / 255,
                       label=f"agency {i}")
            for i in range(env.num_agencies)
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=min(4, len(handles)), fontsize=8, frameon=False)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def save_episode_gif(frame_paths: List[str], out_path: str, duration_ms: int = 120) -> Optional[str]:
    """Assemble PNG frames into an animated GIF using PIL (no extra dependency)."""
    try:
        from PIL import Image
    except Exception:
        return None
    if not frame_paths:
        return None
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return str(out)
