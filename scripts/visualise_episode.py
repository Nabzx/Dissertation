from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, ListedColormap


def animate_episode(
    grid_sequence: List[np.ndarray],
    save_path: Optional[str] = None,
    interval: int = 200,
    block: bool = True,
):
    # --- sanity check ---
    if not grid_sequence:
        raise ValueError("grid_sequence must contain at least one grid state.")

    # --- determine max value in grid (used for colouring) ---
    max_value = int(max(np.max(frame) for frame in grid_sequence))

    # base colours: empty, resource, agents, obstacles, etc.
    base_colours = ["white", "green", "blue", "red", "orange", "purple", "dimgray"]

    # extend colour list if more values exist
    if max_value + 1 > len(base_colours):
        base_colours.extend(
            ["cyan", "magenta", "yellow", "brown"][: max_value + 1 - len(base_colours)]
        )

    # build colour map
    cmap = ListedColormap(base_colours[: max_value + 1])
    norm = BoundaryNorm(np.arange(-0.5, max_value + 1.5, 1.0), cmap.N)

    # --- setup plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # initial frame
    image = ax.imshow(
        grid_sequence[0],
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    title = ax.set_title("Timestep: 0")

    # grid lines for readability
    rows, cols = grid_sequence[0].shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # --- update function (called each frame) ---
    def update(frame_idx: int):
        image.set_data(grid_sequence[frame_idx])   # update grid
        title.set_text(f"Timestep: {frame_idx}")   # update title
        return image, title

    # --- create animation ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(grid_sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )

    # --- optionally save as GIF ---
    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        ani.save(save_path, writer="pillow")

    # --- display animation ---
    plt.show(block=block)

    # non-blocking mode → auto-close after duration
    if not block:
        duration_seconds = max(len(grid_sequence) * interval / 1000.0, 0.1)
        plt.pause(duration_seconds)
        plt.close(fig)

    return ani