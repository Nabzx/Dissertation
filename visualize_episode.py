"""
Utilities for animating a single gridworld episode.
"""

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
    """
    Animate a sequence of grid states for one episode.

    Args:
        grid_sequence: List of 2D numpy arrays representing grid states over time
        save_path: Optional output path for a GIF
        interval: Delay between frames in milliseconds
        block: Whether to block execution while the animation window is shown

    Returns:
        The matplotlib animation object
    """
    if not grid_sequence:
        raise ValueError("grid_sequence must contain at least one grid state.")

    cmap = ListedColormap(["white", "green", "blue", "red"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(grid_sequence[0], cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    title = ax.set_title("Timestep: 0")

    rows, cols = grid_sequence[0].shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int):
        image.set_data(grid_sequence[frame_idx])
        title.set_text(f"Timestep: {frame_idx}")
        return image, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(grid_sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        ani.save(save_path, writer="pillow")

    plt.show(block=block)
    if not block:
        duration_seconds = max(len(grid_sequence) * interval / 1000.0, 0.1)
        plt.pause(duration_seconds)
        plt.close(fig)

    return ani
