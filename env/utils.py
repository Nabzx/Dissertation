import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def save_grid_screenshot(grid: np.ndarray, filename: str, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # colour mapping for agents (cycled if more agents)
    agent_palette = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.6, 0.0],
        [0.5, 0.0, 0.9],
    ]

    obstacle_value = int(np.max(grid))  # assume max value = obstacle
    colour_grid = np.zeros((grid.shape[0], grid.shape[1], 3))

    # convert grid values to RGB colours
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell_value = int(grid[row, col])

            if cell_value == 0:
                colour_grid[row, col] = [1.0, 1.0, 1.0]  # empty
            elif cell_value == 1:
                colour_grid[row, col] = [0.0, 0.8, 0.0]  # resource
            elif cell_value == obstacle_value and obstacle_value > 3:
                colour_grid[row, col] = [0.3, 0.3, 0.3]  # obstacle
            elif cell_value >= 2:
                # map agent values to palette
                colour_grid[row, col] = agent_palette[(cell_value - 2) % len(agent_palette)]

    ax.imshow(colour_grid, origin="upper", interpolation="nearest")

    # draw grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # legend for different objects
    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="Empty"),
        Patch(facecolor="green", label="Resource"),
        Patch(facecolor="blue", label="Agent 0"),
        Patch(facecolor="red", label="Agent 1"),
        Patch(facecolor="orange", label="Agent 2"),
        Patch(facecolor="purple", label="Agent 3"),
        Patch(facecolor="darkgray", label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # ensure folder exists
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_heatmap(heatmap: np.ndarray, filename: str, agent_name: str, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # show heatmap with colour scale
    im = ax.imshow(heatmap, cmap="YlOrRd", origin="upper", interpolation="nearest")

    # colour bar (how many visits)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Visit Count", rotation=270, labelpad=20)

    # grid lines
    ax.set_xticks(np.arange(-0.5, heatmap.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Movement Heatmap: {agent_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_resource_distribution(resources: List[Tuple[int, int]], filename: str, grid_size: int = 15):
    fig, ax = plt.subplots(figsize=(10, 10))

    grid = np.zeros((grid_size, grid_size))

    # mark resource positions
    for row, col in resources:
        grid[row, col] = 1

    im = ax.imshow(grid, cmap="Greens", origin="upper", interpolation="nearest", vmin=0, vmax=1)

    # grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"Resource Distribution ({len(resources)} resources)", fontsize=14, fontweight="bold")

    # label each resource cell
    for row, col in resources:
        ax.text(col, row, "R", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_movement_heatmap(heatmap: np.ndarray, filename: str, cmap: str = "hot"):
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(heatmap, cmap=cmap, origin="upper", interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Total Visit Count", rotation=270, labelpad=20)

    # grid overlay
    ax.set_xticks(np.arange(-0.5, heatmap.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Aggregated Movement Heatmap (All Episodes)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_trajectory_plot(trajectories: Dict[str, List[Tuple[int, int]]], grid_size: int, filename: str):
    fig, ax = plt.subplots(figsize=(10, 10))

    # assign colours to agents
    palette = ["blue", "red", "orange", "purple", "green", "brown"]
    agent_colours = {
        agent_id: palette[idx % len(palette)]
        for idx, agent_id in enumerate(sorted(trajectories.keys()))
    }

    for agent_id, positions in trajectories.items():
        if len(positions) == 0:
            continue

        colour = agent_colours.get(agent_id, "gray")

        # separate x/y coords
        x_coords = [pos[1] for pos in positions]
        y_coords = [pos[0] for pos in positions]

        # draw path line
        ax.plot(x_coords, y_coords, color=colour, alpha=0.6, linewidth=2, label=f"{agent_id}", zorder=1)

        # draw arrows to show direction
        arrow_step = max(1, len(positions) // 20)
        for i in range(0, len(positions) - 1, arrow_step):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]

            if dx != 0 or dy != 0:
                ax.arrow(
                    x_coords[i],
                    y_coords[i],
                    dx * 0.7,
                    dy * 0.7,
                    head_width=0.3,
                    head_length=0.3,
                    fc=colour,
                    ec=colour,
                    alpha=0.7,
                    zorder=2,
                )

        # mark start position
        ax.scatter(
            x_coords[0],
            y_coords[0],
            color=colour,
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2,
            label=f"{agent_id} Start",
            zorder=3,
        )

        # mark end position
        if len(positions) > 1:
            ax.scatter(
                x_coords[-1],
                y_coords[-1],
                color=colour,
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label=f"{agent_id} End",
                zorder=3,
            )

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match grid orientation

    # grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.set_title("Agent Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_reward_curve(reward_history: Dict[str, List[float]], filename: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    # simple colour mapping for agents
    agent_colours = {
        "agent_0": "blue",
        "agent_1": "red",
    }

    for agent_id, rewards in reward_history.items():
        if len(rewards) == 0:
            continue

        colour = agent_colours.get(agent_id, "gray")
        episodes = list(range(1, len(rewards) + 1))

        # plot reward per episode
        ax.plot(
            episodes,
            rewards,
            color=colour,
            marker="o",
            linewidth=2,
            markersize=4,
            label=agent_id,
            alpha=0.8,
        )

    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_title("Episode Reward Curve", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()