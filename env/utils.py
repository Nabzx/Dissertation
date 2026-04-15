"""
Utility functions for visualisation and logging.

This module provides helper functions for saving grid screenshots,
heatmaps, and resource distribution plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
import os


def save_grid_screenshot(grid: np.ndarray, filename: str, title: Optional[str] = None):
    """
    Save a visual representation of the grid state.

    Args:
        grid: 15x15 grid array (0=empty, 1=resource, 2=agent_0, 3=agent_1, 4=obstacle)
        filename: Output file path
        title: Optional title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create color map
    # 0 = empty (white), 1 = resource (green), 2 = agent_0 (blue), 3 = agent_1 (red)
    color_map = np.zeros((grid.shape[0], grid.shape[1], 3))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:  # Empty
                color_map[i, j] = [1.0, 1.0, 1.0]  # White
            elif grid[i, j] == 1:  # Resource
                color_map[i, j] = [0.0, 0.8, 0.0]  # Green
            elif grid[i, j] == 2:  # Agent 0
                color_map[i, j] = [0.0, 0.0, 1.0]  # Blue
            elif grid[i, j] == 3:  # Agent 1
                color_map[i, j] = [1.0, 0.0, 0.0]  # Red
            elif grid[i, j] == 4:  # Obstacle
                color_map[i, j] = [0.3, 0.3, 0.3]  # Dark grey

    ax.imshow(color_map, origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="Empty"),
        Patch(facecolor="green", label="Resource"),
        Patch(facecolor="blue", label="Agent 0"),
        Patch(facecolor="red", label="Agent 1"),
        Patch(facecolor="darkgray", label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_heatmap(heatmap: np.ndarray, filename: str, agent_name: str, title: Optional[str] = None):
    """
    Save a heatmap visualisation of agent movement.

    Args:
        heatmap: 2D array of visit counts
        filename: Output file path
        agent_name: Name of the agent (for title/legend)
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create heatmap
    im = ax.imshow(heatmap, cmap="YlOrRd", origin="upper", interpolation="nearest")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Visit Count", rotation=270, labelpad=20)

    # Add grid lines
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

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_resource_distribution(resources: List[Tuple[int, int]], filename: str, grid_size: int = 15):
    """
    Plot the initial distribution of resources.

    Args:
        resources: List of (row, col) tuples for resource positions
        filename: Output file path
        grid_size: Size of the grid
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create empty grid
    grid = np.zeros((grid_size, grid_size))

    # Mark resource positions
    for row, col in resources:
        grid[row, col] = 1

    # Plot resources
    im = ax.imshow(grid, cmap="Greens", origin="upper", interpolation="nearest", vmin=0, vmax=1)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"Resource Distribution ({len(resources)} resources)", fontsize=14, fontweight="bold")

    # Add text annotations for resource count
    for row, col in resources:
        ax.text(col, row, "R", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_movement_heatmap(heatmap: np.ndarray, filename: str, cmap: str = "hot"):
    """
    Save a movement-density heatmap for an agent (aggregated across episodes).

    Args:
        heatmap: 2D array of aggregated visit counts (15x15)
        filename: Output file path
        cmap: Colormap to use (default: "hot")
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create heatmap with specified colormap
    im = ax.imshow(heatmap, cmap=cmap, origin="upper", interpolation="nearest")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Total Visit Count", rotation=270, labelpad=20)

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, heatmap.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Aggregated Movement Heatmap (All Episodes)", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_trajectory_plot(trajectories: Dict[str, List[Tuple[int, int]]], grid_size: int, filename: str):
    """
    Plot agent trajectories with arrows indicating direction.

    Args:
        trajectories: Dict mapping agent_id to list of (row, col) positions
        grid_size: Size of the grid
        filename: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color mapping for agents
    agent_colors = {
        "agent_0": "blue",
        "agent_1": "red",
    }

    # Plot each agent's trajectory
    for agent_id, positions in trajectories.items():
        if len(positions) == 0:
            continue

        color = agent_colors.get(agent_id, "gray")

        # Extract x (col) and y (row) coordinates
        # Note: matplotlib uses (x, y) where x is column and y is row
        x_coords = [pos[1] for pos in positions]  # columns
        y_coords = [pos[0] for pos in positions]  # rows

        # Plot trajectory line
        ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=2, label=f"{agent_id}", zorder=1)

        # Add arrows to show direction (every few steps to avoid clutter)
        arrow_step = max(1, len(positions) // 20)  # Show ~20 arrows max
        for i in range(0, len(positions) - 1, arrow_step):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
            if dx != 0 or dy != 0:  # Only draw if there's movement
                ax.arrow(
                    x_coords[i],
                    y_coords[i],
                    dx * 0.7,
                    dy * 0.7,
                    head_width=0.3,
                    head_length=0.3,
                    fc=color,
                    ec=color,
                    alpha=0.7,
                    zorder=2,
                )

        # Mark starting position
        if len(positions) > 0:
            ax.scatter(
                x_coords[0],
                y_coords[0],
                color=color,
                s=200,
                marker="o",
                edgecolors="black",
                linewidths=2,
                label=f"{agent_id} Start",
                zorder=3,
            )

        # Mark ending position
        if len(positions) > 1:
            ax.scatter(
                x_coords[-1],
                y_coords[-1],
                color=color,
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label=f"{agent_id} End",
                zorder=3,
            )

    # Set up grid
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Invert y-axis to match grid coordinates (row 0 at top)

    # Add grid lines
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

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_reward_curve(reward_history: Dict[str, List[float]], filename: str):
    """
    Plot reward curves showing episode rewards over time for each agent.

    Args:
        reward_history: Dict mapping agent_id to list of episode total rewards
        filename: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color mapping for agents
    agent_colors = {
        "agent_0": "blue",
        "agent_1": "red",
    }

    # Plot each agent's reward curve
    for agent_id, rewards in reward_history.items():
        if len(rewards) == 0:
            continue

        color = agent_colors.get(agent_id, "gray")
        episodes = list(range(1, len(rewards) + 1))

        ax.plot(episodes, rewards, color=color, marker="o", linewidth=2, markersize=4, label=agent_id, alpha=0.8)

    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    ax.set_title("Episode Reward Curve", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
