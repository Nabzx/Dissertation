"""Renderer for the disaster response environment.

Purely for communication - demos, talks, interviews, figures. It has no bearing on the
science, but it makes the environment legible at a glance, which the raw arrays do not.

  render_frame(env)          -> RGB array
  save_frame(env, path)      -> PNG
  save_episode_gif(frames, path)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# palette
C_STREET = "#e8e6e1"
C_FLOOR = "#f7efe0"
C_WALL = "#5b5750"
C_RUBBLE = "#8a6a4f"
C_DOOR = "#c9a227"
C_MINOR = "#f2c14e"
C_SEVERE = "#e05252"
AGENCY_COLOURS = ["#2b7bba", "#2ca25f", "#8856a7", "#e6812a", "#d94f70", "#17a2b8"]


def render_frame(env, title: Optional[str] = None, figsize: float = 7.0):
    g = env.grid_size
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=110)

    # base: streets everywhere, then building interiors
    ax.add_patch(Rectangle((-0.5, -0.5), g, g, facecolor=C_STREET, edgecolor="none"))
    for (r, c) in getattr(env, "interior", set()):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=C_FLOOR, edgecolor="none"))
    for (r, c) in env.walls:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=C_WALL, edgecolor="none"))
    for (r, c) in getattr(env, "doors", set()):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=C_DOOR, edgecolor="none"))
    for (r, c) in env.rubble:
        ax.add_patch(Rectangle((c - 0.35, r - 0.35), 0.7, 0.7, facecolor=C_RUBBLE,
                               edgecolor="none", alpha=0.9))

    # victims: colour by severity, opacity by urgency (fading = running out of time)
    for v in env.victims:
        urgency = 1.0 - (v.ttl / max(1, v.max_ttl))
        colour = C_SEVERE if v.severity == 2 else C_MINOR
        size = 0.62 if v.severity == 2 else 0.46
        ax.add_patch(plt.Circle((v.col, v.row), size / 2, facecolor=colour,
                                edgecolor="#7a2020" if v.severity == 2 else "#8a6a1e",
                                linewidth=0.8, alpha=0.35 + 0.65 * (1.0 - urgency)))

    # responders: colour by agency
    for a in env.agents:
        r, c = env.agent_positions[a]
        colour = AGENCY_COLOURS[env.agency_of[a] % len(AGENCY_COLOURS)]
        ax.add_patch(plt.Circle((c, r), 0.34, facecolor=colour, edgecolor="white", linewidth=1.1))

    m = env.get_metrics()
    header = title or (
        f"step {m['steps']}   saved {m['lives_saved']}   lost {m['lives_lost']}   "
        f"remaining {m['victims_remaining']}   joint rescues {m['joint_rescues']}"
    )
    ax.set_title(header, fontsize=10)
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(g - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_SEVERE, label="severe victim (needs 2)"),
        plt.Line2D([], [], marker="o", ls="", color=C_MINOR, label="minor victim"),
    ] + [
        plt.Line2D([], [], marker="o", ls="", color=AGENCY_COLOURS[i % len(AGENCY_COLOURS)],
                   label=f"agency {i}")
        for i in range(env.num_agencies)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=min(4, len(handles)), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def save_frame(env, path: str, title: Optional[str] = None) -> str:
    fig = render_frame(env, title=title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def save_episode_gif(env_frames: List[np.ndarray], path: str, fps: int = 8) -> Optional[str]:
    try:
        import imageio.v2 as imageio
    except Exception:
        return None  # optional dependency; PNG frames are the fallback
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(p, env_frames, fps=fps)
    return str(p)
