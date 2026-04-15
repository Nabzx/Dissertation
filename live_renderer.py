"""
Persistent real-time renderer for continuous PPO training episodes.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.path import Path

from env.gridworld_env import GridWorldEnv
from testing.communication import CommunicationLayer
from testing.ppo_agent import PPOAgent
from testing.rewards import apply_reward_scheme


class LiveEpisodeRenderer:
    """
    Keeps a single matplotlib window alive while multiple episodes run.
    """

    def __init__(
        self,
        initial_grid: np.ndarray,
        num_episodes: int,
        max_possible_reward: float,
        max_resources: int,
        view_size: int,
        show_perception: bool = True,
        show_communication: bool = True,
        show_resource_animation: bool = True,
    ):
        self.smoothing_window = 50
        self.plot_update_every = 10
        self.plot_downsample = 5
        self.max_possible_reward = max_possible_reward
        self.max_resources = max_resources
        self.view_size = view_size
        self.show_perception = show_perception
        self.show_communication = show_communication
        self.show_resource_animation = show_resource_animation
        self.trail_length = 20
        self.perception_range = 3
        self.communication_fade_steps = 4
        self.resource_anim_steps = 5
        self.agent_facing = {
            2: (0, 1),   # right
            3: (0, -1),  # left
        }
        self.arena_palette = {
            "void": "#050608",
            "panel": "#0c0f14",
            "floor_a": "#232b33",
            "floor_b": "#1b2229",
            "floor_outside": "#07090d",
            "tile_edge": "#2a323a",
            "wall": "#9ca3af",
            "wall_glow": "#c7a76a",
            "agent_0": "#38bdf8",
            "agent_0_edge": "#0ea5e9",
            "agent_1": "#f43f5e",
            "agent_1_edge": "#be123c",
            "resource": "#8bff63",
            "resource_edge": "#3b7f2d",
            "obstacle": "#4b5563",
            "obstacle_edge": "#6b7280",
            "trail_0": "#7dd3fc",
            "trail_1": "#fda4af",
            "perception_0": "#67e8f9",
            "perception_1": "#fb7185",
            "overlay_text": "#f3f4f6",
            "overlay_box": "#0f172a",
            "comm_0": "#67e8f9",
            "comm_1": "#fb7185",
            "comm_0_alt": "#22d3ee",
            "comm_1_alt": "#fb923c",
            "resource_glow": "#bef264",
            "resource_flash": "#fde68a",
        }
        self.agent_trails = {
            2: deque(maxlen=self.trail_length),
            3: deque(maxlen=self.trail_length),
        }
        self.communication_state = {
            2: {"ttl": 0, "receiver": 3, "preview": ""},
            3: {"ttl": 0, "receiver": 2, "preview": ""},
        }
        self.previous_resource_positions: set[tuple[int, int]] = set()
        self.resource_spawn_state: Dict[tuple[int, int], int] = {}
        self.resource_collect_state: Dict[tuple[int, int], int] = {}

        plt.ion()
        self.fig = plt.figure(figsize=(13.5, 6.4), facecolor=self.arena_palette["panel"])
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1.15, 0.48, 1.0], wspace=0.28, hspace=0.35)
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.ax_hud = self.fig.add_subplot(gs[:, 1])
        self.ax_plot = self.fig.add_subplot(gs[0, 2])
        self.ax_ppo = self.fig.add_subplot(gs[1, 2])

        rows, cols = initial_grid.shape
        self.grid_rows = rows
        self.grid_cols = cols
        self.octagon_vertices = self._build_octagon_vertices(rows, cols)
        self.octagon_path = Path(self.octagon_vertices)
        self.arena_mask = self._compute_octagon_mask(rows, cols)
        self.ax_grid.set_title("Environment")
        self.ax_grid.set_facecolor(self.arena_palette["void"])
        self.ax_grid.set_xlim(0, cols)
        self.ax_grid.set_ylim(rows, 0)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        for spine in self.ax_grid.spines.values():
            spine.set_visible(False)

        self._setup_hud_axis()

        # Draw a dark arena floor once; outside the octagon is treated as void.
        for row in range(rows):
            for col in range(cols):
                inside_arena = self.arena_mask[row, col]
                base_color = (
                    self.arena_palette["floor_a"]
                    if (row + col) % 2 == 0
                    else self.arena_palette["floor_b"]
                )
                tile = Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=base_color if inside_arena else self.arena_palette["floor_outside"],
                    edgecolor=self.arena_palette["tile_edge"] if inside_arena else self.arena_palette["void"],
                    linewidth=0.35 if inside_arena else 0.0,
                    zorder=0,
                )
                self.ax_grid.add_patch(tile)

        # Add a subtle vignette-like dark panel over the whole floor for a more cinematic feel.
        vignette = np.linspace(0.0, 1.0, 300)
        vignette = np.outer(
            np.minimum(vignette, vignette[::-1]),
            np.minimum(vignette, vignette[::-1]),
        )
        vignette_alpha = 0.32 * (1.0 - (vignette / np.max(vignette)))
        self.ax_grid.imshow(
            np.zeros_like(vignette),
            extent=(0, cols, rows, 0),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=vignette_alpha,
            zorder=0.8,
            interpolation="bilinear",
        )

        self.arena_wall_glow = Polygon(
            self.octagon_vertices,
            closed=True,
            fill=False,
            edgecolor=self.arena_palette["wall_glow"],
            linewidth=8.0,
            alpha=0.12,
            joinstyle="round",
            zorder=1.7,
        )
        self.ax_grid.add_patch(self.arena_wall_glow)
        self.arena_wall = Polygon(
            self.octagon_vertices,
            closed=True,
            fill=False,
            edgecolor=self.arena_palette["wall"],
            linewidth=2.2,
            alpha=0.95,
            joinstyle="round",
            zorder=1.8,
        )
        self.ax_grid.add_patch(self.arena_wall)

        self.perception_cell_patches = {
            2: [],
            3: [],
        }
        perception_fill = {
            2: self.arena_palette["perception_0"],
            3: self.arena_palette["perception_1"],
        }
        for agent_value, color in perception_fill.items():
            for _ in range(6):
                patch = Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.14,
                    zorder=2.45,
                    visible=False,
                )
                self.perception_cell_patches[agent_value].append(patch)
                self.ax_grid.add_patch(patch)

        self.perception_ray_lines = {
            2: [],
            3: [],
        }
        ray_colors = {
            2: self.arena_palette["perception_0"],
            3: self.arena_palette["perception_1"],
        }
        for agent_value, color in ray_colors.items():
            for _ in range(5):
                line = Line2D(
                    [],
                    [],
                    color=color,
                    linewidth=1.4,
                    alpha=0.45,
                    solid_capstyle="round",
                    zorder=3.2,
                    visible=False,
                )
                self.perception_ray_lines[agent_value].append(line)
                self.ax_grid.add_line(line)

        self.communication_lines = {}
        self.communication_pulses = {}
        self.communication_text = {}
        communication_colors = {
            2: (self.arena_palette["comm_0"], self.arena_palette["comm_0_alt"]),
            3: (self.arena_palette["comm_1"], self.arena_palette["comm_1_alt"]),
        }
        for agent_value, (primary_color, secondary_color) in communication_colors.items():
            line = Line2D(
                [],
                [],
                color=primary_color,
                linewidth=1.6,
                alpha=0.0,
                solid_capstyle="round",
                zorder=3.35,
                visible=False,
            )
            self.ax_grid.add_line(line)
            self.communication_lines[agent_value] = line

            pulse = Circle(
                (-10.0, -10.0),
                radius=0.42,
                fill=False,
                edgecolor=secondary_color,
                linewidth=1.6,
                alpha=0.0,
                zorder=4.6,
                visible=False,
            )
            self.ax_grid.add_patch(pulse)
            self.communication_pulses[agent_value] = pulse

            text = self.ax_grid.text(
                -10.0,
                -10.0,
                "",
                fontsize=7.5,
                color="#e5e7eb",
                ha="center",
                va="center",
                bbox=dict(facecolor="#111827", edgecolor=primary_color, boxstyle="round,pad=0.2", alpha=0.0),
                zorder=4.7,
                visible=False,
            )
            self.communication_text[agent_value] = text

        self.resource_patches: List[Circle] = []
        for _ in range(max_resources):
            resource_patch = Circle(
                (-10.0, -10.0),
                radius=0.18,
                facecolor=self.arena_palette["resource"],
                edgecolor=self.arena_palette["resource_edge"],
                linewidth=1.2,
                zorder=3,
                visible=False,
            )
            self.resource_patches.append(resource_patch)
            self.ax_grid.add_patch(resource_patch)

        self.resource_glow_patches: List[Circle] = []
        self.resource_collect_patches: List[Circle] = []
        for _ in range(max_resources):
            glow_patch = Circle(
                (-10.0, -10.0),
                radius=0.34,
                facecolor=self.arena_palette["resource_glow"],
                edgecolor="none",
                alpha=0.0,
                zorder=2.85,
                visible=False,
            )
            collect_patch = Circle(
                (-10.0, -10.0),
                radius=0.18,
                facecolor=self.arena_palette["resource_flash"],
                edgecolor="none",
                alpha=0.0,
                zorder=3.05,
                visible=False,
            )
            self.resource_glow_patches.append(glow_patch)
            self.resource_collect_patches.append(collect_patch)
            self.ax_grid.add_patch(glow_patch)
            self.ax_grid.add_patch(collect_patch)

        self.obstacle_patches: List[Rectangle] = []
        for row in range(rows):
            for col in range(cols):
                obstacle_patch = Rectangle(
                    (col + 0.12, row + 0.12),
                    0.76,
                    0.76,
                    facecolor=self.arena_palette["obstacle"],
                    edgecolor=self.arena_palette["obstacle_edge"],
                    linewidth=1.0,
                    zorder=2,
                    visible=False,
                )
                self.obstacle_patches.append(obstacle_patch)
                self.ax_grid.add_patch(obstacle_patch)

        self.agent_patches = {
            2: Circle(
                (0.5, 0.5),
                radius=0.28,
                facecolor=self.arena_palette["agent_0"],
                edgecolor=self.arena_palette["agent_0_edge"],
                linewidth=1.8,
                zorder=4,
                visible=False,
            ),
            3: Circle(
                (0.5, 0.5),
                radius=0.28,
                facecolor=self.arena_palette["agent_1"],
                edgecolor=self.arena_palette["agent_1_edge"],
                linewidth=1.8,
                zorder=4,
                visible=False,
            ),
        }
        for patch in self.agent_patches.values():
            self.ax_grid.add_patch(patch)

        trail_colors = {
            2: self.arena_palette["trail_0"],
            3: self.arena_palette["trail_1"],
        }
        self.trail_patches = {
            2: [],
            3: [],
        }
        for agent_value, color in trail_colors.items():
            for _ in range(self.trail_length):
                trail_patch = Circle(
                    (-10.0, -10.0),
                    radius=0.12,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.0,
                    zorder=3.6,
                    visible=False,
                )
                self.trail_patches[agent_value].append(trail_patch)
                self.ax_grid.add_patch(trail_patch)

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self.arena_palette["agent_0"], markeredgecolor=self.arena_palette["agent_0_edge"], markersize=9, label="Agent 0"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self.arena_palette["agent_1"], markeredgecolor=self.arena_palette["agent_1_edge"], markersize=9, label="Agent 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self.arena_palette["resource"], markeredgecolor=self.arena_palette["resource_edge"], markersize=7, label="Resource"),
            Rectangle((0, 0), 1, 1, facecolor=self.arena_palette["obstacle"], edgecolor=self.arena_palette["obstacle_edge"], label="Obstacle"),
        ]
        if show_perception:
            legend_handles.append(
                Rectangle((0, 0), 1, 1, facecolor=self.arena_palette["perception_0"], edgecolor=self.arena_palette["agent_0"], alpha=0.25, label="Perception")
            )
        if show_communication:
            legend_handles.append(
                Line2D([0], [0], color=self.arena_palette["comm_0"], linewidth=1.6, label="Signal")
            )
        if show_resource_animation:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="none", markerfacecolor=self.arena_palette["resource_flash"], markeredgecolor=self.arena_palette["resource_glow"], markersize=8, label="Spawn Pulse")
            )
        self.ax_grid.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            frameon=True,
            facecolor="#0b1016",
            edgecolor="#313842",
            labelcolor="#d1d5db",
            fontsize=9,
        )
        self.text = self.ax_grid.text(
            0.5,
            -0.08,
            "",
            transform=self.ax_grid.transAxes,
            ha="center",
            fontsize=12,
            color=self.arena_palette["overlay_text"],
            bbox=dict(facecolor=self.arena_palette["overlay_box"], edgecolor="#334155", boxstyle="round,pad=0.35", alpha=0.88),
        )

        self.episode_rewards: List[float] = []
        self.episode_resources: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_values: List[float] = []
        self.reward_line, = self.ax_plot.plot([], [], color="tab:blue", linewidth=2, label="Reward Avg (50)")
        self.resource_line, = self.ax_plot.plot(
            [],
            [],
            color="tab:green",
            linewidth=2,
            label="Resources Avg (50)",
        )
        self.ax_plot.set_xlim(0, max(1, num_episodes))
        self.ax_plot.set_ylim(0, max(1.0, max_possible_reward))
        self._style_ui_axis(self.ax_plot, "Learning Progress")
        self.ax_plot.set_xlabel("Episode")
        self.ax_plot.set_ylabel("Smoothed Value")
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.legend(loc="upper left")

        self.policy_line, = self.ax_ppo.plot(
            [],
            [],
            color="tab:red",
            linewidth=2,
            label="Policy Loss (PPO)",
        )
        self.ax_entropy = self.ax_ppo.twinx()
        self.entropy_line, = self.ax_entropy.plot(
            [],
            [],
            color="tab:purple",
            linewidth=2,
            label="Entropy (Exploration)",
        )
        self.ax_ppo.set_xlim(0, max(1, num_episodes))
        self._style_ui_axis(self.ax_ppo, "PPO Training Dynamics")
        self.ax_ppo.set_xlabel("Episode")
        self.ax_ppo.set_ylabel("Policy Loss (PPO)", color="tab:red")
        self.ax_ppo.tick_params(axis="y", labelcolor="tab:red")
        self.ax_ppo.grid(True, alpha=0.3)
        self.ax_entropy.set_ylabel("Entropy (Exploration)", color="tab:purple")
        self.ax_entropy.tick_params(axis="y", labelcolor="tab:purple")
        self.ax_ppo.set_ylim(-1.0, 2.0)
        self.ax_entropy.set_ylim(0.0, 2.0)
        ppo_handles = [self.policy_line, self.entropy_line]
        self.ax_ppo.legend(ppo_handles, [line.get_label() for line in ppo_handles], loc="upper right")

        self.fig.subplots_adjust(bottom=0.18)
        self._update_environment(initial_grid)

    def _style_ui_axis(self, axis, title: str) -> None:
        axis.set_facecolor("#11161d")
        axis.set_title(title, color="#e5e7eb")
        axis.tick_params(colors="#cbd5e1")
        axis.xaxis.label.set_color("#cbd5e1")
        axis.yaxis.label.set_color("#cbd5e1")
        for spine in axis.spines.values():
            spine.set_color("#374151")

    def _setup_hud_axis(self) -> None:
        self.ax_hud.set_facecolor("#0b1016")
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        self.ax_hud.set_xlim(0, 1)
        self.ax_hud.set_ylim(0, 1)
        for spine in self.ax_hud.spines.values():
            spine.set_visible(False)

        self.ax_hud.text(
            0.08,
            0.97,
            "Arena HUD",
            color="#f3f4f6",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        self.ax_hud.add_patch(
            Rectangle((0.06, 0.89), 0.88, 0.0018, facecolor="#334155", edgecolor="none", alpha=0.9)
        )

        self.hud_episode_text = self.ax_hud.text(
            0.08, 0.85, "", color="#cbd5e1", fontsize=10.5, va="top"
        )
        self.hud_reward_text = self.ax_hud.text(
            0.08, 0.79, "", color="#cbd5e1", fontsize=10, va="top"
        )

        card_specs = {
            "agent_0": {"y": 0.43, "accent": self.arena_palette["agent_0"], "edge": self.arena_palette["agent_0_edge"], "title": "Agent 0"},
            "agent_1": {"y": 0.07, "accent": self.arena_palette["agent_1"], "edge": self.arena_palette["agent_1_edge"], "title": "Agent 1"},
        }
        self.hud_agent_text = {}
        for agent_id, spec in card_specs.items():
            self.ax_hud.add_patch(
                Rectangle(
                    (0.06, spec["y"]),
                    0.88,
                    0.28,
                    facecolor="#111827",
                    edgecolor="#1f2937",
                    linewidth=1.1,
                    zorder=0.5,
                )
            )
            self.ax_hud.add_patch(
                Rectangle(
                    (0.06, spec["y"]),
                    0.018,
                    0.28,
                    facecolor=spec["accent"],
                    edgecolor="none",
                    zorder=0.6,
                )
            )
            self.ax_hud.text(
                0.1,
                spec["y"] + 0.245,
                spec["title"],
                color=spec["accent"],
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            self.hud_agent_text[agent_id] = self.ax_hud.text(
                0.1,
                spec["y"] + 0.205,
                "",
                color="#e5e7eb",
                fontsize=9.3,
                va="top",
                linespacing=1.55,
            )

    def update_hud(self, hud_state: Optional[Dict[str, object]]) -> None:
        if hud_state is None:
            self.hud_episode_text.set_text("")
            self.hud_reward_text.set_text("")
            for text in self.hud_agent_text.values():
                text.set_text("")
            return

        episode = hud_state.get("episode", 0)
        total_episodes = hud_state.get("total_episodes", 0)
        step = hud_state.get("step", 0)
        phase = hud_state.get("phase", "Training")
        episode_reward = float(hud_state.get("episode_reward", 0.0))
        self.hud_episode_text.set_text(
            f"{phase}\nEpisode {episode}/{total_episodes}\nStep {step}"
        )
        self.hud_reward_text.set_text(f"Episode Reward: {episode_reward:.2f}")

        agent_states = hud_state.get("agents", {})
        for agent_id in ["agent_0", "agent_1"]:
            info = agent_states.get(agent_id, {})
            resources = int(info.get("resources", 0))
            cumulative = int(info.get("cumulative_resources", 0))
            position = info.get("position", ("-", "-"))
            facing = info.get("facing", "unknown")
            comm_status = info.get("communication", "offline")
            status = info.get("status", "Active")
            action = info.get("recent_action", "n/a")
            self.hud_agent_text[agent_id].set_text(
                f"Resources: {resources} (run {cumulative})\n"
                f"Position: {position}\n"
                f"Facing: {facing}\n"
                f"Comms: {comm_status}\n"
                f"Status: {status}\n"
                f"Action: {action}"
            )

    def _build_octagon_vertices(self, rows: int, cols: int) -> np.ndarray:
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

    def _compute_octagon_mask(self, rows: int, cols: int) -> np.ndarray:
        mask = np.zeros((rows, cols), dtype=bool)
        for row in range(rows):
            for col in range(cols):
                center = (float(col) + 0.5, float(row) + 0.5)
                mask[row, col] = bool(self.octagon_path.contains_point(center))
        return mask

    def update(
        self,
        grid: np.ndarray,
        episode: int,
        num_episodes: int,
        step: int,
        render_delay: float,
        actions: Optional[Dict[str, int]] = None,
        communication_events: Optional[List[Dict[str, object]]] = None,
        hud_state: Optional[Dict[str, object]] = None,
    ) -> None:
        self._update_environment(grid, actions=actions, communication_events=communication_events)
        self.text.set_text(f"Episode {episode}/{num_episodes} | Step {step}")
        self.update_hud(hud_state)
        self.fig.canvas.draw_idle()
        plt.pause(render_delay)

    def _update_environment(
        self,
        grid: np.ndarray,
        actions: Optional[Dict[str, int]] = None,
        communication_events: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        self._update_facing(actions)
        self._update_perception(grid)
        self._update_resource_animation_state(grid)

        obstacle_mask = grid == 4
        for patch, is_obstacle in zip(self.obstacle_patches, obstacle_mask.flatten()):
            row = int(patch.get_y() - 0.12)
            col = int(patch.get_x() - 0.12)
            patch.set_visible(bool(is_obstacle) and self.arena_mask[row, col])

        resource_idx = 0
        for row, col in np.argwhere(grid == 1):
            if resource_idx >= len(self.resource_patches):
                break
            if not self.arena_mask[row, col]:
                continue
            patch = self.resource_patches[resource_idx]
            patch.center = (float(col) + 0.5, float(row) + 0.5)
            spawn_ttl = self.resource_spawn_state.get((row, col))
            if self.show_resource_animation and spawn_ttl is not None:
                progress = 1.0 - ((spawn_ttl - 1) / float(self.resource_anim_steps))
                patch.set_radius(0.08 + 0.10 * progress)
                patch.set_alpha(0.45 + 0.55 * progress)
            else:
                patch.set_radius(0.18)
                patch.set_alpha(1.0)
            patch.set_visible(True)
            resource_idx += 1

        for idx in range(resource_idx, len(self.resource_patches)):
            self.resource_patches[idx].set_visible(False)

        self._apply_resource_animations()

        current_positions: Dict[int, tuple[int, int]] = {}
        for agent_value, patch in self.agent_patches.items():
            positions = np.argwhere(grid == agent_value)
            if len(positions) == 0:
                patch.set_visible(False)
                self.agent_trails[agent_value].clear()
                self._update_trail_patches(agent_value)
                continue
            row, col = positions[0]
            if not self.arena_mask[row, col]:
                patch.set_visible(False)
                self.agent_trails[agent_value].clear()
                self._update_trail_patches(agent_value)
                continue
            self._append_trail_position(agent_value, int(row), int(col))
            patch.center = (float(col) + 0.5, float(row) + 0.5)
            patch.set_visible(True)
            current_positions[agent_value] = (int(row), int(col))

        self._update_communication_visuals(current_positions, communication_events)

    def _update_resource_animation_state(self, grid: np.ndarray) -> None:
        if not self.show_resource_animation:
            self.previous_resource_positions = set(map(tuple, np.argwhere(grid == 1)))
            self.resource_spawn_state.clear()
            self.resource_collect_state.clear()
            return

        current_resources = {tuple(pos) for pos in np.argwhere(grid == 1)}
        spawned = current_resources - self.previous_resource_positions
        collected = self.previous_resource_positions - current_resources

        for pos in spawned:
            self.resource_spawn_state[pos] = self.resource_anim_steps
        for pos in collected:
            self.resource_collect_state[pos] = self.resource_anim_steps

        self.previous_resource_positions = current_resources

    def _apply_resource_animations(self) -> None:
        for patch in self.resource_glow_patches:
            patch.set_visible(False)
        for patch in self.resource_collect_patches:
            patch.set_visible(False)

        if not self.show_resource_animation:
            return

        active_spawns = list(self.resource_spawn_state.items())[: len(self.resource_glow_patches)]
        for idx, (pos, ttl) in enumerate(active_spawns):
            row, col = pos
            if not self.arena_mask[row, col]:
                continue
            progress = 1.0 - ((ttl - 1) / float(self.resource_anim_steps))
            glow_patch = self.resource_glow_patches[idx]
            glow_patch.center = (float(col) + 0.5, float(row) + 0.5)
            glow_patch.set_radius(0.18 + 0.22 * progress)
            glow_patch.set_alpha(0.35 * (1.0 - progress / 1.2))
            glow_patch.set_visible(True)

        active_collects = list(self.resource_collect_state.items())[: len(self.resource_collect_patches)]
        for idx, (pos, ttl) in enumerate(active_collects):
            row, col = pos
            if not self.arena_mask[row, col]:
                continue
            progress = 1.0 - ((ttl - 1) / float(self.resource_anim_steps))
            collect_patch = self.resource_collect_patches[idx]
            collect_patch.center = (float(col) + 0.5, float(row) + 0.5)
            collect_patch.set_radius(0.16 + 0.18 * progress)
            collect_patch.set_alpha(0.42 * (1.0 - progress))
            collect_patch.set_visible(True)

        self.resource_spawn_state = {
            pos: ttl - 1 for pos, ttl in self.resource_spawn_state.items() if ttl > 1
        }
        self.resource_collect_state = {
            pos: ttl - 1 for pos, ttl in self.resource_collect_state.items() if ttl > 1
        }

    def _append_trail_position(self, agent_value: int, row: int, col: int) -> None:
        trail = self.agent_trails[agent_value]
        new_position = (row, col)
        if not trail or trail[-1] != new_position:
            trail.append(new_position)
        self._update_trail_patches(agent_value)

    def _update_trail_patches(self, agent_value: int) -> None:
        trail = list(self.agent_trails[agent_value])
        patches = self.trail_patches[agent_value]
        visible_trail = trail[:-1] if len(trail) > 1 else []

        for patch in patches:
            patch.set_visible(False)

        if not visible_trail:
            return

        for idx, (row, col) in enumerate(reversed(visible_trail)):
            if idx >= len(patches):
                break
            if not self.arena_mask[row, col]:
                continue
            fade = max(0.12, 0.55 * (1.0 - idx / max(len(visible_trail), 1)))
            patch = patches[idx]
            patch.center = (float(col) + 0.5, float(row) + 0.5)
            patch.set_alpha(fade)
            patch.set_visible(True)

    def _update_facing(self, actions: Optional[Dict[str, int]]) -> None:
        if not actions:
            return
        action_to_direction = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
        }
        for agent_id, action in actions.items():
            if action not in action_to_direction:
                continue
            agent_value = 2 if agent_id == "agent_0" else 3
            self.agent_facing[agent_value] = action_to_direction[action]

    def _update_perception(self, grid: np.ndarray) -> None:
        if not self.show_perception:
            return

        for agent_value in self.perception_ray_lines.keys():
            for patch in self.perception_cell_patches[agent_value]:
                patch.set_visible(False)
            for line in self.perception_ray_lines[agent_value]:
                line.set_visible(False)

        for agent_value in self.perception_ray_lines.keys():
            positions = np.argwhere(grid == agent_value)
            if len(positions) == 0:
                continue

            row, col = positions[0]
            facing = self.agent_facing[agent_value]
            ray_specs = self._compute_forward_rays(int(row), int(col), facing, grid)

            visible_cells = []
            for idx, ray in enumerate(ray_specs):
                if idx < len(self.perception_ray_lines[agent_value]) and len(ray["points"]) >= 2:
                    xs = [point[0] for point in ray["points"]]
                    ys = [point[1] for point in ray["points"]]
                    line = self.perception_ray_lines[agent_value][idx]
                    line.set_data(xs, ys)
                    line.set_visible(True)
                visible_cells.extend(ray["cells"])

            unique_cells = []
            seen = set()
            for cell in visible_cells:
                if cell in seen:
                    continue
                seen.add(cell)
                unique_cells.append(cell)

            for idx, (cell_row, cell_col) in enumerate(unique_cells[: len(self.perception_cell_patches[agent_value])]):
                patch = self.perception_cell_patches[agent_value][idx]
                patch.set_xy((cell_col, cell_row))
                patch.set_width(1)
                patch.set_height(1)
                patch.set_visible(True)

    def _compute_forward_rays(
        self,
        row: int,
        col: int,
        facing: tuple[int, int],
        grid: np.ndarray,
    ) -> List[Dict[str, List]]:
        dr, dc = facing
        if (dr, dc) in [(0, 1), (0, -1)]:
            fan_offsets = [-0.6, -0.3, 0.0, 0.3, 0.6]
            lateral = (1, 0)
        else:
            fan_offsets = [-0.6, -0.3, 0.0, 0.3, 0.6]
            lateral = (0, 1)

        origin_x = float(col) + 0.5
        origin_y = float(row) + 0.5
        rays = []

        for offset in fan_offsets:
            points = [(origin_x, origin_y)]
            cells = []
            blocked = False
            for step in range(1, self.perception_range + 1):
                target_row = row + dr * step + int(round(lateral[0] * offset * step))
                target_col = col + dc * step + int(round(lateral[1] * offset * step))

                if not (0 <= target_row < self.grid_rows and 0 <= target_col < self.grid_cols):
                    break
                if not self.arena_mask[target_row, target_col]:
                    break

                cell_center = (float(target_col) + 0.5, float(target_row) + 0.5)
                points.append(cell_center)
                cells.append((target_row, target_col))

                if grid[target_row, target_col] == 4:
                    blocked = True
                    break

            if blocked or len(points) > 1:
                rays.append({"points": points, "cells": cells})

        return rays

    def _update_communication_visuals(
        self,
        positions: Dict[int, tuple[int, int]],
        communication_events: Optional[List[Dict[str, object]]],
    ) -> None:
        if not self.show_communication:
            for agent_value in self.communication_lines.keys():
                self.communication_lines[agent_value].set_visible(False)
                self.communication_pulses[agent_value].set_visible(False)
                self.communication_text[agent_value].set_visible(False)
            return

        if communication_events:
            for event in communication_events:
                sender = int(event["sender"])
                receiver = int(event["receiver"])
                preview = str(event.get("preview", "msg"))
                self.communication_state[sender]["ttl"] = self.communication_fade_steps
                self.communication_state[sender]["receiver"] = receiver
                self.communication_state[sender]["preview"] = preview

        for sender, state in self.communication_state.items():
            line = self.communication_lines[sender]
            pulse = self.communication_pulses[sender]
            text = self.communication_text[sender]

            if state["ttl"] <= 0 or sender not in positions or state["receiver"] not in positions:
                line.set_visible(False)
                pulse.set_visible(False)
                text.set_visible(False)
                continue

            alpha = state["ttl"] / float(self.communication_fade_steps)
            sender_row, sender_col = positions[sender]
            receiver_row, receiver_col = positions[int(state["receiver"])]
            sx = float(sender_col) + 0.5
            sy = float(sender_row) + 0.5
            rx = float(receiver_col) + 0.5
            ry = float(receiver_row) + 0.5

            curve_x, curve_y = self._build_signal_curve((sx, sy), (rx, ry), sender)
            line.set_data(curve_x, curve_y)
            line.set_alpha(0.15 + 0.45 * alpha)
            line.set_visible(True)

            pulse.center = (sx, sy)
            pulse.set_radius(0.28 + 0.22 * alpha)
            pulse.set_alpha(0.15 + 0.5 * alpha)
            pulse.set_visible(True)

            text.set_position((sx, sy - 0.55))
            text.set_text(str(state["preview"]))
            text.get_bbox_patch().set_alpha(0.2 + 0.45 * alpha)
            text.set_alpha(0.55 + 0.35 * alpha)
            text.set_visible(True)

            state["ttl"] -= 1

    def _build_signal_curve(
        self,
        sender_pos: tuple[float, float],
        receiver_pos: tuple[float, float],
        sender: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sx, sy = sender_pos
        rx, ry = receiver_pos
        mid_x = (sx + rx) / 2.0
        mid_y = (sy + ry) / 2.0
        dx = rx - sx
        dy = ry - sy
        distance = max(np.hypot(dx, dy), 1e-6)
        perp_x = -dy / distance
        perp_y = dx / distance
        bend = 0.18 if sender == 2 else -0.18
        control_x = mid_x + perp_x * distance * bend
        control_y = mid_y + perp_y * distance * bend
        t = np.linspace(0.0, 1.0, 20)
        curve_x = (1 - t) ** 2 * sx + 2 * (1 - t) * t * control_x + t**2 * rx
        curve_y = (1 - t) ** 2 * sy + 2 * (1 - t) * t * control_y + t**2 * ry
        return curve_x, curve_y

    def update_learning_plot(self, total_reward: float, total_resources: float) -> None:
        self.episode_rewards.append(float(total_reward))
        self.episode_resources.append(float(total_resources))
        if len(self.episode_rewards) % self.plot_update_every != 0:
            return

        x_vals = list(range(len(self.episode_rewards)))
        reward_smoothed = self._moving_average(self.episode_rewards, self.smoothing_window)
        resource_smoothed = self._moving_average(self.episode_resources, self.smoothing_window)

        plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
        plot_reward = reward_smoothed[:: self.plot_downsample] or reward_smoothed[-1:]
        plot_resources = resource_smoothed[:: self.plot_downsample] or resource_smoothed[-1:]

        self.reward_line.set_data(plot_x, plot_reward)
        self.resource_line.set_data(plot_x, plot_resources)

    def update_ppo_plot(self, ppo_metrics: Dict[str, float]) -> None:
        self.policy_losses.append(float(ppo_metrics.get("policy_loss", 0.0)))
        self.value_losses.append(float(ppo_metrics.get("value_loss", 0.0)))
        self.entropy_values.append(float(ppo_metrics.get("entropy", 0.0)))
        if len(self.policy_losses) % self.plot_update_every != 0:
            return

        x_vals = list(range(len(self.policy_losses)))
        policy_smoothed = self._moving_average(self.policy_losses, self.smoothing_window)
        entropy_smoothed = self._moving_average(self.entropy_values, self.smoothing_window)

        plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
        plot_policy = policy_smoothed[:: self.plot_downsample] or policy_smoothed[-1:]
        plot_entropy = entropy_smoothed[:: self.plot_downsample] or entropy_smoothed[-1:]

        self.policy_line.set_data(plot_x, plot_policy)
        self.entropy_line.set_data(plot_x, plot_entropy)

    def _moving_average(self, values: List[float], window: int) -> List[float]:
        return [
            float(np.mean(values[max(0, idx - window + 1) : idx + 1]))
            for idx in range(len(values))
        ]

    def refresh(self, render_delay: float) -> None:
        self.fig.canvas.draw_idle()
        plt.pause(render_delay)

    def close(self) -> None:
        plt.ioff()
        plt.show()


def run_live_training(
    num_episodes: int = 1000,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 15,
    num_resources: int = 10,
    max_steps: int = 100,
    render_delay: float = 0.001,
    render_every: int = 50,
    fast_mode: bool = False,
    final_demo_episodes: int = 10,
    show_perception: bool = True,
    show_communication: bool = True,
) -> List[Dict]:
    """
    Run many PPO episodes with a persistent live renderer window.
    """
    env = GridWorldEnv(grid_size=grid_size, num_resources=num_resources, max_steps=max_steps)
    max_possible_reward = float(num_resources * len(env.agents))

    obs_shape = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_shape))
    if use_communication:
        obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)
    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device="cpu")

    raw_obs, _ = env.reset(seed=0)
    renderer = LiveEpisodeRenderer(
        env.grid.copy(),
        num_episodes=num_episodes,
        max_possible_reward=max_possible_reward,
        max_resources=num_resources,
        view_size=env.view_size,
        show_perception=show_perception,
        show_communication=show_communication,
    )

    episode_summaries: List[Dict] = []
    recent_rewards: List[float] = []
    recent_resources: List[int] = []
    cumulative_resources_run = {
        "agent_0": 0,
        "agent_1": 0,
    }
    action_labels = {
        0: "stay",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
    }

    def run_live_episode(
        episode_seed: int,
        render_episode: bool,
        episode_label: int,
        train_policy: bool = True,
    ) -> Dict:
        def facing_label(agent_id: str) -> str:
            agent_value = 2 if agent_id == "agent_0" else 3
            facing = renderer.agent_facing.get(agent_value, (0, 1))
            facing_map = {
                (-1, 0): "up",
                (1, 0): "down",
                (0, -1): "left",
                (0, 1): "right",
            }
            return facing_map.get(tuple(facing), "unknown")

        def build_hud_state(
            step_idx: int,
            actions_map: Optional[Dict[str, int]],
            communication_events: Optional[List[Dict[str, object]]],
            phase: str,
        ) -> Dict[str, object]:
            resources_now = env.get_resources_collected()
            comm_lookup = {
                "agent_0": "idle",
                "agent_1": "idle",
            }
            if communication_events:
                for event in communication_events:
                    sender_id = "agent_0" if int(event["sender"]) == 2 else "agent_1"
                    preview = str(event.get("preview", "msg"))
                    comm_lookup[sender_id] = f"sending {preview}"

            agents_state = {}
            for agent_id in env.agents:
                position = env.agent_positions.get(agent_id, ("-", "-"))
                agents_state[agent_id] = {
                    "resources": resources_now.get(agent_id, 0),
                    "cumulative_resources": cumulative_resources_run[agent_id] + resources_now.get(agent_id, 0),
                    "position": position,
                    "facing": facing_label(agent_id),
                    "communication": comm_lookup[agent_id] if use_communication else "disabled",
                    "status": "Active",
                    "recent_action": action_labels.get(actions_map.get(agent_id, 0), "n/a") if actions_map else "n/a",
                }

            return {
                "phase": phase,
                "episode": episode_label,
                "total_episodes": num_episodes,
                "step": step_idx,
                "episode_reward": total_shaped_reward,
                "agents": agents_state,
            }

        raw_obs, _ = env.reset(seed=episode_seed)

        comm_layer: Optional[CommunicationLayer] = None
        if use_communication:
            comm_layer = CommunicationLayer(env)
            comm_layer.reset()
            obs = comm_layer.build_augment_observation(raw_obs)
        else:
            obs = raw_obs

        if train_policy:
            ppo_agent.reset_buffer()
        cumulative_collected: Dict[str, int] = {agent: 0 for agent in env.agents}
        total_shaped_reward = 0.0

        if render_episode:
            renderer.update(
                env.grid.copy(),
                episode_label,
                num_episodes,
                0,
                render_delay,
                actions=None,
                communication_events=None,
                hud_state=build_hud_state(0, None, None, "Training" if train_policy else "Demo"),
            )

        for step in range(max_steps):
            actions: Dict[str, int] = {}
            step_log_probs: Dict[str, float] = {}
            step_values: Dict[str, float] = {}
            step_flat_obs: Dict[str, np.ndarray] = {}

            for agent_id in env.agents:
                agent_obs = obs[agent_id]
                flat_obs = agent_obs if agent_obs.ndim == 1 else agent_obs.flatten()
                action, log_prob, value = ppo_agent.select_action(flat_obs)
                actions[agent_id] = int(action)
                step_log_probs[agent_id] = float(log_prob)
                step_values[agent_id] = float(value)
                step_flat_obs[agent_id] = flat_obs.astype(np.float32)

            raw_next_obs, raw_rewards, terminations, truncations, _ = env.step(actions)

            for agent_id, reward in raw_rewards.items():
                if reward > 0.0:
                    cumulative_collected[agent_id] += 1

            shaped_rewards = apply_reward_scheme(
                scheme=reward_scheme,
                raw_rewards=raw_rewards,
                cumulative_collected=cumulative_collected,
                total_spawned=env.num_resources,
                alpha=0.5,
            )
            total_shaped_reward += float(sum(shaped_rewards.values()))

            done_flags = {
                agent_id: bool(terminations[agent_id] or truncations[agent_id])
                for agent_id in env.agents
            }

            if train_policy:
                for agent_id in env.agents:
                    ppo_agent.store_transition(
                        obs=step_flat_obs[agent_id],
                        action=actions[agent_id],
                        log_prob=step_log_probs[agent_id],
                        reward=float(shaped_rewards[agent_id]),
                        done=done_flags[agent_id],
                        value=step_values[agent_id],
                    )

            communication_events = None
            if comm_layer is not None:
                sent_messages = comm_layer.update_messages_after_step()
                obs = comm_layer.build_augment_observation(raw_next_obs)
                communication_events = []
                for sender_id, msg in sent_messages.items():
                    receiver_id = next(agent for agent in env.agents if agent != sender_id)
                    preview = f"{int(msg[1]):+d},{int(msg[2]):+d},{int(msg[3])}"
                    communication_events.append(
                        {
                            "sender": 2 if sender_id == "agent_0" else 3,
                            "receiver": 2 if receiver_id == "agent_0" else 3,
                            "preview": preview,
                        }
                    )
            else:
                obs = raw_next_obs

            if render_episode:
                renderer.update(
                    env.grid.copy(),
                    episode_label,
                    num_episodes,
                    step + 1,
                    render_delay,
                    actions=actions,
                    communication_events=communication_events,
                    hud_state=build_hud_state(
                        step + 1,
                        actions,
                        communication_events,
                        "Training" if train_policy else "Demo",
                    ),
                )

            if all(done_flags.values()):
                break

        ppo_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        if train_policy:
            try:
                ppo_metrics = ppo_agent.update(last_value=0.0, last_done=True)
            except RuntimeError as exc:
                print(f"[live] PPO update skipped: {exc}")

        resources = env.get_resources_collected()
        total_resources = int(sum(resources.values()))
        if train_policy:
            for agent_id, count in resources.items():
                cumulative_resources_run[agent_id] += count

        return {
            "resources": resources,
            "total_resources": total_resources,
            "total_reward": total_shaped_reward,
            "ppo_metrics": ppo_metrics,
            "steps": env.step_count,
        }

    for episode in range(num_episodes):
        render_episode = episode % render_every == 0
        episode_result = run_live_episode(
            episode_seed=episode,
            render_episode=render_episode and not fast_mode,
            episode_label=episode + 1,
        )
        resources = episode_result["resources"]
        total_resources = episode_result["total_resources"]
        total_shaped_reward = episode_result["total_reward"]
        ppo_metrics = episode_result["ppo_metrics"]

        episode_summary = {
            "episode_num": episode,
            "resources_collected": resources.copy(),
            "total_reward": total_shaped_reward,
            "steps": episode_result["steps"],
            "ppo_metrics": ppo_metrics,
        }
        episode_summaries.append(episode_summary)
        recent_rewards.append(total_shaped_reward)
        recent_resources.append(total_resources)
        renderer.update_learning_plot(total_shaped_reward, total_resources)
        renderer.update_ppo_plot(ppo_metrics)
        renderer.refresh(render_delay)

        print(
            f"Episode {episode + 1}: "
            f"resources collected={resources}, total reward={total_shaped_reward:.2f}"
        )

        if (episode + 1) % 100 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")

        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(recent_rewards[-10:]))
            avg_resources = float(np.mean(recent_resources[-10:]))
            print(
                f"Average over episodes {episode - 8}-{episode + 1}: "
                f"resources={avg_resources:.2f}, reward={avg_reward:.2f}"
            )

    for demo_idx in range(final_demo_episodes):
        run_live_episode(
            episode_seed=num_episodes + demo_idx,
            render_episode=True,
            episode_label=num_episodes,
            train_policy=False,
        )

    renderer.close()
    return episode_summaries
