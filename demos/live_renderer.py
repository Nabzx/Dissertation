from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.widgets import Button, Slider

from env.arena import build_octagon_vertices, compute_octagon_mask
from env.gridworld_env import GridWorldEnv
from agents.communication import CommunicationLayer
from agents.ppo_agent import PPOAgent
from env.rewards import apply_reward_scheme


class LiveEpisodeRenderer:
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
        obstacle_value: Optional[int] = None,
        agent_styles: Optional[Dict[str, Dict[str, str]]] = None,
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
        self.speed_delay = 0.001
        self.trail_length = 20
        self.perception_range = 3
        self.communication_fade_steps = 3
        self.resource_anim_steps = 5
        self.obstacle_value = int(obstacle_value) if obstacle_value is not None else int(np.max(initial_grid))
        self.agent_values = list(range(2, self.obstacle_value))
        self.agent_facing = {agent_value: (0, 1) for agent_value in self.agent_values}
        self.agent_colours = ["#38bdf8", "#f43f5e", "#f59e0b", "#a78bfa", "#22c55e", "#f97316"]
        self.agent_edge_colours = ["#0ea5e9", "#be123c", "#d97706", "#7c3aed", "#15803d", "#c2410c"]
        self.agent_labels = {
            agent_value: f"Agent {agent_value - 2}"
            for agent_value in self.agent_values
        }
        if agent_styles:
            for agent_value in self.agent_values:
                agent_id = f"agent_{agent_value - 2}"
                style = agent_styles.get(agent_id, {})
                colour = style.get("color")
                edge = style.get("edge_color")
                label = style.get("label")
                if colour:
                    self.agent_colours[self._agent_index(agent_value) % len(self.agent_colours)] = colour
                if edge:
                    self.agent_edge_colours[self._agent_index(agent_value) % len(self.agent_edge_colours)] = edge
                if label:
                    self.agent_labels[agent_value] = label
        self.arena_palette = {
            "void": "#050608",
            "panel": "#0c0f14",
            "floor_a": "#232b33",
            "floor_b": "#1b2229",
            "floor_outside": "#07090d",
            "tile_edge": "#2a323a",
            "wall": "#9ca3af",
            "wall_glow": "#c7a76a",
            "agent_0": self.agent_colours[0],
            "agent_0_edge": self.agent_edge_colours[0],
            "agent_1": self.agent_colours[1],
            "agent_1_edge": self.agent_edge_colours[1],
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
            agent_value: deque(maxlen=self.trail_length)
            for agent_value in self.agent_values
        }
        self.communication_state = {
            agent_value: {"ttl": 0, "receiver": self._default_receiver(agent_value), "preview": ""}
            for agent_value in self.agent_values
        }
        self.previous_resource_positions: set[tuple[int, int]] = set()
        self.resource_spawn_state: Dict[tuple[int, int], int] = {}
        self.resource_collect_state: Dict[tuple[int, int], int] = {}

        plt.ion()
        self.figure_bg = "#f5f0e6"
        self.fig = plt.figure(figsize=(18.0, 8.4), facecolor=self.figure_bg)
        self.fig.patch.set_facecolor(self.figure_bg)
        self._add_figure_background()
        gs = self.fig.add_gridspec(
            2,
            4,
            width_ratios=[1.12, 0.62, 1.0, 1.0],
            wspace=0.42,
            hspace=0.46,
        )
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.ax_hud = self.fig.add_subplot(gs[:, 1])
        self.ax_plot = self.fig.add_subplot(gs[0, 2])
        self.ax_ppo = self.fig.add_subplot(gs[0, 3])
        self.ax_opt = self.fig.add_subplot(gs[1, 2])
        self.ax_coop = self.fig.add_subplot(gs[1, 3])

        rows, cols = initial_grid.shape
        self.grid_rows = rows
        self.grid_cols = cols
        self.octagon_vertices = build_octagon_vertices(rows, cols)
        self.arena_mask = compute_octagon_mask(rows, cols)
        self.ax_grid.set_title(
            "Environment",
            color="#f8fafc",
            fontsize=14,
            fontweight="bold",
            pad=14,
            loc="center",
        )
        self.ax_grid.set_facecolor(self.arena_palette["void"])
        self.ax_grid.set_xlim(0, cols)
        self.ax_grid.set_ylim(rows, 0)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        for spine in self.ax_grid.spines.values():
            spine.set_visible(False)

        self._setup_hud_axis()

        for row in range(rows):
            for col in range(cols):
                inside_arena = self.arena_mask[row, col]
                base_colour = (
                    self.arena_palette["floor_a"]
                    if (row + col) % 2 == 0
                    else self.arena_palette["floor_b"]
                )
                tile = Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=base_colour if inside_arena else self.arena_palette["floor_outside"],
                    edgecolor=self.arena_palette["tile_edge"] if inside_arena else self.arena_palette["void"],
                    linewidth=0.35 if inside_arena else 0.0,
                    zorder=0,
                )
                self.ax_grid.add_patch(tile)

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

        self.perception_cell_patches = {agent_value: [] for agent_value in self.agent_values}
        perception_fill = {
            agent_value: self._agent_colour(agent_value)
            for agent_value in self.agent_values
        }
        for agent_value, colour in perception_fill.items():
            for _ in range(6):
                patch = Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=colour,
                    edgecolor="none",
                    alpha=0.055,
                    zorder=2.45,
                    visible=False,
                )
                self.perception_cell_patches[agent_value].append(patch)
                self.ax_grid.add_patch(patch)

        self.perception_ray_lines = {agent_value: [] for agent_value in self.agent_values}
        ray_colours = {
            agent_value: self._agent_colour(agent_value)
            for agent_value in self.agent_values
        }
        for agent_value, colour in ray_colours.items():
            for _ in range(3):
                line = Line2D(
                    [],
                    [],
                    color=colour,
                    linewidth=0.85,
                    alpha=0.26,
                    solid_capstyle="round",
                    zorder=3.2,
                    visible=False,
                )
                self.perception_ray_lines[agent_value].append(line)
                self.ax_grid.add_line(line)

        self.communication_lines = {}
        self.communication_pulses = {}
        self.communication_text = {}
        communication_colours = {
            agent_value: (self._agent_colour(agent_value), self._agent_edge_colour(agent_value))
            for agent_value in self.agent_values
        }
        for agent_value, (main_colour, edge_colour) in communication_colours.items():
            line = Line2D(
                [],
                [],
                color=main_colour,
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
                edgecolor=edge_colour,
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
                bbox=dict(facecolor="#111827", edgecolor=main_colour, boxstyle="round,pad=0.2", alpha=0.0),
                zorder=4.7,
                visible=False,
            )
            self.communication_text[agent_value] = text

        self.resource_patches: List[Circle] = []
        for _ in range(max_resources):
            resource_patch = Circle(
                (-10.0, -10.0),
                radius=0.19,
                facecolor=self.arena_palette["resource"],
                edgecolor=self.arena_palette["resource_edge"],
                linewidth=1.35,
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

        self.flag_patch = Polygon(
            [(-10.0, -10.0), (-9.6, -9.8), (-10.0, -9.6)],
            closed=True,
            facecolor="#facc15",
            edgecolor="#fde68a",
            linewidth=1.8,
            alpha=0.95,
            zorder=4.2,
            visible=False,
        )
        self.flag_glow_patch = Circle(
            (-10.0, -10.0),
            radius=0.46,
            facecolor="#facc15",
            edgecolor="none",
            alpha=0.0,
            zorder=3.9,
            visible=False,
        )
        self.winner_highlight_patch = Circle(
            (-10.0, -10.0),
            radius=0.46,
            facecolor="none",
            edgecolor="#fde68a",
            linewidth=2.6,
            alpha=0.95,
            zorder=5,
            visible=False,
        )
        self.ax_grid.add_patch(self.flag_glow_patch)
        self.ax_grid.add_patch(self.flag_patch)
        self.ax_grid.add_patch(self.winner_highlight_patch)

        self.obstacle_patches: List[Rectangle] = []
        for row in range(rows):
            for col in range(cols):
                obstacle_patch = Rectangle(
                    (col + 0.12, row + 0.12),
                    0.76,
                    0.76,
                    facecolor="#3f4752",
                    edgecolor=self.arena_palette["obstacle_edge"],
                    linewidth=0.7,
                    zorder=2,
                    visible=False,
                )
                self.obstacle_patches.append(obstacle_patch)
                self.ax_grid.add_patch(obstacle_patch)

        self.agent_patches = {}
        for agent_value in self.agent_values:
            self.agent_patches[agent_value] = Circle(
                (0.5, 0.5),
                radius=0.3,
                facecolor=self._agent_colour(agent_value),
                edgecolor=self._agent_edge_colour(agent_value),
                linewidth=2.0,
                zorder=4,
                visible=False,
            )
        for patch in self.agent_patches.values():
            self.ax_grid.add_patch(patch)

        trail_colours = {
            agent_value: self._agent_colour(agent_value)
            for agent_value in self.agent_values
        }
        self.trail_patches = {agent_value: [] for agent_value in self.agent_values}
        for agent_value, colour in trail_colours.items():
            for _ in range(self.trail_length):
                trail_patch = Circle(
                    (-10.0, -10.0),
                    radius=0.12,
                    facecolor=colour,
                    edgecolor="none",
                    alpha=0.0,
                    zorder=3.6,
                    visible=False,
                )
                self.trail_patches[agent_value].append(trail_patch)
                self.ax_grid.add_patch(trail_patch)

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self._agent_colour(self.agent_values[0]), markeredgecolor=self._agent_edge_colour(self.agent_values[0]), markersize=9, label="Agent"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=self.arena_palette["resource"], markeredgecolor=self.arena_palette["resource_edge"], markersize=7, label="Resource"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#facc15", markeredgecolor="#fde68a", markersize=9, label="Flag"),
            Rectangle((0, 0), 1, 1, facecolor=self.arena_palette["obstacle"], edgecolor=self.arena_palette["obstacle_edge"], label="Obstacle"),
        ]
        self.ax_grid.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=4,
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
        self.gradient_norms: List[float] = []
        self.fairness_values: List[float] = []
        self.balance_values: List[float] = []
        self.reward_line, = self.ax_plot.plot([], [], color="#2563eb", linewidth=2.2, label="Reward Avg (50)")
        self.resource_line, = self.ax_plot.plot(
            [],
            [],
            color="#16a34a",
            linewidth=2.2,
            label="Resources Avg (50)",
        )
        self.ax_plot.set_xlim(0, max(1, num_episodes))
        self.ax_plot.set_ylim(0, max(1.0, max_possible_reward))
        self._style_ui_axis(self.ax_plot, "Learning Progress")
        self.ax_plot.set_xlabel("Episode")
        self.ax_plot.set_ylabel("Smoothed Reward / Resources")
        self.ax_plot.grid(True, color="#e5e7eb", alpha=0.9, linewidth=0.8)
        self.ax_plot.legend(loc="upper left", frameon=False, fontsize=9)

        self.policy_line, = self.ax_ppo.plot(
            [],
            [],
            color="#dc2626",
            linewidth=2.2,
            label="Policy Loss (PPO)",
        )
        self.ax_entropy = self.ax_ppo.twinx()
        self.ax_entropy.set_facecolor("white")
        self.entropy_line, = self.ax_entropy.plot(
            [],
            [],
            color="#7c3aed",
            linewidth=2.2,
            label="Entropy (Exploration)",
        )
        self.ax_ppo.set_xlim(0, max(1, num_episodes))
        self._style_ui_axis(self.ax_ppo, "PPO Training Dynamics")
        self.ax_ppo.set_xlabel("Episode")
        self.ax_ppo.set_ylabel("Policy Loss", color="#dc2626")
        self.ax_ppo.tick_params(axis="y", labelcolor="#dc2626")
        self.ax_ppo.grid(True, color="#e5e7eb", alpha=0.9, linewidth=0.8)
        self.ax_entropy.set_ylabel("Entropy", color="#7c3aed")
        self.ax_entropy.tick_params(axis="y", labelcolor="#7c3aed")
        self.ax_ppo.set_ylim(-1.0, 2.0)
        self.ax_entropy.set_ylim(0.0, 2.0)
        self.ax_entropy.spines["right"].set_color("#9ca3af")
        ppo_handles = [self.policy_line, self.entropy_line]
        self.ax_ppo.legend(ppo_handles, [line.get_label() for line in ppo_handles], loc="upper right", frameon=False, fontsize=9)

        self.value_line, = self.ax_opt.plot(
            [],
            [],
            color="#ea580c",
            linewidth=2.2,
            label="Value Loss Avg (50)",
        )
        self.grad_line, = self.ax_opt.plot(
            [],
            [],
            color="#0891b2",
            linewidth=2.0,
            linestyle="--",
            label="Grad Norm Avg (50)",
        )
        self.ax_opt.set_xlim(0, max(1, num_episodes))
        self._style_ui_axis(self.ax_opt, "PPO Optimisation Signals")
        self.ax_opt.set_xlabel("Episode")
        self.ax_opt.set_ylabel("Loss Value")
        self.ax_opt.grid(True, color="#e5e7eb", alpha=0.9, linewidth=0.8)
        self.ax_opt.legend(loc="upper right", frameon=False, fontsize=9)

        self.fairness_line, = self.ax_coop.plot(
            [],
            [],
            color="#0f766e",
            linewidth=2.2,
            label="Jain Fairness Avg (50)",
        )
        self.balance_line, = self.ax_coop.plot(
            [],
            [],
            color="#9333ea",
            linewidth=2.0,
            label="Resource Balance Avg (50)",
        )
        self.ax_coop.set_xlim(0, max(1, num_episodes))
        self.ax_coop.set_ylim(0.0, 1.05)
        self._style_ui_axis(self.ax_coop, "Cooperation & Fairness")
        self.ax_coop.set_xlabel("Episode")
        self.ax_coop.set_ylabel("Metric Value")
        self.ax_coop.grid(True, color="#e5e7eb", alpha=0.9, linewidth=0.8)
        self.ax_coop.legend(loc="lower right", frameon=False, fontsize=9)

        self.fig.subplots_adjust(left=0.035, right=0.95, top=0.91, bottom=0.18)
        self._shift_environment_axis_right()
        self._update_env(initial_grid)

    def _shift_environment_axis_right(self) -> None:
        pos = self.ax_grid.get_subplotspec().get_position(self.fig)
        self.ax_grid.set_position([pos.x0 + 0.018, pos.y0, pos.width, pos.height])

    def _add_figure_background(self) -> None:
        bg_axis = self.fig.add_axes([0, 0, 1, 1], zorder=-10)
        bg_axis.set_axis_off()
        x_grad = np.linspace(0.0, 1.0, 320)
        y_grad = np.linspace(0.0, 1.0, 180)
        gradient = np.outer(y_grad, x_grad)
        base = np.ones((180, 320, 3))
        warm = np.array([245, 240, 230], dtype=float) / 255.0
        shadow = np.array([232, 222, 207], dtype=float) / 255.0
        base[:] = warm
        base = base * (1.0 - 0.38 * gradient[..., None]) + shadow * (0.38 * gradient[..., None])
        bg_axis.imshow(base, aspect="auto", extent=(0, 1, 0, 1), origin="lower")

    def _agent_index(self, agent_value: int) -> int:
        return int(agent_value) - 2

    def _agent_colour(self, agent_value: int) -> str:
        return self.agent_colours[self._agent_index(agent_value) % len(self.agent_colours)]

    def _agent_edge_colour(self, agent_value: int) -> str:
        return self.agent_edge_colours[self._agent_index(agent_value) % len(self.agent_edge_colours)]

    def _agent_id(self, agent_value: int) -> str:
        return f"agent_{self._agent_index(agent_value)}"

    def _agent_value(self, agent_id: str) -> int:
        return 2 + int(agent_id.split("_")[-1])

    def _default_receiver(self, agent_value: int) -> int:
        if len(self.agent_values) <= 1:
            return agent_value
        idx = self.agent_values.index(agent_value)
        return self.agent_values[(idx + 1) % len(self.agent_values)]

    def _style_ui_axis(self, axis, title: str) -> None:
        axis.set_facecolor("white")
        axis.set_title(title, color="#111827", fontsize=11.5, fontweight="bold", pad=9)
        axis.tick_params(colors="#374151", labelsize=8.5)
        axis.xaxis.label.set_color("#111827")
        axis.yaxis.label.set_color("#111827")
        axis.xaxis.label.set_size(9.5)
        axis.yaxis.label.set_size(9.5)
        for spine in axis.spines.values():
            spine.set_color("#d1d5db")

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
            fontsize=13.5,
            fontweight="bold",
            va="top",
        )
        self.ax_hud.add_patch(
            Rectangle((0.06, 0.905), 0.88, 0.0018, facecolor="#334155", edgecolor="none", alpha=0.9)
        )

        self.ax_hud.add_patch(
            Rectangle(
                (0.06, 0.70),
                0.88,
                0.17,
                facecolor="#111827",
                edgecolor="#1f2937",
                linewidth=1.0,
                zorder=0.5,
            )
        )
        self.hud_phase_text = self.ax_hud.text(
            0.1, 0.845, "", color="#f8fafc", fontsize=9.4, fontweight="bold", va="top"
        )
        self.hud_episode_text = self.ax_hud.text(
            0.1, 0.798, "", color="#cbd5e1", fontsize=8.2, va="top"
        )
        self.hud_step_text = self.ax_hud.text(
            0.1, 0.758, "", color="#cbd5e1", fontsize=8.2, va="top"
        )
        self.hud_reward_text = self.ax_hud.text(
            0.1, 0.718, "", color="#cbd5e1", fontsize=8.2, va="top"
        )

        self.hud_agent_text = {}
        card_count = max(1, len(self.agent_values))
        top = 0.665
        bottom = 0.035
        gap = 0.016
        card_height = max(0.112, min(0.152, (top - bottom - gap * (card_count - 1)) / card_count))
        for idx, agent_value in enumerate(self.agent_values):
            agent_id = self._agent_id(agent_value)
            y = top - (idx + 1) * card_height - idx * gap
            accent = self._agent_colour(agent_value)
            self.ax_hud.add_patch(
                Rectangle(
                    (0.06, y),
                    0.88,
                    card_height,
                    facecolor="#111827",
                    edgecolor="#1f2937",
                    linewidth=1.1,
                    zorder=0.5,
                )
            )
            self.ax_hud.add_patch(
                Rectangle(
                    (0.06, y),
                    0.018,
                    card_height,
                    facecolor=accent,
                    edgecolor="none",
                    zorder=0.6,
                )
            )
            self.ax_hud.text(
                0.1,
                y + card_height - 0.026,
                f"Agent {idx}",
                color=accent,
                fontsize=8.9,
                fontweight="bold",
                va="top",
            )
            self.hud_agent_text[agent_id] = self.ax_hud.text(
                0.1,
                y + card_height - 0.057,
                "",
                color="#e5e7eb",
                fontsize=6.3 if card_count > 2 else 7.2,
                va="top",
                linespacing=1.18,
            )

    def update_hud(self, hud_state: Optional[Dict[str, object]]) -> None:
        if hud_state is None:
            self.hud_phase_text.set_text("")
            self.hud_episode_text.set_text("")
            self.hud_step_text.set_text("")
            self.hud_reward_text.set_text("")
            for text in self.hud_agent_text.values():
                text.set_text("")
            return

        episode = hud_state.get("episode", 0)
        total_episodes = hud_state.get("total_episodes", 0)
        step = hud_state.get("step", 0)
        phase = hud_state.get("phase", "Training")
        episode_reward = float(hud_state.get("episode_reward", 0.0))
        game_metrics = hud_state.get("game_metrics", {}) or {}
        game_mode = game_metrics.get("mode") or hud_state.get("game_mode")
        winner = game_metrics.get("winner") if isinstance(game_metrics, dict) else None
        avg_time = game_metrics.get("average_time_to_flag") if isinstance(game_metrics, dict) else None
        flag_position = game_metrics.get("flag_position") if isinstance(game_metrics, dict) else None
        phase_label = "CAPTURE THE FLAG" if game_mode == "capture_flag" else str(phase).upper()
        self.hud_phase_text.set_text(phase_label)
        self.hud_episode_text.set_text(f"Episode: {episode}/{total_episodes}")
        flag_text = f" | Flag: {flag_position}" if flag_position is not None else ""
        self.hud_step_text.set_text(f"Step: {step}{flag_text}")
        winner_text = f" | Winner: {winner}" if winner else ""
        avg_text = f" | Avg time: {avg_time:.1f}" if avg_time is not None else ""
        self.hud_reward_text.set_text(f"Episode reward: {episode_reward:.2f}{winner_text}{avg_text}")

        agent_states = hud_state.get("agents", {})
        distances = game_metrics.get("distances_to_flag", {}) if isinstance(game_metrics, dict) else {}
        wins = game_metrics.get("wins", {}) if isinstance(game_metrics, dict) else {}
        for agent_id in self.hud_agent_text.keys():
            info = agent_states.get(agent_id, {})
            resources = int(info.get("resources", 0))
            cumulative = int(info.get("cumulative_resources", 0))
            position = info.get("position", ("-", "-"))
            facing = info.get("facing", "unknown")
            comm_status = info.get("communication", "offline")
            agent_type = info.get("agent_type", "unknown")
            status = info.get("status", "Active")
            action = info.get("recent_action", "n/a")
            lines = [
                f"Type: {agent_type}",
                f"Res: {resources} | Run: {cumulative}",
                f"Pos: {position}",
                f"Act: {action}",
            ]
            if agent_id in distances:
                lines.append(f"Flag dist: {distances[agent_id]}")
            if agent_id in wins:
                lines.append(f"Wins: {wins[agent_id]}")
            self.hud_agent_text[agent_id].set_text("\n".join(lines))

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
        render_info: Optional[Dict[str, object]] = None,
    ) -> None:
        self._update_env(
            grid,
            actions=actions,
            communication_events=communication_events,
            render_info=render_info,
        )
        self.text.set_text(f"Episode {episode}/{num_episodes} | Step {step}")
        self.update_hud(hud_state)
        self.fig.canvas.draw_idle()
        plt.pause(self.get_speed_delay(render_delay))

    def _update_env(
        self,
        grid: np.ndarray,
        actions: Optional[Dict[str, int]] = None,
        communication_events: Optional[List[Dict[str, object]]] = None,
        render_info: Optional[Dict[str, object]] = None,
    ) -> None:
        self._update_facing(actions)
        self._update_perception(grid)
        self._update_resource_animation_state(grid)

        obstacle_mask = grid == self.obstacle_value
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

        now_positions: Dict[int, tuple[int, int]] = {}
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
            now_positions[agent_value] = (int(row), int(col))

        self._update_communication_visuals(now_positions, communication_events)
        self._update_game_mode_visuals(render_info, now_positions)

    def _update_game_mode_visuals(
        self,
        render_info: Optional[Dict[str, object]],
        now_positions: Dict[int, tuple[int, int]],
    ) -> None:
        self.flag_patch.set_visible(False)
        self.flag_glow_patch.set_visible(False)
        self.winner_highlight_patch.set_visible(False)

        if not render_info:
            return

        flag_position = render_info.get("flag_position")
        if flag_position is not None:
            row, col = int(flag_position[0]), int(flag_position[1])
            if 0 <= row < self.arena_mask.shape[0] and 0 <= col < self.arena_mask.shape[1]:
                if self.arena_mask[row, col]:
                    cx, cy = float(col) + 0.5, float(row) + 0.5
                    self.flag_patch.set_xy([
                        (cx - 0.20, cy + 0.24),
                        (cx + 0.26, cy + 0.08),
                        (cx - 0.20, cy - 0.08),
                    ])
                    self.flag_patch.set_visible(True)
                    self.flag_glow_patch.center = (cx, cy)
                    self.flag_glow_patch.set_alpha(0.22)
                    self.flag_glow_patch.set_visible(True)

        winner = render_info.get("winner")
        if winner:
            winner_value = self._agent_value(str(winner))
            winner_position = now_positions.get(winner_value)
            if winner_position is not None:
                row, col = winner_position
                self.winner_highlight_patch.center = (float(col) + 0.5, float(row) + 0.5)
                self.winner_highlight_patch.set_visible(True)

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
            agent_value = self._agent_value(agent_id)
            if agent_value not in self.agent_facing:
                continue
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
            fan_offsets = [-0.45, 0.0, 0.45]
            lateral = (1, 0)
        else:
            fan_offsets = [-0.45, 0.0, 0.45]
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

                cell_centre = (float(target_col) + 0.5, float(target_row) + 0.5)
                points.append(cell_centre)
                cells.append((target_row, target_col))

                if grid[target_row, target_col] == self.obstacle_value:
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

    def reset_communication_visuals(self) -> None:
        for state in self.communication_state.values():
            state["ttl"] = 0
            state["preview"] = ""
        for agent_value in self.communication_lines.keys():
            self.communication_lines[agent_value].set_visible(False)
            self.communication_pulses[agent_value].set_visible(False)
            self.communication_text[agent_value].set_visible(False)

    def reset_playback_visual_state(self, grid: np.ndarray) -> None:
        self.reset_communication_visuals()
        self.resource_spawn_state.clear()
        self.resource_collect_state.clear()
        self.previous_resource_positions = {tuple(pos) for pos in np.argwhere(grid == 1)}
        for agent_value in self.agent_trails:
            self.agent_trails[agent_value].clear()
            positions = np.argwhere(grid == agent_value)
            if len(positions):
                row, col = positions[0]
                self.agent_trails[agent_value].append((int(row), int(col)))
            self._update_trail_patches(agent_value)

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
        bend = 0.18 if self._agent_index(sender) % 2 == 0 else -0.18
        control_x = mid_x + perp_x * distance * bend
        control_y = mid_y + perp_y * distance * bend
        t = np.linspace(0.0, 1.0, 20)
        curve_x = (1 - t) ** 2 * sx + 2 * (1 - t) * t * control_x + t**2 * rx
        curve_y = (1 - t) ** 2 * sy + 2 * (1 - t) * t * control_y + t**2 * ry
        return curve_x, curve_y

    def update_learning_plot(
        self,
        total_reward: float,
        total_resources: float,
        per_agent_resources: Optional[Dict[str, float]] = None,
    ) -> None:
        self.episode_rewards.append(float(total_reward))
        self.episode_resources.append(float(total_resources))
        self._append_cooperation_metrics(per_agent_resources)
        if len(self.episode_rewards) % self.plot_update_every != 0:
            return

        x_vals = list(range(len(self.episode_rewards)))
        reward_smoothed = self._moving_avg(self.episode_rewards, self.smoothing_window)
        resource_smoothed = self._moving_avg(self.episode_resources, self.smoothing_window)

        plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
        plot_reward = reward_smoothed[:: self.plot_downsample] or reward_smoothed[-1:]
        plot_resources = resource_smoothed[:: self.plot_downsample] or resource_smoothed[-1:]

        self.reward_line.set_data(plot_x, plot_reward)
        self.resource_line.set_data(plot_x, plot_resources)
        self._update_cooperation_lines(x_vals)

    def update_ppo_plot(self, ppo_metrics: Optional[Dict[str, float]]) -> None:
        if not ppo_metrics:
            ppo_metrics = {}
        self.policy_losses.append(float(ppo_metrics.get("policy_loss", 0.0)))
        self.value_losses.append(float(ppo_metrics.get("value_loss", 0.0)))
        self.entropy_values.append(float(ppo_metrics.get("entropy", 0.0)))
        self.gradient_norms.append(float(ppo_metrics.get("grad_norm", ppo_metrics.get("gradient_norm", np.nan))))
        if len(self.policy_losses) % self.plot_update_every != 0:
            return

        x_vals = list(range(len(self.policy_losses)))
        policy_smoothed = self._moving_avg(self.policy_losses, self.smoothing_window)
        entropy_smoothed = self._moving_avg(self.entropy_values, self.smoothing_window)
        value_smoothed = self._moving_avg(self.value_losses, self.smoothing_window)
        grad_smoothed = self._moving_avg_skip_nan(self.gradient_norms, self.smoothing_window)

        plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
        plot_policy = policy_smoothed[:: self.plot_downsample] or policy_smoothed[-1:]
        plot_entropy = entropy_smoothed[:: self.plot_downsample] or entropy_smoothed[-1:]
        plot_value = value_smoothed[:: self.plot_downsample] or value_smoothed[-1:]
        plot_grad = grad_smoothed[:: self.plot_downsample] or grad_smoothed[-1:]

        self.policy_line.set_data(plot_x, plot_policy)
        self.entropy_line.set_data(plot_x, plot_entropy)
        self.value_line.set_data(plot_x, plot_value)
        self.grad_line.set_data(plot_x, plot_grad)
        self.ax_opt.relim()
        self.ax_opt.autoscale_view(scalex=False, scaley=True)

    def _append_cooperation_metrics(self, per_agent_resources: Optional[Dict[str, float]]) -> None:
        if not per_agent_resources:
            self.fairness_values.append(np.nan)
            self.balance_values.append(np.nan)
            return

        values = np.array(list(per_agent_resources.values()), dtype=float)
        total = float(np.sum(values))
        squared_sum = float(np.sum(values**2))
        if len(values) == 0 or squared_sum <= 0.0:
            fairness = np.nan
            balance = np.nan
        else:
            fairness = float((total**2) / (len(values) * squared_sum))
            balance = float(1.0 - ((np.max(values) - np.min(values)) / max(total, 1.0)))
            balance = float(np.clip(balance, 0.0, 1.0))

        self.fairness_values.append(fairness)
        self.balance_values.append(balance)

    def _update_cooperation_lines(self, x_vals: List[int]) -> None:
        fairness_smoothed = self._moving_avg_skip_nan(self.fairness_values, self.smoothing_window)
        balance_smoothed = self._moving_avg_skip_nan(self.balance_values, self.smoothing_window)

        plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
        plot_fairness = fairness_smoothed[:: self.plot_downsample] or fairness_smoothed[-1:]
        plot_balance = balance_smoothed[:: self.plot_downsample] or balance_smoothed[-1:]

        self.fairness_line.set_data(plot_x, plot_fairness)
        self.balance_line.set_data(plot_x, plot_balance)

    def _moving_avg(self, values: List[float], window: int) -> List[float]:
        return [
            float(np.mean(values[max(0, idx - window + 1) : idx + 1]))
            for idx in range(len(values))
        ]

    def _moving_avg_skip_nan(self, values: List[float], window: int) -> List[float]:
        smoothed: List[float] = []
        for idx in range(len(values)):
            window_values = np.array(values[max(0, idx - window + 1) : idx + 1], dtype=float)
            valid = window_values[~np.isnan(window_values)]
            smoothed.append(float(np.mean(valid)) if len(valid) else np.nan)
        return smoothed

    def refresh(self, render_delay: float) -> None:
        self.fig.canvas.draw_idle()
        plt.pause(self.get_speed_delay(render_delay))

    def get_speed_delay(self, fallback_delay: float = 0.001) -> float:
        delay = getattr(self, "speed_delay", fallback_delay)
        return float(np.clip(delay, 0.0001, 0.1))

    def set_speed_delay(self, delay: float) -> None:
        self.speed_delay = float(np.clip(delay, 0.0001, 0.1))
        self._update_speed_status_label()

    def set_speed_preset(self, delay: float) -> None:
        self.set_speed_delay(delay)
        if hasattr(self, "live_speed_slider"):
            self.live_speed_slider.set_val(self.speed_delay)

    def attach_speed_status_label(self, mode_label: str) -> None:
        self.speed_status_mode = mode_label
        if not hasattr(self, "speed_status_text"):
            self.speed_status_text = self.fig.text(
                0.835,
                0.025,
                "",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#e5e7eb",
                bbox=dict(facecolor="#0f172a", edgecolor="#334155", boxstyle="round,pad=0.28", alpha=0.9),
            )
        self._update_speed_status_label()

    def _update_speed_status_label(self) -> None:
        if not hasattr(self, "speed_status_text"):
            return
        mode_label = getattr(self, "speed_status_mode", "Playback")
        self.speed_status_text.set_text(f"{mode_label} delay: {self.speed_delay:.4f}s/frame")

    def setup_live_speed_controls(self, initial_delay: float) -> None:
        self.set_speed_delay(initial_delay)
        self.fig.subplots_adjust(bottom=0.24)
        self._shift_environment_axis_right()

        ax_speed = self.fig.add_axes([0.08, 0.06, 0.46, 0.025])
        self.live_speed_slider = Slider(
            ax_speed,
            "Delay",
            0.0001,
            0.1,
            valinit=self.speed_delay,
            valfmt="%.4f",
        )
        self.live_speed_slider.on_changed(self.set_speed_delay)
        self.attach_speed_status_label("Live")

        preset_specs = [
            ("Slow", [0.58, 0.045, 0.07, 0.045], 0.1),
            ("Normal", [0.66, 0.045, 0.08, 0.045], 0.01),
            ("Fast", [0.75, 0.045, 0.07, 0.045], 0.0001),
        ]
        self.live_speed_buttons = []
        for label, rect, delay in preset_specs:
            button = Button(self.fig.add_axes(rect), label)
            button.on_clicked(lambda _event, d=delay: self.set_speed_preset(d))
            self.live_speed_buttons.append(button)

    def close(self) -> None:
        plt.ioff()
        plt.show()


class PlaybackController:
    def __init__(
        self,
        renderer: LiveEpisodeRenderer,
        history: List[List[Dict[str, object]]],
        summaries: List[Dict],
    ):
        self.renderer = renderer
        self.history = history
        self.summaries = summaries
        self.current_episode = 0
        self.current_step = 0
        self.playing = False
        self.updating_sliders = False

        self.renderer.fig.subplots_adjust(bottom=0.24)
        self.renderer._shift_environment_axis_right()
        self._create_controls()
        self._connect_keys()
        self._show_frame()

    def _create_controls(self) -> None:
        max_episode = max(0, len(self.history) - 1)
        max_step = max(0, len(self.history[0]) - 1) if self.history else 0

        ax_episode = self.renderer.fig.add_axes([0.08, 0.15, 0.56, 0.025])
        ax_step = self.renderer.fig.add_axes([0.08, 0.105, 0.56, 0.025])
        ax_speed = self.renderer.fig.add_axes([0.08, 0.06, 0.46, 0.025])

        self.episode_slider = Slider(
            ax_episode,
            "Episode",
            0,
            max_episode,
            valinit=0,
            valstep=1,
        )
        self.step_slider = Slider(
            ax_step,
            "Step",
            0,
            max_step,
            valinit=0,
            valstep=1,
        )
        self.speed_slider = Slider(
            ax_speed,
            "Delay",
            0.0001,
            0.1,
            valinit=self.renderer.get_speed_delay(0.01),
            valfmt="%.4f",
        )
        self.speed_slider.on_changed(self.renderer.set_speed_delay)
        self.renderer.attach_speed_status_label("Playback")

        self.episode_slider.on_changed(self._on_episode_slider)
        self.step_slider.on_changed(self._on_step_slider)

        button_specs = [
            ("Play/Pause", [0.69, 0.13, 0.1, 0.045], self.toggle_play),
            ("<< Ep", [0.805, 0.13, 0.075, 0.045], self.prev_episode),
            ("Ep >>", [0.89, 0.13, 0.075, 0.045], self.next_episode),
            ("- Step", [0.805, 0.07, 0.075, 0.045], self.prev_step),
            ("Step +", [0.89, 0.07, 0.075, 0.045], self.next_step),
            ("Slow", [0.58, 0.045, 0.07, 0.045], lambda: self.set_speed_preset(0.1)),
            ("Normal", [0.66, 0.045, 0.08, 0.045], lambda: self.set_speed_preset(0.01)),
            ("Fast", [0.75, 0.045, 0.07, 0.045], lambda: self.set_speed_preset(0.0001)),
        ]
        self.buttons = []
        for label, rect, callback in button_specs:
            button = Button(self.renderer.fig.add_axes(rect), label)
            button.on_clicked(lambda _event, cb=callback: cb())
            self.buttons.append(button)

    def _connect_keys(self) -> None:
        self.renderer.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self.prev_step()
        elif event.key == "right":
            self.next_step()
        elif event.key == "up":
            self.prev_episode()
        elif event.key == "down":
            self.next_episode()

    def _on_episode_slider(self, value: float) -> None:
        if self.updating_sliders:
            return
        self.current_episode = int(value)
        self.current_step = min(self.current_step, len(self.history[self.current_episode]) - 1)
        self._sync_step_slider_bounds()
        self._show_frame()

    def _on_step_slider(self, value: float) -> None:
        if self.updating_sliders:
            return
        self.current_step = int(value)
        self._show_frame()

    def _sync_step_slider_bounds(self) -> None:
        max_step = max(0, len(self.history[self.current_episode]) - 1)
        self.step_slider.valmax = max_step
        self.step_slider.ax.set_xlim(self.step_slider.valmin, max_step)
        self.current_step = min(self.current_step, max_step)
        self.updating_sliders = True
        self.step_slider.set_val(self.current_step)
        self.updating_sliders = False

    def _sync_sliders(self) -> None:
        self.updating_sliders = True
        self.episode_slider.set_val(self.current_episode)
        self._sync_step_slider_bounds()
        self.step_slider.set_val(self.current_step)
        self.updating_sliders = False

    def _show_frame(self) -> None:
        frame = self._current_frame()
        grid = self._frame_grid(frame)
        self.renderer.reset_playback_visual_state(grid)
        hud_state = self._build_playback_hud_state(frame)
        self.renderer.update(
            grid,
            int(frame.get("episode", self.current_episode + 1)),
            len(self.history),
            int(frame.get("step", self.current_step)),
            self.renderer.get_speed_delay(0.001),
            actions=None,
            communication_events=None,
            hud_state=hud_state,
            render_info=frame.get("render_info"),
        )

    def _current_frame(self) -> Dict[str, object]:
        frame = self.history[self.current_episode][self.current_step]
        if isinstance(frame, np.ndarray):
            return {
                "episode": self.current_episode + 1,
                "step": self.current_step,
                "grid": frame,
                "agent_positions": {},
                "resource_positions": [],
            }
        return frame

    def _frame_grid(self, frame: Dict[str, object]) -> np.ndarray:
        return np.asarray(frame["grid"])

    def _build_playback_hud_state(self, frame: Dict[str, object]) -> Dict[str, object]:
        grid = self._frame_grid(frame)
        summary = self.summaries[self.current_episode] if self.current_episode < len(self.summaries) else {}
        resources_collected = summary.get("resources_collected", {})
        game_metrics = frame.get("game_metrics", summary.get("game_metrics", {}))
        recorded_positions = frame.get("agent_positions", {})
        agents_state = {}
        for agent_value in self.renderer.agent_values:
            agent_id = self.renderer._agent_id(agent_value)
            positions = np.argwhere(grid == agent_value)
            position = recorded_positions.get(agent_id)
            if position is None:
                position = tuple(map(int, positions[0])) if len(positions) else ("-", "-")
            agents_state[agent_id] = {
                "resources": resources_collected.get(agent_id, 0),
                "cumulative_resources": resources_collected.get(agent_id, 0),
                "position": position,
                "facing": "playback",
                "communication": "recorded",
                "status": "Active" if len(positions) else "Hidden",
                "recent_action": "n/a",
            }

        return {
            "phase": "Playback",
            "episode": int(frame.get("episode", self.current_episode + 1)),
            "total_episodes": len(self.history),
            "step": int(frame.get("step", self.current_step)),
            "episode_reward": float(summary.get("total_reward", 0.0)),
            "game_metrics": game_metrics,
            "agents": agents_state,
        }

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def set_speed_preset(self, delay: float) -> None:
        self.speed_slider.set_val(float(np.clip(delay, 0.0001, 0.1)))

    def prev_episode(self) -> None:
        self.current_episode = max(0, self.current_episode - 1)
        self.current_step = 0
        self._sync_sliders()
        self._show_frame()

    def next_episode(self) -> None:
        self.current_episode = min(len(self.history) - 1, self.current_episode + 1)
        self.current_step = 0
        self._sync_sliders()
        self._show_frame()

    def prev_step(self) -> None:
        if self.current_step > 0:
            self.current_step -= 1
        elif self.current_episode > 0:
            self.current_episode -= 1
            self.current_step = len(self.history[self.current_episode]) - 1
        self._sync_sliders()
        self._show_frame()

    def next_step(self) -> None:
        if self.current_step < len(self.history[self.current_episode]) - 1:
            self.current_step += 1
        elif self.current_episode < len(self.history) - 1:
            self.current_episode += 1
            self.current_step = 0
        else:
            self.playing = False
        self._sync_sliders()
        self._show_frame()

    def run(self) -> None:
        plt.ion()
        while plt.fignum_exists(self.renderer.fig.number):
            if self.playing:
                self.next_step()
            delay = self.renderer.get_speed_delay(float(self.speed_slider.val))
            plt.pause(delay)


def run_live_training(
    num_episodes: int = 1000,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 25,
    num_agents: int = 4,
    num_resources: int = 25,
    num_obstacles: int = 45,
    max_steps: int = 250,
    render_delay: float = 0.001,
    render_every: int = 50,
    fast_mode: bool = False,
    final_demo_episodes: int = 10,
    show_perception: bool = True,
    show_communication: bool = True,
    mode: str = "playback",
) -> List[Dict]:
    mode = mode.lower().strip()
    if mode not in {"live", "playback"}:
        raise ValueError("mode must be 'live' or 'playback'")

    env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )
    max_possible_reward = float(num_resources * len(env.agents))

    obs_size = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_size))
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
        obstacle_value=env.obstacle_value,
    )
    renderer.set_speed_delay(render_delay)
    if mode == "live":
        renderer.setup_live_speed_controls(render_delay)

    episode_summaries: List[Dict] = []
    history: List[List[Dict[str, object]]] = []
    recent_rewards: List[float] = []
    recent_resources: List[int] = []
    cumulative_resources_run = {agent: 0 for agent in env.agents}
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
        record_history: bool = True,
    ) -> Dict:
        def facing_label(agent_id: str) -> str:
            agent_value = renderer._agent_value(agent_id)
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
            comm_lookup = {agent: "idle" for agent in env.agents}
            if communication_events:
                for event in communication_events:
                    sender_id = renderer._agent_id(int(event["sender"]))
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
        episode_history: List[Dict[str, object]] = []

        def build_history_frame(step_idx: int) -> Dict[str, object]:
            grid_snapshot = env.grid.copy()
            return {
                "episode": episode_label,
                "step": step_idx,
                "grid": grid_snapshot,
                "agent_positions": {
                    agent_id: tuple(map(int, position))
                    for agent_id, position in env.agent_positions.items()
                },
                "resource_positions": [
                    tuple(map(int, position))
                    for position in np.argwhere(grid_snapshot == 1)
                ],
            }

        if record_history:
            episode_history.append(build_history_frame(0))

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
            renderer.reset_communication_visuals()
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
            if record_history:
                episode_history.append(build_history_frame(step + 1))

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
                        trajectory_id=agent_id,
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
                            "sender": env.agent_value(sender_id),
                            "receiver": env.agent_value(receiver_id),
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
            "history": episode_history,
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
        history.append(episode_result["history"])
        recent_rewards.append(total_shaped_reward)
        recent_resources.append(total_resources)
        renderer.update_learning_plot(total_shaped_reward, total_resources, resources)
        renderer.update_ppo_plot(ppo_metrics)
        renderer.refresh(render_delay)

        print(
            f"Episode {episode + 1}: "
            f"resources collected={resources}, total reward={total_shaped_reward:.2f}, "
            f"policy_loss={ppo_metrics.get('policy_loss', 0.0):.4f}, "
            f"value_loss={ppo_metrics.get('value_loss', 0.0):.4f}, "
            f"entropy={ppo_metrics.get('entropy', 0.0):.4f}, "
            f"mean_reward={ppo_metrics.get('mean_reward', 0.0):.4f}"
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

    if mode == "playback":
        PlaybackController(renderer, history, episode_summaries).run()
    else:
        for demo_idx in range(final_demo_episodes):
            run_live_episode(
                episode_seed=num_episodes + demo_idx,
                render_episode=True,
                episode_label=num_episodes,
                train_policy=False,
                record_history=False,
            )
        renderer.close()

    return episode_summaries
