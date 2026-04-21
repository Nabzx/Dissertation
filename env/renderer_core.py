from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from env.arena import build_octagon_vertices, compute_octagon_mask
from env.gridworld_env import GridWorldEnv
from agents.communication import CommunicationLayer
from agents.ppo_agent import PPOAgent
from env.rewards import apply_reward_scheme

from env import renderer_draw, renderer_ui

FRAME_TIME = 1.0 / 60.0


def build_communication_events_from_flags(
    env: GridWorldEnv,
    previews: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    comms: List[Dict[str, object]] = []
    flags = getattr(env, "just_communicated", {})
    previews = previews or {}

    for agent_id in env.agents:
        if not flags.get(agent_id, False):
            continue

        receiver_id = next((other for other in env.agents if other != agent_id), agent_id)
        comms.append(
            {
                "sender": env.agent_value(agent_id),
                "receiver": env.agent_value(receiver_id),
                "preview": previews.get(agent_id, "proximity"),
            }
        )

    return comms


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
        self.trail_length = 20
        self.perception_range = 3
        self.communication_growth_rate = 0.18
        self.communication_fade_rate = 0.08
        self.communication_start_radius = 0.45
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
        self.pulses: List[Dict[str, object]] = []
        self.previous_resource_positions: set[tuple[int, int]] = set()
        self.resource_spawn_state: Dict[tuple[int, int], int] = {}
        self.resource_collect_state: Dict[tuple[int, int], int] = {}

        plt.ion()
        self.figure_bg = "#2e2e2e"
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

        self.communication_pulse_patches: List[Circle] = []
        for _ in range(max(6, len(self.agent_values) * 6)):
            pulse = Circle(
                (-10.0, -10.0),
                radius=0.0,
                fill=False,
                edgecolor="#facc15",
                linewidth=1.8,
                alpha=0.0,
                zorder=3.85,
                visible=False,
            )
            self.ax_grid.add_patch(pulse)
            self.communication_pulse_patches.append(pulse)

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
            Rectangle((0, 0), 1, 1, facecolor=self.arena_palette["obstacle"], edgecolor=self.arena_palette["obstacle_edge"], label="Obstacle"),
        ]
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
        self.episode_entropy: List[float] = []
        self.episode_cooperation: List[float] = []
        self.agent_resource_totals: Dict[str, List[float]] = {
            self._agent_id(agent_value): [] for agent_value in self.agent_values
        }
        self.resource_line, = self.ax_plot.plot(
            [],
            [],
            color="#16a34a",
            linewidth=2.2,
            label="Resources Avg (50)",
        )
        self.ax_plot.set_xlim(0, max(1, num_episodes))
        self.ax_plot.set_ylim(0, max(1.0, max_possible_reward))
        self._style_ui_axis(self.ax_plot, "Resources Collected Over Time")
        self.ax_plot.set_xlabel("Episode")
        self.ax_plot.set_ylabel("Resources Collected (Smoothed)")
        self.ax_plot.grid(True, color="#9ca3af", alpha=0.2, linewidth=0.8)
        self.ax_plot.legend(loc="upper right", framealpha=0.9, fontsize=9)

        self.entropy_line, = self.ax_ppo.plot(
            [],
            [],
            color="#7c3aed",
            linewidth=2.2,
            label="Policy Entropy Avg (50)",
        )
        self.ax_ppo.set_xlim(0, max(1, num_episodes))
        self._style_ui_axis(self.ax_ppo, "Exploration Behaviour")
        self.ax_ppo.set_xlabel("Episode")
        self.ax_ppo.set_ylabel("Policy Entropy")
        self.ax_ppo.grid(True, color="#9ca3af", alpha=0.2, linewidth=0.8)
        self.ax_ppo.set_ylim(0.0, 2.0)
        self.ax_ppo.legend(loc="upper right", framealpha=0.9, fontsize=9)

        self.contribution_agent_ids = [self._agent_id(agent_value) for agent_value in self.agent_values]
        self.contribution_labels = [f"Agent {self._agent_index(agent_value)}" for agent_value in self.agent_values]
        contribution_colours = [self._agent_colour(agent_value) for agent_value in self.agent_values]
        contribution_x = np.arange(len(self.contribution_agent_ids))
        self.contribution_bars = self.ax_opt.bar(
            contribution_x,
            np.zeros(len(self.contribution_agent_ids)),
            color=contribution_colours,
            width=0.62,
            label=self.contribution_labels,
        )
        self.contribution_value_labels = [
            self.ax_opt.text(x_pos, 0.0, "0.00", ha="center", va="bottom", fontsize=8.2, color="#111111")
            for x_pos in contribution_x
        ]
        self._style_ui_axis(self.ax_opt, "Average Contribution Per Agent")
        self.ax_opt.set_xlabel("Agent")
        self.ax_opt.set_ylabel("Average Resources Collected")
        self.ax_opt.set_xticks(contribution_x)
        self.ax_opt.set_xticklabels(self.contribution_labels)
        self.ax_opt.set_ylim(0.0, 1.0)
        self.ax_opt.grid(True, color="#9ca3af", alpha=0.2, linewidth=0.8)
        self.ax_opt.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

        self.cooperation_line, = self.ax_coop.plot(
            [],
            [],
            color="#0f766e",
            linewidth=2.2,
            label="Cooperation Avg (50)",
        )
        self.ax_coop.set_xlim(0, max(1, num_episodes))
        self.ax_coop.set_ylim(0.0, 1.0)
        self._style_ui_axis(self.ax_coop, "Cooperation Score")
        self.ax_coop.set_xlabel("Episode")
        self.ax_coop.set_ylabel("Cooperation Level")
        self.ax_coop.grid(True, color="#9ca3af", alpha=0.2, linewidth=0.8)
        self.ax_coop.legend(loc="upper right", framealpha=0.9, fontsize=9)

        self.fig.subplots_adjust(left=0.035, right=0.95, top=0.91, bottom=0.18)
        self._shift_environment_axis_right()
        self._update_env(initial_grid)

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
        plt.pause(FRAME_TIME)

    def refresh(self, render_delay: float) -> None:
        self.fig.canvas.draw_idle()
        plt.pause(FRAME_TIME)

    def close(self) -> None:
        plt.ioff()
        plt.show()



for _name in (
    "_update_env",
    "_update_game_mode_visuals",
    "_update_resource_animation_state",
    "_apply_resource_animations",
    "_append_trail_position",
    "_update_trail_patches",
    "_update_facing",
    "_update_perception",
    "_compute_forward_rays",
    "_update_communication_visuals",
    "reset_communication_visuals",
):
    setattr(LiveEpisodeRenderer, _name, getattr(renderer_draw, _name))

for _name in (
    "_shift_environment_axis_right",
    "_add_figure_background",
    "_style_ui_axis",
    "_setup_hud_axis",
    "setup_team_comparison_view",
    "update_team_comparison_view",
    "update_hud",
    "update_learning_plot",
    "update_ppo_plot",
    "_append_cooperation_metrics",
    "_update_agent_contribution_bars",
    "_update_cooperation_lines",
    "_moving_avg",
    "setup_live_layout",
):
    setattr(LiveEpisodeRenderer, _name, getattr(renderer_ui, _name))


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
    mode: str = "train",
) -> List[Dict]:
    mode = mode.lower().strip()

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
    episode_summaries: List[Dict] = []
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
        n_comms = 0

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

            flags = getattr(env, "just_communicated", {})
            active_agents = [agent for agent in env.agents if flags.get(agent, False)]
            communication_events: Optional[List[Dict[str, object]]] = None
            if comm_layer is not None:
                previews: Dict[str, str] = {}
                if active_agents:
                    sent_messages = comm_layer.update_messages_after_step(active_agents)
                    for sender_id, msg in sent_messages.items():
                        previews[sender_id] = f"{int(msg[1]):+d},{int(msg[2]):+d},{int(msg[3])}"
                obs = comm_layer.build_augment_observation(raw_next_obs)
                communication_events = build_communication_events_from_flags(env, previews=previews)
            else:
                obs = raw_next_obs
                communication_events = build_communication_events_from_flags(env)
            n_comms += len(communication_events)

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
            "comms": n_comms,
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
        n_comms = episode_result["comms"]

        episode_summary = {
            "episode_num": episode,
            "resources_collected": resources.copy(),
            "total_reward": total_shaped_reward,
            "steps": episode_result["steps"],
            "ppo_metrics": ppo_metrics,
            "comms": n_comms,
        }
        episode_summaries.append(episode_summary)
        renderer.update_learning_plot(total_shaped_reward, total_resources, resources)
        renderer.update_ppo_plot(ppo_metrics)
        renderer.refresh(render_delay)

        print(f"Episode {episode + 1} | reward: {total_shaped_reward:.2f} | comms: {n_comms}")

    if mode == "live":
        for demo_idx in range(final_demo_episodes):
            run_live_episode(
                episode_seed=num_episodes + demo_idx,
                render_episode=True,
                episode_label=num_episodes,
                train_policy=False,
            )
    renderer.close()

    return episode_summaries
