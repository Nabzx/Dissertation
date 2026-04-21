from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


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
    warm = np.array([46, 46, 46], dtype=float) / 255.0
    shadow = np.array([46, 46, 46], dtype=float) / 255.0
    base[:] = warm
    base = base * (1.0 - 0.38 * gradient[..., None]) + shadow * (0.38 * gradient[..., None])
    bg_axis.imshow(base, aspect="auto", extent=(0, 1, 0, 1), origin="lower")

def _style_ui_axis(self, axis, title: str) -> None:
    axis.set_facecolor("#ffffff")
    axis.set_title(title, color="#f5f5f5", fontsize=11.5, fontweight="bold", pad=9)
    axis.tick_params(colors="#f5f5f5", labelsize=8.5)
    axis.xaxis.label.set_color("#f5f5f5")
    axis.yaxis.label.set_color("#f5f5f5")
    axis.xaxis.label.set_size(9.5)
    axis.yaxis.label.set_size(9.5)
    for spine in axis.spines.values():
        spine.set_color("#bbbbbb")

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
        0.1, 0.842, "", color="#f8fafc", fontsize=9.0, fontweight="bold", va="top"
    )
    self.hud_episode_text = self.ax_hud.text(
        0.1, 0.802, "", color="#cbd5e1", fontsize=7.9, va="top"
    )
    self.hud_step_text = self.ax_hud.text(
        0.1, 0.768, "", color="#cbd5e1", fontsize=7.9, va="top"
    )
    self.hud_reward_text = self.ax_hud.text(
        0.1, 0.734, "", color="#cbd5e1", fontsize=7.9, va="top"
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

def setup_team_comparison_view(self, team_specs: List[Dict[str, object]]) -> None:
    self.comparison_team_specs = team_specs
    panel_bg = "#1e1e1e"
    panel_edge = "#4b5563"
    muted_text = "#a3a3a3"
    main_text = "#f5f5f5"
    for axis in (self.ax_plot, self.ax_ppo, self.ax_opt, self.ax_coop):
        axis.clear()
        axis.set_facecolor(panel_bg)
        axis.tick_params(colors=main_text, labelsize=8.5)
        for spine in axis.spines.values():
            spine.set_color(panel_edge)
            spine.set_linewidth(1.0)

    self.comparison_metric_values = {}
    for axis, spec in zip((self.ax_plot, self.ax_ppo), team_specs):
        team_key = str(spec["key"])
        team_color = str(spec["color"])
        box_x = 0.035
        box_y = 0.055
        box_width = 0.93
        box_height = 0.88
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.add_patch(
            FancyBboxPatch(
                (box_x, box_y),
                box_width,
                box_height,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                transform=axis.transAxes,
                facecolor=panel_bg,
                edgecolor=panel_edge,
                linewidth=1.0,
                zorder=0.2,
            )
        )
        axis.text(
            box_x + 0.055,
            box_y + box_height * 0.915,
            str(spec["title"]),
            color=team_color,
            fontsize=11.2,
            fontweight="bold",
            va="top",
            zorder=2,
        )
        self.comparison_metric_values[team_key] = {}
        text_x = box_x + 0.065
        base_y = box_y + box_height * 0.72
        metric_specs = [
            ("total_resources", "TOTAL", base_y, base_y - 0.07),
            ("average_per_agent", "AVG / AGENT", base_y - 0.16, base_y - 0.23),
            ("efficiency", "EFFICIENCY", base_y - 0.32, base_y - 0.39),
        ]
        for metric_key, label, label_y, value_y in metric_specs:
            axis.text(
                text_x,
                label_y,
                label,
                color=main_text,
                alpha=0.6,
                fontsize=9,
                fontweight="medium",
                ha="left",
                va="top",
                zorder=2,
            )
            self.comparison_metric_values[team_key][metric_key] = axis.text(
                text_x,
                value_y,
                "0",
                color=main_text,
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="top",
                zorder=2,
            )

    labels = [str(spec["label"]) for spec in team_specs]
    colors = [str(spec["color"]) for spec in team_specs]
    x_vals = np.arange(len(team_specs))
    self.comparison_bars = self.ax_opt.bar(
        x_vals,
        np.zeros(len(team_specs)),
        color=colors,
        width=0.56,
        label=labels,
    )
    self.comparison_bar_labels = [
        self.ax_opt.text(x_val, 0.0, "0.00", ha="center", va="bottom", fontsize=8.2, color=main_text)
        for x_val in x_vals
    ]
    self.ax_opt.set_title("Average Resources Per Episode", color=main_text, fontsize=11.2, fontweight="bold", pad=11)
    self.ax_opt.set_xticks(x_vals)
    self.ax_opt.set_xticklabels(labels, color=main_text)
    self.ax_opt.set_ylabel("Average Resources", color=main_text)
    self.ax_opt.tick_params(axis="x", pad=7)
    self.ax_opt.set_ylim(0.0, 1.0)
    self.ax_opt.grid(True, axis="y", color="#6b7280", alpha=0.28, linewidth=0.8)
    legend = self.ax_opt.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    legend.get_frame().set_facecolor("#262626")
    legend.get_frame().set_edgecolor(panel_edge)
    for text in legend.get_texts():
        text.set_color(main_text)

    self.ax_coop.set_xlim(0, 1)
    self.ax_coop.set_ylim(0, 1)
    self.ax_coop.set_xticks([])
    self.ax_coop.set_yticks([])
    self.ax_coop.add_patch(
        FancyBboxPatch(
            (0.035, 0.055),
            0.93,
            0.88,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            transform=self.ax_coop.transAxes,
            facecolor=panel_bg,
            edgecolor=panel_edge,
            linewidth=1.0,
            zorder=0.2,
        )
    )
    self.ax_coop.set_title("Comparison Summary", color=main_text, fontsize=11.2, fontweight="bold", pad=11)
    summary_box_x = 0.035
    summary_box_y = 0.055
    summary_box_height = 0.88
    summary_text_x = summary_box_x + 0.04
    summary_top_y = summary_box_y + summary_box_height - 0.05
    self.comparison_summary_layout = {
        "top_y": summary_top_y,
        "line_gap": 0.08,
        "text_x": summary_text_x,
    }
    self.comparison_summary_values = {}
    for key in ("win_header", "pretrained_wins", "scratch_wins", "performance_header", "performance", "gap_header", "gap"):
        is_header = key.endswith("_header")
        self.comparison_summary_values[key] = self.ax_coop.text(
            summary_text_x,
            summary_top_y,
            "",
            color=muted_text if is_header else main_text,
            alpha=0.5 if is_header else 1.0,
            fontsize=9 if is_header else (9.4 if key == "performance" else 10.8),
            fontweight="bold",
            va="top",
            ha="left",
            linespacing=1.45,
            wrap=True,
            clip_on=True,
            zorder=2,
        )

def update_team_comparison_view(
    self,
    team_metrics: Dict[str, Dict[str, float]],
    win_counts: Dict[str, int],
    summary_text: str,
) -> None:
    if not hasattr(self, "comparison_team_specs"):
        return

    avg_values = []
    for spec in self.comparison_team_specs:
        team_key = str(spec["key"])
        metrics = team_metrics.get(team_key, {})
        total = float(metrics.get("total_resources", 0.0))
        average = float(metrics.get("average_per_agent", 0.0))
        efficiency = float(metrics.get("efficiency", 0.0))
        avg_per_episode = float(metrics.get("average_per_episode", 0.0))
        avg_values.append(avg_per_episode)
        self.comparison_metric_values[team_key]["total_resources"].set_text(f"{total:.0f}")
        self.comparison_metric_values[team_key]["average_per_agent"].set_text(f"{average:.2f}")
        self.comparison_metric_values[team_key]["efficiency"].set_text(f"{efficiency:.3f}")

    top = max(avg_values, default=0.0)
    y_top = max(1.0, top * 1.22)
    self.ax_opt.set_ylim(0.0, y_top)
    for bar, label, value in zip(self.comparison_bars, self.comparison_bar_labels, avg_values):
        bar.set_height(value)
        label.set_position((bar.get_x() + bar.get_width() / 2.0, value + y_top * 0.025))
        label.set_text(f"{value:.2f}")

    summary_lines = summary_text.splitlines()
    performance = summary_lines[0] if summary_lines else summary_text
    if performance == "Pretrained agents outperforming scratch agents":
        performance = "Pretrained agents\noutperforming scratch\nagents"
    elif performance == "Scratch agents outperforming pretrained agents":
        performance = "Scratch agents\noutperforming pretrained\nagents"
    elif performance == "Pretrained and scratch agents are currently tied":
        performance = "Pretrained and scratch\nagents are currently tied"
    gap = summary_lines[1].replace("Performance gap: ", "") if len(summary_lines) > 1 else ""

    layout = self.comparison_summary_layout
    y = float(layout["top_y"])
    line_gap = float(layout["line_gap"])
    text_x = float(layout["text_x"])
    summary_items = []
    summary_items.append(("win_header", "WIN COUNT", y))
    y -= line_gap
    summary_items.append(("pretrained_wins", f"Pretrained: {int(win_counts.get('pretrained', 0))}", y))
    y -= line_gap
    summary_items.append(("scratch_wins", f"Scratch: {int(win_counts.get('scratch', 0))}", y))
    y -= line_gap * 1.2
    summary_items.append(("performance_header", "PERFORMANCE", y))
    y -= line_gap
    summary_items.append(("performance", performance, y))
    num_lines = performance.count("\n") + 1
    y -= line_gap * num_lines * 1.1
    summary_items.append(("gap_header", "GAP", y))
    y -= line_gap
    summary_items.append(("gap", gap, y))
    for key, value, y_pos in summary_items:
        self.comparison_summary_values[key].set_position((text_x, y_pos))
        self.comparison_summary_values[key].set_text(value)

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
    winner = game_metrics.get("winner") if isinstance(game_metrics, dict) else None
    avg_time = game_metrics.get("average_time_to_flag") if isinstance(game_metrics, dict) else None
    flag_position = game_metrics.get("flag_position") if isinstance(game_metrics, dict) else None
    phase_label = str(phase).upper()
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
        efficiency = resources / max(1, int(step))
        position = info.get("position", ("-", "-"))
        facing = info.get("facing", "unknown")
        comm_status = info.get("communication", "offline")
        agent_type = info.get("agent_type", "unknown")
        status = info.get("status", "Active")
        action = info.get("recent_action", "n/a")
        lines = [
            f"Type: {agent_type}",
            f"Resources: {resources} | Efficiency: {efficiency:.3f}",
            f"Pos: {position}",
            f"Act: {action}",
        ]
        if agent_id in distances:
            lines.append(f"Flag dist: {distances[agent_id]}")
        if agent_id in wins:
            lines.append(f"Wins: {wins[agent_id]}")
        self.hud_agent_text[agent_id].set_text("\n".join(lines))

def update_learning_plot(
    self,
    total_reward: float,
    total_resources: float,
    per_agent_resources: Optional[Dict[str, float]] = None,
) -> None:
    self.episode_rewards.append(float(total_reward))
    self.episode_resources.append(float(total_resources))
    self._append_cooperation_metrics(per_agent_resources)
    self._update_agent_contribution_bars(per_agent_resources)
    if len(self.episode_resources) % self.plot_update_every != 0:
        return

    x_vals = list(range(len(self.episode_resources)))
    smoothed_resources = self._moving_avg(self.episode_resources, 50)

    plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
    plot_resources = smoothed_resources[:: self.plot_downsample] or smoothed_resources[-1:]

    self.resource_line.set_data(plot_x, plot_resources)
    self._update_cooperation_lines(x_vals)

def update_ppo_plot(self, ppo_metrics: Optional[Dict[str, float]]) -> None:
    if not ppo_metrics:
        ppo_metrics = {}
    self.episode_entropy.append(float(ppo_metrics.get("entropy", 0.0)))

    entropy_x_vals = list(range(len(self.episode_entropy)))
    smoothed_entropy = self._moving_avg(self.episode_entropy, 50)
    plot_entropy_x = entropy_x_vals[:: self.plot_downsample] or entropy_x_vals[-1:]
    plot_entropy = smoothed_entropy[:: self.plot_downsample] or smoothed_entropy[-1:]
    self.entropy_line.set_data(plot_entropy_x, plot_entropy)

def _append_cooperation_metrics(self, per_agent_resources: Optional[Dict[str, float]]) -> None:
    if not per_agent_resources:
        self.episode_cooperation.append(0.0)
        return

    values = np.array(list(per_agent_resources.values()), dtype=float)
    mean = float(np.mean(values)) if len(values) > 0 else 0.0
    std = float(np.std(values)) if len(values) > 0 else 0.0
    if mean > 0.0:
        cooperation_score = 1.0 - (std / mean)
    else:
        cooperation_score = 0.0

    self.episode_cooperation.append(float(np.clip(cooperation_score, 0.0, 1.0)))

def _update_agent_contribution_bars(self, per_agent_resources: Optional[Dict[str, float]]) -> None:
    resources = per_agent_resources or {}
    averages = []
    for agent_id in self.contribution_agent_ids:
        self.agent_resource_totals[agent_id].append(float(resources.get(agent_id, 0.0)))
        averages.append(float(np.mean(self.agent_resource_totals[agent_id])))

    top = max(averages, default=0.0)
    self.ax_opt.set_ylim(0.0, max(1.0, top * 1.18))
    for bar, label, value in zip(self.contribution_bars, self.contribution_value_labels, averages):
        bar.set_height(value)
        label.set_position((bar.get_x() + bar.get_width() / 2.0, value))
        label.set_text(f"{value:.2f}")

def _update_cooperation_lines(self, x_vals: List[int]) -> None:
    smoothed_cooperation = self._moving_avg(self.episode_cooperation, 50)

    plot_x = x_vals[:: self.plot_downsample] or x_vals[-1:]
    plot_cooperation = smoothed_cooperation[:: self.plot_downsample] or smoothed_cooperation[-1:]

    self.cooperation_line.set_data(plot_x, plot_cooperation)

def _moving_avg(self, values: List[float], window: int) -> List[float]:
    return [
        float(np.mean(values[max(0, idx - window + 1) : idx + 1]))
        for idx in range(len(values))
    ]

def setup_live_layout(self) -> None:
    self.fig.subplots_adjust(bottom=0.24)
    self._shift_environment_axis_right()



def draw_hud(self, hud_state=None):
    return update_hud(self, hud_state)


def draw_team_boxes(self, team_specs):
    return setup_team_comparison_view(self, team_specs)


def draw_summary_box(self, team_metrics, win_counts, summary):
    return update_team_comparison_view(self, team_metrics, win_counts, summary)
