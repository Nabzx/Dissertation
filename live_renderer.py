"""
Persistent real-time renderer for continuous PPO training episodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

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
    ):
        self.smoothing_window = 50
        self.plot_update_every = 10
        self.plot_downsample = 5
        self.max_possible_reward = max_possible_reward
        self.max_resources = max_resources
        self.view_size = view_size
        self.show_perception = show_perception

        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], wspace=0.35, hspace=0.35)
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.ax_plot = self.fig.add_subplot(gs[0, 1])
        self.ax_ppo = self.fig.add_subplot(gs[1, 1])

        rows, cols = initial_grid.shape
        self.grid_rows = rows
        self.grid_cols = cols
        self.ax_grid.set_title("Environment")
        self.ax_grid.set_facecolor("#eef3ec")
        self.ax_grid.set_xlim(0, cols)
        self.ax_grid.set_ylim(rows, 0)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        for spine in self.ax_grid.spines.values():
            spine.set_visible(False)

        # Draw a soft tile field once so the world feels more like a simulation map.
        tile_colors = ("#f6f4ee", "#edf2e8")
        for row in range(rows):
            for col in range(cols):
                tile = Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=tile_colors[(row + col) % 2],
                    edgecolor="#dde5d8",
                    linewidth=0.3,
                    zorder=0,
                )
                self.ax_grid.add_patch(tile)

        self.dim_overlay = Rectangle(
            (0, 0),
            cols,
            rows,
            facecolor="#1f2937",
            edgecolor="none",
            alpha=0.18 if show_perception else 0.0,
            zorder=1,
            visible=show_perception,
        )
        self.ax_grid.add_patch(self.dim_overlay)

        self.perception_patches = {
            2: Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#93c5fd",
                edgecolor="#60a5fa",
                linewidth=1.0,
                alpha=0.22,
                zorder=2.5,
                visible=False,
            ),
            3: Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#fca5a5",
                edgecolor="#f87171",
                linewidth=1.0,
                alpha=0.22,
                zorder=2.5,
                visible=False,
            ),
        }
        for patch in self.perception_patches.values():
            self.ax_grid.add_patch(patch)

        self.resource_patches: List[Circle] = []
        for _ in range(max_resources):
            resource_patch = Circle(
                (-10.0, -10.0),
                radius=0.18,
                facecolor="#4fae68",
                edgecolor="#2f6f41",
                linewidth=1.0,
                zorder=3,
                visible=False,
            )
            self.resource_patches.append(resource_patch)
            self.ax_grid.add_patch(resource_patch)

        self.obstacle_patches: List[Rectangle] = []
        for row in range(rows):
            for col in range(cols):
                obstacle_patch = Rectangle(
                    (col + 0.12, row + 0.12),
                    0.76,
                    0.76,
                    facecolor="#4b5563",
                    edgecolor="#374151",
                    linewidth=0.8,
                    zorder=2,
                    visible=False,
                )
                self.obstacle_patches.append(obstacle_patch)
                self.ax_grid.add_patch(obstacle_patch)

        self.agent_patches = {
            2: Circle(
                (0.5, 0.5),
                radius=0.28,
                facecolor="#3b82f6",
                edgecolor="#1d4ed8",
                linewidth=1.5,
                zorder=4,
                visible=False,
            ),
            3: Circle(
                (0.5, 0.5),
                radius=0.28,
                facecolor="#ef4444",
                edgecolor="#b91c1c",
                linewidth=1.5,
                zorder=4,
                visible=False,
            ),
        }
        for patch in self.agent_patches.values():
            self.ax_grid.add_patch(patch)

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#3b82f6", markeredgecolor="#1d4ed8", markersize=9, label="Agent 0"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#ef4444", markeredgecolor="#b91c1c", markersize=9, label="Agent 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#4fae68", markeredgecolor="#2f6f41", markersize=7, label="Resource"),
            Rectangle((0, 0), 1, 1, facecolor="#4b5563", edgecolor="#374151", label="Obstacle"),
        ]
        if show_perception:
            legend_handles.append(
                Rectangle((0, 0), 1, 1, facecolor="#bfdbfe", edgecolor="#60a5fa", alpha=0.35, label="Perception")
            )
        self.ax_grid.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        self.text = self.ax_grid.text(
            0.5,
            -0.08,
            "",
            transform=self.ax_grid.transAxes,
            ha="center",
            fontsize=12,
            color="black",
            bbox=dict(facecolor="white", alpha=0.8),
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
        self.ax_plot.set_title("Learning Progress")
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
        self.ax_ppo.set_title("PPO Training Dynamics")
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

    def update(
        self,
        grid: np.ndarray,
        episode: int,
        num_episodes: int,
        step: int,
        render_delay: float,
    ) -> None:
        self._update_environment(grid)
        self.text.set_text(f"Episode {episode}/{num_episodes} | Step {step}")
        self.fig.canvas.draw_idle()
        plt.pause(render_delay)

    def _update_environment(self, grid: np.ndarray) -> None:
        self._update_perception(grid)

        obstacle_mask = grid == 4
        for patch, is_obstacle in zip(self.obstacle_patches, obstacle_mask.flatten()):
            patch.set_visible(bool(is_obstacle))

        resource_idx = 0
        for row, col in np.argwhere(grid == 1):
            if resource_idx >= len(self.resource_patches):
                break
            patch = self.resource_patches[resource_idx]
            patch.center = (float(col) + 0.5, float(row) + 0.5)
            patch.set_visible(True)
            resource_idx += 1

        for idx in range(resource_idx, len(self.resource_patches)):
            self.resource_patches[idx].set_visible(False)

        for agent_value, patch in self.agent_patches.items():
            positions = np.argwhere(grid == agent_value)
            if len(positions) == 0:
                patch.set_visible(False)
                continue
            row, col = positions[0]
            patch.center = (float(col) + 0.5, float(row) + 0.5)
            patch.set_visible(True)

    def _update_perception(self, grid: np.ndarray) -> None:
        if not self.show_perception:
            return

        radius = self.view_size // 2
        for agent_value, patch in self.perception_patches.items():
            positions = np.argwhere(grid == agent_value)
            if len(positions) == 0:
                patch.set_visible(False)
                continue

            row, col = positions[0]
            row_start = max(0, int(row) - radius)
            row_end = min(self.grid_rows, int(row) + radius + 1)
            col_start = max(0, int(col) - radius)
            col_end = min(self.grid_cols, int(col) + radius + 1)

            patch.set_xy((col_start, row_start))
            patch.set_width(col_end - col_start)
            patch.set_height(row_end - row_start)
            patch.set_visible(True)

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
    )

    episode_summaries: List[Dict] = []
    recent_rewards: List[float] = []
    recent_resources: List[int] = []

    def run_live_episode(
        episode_seed: int,
        render_episode: bool,
        episode_label: int,
        train_policy: bool = True,
    ) -> Dict:
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
            renderer.update(env.grid.copy(), episode_label, num_episodes, 0, render_delay)

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

            if comm_layer is not None:
                comm_layer.update_messages_after_step()
                obs = comm_layer.build_augment_observation(raw_next_obs)
            else:
                obs = raw_next_obs

            if render_episode:
                renderer.update(env.grid.copy(), episode_label, num_episodes, step + 1, render_delay)

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
