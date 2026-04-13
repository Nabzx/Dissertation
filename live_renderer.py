"""
Persistent real-time renderer for continuous PPO training episodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from env.gridworld_env import GridWorldEnv
from testing.communication import CommunicationLayer
from testing.ppo_agent import PPOAgent
from testing.rewards import apply_reward_scheme


class LiveEpisodeRenderer:
    """
    Keeps a single matplotlib window alive while multiple episodes run.
    """

    def __init__(self, initial_grid: np.ndarray, num_episodes: int, max_possible_reward: float):
        cmap = ListedColormap(["white", "green", "blue", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], wspace=0.35, hspace=0.35)
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.ax_plot = self.fig.add_subplot(gs[0, 1])
        self.ax_ppo = self.fig.add_subplot(gs[1, 1])
        self.im = self.ax_grid.imshow(
            initial_grid,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        rows, cols = initial_grid.shape
        self.ax_grid.set_title("Environment")
        self.ax_grid.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        self.ax_grid.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
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
        self.raw_line, = self.ax_plot.plot([], [], color="tab:blue", label="Reward (raw)")
        self.smooth_line, = self.ax_plot.plot([], [], color="tab:orange", label="Moving Avg (20)")
        self.resource_line, = self.ax_plot.plot([], [], color="tab:green", label="Resources collected")
        self.ax_plot.set_xlim(0, max(1, num_episodes))
        self.ax_plot.set_ylim(0, max(1.0, max_possible_reward))
        self.ax_plot.set_title("Learning Progress")
        self.ax_plot.set_xlabel("Episode")
        self.ax_plot.set_ylabel("Reward")
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.legend(loc="upper left")

        self.policy_line, = self.ax_ppo.plot([], [], color="tab:red", label="Policy Loss (PPO)")
        self.ax_entropy = self.ax_ppo.twinx()
        self.entropy_line, = self.ax_entropy.plot([], [], color="tab:purple", label="Entropy (Exploration)")
        self.ax_ppo.set_xlim(0, max(1, num_episodes))
        self.ax_ppo.set_title("PPO Training Dynamics")
        self.ax_ppo.set_xlabel("Episode")
        self.ax_ppo.set_ylabel("Policy Loss (PPO)", color="tab:red")
        self.ax_ppo.tick_params(axis="y", labelcolor="tab:red")
        self.ax_ppo.grid(True, alpha=0.3)
        self.ax_entropy.set_ylabel("Entropy (Exploration)", color="tab:purple")
        self.ax_entropy.tick_params(axis="y", labelcolor="tab:purple")
        ppo_handles = [self.policy_line, self.entropy_line]
        self.ax_ppo.legend(ppo_handles, [line.get_label() for line in ppo_handles], loc="upper right")

        self.fig.subplots_adjust(bottom=0.18)

    def update(self, grid: np.ndarray, episode: int, step: int, render_delay: float) -> None:
        self.im.set_data(grid)
        self.text.set_text(f"Episode {episode} | Step {step}")
        self.fig.canvas.draw_idle()
        plt.pause(render_delay)

    def update_learning_plot(self, total_reward: float, total_resources: float) -> None:
        self.episode_rewards.append(float(total_reward))
        self.episode_resources.append(float(total_resources))
        x_vals = list(range(len(self.episode_rewards)))
        self.raw_line.set_data(x_vals, self.episode_rewards)
        self.resource_line.set_data(x_vals, self.episode_resources)

        window = 20
        if self.episode_rewards:
            smoothed = [
                float(np.mean(self.episode_rewards[max(0, idx - window + 1) : idx + 1]))
                for idx in range(len(self.episode_rewards))
            ]
            self.smooth_line.set_data(x_vals, smoothed)

        self.ax_plot.relim()
        self.ax_plot.autoscale_view()

    def update_ppo_plot(self, ppo_metrics: Dict[str, float]) -> None:
        self.policy_losses.append(float(ppo_metrics.get("policy_loss", 0.0)))
        self.value_losses.append(float(ppo_metrics.get("value_loss", 0.0)))
        self.entropy_values.append(float(ppo_metrics.get("entropy", 0.0)))

        x_vals = list(range(len(self.policy_losses)))
        self.policy_line.set_data(x_vals, self.policy_losses)
        self.entropy_line.set_data(x_vals, self.entropy_values)

        self.ax_ppo.relim()
        self.ax_ppo.autoscale_view()
        self.ax_entropy.relim()
        self.ax_entropy.autoscale_view()

    def close(self) -> None:
        plt.ioff()
        plt.show()


def run_live_training(
    num_episodes: int = 50,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 15,
    num_resources: int = 10,
    max_steps: int = 100,
    render_delay: float = 0.01,
) -> List[Dict]:
    """
    Run many PPO episodes with a single persistent live renderer window.
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
    )

    episode_summaries: List[Dict] = []
    recent_rewards: List[float] = []
    recent_resources: List[int] = []

    for episode in range(num_episodes):
        raw_obs, _ = env.reset(seed=episode)

        comm_layer: Optional[CommunicationLayer] = None
        if use_communication:
            comm_layer = CommunicationLayer(env)
            comm_layer.reset()
            obs = comm_layer.build_augment_observation(raw_obs)
        else:
            obs = raw_obs

        ppo_agent.reset_buffer()
        cumulative_collected: Dict[str, int] = {agent: 0 for agent in env.agents}
        total_shaped_reward = 0.0

        renderer.update(env.grid.copy(), episode, 0, render_delay)

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

            renderer.update(env.grid.copy(), episode, step + 1, render_delay)

            if all(done_flags.values()):
                break

        ppo_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        try:
            ppo_metrics = ppo_agent.update(last_value=0.0, last_done=True)
        except RuntimeError as exc:
            print(f"[live] PPO update skipped: {exc}")

        resources = env.get_resources_collected()
        total_resources = int(sum(resources.values()))

        episode_summary = {
            "episode_num": episode,
            "resources_collected": resources.copy(),
            "total_reward": total_shaped_reward,
            "steps": env.step_count,
            "ppo_metrics": ppo_metrics,
        }
        episode_summaries.append(episode_summary)
        recent_rewards.append(total_shaped_reward)
        recent_resources.append(total_resources)
        renderer.update_learning_plot(total_shaped_reward, total_resources)
        renderer.update_ppo_plot(ppo_metrics)
        plt.pause(render_delay)

        print(
            f"Episode {episode + 1}: "
            f"resources collected={resources}, total reward={total_shaped_reward:.2f}"
        )

        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(recent_rewards[-10:]))
            avg_resources = float(np.mean(recent_resources[-10:]))
            print(
                f"Average over episodes {episode - 8}-{episode + 1}: "
                f"resources={avg_resources:.2f}, reward={avg_reward:.2f}"
            )

    renderer.close()
    return episode_summaries
