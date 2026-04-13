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

    def __init__(self, initial_grid: np.ndarray):
        cmap = ListedColormap(["white", "green", "blue", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(
            initial_grid,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        rows, cols = initial_grid.shape
        self.ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

    def update(self, grid: np.ndarray, episode: int, step: int, render_delay: float) -> None:
        self.im.set_data(grid)
        self.ax.set_title(f"Episode {episode} | Step {step}")
        self.fig.canvas.draw_idle()
        plt.pause(render_delay)

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

    obs_shape = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_shape))
    if use_communication:
        obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)
    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device="cpu")

    raw_obs, _ = env.reset(seed=0)
    renderer = LiveEpisodeRenderer(env.grid.copy())

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

        try:
            ppo_agent.update(last_value=0.0, last_done=True)
        except RuntimeError as exc:
            print(f"[live] PPO update skipped: {exc}")

        resources = env.get_resources_collected()
        total_resources = int(sum(resources.values()))

        episode_summary = {
            "episode_num": episode,
            "resources_collected": resources.copy(),
            "total_reward": total_shaped_reward,
            "steps": env.step_count,
        }
        episode_summaries.append(episode_summary)
        recent_rewards.append(total_shaped_reward)
        recent_resources.append(total_resources)

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
