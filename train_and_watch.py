"""Continuous PPO training with occasional episode animation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from env.gridworld_env import GridWorldEnv
from testing.communication import CommunicationLayer
from testing.ppo_agent import PPOAgent
from train.run_simulation import run_episode
from visualize_episode import animate_episode


def train_and_watch(
    num_episodes: int = 5000,
    render_every: int = 100,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 25,
    num_resources: int = 25,
    num_obstacles: int = 45,
    max_steps: int = 250,
    render_interval: int = 50,
    save_gif_dir: Optional[str] = None,
) -> List[Dict]:
    """Train one persistent PPO agent and optionally animate sampled episodes."""
    env = GridWorldEnv(
        grid_size=grid_size,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )

    obs_shape = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_shape))
    if use_communication:
        obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)

    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device="cpu")

    episode_summaries: List[Dict] = []
    recent_rewards: List[float] = []
    recent_resources: List[int] = []

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        should_render = render_every > 0 and episode % render_every == 0

        episode_data = run_episode(
            env=env,
            agents=None,
            episode_num=episode,
            save_screenshots=False,
            save_heatmaps=False,
            save_artifacts=False,
            agent_type="ppo",
            ppo_agent=ppo_agent,
            reward_scheme=reward_scheme,
            use_communication=use_communication,
            render=should_render,
        )

        resources = episode_data["resources_collected"]
        total_resources = int(sum(resources.values()))
        total_reward = float(episode_data["total_shaped_reward"])
        ppo_metrics = episode_data.get("ppo_metrics") or {}

        episode_summaries.append(episode_data)
        recent_rewards.append(total_reward)
        recent_resources.append(total_resources)

        print(
            f"Episode {episode + 1}: "
            f"resources collected={resources}, total reward={total_reward:.2f}, "
            f"policy_loss={ppo_metrics.get('policy_loss', 0.0):.4f}, "
            f"value_loss={ppo_metrics.get('value_loss', 0.0):.4f}, "
            f"entropy={ppo_metrics.get('entropy', 0.0):.4f}"
        )

        if should_render:
            save_path = None
            if save_gif_dir is not None:
                save_path = f"{save_gif_dir}/episode_{episode + 1:04d}.gif"
            animate_episode(
                episode_data["grid_sequence"],
                save_path=save_path,
                interval=render_interval,
                block=False,
            )

        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(recent_rewards[-10:]))
            avg_resources = float(np.mean(recent_resources[-10:]))
            print(
                f"Average over episodes {episode - 8}-{episode + 1}: "
                f"resources={avg_resources:.2f}, reward={avg_reward:.2f}"
            )

    return episode_summaries


if __name__ == "__main__":
    train_and_watch(num_episodes=50, render_every=10)
