"""
Load a trained PPO checkpoint and render evaluation episodes.

This script does not train or update PPO. It is only for visualizing a saved
policy checkpoint produced by train_headless.py.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from env.gridworld_env import GridWorldEnv
from testing.communication import CommunicationLayer
from testing.ppo_agent import PPOAgent, TORCH_AVAILABLE
from train.run_simulation import run_episode
from visualize_episode import animate_episode


def demo_trained_agent(
    checkpoint_path: str = "checkpoints/ppo_latest.pt",
    num_episodes: int = 5,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 15,
    num_resources: int = 10,
    max_steps: int = 100,
    interval: int = 50,
    save_gif_dir: Optional[str] = None,
    device: str = "cpu",
) -> None:
    """
    Load a trained PPO policy and visualize it without further training.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Demoing a trained PPO checkpoint requires PyTorch.")

    env = GridWorldEnv(grid_size=grid_size, num_resources=num_resources, max_steps=max_steps)
    ppo_agent = PPOAgent.load(checkpoint_path, device=device)
    ppo_agent.model.eval()

    expected_obs_dim = int(np.prod(env.observation_spaces[env.agents[0]].shape))
    if use_communication:
        expected_obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)

    if ppo_agent.obs_dim != expected_obs_dim or ppo_agent.n_actions != action_dim:
        raise ValueError(
            "Checkpoint shape does not match this environment/demo config. "
            f"Checkpoint obs/action=({ppo_agent.obs_dim}, {ppo_agent.n_actions}); "
            f"expected=({expected_obs_dim}, {action_dim})."
        )

    for episode in range(num_episodes):
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
            render=True,
            train_policy=False,
        )

        resources = episode_data["resources_collected"]
        total_reward = float(episode_data["total_shaped_reward"])
        print(
            f"Demo episode {episode + 1}/{num_episodes}: "
            f"resources={resources}, total_reward={total_reward:.2f}"
        )

        save_path = None
        if save_gif_dir is not None:
            os.makedirs(save_gif_dir, exist_ok=True)
            save_path = os.path.join(save_gif_dir, f"trained_episode_{episode + 1:03d}.gif")

        animate_episode(
            episode_data["grid_sequence"],
            save_path=save_path,
            interval=interval,
            block=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained PPO checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_latest.pt")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--num-resources", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--save-gif-dir", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo_trained_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        reward_scheme=args.reward_scheme,
        use_communication=args.communication,
        grid_size=args.grid_size,
        num_resources=args.num_resources,
        max_steps=args.max_steps,
        interval=args.interval,
        save_gif_dir=args.save_gif_dir,
        device=args.device,
    )
