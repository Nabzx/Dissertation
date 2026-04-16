"""
Headless PPO training pipeline with checkpointing.

This script intentionally does not import or use the live renderer. It trains
one persistent PPOAgent across many episodes, writes lightweight metrics, and
saves checkpoints that can later be loaded by demo_trained_agent.py.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from env.gridworld_env import GridWorldEnv
from testing.communication import CommunicationLayer
from testing.ppo_agent import PPOAgent, TORCH_AVAILABLE
from train.run_simulation import run_episode


def train_headless(
    num_episodes: int = 5000,
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "results/headless_training_metrics.json",
    checkpoint_every: int = 500,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 15,
    num_resources: int = 10,
    max_steps: int = 100,
    device: str = "cpu",
) -> List[Dict]:
    """
    Train PPO without rendering and periodically save model checkpoints.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Headless PPO training requires PyTorch.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    env = GridWorldEnv(grid_size=grid_size, num_resources=num_resources, max_steps=max_steps)
    obs_dim = int(np.prod(env.observation_spaces[env.agents[0]].shape))
    if use_communication:
        obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)

    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device=device)
    metrics: List[Dict] = []

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
            render=False,
            train_policy=True,
        )

        resources = episode_data["resources_collected"]
        total_resources = int(sum(resources.values()))
        total_reward = float(episode_data["total_shaped_reward"])
        ppo_metrics = episode_data.get("ppo_metrics") or {}

        row = {
            "episode": episode + 1,
            "total_reward": total_reward,
            "total_resources": total_resources,
            "resources_collected": resources,
            "steps": int(episode_data["total_steps"]),
            "ppo_metrics": ppo_metrics,
        }
        metrics.append(row)

        if (episode + 1) % 10 == 0:
            recent = metrics[-10:]
            avg_reward = float(np.mean([item["total_reward"] for item in recent]))
            avg_resources = float(np.mean([item["total_resources"] for item in recent]))
            print(
                f"Episode {episode + 1}/{num_episodes}: "
                f"avg_reward={avg_reward:.2f}, avg_resources={avg_resources:.2f}, "
                f"policy_loss={ppo_metrics.get('policy_loss', 0.0):.4f}, "
                f"value_loss={ppo_metrics.get('value_loss', 0.0):.4f}, "
                f"entropy={ppo_metrics.get('entropy', 0.0):.4f}"
            )

        should_checkpoint = checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0
        if should_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_episode_{episode + 1:06d}.pt")
            ppo_agent.save(checkpoint_path)
            ppo_agent.save(os.path.join(checkpoint_dir, "ppo_latest.pt"))
            _write_metrics(metrics_path, metrics)
            print(f"Saved checkpoint: {checkpoint_path}")

    final_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    ppo_agent.save(final_path)
    ppo_agent.save(os.path.join(checkpoint_dir, "ppo_latest.pt"))
    _write_metrics(metrics_path, metrics)
    print(f"Saved final checkpoint: {final_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def _write_metrics(path: str, metrics: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO headlessly with checkpoints.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-path", default="results/headless_training_metrics.json")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--num-resources", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_headless(
        num_episodes=args.num_episodes,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        checkpoint_every=args.checkpoint_every,
        reward_scheme=args.reward_scheme,
        use_communication=args.communication,
        grid_size=args.grid_size,
        num_resources=args.num_resources,
        max_steps=args.max_steps,
        device=args.device,
    )
