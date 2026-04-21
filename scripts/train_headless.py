from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.gridworld_env import GridWorldEnv
from agents.communication import CommunicationLayer
from agents.ppo_agent import PPOAgent, TORCH_AVAILABLE
from train.run_simulation import run_episode


def train_headless(
    num_episodes: int = 5000,
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "results/headless_training_metrics.json",
    csv_path: str = "results/headless_training_metrics.csv",
    checkpoint_every: int = 500,
    smoothing_window: int = 50,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    grid_size: int = 25,
    num_agents: int = 4,
    num_resources: int = 25,
    num_obstacles: int = 45,
    max_steps: int = 250,
    device: str = "cpu",
) -> List[Dict]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Headless PPO training requires PyTorch.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    for output_path in [metrics_path, csv_path]:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )
    obs_dim = int(np.prod(env.observation_spaces[env.agents[0]].shape))
    if use_communication:
        obs_dim += int(CommunicationLayer(env).config.max_ints)
    action_dim = int(env.action_spaces[env.agents[0]].n)

    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device=device)
    metrics: List[Dict] = []
    _make_csv(csv_path)

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
        reward_ma = _moving_avg_value(metrics, "total_reward", total_reward, smoothing_window)
        resources_ma = _moving_avg_value(metrics, "total_resources", total_resources, smoothing_window)
        entropy_ma = _moving_avg_metric(metrics, "entropy", ppo_metrics.get("entropy", 0.0), smoothing_window)
        policy_loss_ma = _moving_avg_metric(
            metrics,
            "policy_loss",
            ppo_metrics.get("policy_loss", 0.0),
            smoothing_window,
        )
        value_loss_ma = _moving_avg_metric(
            metrics,
            "value_loss",
            ppo_metrics.get("value_loss", 0.0),
            smoothing_window,
        )

        row = {
            "episode": episode + 1,
            "total_reward": total_reward,
            "total_resources": total_resources,
            "resources_collected": resources,
            "steps": int(episode_data["total_steps"]),
            "reward_ma": reward_ma,
            "resources_ma": resources_ma,
            "entropy_ma": entropy_ma,
            "policy_loss_ma": policy_loss_ma,
            "value_loss_ma": value_loss_ma,
            "ppo_metrics": ppo_metrics,
        }
        metrics.append(row)
        _append_csv_row(csv_path, row)

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
    print(f"Saved per-episode CSV: {csv_path}")
    return metrics


def _write_metrics(path: str, metrics: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _moving_avg_value(
    previous_rows: List[Dict],
    key: str,
    current_value: float,
    window: int,
) -> float:
    values = [float(row[key]) for row in previous_rows[-max(0, window - 1) :]]
    values.append(float(current_value))
    return float(np.mean(values)) if values else 0.0


def _moving_avg_metric(
    previous_rows: List[Dict],
    metric_key: str,
    current_value: float,
    window: int,
) -> float:
    values = [
        float(row.get("ppo_metrics", {}).get(metric_key, 0.0))
        for row in previous_rows[-max(0, window - 1) :]
    ]
    values.append(float(current_value))
    return float(np.mean(values)) if values else 0.0


def _csv_fields() -> List[str]:
    return [
        "episode",
        "total_reward",
        "reward_ma",
        "total_resources",
        "resources_ma",
        "resources_collected_json",
        "steps",
        "policy_loss",
        "policy_loss_ma",
        "value_loss",
        "value_loss_ma",
        "entropy",
        "entropy_ma",
        "mean_reward",
        "approx_kl",
        "clip_fraction",
    ]


def _make_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields())
        writer.writeheader()


def _append_csv_row(path: str, row: Dict) -> None:
    ppo_metrics = row.get("ppo_metrics") or {}
    resources = row.get("resources_collected") or {}
    flat_row = {
        "episode": row["episode"],
        "total_reward": row["total_reward"],
        "reward_ma": row["reward_ma"],
        "total_resources": row["total_resources"],
        "resources_ma": row["resources_ma"],
        "resources_collected_json": json.dumps(resources, sort_keys=True),
        "steps": row["steps"],
        "policy_loss": ppo_metrics.get("policy_loss", 0.0),
        "policy_loss_ma": row["policy_loss_ma"],
        "value_loss": ppo_metrics.get("value_loss", 0.0),
        "value_loss_ma": row["value_loss_ma"],
        "entropy": ppo_metrics.get("entropy", 0.0),
        "entropy_ma": row["entropy_ma"],
        "mean_reward": ppo_metrics.get("mean_reward", 0.0),
        "approx_kl": ppo_metrics.get("approx_kl", 0.0),
        "clip_fraction": ppo_metrics.get("clip_fraction", 0.0),
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields())
        writer.writerow(flat_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO headlessly with checkpoints.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--metrics-path", default="results/headless_training_metrics.json")
    parser.add_argument("--csv-path", default="results/headless_training_metrics.csv")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=25)
    parser.add_argument("--num-obstacles", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_headless(
        num_episodes=args.num_episodes,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        csv_path=args.csv_path,
        checkpoint_every=args.checkpoint_every,
        smoothing_window=args.smoothing_window,
        reward_scheme=args.reward_scheme,
        use_communication=args.communication,
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_resources=args.num_resources,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
        device=args.device,
    )
