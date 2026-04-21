from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.gridworld_env import GridWorldEnv
from agents.communication import CommunicationLayer
from agents.ppo_agent import PPOAgent, TORCH_AVAILABLE
from train.run_simulation import run_episode


def evaluate_policy(
    label: str,
    ppo_agent: PPOAgent,
    num_episodes: int,
    reward_scheme: str,
    use_communication: bool,
    grid_size: int,
    num_agents: int,
    num_resources: int,
    num_obstacles: int,
    max_steps: int,
) -> List[Dict]:
    env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )
    rows: List[Dict] = []

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
            train_policy=False,
        )
        resources = episode_data["resources_collected"]
        total_resources = int(sum(resources.values()))
        survived = {
            agent: int(count > 0)
            for agent, count in resources.items()
        }
        rows.append(
            {
                "policy": label,
                "episode": episode + 1,
                "total_reward": float(episode_data["total_shaped_reward"]),
                "total_resources": total_resources,
                "resources_collected": resources,
                "all_survived": all(survived.values()) if survived else False,
                "any_survived": any(survived.values()) if survived else False,
                "steps": int(episode_data["total_steps"]),
            }
        )

    return rows


def summarize(rows: List[Dict]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for label in sorted({row["policy"] for row in rows}):
        subset = [row for row in rows if row["policy"] == label]
        rewards = [float(row["total_reward"]) for row in subset]
        resources = [float(row["total_resources"]) for row in subset]
        summary[label] = {
            "episodes": len(subset),
            "mean_reward": mean(rewards) if rewards else 0.0,
            "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_resources": mean(resources) if resources else 0.0,
            "std_resources": pstdev(resources) if len(resources) > 1 else 0.0,
            "all_survival_rate": _rate(row["all_survived"] for row in subset),
            "any_survival_rate": _rate(row["any_survived"] for row in subset),
        }
    return summary


def _rate(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def write_outputs(rows: List[Dict], summary: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pretrained_vs_scratch.csv")
    json_path = os.path.join(output_dir, "pretrained_vs_scratch.json")
    summary_path = os.path.join(output_dir, "pretrained_vs_scratch_summary.json")

    flat_rows = []
    for row in rows:
        flat = row.copy()
        flat["resources_collected"] = json.dumps(row["resources_collected"], sort_keys=True)
        flat_rows.append(flat)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-episode comparison CSV: {csv_path}")
    print(f"Saved per-episode comparison JSON: {json_path}")
    print(f"Saved comparison summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pretrained PPO with scratch PPO in the main arena.")
    parser.add_argument("--checkpoint", default="checkpoints/run_10000/ppo_latest.pt")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", default="results/run_10000/pretrained_comparison")
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=25)
    parser.add_argument("--num-obstacles", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Comparison requires PyTorch.")

    args = parse_args()
    print(f"Loading pretrained checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print()
        print("Pretrained checkpoint not found.")
        print("Run pretraining first to generate the checkpoint:")
        print("python3 train/train_simple_env.py --num-episodes 5000")
        print()
        print(f"Expected checkpoint path: {args.checkpoint}")
        sys.exit(1)

    probe_env = GridWorldEnv(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_resources=args.num_resources,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
    )
    obs_dim = int(np.prod(probe_env.observation_spaces[probe_env.agents[0]].shape))
    if args.communication:
        obs_dim += int(CommunicationLayer(probe_env).config.max_ints)
    action_dim = int(probe_env.action_spaces[probe_env.agents[0]].n)

    pretrained = PPOAgent.load(args.checkpoint, device=args.device)
    if pretrained.obs_dim != obs_dim or pretrained.n_actions != action_dim:
        raise ValueError(
            "Checkpoint shape does not match evaluation arena. "
            f"Checkpoint=({pretrained.obs_dim}, {pretrained.n_actions}); "
            f"arena=({obs_dim}, {action_dim})."
        )
    pretrained.model.eval()

    scratch = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device=args.device)
    scratch.model.eval()

    rows = []
    rows.extend(
        evaluate_policy(
            "pretrained",
            pretrained,
            args.num_episodes,
            args.reward_scheme,
            args.communication,
            args.grid_size,
            args.num_agents,
            args.num_resources,
            args.num_obstacles,
            args.max_steps,
        )
    )
    rows.extend(
        evaluate_policy(
            "scratch",
            scratch,
            args.num_episodes,
            args.reward_scheme,
            args.communication,
            args.grid_size,
            args.num_agents,
            args.num_resources,
            args.num_obstacles,
            args.max_steps,
        )
    )

    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
