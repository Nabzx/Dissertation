from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_headless import train_headless


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO in an easier curriculum environment.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=45)
    parser.add_argument("--num-obstacles", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num_episodes = args.num_episodes
    run_name = f"run_{num_episodes}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/{run_name}"
    if args.metrics_path is None:
        args.metrics_path = f"results/{run_name}/simple_env_training_metrics.json"
    if args.csv_path is None:
        args.csv_path = f"results/{run_name}/simple_env_training_metrics.csv"

    latest_checkpoint_path = f"{args.checkpoint_dir.rstrip('/')}/ppo_latest.pt"
    print("Simple-environment PPO pretraining")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Latest checkpoint:    {latest_checkpoint_path}")
    print(f"Metrics JSON:         {args.metrics_path}")
    print(f"Metrics CSV:          {args.csv_path}")

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
    print(f"Checkpoint written successfully: {latest_checkpoint_path}")
