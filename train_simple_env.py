from __future__ import annotations

import argparse

from train_headless import train_headless


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO in an easier curriculum environment.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default="checkpoints/simple_env")
    parser.add_argument("--metrics-path", default="results/simple_env_training_metrics.json")
    parser.add_argument("--csv-path", default="results/simple_env_training_metrics.csv")
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
