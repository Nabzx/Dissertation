"""Command-line runner for gridworld experiments."""

from __future__ import annotations

import argparse

import numpy as np

from train.run_simulation import run_batch_simulation
from train.generate_preliminary_results import generate_preliminary_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gridworld experiments with agent and reward settings.")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="heuristic",
        choices=("heuristic", "ppo"),
        help="Agent policy: heuristic or PPO (default: heuristic)",
    )
    parser.add_argument(
        "--reward_scheme",
        type=str,
        default="selfish",
        choices=("selfish", "mixed", "fully_cooperative"),
        help="Reward shaping scheme (default: selfish)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of episodes to run (default: 20)",
    )
    parser.add_argument(
        "--communication",
        action="store_true",
        help="Enable bandwidth-limited inter-agent communication (default: off)",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="Number of agents in the gridworld (default: 4)",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="default",
        choices=("default", "capture_flag"),
        help="Optional minigame mode: default or capture_flag",
    )
    args = parser.parse_args()

    agent_type = args.agent_type.lower()
    reward_scheme = args.reward_scheme.lower()
    game_mode = args.game_mode.lower()
    mode_tag = "" if game_mode == "default" else f"_{game_mode}"
    run_tag = f"{agent_type}_{reward_scheme}{mode_tag}" + ("_comm" if args.communication else "")

    print("=" * 60)
    print("Experiment")
    print("=" * 60)
    print(f"  agent_type:     {agent_type}")
    print(f"  reward_scheme:  {reward_scheme}")
    print(f"  communication:  {args.communication}")
    print(f"  game_mode:      {game_mode}")
    print(f"  num_agents:     {args.num_agents}")
    print(f"  num_episodes:   {args.num_episodes}")
    print(f"  logs/results:   logs/{run_tag}/  |  results/{run_tag}/")
    print("=" * 60)
    print()

    episode_data = run_batch_simulation(
        num_episodes=args.num_episodes,
        agent_type=agent_type,
        reward_scheme=reward_scheme,
        use_communication=args.communication,
        num_agents=args.num_agents,
        game_mode=game_mode,
    )

    generate_preliminary_results(
        logs_dir=f"logs/{run_tag}/episodes",
        results_dir=f"results/{run_tag}",
    )

    # Summary statistics
    total_resources = [sum(ep["resources_collected"].values()) for ep in episode_data]
    mean_resources = float(np.mean(total_resources)) if total_resources else 0.0

    shaped = [ep.get("total_shaped_reward", 0.0) for ep in episode_data]
    mean_shaped = float(np.mean(shaped)) if shaped else 0.0

    print()
    print("=" * 60)
    print("Experiment summary")
    print("=" * 60)
    print(f"  agent_type:              {agent_type}")
    print(f"  reward_scheme:          {reward_scheme}")
    print(f"  game_mode:              {game_mode}")
    print(f"  episodes:                {len(episode_data)}")
    print(f"  mean total resources:    {mean_resources:.2f} (all agents per episode)")
    print(f"  mean total shaped reward: {mean_shaped:.2f} (sum over agents & steps per episode)")
    print("=" * 60)


if __name__ == "__main__":
    main()
