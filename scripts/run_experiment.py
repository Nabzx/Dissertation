from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make sure project root is on path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# main simulation + analysis utilities
from train.run_simulation import run_batch_simulation
from train.generate_preliminary_results import generate_preliminary_results


def main() -> None:
    # --- CLI args (controls experiment setup) ---
    parser = argparse.ArgumentParser(
        description="Run gridworld experiments with agent and reward settings."
    )

    # which type of agent to use
    parser.add_argument(
        "--agent_type",
        type=str,
        default="heuristic",
        choices=("heuristic", "ppo"),
        help="Agent policy: heuristic or PPO (default: heuristic)",
    )

    # reward shaping scheme
    parser.add_argument(
        "--reward_scheme",
        type=str,
        default="selfish",
        choices=("selfish", "mixed", "fully_cooperative"),
        help="Reward shaping scheme (default: selfish)",
    )

    # number of episodes to simulate
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of episodes to run (default: 20)",
    )

    # optional communication between agents
    parser.add_argument(
        "--communication",
        action="store_true",
        help="Enable bandwidth-limited inter-agent communication (default: off)",
    )

    # number of agents in the environment
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="Number of agents in the gridworld (default: 4)",
    )

    args = parser.parse_args()

    # normalise inputs
    agent_type = args.agent_type.lower()
    reward_scheme = args.reward_scheme.lower()

    # used to name output folders
    run_tag = f"{agent_type}_{reward_scheme}" + ("_comm" if args.communication else "")

    # --- print config summary ---
    print("=" * 60)
    print("Experiment")
    print("=" * 60)
    print(f"  agent_type:     {agent_type}")
    print(f"  reward_scheme:  {reward_scheme}")
    print(f"  communication:  {args.communication}")
    print(f"  num_agents:     {args.num_agents}")
    print(f"  num_episodes:   {args.num_episodes}")
    print(f"  logs/results:   logs/{run_tag}/  |  results/{run_tag}/")
    print("=" * 60)
    print()

    # --- run the actual simulation ---
    episode_data = run_batch_simulation(
        num_episodes=args.num_episodes,
        agent_type=agent_type,
        reward_scheme=reward_scheme,
        use_communication=args.communication,
        num_agents=args.num_agents,
    )

    # --- generate plots + summary files ---
    generate_preliminary_results(
        logs_dir=f"logs/{run_tag}/episodes",
        results_dir=f"results/{run_tag}",
    )

    # --- quick summary stats ---
    total_resources = [
        sum(ep["resources_collected"].values())
        for ep in episode_data
    ]
    mean_resources = float(np.mean(total_resources)) if total_resources else 0.0

    shaped = [ep.get("total_shaped_reward", 0.0) for ep in episode_data]
    mean_shaped = float(np.mean(shaped)) if shaped else 0.0

    # --- print final summary ---
    print()
    print("=" * 60)
    print("Experiment summary")
    print("=" * 60)
    print(f"  agent_type:              {agent_type}")
    print(f"  reward_scheme:          {reward_scheme}")
    print(f"  episodes:                {len(episode_data)}")
    print(f"  mean total resources:    {mean_resources:.2f} (all agents per episode)")
    print(f"  mean total shaped reward: {mean_shaped:.2f} (sum over agents & steps per episode)")
    print("=" * 60)


# --- entry point ---
if __name__ == "__main__":
    main()