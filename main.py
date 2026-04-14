"""
Main entry point for the multi-agent resource-scarcity simulation.

This script runs simulations and generates preliminary results.
"""

from train.run_simulation import run_batch_simulation
from train.generate_preliminary_results import generate_preliminary_results


def main():
    """
    Main function to run simulations and generate results.
    """
    print("=" * 60)
    print("Multi-Agent Resource-Scarcity Simulation")
    print("=" * 60)
    print()

    # Configuration
    num_episodes = 20
    grid_size = 15
    num_resources = 10
    max_steps = 200
    agent_type = "heuristic"  # change to "ppo" to enable RL
    reward_scheme = "selfish"  # selfish | mixed | fully_cooperative
    run_tag = f"{agent_type}_{reward_scheme}"

    print("Configuration:")
    print(f"  Grid Size: {grid_size}x{grid_size}")
    print(f"  Number of Resources: {num_resources}")
    print(f"  Max Steps per Episode: {max_steps}")
    print(f"  Number of Episodes: {num_episodes}")
    print(f"  Agent Type: {agent_type}")
    print(f"  Reward Scheme: {reward_scheme}")
    print()

    # Run simulations
    print("Step 1: Running simulations...")
    print("-" * 60)
    episode_data = run_batch_simulation(
        num_episodes=num_episodes,
        grid_size=grid_size,
        num_resources=num_resources,
        max_steps=max_steps,
        save_screenshots=True,
        save_heatmaps=True,
        agent_type=agent_type,
        reward_scheme=reward_scheme,
    )
    print()

    # Generate preliminary results
    print("Step 2: Generating preliminary results...")
    print("-" * 60)
    generate_preliminary_results(
        logs_dir=f"logs/{run_tag}/episodes",
        results_dir=f"results/{run_tag}",
    )
    print()

    print("=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - logs/{run_tag}/episodes/*.json (episode data)")
    print(f"  - logs/{run_tag}/screenshots/*.png (grid screenshots)")
    print(f"  - logs/{run_tag}/heatmaps/*.png (per-episode movement heatmaps)")
    print(f"  - logs/{run_tag}/heatmaps/agent_0_heatmap.png (aggregated heatmap)")
    print(f"  - logs/{run_tag}/heatmaps/agent_1_heatmap.png (aggregated heatmap)")
    print(f"  - logs/{run_tag}/resources/*.png (resource distributions)")
    print(f"  - results/{run_tag}/preliminary_results_summary.json")
    print(f"  - results/{run_tag}/resource_distribution_heatmap.png")
    print(f"  - results/{run_tag}/screenshot_montage.png")
    print(f"  - results/{run_tag}/trajectories/episode_0_trajectory.png")
    print(f"  - results/{run_tag}/trajectories/episode_1_trajectory.png")
    print(f"  - results/{run_tag}/reward_curves/reward_curve.png")
    print()
    print(f"Check the results/{run_tag}/ directory for analysis outputs.")


if __name__ == "__main__":
    main()

