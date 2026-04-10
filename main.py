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

    print("Configuration:")
    print(f"  Grid Size: {grid_size}x{grid_size}")
    print(f"  Number of Resources: {num_resources}")
    print(f"  Max Steps per Episode: {max_steps}")
    print(f"  Number of Episodes: {num_episodes}")
    print(f"  Agent Type: {agent_type}")
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
    )
    print()

    # Generate preliminary results
    print("Step 2: Generating preliminary results...")
    print("-" * 60)
    generate_preliminary_results(
        logs_dir=f"logs/{agent_type}/episodes",
        results_dir=f"results/{agent_type}",
    )
    print()

    print("=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - logs/{agent_type}/episodes/*.json (episode data)")
    print(f"  - logs/{agent_type}/screenshots/*.png (grid screenshots)")
    print(f"  - logs/{agent_type}/heatmaps/*.png (per-episode movement heatmaps)")
    print(f"  - logs/{agent_type}/heatmaps/agent_0_heatmap.png (aggregated heatmap)")
    print(f"  - logs/{agent_type}/heatmaps/agent_1_heatmap.png (aggregated heatmap)")
    print(f"  - logs/{agent_type}/resources/*.png (resource distributions)")
    print(f"  - results/{agent_type}/preliminary_results_summary.json")
    print(f"  - results/{agent_type}/resource_distribution_heatmap.png")
    print(f"  - results/{agent_type}/screenshot_montage.png")
    print(f"  - results/{agent_type}/trajectories/episode_0_trajectory.png")
    print(f"  - results/{agent_type}/trajectories/episode_1_trajectory.png")
    print(f"  - results/{agent_type}/reward_curves/reward_curve.png")
    print()
    print(f"Check the results/{agent_type}/ directory for analysis outputs.")


if __name__ == "__main__":
    main()

