"""
Simulation runner for the grid world environment.

This script runs batches of episodes and collects data for analysis.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional

from env.gridworld_env import GridWorldEnv
from agents.heuristic_agent import HeuristicAgent
from env.utils import (
    save_grid_screenshot,
    save_heatmap,
    plot_resource_distribution,
    save_trajectory_plot,
)

from testing.ppo_agent import PPOAgent


def run_episode(
    env: GridWorldEnv,
    agents: Optional[Dict[str, HeuristicAgent]],
    episode_num: int,
    save_screenshots: bool = True,
    save_heatmaps: bool = True,
    agent_type: str = "heuristic",
    ppo_agent: Optional[PPOAgent] = None,
    logs_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict:
    """
    Run a single episode of the simulation.

    Args:
        env: GridWorldEnv instance
        agents: Dict mapping agent IDs to HeuristicAgent instances
        episode_num: Episode number for logging
        save_screenshots: Whether to save grid screenshots
        save_heatmaps: Whether to save heatmap visualizations
        agent_type: "heuristic" (default) or "ppo"
        ppo_agent: Shared PPOAgent instance when agent_type == "ppo"
        logs_dir: Base logs directory (e.g., "logs/heuristic")
        results_dir: Base results directory (e.g., "results/heuristic")

    Returns:
        Dict containing episode data
    """
    observations, infos = env.reset(seed=episode_num)

    agent_type = agent_type.lower()
    logs_dir = logs_dir or "logs"
    results_dir = results_dir or "results"

    if agent_type == "ppo" and ppo_agent is None:
        raise ValueError("agent_type='ppo' requires a ppo_agent instance.")

    # Reset agents / buffers
    if agent_type == "heuristic":
        for agent in agents.values():
            agent.reset()
    else:
        # Keep PPO updates episode-scoped for simplicity.
        ppo_agent.reset_buffer()

    # Store episode trajectory and agent positions
    trajectory = []
    agent_trajectories = {
        "agent_0": [],
        "agent_1": [],
    }
    step_count = 0

    # Store initial positions
    for agent_id in env.agents:
        pos = env.agent_positions[agent_id]
        agent_trajectories[agent_id].append(pos)

    # Run episode
    while True:
        # Get actions from agents
        actions = {}
        step_log_probs: Dict[str, float] = {}
        step_values: Dict[str, float] = {}
        step_flat_obs: Dict[str, np.ndarray] = {}

        if agent_type == "heuristic":
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(observations[agent_id])
        else:
            for agent_id in env.agents:
                flat_obs = observations[agent_id].flatten()
                action, log_prob, value = ppo_agent.select_action(flat_obs)
                actions[agent_id] = int(action)
                step_log_probs[agent_id] = float(log_prob)
                step_values[agent_id] = float(value)
                step_flat_obs[agent_id] = flat_obs

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store PPO transitions (after env.step, using reward + done).
        if agent_type == "ppo":
            for agent_id in env.agents:
                done = bool(terminations[agent_id] or truncations[agent_id])
                ppo_agent.store_transition(
                    obs=step_flat_obs[agent_id],
                    action=actions[agent_id],
                    log_prob=step_log_probs[agent_id],
                    reward=float(rewards[agent_id]),
                    done=done,
                    value=step_values[agent_id],
                )

        # Store agent positions after step
        for agent_id in env.agents:
            pos = env.agent_positions[agent_id]
            agent_trajectories[agent_id].append(pos)

        # Store step data
        trajectory.append(
            {
                "step": step_count,
                "actions": actions.copy(),
                "rewards": rewards.copy(),
            }
        )

        step_count += 1

        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()):
            break

    # PPO update at the end of the episode.
    if agent_type == "ppo":
        try:
            ppo_agent.update(last_value=0.0, last_done=True)
        except RuntimeError as exc:
            # PyTorch missing or other training-time issue. Keep simulation running.
            print(f"[ppo] Update skipped: {exc}")

    # Collect episode data
    heatmaps = env.get_heatmaps()
    resources_collected = env.get_resources_collected()
    final_grid = env.get_final_grid()
    initial_resources = env.get_initial_resource_positions()

    # Save screenshots
    if save_screenshots:
        screenshot_path = os.path.join(logs_dir, "screenshots", f"episode_{episode_num:04d}_final.png")
        save_grid_screenshot(final_grid, screenshot_path, title=f"Episode {episode_num} - Final State")

    # Save heatmaps
    if save_heatmaps:
        for agent_id, heatmap in heatmaps.items():
            heatmap_path = os.path.join(logs_dir, "heatmaps", f"episode_{episode_num:04d}_{agent_id}.png")
            save_heatmap(heatmap, heatmap_path, agent_id, title=f"Episode {episode_num} - {agent_id}")

    # Save resource distribution
    resource_dist_path = os.path.join(logs_dir, "resources", f"episode_{episode_num:04d}_distribution.png")
    plot_resource_distribution(initial_resources, resource_dist_path, grid_size=env.grid_size)

    # Save trajectory plots for episodes 0 and 1
    if episode_num in [0, 1]:
        trajectory_path = os.path.join(results_dir, "trajectories", f"episode_{episode_num}_trajectory.png")
        save_trajectory_plot(agent_trajectories, env.grid_size, trajectory_path)

    # Create episode summary
    episode_data = {
        "episode_num": episode_num,
        "total_steps": step_count,
        "resources_collected": resources_collected.copy(),
        "total_resources_spawned": len(initial_resources),
        "initial_resource_positions": initial_resources,
        "agent_0_survived": resources_collected["agent_0"] >= 1,
        "agent_1_survived": resources_collected["agent_1"] >= 1,
        "both_survived": resources_collected["agent_0"] >= 1 and resources_collected["agent_1"] >= 1,
        "screenshot_path": screenshot_path if save_screenshots else None,
        "heatmap_paths": {
            agent_id: os.path.join(logs_dir, "heatmaps", f"episode_{episode_num:04d}_{agent_id}.png")
            for agent_id in heatmaps.keys()
        }
        if save_heatmaps
        else {},
        "resource_dist_path": resource_dist_path,
        # Store heatmaps as lists for JSON serialisation
        "heatmaps": {agent_id: heatmap.tolist() for agent_id, heatmap in heatmaps.items()},
        # Store trajectories
        "trajectories": {agent_id: positions for agent_id, positions in agent_trajectories.items()},
    }

    # Save episode JSON
    episode_json_path = os.path.join(logs_dir, "episodes", f"episode_{episode_num:04d}.json")
    os.makedirs(os.path.dirname(episode_json_path), exist_ok=True)
    with open(episode_json_path, "w") as f:
        json.dump(episode_data, f, indent=2)

    return episode_data


def run_batch_simulation(
    num_episodes: int = 20,
    grid_size: int = 15,
    num_resources: int = 10,
    max_steps: int = 200,
    save_screenshots: bool = True,
    save_heatmaps: bool = True,
    agent_type: str = "heuristic",
) -> List[Dict]:
    """
    Run a batch of simulation episodes.

    Args:
        num_episodes: Number of episodes to run
        grid_size: Size of the grid
        num_resources: Number of resources per episode
        max_steps: Maximum steps per episode
        save_screenshots: Whether to save grid screenshots
        save_heatmaps: Whether to save heatmap visualisations

    Returns:
        List of episode data dictionaries
    """
    # Create environment
    env = GridWorldEnv(grid_size=grid_size, num_resources=num_resources, max_steps=max_steps)

    agent_type = agent_type.lower()

    # Agent-type scoped output directories
    logs_dir = f"logs/{agent_type}"
    results_dir = f"results/{agent_type}"
    os.makedirs(os.path.join(logs_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "resources"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "reward_curves"), exist_ok=True)

    # Create agents (heuristic) or PPO policy (shared)
    ppo_agent: Optional[PPOAgent] = None
    if agent_type == "heuristic":
        agents: Optional[Dict[str, HeuristicAgent]] = {
            "agent_0": HeuristicAgent("agent_0"),
            "agent_1": HeuristicAgent("agent_1"),
        }
    elif agent_type == "ppo":
        agents = None
        obs_dim = grid_size * grid_size
        action_dim = 5
        ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim)
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'. Expected 'heuristic' or 'ppo'.")

    # Run episodes
    all_episode_data = []

    print(f"Running {num_episodes} episodes...")
    for episode_num in range(num_episodes):
        print(f"Episode {episode_num + 1}/{num_episodes}...", end=" ")
        episode_data = run_episode(
            env,
            agents,
            episode_num,
            save_screenshots=save_screenshots,
            save_heatmaps=save_heatmaps,
            agent_type=agent_type,
            ppo_agent=ppo_agent,
            logs_dir=logs_dir,
            results_dir=results_dir,
        )
        all_episode_data.append(episode_data)

        # Print summary
        resources = episode_data["resources_collected"]
        print(
            f"Agent 0: {resources['agent_0']} resources, "
            f"Agent 1: {resources['agent_1']} resources, "
            f"Steps: {episode_data['total_steps']}"
        )

    print(f"\nCompleted {num_episodes} episodes!")
    print(f"Results saved to {logs_dir}/ and {results_dir}/")

    return all_episode_data


if __name__ == "__main__":
    # Run a batch of simulations
    episode_data = run_batch_simulation(
        num_episodes=20,
        grid_size=15,
        num_resources=10,
        max_steps=200,
        save_screenshots=True,
        save_heatmaps=True,
    )

