import json
import os
from typing import Dict, List, Optional

import numpy as np

from env.gridworld_env import GridWorldEnv
from agents.heuristic_agent import HeuristicAgent
from env.utils import (
    save_grid_screenshot,
    save_heatmap,
    plot_resource_distribution,
    save_trajectory_plot,
)

from testing.ppo_agent import PPOAgent
from testing.rewards import apply_reward_scheme
from testing.communication import CommunicationLayer
from minigames import CaptureFlagGame, GameModeWrapper


def run_episode(
    env: GridWorldEnv,
    agents: Optional[Dict[str, HeuristicAgent]],
    episode_num: int,
    save_screenshots: bool = True,
    save_heatmaps: bool = True,
    save_artifacts: bool = True,
    agent_type: str = "heuristic",
    ppo_agent: Optional[PPOAgent] = None,
    logs_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    render: bool = False,
    train_policy: bool = True,
) -> Dict:
    raw_obs, infos = env.reset(seed=episode_num)

    agent_type = agent_type.lower()
    reward_scheme = reward_scheme.lower()
    logs_dir = logs_dir or "logs"
    results_dir = results_dir or "results"

    total_spawned = env.num_resources
    cumulative_collected: Dict[str, int] = {a: 0 for a in env.agents}
    total_shaped_reward = 0.0
    game_mode_active = getattr(env, "game_mode", None) is not None

    if agent_type == "ppo" and ppo_agent is None:
        raise ValueError("agent_type='ppo' requires a ppo_agent instance.")

    comm_layer: Optional[CommunicationLayer] = None
    if agent_type == "ppo" and use_communication:
        comm_layer = CommunicationLayer(env)
        comm_layer.reset()
        obs = comm_layer.build_augment_observation(raw_obs)
    else:
        obs = raw_obs

    if agent_type == "heuristic":
        for agent in agents.values():
            agent.reset()
    elif train_policy:
        ppo_agent.reset_buffer()

    trajectory = []
    agent_trajectories = {agent: [] for agent in env.agents}
    grid_sequence: List[np.ndarray] = []
    step_count = 0

    if render:
        grid_sequence.append(env.grid.copy())

    for agent_id in env.agents:
        pos = env.agent_positions[agent_id]
        agent_trajectories[agent_id].append(pos)

    while True:
        actions = {}
        step_log_probs: Dict[str, float] = {}
        step_values: Dict[str, float] = {}
        step_flat_obs: Dict[str, np.ndarray] = {}

        if agent_type == "heuristic":
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(obs[agent_id])
        else:
            for agent_id in env.agents:
                agent_obs = obs[agent_id]
                flat_obs = agent_obs if agent_obs.ndim == 1 else agent_obs.flatten()
                action, log_prob, value = ppo_agent.select_action(flat_obs)
                actions[agent_id] = int(action)
                step_log_probs[agent_id] = float(log_prob)
                step_values[agent_id] = float(value)
                step_flat_obs[agent_id] = flat_obs

        raw_next_obs, rewards, terminations, truncations, infos = env.step(actions)

        if game_mode_active:
            shaped_rewards = rewards.copy()
        else:
            for agent_id, r in rewards.items():
                if r > 0.0:
                    cumulative_collected[agent_id] += 1

            shaped_rewards = apply_reward_scheme(
                scheme=reward_scheme,
                raw_rewards=rewards,
                cumulative_collected=cumulative_collected,
                total_spawned=total_spawned,
                alpha=0.5,
            )
        total_shaped_reward += sum(shaped_rewards.values())

        if agent_type == "ppo" and train_policy:
            for agent_id in env.agents:
                done = bool(terminations[agent_id] or truncations[agent_id])
                ppo_agent.store_transition(
                    obs=step_flat_obs[agent_id],
                    action=actions[agent_id],
                    log_prob=step_log_probs[agent_id],
                    reward=float(shaped_rewards[agent_id]),
                    done=done,
                    value=step_values[agent_id],
                    trajectory_id=agent_id,
                )

        if comm_layer is not None:
            comm_layer.update_messages_after_step()
            obs = comm_layer.build_augment_observation(raw_next_obs)
        else:
            obs = raw_next_obs

        if render:
            grid_sequence.append(env.grid.copy())

        for agent_id in env.agents:
            pos = env.agent_positions[agent_id]
            agent_trajectories[agent_id].append(pos)

        trajectory.append(
            {
                "step": step_count,
                "actions": actions.copy(),
                "rewards": shaped_rewards.copy(),
                "raw_rewards": rewards.copy(),
            }
        )

        step_count += 1

        if all(terminations.values()) or all(truncations.values()):
            break

    ppo_metrics = None
    if agent_type == "ppo" and train_policy:
        try:
            ppo_metrics = ppo_agent.update(last_value=0.0, last_done=True)
        except RuntimeError as exc:
            print(f"[ppo] Update skipped: {exc}")

    heatmaps = env.get_heatmaps()
    resources_collected = env.get_resources_collected()
    final_grid = env.get_final_grid()
    initial_resources = env.get_initial_resource_positions()

    screenshot_path = None
    if save_artifacts and save_screenshots:
        screenshot_path = os.path.join(logs_dir, "screenshots", f"episode_{episode_num:04d}_final.png")
        save_grid_screenshot(final_grid, screenshot_path, title=f"Episode {episode_num} - Final State")

    heatmap_paths = {}
    if save_artifacts and save_heatmaps:
        for agent_id, heatmap in heatmaps.items():
            heatmap_path = os.path.join(logs_dir, "heatmaps", f"episode_{episode_num:04d}_{agent_id}.png")
            save_heatmap(heatmap, heatmap_path, agent_id, title=f"Episode {episode_num} - {agent_id}")
            heatmap_paths[agent_id] = heatmap_path

    resource_dist_path = None
    if save_artifacts:
        resource_dist_path = os.path.join(logs_dir, "resources", f"episode_{episode_num:04d}_distribution.png")
        plot_resource_distribution(initial_resources, resource_dist_path, grid_size=env.grid_size)

    if save_artifacts and episode_num in [0, 1]:
        trajectory_path = os.path.join(results_dir, "trajectories", f"episode_{episode_num}_trajectory.png")
        save_trajectory_plot(agent_trajectories, env.grid_size, trajectory_path)

    episode_data = {
        "episode_num": episode_num,
        "reward_scheme": reward_scheme,
        "game_mode": env.game_mode.__class__.__name__ if game_mode_active else "default",
        "game_metrics": env.get_metrics() if game_mode_active else {},
        "render_info": env.get_render_info() if game_mode_active else {},
        "total_steps": step_count,
        "total_shaped_reward": total_shaped_reward,
        "resources_collected": resources_collected.copy(),
        "total_resources_spawned": len(initial_resources),
        "initial_resource_positions": initial_resources,
        "agent_survival": {
            agent: resources_collected.get(agent, 0) >= 1
            for agent in env.agents
        },
        "all_survived": all(resources_collected.get(agent, 0) >= 1 for agent in env.agents),
        "any_survived": any(resources_collected.get(agent, 0) >= 1 for agent in env.agents),
        "agent_0_survived": resources_collected.get("agent_0", 0) >= 1,
        "agent_1_survived": resources_collected.get("agent_1", 0) >= 1,
        "both_survived": all(
            resources_collected.get(agent, 0) >= 1
            for agent in ["agent_0", "agent_1"]
            if agent in resources_collected
        ),
        "screenshot_path": screenshot_path,
        "heatmap_paths": heatmap_paths,
        "resource_dist_path": resource_dist_path,
        "heatmaps": {agent_id: heatmap.tolist() for agent_id, heatmap in heatmaps.items()},
        "trajectories": {agent_id: positions for agent_id, positions in agent_trajectories.items()},
        "ppo_metrics": ppo_metrics,
    }

    if save_artifacts:
        episode_json_path = os.path.join(logs_dir, "episodes", f"episode_{episode_num:04d}.json")
        os.makedirs(os.path.dirname(episode_json_path), exist_ok=True)
        with open(episode_json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

    if render:
        episode_data["grid_sequence"] = grid_sequence

    return episode_data


def run_batch_simulation(
    num_episodes: int = 20,
    grid_size: int = 25,
    num_agents: int = 4,
    num_resources: int = 25,
    num_obstacles: int = 45,
    max_steps: int = 250,
    save_screenshots: bool = True,
    save_heatmaps: bool = True,
    agent_type: str = "heuristic",
    reward_scheme: str = "selfish",
    use_communication: bool = False,
    game_mode: str = "default",
) -> List[Dict]:
    env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )

    game_mode = game_mode.lower()
    if game_mode == "capture_flag":
        env = GameModeWrapper(env, CaptureFlagGame(env))
    elif game_mode not in ("default", "none"):
        raise ValueError(f"Unknown game_mode '{game_mode}'. Expected 'default' or 'capture_flag'.")

    agent_type = agent_type.lower()
    reward_scheme = reward_scheme.lower()
    mode_tag = "" if game_mode in ("default", "none") else f"_{game_mode}"
    run_tag = f"{agent_type}_{reward_scheme}{mode_tag}" + ("_comm" if use_communication else "")

    logs_dir = f"logs/{run_tag}"
    results_dir = f"results/{run_tag}"
    os.makedirs(os.path.join(logs_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "resources"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "reward_curves"), exist_ok=True)

    ppo_agent: Optional[PPOAgent] = None
    if agent_type == "heuristic":
        agents: Optional[Dict[str, HeuristicAgent]] = {
            agent: HeuristicAgent(agent) for agent in env.agents
        }
    elif agent_type == "ppo":
        agents = None
        obs_size = env.observation_spaces[env.agents[0]].shape
        obs_dim = int(np.prod(obs_size))
        if use_communication:
            obs_dim += int(CommunicationLayer(env).config.max_ints)
        action_dim = 5
        ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim)
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'. Expected 'heuristic' or 'ppo'.")

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
            reward_scheme=reward_scheme,
            use_communication=use_communication,
        )
        all_episode_data.append(episode_data)

        resources = episode_data["resources_collected"]
        resource_text = ", ".join(f"{agent}: {count}" for agent, count in resources.items())
        print(f"{resource_text}, Steps: {episode_data['total_steps']}")

    print(f"\nCompleted {num_episodes} episodes!")
    print(f"Results saved to {logs_dir}/ and {results_dir}/")

    return all_episode_data


if __name__ == "__main__":
    episode_data = run_batch_simulation(
        num_episodes=20,
        grid_size=15,
        num_resources=10,
        max_steps=200,
        save_screenshots=True,
        save_heatmaps=True,
        reward_scheme="selfish",
    )
