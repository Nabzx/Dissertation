from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from agents.communication import CommunicationLayer


def run_post_training_analysis(env, agents, run_name):
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _run_final_episode(env, agents)
    _save_final_frame(data["final_grid"], out_dir / "final_episode.png")
    _save_trajectory_plot(data["trajectories"], env.grid_size, out_dir / "trajectory_plot.png")
    _save_heatmap(data["heatmap"], out_dir / "heatmap.png")
    _save_reward_curve(run_name, out_dir / "reward_curve.png")
    _save_resource_bars(data["resources"], out_dir / "resources_per_agent.png")


def _run_final_episode(env, agents) -> Dict:
    raw_obs, _ = env.reset(seed=999999)
    obs = raw_obs
    comm = _make_comm_layer(env, agents, raw_obs)
    if comm is not None:
        obs = comm.build_augment_observation(raw_obs)

    traj = {agent: [env.agent_positions[agent]] for agent in env.agents}
    rewards: List[Dict[str, float]] = []
    comms: List[Dict[str, object]] = []

    for step in range(env.max_steps):
        actions = {}
        for agent in env.agents:
            actions[agent] = _select_action(agents, agent, obs[agent])

        raw_obs, reward, term, trunc, _ = env.step(actions)
        rewards.append({agent: float(value) for agent, value in reward.items()})

        active = [agent for agent in env.agents if env.just_communicated.get(agent, False)]
        if active:
            comms.extend({"step": step + 1, "agent": agent, "pos": env.agent_positions[agent]} for agent in active)

        if comm is not None:
            comm.update_messages_after_step(active)
            obs = comm.build_augment_observation(raw_obs)
        else:
            obs = raw_obs

        for agent in env.agents:
            traj[agent].append(env.agent_positions[agent])

        if all(term.values()) or all(trunc.values()):
            break

    return {
        "trajectories": traj,
        "resources": env.get_resources_collected(),
        "rewards": rewards,
        "communication_events": comms,
        "heatmap": _combined_heatmap(env.get_heatmaps()),
        "final_grid": env.get_final_grid(),
    }


def _make_comm_layer(env, agents, raw_obs):
    if not hasattr(agents, "obs_dim"):
        return None

    base_dim = int(np.prod(raw_obs[env.agents[0]].shape))
    if int(agents.obs_dim) <= base_dim:
        return None

    comm = CommunicationLayer(env)
    comm.reset()
    return comm


def _select_action(agents, agent_id: str, obs: np.ndarray) -> int:
    if hasattr(agents, "select_action"):
        flat = obs if obs.ndim == 1 else obs.flatten()
        action, _, _ = agents.select_action(flat.astype(np.float32))
        return int(action)

    return int(agents[agent_id].get_action(obs))


def _combined_heatmap(heatmaps: Dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for heatmap in heatmaps.values():
        total = heatmap.copy() if total is None else total + heatmap
    return total if total is not None else np.zeros((1, 1), dtype=np.int32)


def _save_final_frame(grid: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, origin="upper", interpolation="nearest")
    ax.set_title("Final episode")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_trajectory_plot(traj: Dict[str, List[Tuple[int, int]]], grid_size: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green", "tab:brown"]

    for idx, (agent, positions) in enumerate(traj.items()):
        if not positions:
            continue
        xs = [pos[1] for pos in positions]
        ys = [pos[0] for pos in positions]
        colour = colours[idx % len(colours)]
        ax.plot(xs, ys, color=colour, linewidth=2, label=agent)
        ax.scatter(xs[0], ys[0], color=colour, marker="o", edgecolors="black")
        ax.scatter(xs[-1], ys[-1], color=colour, marker="x")

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_title("Agent trajectories")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_heatmap(heatmap: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap="hot", origin="upper", interpolation="nearest")
    ax.set_title("Visitation heatmap")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_reward_curve(run_name: str, path: Path) -> None:
    rows = _load_metrics(run_name)
    if not rows:
        return

    episodes = [int(row.get("episode", idx + 1)) for idx, row in enumerate(rows)]
    rewards = [float(row.get("total_reward", 0.0)) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, linewidth=1.5)
    ax.set_title("Reward curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_metrics(run_name: str) -> List[Dict]:
    result_dir = Path("results") / run_name
    paths = [
        result_dir / "headless_training_metrics.json",
        result_dir / "simple_env_training_metrics.json",
    ]
    paths.extend(sorted(result_dir.glob("*training_metrics*.json")))

    for path in paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return []


def _save_resource_bars(resources: Dict[str, int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    agents = list(resources.keys())
    counts = [int(resources[agent]) for agent in agents]
    ax.bar(agents, counts, color="tab:green")
    ax.set_title("Resources collected")
    ax.set_ylabel("Resources")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
