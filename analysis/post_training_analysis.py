from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from agents.communication import CommunicationLayer
from agents.ppo_agent import PPOAgent
from env.gridworld_env import GridWorldEnv


MAX_RESOURCES = 25


def run_post_training_analysis(run_name: str) -> None:
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_training_rows(run_name)
    stats = _build_curves(rows)

    _plot_curve(stats["episodes"], stats["rewards"], "Reward", "Reward", out_dir / "reward_curve.png")
    _plot_curve(
        stats["episodes"],
        stats["efficiency"],
        "Resource efficiency",
        "Efficiency",
        out_dir / "efficiency_curve.png",
    )
    _plot_curve(stats["episodes"], stats["fairness"], "Fairness", "Jain index", out_dir / "fairness_curve.png")
    _plot_curve(
        stats["episodes"],
        stats["cooperation"],
        "Cooperation score",
        "Score",
        out_dir / "cooperation_curve.png",
    )

    final = _run_final_episode(run_name)
    _plot_trajectory(final["trajectories"], final["grid_size"], out_dir / "trajectory.png")
    _plot_heatmap(final["visits"], out_dir / "heatmap.png")
    _plot_final_frame(final["final_grid"], out_dir / "final_frame.png")

    summary = {
        "mean_reward": _mean(stats["rewards"]),
        "mean_resources": _mean(stats["resources"]),
        "mean_efficiency": _mean(stats["efficiency"]),
        "mean_fairness": _mean(stats["fairness"]),
        "mean_cooperation": _mean(stats["cooperation"]),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _load_training_rows(run_name: str) -> List[Dict]:
    path = Path("results") / run_name / "headless_training_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Training metrics not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _build_curves(rows: List[Dict]) -> Dict[str, List[float]]:
    episodes = []
    rewards = []
    resources = []
    efficiency = []
    fairness = []
    cooperation = []
    entropy = []

    for idx, row in enumerate(rows):
        res = row.get("resources_collected", {})
        counts = [float(value) for value in res.values()]
        total_res = float(row.get("total_resources", sum(counts)))
        eff = total_res / max(1, MAX_RESOURCES)
        fair = _jain_index(counts)

        episodes.append(int(row.get("episode", idx + 1)))
        rewards.append(float(row.get("total_reward", 0.0)))
        resources.append(total_res)
        efficiency.append(eff)
        fairness.append(fair)
        cooperation.append(eff * fair)
        entropy.append(float((row.get("ppo_metrics") or {}).get("entropy", 0.0)))

    return {
        "episodes": episodes,
        "rewards": rewards,
        "resources": resources,
        "efficiency": efficiency,
        "fairness": fairness,
        "cooperation": cooperation,
        "entropy": entropy,
    }


def _run_final_episode(run_name: str) -> Dict:
    env = GridWorldEnv()
    agent = PPOAgent.load(str(Path("checkpoints") / run_name / "ppo_latest.pt"), device="cpu")
    raw_obs, _ = env.reset(seed=999999)
    obs = raw_obs

    comm = None
    base_dim = int(np.prod(raw_obs[env.agents[0]].shape))
    if int(agent.obs_dim) > base_dim:
        comm = CommunicationLayer(env)
        comm.reset()
        obs = comm.build_augment_observation(raw_obs)

    trajectories = {agent_id: [env.agent_positions[agent_id]] for agent_id in env.agents}
    visits = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    for pos in env.agent_positions.values():
        visits[pos[0], pos[1]] += 1

    for _ in range(env.max_steps):
        actions = {}
        for agent_id in env.agents:
            flat = obs[agent_id] if obs[agent_id].ndim == 1 else obs[agent_id].flatten()
            action, _, _ = agent.select_action(flat.astype(np.float32))
            actions[agent_id] = int(action)

        raw_obs, _, term, trunc, _ = env.step(actions)

        active = [agent_id for agent_id in env.agents if env.just_communicated.get(agent_id, False)]
        if comm is not None:
            comm.update_messages_after_step(active)
            obs = comm.build_augment_observation(raw_obs)
        else:
            obs = raw_obs

        for agent_id in env.agents:
            pos = env.agent_positions[agent_id]
            trajectories[agent_id].append(pos)
            visits[pos[0], pos[1]] += 1

        if all(term.values()) or all(trunc.values()):
            break

    return {
        "trajectories": trajectories,
        "visits": visits,
        "final_grid": env.get_final_grid(),
        "grid_size": env.grid_size,
    }


def _plot_curve(x: List[float], y: List[float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_trajectory(trajectories: Dict[str, List[tuple]], grid_size: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green", "tab:brown"]

    for idx, (agent_id, positions) in enumerate(trajectories.items()):
        xs = [pos[1] for pos in positions]
        ys = [pos[0] for pos in positions]
        colour = colours[idx % len(colours)]
        ax.plot(xs, ys, color=colour, linewidth=2, label=agent_id)
        ax.scatter(xs[0], ys[0], color=colour, marker="o", edgecolors="black")
        ax.scatter(xs[-1], ys[-1], color=colour, marker="x")

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_title("Final episode trajectories")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_heatmap(visits: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(visits, cmap="hot", origin="upper", interpolation="nearest")
    ax.set_title("Final episode visitation heatmap")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_final_frame(grid: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, origin="upper", interpolation="nearest")
    ax.set_title("Final episode frame")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _jain_index(values: List[float]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    denom = len(values) * sum(value * value for value in values)
    if denom == 0:
        return 0.0
    return float((total * total) / denom)


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0
