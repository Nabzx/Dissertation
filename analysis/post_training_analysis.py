from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


MAX_RESOURCES = 25  # used to normalise efficiency


def run_post_training_analysis(run_name: str) -> None:
    out_dir = Path("results") / run_name
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)  
    # create plot folder if needed

    rows = _load_json(out_dir / "headless_training_metrics.json")  
    final_eps = _load_json(out_dir / "final_episodes.json")  
    # load training + final eval data

    curves = _build_curves(rows)
    final_stats = _build_final_stats(final_eps)

    # generate plots for different metrics
    _plot_curve(curves["episodes"], curves["rewards"], "Reward", "Reward", plot_dir / "reward_curve.png")
    _plot_curve(curves["episodes"], curves["efficiency"], "Resource efficiency", "Efficiency", plot_dir / "efficiency_curve.png")
    _plot_curve(curves["episodes"], curves["fairness"], "Fairness", "Jain index", plot_dir / "fairness_curve.png")
    _plot_curve(curves["episodes"], curves["cooperation"], "Cooperation score", "Score", plot_dir / "cooperation_curve.png")

    _plot_avg_heatmap(final_eps, plot_dir / "heatmap.png")  
    # average agent visitation

    _plot_sample_trajectories(final_eps, plot_dir / "trajectory.png")  
    # example movement paths

    _plot_metric_bars(final_stats, plot_dir / "final_metrics.png")  
    # summary bar chart

    with open(out_dir / "summary.json", "w") as f:
        json.dump(final_stats, f, indent=2)  
        # save summary stats


def _load_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required analysis file not found: {path}")  
        # fail early if missing

    with open(path, "r") as f:
        return json.load(f)


def _build_curves(rows: List[Dict]) -> Dict[str, List[float]]:
    episodes = []
    rewards = []
    efficiency = []
    fairness = []
    cooperation = []

    for idx, row in enumerate(rows):
        res = row.get("resources_collected", {})
        counts = [float(value) for value in res.values()]  
        # resources per agent

        total_res = float(row.get("total_resources", sum(counts)))
        eff = total_res / max(1, MAX_RESOURCES)  
        # normalised efficiency

        fair = _jain_index(counts)  
        # fairness score

        episodes.append(int(row.get("episode", idx + 1)))
        rewards.append(float(row.get("total_reward", 0.0)))
        efficiency.append(eff)
        fairness.append(fair)
        cooperation.append(eff * fair)  
        # combined metric

    return {
        "episodes": episodes,
        "rewards": rewards,
        "efficiency": efficiency,
        "fairness": fairness,
        "cooperation": cooperation,
    }


def _build_final_stats(final_eps: List[Dict]) -> Dict[str, float]:
    rewards = []
    resources = []
    efficiency = []
    fairness = []
    cooperation = []

    for ep in final_eps:
        res = ep.get("resources_collected", {})
        counts = [float(value) for value in res.values()]

        total_res = float(sum(counts))
        eff = total_res / max(1, MAX_RESOURCES)
        fair = _jain_index(counts)

        rewards.append(float(ep.get("total_reward", 0.0)))
        resources.append(total_res)
        efficiency.append(eff)
        fairness.append(fair)
        cooperation.append(eff * fair)

    return {
        "mean_reward": _mean(rewards),
        "mean_resources": _mean(resources),
        "mean_efficiency": _mean(efficiency),
        "mean_fairness": _mean(fairness),
        "mean_cooperation": _mean(cooperation),
        "num_episodes": len(final_eps),
    }  # averages over final episodes


def _plot_curve(x: List[float], y: List[float], title: str, ylabel: str, path: Path) -> None:
    sx, sy = downsample(x, moving_average(y, window=200), factor=50)  
    # smooth + reduce points for nicer plot

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sx, sy, linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_avg_heatmap(final_eps: List[Dict], path: Path) -> None:
    heatmaps = [np.array(ep["heatmap"], dtype=np.float32) for ep in final_eps if "heatmap" in ep]

    if not heatmaps:
        return  # skip if no data

    avg_heatmap = np.mean(heatmaps, axis=0)  
    avg_heatmap = np.log1p(avg_heatmap)  
    # log scaling to make differences clearer

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(avg_heatmap, cmap="hot", origin="upper", interpolation="nearest")

    ax.set_title("Average visitation heatmap (final 100 episodes)")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sample_trajectories(final_eps: List[Dict], path: Path) -> None:
    if not final_eps:
        return

    rng = np.random.default_rng(7)  
    # fixed seed for reproducibility

    count = min(10, len(final_eps))
    indices = sorted(rng.choice(len(final_eps), size=count, replace=False))  
    # sample episodes

    colours = ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:green", "tab:brown"]

    fig, ax = plt.subplots(figsize=(8, 8))
    seen_agents = set()

    for idx in indices:
        trajectories = final_eps[idx].get("trajectories", {})

        for agent_idx, (agent_id, positions) in enumerate(trajectories.items()):
            if not positions:
                continue

            xs = [pos[1] for pos in positions]
            ys = [pos[0] for pos in positions]

            label = agent_id if agent_id not in seen_agents else None
            seen_agents.add(agent_id)

            ax.plot(
                xs,
                ys,
                color=colours[agent_idx % len(colours)],
                linewidth=0.8,
                alpha=0.45,
                label=label,
            )

    grid_size = len(final_eps[0].get("heatmap", [])) or 25

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  
    # flip y-axis to match grid coords

    ax.set_title("Sample trajectories (10 episodes from final 100)")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metric_bars(stats: Dict[str, float], path: Path) -> None:
    labels = ["Reward", "Efficiency", "Fairness", "Cooperation"]

    values = [
        stats["mean_reward"],
        stats["mean_efficiency"],
        stats["mean_fairness"],
        stats["mean_cooperation"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["tab:blue", "tab:green", "tab:orange", "tab:purple"])

    ax.set_title("Final 100 episode metrics")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def moving_average(values: List[float], window: int = 200) -> List[float]:
    if not values:
        return []

    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)  
        out.append(float(np.mean(values[start : idx + 1])))  
        # average over sliding window

    return out


def downsample(x: List[float], y: List[float], factor: int = 50):
    return x[::factor], y[::factor]  
    # take every nth point


def _jain_index(values: List[float]) -> float:
    if not values:
        return 0.0

    total = sum(values)
    denom = len(values) * sum(value * value for value in values)

    if denom == 0:
        return 0.0

    return float((total * total) / denom)  
    # fairness calculation


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0  
    # safe mean