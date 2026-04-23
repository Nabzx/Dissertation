from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "selfish": Path("results/run_50000_selfish"),
    "cooperative": Path("results/run_50000_cooperative"),
    "mixed": Path("results/run_50000_mixed"),
}
OUTPUT_DIR = Path("results/final_comparison")
MAX_RESOURCES = 25
SMOOTHING_WINDOW = 200
PLOT_DOWNSAMPLE = 50
SCHEME_COLORS = {
    "selfish": "tab:red",
    "cooperative": "tab:blue",
    "mixed": "tab:green",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {scheme: load_run_data(run_dir) for scheme, run_dir in RUNS.items()}
    curve_data = {scheme: build_curves(data["metrics"]) for scheme, data in runs.items()}
    final_stats = {scheme: build_final_stats(data["final_episodes"]) for scheme, data in runs.items()}

    plot_multi_curve(curve_data, "reward", "Reward", "Smoothed reward", OUTPUT_DIR / "reward_comparison.png")
    plot_multi_curve(curve_data, "fairness", "Jain fairness", "Smoothed fairness", OUTPUT_DIR / "fairness_comparison.png")
    plot_multi_curve(
        curve_data,
        "cooperation",
        "Cooperation score",
        "Smoothed cooperation",
        OUTPUT_DIR / "cooperation_comparison.png",
    )
    plot_final_bars(final_stats, OUTPUT_DIR / "final_metrics_comparison.png")

    summary = {
        scheme: {
            "source_dir": str(RUNS[scheme]),
            **stats,
        }
        for scheme, stats in final_stats.items()
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison outputs to {OUTPUT_DIR}")


def load_run_data(run_dir: Path) -> Dict[str, List[Dict]]:
    metrics_path = run_dir / "headless_training_metrics.json"
    final_episodes_path = run_dir / "final_episodes.json"

    return {
        "metrics": load_json(metrics_path),
        "final_episodes": load_json(final_episodes_path),
    }


def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_curves(rows: List[Dict]) -> Dict[str, List[float]]:
    episodes = []
    rewards = []
    fairness = []
    cooperation = []

    for idx, row in enumerate(rows):
        resources = row.get("resources_collected", {})
        counts = [float(value) for value in resources.values()]
        total_resources = float(row.get("total_resources", sum(counts)))
        efficiency = total_resources / max(1, MAX_RESOURCES)

        episodes.append(int(row.get("episode", idx + 1)))
        rewards.append(float(row.get("total_reward", 0.0)))
        fairness.append(jain_index(counts))
        cooperation.append(efficiency * fairness[-1])

    return {
        "episodes": episodes,
        "reward": moving_average(rewards, SMOOTHING_WINDOW),
        "fairness": moving_average(fairness, SMOOTHING_WINDOW),
        "cooperation": moving_average(cooperation, SMOOTHING_WINDOW),
    }


def build_final_stats(final_episodes: List[Dict]) -> Dict[str, float]:
    rewards = []
    efficiency = []
    fairness = []
    cooperation = []

    for episode in final_episodes:
        resources = episode.get("resources_collected", {})
        counts = [float(value) for value in resources.values()]
        total_resources = float(sum(counts))
        eff = total_resources / max(1, MAX_RESOURCES)
        fair = jain_index(counts)

        rewards.append(float(episode.get("total_reward", 0.0)))
        efficiency.append(eff)
        fairness.append(fair)
        cooperation.append(eff * fair)

    return {
        "mean_reward": mean(rewards),
        "mean_efficiency": mean(efficiency),
        "mean_fairness": mean(fairness),
        "mean_cooperation": mean(cooperation),
        "num_final_episodes": len(final_episodes),
    }


def plot_multi_curve(
    curve_data: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for scheme, data in curve_data.items():
        x = data["episodes"][::PLOT_DOWNSAMPLE]
        y = data[metric_key][::PLOT_DOWNSAMPLE]
        ax.plot(x, y, label=scheme.capitalize(), linewidth=2.0, color=SCHEME_COLORS[scheme])

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_final_bars(final_stats: Dict[str, Dict[str, float]], output_path: Path) -> None:
    schemes = list(final_stats.keys())
    metric_keys = ["mean_reward", "mean_efficiency", "mean_fairness", "mean_cooperation"]
    metric_labels = ["Mean reward", "Mean efficiency", "Mean fairness", "Mean cooperation"]
    x = np.arange(len(metric_keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, scheme in enumerate(schemes):
        values = [final_stats[scheme][key] for key in metric_keys]
        offset = (idx - 1) * width
        ax.bar(x + offset, values, width=width, label=scheme.capitalize(), color=SCHEME_COLORS[scheme])

    ax.set_title("Final 100 episode comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []

    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(float(np.mean(values[start : idx + 1])))
    return smoothed


def jain_index(values: List[float]) -> float:
    if not values:
        return 0.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
