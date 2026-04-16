"""
Compute cooperation and performance metrics from headless PPO training output.

The input is the JSON metrics file produced by train_headless.py. Outputs:
- per-episode CSV with cooperation metrics
- per-episode JSON with the same metrics
- summary JSON with aggregate statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List


def jains_fairness_index(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0

    numerator = sum(values) ** 2
    denominator = len(values) * sum(value**2 for value in values)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def moving_average(values: List[float], window: int) -> List[float]:
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(mean(values[start : idx + 1]))
    return smoothed


def compute_episode_metrics(
    rows: List[Dict],
    total_available: int,
    smoothing_window: int,
) -> List[Dict]:
    per_episode = []

    for row in rows:
        resources = row.get("resources_collected", {})
        agent_0 = int(resources.get("agent_0", 0))
        agent_1 = int(resources.get("agent_1", 0))
        collected = int(row.get("total_resources", agent_0 + agent_1))
        available = max(int(total_available), collected)

        efficiency = collected / available if available > 0 else 0.0
        fairness = jains_fairness_index([agent_0, agent_1])
        balance_gap = abs(agent_0 - agent_1)
        normalized_balance_gap = balance_gap / collected if collected > 0 else 0.0
        cooperation_score = efficiency * fairness
        agent_0_survived = agent_0 > 0
        agent_1_survived = agent_1 > 0
        both_survived = agent_0_survived and agent_1_survived
        any_survived = agent_0_survived or agent_1_survived

        metrics = {
            "episode": int(row["episode"]),
            "total_reward": float(row.get("total_reward", 0.0)),
            "total_collected": collected,
            "total_available_estimate": available,
            "agent_0_resources": agent_0,
            "agent_1_resources": agent_1,
            "resource_efficiency": efficiency,
            "jain_fairness": fairness,
            "cooperation_score": cooperation_score,
            "balance_gap": balance_gap,
            "normalized_balance_gap": normalized_balance_gap,
            "agent_0_survived": agent_0_survived,
            "agent_1_survived": agent_1_survived,
            "both_survived": both_survived,
            "any_survived": any_survived,
            "entropy": float((row.get("ppo_metrics") or {}).get("entropy", 0.0)),
        }
        per_episode.append(metrics)

    for key in ["resource_efficiency", "jain_fairness", "cooperation_score", "entropy"]:
        values = [float(row[key]) for row in per_episode]
        smoothed = moving_average(values, smoothing_window)
        for row, value in zip(per_episode, smoothed):
            row[f"{key}_ma"] = value

    return per_episode


def summarize_metrics(per_episode: List[Dict]) -> Dict:
    summary = {
        "num_episodes": len(per_episode),
        "survival_rates": {
            "agent_0": _rate(row["agent_0_survived"] for row in per_episode),
            "agent_1": _rate(row["agent_1_survived"] for row in per_episode),
            "both_agents": _rate(row["both_survived"] for row in per_episode),
            "at_least_one_agent": _rate(row["any_survived"] for row in per_episode),
        },
        "metrics": {},
        "early_vs_late": {},
    }

    metric_keys = [
        "total_reward",
        "total_collected",
        "resource_efficiency",
        "jain_fairness",
        "cooperation_score",
        "balance_gap",
        "normalized_balance_gap",
        "entropy",
    ]
    for key in metric_keys:
        values = [float(row[key]) for row in per_episode]
        summary["metrics"][key] = _describe(values)

    split = min(1000, max(1, len(per_episode) // 10))
    early = per_episode[:split]
    late = per_episode[-split:]
    for key in ["resource_efficiency", "jain_fairness", "cooperation_score", "total_collected", "entropy"]:
        early_mean = mean(float(row[key]) for row in early)
        late_mean = mean(float(row[key]) for row in late)
        summary["early_vs_late"][key] = {
            "first_window_mean": early_mean,
            "last_window_mean": late_mean,
            "delta": late_mean - early_mean,
            "window_size": split,
        }

    return summary


def _rate(flags: Iterable[bool]) -> float:
    flags = list(flags)
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def _describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def write_json(path: str, data) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str, rows: List[Dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-agent cooperation metrics.")
    parser.add_argument("--input", default="results/headless_training_metrics_10000.json")
    parser.add_argument("--per-episode-csv", default="results/cooperation_metrics_10000.csv")
    parser.add_argument("--per-episode-json", default="results/cooperation_metrics_10000.json")
    parser.add_argument("--summary-json", default="results/cooperation_summary_10000.json")
    parser.add_argument("--total-available", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.input, "r") as f:
        rows = json.load(f)

    per_episode = compute_episode_metrics(
        rows,
        total_available=args.total_available,
        smoothing_window=args.smoothing_window,
    )
    summary = summarize_metrics(per_episode)

    write_csv(args.per_episode_csv, per_episode)
    write_json(args.per_episode_json, per_episode)
    write_json(args.summary_json, summary)

    print(f"Analyzed {len(per_episode)} episodes")
    print(f"Saved per-episode CSV: {args.per_episode_csv}")
    print(f"Saved per-episode JSON: {args.per_episode_json}")
    print(f"Saved summary JSON: {args.summary_json}")
    print(
        "Summary: "
        f"efficiency={summary['metrics']['resource_efficiency']['mean']:.3f}, "
        f"fairness={summary['metrics']['jain_fairness']['mean']:.3f}, "
        f"cooperation={summary['metrics']['cooperation_score']['mean']:.3f}, "
        f"both_survival={summary['survival_rates']['both_agents']:.3f}"
    )


if __name__ == "__main__":
    main()
