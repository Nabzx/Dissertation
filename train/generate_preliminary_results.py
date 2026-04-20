"""Generate preliminary plots and summaries from saved episodes."""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from env.utils import save_heatmap, save_movement_heatmap, save_reward_curve


def load_episode_data(logs_dir: str = "logs/episodes") -> List[Dict]:
    """Load episode JSON files from a logs directory."""
    episode_data = []

    if not os.path.exists(logs_dir):
        print(f"Warning: {logs_dir} does not exist. No episodes to process.")
        return episode_data

    episode_files = sorted(Path(logs_dir).glob("episode_*.json"))

    for episode_file in episode_files:
        with open(episode_file, "r") as f:
            data = json.load(f)
            episode_data.append(data)

    return episode_data


def calculate_survival_rate(episode_data: List[Dict]) -> Dict[str, float]:
    """Calculate the legacy two-agent survival summary."""
    if len(episode_data) == 0:
        return {
            "agent_0_survival_rate": 0.0,
            "agent_1_survival_rate": 0.0,
            "both_survival_rate": 0.0,
            "total_episodes": 0,
        }

    agent_0_survived = sum(1 for ep in episode_data if ep["agent_0_survived"])
    agent_1_survived = sum(1 for ep in episode_data if ep["agent_1_survived"])
    both_survived = sum(1 for ep in episode_data if ep["both_survived"])
    total_episodes = len(episode_data)

    return {
        "agent_0_survival_rate": agent_0_survived / total_episodes,
        "agent_1_survival_rate": agent_1_survived / total_episodes,
        "both_survival_rate": both_survived / total_episodes,
        "agent_0_survived_count": agent_0_survived,
        "agent_1_survived_count": agent_1_survived,
        "both_survived_count": both_survived,
        "total_episodes": total_episodes,
    }


def calculate_cooperation_scores(episode_data: List[Dict]) -> Dict[str, float]:
    """Calculate the original two-agent cooperation score."""
    if len(episode_data) == 0:
        return {
            "average_cooperation": 0.0,
            "episode_cooperations": [],
        }

    episode_cooperations = []

    for ep in episode_data:
        agent_0_resources = ep["resources_collected"]["agent_0"]
        agent_1_resources = ep["resources_collected"]["agent_1"]
        total_spawned = ep["total_resources_spawned"]

        cooperation = min(agent_0_resources, agent_1_resources) / max(1, total_spawned)
        episode_cooperations.append(cooperation)

    return {
        "average_cooperation": np.mean(episode_cooperations),
        "std_cooperation": np.std(episode_cooperations),
        "min_cooperation": np.min(episode_cooperations),
        "max_cooperation": np.max(episode_cooperations),
        "episode_cooperations": episode_cooperations,
    }


def aggregate_heatmaps(episode_data: List[Dict], grid_size: int = 15) -> Dict[str, np.ndarray]:
    """Add episode heatmaps together for agent_0 and agent_1."""
    aggregated = {
        "agent_0": np.zeros((grid_size, grid_size), dtype=np.int32),
        "agent_1": np.zeros((grid_size, grid_size), dtype=np.int32),
    }

    for ep in episode_data:
        if "heatmaps" in ep:
            for agent_id, heatmap_list in ep["heatmaps"].items():
                if agent_id in aggregated:
                    heatmap_array = np.array(heatmap_list, dtype=np.int32)
                    aggregated[agent_id] += heatmap_array

    return aggregated


def create_resource_distribution_heatmap(episode_data: List[Dict], grid_size: int = 15) -> np.ndarray:
    """Count initial resource positions across episodes."""
    resource_heatmap = np.zeros((grid_size, grid_size), dtype=np.int32)

    for ep in episode_data:
        for row, col in ep["initial_resource_positions"]:
            resource_heatmap[row, col] += 1

    return resource_heatmap


def create_screenshot_montage(
    episode_data: List[Dict],
    num_samples: int = 9,
    output_path: str = "results/screenshot_montage.png",
):
    """Build a square-ish montage of saved final screenshots."""
    if len(episode_data) == 0:
        print("No episode data to create montage")
        return

    sample_indices = np.linspace(0, len(episode_data) - 1, num_samples, dtype=int)
    sample_episodes = [episode_data[i] for i in sample_indices]

    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx, ep in enumerate(sample_episodes):
        if idx >= len(axes):
            break

        screenshot_path = ep.get("screenshot_path")
        if screenshot_path and os.path.exists(screenshot_path):
            img = plt.imread(screenshot_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Episode {ep['episode_num']}", fontsize=8)
        else:
            axes[idx].text(
                0.5,
                0.5,
                f"Episode {ep['episode_num']}\n(No screenshot)",
                ha="center",
                va="center",
            )
        axes[idx].axis("off")

    for idx in range(len(sample_episodes), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Sample Episode Final States", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved screenshot montage to {output_path}")


def generate_preliminary_results(logs_dir: str = "logs/episodes", results_dir: str = "results"):
    """Generate all preliminary result files."""
    print("Loading episode data...")
    episode_data = load_episode_data(logs_dir)

    if len(episode_data) == 0:
        print("No episode data found. Please run simulations first.")
        return

    print(f"Processing {len(episode_data)} episodes...")

    print("Calculating survival rates...")
    survival_stats = calculate_survival_rate(episode_data)

    print("Calculating cooperation scores...")
    cooperation_stats = calculate_cooperation_scores(episode_data)

    print("Creating resource distribution heatmap...")
    resource_heatmap = create_resource_distribution_heatmap(episode_data, grid_size=15)
    resource_heatmap_path = os.path.join(results_dir, "resource_distribution_heatmap.png")
    os.makedirs(results_dir, exist_ok=True)
    save_heatmap(
        resource_heatmap,
        resource_heatmap_path,
        "All Episodes",
        title="Resource Distribution Heatmap (All Episodes)",
    )

    print("Creating screenshot montage...")
    montage_path = os.path.join(results_dir, "screenshot_montage.png")
    create_screenshot_montage(episode_data, num_samples=9, output_path=montage_path)

    print("Aggregating movement heatmaps...")
    aggregated_heatmaps = aggregate_heatmaps(episode_data, grid_size=15)

    logs_base_dir = os.path.dirname(logs_dir.rstrip("/\\"))
    for agent_id, heatmap in aggregated_heatmaps.items():
        heatmap_path = os.path.join(logs_base_dir, "heatmaps", f"{agent_id}_heatmap.png")
        save_movement_heatmap(
            heatmap,
            heatmap_path,
            cmap="hot",
        )
        print(f"  Saved aggregated heatmap for {agent_id}")

    print("Generating reward curves...")
    reward_history = {
        "agent_0": [],
        "agent_1": [],
    }

    for ep in episode_data:
        resources = ep["resources_collected"]
        reward_history["agent_0"].append(resources["agent_0"])
        reward_history["agent_1"].append(resources["agent_1"])

    reward_curve_path = os.path.join(results_dir, "reward_curves", "reward_curve.png")
    save_reward_curve(reward_history, reward_curve_path)
    print(f"  Saved reward curve to {reward_curve_path}")

    summary = {
        "total_episodes": len(episode_data),
        "survival_rates": survival_stats,
        "cooperation_scores": {
            "average": float(cooperation_stats["average_cooperation"]),
            "std": float(cooperation_stats["std_cooperation"]),
            "min": float(cooperation_stats["min_cooperation"]),
            "max": float(cooperation_stats["max_cooperation"]),
        },
        "resource_statistics": {
            "average_resources_per_episode": float(np.mean([ep["total_resources_spawned"] for ep in episode_data])),
            "average_agent_0_collected": float(np.mean([ep["resources_collected"]["agent_0"] for ep in episode_data])),
            "average_agent_1_collected": float(np.mean([ep["resources_collected"]["agent_1"] for ep in episode_data])),
        },
    }

    summary_path = os.path.join(results_dir, "preliminary_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PRELIMINARY RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Episodes: {summary['total_episodes']}")
    print("\nSurvival Rates:")
    print(
        f"  Agent 0: {survival_stats['agent_0_survival_rate']:.2%} "
        f"({survival_stats['agent_0_survived_count']}/{survival_stats['total_episodes']})"
    )
    print(
        f"  Agent 1: {survival_stats['agent_1_survival_rate']:.2%} "
        f"({survival_stats['agent_1_survived_count']}/{survival_stats['total_episodes']})"
    )
    print(
        f"  Both: {survival_stats['both_survival_rate']:.2%} "
        f"({survival_stats['both_survived_count']}/{survival_stats['total_episodes']})"
    )
    print("\nCooperation Score:")
    print(f"  Average: {cooperation_stats['average_cooperation']:.4f}")
    print(f"  Std Dev: {cooperation_stats['std_cooperation']:.4f}")
    print(f"  Range: [{cooperation_stats['min_cooperation']:.4f}, {cooperation_stats['max_cooperation']:.4f}]")
    print("\nResource Statistics:")
    print(f"  Avg Resources/Episode: {summary['resource_statistics']['average_resources_per_episode']:.2f}")
    print(f"  Avg Agent 0 Collected: {summary['resource_statistics']['average_agent_0_collected']:.2f}")
    print(f"  Avg Agent 1 Collected: {summary['resource_statistics']['average_agent_1_collected']:.2f}")
    print("=" * 60)

    print(f"\nResults saved to {results_dir}/")
    print("  - preliminary_results_summary.json")
    print("  - resource_distribution_heatmap.png")
    print("  - screenshot_montage.png")
    print("  - reward_curves/reward_curve.png")
    print(f"\nAggregated heatmaps saved to {os.path.join(logs_base_dir, 'heatmaps')}/")
    print("  - agent_0_heatmap.png")
    print("  - agent_1_heatmap.png")
    print("=" * 60)


if __name__ == "__main__":
    generate_preliminary_results()
