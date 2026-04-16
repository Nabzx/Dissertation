"""
Entry point for running small testing-mode experiments.

USAGE (from project root):
    python testing/run_testing_experiments.py

This script:
    - trains a tiny PPO agent (if PyTorch is available) or runs with a
      random policy (if not),
    - uses a simple bandwidth-limited communication layer,
    - evaluates several reward schemes across multiple random seeds,
    - computes additional metrics and basic statistics,
    - writes a compact JSON summary to `testing/results/testing_summary.json`,
    - prints a short human-readable summary to the console.

IMPORTANT:
    - Existing project files (including the main entrypoint) are not touched.
    - All logs / results are written under the `testing/` folder only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from env.gridworld_env import GridWorldEnv

# NOTE:
# We use absolute imports from the `testing` package so that this script can
# be executed directly via:
#     python testing/run_testing_experiments.py
from testing.analysis import mann_whitney_u, multi_seed_summary
from testing.communication import CommunicationLayer
from testing.metrics import (
    jains_fairness_index,
    resource_utilisation_efficiency,
    spatial_entropy,
)
from testing.ppo_agent import PPOAgent, PPOConfig, TORCH_AVAILABLE
from testing.rewards import apply_reward_scheme


@dataclass
class EpisodeMetrics:
    """Container for key metrics collected at the end of an episode."""

    fairness: float
    spatial_entropy_mean: float
    resource_efficiency: float


def _ensure_dirs() -> Tuple[str, str]:
    """Create testing subdirectories (logs + results) if needed."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return logs_dir, results_dir


def run_single_episode(
    seed: int,
    reward_scheme: str,
    ppo_agent: PPOAgent,
    n_steps_per_update: int = 128,
) -> EpisodeMetrics:
    """
    Run one episode under the specified reward scheme, collecting
    trajectories for PPO and computing additional metrics.
    """
    # Instantiate a small environment; config kept conservative for speed.
    env = GridWorldEnv(grid_size=15, num_resources=10, max_steps=100, seed=seed)
    comm_layer = CommunicationLayer(env)

    # Reset environment and communication.
    raw_obs, _ = env.reset(seed=seed)
    comm_layer.reset()
    obs = comm_layer.build_augment_observation(raw_obs)

    # Track how many resources each agent collects over the episode.
    cumulative_collected: Dict[str, int] = {agent: 0 for agent in env.agents}

    done_flags: Dict[str, bool] = {agent: False for agent in env.agents}
    step_count = 0

    # We collect at most n_steps_per_update transitions for training.
    ppo_agent.reset_buffer()

    while not all(done_flags.values()):
        # Convert current observations to flat vectors (already flattened by
        # the communication layer).
        actions: Dict[str, int] = {}
        log_probs: Dict[str, float] = {}
        values: Dict[str, float] = {}

        for agent_id in env.agents:
            if done_flags[agent_id]:
                # If already done, choose a no-op action (stay).
                actions[agent_id] = 0
                log_probs[agent_id] = 0.0
                values[agent_id] = 0.0
                continue

            agent_obs = obs[agent_id]
            action, log_p, value = ppo_agent.select_action(agent_obs)
            actions[agent_id] = action
            log_probs[agent_id] = log_p
            values[agent_id] = value

        # Step the environment with the chosen actions.
        raw_next_obs, raw_rewards, terminations, truncations, _ = env.step(actions)

        # Update counts of collected resources using the *environment* rewards
        # (before any shaping).
        for agent_id, r in raw_rewards.items():
            if r > 0.0:
                cumulative_collected[agent_id] += 1

        # Apply testing-mode reward shaping.
        shaped_rewards = apply_reward_scheme(
            scheme=reward_scheme,
            raw_rewards=raw_rewards,
            cumulative_collected=cumulative_collected,
            total_spawned=env.num_resources,
            alpha=0.5,
        )

        # Update done flags.
        done_flags = {
            agent_id: bool(terminations[agent_id] or truncations[agent_id])
            for agent_id in env.agents
        }

        # Let the communication layer compute new messages and use them to
        # augment the *next* observations.
        comm_layer.update_messages_after_step()
        next_obs = comm_layer.build_augment_observation(raw_next_obs)

        # Store transitions for PPO training.
        for agent_id in env.agents:
            # A simple approach is to treat each agent's transition
            # independently but share the same policy network.
            ppo_agent.store_transition(
                obs=obs[agent_id],
                action=actions[agent_id],
                log_prob=log_probs[agent_id],
                reward=shaped_rewards.get(agent_id, 0.0),
                done=done_flags[agent_id],
                value=values[agent_id],
                trajectory_id=agent_id,
            )

        obs = next_obs
        step_count += 1

        if all(done_flags.values()) or step_count >= n_steps_per_update:
            break

    # Perform a PPO update (if supported); if torch is not available this
    # will raise a clear RuntimeError which we allow to bubble up.
    try:
        ppo_agent.update(last_value=0.0, last_done=True)
    except RuntimeError as exc:
        # If torch is missing we log a short note and skip learning.
        # The experiment can still proceed and metrics are still gathered.
        print(f"[testing] PPO update skipped: {exc}")

    # ------------------ metric computation at end of episode ------------------
    # Use the environment's helper methods to get heatmaps and resource counts.
    heatmaps = env.get_heatmaps()
    resources_collected = env.get_resources_collected()

    fairness = jains_fairness_index(list(resources_collected.values()))
    entropies = [spatial_entropy(hm) for hm in heatmaps.values()]
    entropy_mean = float(np.mean(entropies)) if entropies else 0.0

    util_metrics = resource_utilisation_efficiency(
        resources_collected, total_spawned=env.num_resources, steps_taken=step_count
    )
    efficiency = float(util_metrics.get("efficiency", 0.0))

    return EpisodeMetrics(
        fairness=fairness,
        spatial_entropy_mean=entropy_mean,
        resource_efficiency=efficiency,
    )


def run_testing_experiments() -> Dict:
    """
    High-level orchestration:
        - Multiple seeds
        - Multiple reward schemes
        - Metric aggregation + statistical tests
    """
    logs_dir, results_dir = _ensure_dirs()

    # Keep experiments small and quick.
    seeds: List[int] = [0, 1, 2]
    reward_schemes: List[str] = ["selfish", "mixed", "fully_cooperative"]
    episodes_per_seed = 10

    # Observation dimension: 15x15 grid + communication vector.
    dummy_env = GridWorldEnv(grid_size=15, num_resources=10, max_steps=10, seed=0)
    dummy_comm = CommunicationLayer(dummy_env)
    dummy_obs, _ = dummy_env.reset(seed=0)
    aug_obs = dummy_comm.build_augment_observation(dummy_obs)
    obs_dim = int(next(iter(aug_obs.values())).size)
    action_dim = int(dummy_env.action_spaces[dummy_env.agents[0]].n)

    # Small PPO config for quick updates.
    ppo_cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        train_epochs=2,
        mini_batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    all_results: Dict[str, Dict] = {}

    for scheme in reward_schemes:
        scheme_seed_metrics: Dict[int, Dict[str, float]] = {}
        per_seed_episode_metrics: Dict[int, List[EpisodeMetrics]] = {}

        for seed in seeds:
            # Fresh PPO agent per (scheme, seed) pair keeps things simple.
            ppo_agent = PPOAgent(
                obs_dim=obs_dim,
                n_actions=action_dim,
                config=ppo_cfg,
                device="cpu",
            )

            episode_metrics: List[EpisodeMetrics] = []
            for _ep in range(episodes_per_seed):
                m = run_single_episode(
                    seed=seed,
                    reward_scheme=scheme,
                    ppo_agent=ppo_agent,
                    n_steps_per_update=128,
                )
                episode_metrics.append(m)

            per_seed_episode_metrics[seed] = episode_metrics

            # Aggregate metrics across episodes (mean per seed).
            fairness_vals = [em.fairness for em in episode_metrics]
            entropy_vals = [em.spatial_entropy_mean for em in episode_metrics]
            efficiency_vals = [em.resource_efficiency for em in episode_metrics]

            seed_summary = {
                "fairness_mean": float(np.mean(fairness_vals)),
                "spatial_entropy_mean": float(np.mean(entropy_vals)),
                "resource_efficiency_mean": float(np.mean(efficiency_vals)),
            }
            scheme_seed_metrics[seed] = seed_summary

        # Multi-seed stability analysis for this scheme.
        scheme_summary = multi_seed_summary(scheme_seed_metrics)

        all_results[scheme] = {
            "per_seed": scheme_seed_metrics,
            "multi_seed_summary": scheme_summary,
        }

    # ------------------------ statistical testing ------------------------
    # Example: compare fairness between "selfish" and "fully_cooperative".
    stat_tests: Dict[str, Dict] = {}
    base_scheme = "selfish"
    compare_scheme = "fully_cooperative"

    if base_scheme in all_results and compare_scheme in all_results:
        base_vals = [
            v["fairness_mean"] for v in all_results[base_scheme]["per_seed"].values()
        ]
        comp_vals = [
            v["fairness_mean"] for v in all_results[compare_scheme]["per_seed"].values()
        ]
        stat_tests["fairness_selfish_vs_fully_cooperative"] = mann_whitney_u(
            base_vals, comp_vals
        )

    summary = {
        "torch_available": TORCH_AVAILABLE,
        "seeds": seeds,
        "reward_schemes": reward_schemes,
        "results": all_results,
        "statistical_tests": stat_tests,
    }

    # ----------------------------- persistence -----------------------------
    summary_path = os.path.join(results_dir, "testing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Short console summary.
    print("\n=== Testing Mode Summary ===")
    print(f"Torch available for PPO training: {TORCH_AVAILABLE}")
    for scheme in reward_schemes:
        if scheme not in all_results:
            continue
        multi = all_results[scheme]["multi_seed_summary"]
        fairness_info = multi.get("fairness_mean", {})
        eff_info = multi.get("resource_efficiency_mean", {})
        print(
            f"  Scheme '{scheme}': "
            f"fairness mean≈{fairness_info.get('mean', 0.0):.3f}, "
            f"efficiency mean≈{eff_info.get('mean', 0.0):.3f}"
        )

    print(f"\nJSON summary written to: {summary_path}")
    return summary


if __name__ == "__main__":
    run_testing_experiments()
