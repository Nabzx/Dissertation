"""
Reward shaping schemes for the testing pipeline.

The environment itself is left completely unchanged; instead we wrap the
raw rewards returned by `env.step` inside the testing runner and apply
simple reward transformations defined here.
"""

from __future__ import annotations

from typing import Dict, List


def selfish_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    """
    Selfish scheme: each agent only receives its own environment reward.

    The GridWorld environment already gives +1.0 to the agent that collects
    a resource, and 0.0 otherwise, so this function simply returns a copy.
    """
    return {k: float(v) for k, v in raw_rewards.items()}


def fully_cooperative_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    """
    Fully cooperative scheme: whenever any agent collects a resource,
    *both* agents receive +1 reward for that step.
    """
    if not raw_rewards:
        return {}

    any_collected = any(v > 0.0 for v in raw_rewards.values())
    if not any_collected:
        return {k: 0.0 for k in raw_rewards.keys()}

    # At least one agent collected a resource.
    return {k: 1.0 for k in raw_rewards.keys()}


def mixed_rewards(
    raw_rewards: Dict[str, float],
    cumulative_collected: Dict[str, int],
    total_spawned: int,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Mixed scheme: selfish reward plus a small cooperative bonus.

    For each step we add:

        alpha * (global_collected_so_far / total_spawned)

    to each agent's reward. This is intentionally simple and easy to
    reason about in an interview.
    """
    selfish = selfish_rewards(raw_rewards)
    total_collected = float(sum(cumulative_collected.values()))

    if total_spawned <= 0:
        bonus = 0.0
    else:
        bonus = alpha * (total_collected / float(total_spawned))

    shaped: Dict[str, float] = {}
    for agent, r in selfish.items():
        shaped[agent] = float(r + bonus)
    return shaped


def apply_reward_scheme(
    scheme: str,
    raw_rewards: Dict[str, float],
    cumulative_collected: Dict[str, int],
    total_spawned: int,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Helper that picks the appropriate shaping function based on a simple
    string name ("selfish", "mixed", "fully_cooperative").
    """
    scheme = scheme.lower()
    if scheme == "selfish":
        return selfish_rewards(raw_rewards)
    if scheme == "mixed":
        return mixed_rewards(
            raw_rewards,
            cumulative_collected=cumulative_collected,
            total_spawned=total_spawned,
            alpha=alpha,
        )
    if scheme == "fully_cooperative":
        return fully_cooperative_rewards(raw_rewards)

    # Fallback: behave like selfish if the scheme is unknown.
    return selfish_rewards(raw_rewards)

