"""
Additional metrics for analysing GridWorld experiments in testing mode.

All functions here are self-contained and work with simple numpy arrays or
Python containers so that they remain easy to understand and re-use.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

 
def jains_fairness_index(values: List[float] | np.ndarray) -> float:
    """
    Compute Jain's fairness index for a list of non‑negative values.

        J(x) = (sum_i x_i)^2 / (n * sum_i x_i^2)

    The result lies in (0, 1]. We define J = 0 if all values are zero to
    avoid division by zero.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0

    num = float(np.sum(x) ** 2)
    denom = float(x.size * np.sum(x ** 2))
    if denom == 0.0:
        return 0.0
    return num / denom


def spatial_entropy(heatmap: np.ndarray) -> float:
    """
    Compute entropy over a (possibly 2D) visit-frequency heatmap.

    The heatmap is first flattened and normalised to a probability
    distribution p. The spatial entropy is then:

        H = - sum_i p_i log(p_i)

    using natural logarithms. We define H = 0 when no visits occurred.
    """
    counts = np.asarray(heatmap, dtype=np.float64).flatten()
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    p = counts / total
    # Avoid log(0) by adding a tiny epsilon where needed.
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    entropy = -float(np.sum(p_safe * np.log(p_safe)))
    return entropy


def resource_utilisation_efficiency(
    resources_collected: Dict[str, int],
    total_spawned: int,
    steps_taken: int | None = None,
) -> Dict[str, float]:
    """
    Compute simple resource utilisation metrics.

    Args:
        resources_collected: mapping agent_id -> number of resources collected.
        total_spawned: total number of resources spawned in the episode.
        steps_taken: optional number of steps used in the episode, for
            computing "resources collected per step".

    Returns:
        Dict with keys:
            - "efficiency": total_collected / total_spawned
            - "per_step": total_collected / steps_taken (if steps_taken given)
    """
    total_collected = float(sum(resources_collected.values()))
    metrics: Dict[str, float] = {}

    if total_spawned > 0:
        metrics["efficiency"] = total_collected / float(total_spawned)
    else:
        metrics["efficiency"] = 0.0

    if steps_taken is not None and steps_taken > 0:
        metrics["per_step"] = total_collected / float(steps_taken)

    return metrics

