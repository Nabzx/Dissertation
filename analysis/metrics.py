from __future__ import annotations

from typing import Dict, List

import numpy as np

 
def jains_fairness_index(values: List[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)  # convert to numpy array for easy ops

    if x.size == 0:
        return 0.0  # no values -> no fairness

    num = float(np.sum(x) ** 2)  # (sum x)^2
    denom = float(x.size * np.sum(x ** 2))  # n * sum(x^2)

    if denom == 0.0:
        return 0.0  # avoid divide by zero (all values zero)

    return num / denom  # final Jain index


def spatial_entropy(heatmap: np.ndarray) -> float:

    # Compute entropy over a (possibly 2D) visit-frequency heatmap.
    counts = np.asarray(heatmap, dtype=np.float64).flatten()  
    # flatten grid into 1D

    total = float(np.sum(counts))  # total visits

    if total <= 0.0:
        return 0.0  # no visits -> zero entropy

    p = counts / total  # convert to probability distribution

    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)  
    # avoid log(0) by clipping small values

    entropy = -float(np.sum(p_safe * np.log(p_safe)))  
    # standard entropy formula

    return entropy


def resource_utilisation_efficiency(
    resources_collected: Dict[str, int],
    total_spawned: int,
    steps_taken: int | None = None,
) -> Dict[str, float]:

    total_collected = float(sum(resources_collected.values()))  
    # total resources collected across all agents

    metrics: Dict[str, float] = {}

    if total_spawned > 0:
        metrics["efficiency"] = total_collected / float(total_spawned)  
        # proportion of resources actually used
    else:
        metrics["efficiency"] = 0.0

    if steps_taken is not None and steps_taken > 0:
        metrics["per_step"] = total_collected / float(steps_taken)  
        # how fast resources were collected

    return metrics