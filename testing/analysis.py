"""
Analysis utilities for the testing pipeline.

This module focuses on:
- Multi-seed stability summaries (mean / std of metrics)
- Simple statistical testing using the Mann–Whitney U test

SciPy is optional:
- If it is installed, we call `scipy.stats.mannwhitneyu`.
- If not, we compute the U statistic manually and return it together with
  a friendly note explaining that the p-value is omitted.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from scipy.stats import mannwhitneyu as _scipy_mwu  # type: ignore

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - gracefully handle missing SciPy
    SCIPY_AVAILABLE = False
    _scipy_mwu = None  # type: ignore


def multi_seed_summary(
    seed_metrics: Dict[int, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-seed metrics into mean / std for each metric key.

    Args:
        seed_metrics: mapping seed -> {metric_name: value}

    Returns:
        {metric_name: {"mean": ..., "std": ...}, ...}
    """
    if not seed_metrics:
        return {}

    # Collect values per metric name.
    per_metric: Dict[str, list[float]] = {}
    for _seed, metrics in seed_metrics.items():
        for name, value in metrics.items():
            per_metric.setdefault(name, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for name, values in per_metric.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }
    return summary


def mann_whitney_u(
    sample_a: Iterable[float], sample_b: Iterable[float]
) -> Dict[str, float | str | None]:
    """
    Compare two samples using the Mann–Whitney U test.

    When SciPy is available, this simply forwards to scipy and returns
    both the U statistic and the p-value. If SciPy is not installed,
    we compute the U statistic manually and return a result with
    `p_value = None` and a note explaining the limitation.
    """
    a = np.asarray(list(sample_a), dtype=np.float64)
    b = np.asarray(list(sample_b), dtype=np.float64)

    if a.size == 0 or b.size == 0:
        return {
            "u_stat": float("nan"),
            "p_value": None,
            "note": "One or both samples are empty; Mann–Whitney U undefined.",
        }

    if SCIPY_AVAILABLE:
        result = _scipy_mwu(a, b, alternative="two-sided")
        return {
            "u_stat": float(result.statistic),
            "p_value": float(result.pvalue),
            "note": "Computed using SciPy.",
        }

    # -------------------- manual fallback (no SciPy) --------------------
    # We compute the U statistic via ranking, but we deliberately do not
    # approximate the p-value to keep the implementation short and clear.
    combined = np.concatenate([a, b])
    ranks = _rankdata(combined)

    ranks_a = ranks[: a.size]
    ranks_b = ranks[a.size :]

    n1 = float(a.size)
    n2 = float(b.size)

    R1 = float(np.sum(ranks_a))
    # R2 = float(np.sum(ranks_b))  # Not needed for U1 if R1 known

    U1 = R1 - n1 * (n1 + 1.0) / 2.0
    # U2 = n1 * n2 - U1  # Could also be reported

    return {
        "u_stat": float(U1),
        "p_value": None,
        "note": "SciPy not installed; computed U statistic only.",
    }


def _rankdata(x: np.ndarray) -> np.ndarray:
    """
    Simple rank assignment with average ranks for ties.

    This mirrors the behaviour of `scipy.stats.rankdata(method="average")`
    but is implemented here in a few lines to avoid depending on SciPy.
    """
    # Argsort twice to obtain ranks.
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)

    # Handle ties: for each unique value, assign the mean of its ranks.
    unique_vals, inverse_indices, counts = np.unique(
        x, return_inverse=True, return_counts=True
    )

    if unique_vals.size < x.size:
        # There are duplicates; adjust their ranks.
        for idx, count in enumerate(counts):
            if count == 1:
                continue
            # Indices in the original array that equal this unique value.
            mask = inverse_indices == idx
            # Average rank for all these entries.
            avg_rank = float(np.mean(ranks[mask]))
            ranks[mask] = avg_rank

    return ranks

