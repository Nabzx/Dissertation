"""Mandate reward for the disaster environment.

The premise is that each *agency* is measured on its own outcomes while the goal is the
collective one. So the credited unit is the agency, not the individual:

    r_i = alpha * mean(raw over agent i's agency) + (1 - alpha) * mean(raw over all agents)

  alpha = 1  -> agencies are credited only for what they themselves rescue
  alpha = 0  -> everyone is credited with the collective outcome

Setting num_agencies = num_agents recovers the purely individual case, so this generalises
rather than replaces the gridworld paper's formulation.

`credit="agent"` keeps the older individual-vs-collective form for comparison:
    r_i = alpha * own + (1 - alpha) * mean(raw over all agents)

Kept separate from env/rewards.py so the gridworld paper's reward code is untouched.
"""
from __future__ import annotations

from typing import Dict


def mandate_rewards(
    raw: Dict[str, float],
    agency_of: Dict[str, int],
    alpha: float,
    credit: str = "agency",
) -> Dict[str, float]:
    if not raw:
        return {}

    n = len(raw)
    global_mean = sum(raw.values()) / n

    if credit == "agent":
        return {a: alpha * raw[a] + (1.0 - alpha) * global_mean for a in raw}

    if credit != "agency":
        raise ValueError(f"unknown credit scheme '{credit}' (use 'agency' or 'agent')")

    totals: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for a, v in raw.items():
        g = agency_of[a]
        totals[g] = totals.get(g, 0.0) + v
        counts[g] = counts.get(g, 0) + 1

    agency_mean = {g: totals[g] / counts[g] for g in totals}
    return {a: alpha * agency_mean[agency_of[a]] + (1.0 - alpha) * global_mean for a in raw}
