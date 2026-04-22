from __future__ import annotations

from typing import Dict


def compute_reward(agent_id: str, collected: Dict[str, float], team_total: float, scheme: str) -> float:
    scheme = scheme.lower()
    num_agents = max(1, len(collected))
    own = float(collected.get(agent_id, 0.0))
    team_avg = float(team_total) / num_agents

    if scheme == "selfish":
        return own
    if scheme in ("cooperative", "fully_cooperative"):
        return team_avg
    if scheme == "mixed":
        alpha = 0.5
        return alpha * own + (1 - alpha) * team_avg

    return own


def selfish_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    return {agent: compute_reward(agent, raw_rewards, sum(raw_rewards.values()), "selfish") for agent in raw_rewards}


def cooperative_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    team_total = float(sum(raw_rewards.values()))
    return {agent: compute_reward(agent, raw_rewards, team_total, "cooperative") for agent in raw_rewards}


def fully_cooperative_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    return cooperative_rewards(raw_rewards)


def mixed_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    team_total = float(sum(raw_rewards.values()))
    return {agent: compute_reward(agent, raw_rewards, team_total, "mixed") for agent in raw_rewards}


def apply_reward_scheme(
    scheme: str,
    raw_rewards: Dict[str, float],
    cumulative_collected: Dict[str, int],
    total_spawned: int,
    alpha: float = 0.5,
) -> Dict[str, float]:
    scheme = scheme.lower()
    if scheme == "selfish":
        return selfish_rewards(raw_rewards)
    if scheme in ("cooperative", "fully_cooperative"):
        return cooperative_rewards(raw_rewards)
    if scheme == "mixed":
        return mixed_rewards(raw_rewards)

    return selfish_rewards(raw_rewards)
