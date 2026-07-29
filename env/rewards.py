from __future__ import annotations

from typing import Dict


def compute_reward(
    agent_id: str,
    collected: Dict[str, float],
    team_total: float,
    scheme: str,
    cooperative_variant: str = "plus_own",
    alpha: float = 0.5,
) -> float:
    scheme = scheme.lower()  # normalise input

    num_agents = max(1, len(collected))  # avoid divide by zero
    own = float(collected.get(agent_id, 0.0))  # agent’s own reward
    team_avg = float(team_total) / num_agents  # average team reward

    # purely selfish reward
    if scheme == "selfish":
        return own

    # cooperative reward. Two variants (see research-paper/ISSUES.md #1):
    #   "plus_own"  -> team_avg + 0.1*own  (what the frozen 50k runs used)
    #   "team_avg"  -> pure team average   (what the dissertation eq 2.7 states)
    # Default preserves original behaviour until a reproduction run decides which
    # to standardise on for the paper.
    if scheme in ("cooperative", "fully_cooperative"):
        if cooperative_variant == "team_avg":
            return team_avg
        return team_avg + 0.1 * own

    # mix of selfish + cooperative: r_i = alpha*own + (1-alpha)*team_avg.
    # alpha=1 recovers selfish, alpha=0 recovers cooperative (team_avg). Enables the
    # alpha-sweep that turns the three discrete conditions into a continuous curve.
    if scheme == "mixed":
        return alpha * own + (1 - alpha) * team_avg

    return own  # fallback


def selfish_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    # each agent only cares about its own reward
    return {
        agent: compute_reward(agent, raw_rewards, sum(raw_rewards.values()), "selfish")
        for agent in raw_rewards
    }


def cooperative_rewards(
    raw_rewards: Dict[str, float], cooperative_variant: str = "plus_own"
) -> Dict[str, float]:
    team_total = float(sum(raw_rewards.values()))  # total team reward

    # each agent gets same team-based reward
    return {
        agent: compute_reward(agent, raw_rewards, team_total, "cooperative", cooperative_variant)
        for agent in raw_rewards
    }


def fully_cooperative_rewards(raw_rewards: Dict[str, float]) -> Dict[str, float]:
    # identical to cooperative (just alias)
    return cooperative_rewards(raw_rewards)


def mixed_rewards(raw_rewards: Dict[str, float], alpha: float = 0.5) -> Dict[str, float]:
    team_total = float(sum(raw_rewards.values()))

    # blend of individual + team reward, controlled by alpha
    return {
        agent: compute_reward(agent, raw_rewards, team_total, "mixed", alpha=alpha)
        for agent in raw_rewards
    }


def apply_reward_scheme(
    scheme: str,
    raw_rewards: Dict[str, float],
    cumulative_collected: Dict[str, int],
    total_spawned: int,
    alpha: float = 0.5,
    cooperative_variant: str = "plus_own",
) -> Dict[str, float]:
    scheme = scheme.lower()  # normalise

    # route to correct reward function
    if scheme == "selfish":
        return selfish_rewards(raw_rewards)

    if scheme in ("cooperative", "fully_cooperative"):
        return cooperative_rewards(raw_rewards, cooperative_variant)

    if scheme == "mixed":
        return mixed_rewards(raw_rewards, alpha)

    return selfish_rewards(raw_rewards)  # default fallback