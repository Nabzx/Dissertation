"""Independent (non-parameter-shared) PPO: one network per agent.

Ablation for the paper's main threat to validity. The default setup shares a single
policy across all four agents, so the "four agents" are really one policy generalising
over four starting positions. This wrapper gives each agent its own network, optimiser
and buffer, so they are genuinely independent learners.

It duck-types the PPOAgent interface used by train.run_simulation, with an extra
agent_id argument, and sets is_multi = True so callers can dispatch.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from agents.ppo_agent import PPOAgent, PPOConfig


class IndependentPPO:
    is_multi = True

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int,
        n_actions: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        # one fully independent learner per agent
        self.policies: Dict[str, PPOAgent] = {
            aid: PPOAgent(obs_dim=obs_dim, n_actions=n_actions, config=config, device=device)
            for aid in self.agent_ids
        }

    def reset_buffer(self) -> None:
        for p in self.policies.values():
            p.reset_buffer()

    def select_action(self, obs: np.ndarray, agent_id: str) -> Tuple[int, float, float]:
        return self.policies[agent_id].select_action(obs)

    def get_value(self, obs: np.ndarray, agent_id: str) -> float:
        return self.policies[agent_id].get_value(obs)

    def store_transition(self, agent_id: str, **kwargs) -> None:
        kwargs.pop("trajectory_id", None)  # each policy owns a single trajectory
        self.policies[agent_id].store_transition(trajectory_id=agent_id, **kwargs)

    def update(self, last_value=0.0, last_done=True) -> Dict[str, float]:
        # update each policy with its own bootstrap value, then average the metrics
        per_agent = []
        for aid, p in self.policies.items():
            lv = last_value.get(aid, 0.0) if isinstance(last_value, dict) else last_value
            per_agent.append(p.update(last_value=lv, last_done=last_done))
        keys = per_agent[0].keys() if per_agent else []
        return {k: float(np.mean([m[k] for m in per_agent])) for k in keys}

    def save(self, path: str) -> None:
        root, ext = os.path.splitext(path)
        for aid, p in self.policies.items():
            p.save(f"{root}_{aid}{ext}")
