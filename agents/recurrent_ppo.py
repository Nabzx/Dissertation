"""Recurrent (GRU) policy for the disaster environment — Phase 3.

The self-trace channel gives an agent a spatial memory of *where it has been*, but not a
memory of *what it saw*. A victim glimpsed through a doorway and not yet reached vanishes from
the observation the moment line of sight breaks. A recurrent policy can carry that across
time, which is exactly the capability a searching responder needs.

Design notes:
  - One hidden state per (agent, episode), reset at episode start.
  - The hidden state is recomputed during the PPO update by replaying each trajectory in
    order, rather than being stored per transition. Replaying is simpler and avoids the stale
    hidden-state problem, at the cost of one extra forward pass per epoch.
  - Trajectories are kept per agent (trajectory_id), so replay never mixes agents.

Subclasses PPOAgent so buffers, GAE, the clipped objective and gradient clipping are all
inherited; only the network and the update's forward pass differ.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.ppo_agent import PPOAgent, PPOConfig, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _RecurrentPolicyValue(nn.Module):
        def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, enc: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(obs_dim, enc), nn.ReLU())
            self.gru = nn.GRU(enc, hidden, batch_first=True)
            self.policy_head = nn.Linear(hidden, n_actions)
            self.value_head = nn.Linear(hidden, 1)
            self.hidden_size = hidden

        def forward(self, x: "torch.Tensor", h: Optional["torch.Tensor"] = None):
            """x: (B, T, obs_dim) -> logits (B, T, A), values (B, T), new hidden."""
            b, t, _ = x.shape
            e = self.encoder(x.reshape(b * t, -1)).reshape(b, t, -1)
            out, h_new = self.gru(e, h)
            logits = self.policy_head(out)
            values = self.value_head(out).squeeze(-1)
            return logits, values, h_new


class RecurrentPPOAgent(PPOAgent):
    """PPO with a GRU. Acting is stepwise (T=1); the update replays whole trajectories."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
        hidden: int = 128,
    ) -> None:
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, config=config, device=device)
        if TORCH_AVAILABLE:
            self.model = _RecurrentPolicyValue(obs_dim, n_actions, hidden=hidden).to(self.device_t)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.hidden_size = hidden
        self._h: Dict[str, "torch.Tensor"] = {}

    # ---------- acting ----------
    def reset_hidden(self, agent_id: Optional[str] = None) -> None:
        if agent_id is None:
            self._h = {}
        else:
            self._h.pop(agent_id, None)

    def _step(self, obs: np.ndarray, agent_id: str):
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device_t).view(1, 1, -1)
        h = self._h.get(agent_id)
        with torch.no_grad():
            logits, value, h_new = self.model(x, h)
        self._h[agent_id] = h_new
        return logits[:, -1, :], value[:, -1]

    def select_action(self, obs: np.ndarray, agent_id: str = "default"):
        if not TORCH_AVAILABLE:
            return super().select_action(obs)
        logits, value = self._step(obs, agent_id)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def select_action_greedy(self, obs: np.ndarray, agent_id: str = "default") -> int:
        if not TORCH_AVAILABLE:
            return super().select_action_greedy(obs)
        logits, _ = self._step(obs, agent_id)
        return int(torch.argmax(logits, dim=-1).item())

    def get_value(self, obs: np.ndarray, agent_id: str = "default") -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        _, value = self._step(obs, agent_id)
        return float(value.item())

    # ---------- update ----------
    def _trajectory_slices(self) -> Dict[str, List[int]]:
        slices: Dict[str, List[int]] = {}
        for i, tid in enumerate(self.buffer["trajectory_ids"]):
            slices.setdefault(tid, []).append(i)
        return slices

    def update(self, last_value=0.0, last_done=True) -> Dict[str, float]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("RecurrentPPOAgent.update() called but torch not installed")
        if len(self.buffer["rewards"]) == 0:
            return super().update(last_value=last_value, last_done=last_done)

        obs_all = np.stack(self.buffer["obs"])
        actions_all = np.stack(self.buffer["actions"])
        old_lp_all = np.stack(self.buffer["log_probs"])
        returns, advantages = self._compute_returns_and_advantages(last_value, last_done)

        obs_t = torch.from_numpy(obs_all).to(self.device_t)
        act_t = torch.from_numpy(actions_all).to(self.device_t)
        old_lp_t = torch.from_numpy(old_lp_all).to(self.device_t)
        ret_t = torch.from_numpy(returns).to(self.device_t)
        adv_t = torch.from_numpy(advantages).to(self.device_t)

        slices = self._trajectory_slices()
        pl, vl, ent, kls, clips = [], [], [], [], []

        for _ in range(self.config.train_epochs):
            # replay each agent's trajectory in order so the GRU sees the true sequence
            for idx in slices.values():
                sel = torch.tensor(idx, dtype=torch.long, device=self.device_t)
                x = obs_t[sel].unsqueeze(0)                    # (1, T, obs)
                logits, values, _ = self.model(x)
                logits = logits.squeeze(0)
                values = values.squeeze(0)

                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[sel])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_lp_t[sel])
                a = adv_t[sel]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret_t[sel])

                loss = (policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = logp - old_lp_t[sel]
                    kls.append(float(((ratio - 1.0) - log_ratio).mean().item()))
                    clips.append(float((torch.abs(ratio - 1.0) > self.config.clip_ratio)
                                       .float().mean().item()))
                pl.append(float(policy_loss.item()))
                vl.append(float(value_loss.item()))
                ent.append(float(entropy.item()))

        rewards_np = np.array(self.buffer["rewards"], dtype=np.float32)
        self.reset_buffer()
        metrics = {
            "policy_loss": float(np.mean(pl)) if pl else 0.0,
            "value_loss": float(np.mean(vl)) if vl else 0.0,
            "entropy": float(np.mean(ent)) if ent else 0.0,
            "mean_reward": float(np.mean(rewards_np)) if rewards_np.size else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clip_fraction": float(np.mean(clips)) if clips else 0.0,
        }
        self.update_history.append(metrics)
        return metrics
