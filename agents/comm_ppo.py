"""Two-headed policy for the communication phase — Phase 4.

The agent makes two decisions each step: how to move, and whether to broadcast a victim
sighting. These are **factored**, not combined into one 12-way action, for two reasons:

  1. Broadcasting costs no movement, so declining to share carries no opportunity cost. Any
     reduction in sharing is then attributable to the incentive rather than to the agent
     being busy — which is exactly what H2 needs to be identifiable.
  2. The broadcast probability can be read directly from the head, giving a clean behavioural
     measure of willingness to share.

Joint log-probability is the sum of the two heads' log-probs (they are conditionally
independent given the observation), which is the standard factored-policy formulation.

Subclasses PPOAgent so GAE, the clipped objective, entropy and gradient clipping are all
inherited; only the network and the loss's forward pass change.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from agents.ppo_agent import PPOAgent, PPOConfig, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _TwoHeadPolicyValue(nn.Module):
        def __init__(self, obs_dim: int, n_moves: int, hidden: int = 128):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.move_head = nn.Linear(hidden, n_moves)
            self.comm_head = nn.Linear(hidden, 2)     # broadcast: no / yes
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return self.move_head(h), self.comm_head(h), self.value_head(h).squeeze(-1)


class CommPPOAgent(PPOAgent):
    """PPO with factored (move, broadcast) actions."""

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
            self.model = _TwoHeadPolicyValue(obs_dim, n_actions, hidden).to(self.device_t)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

    # ---------- acting ----------
    def select_action(self, obs: np.ndarray) -> Tuple[Tuple[int, int], float, float]:
        if not TORCH_AVAILABLE:
            return (int(np.random.randint(self.n_actions)), int(np.random.randint(2))), 0.0, 0.0
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device_t).unsqueeze(0)
        with torch.no_grad():
            m_logits, c_logits, value = self.model(x)
            m_dist = torch.distributions.Categorical(logits=m_logits)
            c_dist = torch.distributions.Categorical(logits=c_logits)
            move, comm = m_dist.sample(), c_dist.sample()
            logp = m_dist.log_prob(move) + c_dist.log_prob(comm)
        return (int(move.item()), int(comm.item())), float(logp.item()), float(value.item())

    def select_action_greedy(self, obs: np.ndarray) -> Tuple[int, int]:
        if not TORCH_AVAILABLE:
            return (int(np.random.randint(self.n_actions)), 0)
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device_t).unsqueeze(0)
        with torch.no_grad():
            m_logits, c_logits, _ = self.model(x)
        return (int(torch.argmax(m_logits, -1).item()), int(torch.argmax(c_logits, -1).item()))

    def get_value(self, obs: np.ndarray) -> float:
        if not TORCH_AVAILABLE:
            return 0.0
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device_t).unsqueeze(0)
        with torch.no_grad():
            _, _, value = self.model(x)
        return float(value.item())

    def broadcast_prob(self, obs: np.ndarray) -> float:
        """P(broadcast) for analysis — the direct behavioural measure H2 is about."""
        if not TORCH_AVAILABLE:
            return 0.0
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device_t).unsqueeze(0)
        with torch.no_grad():
            _, c_logits, _ = self.model(x)
            return float(torch.softmax(c_logits, dim=-1)[0, 1].item())

    # ---------- update ----------
    def update(self, last_value=0.0, last_done=True) -> Dict[str, float]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("CommPPOAgent.update() called but torch not installed")
        if len(self.buffer["rewards"]) == 0:
            return super().update(last_value=last_value, last_done=last_done)

        obs = torch.from_numpy(np.stack(self.buffer["obs"])).to(self.device_t)
        acts = torch.from_numpy(np.stack(self.buffer["actions"])).to(self.device_t)  # (T, 2)
        old_lp = torch.from_numpy(np.stack(self.buffer["log_probs"])).to(self.device_t)
        returns, advantages = self._compute_returns_and_advantages(last_value, last_done)
        ret_t = torch.from_numpy(returns).to(self.device_t)
        adv_t = torch.from_numpy(advantages).to(self.device_t)
        rewards_np = np.array(self.buffer["rewards"], dtype=np.float32)

        n = obs.size(0)
        bs = min(self.config.mini_batch_size, n)
        pl, vl, ent, kls, clips = [], [], [], [], []

        for _ in range(self.config.train_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, bs):
                mb = idx[start:start + bs]
                m_logits, c_logits, values = self.model(obs[mb])
                m_dist = torch.distributions.Categorical(logits=m_logits)
                c_dist = torch.distributions.Categorical(logits=c_logits)

                logp = m_dist.log_prob(acts[mb, 0]) + c_dist.log_prob(acts[mb, 1])
                entropy = (m_dist.entropy() + c_dist.entropy()).mean()

                ratio = torch.exp(logp - old_lp[mb])
                a = adv_t[mb]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                    1 + self.config.clip_ratio) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, ret_t[mb])
                loss = (policy_loss + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = logp - old_lp[mb]
                    kls.append(float(((ratio - 1.0) - log_ratio).mean().item()))
                    clips.append(float((torch.abs(ratio - 1.0) > self.config.clip_ratio)
                                       .float().mean().item()))
                pl.append(float(policy_loss.item()))
                vl.append(float(value_loss.item()))
                ent.append(float(entropy.item()))

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
