"""
Minimal PPO implementation for the GridWorld testing pipeline.

This implementation is intentionally small and beginner friendly. It is not
optimised and is only meant to demonstrate a reasonable PPO-like workflow
for interview / experimentation purposes.

The agent expects *flat* vector observations. In the testing pipeline we
achieve this by flattening the 15x15 grid observation (and optionally
concatenating a small communication vector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - safety net for environments without torch
    # Torch is optional. The rest of the testing pipeline should still import
    # cleanly even if PPO training cannot run.
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = object  # type: ignore
    optim = object  # type: ignore


@dataclass
class PPOConfig:
    """Small configuration bundle for PPO hyperparameters."""

    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda (used in a simplified way)
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 4
    mini_batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01


if TORCH_AVAILABLE:

    class _MLPPolicyValue(nn.Module):
        """
        Tiny shared-body MLP with separate policy and value heads.

        This is deliberately small: two hidden layers with ReLU activations.
        """

        def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_dim, n_actions)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.body(x)
            logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            return logits, value


class PPOAgent:
    """
    Simple PPO agent for discrete actions.

    When PyTorch is available, this uses a small neural network. When PyTorch
    is NOT available, the class still initialises but any attempt to call
    ``update`` will raise a friendly error explaining the limitation. This
    allows the rest of the testing pipeline to run (e.g., with random actions)
    in environments that do not ship with torch by default.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config or PPOConfig()

        self.device = device
        if TORCH_AVAILABLE:
            self.device_t = torch.device(device)
            self.model = _MLPPolicyValue(obs_dim, n_actions).to(self.device_t)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        else:
            self.model = None  # type: ignore
            self.optimizer = None  # type: ignore

        # Simple buffer to collect trajectory data for a batch update.
        self.reset_buffer()

    # ------------------------------------------------------------------
    # Trajectory buffer helpers
    # ------------------------------------------------------------------
    def reset_buffer(self) -> None:
        self.buffer: Dict[str, List[np.ndarray]] = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Choose an action given a single flat observation.

        Returns:
            action (int), log_prob (float), value_estimate (float)
        """
        if TORCH_AVAILABLE:
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device_t)
            with torch.no_grad():
                logits, value = self.model(obs_t.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item()),
            )

        # Fallback: random policy with a dummy value estimate.
        action = np.random.randint(self.n_actions)
        log_prob = -math.log(self.n_actions)
        value_estimate = 0.0
        return action, log_prob, value_estimate

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store one time-step transition in the on-policy buffer."""
        self.buffer["obs"].append(obs.astype(np.float32))
        self.buffer["actions"].append(np.array(action, dtype=np.int64))
        self.buffer["log_probs"].append(np.array(log_prob, dtype=np.float32))
        self.buffer["rewards"].append(np.array(reward, dtype=np.float32))
        self.buffer["dones"].append(np.array(done, dtype=np.float32))
        self.buffer["values"].append(np.array(value, dtype=np.float32))

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def _compute_returns_and_advantages(
        self, last_value: float = 0.0, last_done: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and simple (non-GAE) advantages.

        This keeps the maths beginner-friendly: we use standard
        discounted returns and then define advantages as

            A_t = R_t - V_t
        """
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)

        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        next_return = 0.0 if last_done else last_value

        for t in reversed(range(T)):
            next_return = rewards[t] + self.config.gamma * next_return * (1.0 - dones[t])
            returns[t] = next_return

        advantages = returns - values
        # Normalise advantages for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def update(self, last_value: float = 0.0, last_done: bool = True) -> None:
        """
        Perform a PPO update using the collected on-policy batch.

        If torch is not available, this method will raise a friendly error to
        explain that learning cannot proceed in the current environment.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PPOAgent.update() was called but PyTorch is not installed. "
                "Either install torch to enable training or treat this agent "
                "as a random (non-learning) policy for demonstration."
            )

        if len(self.buffer["rewards"]) == 0:
            return

        # Prepare tensors
        obs = torch.from_numpy(np.stack(self.buffer["obs"])).to(self.device_t)
        actions = torch.from_numpy(np.stack(self.buffer["actions"])).to(self.device_t)
        old_log_probs = torch.from_numpy(
            np.stack(self.buffer["log_probs"])
        ).to(self.device_t)

        returns, advantages = self._compute_returns_and_advantages(last_value, last_done)
        returns_t = torch.from_numpy(returns).to(self.device_t)
        adv_t = torch.from_numpy(advantages).to(self.device_t)

        dataset_size = obs.size(0)
        batch_size = self.config.mini_batch_size

        for _ in range(self.config.train_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                batch_obs = obs[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log_probs = old_log_probs[mb_idx]
                batch_returns = returns_t[mb_idx]
                batch_adv = adv_t[mb_idx]

                logits, values = self.model(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (batch_returns - values).pow(2).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        # Clear buffer after update
        self.reset_buffer()

