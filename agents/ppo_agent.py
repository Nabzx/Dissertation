from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    optim = object


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 4
    mini_batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


if TORCH_AVAILABLE:

    class _MLPPolicyValue(nn.Module):
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
        self.just_communicated = False

        self.device = device
        if TORCH_AVAILABLE:
            self.device_t = torch.device(device)
            self.model = _MLPPolicyValue(obs_dim, n_actions).to(self.device_t)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        else:
            self.model = None
            self.optimizer = None

        self.reset_buffer()
        self.update_history: List[Dict[str, float]] = []

    def reset_buffer(self) -> None:
        self.buffer: Dict[str, List[np.ndarray]] = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "trajectory_ids": [],
        }

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
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
        trajectory_id: str = "default",
    ) -> None:
        self.buffer["obs"].append(obs.astype(np.float32))
        self.buffer["actions"].append(np.array(action, dtype=np.int64))
        self.buffer["log_probs"].append(np.array(log_prob, dtype=np.float32))
        self.buffer["rewards"].append(np.array(reward, dtype=np.float32))
        self.buffer["dones"].append(np.array(done, dtype=np.float32))
        self.buffer["values"].append(np.array(value, dtype=np.float32))
        self.buffer["trajectory_ids"].append(trajectory_id)

    def _compute_returns_and_advantages(
        self, last_value: float = 0.0, last_done: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        trajectory_ids = self.buffer.get("trajectory_ids", ["default"] * T)
        unique_trajectories = list(dict.fromkeys(trajectory_ids))

        for trajectory_id in unique_trajectories:
            indices = [idx for idx, tid in enumerate(trajectory_ids) if tid == trajectory_id]
            last_gae = 0.0

            for pos in reversed(range(len(indices))):
                idx = indices[pos]
                if pos == len(indices) - 1:
                    next_value = 0.0 if last_done else float(last_value)
                else:
                    next_value = values[indices[pos + 1]]

                next_non_terminal = 1.0 - dones[idx]
                delta = (
                    rewards[idx]
                    + self.config.gamma * next_value * next_non_terminal
                    - values[idx]
                )
                last_gae = (
                    delta
                    + self.config.gamma
                    * self.config.lam
                    * next_non_terminal
                    * last_gae
                )
                advantages[idx] = last_gae

        returns = advantages + values

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def update(self, last_value: float = 0.0, last_done: bool = True) -> Dict[str, float]:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PPOAgent.update() was called but PyTorch is not installed. "
                "Either install torch to enable training or treat this agent "
                "as a random (non-learning) policy for demonstration."
            )

        if len(self.buffer["rewards"]) == 0:
            metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "mean_reward": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }
            self.update_history.append(metrics)
            return metrics

        obs = torch.from_numpy(np.stack(self.buffer["obs"])).to(self.device_t)
        actions = torch.from_numpy(np.stack(self.buffer["actions"])).to(self.device_t)
        old_log_probs = torch.from_numpy(
            np.stack(self.buffer["log_probs"])
        ).to(self.device_t)
        rewards_np = np.array(self.buffer["rewards"], dtype=np.float32)

        returns, advantages = self._compute_returns_and_advantages(last_value, last_done)
        returns_t = torch.from_numpy(returns).to(self.device_t)
        adv_t = torch.from_numpy(advantages).to(self.device_t)

        dataset_size = obs.size(0)
        batch_size = min(self.config.mini_batch_size, dataset_size)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []

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

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                        .float()
                        .mean()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

        self.reset_buffer()

        metrics = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_reward": float(np.mean(rewards_np)) if len(rewards_np) else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        }
        self.update_history.append(metrics)
        return metrics

    def save(self, path: str) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Cannot save PPOAgent because PyTorch is not installed.")

        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_history": self.update_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPOAgent":
        if not TORCH_AVAILABLE:
            raise RuntimeError("Cannot load PPOAgent because PyTorch is not installed.")

        checkpoint = torch.load(path, map_location=torch.device(device))
        config = PPOConfig(**checkpoint.get("config", {}))
        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            config=config,
            device=device,
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            agent.optimizer.load_state_dict(optimizer_state)
        agent.update_history = list(checkpoint.get("update_history", []))
        agent.reset_buffer()
        return agent
