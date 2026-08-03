"""Convolutional policy/value network for the disaster environment.

The observation is a C x W x W stack of spatial channels. Flattening it into an MLP discards
the spatial structure - the network has to relearn that neighbouring cells are related. A
small conv encoder gives that structure for free and is cheap at this window size.

Subclasses PPOAgent so all the corrected PPO machinery (GAE with per-trajectory bootstrapping,
clipped objective, entropy bonus, gradient clipping) is inherited unchanged; only the network
differs. That keeps the conv-vs-MLP comparison a clean ablation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from agents.ppo_agent import PPOAgent, PPOConfig, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class _ConvPolicyValue(nn.Module):
        def __init__(self, in_channels: int, view: int, n_actions: int, width: int = 32):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            feat = width * view * view
            self.trunk = nn.Sequential(nn.Linear(feat, 128), nn.ReLU())
            self.policy_head = nn.Linear(128, n_actions)
            self.value_head = nn.Linear(128, 1)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.body(x)
            h = self.trunk(h.flatten(1))
            return self.policy_head(h), self.value_head(h).squeeze(-1)


class ConvPPOAgent(PPOAgent):
    """PPOAgent with a conv encoder. Observations are stored flat (so the buffer, GAE and
    update code are untouched) and reshaped to (C, W, W) inside the forward pass."""

    def __init__(
        self,
        in_channels: int,
        view_size: int,
        n_actions: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
        width: int = 32,
    ) -> None:
        obs_dim = in_channels * view_size * view_size
        self.in_channels = in_channels
        self.view_size = view_size
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, config=config, device=device)
        if TORCH_AVAILABLE:
            import torch.optim as optim
            self.model = _ConvShim(
                _ConvPolicyValue(in_channels, view_size, n_actions, width),
                in_channels, view_size,
            ).to(self.device_t)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)


if TORCH_AVAILABLE:
    class _ConvShim(nn.Module):
        """Accepts flat observations and reshapes them for the conv net, so PPOAgent's
        buffer/update code needs no changes."""

        def __init__(self, net: "_ConvPolicyValue", c: int, w: int):
            super().__init__()
            self.net = net
            self.c, self.w = c, w

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            if x.dim() == 2:
                x = x.view(-1, self.c, self.w, self.w)
            elif x.dim() == 1:
                x = x.view(1, self.c, self.w, self.w)
            return self.net(x)
