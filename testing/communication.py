"""
Simple bandwidth-limited communication layer for the testing pipeline.

The goal is to demonstrate the *idea* of communication, not to build a
full multi-agent communication framework. Each agent can send a short
message vector every step, which is then concatenated to the other
agent's observation before it is passed to the policy.

Message definition (per step, per agent):
    [agent_id, nearest_resource_dx, nearest_resource_dy, remaining_resources_count]

Bandwidth limit:
    - We interpret the bandwidth limit as a maximum number of integers.
    - By default we only send the first 4 entries, which easily fits into
      a "max 8 integers" budget.
    - If you increase the message, the `truncate_message` helper will clip
      the length automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class CommunicationConfig:
    """Configuration for the communication layer."""

    max_ints: int = 8  # "bandwidth" in integers
    grid_size: int = 15  # used for simple clipping of dx / dy


@dataclass
class CommunicationState:
    """Keeps track of the most recent message each agent has received."""

    last_messages: Dict[str, np.ndarray] = field(default_factory=dict)


class CommunicationLayer:
    """
    Helper that computes and maintains communication messages between agents.

    This layer does **not** modify the environment. It only:
    - reads environment attributes (agent positions, resource positions),
    - computes short messages,
    - and exposes convenience functions to augment observations.
    """

    def __init__(self, env, config: CommunicationConfig | None = None):
        self.env = env
        self.config = config or CommunicationConfig(grid_size=getattr(env, "grid_size", 15))
        self.state = CommunicationState()

        # Initialise zero messages for all agents.
        zero_msg = np.zeros(self.config.max_ints, dtype=np.float32)
        for agent in env.agents:
            self.state.last_messages[agent] = zero_msg.copy()

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _compute_message_for_agent(self, agent_id: str) -> np.ndarray:
        """
        Build a simple message vector for a given agent.

        Contents:
            [agent_id_index, nearest_resource_dx, nearest_resource_dy,
             remaining_resources_count]
        """
        # Map agent IDs to small integer indices.
        agent_index = self.env.agents.index(agent_id)

        agent_pos = self.env.agent_positions[agent_id]
        resources = list(self.env.resource_positions)

        if resources:
            # Find nearest resource in Manhattan distance.
            dists: List[tuple[int, tuple[int, int]]] = []
            for r in resources:
                dx = r[0] - agent_pos[0]
                dy = r[1] - agent_pos[1]
                d = abs(dx) + abs(dy)
                dists.append((d, r))
            dists.sort(key=lambda x: x[0])
            nearest = dists[0][1]
            dx = nearest[0] - agent_pos[0]
            dy = nearest[1] - agent_pos[1]
        else:
            dx, dy = 0, 0

        # Clip deltas to the grid size for sanity.
        dx = int(np.clip(dx, -self.config.grid_size, self.config.grid_size))
        dy = int(np.clip(dy, -self.config.grid_size, self.config.grid_size))

        remaining = len(resources)

        msg = np.array([agent_index, dx, dy, remaining], dtype=np.float32)
        return self.truncate_message(msg)

    def truncate_message(self, msg: np.ndarray) -> np.ndarray:
        """Enforce the bandwidth limit by truncating the message length."""
        if msg.size > self.config.max_ints:
            return msg[: self.config.max_ints].astype(np.float32)

        # Pad to a fixed length for convenience when concatenating.
        if msg.size < self.config.max_ints:
            padded = np.zeros(self.config.max_ints, dtype=np.float32)
            padded[: msg.size] = msg
            return padded

        return msg.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API used by the testing runner
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset messages at the start of an episode."""
        zero_msg = np.zeros(self.config.max_ints, dtype=np.float32)
        for agent in self.env.agents:
            self.state.last_messages[agent] = zero_msg.copy()

    def build_augment_observation(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Concatenate the most recent *received* message to each agent's
        flattened observation.
        """
        augmented: Dict[str, np.ndarray] = {}
        for agent, obs in raw_obs.items():
            flat = obs.astype(np.float32).flatten()
            msg = self.state.last_messages.get(agent)
            if msg is None:
                msg = np.zeros(self.config.max_ints, dtype=np.float32)
            augmented[agent] = np.concatenate([flat, msg], axis=0)
        return augmented

    def update_messages_after_step(self) -> Dict[str, np.ndarray]:
        """
        After the environment step, compute *new* messages for each agent
        and deliver them to the other agent. These messages will be used
        for the next decision step.
        """
        new_messages: Dict[str, np.ndarray] = {}
        for agent in self.env.agents:
            new_messages[agent] = self._compute_message_for_agent(agent)

        # Deliver each agent's message to all *other* agents.
        for agent in self.env.agents:
            for other in self.env.agents:
                if other == agent:
                    continue
                self.state.last_messages[other] = new_messages[agent].copy()

        return new_messages
