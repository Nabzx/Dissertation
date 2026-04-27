from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CommunicationConfig:

    max_ints: int = 8  # max message length (acts like bandwidth limit)
    grid_size: int = 15  # used to clip dx/dy so values stay reasonable


@dataclass
class CommunicationState:
    last_messages: Dict[str, np.ndarray] = field(default_factory=dict)  
    # stores the most recent message each agent has received


class CommunicationLayer:
    def __init__(self, env, config: CommunicationConfig | None = None):
        self.env = env  # keep reference to env to read positions/resources
        self.config = config or CommunicationConfig(
            grid_size=getattr(env, "grid_size", 15)
        )  # use given config or fallback to default
        self.state = CommunicationState()  # tracks message state between agents

        zero_msg = np.zeros(self.config.max_ints, dtype=np.float32)  # empty message template
        for agent in env.agents:
            self.state.last_messages[agent] = zero_msg.copy()  
            # initialise each agent with an empty message


    def _compute_message_for_agent(self, agent_id: str) -> np.ndarray:
        agent_index = self.env.agents.index(agent_id)  
        # convert agent id into a small integer (easier to send in message)

        agent_pos = self.env.agent_positions[agent_id]
        resources = list(self.env.resource_positions)  
        # copy resource list so we can safely iterate

        if resources:
            dists: List[tuple[int, tuple[int, int]]] = []
            for r in resources:
                dx = r[0] - agent_pos[0]
                dy = r[1] - agent_pos[1]
                d = abs(dx) + abs(dy)  
                # manhattan distance (cheaper than euclidean and fine here)
                dists.append((d, r))

            dists.sort(key=lambda x: x[0])  
            # sort resources by distance (closest first)

            nearest = dists[0][1]  
            # pick the closest resource

            dx = nearest[0] - agent_pos[0]
            dy = nearest[1] - agent_pos[1]
        else:
            dx, dy = 0, 0  
            # if no resources left just send zeros

        dx = int(np.clip(dx, -self.config.grid_size, self.config.grid_size))  
        dy = int(np.clip(dy, -self.config.grid_size, self.config.grid_size))  
        # clip so values don't get too large (keeps things stable)

        remaining = len(resources)  
        # how many resources are left in total

        msg = np.array([agent_index, dx, dy, remaining], dtype=np.float32)
        return self.truncate_message(msg)  
        # make sure message fits within bandwidth limit


    def truncate_message(self, msg: np.ndarray) -> np.ndarray:
        """Enforce the bandwidth limit by truncating the message length."""
        if msg.size > self.config.max_ints:
            return msg[: self.config.max_ints].astype(np.float32)  
            # cut off anything beyond max_ints

        if msg.size < self.config.max_ints:
            padded = np.zeros(self.config.max_ints, dtype=np.float32)  
            padded[: msg.size] = msg
            return padded  
            # pad with zeros so all messages have same size (easier to concat)

        return msg.astype(np.float32)  
        # already correct size


    def reset(self) -> None:
        zero_msg = np.zeros(self.config.max_ints, dtype=np.float32)  
        # reset all messages back to empty at start of episode
        for agent in self.env.agents:
            self.state.last_messages[agent] = zero_msg.copy()


    def build_augment_observation(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        augmented: Dict[str, np.ndarray] = {}
        for agent, obs in raw_obs.items():
            flat = obs.astype(np.float32).flatten()  
            # flatten observation so we can append message easily

            msg = self.state.last_messages.get(agent)  
            # get most recent message this agent received

            if msg is None:
                msg = np.zeros(self.config.max_ints, dtype=np.float32)  
                # fallback just in case (shouldn't normally happen)

            augmented[agent] = np.concatenate([flat, msg], axis=0)  
            # final observation = original obs + message

        return augmented


    def update_messages_after_step(self, active_agents: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        senders = active_agents if active_agents is not None else self.env.agents  
        # decide which agents are sending messages this step

        new_messages: Dict[str, np.ndarray] = {}
        for agent in senders:
            new_messages[agent] = self._compute_message_for_agent(agent)  
            # compute fresh message for each sender

        changed_messages: Dict[str, np.ndarray] = {}

        for agent in senders:
            for other in self.env.agents:
                if other == agent:
                    continue  # skip sending to self

                previous_msg = self.state.last_messages.get(other)

                if previous_msg is None or not np.array_equal(previous_msg, new_messages[agent]):
                    changed_messages[agent] = new_messages[agent].copy()  
                    # only track messages that actually changed (useful for visualisation/debug)

                self.state.last_messages[other] = new_messages[agent].copy()  
                # deliver message to the other agent

        return changed_messages