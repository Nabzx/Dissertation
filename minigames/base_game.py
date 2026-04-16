"""
Base interfaces for modular GridWorld minigames.

The minigame layer is intentionally separate from ``GridWorldEnv``. A
``GameModeWrapper`` owns an existing environment instance, lets the environment
handle core physics, and gives a ``GameMode`` a chance to customize reset
state, rewards, termination, metrics, and renderer metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


AgentDict = Dict[str, Any]


class GameMode:
    """Base class for custom arena minigames."""

    def __init__(self, env):
        self.env = env
        self.last_actions: Optional[Dict[str, int]] = None
        self.last_observations: Optional[AgentDict] = None
        self.last_raw_rewards: Optional[Dict[str, float]] = None
        self.last_terminations: Optional[Dict[str, bool]] = None
        self.last_truncations: Optional[Dict[str, bool]] = None
        self.last_infos: Optional[AgentDict] = None

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_rewards(self):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def get_metrics(self):
        return {}

    def get_render_info(self):
        """Return optional renderer metadata such as zones, flags, or scores."""
        return {}


class GameModeWrapper:
    """
    PettingZoo-style wrapper that lets a GameMode control episode rules.

    The wrapped environment still owns movement, resource spawning, collision
    handling, observations, and grid state. The game mode can adjust:
    - reset-time state
    - rewards returned to learners
    - termination flags
    - metrics and renderer-facing info dictionaries
    """

    def __init__(self, env, game_mode: GameMode):
        if game_mode.env is not env:
            raise ValueError("GameModeWrapper requires game_mode.env to be the wrapped env.")

        self.env = env
        self.game_mode = game_mode
        self.agents = env.agents
        self.possible_agents = getattr(env, "possible_agents", env.agents)
        self.action_spaces = env.action_spaces
        self.observation_spaces = env.observation_spaces
        self.metadata = getattr(env, "metadata", {})

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        reset_result = self.game_mode.reset()

        if reset_result is not None:
            observations, infos = self._merge_reset_result(observations, infos, reset_result)

        infos = self._attach_mode_info(infos)
        return observations, infos

    def step(self, actions: Dict[str, int]):
        observations, raw_rewards, terminations, truncations, infos = self.env.step(actions)

        self.game_mode.last_actions = actions
        self.game_mode.last_observations = observations
        self.game_mode.last_raw_rewards = raw_rewards
        self.game_mode.last_terminations = terminations
        self.game_mode.last_truncations = truncations
        self.game_mode.last_infos = infos

        step_result = self.game_mode.step(actions)
        if step_result is not None:
            observations, raw_rewards, terminations, truncations, infos = self._merge_step_result(
                observations,
                raw_rewards,
                terminations,
                truncations,
                infos,
                step_result,
            )

        rewards = self.game_mode.compute_rewards()
        if rewards is None:
            rewards = raw_rewards

        mode_done = self.game_mode.is_done()
        terminations, truncations = self._apply_done_override(
            mode_done,
            terminations,
            truncations,
        )

        infos = self._attach_mode_info(infos)
        return observations, rewards, terminations, truncations, infos

    def get_metrics(self) -> Dict[str, Any]:
        return self.game_mode.get_metrics()

    def get_render_info(self) -> Dict[str, Any]:
        return self.game_mode.get_render_info()

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.env, name)

    def _attach_mode_info(self, infos: AgentDict) -> AgentDict:
        metrics = self.game_mode.get_metrics()
        render_info = self.game_mode.get_render_info()
        for agent in self.env.agents:
            agent_info = dict(infos.get(agent, {}))
            agent_info["game_mode"] = self.game_mode.__class__.__name__
            agent_info["game_metrics"] = metrics
            agent_info["render_info"] = render_info
            infos[agent] = agent_info
        return infos

    def _apply_done_override(
        self,
        mode_done,
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        if mode_done is None:
            return terminations, truncations

        if isinstance(mode_done, bool):
            if mode_done:
                terminations = {agent: True for agent in self.env.agents}
            return terminations, truncations

        if isinstance(mode_done, dict):
            for agent, done in mode_done.items():
                if agent in terminations:
                    terminations[agent] = bool(terminations[agent] or done)
            return terminations, truncations

        raise TypeError("GameMode.is_done() must return None, bool, or a dict of agent flags.")

    def _merge_reset_result(self, observations: AgentDict, infos: AgentDict, reset_result):
        if isinstance(reset_result, tuple):
            if len(reset_result) != 2:
                raise ValueError("GameMode.reset() tuple result must be (observations, infos).")
            return reset_result

        if isinstance(reset_result, dict):
            observations = reset_result.get("observations", observations)
            infos = reset_result.get("infos", infos)
            return observations, infos

        raise TypeError("GameMode.reset() must return None, a dict, or (observations, infos).")

    def _merge_step_result(
        self,
        observations: AgentDict,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        infos: AgentDict,
        step_result,
    ):
        if isinstance(step_result, tuple):
            if len(step_result) != 5:
                raise ValueError(
                    "GameMode.step() tuple result must be "
                    "(observations, rewards, terminations, truncations, infos)."
                )
            return step_result

        if isinstance(step_result, dict):
            observations = step_result.get("observations", observations)
            rewards = step_result.get("rewards", rewards)
            terminations = step_result.get("terminations", terminations)
            truncations = step_result.get("truncations", truncations)
            infos = step_result.get("infos", infos)
            return observations, rewards, terminations, truncations, infos

        raise TypeError("GameMode.step() must return None, a dict, or a 5-item step tuple.")
