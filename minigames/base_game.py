"""Small wrapper layer for arena-specific minigames."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


AgentInfo = Dict[str, Any]


class ArenaScenario:
    """Base class for rule sets that sit on top of ``GridWorldEnv``."""

    def __init__(self, env):
        self.env = env
        self.last_actions: Optional[Dict[str, int]] = None
        self.last_observations: Optional[AgentInfo] = None
        self.last_raw_rewards: Optional[Dict[str, float]] = None
        self.last_terminations: Optional[Dict[str, bool]] = None
        self.last_truncations: Optional[Dict[str, bool]] = None
        self.last_infos: Optional[AgentInfo] = None

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_arena_rewards(self):
        raise NotImplementedError

    def is_round_done(self):
        raise NotImplementedError

    def get_metrics(self):
        return {}

    def get_arena_render_info(self):
        return {}

    # Backward-compatible names used by the first minigame wrapper.
    def compute_rewards(self):
        return self.compute_arena_rewards()

    def is_done(self):
        return self.is_round_done()

    def get_render_info(self):
        return self.get_arena_render_info()


class ArenaScenarioWrapper:
    """Let an arena scenario adjust rewards, done flags, metrics, and render info."""

    def __init__(self, env, arena_scenario: ArenaScenario):
        if arena_scenario.env is not env:
            raise ValueError("GameModeWrapper requires game_mode.env to be the wrapped env.")

        self.env = env
        self.arena_scenario = arena_scenario
        self.game_mode = arena_scenario
        self.agents = env.agents
        self.possible_agents = getattr(env, "possible_agents", env.agents)
        self.action_spaces = env.action_spaces
        self.observation_spaces = env.observation_spaces
        self.metadata = getattr(env, "metadata", {})

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        scenario_result = self.arena_scenario.reset()

        if scenario_result is not None:
            observations, infos = self._merge_reset_result(observations, infos, scenario_result)

        return observations, self._attach_scenario_info(infos)

    def step(self, actions: Dict[str, int]):
        observations, raw_rewards, terminations, truncations, infos = self.env.step(actions)

        self.arena_scenario.last_actions = actions
        self.arena_scenario.last_observations = observations
        self.arena_scenario.last_raw_rewards = raw_rewards
        self.arena_scenario.last_terminations = terminations
        self.arena_scenario.last_truncations = truncations
        self.arena_scenario.last_infos = infos

        scenario_result = self.arena_scenario.step(actions)
        if scenario_result is not None:
            observations, raw_rewards, terminations, truncations, infos = self._merge_step_result(
                observations,
                raw_rewards,
                terminations,
                truncations,
                infos,
                scenario_result,
            )

        rewards = self.arena_scenario.compute_rewards()
        if rewards is None:
            rewards = raw_rewards

        terminations, truncations = self._apply_done_override(
            self.arena_scenario.is_done(),
            terminations,
            truncations,
        )

        return observations, rewards, terminations, truncations, self._attach_scenario_info(infos)

    def get_metrics(self) -> Dict[str, Any]:
        return self.arena_scenario.get_metrics()

    def get_render_info(self) -> Dict[str, Any]:
        return self.arena_scenario.get_render_info()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _attach_scenario_info(self, infos: AgentInfo) -> AgentInfo:
        metrics = self.arena_scenario.get_metrics()
        render_info = self.arena_scenario.get_render_info()
        for agent in self.env.agents:
            agent_info = dict(infos.get(agent, {}))
            agent_info["game_mode"] = self.arena_scenario.__class__.__name__
            agent_info["game_metrics"] = metrics
            agent_info["render_info"] = render_info
            infos[agent] = agent_info
        return infos

    def _apply_done_override(
        self,
        scenario_done,
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        if scenario_done is None:
            return terminations, truncations

        if isinstance(scenario_done, bool):
            if scenario_done:
                terminations = {agent: True for agent in self.env.agents}
            return terminations, truncations

        if isinstance(scenario_done, dict):
            for agent, done in scenario_done.items():
                if agent in terminations:
                    terminations[agent] = bool(terminations[agent] or done)
            return terminations, truncations

        raise TypeError("GameMode.is_done() must return None, bool, or a dict of agent flags.")

    def _merge_reset_result(self, observations: AgentInfo, infos: AgentInfo, reset_result):
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
        observations: AgentInfo,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        infos: AgentInfo,
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


class GameMode(ArenaScenario):
    """Compatibility name for older minigame code."""


class GameModeWrapper(ArenaScenarioWrapper):
    """Compatibility wrapper name used by existing runners."""
