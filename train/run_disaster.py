"""Episode runner for the disaster response environment.

Mirrors train/run_simulation.py but for DisasterEnv: flattens the multi-channel observation,
applies the mandate reward r_i = alpha*own + (1-alpha)*team_avg, stores PPO transitions, and
bootstraps the value target on truncation (the same correctness fix used in the paper branch).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from env.disaster_env import DisasterEnv
from env.disaster_rewards import mandate_rewards


def flatten_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {a: o.reshape(-1).astype(np.float32) for a, o in obs.items()}


def run_disaster_episode(
    env: DisasterEnv,
    policy,                       # PPOAgent / IndependentPPO / RandomPolicy / GreedyPolicy
    episode_num: int = 0,
    alpha: float = 1.0,
    train_policy: bool = True,
    episode_seed: Optional[int] = None,
    render_dir: Optional[str] = None,
    render_every: int = 0,
    credit: str = "agency",
    deterministic: bool = False,
) -> Dict:
    seed = episode_seed if episode_seed is not None else episode_num
    raw_obs, _ = env.reset(seed=seed)
    obs = flatten_obs(raw_obs)

    learning = getattr(policy, "is_learning", True)
    multi = getattr(policy, "is_multi", False)      # IndependentPPO exposes this
    recurrent = hasattr(policy, "reset_hidden")     # RecurrentPPOAgent
    if recurrent:
        policy.reset_hidden()                       # memory must not leak across episodes
    if learning and train_policy:
        policy.reset_buffer()
    if not learning:
        policy.reset()

    frames: List[str] = []
    step = 0

    while True:
        actions: Dict[str, int] = {}
        log_probs: Dict[str, float] = {}
        values: Dict[str, float] = {}
        step_obs: Dict[str, np.ndarray] = {}

        for a in env.agents:
            flat = obs[a]
            if learning:
                needs_id = multi or recurrent
                if deterministic:
                    act = (policy.select_action_greedy(flat, a) if needs_id
                           else policy.select_action_greedy(flat))
                    lp, val = 0.0, 0.0
                elif needs_id:
                    act, lp, val = policy.select_action(flat, a)
                else:
                    act, lp, val = policy.select_action(flat)
                log_probs[a] = float(lp)
                values[a] = float(val)
                step_obs[a] = flat
            else:
                act = policy.act(raw_obs[a], a)
            actions[a] = int(act)

        raw_next, raw_rewards, terminations, truncations, _ = env.step(actions)

        # the mandate: alpha * (own agency's mean) + (1-alpha) * (collective mean)
        shaped = mandate_rewards(raw_rewards, env.agency_of, alpha, credit=credit)

        if learning and train_policy:
            for a in env.agents:
                done = bool(terminations[a])   # truncation is NOT terminal (bootstrap instead)
                kwargs = dict(
                    obs=step_obs[a],
                    action=actions[a],
                    log_prob=log_probs[a],
                    reward=float(shaped[a]),
                    done=done,
                    value=values[a],
                    trajectory_id=a,
                )
                if multi:
                    policy.store_transition(agent_id=a, **kwargs)
                else:
                    policy.store_transition(**kwargs)

        if render_dir and render_every and step % render_every == 0:
            from env.disaster_render import save_frame
            frames.append(save_frame(env, f"{render_dir}/ep{episode_num:05d}_t{step:04d}.png"))

        raw_obs = raw_next
        obs = flatten_obs(raw_obs)
        step += 1

        if all(terminations.values()) or all(truncations.values()):
            break

    ppo_metrics = None
    if learning and train_policy:
        terminated = all(terminations.values())
        if terminated:
            last_value, last_done = 0.0, True
        else:
            last_value = {}
            for a in env.agents:
                last_value[a] = (
                    policy.get_value(obs[a], a) if (multi or recurrent)
                    else policy.get_value(obs[a])
                )
            last_done = False
        try:
            ppo_metrics = policy.update(last_value=last_value, last_done=last_done)
        except RuntimeError as exc:
            print(f"[ppo] update skipped: {exc}")

    m = env.get_metrics()
    m["episode"] = episode_num
    m["alpha"] = alpha
    m["ppo_metrics"] = ppo_metrics
    m["frames"] = frames
    m["total_shaped_reward"] = float(sum(shaped.values()))
    return m
