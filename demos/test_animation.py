from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.gridworld_env import GridWorldEnv
from agents.ppo_agent import PPOAgent
from train.run_simulation import run_episode
from demos.visualise_episode import animate_episode


def run_animation_test(save_path: str | None = None):
    agent_type = "ppo"
    reward_scheme = "selfish"
    communication = False

    env = GridWorldEnv(grid_size=25, num_resources=25, num_obstacles=45, max_steps=250)

    obs_size = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_size))
    action_dim = int(env.action_spaces[env.agents[0]].n)
    ppo_agent = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device="cpu")

    episode_data = run_episode(
        env=env,
        agents=None,
        episode_num=0,
        save_screenshots=False,
        save_heatmaps=False,
        agent_type=agent_type,
        ppo_agent=ppo_agent,
        reward_scheme=reward_scheme,
        use_communication=communication,
        render=True,
    )

    grid_sequence = episode_data["grid_sequence"]
    animate_episode(grid_sequence, save_path=save_path)
    return grid_sequence


if __name__ == "__main__":
    run_animation_test()
