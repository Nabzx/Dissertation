"""Run one PPO episode and animate the resulting grid sequence."""

from __future__ import annotations

import numpy as np

from env.gridworld_env import GridWorldEnv
from testing.ppo_agent import PPOAgent
from train.run_simulation import run_episode
from visualize_episode import animate_episode


def run_animation_test(save_path: str | None = None):
    """Run one PPO episode and return the captured grid sequence."""
    agent_type = "ppo"
    reward_scheme = "selfish"
    communication = False

    env = GridWorldEnv(grid_size=25, num_resources=25, num_obstacles=45, max_steps=250)

    obs_shape = env.observation_spaces[env.agents[0]].shape
    obs_dim = int(np.prod(obs_shape))
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
