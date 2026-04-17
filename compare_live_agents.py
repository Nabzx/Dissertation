"""
Live Hunger Games arena comparison for multiple PPO policy types.

The environment contains multiple agents at the same time, but each agent is
assigned to a policy group:
- pretrained: loaded from a curriculum/simple-environment checkpoint
- scratch: fresh random PPO network
- trained: optional long-trained checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from env.gridworld_env import GridWorldEnv
from live_renderer import LiveEpisodeRenderer
from testing.ppo_agent import PPOAgent, TORCH_AVAILABLE
from testing.rewards import apply_reward_scheme


ACTION_LABELS = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


def build_agent_assignments(env: GridWorldEnv, trained_checkpoint: Optional[str]) -> Dict[str, str]:
    if len(env.agents) < 4:
        raise ValueError("compare_live_agents.py expects at least 4 agents.")

    assignments = {
        env.agents[0]: "pretrained",
        env.agents[1]: "pretrained",
        env.agents[2]: "scratch",
        env.agents[3]: "trained" if trained_checkpoint else "scratch",
    }
    for agent in env.agents[4:]:
        assignments[agent] = "scratch"
    return assignments


def build_agent_styles(assignments: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    colors = {
        "pretrained": [("#00d9ff", "#8ff3ff"), ("#39ff88", "#b6ffd2")],
        "scratch": [("#64748b", "#94a3b8"), ("#78716c", "#a8a29e")],
        "trained": [("#facc15", "#fde68a"), ("#fb923c", "#fed7aa")],
    }
    seen = {"pretrained": 0, "scratch": 0, "trained": 0}
    styles: Dict[str, Dict[str, str]] = {}
    for agent, policy_type in assignments.items():
        palette = colors[policy_type]
        idx = seen[policy_type] % len(palette)
        seen[policy_type] += 1
        color, edge_color = palette[idx]
        styles[agent] = {
            "color": color,
            "edge_color": edge_color,
            "label": f"{agent} {policy_type}",
        }
    return styles


def create_policy_pool(
    obs_dim: int,
    action_dim: int,
    checkpoint: str,
    trained_checkpoint: Optional[str],
    device: str,
) -> Dict[str, PPOAgent]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("compare_live_agents.py requires PyTorch.")

    if not os.path.exists(checkpoint):
        print("Pretrained checkpoint not found.")
        print("Run pretraining first:")
        print("python3 train_simple_env.py --num-episodes 5000")
        print(f"Expected checkpoint path: {checkpoint}")
        sys.exit(1)

    print(f"Loading pretrained checkpoint: {checkpoint}")
    pretrained = PPOAgent.load(checkpoint, device=device)
    validate_policy_shape("pretrained", pretrained, obs_dim, action_dim)

    scratch = PPOAgent(obs_dim=obs_dim, n_actions=action_dim, device=device)

    policies = {
        "pretrained": pretrained,
        "scratch": scratch,
    }

    if trained_checkpoint:
        if not os.path.exists(trained_checkpoint):
            print(f"Optional trained checkpoint not found, skipping: {trained_checkpoint}")
        else:
            print(f"Loading trained checkpoint: {trained_checkpoint}")
            trained = PPOAgent.load(trained_checkpoint, device=device)
            validate_policy_shape("trained", trained, obs_dim, action_dim)
            policies["trained"] = trained

    for policy in policies.values():
        policy.model.eval()
    return policies


def validate_policy_shape(label: str, policy: PPOAgent, obs_dim: int, action_dim: int) -> None:
    if policy.obs_dim != obs_dim or policy.n_actions != action_dim:
        raise ValueError(
            f"{label} checkpoint shape does not match arena. "
            f"Checkpoint=({policy.obs_dim}, {policy.n_actions}); "
            f"arena=({obs_dim}, {action_dim})."
        )


def run_live_comparison(
    checkpoint: str,
    trained_checkpoint: Optional[str],
    learning_mode: str,
    num_episodes: int,
    grid_size: int,
    num_agents: int,
    num_resources: int,
    num_obstacles: int,
    max_steps: int,
    reward_scheme: str,
    render_delay: float,
    output_dir: str,
    device: str,
) -> List[Dict]:
    env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=num_resources,
        num_obstacles=num_obstacles,
        max_steps=max_steps,
    )
    obs_dim = int(np.prod(env.observation_spaces[env.agents[0]].shape))
    action_dim = int(env.action_spaces[env.agents[0]].n)
    policies = create_policy_pool(obs_dim, action_dim, checkpoint, trained_checkpoint, device)
    assignments = build_agent_assignments(env, trained_checkpoint if "trained" in policies else None)
    agent_styles = build_agent_styles(assignments)
    learning_enabled = learning_mode.lower() == "on"

    env.reset(seed=0)
    renderer = LiveEpisodeRenderer(
        env.grid.copy(),
        num_episodes=num_episodes,
        max_possible_reward=float(num_resources * len(env.agents)),
        max_resources=max(env.max_resources, num_resources),
        view_size=env.view_size,
        show_perception=True,
        show_communication=False,
        obstacle_value=env.obstacle_value,
        agent_styles=agent_styles,
    )
    renderer.setup_live_speed_controls(render_delay)

    rows: List[Dict] = []
    cumulative_resources = {agent: 0 for agent in env.agents}

    for episode in range(num_episodes):
        raw_obs, _ = env.reset(seed=episode)
        obs = raw_obs
        episode_reward = 0.0
        episode_policy_rewards = {policy_type: 0.0 for policy_type in set(assignments.values())}
        episode_policy_resources = {policy_type: 0 for policy_type in set(assignments.values())}
        cumulative_collected = {agent: 0 for agent in env.agents}

        if learning_enabled:
            for policy in policies.values():
                policy.reset_buffer()

        renderer.reset_communication_visuals()
        renderer.update(
            env.grid.copy(),
            episode + 1,
            num_episodes,
            0,
            render_delay,
            actions=None,
            hud_state=build_hud_state(
                env,
                assignments,
                cumulative_resources,
                episode + 1,
                num_episodes,
                0,
                episode_reward,
                None,
                learning_enabled,
            ),
        )

        for step in range(max_steps):
            actions: Dict[str, int] = {}
            transition_data: Dict[str, Dict[str, object]] = {}

            for agent_id in env.agents:
                policy_type = assignments[agent_id]
                policy = policies[policy_type]
                flat_obs = obs[agent_id].flatten().astype(np.float32)
                action, log_prob, value = policy.select_action(flat_obs)
                actions[agent_id] = int(action)
                transition_data[agent_id] = {
                    "obs": flat_obs,
                    "log_prob": float(log_prob),
                    "value": float(value),
                }

            raw_next_obs, raw_rewards, terminations, truncations, _ = env.step(actions)
            for agent_id, reward in raw_rewards.items():
                if reward > 0.0:
                    cumulative_collected[agent_id] += 1

            shaped_rewards = apply_reward_scheme(
                scheme=reward_scheme,
                raw_rewards=raw_rewards,
                cumulative_collected=cumulative_collected,
                total_spawned=env.num_resources,
                alpha=0.5,
            )
            episode_reward += float(sum(shaped_rewards.values()))

            done_flags = {
                agent_id: bool(terminations[agent_id] or truncations[agent_id])
                for agent_id in env.agents
            }

            for agent_id in env.agents:
                policy_type = assignments[agent_id]
                if raw_rewards[agent_id] > 0.0:
                    episode_policy_resources[policy_type] += 1
                episode_policy_rewards[policy_type] += float(shaped_rewards[agent_id])

                if learning_enabled:
                    policies[policy_type].store_transition(
                        obs=transition_data[agent_id]["obs"],
                        action=actions[agent_id],
                        log_prob=transition_data[agent_id]["log_prob"],
                        reward=float(shaped_rewards[agent_id]),
                        done=done_flags[agent_id],
                        value=transition_data[agent_id]["value"],
                        trajectory_id=agent_id,
                    )

            obs = raw_next_obs
            renderer.update(
                env.grid.copy(),
                episode + 1,
                num_episodes,
                step + 1,
                render_delay,
                actions=actions,
                hud_state=build_hud_state(
                    env,
                    assignments,
                    cumulative_resources,
                    episode + 1,
                    num_episodes,
                    step + 1,
                    episode_reward,
                    actions,
                    learning_enabled,
                ),
            )

            if all(done_flags.values()):
                break

        resources = env.get_resources_collected()
        for agent_id, count in resources.items():
            cumulative_resources[agent_id] += count

        if learning_enabled:
            for policy_type, policy in policies.items():
                try:
                    metrics = policy.update(last_value=0.0, last_done=True)
                    print(
                        f"Episode {episode + 1} {policy_type}: "
                        f"policy_loss={metrics.get('policy_loss', 0.0):.4f}, "
                        f"entropy={metrics.get('entropy', 0.0):.4f}"
                    )
                except RuntimeError as exc:
                    print(f"[compare-live] PPO update skipped for {policy_type}: {exc}")

        row = {
            "episode": episode + 1,
            "learning_mode": learning_mode,
            "total_reward": episode_reward,
            "resources_collected": resources,
            "policy_rewards": episode_policy_rewards,
            "policy_resources": episode_policy_resources,
        }
        rows.append(row)
        renderer.update_learning_plot(episode_reward, sum(resources.values()), resources)
        renderer.refresh(render_delay)
        print(
            f"Episode {episode + 1}/{num_episodes}: "
            f"resources={resources}, policy_resources={episode_policy_resources}"
        )

    write_logs(rows, output_dir)
    renderer.close()
    return rows


def build_hud_state(
    env: GridWorldEnv,
    assignments: Dict[str, str],
    cumulative_resources: Dict[str, int],
    episode: int,
    total_episodes: int,
    step: int,
    episode_reward: float,
    actions: Optional[Dict[str, int]],
    learning_enabled: bool,
) -> Dict[str, object]:
    resources_now = env.get_resources_collected()
    agents_state = {}
    for agent_id in env.agents:
        agents_state[agent_id] = {
            "resources": resources_now.get(agent_id, 0),
            "cumulative_resources": cumulative_resources.get(agent_id, 0) + resources_now.get(agent_id, 0),
            "position": env.agent_positions.get(agent_id, ("-", "-")),
            "facing": "live",
            "communication": "disabled",
            "agent_type": assignments[agent_id],
            "recent_action": ACTION_LABELS.get(actions.get(agent_id, 0), "n/a") if actions else "n/a",
        }

    phase = "Learning Compare" if learning_enabled else "Fixed Compare"
    return {
        "phase": phase,
        "episode": episode,
        "total_episodes": total_episodes,
        "step": step,
        "episode_reward": episode_reward,
        "agents": agents_state,
    }


def write_logs(rows: List[Dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "compare_live_agents.json")
    csv_path = os.path.join(output_dir, "compare_live_agents.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    flat_rows = []
    for row in rows:
        flat = row.copy()
        flat["resources_collected"] = json.dumps(row["resources_collected"], sort_keys=True)
        flat["policy_rewards"] = json.dumps(row["policy_rewards"], sort_keys=True)
        flat["policy_resources"] = json.dumps(row["policy_resources"], sort_keys=True)
        flat_rows.append(flat)

    with open(csv_path, "w", newline="") as f:
        fieldnames = (
            list(flat_rows[0].keys())
            if flat_rows
            else [
                "episode",
                "learning_mode",
                "total_reward",
                "resources_collected",
                "policy_rewards",
                "policy_resources",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Saved live comparison JSON: {json_path}")
    print(f"Saved live comparison CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live visual comparison of PPO agent types.")
    parser.add_argument("--checkpoint", default="checkpoints/simple_env/ppo_latest.pt")
    parser.add_argument("--trained-checkpoint", default=None)
    parser.add_argument("--learning-mode", choices=("on", "off"), default="off")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=25)
    parser.add_argument("--num-obstacles", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--render-delay", type=float, default=0.01)
    parser.add_argument("--output-dir", default="results/live_agent_comparison")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live_comparison(
        checkpoint=args.checkpoint,
        trained_checkpoint=args.trained_checkpoint,
        learning_mode=args.learning_mode,
        num_episodes=args.num_episodes,
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_resources=args.num_resources,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
        reward_scheme=args.reward_scheme,
        render_delay=args.render_delay,
        output_dir=args.output_dir,
        device=args.device,
    )
