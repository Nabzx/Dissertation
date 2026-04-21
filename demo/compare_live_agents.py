"""Live arena demo comparing pretrained agents against scratch agents."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.gridworld_env import GridWorldEnv
from env.live_renderer import LiveEpisodeRenderer, build_communication_events_from_flags
from agents.ppo_agent import PPOAgent, TORCH_AVAILABLE
from env.rewards import apply_reward_scheme


ACTION_LABELS = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}

TEAM_SPECS = [
    {
        "key": "pretrained",
        "label": "Pretrained",
        "title": "PRETRAINED TEAM (RED)",
        "color": "#dc2626",
    },
    {
        "key": "scratch",
        "label": "Scratch",
        "title": "SCRATCH TEAM (BLUE)",
        "color": "#2563eb",
    },
]


def build_agent_assignments(env: GridWorldEnv) -> Dict[str, str]:
    if len(env.agents) < 4:
        raise ValueError("compare_live_agents.py expects at least 4 agents.")

    assignments = {
        env.agents[0]: "pretrained",
        env.agents[1]: "pretrained",
        env.agents[2]: "scratch",
        env.agents[3]: "scratch",
    }
    for agent in env.agents[4:]:
        assignments[agent] = "scratch"
    return assignments


def build_agent_styles(assignments: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    colors = {
        "pretrained": [("#ef4444", "#fecaca"), ("#b91c1c", "#fca5a5")],
        "scratch": [("#3b82f6", "#bfdbfe"), ("#1d4ed8", "#93c5fd")],
    }
    seen = {"pretrained": 0, "scratch": 0}
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


def build_team_agents(env: GridWorldEnv) -> Dict[str, List[str]]:
    return {
        "pretrained": list(env.agents[:2]),
        "scratch": list(env.agents[2:4]),
    }


def build_team_metrics(
    team_agents: Dict[str, List[str]],
    resources: Dict[str, int],
    step: int,
    episode_resource_history: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for team_key, agents in team_agents.items():
        total = float(sum(resources.get(agent_id, 0) for agent_id in agents))
        history = episode_resource_history.get(team_key, [])
        metrics[team_key] = {
            "total_resources": total,
            "average_per_agent": total / max(1, len(agents)),
            "efficiency": total / max(1, step),
            "average_per_episode": float(np.mean(history)) if history else 0.0,
        }
    return metrics


def build_comparison_summary(team_metrics: Dict[str, Dict[str, float]]) -> str:
    pretrained_total = float(team_metrics.get("pretrained", {}).get("total_resources", 0.0))
    scratch_total = float(team_metrics.get("scratch", {}).get("total_resources", 0.0))
    gap = pretrained_total - scratch_total
    if gap > 0:
        leader = "Pretrained agents outperforming scratch agents"
    elif gap < 0:
        leader = "Scratch agents outperforming pretrained agents"
    else:
        leader = "Pretrained and scratch agents are currently tied"
    return f"{leader}\nPerformance gap: {gap:+.0f} resources"


def create_policy_pool(
    obs_dim: int,
    action_dim: int,
    checkpoint: str,
    device: str,
) -> Dict[str, PPOAgent]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("compare_live_agents.py requires PyTorch.")

    if not os.path.exists(checkpoint):
        print("Pretrained checkpoint not found.")
        print("Run pretraining first:")
        print("python3 train/train_headless.py --num-episodes 50000")
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
    policies = create_policy_pool(obs_dim, action_dim, checkpoint, device)
    assignments = build_agent_assignments(env)
    agent_styles = build_agent_styles(assignments)
    team_agents = build_team_agents(env)
    learning_enabled = learning_mode.lower() == "on"

    env.reset(seed=0)
    renderer = LiveEpisodeRenderer(
        env.grid.copy(),
        num_episodes=num_episodes,
        max_possible_reward=float(num_resources * len(env.agents)),
        max_resources=max(env.max_resources, num_resources),
        view_size=env.view_size,
        show_perception=True,
        show_communication=True,
        obstacle_value=env.obstacle_value,
        agent_styles=agent_styles,
    )
    renderer.setup_team_comparison_view(TEAM_SPECS)
    renderer.setup_live_layout()

    rows: List[Dict] = []
    cumulative_resources = {agent: 0 for agent in env.agents}
    episode_resource_history = {str(team["key"]): [] for team in TEAM_SPECS}
    win_counts = {str(team["key"]): 0 for team in TEAM_SPECS}

    for episode in range(num_episodes):
        raw_obs, _ = env.reset(seed=episode)
        obs = raw_obs
        episode_reward = 0.0
        episode_policy_rewards = {policy_type: 0.0 for policy_type in set(assignments.values())}
        episode_policy_resources = {policy_type: 0 for policy_type in set(assignments.values())}
        episode_policy_comms = {policy_type: 0 for policy_type in set(assignments.values())}
        cumulative_collected = {agent: 0 for agent in env.agents}

        if learning_enabled:
            for policy in policies.values():
                policy.reset_buffer()

        renderer.reset_communication_visuals()
        team_metrics = build_team_metrics(team_agents, env.get_resources_collected(), 0, episode_resource_history)
        renderer.update_team_comparison_view(team_metrics, win_counts, build_comparison_summary(team_metrics))
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
            communication_events = build_communication_events_from_flags(env)
            for event in communication_events:
                sender_id = env.agent_id_from_value(int(event["sender"]))
                if sender_id is not None:
                    episode_policy_comms[assignments[sender_id]] += 1
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
            team_metrics = build_team_metrics(team_agents, env.get_resources_collected(), step + 1, episode_resource_history)
            renderer.update_team_comparison_view(team_metrics, win_counts, build_comparison_summary(team_metrics))
            renderer.update(
                env.grid.copy(),
                episode + 1,
                num_episodes,
                step + 1,
                render_delay,
                actions=actions,
                communication_events=communication_events,
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
                    communication_events,
                ),
            )

            if all(done_flags.values()):
                break

        resources = env.get_resources_collected()
        for agent_id, count in resources.items():
            cumulative_resources[agent_id] += count
        team_episode_totals = {
            team_key: int(sum(resources.get(agent_id, 0) for agent_id in agents))
            for team_key, agents in team_agents.items()
        }
        for team_key, total in team_episode_totals.items():
            episode_resource_history[team_key].append(total)
        if team_episode_totals["pretrained"] > team_episode_totals["scratch"]:
            win_counts["pretrained"] += 1
        elif team_episode_totals["scratch"] > team_episode_totals["pretrained"]:
            win_counts["scratch"] += 1

        if learning_enabled:
            for policy_type in sorted(set(assignments.values())):
                policy = policies[policy_type]
                try:
                    policy.update(last_value=0.0, last_done=True)
                except RuntimeError as exc:
                    print(f"[compare-live] PPO update skipped for {policy_type}: {exc}")

        row = {
            "episode": episode + 1,
            "learning_mode": learning_mode,
            "total_reward": episode_reward,
            "resources_collected": resources,
            "policy_rewards": episode_policy_rewards,
            "policy_resources": episode_policy_resources,
            "policy_comms": episode_policy_comms,
        }
        rows.append(row)
        team_metrics = build_team_metrics(team_agents, resources, max(1, env.step_count), episode_resource_history)
        renderer.update_team_comparison_view(team_metrics, win_counts, build_comparison_summary(team_metrics))
        renderer.refresh(render_delay)
        print(
            f"[Pretrained] ep {episode + 1}: "
            f"reward={episode_policy_rewards.get('pretrained', 0.0):.2f}, "
            f"comms={episode_policy_comms.get('pretrained', 0)}"
        )
        print(
            f"[Scratch] ep {episode + 1}: "
            f"reward={episode_policy_rewards.get('scratch', 0.0):.2f}, "
            f"comms={episode_policy_comms.get('scratch', 0)}"
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
    communication_events: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    resources_now = env.get_resources_collected()
    communicating_agents = set()
    if communication_events:
        for event in communication_events:
            communicating_agents.add(env.agent_id_from_value(int(event["sender"])))
    agents_state = {}
    for agent_id in env.agents:
        agents_state[agent_id] = {
            "resources": resources_now.get(agent_id, 0),
            "cumulative_resources": cumulative_resources.get(agent_id, 0) + resources_now.get(agent_id, 0),
            "position": env.agent_positions.get(agent_id, ("-", "-")),
            "facing": "live",
            "communication": "sending" if agent_id in communicating_agents else "idle",
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
        flat["policy_comms"] = json.dumps(row["policy_comms"], sort_keys=True)
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
                "policy_comms",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Saved live comparison JSON: {json_path}")
    print(f"Saved live comparison CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live visual comparison of pretrained vs scratch agents.")
    parser.add_argument("--checkpoint", default="checkpoints/run_50000/ppo_latest.pt")
    parser.add_argument("--trained-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-mode", choices=("on", "off"), default="off")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-resources", type=int, default=25)
    parser.add_argument("--num-obstacles", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--reward-scheme", default="selfish")
    parser.add_argument("--render-delay", type=float, default=0.01)
    parser.add_argument("--output-dir", default="results/run_10000/live_agent_comparison")
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
