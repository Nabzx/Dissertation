"""
Dedicated live demo for the Capture the Flag minigame.

Run with:
    python3 demo_capture_flag.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from env.gridworld_env import GridWorldEnv
from live_renderer import LiveEpisodeRenderer, PlaybackController
from minigames import CaptureFlagGame, GameModeWrapper


ACTION_LABELS = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


def make_capture_flag_env(
    grid_size: int,
    num_agents: int,
    num_obstacles: int,
    max_steps: int,
) -> GameModeWrapper:
    base_env = GridWorldEnv(
        grid_size=grid_size,
        num_agents=num_agents,
        num_resources=0,
        num_obstacles=num_obstacles,
        resource_respawn_prob=0.0,
        max_resources=0,
        max_steps=max_steps,
    )
    return GameModeWrapper(base_env, CaptureFlagGame(base_env))


def build_agent_styles(env: GameModeWrapper) -> Dict[str, Dict[str, str]]:
    colors = [
        ("#00d9ff", "#8ff3ff"),
        ("#ff315a", "#ff9aaa"),
        ("#39ff88", "#b6ffd2"),
        ("#facc15", "#fde68a"),
        ("#a78bfa", "#ddd6fe"),
        ("#fb923c", "#fed7aa"),
    ]
    styles = {}
    for idx, agent_id in enumerate(env.agents):
        color, edge_color = colors[idx % len(colors)]
        styles[agent_id] = {
            "color": color,
            "edge_color": edge_color,
            "label": f"{agent_id} racer",
        }
    return styles


def choose_flag_action(env: GameModeWrapper, agent_id: str, occupied: set[Tuple[int, int]]) -> int:
    flag_position = env.get_metrics().get("flag_position")
    if flag_position is None:
        return 0

    current = env.agent_positions[agent_id]
    best_action = 0
    best_distance = manhattan(current, flag_position)

    # Prefer direct progress, but let the env's movement helper enforce arena
    # walls and obstacles so this demo stays compatible with future arena rules.
    for action in [1, 2, 3, 4]:
        candidate = env._move_agent(current, action)
        if candidate in occupied and candidate != current:
            continue
        distance = manhattan(candidate, flag_position)
        if distance < best_distance:
            best_action = action
            best_distance = distance

    if best_action == 0:
        shuffled = [1, 2, 3, 4]
        np.random.shuffle(shuffled)
        for action in shuffled:
            candidate = env._move_agent(current, action)
            if candidate != current and candidate not in occupied:
                return action

    return best_action


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def build_hud_state(
    env: GameModeWrapper,
    episode: int,
    num_episodes: int,
    step: int,
    total_reward: float,
    actions: Optional[Dict[str, int]],
    wins: Dict[str, int],
    capture_times: List[int],
) -> Dict[str, object]:
    game_metrics = env.get_metrics()
    game_metrics = dict(game_metrics)
    game_metrics["wins"] = wins.copy()
    game_metrics["average_time_to_flag"] = (
        float(np.mean(capture_times)) if capture_times else None
    )

    agents_state = {}
    for agent_id in env.agents:
        agents_state[agent_id] = {
            "resources": 0,
            "cumulative_resources": 0,
            "position": env.agent_positions.get(agent_id, ("-", "-")),
            "facing": "flag",
            "communication": "disabled",
            "agent_type": "racer",
            "recent_action": ACTION_LABELS.get(actions.get(agent_id, 0), "n/a") if actions else "n/a",
        }

    return {
        "phase": "Capture the Flag",
        "game_mode": "capture_flag",
        "episode": episode,
        "total_episodes": num_episodes,
        "step": step,
        "episode_reward": total_reward,
        "game_metrics": game_metrics,
        "agents": agents_state,
    }


def build_history_frame(
    env: GameModeWrapper,
    episode: int,
    step: int,
    total_reward: float,
    actions: Optional[Dict[str, int]],
    wins: Dict[str, int],
    capture_times: List[int],
) -> Dict[str, object]:
    return {
        "episode": episode,
        "step": step,
        "grid": env.grid.copy(),
        "agent_positions": {
            agent_id: tuple(map(int, position))
            for agent_id, position in env.agent_positions.items()
        },
        "resource_positions": [],
        "render_info": env.get_render_info(),
        "game_metrics": build_hud_state(
            env,
            episode,
            episode,
            step,
            total_reward,
            actions,
            wins,
            capture_times,
        )["game_metrics"],
    }


def run_capture_flag_demo(
    num_episodes: int = 10,
    grid_size: int = 25,
    num_agents: int = 4,
    num_obstacles: int = 45,
    max_steps: int = 150,
    render_delay: float = 0.01,
    mode: str = "live",
) -> List[Dict[str, object]]:
    mode = mode.lower()
    if mode not in {"live", "playback", "both"}:
        raise ValueError("mode must be 'live', 'playback', or 'both'.")
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1.")

    env = make_capture_flag_env(grid_size, num_agents, num_obstacles, max_steps)
    env.reset(seed=0)

    renderer: Optional[LiveEpisodeRenderer] = None
    if mode in {"live", "both"}:
        renderer = LiveEpisodeRenderer(
            env.grid.copy(),
            num_episodes=num_episodes,
            max_possible_reward=10.0,
            max_resources=1,
            view_size=env.view_size,
            show_perception=True,
            show_communication=False,
            show_resource_animation=False,
            obstacle_value=env.obstacle_value,
            agent_styles=build_agent_styles(env),
        )
        renderer.setup_live_speed_controls(render_delay)

    wins = {agent: 0 for agent in env.agents}
    capture_times: List[int] = []
    history: List[List[Dict[str, object]]] = []
    summaries: List[Dict[str, object]] = []

    for episode_idx in range(num_episodes):
        episode = episode_idx + 1
        env.reset(seed=episode_idx)
        total_reward = 0.0
        episode_history: List[Dict[str, object]] = []
        actions: Optional[Dict[str, int]] = None

        episode_history.append(
            build_history_frame(env, episode, 0, total_reward, actions, wins, capture_times)
        )

        if renderer is not None:
            renderer.reset_communication_visuals()
            renderer.update(
                env.grid.copy(),
                episode,
                num_episodes,
                0,
                render_delay,
                actions=None,
                hud_state=build_hud_state(
                    env,
                    episode,
                    num_episodes,
                    0,
                    total_reward,
                    None,
                    wins,
                    capture_times,
                ),
                render_info=env.get_render_info(),
            )

        winner = None
        final_step = 0
        for step in range(1, max_steps + 1):
            occupied = set(env.agent_positions.values())
            actions = {
                agent_id: choose_flag_action(env, agent_id, occupied)
                for agent_id in env.agents
            }
            _obs, rewards, terminations, truncations, _infos = env.step(actions)
            total_reward += float(sum(rewards.values()))
            final_step = step
            winner = env.get_metrics().get("winner")

            if winner is not None:
                wins[winner] += 1
                capture_times.append(step)

            episode_history.append(
                build_history_frame(env, episode, step, total_reward, actions, wins, capture_times)
            )

            if renderer is not None:
                renderer.update(
                    env.grid.copy(),
                    episode,
                    num_episodes,
                    step,
                    render_delay,
                    actions=actions,
                    hud_state=build_hud_state(
                        env,
                        episode,
                        num_episodes,
                        step,
                        total_reward,
                        actions,
                        wins,
                        capture_times,
                    ),
                    render_info=env.get_render_info(),
                )

            if all(terminations.values()) or all(truncations.values()):
                break

        summary = {
            "episode": episode,
            "winner": winner,
            "time_to_flag": final_step if winner else None,
            "total_reward": total_reward,
            "resources_collected": {agent: 0 for agent in env.agents},
            "wins": wins.copy(),
            "average_time_to_flag": float(np.mean(capture_times)) if capture_times else None,
            "game_metrics": env.get_metrics(),
        }
        history.append(episode_history)
        summaries.append(summary)
        renderer_for_plot = renderer
        if renderer_for_plot is not None:
            renderer_for_plot.update_learning_plot(total_reward, 1 if winner else 0)
            renderer_for_plot.refresh(render_delay)

        print(
            f"Capture Flag episode {episode}/{num_episodes}: "
            f"winner={winner or 'none'}, steps={final_step}, wins={wins}"
        )

    if mode == "both" and renderer is not None:
        plt.close(renderer.fig)
        renderer = None

    if mode in {"playback", "both"}:
        playback_env = make_capture_flag_env(grid_size, num_agents, num_obstacles, max_steps)
        playback_env.reset(seed=0)
        playback_renderer = LiveEpisodeRenderer(
            history[0][0]["grid"],
            num_episodes=num_episodes,
            max_possible_reward=10.0,
            max_resources=1,
            view_size=playback_env.view_size,
            show_perception=True,
            show_communication=False,
            show_resource_animation=False,
            obstacle_value=playback_env.obstacle_value,
            agent_styles=build_agent_styles(playback_env),
        )
        playback_renderer.set_speed_delay(render_delay)
        controller = PlaybackController(playback_renderer, history, summaries)
        controller.run()
    elif renderer is not None:
        renderer.close()

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Capture the Flag minigame demo.")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-obstacles", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--render-delay", type=float, default=0.01)
    parser.add_argument(
        "--mode",
        choices=("live", "playback", "both"),
        default="live",
        help="live shows episodes as they run; playback records first; both does live then opens scrub controls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_capture_flag_demo(
        num_episodes=args.num_episodes,
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_obstacles=args.num_obstacles,
        max_steps=args.max_steps,
        render_delay=args.render_delay,
        mode=args.mode,
    )
