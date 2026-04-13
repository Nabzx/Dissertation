"""
Run a persistent live PPO training demo.
"""

from __future__ import annotations

from live_renderer import run_live_training


if __name__ == "__main__":
    run_live_training(num_episodes=50, render_delay=0.01)
