"""
Run a persistent live PPO training demo.
"""

from __future__ import annotations

from live_renderer import run_live_training


if __name__ == "__main__":
    run_live_training(num_episodes=5000, render_every=200, render_delay=0.001, fast_mode=False, mode="playback")
    #run_live_training(num_episodes=1000, mode="playback")
