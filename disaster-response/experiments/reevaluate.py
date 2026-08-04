"""Re-evaluate trained checkpoints on held-out seeds.

Needed because the first sweep evaluated with argmax actions, which understates performance
in this task by roughly 40% (see results/eval_mode.md): a deterministic policy loops in a
search problem, while a stochastic one keeps covering ground. Rather than retrain, we reload
each run's final checkpoint and evaluate again.

Writes `eval_stochastic.json` beside each run's summary, leaving the original summary intact.

Usage:
  python reevaluate.py --episodes 8000 --n 100
  python reevaluate.py --pattern 'disaster_8000_a*' --n 50
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from env.disaster_env import DisasterEnv, N_ACTIONS          # noqa: E402
from agents.ppo_agent import PPOAgent, set_global_seeds       # noqa: E402
from train.run_disaster import run_disaster_episode           # noqa: E402

EVAL_BASE = 10_000_000     # held-out seed range, disjoint from training


def load_policy(ckpt: str, comm: bool):
    if comm:
        from agents.comm_ppo import CommPPOAgent
        base = PPOAgent.load(ckpt)
        agent = CommPPOAgent(obs_dim=base.obs_dim, n_actions=base.n_actions)
        agent.model.load_state_dict(base.model.state_dict(), strict=False)
        return agent
    return PPOAgent.load(ckpt)


def evaluate(run_dir: Path, n: int, deterministic: bool) -> Optional[Dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    cfg = json.loads(summary_path.read_text())

    ckpts = sorted((REPO / "disaster-response" / "runs" / "checkpoints" / run_dir.name).glob("*.pt"))
    final = [c for c in ckpts if c.name == "final.pt"] or ckpts
    if not final:
        return None

    comm = "_comm" in run_dir.name
    set_global_seeds(1234)
    env = DisasterEnv(grid_size=50, num_agents=6, num_agencies=2, num_victims=40,
                      max_steps=300, view_size=9, communication=comm)
    policy = load_policy(str(final[-1]), comm)

    rows = []
    for i in range(n):
        m = run_disaster_episode(
            env=env, policy=policy, episode_num=i, alpha=cfg["alpha"],
            train_policy=False, episode_seed=EVAL_BASE + cfg["seed"] * n + i,
            deterministic=deterministic,
        )
        rows.append(m)

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    return {
        "run_name": run_dir.name,
        "alpha": cfg["alpha"],
        "seed": cfg["seed"],
        "deterministic": deterministic,
        "n": n,
        "lives_saved": mean("lives_saved"),
        "save_rate": mean("save_rate"),
        "severe_save_rate": mean("severe_save_rate"),
        "minor_save_rate": mean("minor_save_rate"),
        "joint_rescues": mean("joint_rescues"),
        "sharing_rate": mean("sharing_rate") if comm else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--runs-root", default=str(here / ".." / "runs"))
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    root = Path(args.runs_root).resolve()
    pattern = args.pattern or f"disaster_{args.episodes}_a*"
    dirs = sorted(d for d in root.glob(pattern) if d.is_dir())
    if not dirs:
        print(f"no runs matching {pattern} under {root}")
        return

    out_name = "eval_deterministic.json" if args.deterministic else "eval_stochastic.json"
    print(f"{'run':<34} {'lives':>7} {'severe':>7} {'minor':>7}")
    for d in dirs:
        res = evaluate(d, args.n, args.deterministic)
        if res is None:
            print(f"{d.name:<34} (skipped: no checkpoint)")
            continue
        (d / out_name).write_text(json.dumps(res, indent=2))
        print(f"{d.name:<34} {res['lives_saved']:>7.2f} "
              f"{res['severe_save_rate']:>7.3f} {res['minor_save_rate']:>7.3f}")


if __name__ == "__main__":
    main()
