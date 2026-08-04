"""Training / evaluation driver for the disaster response environment.

Runs PPO (shared or independent policies) or a non-learning baseline, logging per-episode
metrics to CSV and a final summary to JSON. Output layout mirrors the paper branch so the
same analysis habits apply.

Examples:
  # baselines (no learning) - establishes the floor and a strong heuristic reference
  python -m train.train_disaster --policy random --episodes 200 --out-root disaster-response/runs
  python -m train.train_disaster --policy greedy --episodes 200 --out-root disaster-response/runs

  # PPO at a given mandate
  python -m train.train_disaster --policy ppo --alpha 1.0 --episodes 8000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                                    # graceful fallback
    def tqdm(it, **kw):
        return it

from env.disaster_env import DisasterEnv, N_ACTIONS
from agents.ppo_agent import PPOAgent, TORCH_AVAILABLE, set_global_seeds
from agents.disaster_baselines import RandomPolicy, GreedyPolicy
from agents.disaster_coordinated import CoordinatedGreedyPolicy
from train.run_disaster import run_disaster_episode

CSV_FIELDS = [
    "episode", "lives_saved", "lives_lost", "victims_remaining", "save_rate",
    "severe_save_rate", "minor_save_rate", "severe_saved", "minor_saved",
    "joint_rescues", "mean_idle_rate", "steps", "total_shaped_reward",
    "policy_loss", "value_loss", "entropy", "approx_kl",
]


def _row(m: Dict) -> Dict:
    p = m.get("ppo_metrics") or {}
    return {
        "episode": m["episode"],
        "lives_saved": m["lives_saved"],
        "lives_lost": m["lives_lost"],
        "victims_remaining": m["victims_remaining"],
        "save_rate": round(m["save_rate"], 5),
        "severe_save_rate": round(m["severe_save_rate"], 5),
        "minor_save_rate": round(m["minor_save_rate"], 5),
        "severe_saved": m["severe_saved"],
        "minor_saved": m["minor_saved"],
        "joint_rescues": m["joint_rescues"],
        "mean_idle_rate": round(float(np.mean(list(m["idle_rate"].values()))), 5),
        "steps": m["steps"],
        "total_shaped_reward": round(m["total_shaped_reward"], 4),
        "policy_loss": round(p.get("policy_loss", 0.0), 5),
        "value_loss": round(p.get("value_loss", 0.0), 5),
        "entropy": round(p.get("entropy", 0.0), 5),
        "approx_kl": round(p.get("approx_kl", 0.0), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ppo", choices=["ppo", "random", "greedy", "coordinated"])
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="mandate: r = alpha*own + (1-alpha)*team_avg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--independent", action="store_true", help="one network per responder")
    ap.add_argument("--encoder", default="mlp", choices=["mlp", "conv", "gru"],
                    help="conv keeps the spatial structure of the observation")
    ap.add_argument("--entropy-start", type=float, default=0.03,
                    help="entropy coefficient at the start of training")
    ap.add_argument("--entropy-end", type=float, default=0.005,
                    help="entropy coefficient at the end; annealed linearly")
    ap.add_argument("--grid-size", type=int, default=60)
    ap.add_argument("--num-agents", type=int, default=12)
    ap.add_argument("--num-agencies", type=int, default=3)
    ap.add_argument("--num-victims", type=int, default=45)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--view-size", type=int, default=9)
    ap.add_argument("--layout", default="village", choices=["village", "open"])
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--out-root", default="disaster-response/runs")
    ap.add_argument("--credit", default="agency", choices=["agency", "agent"],
                    help="agency: alpha weights the agency mean; agent: weights own reward")
    ap.add_argument("--eval-episodes", type=int, default=100,
                    help="deterministic evaluation episodes on held-out seeds after training")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_global_seeds(args.seed)

    arch = "_indep" if args.independent else ""
    tag = f"_{args.tag}" if args.tag else ""
    if args.policy == "ppo":
        enc = "" if args.encoder == "mlp" else f"_{args.encoder}"
        run_name = f"disaster_{args.episodes}_a{args.alpha:g}_seed{args.seed}{arch}{enc}{tag}"
    else:
        run_name = f"disaster_{args.policy}_{args.episodes}_seed{args.seed}{tag}"

    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.out_root) / "checkpoints" / run_name

    env = DisasterEnv(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        num_agencies=args.num_agencies,
        num_victims=args.num_victims,
        max_steps=args.max_steps,
        view_size=args.view_size,
        layout=args.layout,
    )

    obs_dim = int(np.prod(env.observation_spaces[env.agents[0]].shape))
    if args.policy == "ppo":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PPO requires PyTorch.")
        if args.independent:
            from agents.independent_ppo import IndependentPPO
            policy = IndependentPPO(agent_ids=list(env.agents), obs_dim=obs_dim, n_actions=N_ACTIONS)
        elif args.encoder == "gru":
            from agents.recurrent_ppo import RecurrentPPOAgent
            policy = RecurrentPPOAgent(obs_dim=obs_dim, n_actions=N_ACTIONS)
        elif args.encoder == "conv":
            from agents.conv_ppo import ConvPPOAgent
            from env.disaster_env import N_CHANNELS
            policy = ConvPPOAgent(in_channels=N_CHANNELS, view_size=args.view_size,
                                  n_actions=N_ACTIONS)
        else:
            policy = PPOAgent(obs_dim=obs_dim, n_actions=N_ACTIONS)
    elif args.policy == "random":
        policy = RandomPolicy(seed=args.seed)
    elif args.policy == "coordinated":
        policy = CoordinatedGreedyPolicy(env, seed=args.seed)
    else:
        policy = GreedyPolicy(view_size=args.view_size, seed=args.seed)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    rows: List[Dict] = []
    pbar = tqdm(range(args.episodes), desc=run_name, unit="ep")
    for ep in pbar:
        # anneal exploration: search needs wide exploration early, commitment later
        if args.policy == "ppo":
            frac = ep / max(1, args.episodes - 1)
            coef = args.entropy_start + frac * (args.entropy_end - args.entropy_start)
            if hasattr(policy, "config"):
                policy.config.entropy_coef = coef
            elif hasattr(policy, "policies"):
                for pol in policy.policies.values():
                    pol.config.entropy_coef = coef

        m = run_disaster_episode(
            env=env,
            policy=policy,
            episode_num=ep,
            alpha=args.alpha,
            train_policy=(args.policy == "ppo"),
            episode_seed=args.seed * args.episodes + ep,
            credit=args.credit,
        )
        row = _row(m)
        rows.append(row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        if hasattr(pbar, "set_postfix") and ep % 10 == 0:
            recent = rows[-50:]
            pbar.set_postfix({
                "saved": f"{np.mean([r['lives_saved'] for r in recent]):.1f}",
                "sev": f"{np.mean([r['severe_save_rate'] for r in recent]):.2f}",
            })

        if args.policy == "ppo" and args.checkpoint_every and (ep + 1) % args.checkpoint_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save(str(ckpt_dir / f"ep{ep+1:06d}.pt"))

    # summary over the final 10% of episodes (or final 100, whichever is larger)
    n_tail = max(100, len(rows) // 10)
    tail = rows[-n_tail:]
    summary = {
        "run_name": run_name,
        "policy": args.policy,
        "alpha": args.alpha,
        "seed": args.seed,
        "independent": args.independent,
        "episodes": args.episodes,
        "tail_episodes": len(tail),
        "mean_lives_saved": float(np.mean([r["lives_saved"] for r in tail])),
        "mean_lives_lost": float(np.mean([r["lives_lost"] for r in tail])),
        "mean_save_rate": float(np.mean([r["save_rate"] for r in tail])),
        "mean_severe_save_rate": float(np.mean([r["severe_save_rate"] for r in tail])),
        "mean_minor_save_rate": float(np.mean([r["minor_save_rate"] for r in tail])),
        "mean_joint_rescues": float(np.mean([r["joint_rescues"] for r in tail])),
        "mean_idle_rate": float(np.mean([r["mean_idle_rate"] for r in tail])),
    }
    # deterministic evaluation on held-out seeds: training metrics carry exploration noise,
    # so final numbers should not come from the training episodes themselves.
    if args.eval_episodes > 0:
        EVAL_BASE = 10_000_000
        ev: List[Dict] = []
        for i in range(args.eval_episodes):
            m = run_disaster_episode(
                env=env, policy=policy, episode_num=i, alpha=args.alpha,
                train_policy=False, episode_seed=EVAL_BASE + args.seed * args.eval_episodes + i,
                credit=args.credit, deterministic=(args.policy == "ppo"),
            )
            ev.append(_row(m))
        summary["eval_episodes"] = len(ev)
        summary["eval_lives_saved"] = float(np.mean([r["lives_saved"] for r in ev]))
        summary["eval_save_rate"] = float(np.mean([r["save_rate"] for r in ev]))
        summary["eval_severe_save_rate"] = float(np.mean([r["severe_save_rate"] for r in ev]))
        summary["eval_minor_save_rate"] = float(np.mean([r["minor_save_rate"] for r in ev]))
        summary["eval_joint_rescues"] = float(np.mean([r["joint_rescues"] for r in ev]))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.policy == "ppo":
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        policy.save(str(ckpt_dir / "final.pt"))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}\nwrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
