"""Multi-seed learning curves with confidence bands (paper Figure 2).

For each reward scheme, reads every seed's per-episode CSV, computes per-episode
efficiency (resources/25) and Jain fairness (matching analysis/post_training_analysis.py),
smooths each seed, then plots the across-seed mean with a shaded +/-1 s.d. band.

Pure NumPy + matplotlib (already deps). Streams the CSVs (light on memory).

Usage:
  python plot_curves.py --episodes 30000
  python plot_curves.py --episodes 30000 --window 200 --metric efficiency fairness
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_RES = 25
COLOURS = {"selfish": "tab:red", "cooperative": "tab:blue", "mixed": "tab:green"}
PRETTY = {"efficiency": "Resource efficiency", "fairness": "Jain fairness index",
          "cooperation": "Cooperation score", "reward": "Total reward"}


def jain(counts: List[float]) -> float:
    if not counts:
        return 0.0
    s = sum(counts)
    d = len(counts) * sum(c * c for c in counts)
    return float(s * s / d) if d else 0.0


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    csum = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(y.size):
        s = max(0, i - w + 1)
        out[i] = (csum[i + 1] - csum[s]) / (i + 1 - s)
    return out


def load_seed_series(run_dir: Path, max_res: int = MAX_RES) -> Dict[str, np.ndarray]:
    # max_res must match the run's num_resources (see the _r<N> suffix on density runs),
    # otherwise efficiency is normalised against the wrong denominator.
    eff, fair, rew, coop = [], [], [], []
    with open(run_dir / "headless_training_metrics.csv") as f:
        for row in csv.DictReader(f):
            total = float(row["total_resources"])
            e = total / max_res
            counts = [float(v) for v in json.loads(row["resources_collected_json"]).values()]
            fr = jain(counts)
            eff.append(e)
            fair.append(fr)
            rew.append(float(row["total_reward"]))
            coop.append(e * fr)
    return {"efficiency": np.array(eff), "fairness": np.array(fair),
            "reward": np.array(rew), "cooperation": np.array(coop)}


def discover(results_root: Path, episodes: int, scheme: str) -> List[Path]:
    dirs = []
    for d in sorted(results_root.glob(f"run_{episodes}_{scheme}_seed*")):
        if re.match(rf"run_{episodes}_{scheme}_seed\d+(_(plus_own|team_avg))?$", d.name) \
           and (d / "headless_training_metrics.csv").is_file():
            dirs.append(d)
    return dirs


def stacked(results_root: Path, episodes: int, scheme: str, metric: str, window: int
            ) -> Tuple[np.ndarray, np.ndarray, int]:
    seeds = discover(results_root, episodes, scheme)
    series = [moving_average(load_seed_series(d)[metric], window) for d in seeds]
    if not series:
        return np.array([]), np.array([]), 0
    T = min(s.size for s in series)
    mat = np.stack([s[:T] for s in series])  # (n_seeds, T)
    return mat.mean(0), mat.std(0), len(seeds)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--schemes", nargs="+", default=["selfish", "cooperative", "mixed"])
    ap.add_argument("--metric", nargs="+", default=["efficiency", "fairness"])
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--downsample", type=int, default=50)
    ap.add_argument("--out", default=str(here / ".." / "results" / "figures" / "learning_curves.png"))
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    metrics = args.metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6), squeeze=False)
    n_seeds_seen = 0

    for ax, metric in zip(axes[0], metrics):
        for scheme in args.schemes:
            mean, std, n = stacked(root, args.episodes, scheme, metric, args.window)
            if mean.size == 0:
                continue
            n_seeds_seen = max(n_seeds_seen, n)
            x = np.arange(mean.size)[:: args.downsample]
            m = mean[:: args.downsample]
            s = std[:: args.downsample]
            c = COLOURS.get(scheme, None)
            ax.plot(x, m, color=c, linewidth=1.8, label=f"{scheme} (n={n})")
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.18)
        ax.set_title(PRETTY.get(metric, metric))
        ax.set_xlabel("Episode")
        ax.set_ylabel(PRETTY.get(metric, metric))
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Learning curves across {n_seeds_seen} seeds "
                 f"(smoothed, window={args.window}; shaded = ±1 s.d.)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
