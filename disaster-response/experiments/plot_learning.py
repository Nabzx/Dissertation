"""Learning curves for disaster runs, with an optional baseline reference line.

Used to answer two questions the mandate sweep depends on:
  1. Has training converged? (read the plateau, not the endpoint)
  2. Has PPO overtaken the greedy heuristic? (the bar for any coordination claim)

Usage:
  python plot_learning.py --run disaster_6000_a1_seed0_probe
  python plot_learning.py --run <run> --baseline 6.08     # greedy reference
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("lives_saved", "Lives saved"),
    ("severe_save_rate", "Severe save rate"),
    ("minor_save_rate", "Minor save rate"),
    ("entropy", "Policy entropy"),
]


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if y.size == 0:
        return y
    c = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    out = np.empty(y.size)
    for i in range(y.size):
        s = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[s]) / (i + 1 - s)
    return out


def load(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def plateau_estimate(y: np.ndarray, window: int, tol: float = 0.02) -> int:
    """First episode after which the smoothed curve stays within `tol` (relative) of its
    final value - a rough, honest convergence read rather than eyeballing."""
    s = moving_average(y, window)
    if s.size < window * 2:
        return -1
    final = s[-1]
    if abs(final) < 1e-9:
        return -1
    for i in range(s.size):
        if np.all(np.abs(s[i:] - final) / abs(final) <= tol):
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--runs-root", default=str(here / ".." / "runs"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--baseline", type=float, default=None,
                    help="greedy lives-saved reference to draw on the first panel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.runs_root).resolve() / args.run
    data = load(run_dir)
    if not data:
        print(f"no data in {run_dir}")
        return
    n = len(data["episode"])

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.0))
    for ax, (key, label) in zip(axes, METRICS):
        if key not in data:
            ax.set_visible(False)
            continue
        y = moving_average(data[key], args.window)
        ax.plot(data["episode"], y, linewidth=1.7, color="tab:blue")
        if key == "lives_saved" and args.baseline is not None:
            ax.axhline(args.baseline, ls="--", color="tab:red", linewidth=1.3,
                       label=f"greedy = {args.baseline:g}")
            ax.legend(fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.run}  (smoothed, window={args.window}; {n} episodes)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(args.out) if args.out else (here / ".." / "figures" / f"learning_{args.run}.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # convergence + baseline read
    for key, label in METRICS:
        if key in data:
            i = plateau_estimate(data[key], args.window)
            msg = f"~episode {i}" if i >= 0 else "not yet converged"
            print(f"{label:20s} plateau: {msg}")
    if args.baseline is not None:
        smooth = moving_average(data["lives_saved"], args.window)
        above = np.where(smooth > args.baseline)[0]
        print(f"\novertakes greedy ({args.baseline:g}) at: "
              f"{'episode ' + str(int(above[0])) if above.size else 'NOT YET'}")
        print(f"final smoothed lives saved: {smooth[-1]:.2f}")


if __name__ == "__main__":
    main()
