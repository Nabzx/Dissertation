"""Resource-density sweep figure.

Left:  efficiency of selfish vs cooperative across resource densities (they converge as
       resources become abundant).
Right: the selfish-minus-cooperative gap for each metric, shrinking with abundance.

Usage: python plot_density.py --episodes 30000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["mean_efficiency", "mean_fairness", "mean_cooperation"]
PRETTY = {"mean_efficiency": "Efficiency", "mean_fairness": "Jain fairness",
          "mean_cooperation": "Cooperation"}
COLOURS = {"mean_efficiency": "tab:green", "mean_fairness": "tab:orange",
           "mean_cooperation": "tab:purple"}


def collect(root: Path, episodes: int, scheme: str, density: int) -> Dict[str, np.ndarray]:
    # density 25 is the default configuration and carries no _r suffix
    suffix = "" if density == 25 else f"_r{density}"
    vals: Dict[str, List[float]] = {m: [] for m in METRICS}
    for d in sorted(root.glob(f"run_{episodes}_{scheme}_seed*")):
        if not re.match(
            rf"run_{episodes}_{scheme}_seed\d+(_(plus_own|team_avg))?{suffix}$", d.name
        ):
            continue
        p = d / "summary.json"
        if p.is_file():
            s = json.loads(p.read_text())
            for m in METRICS:
                vals[m].append(s[m])
    return {m: np.array(v, float) for m, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--densities", nargs="+", type=int, default=[15, 25, 40])
    ap.add_argument("--out", default=str(here / ".." / "results" / "figures" / "density.png"))
    args = ap.parse_args()
    root = Path(args.results_root).resolve()

    dens = args.densities
    sel = {d: collect(root, args.episodes, "selfish", d) for d in dens}
    coop = {d: collect(root, args.episodes, "cooperative", d) for d in dens}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: absolute efficiency, both schemes
    for name, group, colour in (("Selfish", sel, "tab:red"), ("Cooperative", coop, "tab:blue")):
        m = np.array([group[d]["mean_efficiency"].mean() for d in dens])
        s = np.array([group[d]["mean_efficiency"].std(ddof=1)
                      if group[d]["mean_efficiency"].size > 1 else 0.0 for d in dens])
        ax1.plot(dens, m, "-o", color=colour, label=name, linewidth=1.8)
        ax1.fill_between(dens, m - s, m + s, color=colour, alpha=0.15)
    ax1.set_xlabel("Number of resources (scarcity $\\rightarrow$ abundance)")
    ax1.set_ylabel("Efficiency")
    ax1.set_title("Efficiency vs resource density")
    ax1.set_xticks(dens)
    ax1.grid(alpha=0.25)
    ax1.legend()

    # right: the gap per metric
    for m in METRICS:
        gap = [sel[d][m].mean() - coop[d][m].mean() for d in dens]
        ax2.plot(dens, gap, "-o", color=COLOURS[m], label=PRETTY[m], linewidth=1.8)
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Number of resources (scarcity $\\rightarrow$ abundance)")
    ax2.set_ylabel("Selfish $-$ Cooperative")
    ax2.set_title("The advantage of selfish reward shrinks with abundance")
    ax2.set_xticks(dens)
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    for d in dens:
        print(f"density {d}: " + "  ".join(
            f"{PRETTY[m]} gap={sel[d][m].mean() - coop[d][m].mean():+.3f}" for m in METRICS))


if __name__ == "__main__":
    main()
