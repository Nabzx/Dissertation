"""Shared-weight vs independent-policies ablation figure.

Grouped bars: for each metric, selfish and cooperative under both architectures, with
across-seed s.d. error bars. Shows selfish is unaffected by removing parameter sharing
while cooperative collapses.

Usage: python plot_ablation.py --episodes 30000
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


def collect(root: Path, episodes: int, base: str, indep: bool) -> Dict[str, np.ndarray]:
    suffix = "_indep" if indep else ""
    vals: Dict[str, List[float]] = {m: [] for m in METRICS}
    for d in sorted(root.glob(f"run_{episodes}_{base}_seed*")):
        if not re.match(rf"run_{episodes}_{base}_seed\d+(_(plus_own|team_avg))?{suffix}$", d.name):
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
    ap.add_argument("--out", default=str(here / ".." / "results" / "figures" / "ablation.png"))
    args = ap.parse_args()
    root = Path(args.results_root).resolve()

    conds = [
        ("selfish",     False, "Selfish\n(shared)",      "tab:red"),
        ("selfish",     True,  "Selfish\n(independent)", "salmon"),
        ("cooperative", False, "Cooperative\n(shared)",  "tab:blue"),
        ("cooperative", True,  "Cooperative\n(independent)", "lightsteelblue"),
    ]
    data = [(lbl, c, collect(root, args.episodes, base, ind)) for base, ind, lbl, c in conds]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 4.4))
    for ax, m in zip(axes, METRICS):
        labels, means, errs, colours = [], [], [], []
        for lbl, c, d in data:
            v = d[m]
            if v.size == 0:
                continue
            labels.append(lbl)
            means.append(v.mean())
            errs.append(v.std(ddof=1) if v.size > 1 else 0.0)
            colours.append(c)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errs, capsize=4, color=colours, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(PRETTY[m])
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("Removing parameter sharing: selfish is unaffected, cooperative collapses\n"
                 "(error bars = 1 s.d. across seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
