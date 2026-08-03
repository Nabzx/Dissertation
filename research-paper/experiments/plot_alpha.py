"""Alpha-sweep curve (paper Phase 2 centrepiece).

Reads the mixed-reward alpha-sweep runs (run_<EP>_mixed_seed<N>_a<alpha>), groups by
alpha across seeds, and plots each metric as a function of alpha with +/-1 s.d. bands.
alpha = 1 -> selfish, alpha = 0 -> cooperative(team_avg); overlays the Phase 1 selfish /
cooperative points as reference so the endpoints can be seen to line up.

Pure NumPy + matplotlib. Also prints a table and writes json.

Usage:
  python plot_alpha.py --episodes 30000
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


def _load(root: Path, name: str) -> dict | None:
    p = root / name / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else None


def discover_alpha(root: Path, episodes: int) -> Dict[float, List[dict]]:
    groups: Dict[float, List[dict]] = {}
    for d in sorted(root.glob(f"run_{episodes}_mixed_seed*_a*")):
        m = re.match(rf"run_{episodes}_mixed_seed(\d+)_a([0-9.]+)$", d.name)
        if not m or not (d / "summary.json").is_file():
            continue
        alpha = float(m.group(2))
        groups.setdefault(alpha, []).append(json.loads((d / "summary.json").read_text()))
    return groups


def ref_points(root: Path, episodes: int, scheme: str, suffix: str) -> Dict[str, float]:
    # mean over seeds of the Phase 1 selfish / cooperative runs, for endpoint overlay
    vals: Dict[str, List[float]] = {m: [] for m in METRICS}
    for d in sorted(root.glob(f"run_{episodes}_{scheme}_seed*{suffix}")):
        s = _load(root, d.name)
        if s:
            for m in METRICS:
                vals[m].append(s[m])
    return {m: float(np.mean(v)) for m, v in vals.items() if v}


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--out", default=str(here / ".." / "results" / "figures" / "alpha_sweep.png"))
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    groups = discover_alpha(root, args.episodes)
    if not groups:
        print(f"No alpha-sweep runs found under {root} (run_{args.episodes}_mixed_seed*_a*).")
        return
    alphas = sorted(groups)

    # reference endpoints from Phase 1
    ref_selfish = ref_points(root, args.episodes, "selfish", "")            # ~ alpha=1
    ref_coop = ref_points(root, args.episodes, "cooperative", "_team_avg")  # ~ alpha=0

    # table
    print(f"{'alpha':>6} {'n':>3}  " + "  ".join(f"{PRETTY[m]:>22}" for m in METRICS))
    table: Dict[str, list] = {"alpha": alphas}
    stats = {m: {"mean": [], "std": []} for m in METRICS}
    for a in alphas:
        runs = groups[a]
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in runs], float)
            stats[m]["mean"].append(v.mean())
            stats[m]["std"].append(v.std(ddof=1) if v.size > 1 else 0.0)
            cells.append(f"{v.mean():.3f} +/- {v.std(ddof=1) if v.size>1 else 0.0:.3f}")
        print(f"{a:>6.2f} {len(runs):>3}  " + "  ".join(f"{c:>22}" for c in cells))

    # plot
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for m in METRICS:
        mean = np.array(stats[m]["mean"])
        std = np.array(stats[m]["std"])
        ax.plot(alphas, mean, "-o", color=COLOURS[m], label=PRETTY[m], linewidth=1.8)
        ax.fill_between(alphas, mean - std, mean + std, color=COLOURS[m], alpha=0.15)
        # endpoint reference markers (hollow) from the standalone selfish/cooperative runs
        if ref_coop.get(m) is not None:
            ax.plot(0.0, ref_coop[m], marker="s", mfc="none", mec=COLOURS[m], ms=9)
        if ref_selfish.get(m) is not None:
            ax.plot(1.0, ref_selfish[m], marker="s", mfc="none", mec=COLOURS[m], ms=9)

    ax.set_xlabel(r"$\alpha$  (0 = cooperative / team-average,  1 = selfish / individual)")
    ax.set_ylabel("Final-100-episode metric")
    ax.set_title(r"Effect of individual-incentive weight $\alpha$ on emergent behaviour"
                 "\n(shaded = ±1 s.d. across seeds; hollow squares = standalone selfish/cooperative runs)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    data = {str(a): {m: [r[m] for r in groups[a]] for m in METRICS} for a in alphas}
    (out.parent.parent / "alpha_sweep.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
