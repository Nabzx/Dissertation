"""Behavioural metrics: territoriality and passivity.

These measure *coordination* and *inaction* directly, rather than inferring them from
outcome metrics. Both are computed from the per-agent trajectories already stored in each
run's final_episodes.json (100 episodes x 4 agents x 251 positions), so no retraining.

  territoriality  - 1 - mean pairwise overlap of agents' spatial occupancy distributions.
                    1.0 = agents occupy completely disjoint regions (perfect separation),
                    0.0 = identical occupancy. Measures emergent spatial coordination.
  passivity       - fraction of timesteps an agent does not change cell. Directly tests
                    the "agents learn to do nothing" account of free-riding.

Usage:
  python behaviour.py --episodes 30000 --schemes selfish cooperative cooperative_indep
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from aggregate import welch_ttest, ci95  # reuse the stats machinery


def _occupancy(positions: List[List[int]], grid: int = 25) -> np.ndarray:
    h = np.zeros(grid * grid, dtype=np.float64)
    for r, c in positions:
        if 0 <= r < grid and 0 <= c < grid:
            h[r * grid + c] += 1.0
    s = h.sum()
    return h / s if s > 0 else h


def episode_metrics(ep: Dict, grid: int = 25) -> Dict[str, float]:
    trajs = ep.get("trajectories", {})
    agents = sorted(trajs)
    if len(agents) < 2:
        return {}

    # territoriality: 1 - mean pairwise histogram intersection of occupancy
    occ = {a: _occupancy(trajs[a], grid) for a in agents}
    overlaps = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            overlaps.append(float(np.minimum(occ[agents[i]], occ[agents[j]]).sum()))
    territoriality = 1.0 - float(np.mean(overlaps)) if overlaps else 0.0

    # passivity: fraction of steps where the agent stayed in the same cell
    stays, total = 0, 0
    for a in agents:
        p = trajs[a]
        for t in range(1, len(p)):
            total += 1
            if p[t][0] == p[t - 1][0] and p[t][1] == p[t - 1][1]:
                stays += 1
    passivity = stays / total if total else 0.0

    return {"territoriality": territoriality, "passivity": passivity}


def run_metrics(run_dir: Path) -> Dict[str, float]:
    eps = json.loads((run_dir / "final_episodes.json").read_text())
    vals: Dict[str, List[float]] = {"territoriality": [], "passivity": []}
    for ep in eps:
        m = episode_metrics(ep)
        for k, v in m.items():
            vals[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}


def discover(root: Path, episodes: int, scheme: str) -> List[Path]:
    base, indep = (scheme[:-6], True) if scheme.endswith("_indep") else (scheme, False)
    suffix = "_indep" if indep else ""
    out = []
    for d in sorted(root.glob(f"run_{episodes}_{base}_seed*")):
        if re.match(rf"run_{episodes}_{base}_seed\d+(_(plus_own|team_avg))?{suffix}$", d.name) \
           and (d / "final_episodes.json").is_file():
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--schemes", nargs="+",
                    default=["selfish", "cooperative", "mixed", "selfish_indep", "cooperative_indep"])
    ap.add_argument("--out", default=str(here / ".." / "results" / "behaviour.md"))
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for s in args.schemes:
        dirs = discover(root, args.episodes, s)
        if not dirs:
            continue
        per = [run_metrics(d) for d in dirs]
        groups[s] = {k: np.array([p[k] for p in per], float) for k in ("territoriality", "passivity")}

    if not groups:
        print("no runs found")
        return

    lines = ["# Behavioural metrics (final 100 episodes, mean +/- s.d. across seeds)\n"]
    schemes = list(groups)
    lines.append("| Metric | " + " | ".join(schemes) + " |")
    lines.append("|" + "---|" * (len(schemes) + 1))
    for k, label in (("territoriality", "Territoriality (higher = more separated)"),
                     ("passivity", "Passivity (higher = more inaction)")):
        row = [label]
        for s in schemes:
            v = groups[s][k]
            row.append(f"{v.mean():.3f} +/- {v.std(ddof=1) if v.size>1 else 0.0:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Key comparisons (Welch t-test)\n")
    lines.append("| Metric | comparison | diff | p | Cohen's d |")
    lines.append("|---|---|---|---|---|")
    pairs = [("selfish", "cooperative"), ("cooperative", "cooperative_indep"),
             ("selfish_indep", "cooperative_indep"), ("selfish", "mixed")]
    for k in ("territoriality", "passivity"):
        for a, b in pairs:
            if a in groups and b in groups:
                r = welch_ttest(groups[a][k], groups[b][k])
                diff = groups[a][k].mean() - groups[b][k].mean()
                p = r["p"]
                sig = "" if p != p else (" ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else "")
                d = r["cohen_d"]
                p_str = "n/a" if p != p else f"{p:.4f}{sig}"
                d_str = "n/a" if d != d else f"{d:.2f}"
                lines.append(f"| {k} | {a} vs {b} | {diff:+.3f} | {p_str} | {d_str} |")
    report = "\n".join(lines)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
