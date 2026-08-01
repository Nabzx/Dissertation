"""Non-circular free-riding metrics (answers the 'cooperation = eff x fair is circular'
reviewer objection). Computed from existing runs -- no training needed.

For each condition, over the final 100 episodes and across seeds, we compute two measures
of contribution inequality that directly operationalise free-riding:

  * free-rider fraction : mean fraction of agents collecting < 0.5x the per-episode
    per-agent mean. An interpretable count of how many agents coast on the others.
  * contribution Gini   : Gini coefficient of per-agent collection counts (0 = equal,
    1 = one agent does everything). Familiar cross-check; note it is related to the Jain
    fairness index (both measure dispersion of the same counts), whereas the free-rider
    fraction is a distinct thresholded behavioural measure.

Hypothesis: pure team-average (cooperative) reward induces MORE free-riding than schemes
that keep individual incentive.

Usage:
  python freerider.py --episodes 30000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from aggregate import welch_ttest, ci95  # reuse pure-numpy stats


def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, float))
    n = x.size
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    # mean absolute difference formula
    cum = np.cumsum(x)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * s) / (n * s))


def free_rider_fraction(counts: np.ndarray, thresh: float = 0.5) -> float:
    x = np.asarray(counts, float)
    m = x.mean()
    if m == 0:
        return 0.0
    return float(np.mean(x < thresh * m))


def last_n_metrics(run_dir: Path, n: int = 100) -> Dict[str, float]:
    rows = []
    with open(run_dir / "headless_training_metrics.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    tail = rows[-n:]
    fr, gi = [], []
    for row in tail:
        counts = np.array([float(v) for v in json.loads(row["resources_collected_json"]).values()])
        fr.append(free_rider_fraction(counts))
        gi.append(gini(counts))
    return {"free_rider_fraction": float(np.mean(fr)), "contribution_gini": float(np.mean(gi))}


def discover(root: Path, episodes: int, scheme: str) -> List[Path]:
    # "<scheme>_indep" selects the independent-policy ablation runs, which carry an
    # _indep suffix; a plain scheme name matches only the shared-weight runs.
    base, want_indep = (scheme[:-6], True) if scheme.endswith("_indep") else (scheme, False)
    suffix = "_indep" if want_indep else ""
    out = []
    for d in sorted(root.glob(f"run_{episodes}_{base}_seed*")):
        if re.match(rf"run_{episodes}_{base}_seed\d+(_(plus_own|team_avg))?{suffix}$", d.name) \
           and (d / "headless_training_metrics.csv").is_file():
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--schemes", nargs="+", default=["selfish", "cooperative", "mixed"])
    ap.add_argument("--out", default=str(here / ".." / "results" / "freerider.md"))
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    METRICS = ["free_rider_fraction", "contribution_gini"]
    PRETTY = {"free_rider_fraction": "Free-rider fraction", "contribution_gini": "Contribution Gini"}

    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for s in args.schemes:
        dirs = discover(root, args.episodes, s)
        if not dirs:
            continue
        vals = {m: [] for m in METRICS}
        for d in dirs:
            r = last_n_metrics(d)
            for m in METRICS:
                vals[m].append(r[m])
        groups[s] = {m: np.array(vals[m], float) for m in METRICS}

    schemes = list(groups)
    lines = ["# Free-riding metrics (final 100 episodes, mean +/- std [95% CI])\n"]
    lines.append("| Metric | " + " | ".join(schemes) + " |")
    lines.append("|" + "---|" * (len(schemes) + 1))
    for m in METRICS:
        cells = [PRETTY[m]]
        for s in schemes:
            v = groups[s][m]
            cells.append(f"{v.mean():.3f} +/- {v.std(ddof=1):.3f} [±{ci95(v):.3f}]" if v.size > 1
                         else f"{v.mean():.3f} (n=1)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Cooperative vs others (Welch t-test; higher = more free-riding)\n")
    lines.append("| Metric | comparison | diff | p | Cohen's d |")
    lines.append("|---|---|---|---|---|")
    if "cooperative" in groups:
        for m in METRICS:
            for s in schemes:
                if s == "cooperative":
                    continue
                a, b = groups["cooperative"][m], groups[s][m]
                r = welch_ttest(a, b)
                sig = ""
                if r["p"] == r["p"]:
                    sig = " ***" if r["p"] < 0.001 else " **" if r["p"] < 0.01 else " *" if r["p"] < 0.05 else ""
                p = "n/a" if r["p"] != r["p"] else f"{r['p']:.4f}{sig}"
                d = "n/a" if r["cohen_d"] != r["cohen_d"] else f"{r['cohen_d']:.2f}"
                lines.append(f"| {PRETTY[m]} | cooperative vs {s} | {a.mean()-b.mean():+.3f} | {p} | {d} |")
    lines.append("\n_* p<0.05, ** p<0.01, *** p<0.001._")

    report = "\n".join(lines)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
