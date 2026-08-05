"""Aggregate disaster runs into the mandate table + significance tests.

Groups runs by alpha across seeds and reports mean +/- s.d., 95% CI, and pairwise Welch
t-tests with Cohen's d. Reuses the pure-NumPy statistics written for the gridworld paper, so
there is one implementation of the stats and no scipy dependency.

Usage:
  python aggregate.py --episodes 8000
  python aggregate.py --episodes 8000 --metric mean_severe_save_rate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# reuse the paper branch's stats implementation
PAPER_EXP = Path(__file__).resolve().parents[2] / "research-paper" / "experiments"
if str(PAPER_EXP) not in sys.path:
    sys.path.insert(0, str(PAPER_EXP))
from aggregate import welch_ttest, ci95  # noqa: E402

METRICS = [
    "mean_lives_saved",
    "mean_save_rate",
    "mean_severe_save_rate",
    "mean_minor_save_rate",
    "mean_joint_rescues",
    "mean_idle_rate",
]
PRETTY = {
    "mean_lives_saved": "Lives saved",
    "mean_save_rate": "Save rate",
    "mean_severe_save_rate": "Severe save rate",
    "mean_minor_save_rate": "Minor save rate",
    "mean_joint_rescues": "Joint rescues",
    "mean_idle_rate": "Idle rate",
}


def discover(root: Path, episodes: int) -> Dict[float, List[dict]]:
    """Prefer eval_stochastic.json: the summary's own eval used argmax actions, which
    understates performance by ~40% in this search task (see results/eval_mode.md)."""
    groups: Dict[float, List[dict]] = {}
    for d in sorted(root.glob(f"disaster_{episodes}_a*")):
        stoch = d / "eval_stochastic.json"
        s = d / "summary.json"
        if not s.is_file():
            continue
        m = re.match(rf"disaster_{episodes}_a([0-9.]+)_seed(\d+)(.*)$", d.name)
        if not m:
            continue
        data = json.loads(s.read_text())
        if stoch.is_file():
            ev = json.loads(stoch.read_text())
            # overwrite the headline metrics with the stochastic evaluation
            data["mean_lives_saved"] = ev["lives_saved"]
            data["mean_save_rate"] = ev["save_rate"]
            data["mean_severe_save_rate"] = ev["severe_save_rate"]
            data["mean_minor_save_rate"] = ev["minor_save_rate"]
            data["mean_joint_rescues"] = ev["joint_rescues"]
            data["_eval"] = "stochastic"
        else:
            data["_eval"] = "training-window"
        data["_variant"] = m.group(3)
        groups.setdefault(float(m.group(1)), []).append(data)
    return groups


def build_report(groups: Dict[float, List[dict]]) -> str:
    alphas = sorted(groups)
    lines = ["# Disaster mandate sweep\n"]
    for a in alphas:
        seeds = sorted(r["seed"] for r in groups[a])
        lines.append(f"- **alpha={a}**: n={len(groups[a])}, seeds={seeds}")
    lines.append("")

    lines.append("## Final-window metrics (mean +/- s.d. [95% CI])\n")
    lines.append("| Metric | " + " | ".join(f"a={a}" for a in alphas) + " |")
    lines.append("|" + "---|" * (len(alphas) + 1))
    arrs: Dict[str, Dict[float, np.ndarray]] = {}
    for met in METRICS:
        row = [PRETTY[met]]
        for a in alphas:
            v = np.array([r[met] for r in groups[a]], float)
            arrs.setdefault(met, {})[a] = v
            if v.size >= 2:
                row.append(f"{v.mean():.3f} +/- {v.std(ddof=1):.3f} [±{ci95(v):.3f}]")
            else:
                row.append(f"{v.mean():.3f} (n=1)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # H1: severe save rate should fall as alpha rises
    lines.append("## H1 — does an individual mandate suppress cooperative rescues?\n")
    lines.append("| Metric | comparison | diff | p | Cohen's d |")
    lines.append("|---|---|---|---|---|")
    if len(alphas) >= 2:
        lo, hi = alphas[0], alphas[-1]
        for met in ("mean_severe_save_rate", "mean_minor_save_rate",
                    "mean_joint_rescues", "mean_lives_saved"):
            a_lo, a_hi = arrs[met][lo], arrs[met][hi]
            r = welch_ttest(a_lo, a_hi)
            diff = a_lo.mean() - a_hi.mean()
            p, d = r["p"], r["cohen_d"]
            sig = "" if p != p else (" ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else "")
            lines.append(
                f"| {PRETTY[met]} | a={lo} vs a={hi} | {diff:+.3f} | "
                f"{'n/a' if p != p else f'{p:.4f}{sig}'} | "
                f"{'n/a' if d != d else f'{d:.2f}'} |"
            )
    lines.append("\n_H1 predicts severe save rate and joint rescues fall as alpha rises._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs"))
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    groups = discover(root, args.episodes)
    if not groups:
        print(f"no runs found under {root} matching disaster_{args.episodes}_a*")
        return
    report = build_report(groups)
    print(report)

    out = Path(args.out) if args.out else (here / ".." / "results" / f"mandate_{args.episodes}.md").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    raw = {str(a): {m: [r[m] for r in groups[a]] for m in METRICS} for a in sorted(groups)}
    out.with_suffix(".json").write_text(json.dumps(raw, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
