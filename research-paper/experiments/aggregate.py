"""Aggregate multi-seed runs into the paper's headline table + significance tests.

Reads every run's summary.json (mean_reward / mean_efficiency / mean_fairness /
mean_cooperation over the final 100 episodes), groups by reward scheme across seeds,
and reports mean +/- std, 95% CI, pairwise Welch t-tests and Cohen's d.

Pure NumPy + stdlib only (no scipy/pandas) so the environment stays dependency-free.
Runs on the frozen single-seed results too (n=1 -> variance/p-values reported as n/a).

Usage:
  python aggregate.py                                   # defaults (workspace runs, 30k)
  python aggregate.py --results-root ../../results --episodes 50000   # frozen data
  python aggregate.py --episodes 30000 --schemes selfish cooperative mixed
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

METRICS = ["mean_reward", "mean_efficiency", "mean_fairness", "mean_cooperation"]
PRETTY = {
    "mean_reward": "Reward",
    "mean_efficiency": "Efficiency",
    "mean_fairness": "Jain fairness",
    "mean_cooperation": "Cooperation",
}


# ---------- statistics (pure numpy / math) ----------
def _betacf(a: float, b: float, x: float) -> float:
    # continued fraction for the incomplete beta function (Numerical Recipes)
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    # regularised incomplete beta function I_x(a, b)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_two_sided(t: float, df: float) -> float:
    # two-sided p-value from Student's t with df degrees of freedom
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"), "cohen_d": float("nan")}
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"), "cohen_d": float("nan")}
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = _t_sf_two_sided(t, df)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (ma - mb) / pooled if pooled else float("nan")
    return {"t": t, "df": df, "p": p, "cohen_d": d}


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return float("nan")
    # t critical value for 95% two-sided via inverse of the p-value (bisection)
    se = x.std(ddof=1) / math.sqrt(n)
    df = n - 1
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _t_sf_two_sided(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return mid * se


# ---------- discovery ----------
def discover(results_root: Path, episodes: int, schemes: List[str]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {s: [] for s in schemes}
    # match run_<EP>_<scheme>[_seed<N>] , optional trailing _<variant>
    for d in sorted(results_root.glob(f"run_{episodes}_*")):
        summ = d / "summary.json"
        if not summ.is_file():
            continue
        m = re.match(rf"run_{episodes}_([a-z_]+?)(?:_seed(\d+))?(?:_(plus_own|team_avg))?$", d.name)
        if not m:
            continue
        scheme, seed, variant = m.group(1), m.group(2), m.group(3)
        if scheme not in out:
            continue
        data = json.loads(summ.read_text())
        data["_dir"] = d.name
        data["_seed"] = int(seed) if seed is not None else None
        out[scheme].append(data)
    return out


# ---------- reporting ----------
def build_report(groups: Dict[str, List[dict]]) -> str:
    lines: List[str] = []
    schemes = [s for s in groups if groups[s]]

    lines.append("# Multi-seed aggregation\n")
    for s in schemes:
        seeds = [r["_seed"] for r in groups[s]]
        lines.append(f"- **{s}**: n={len(groups[s])} runs, seeds={seeds}")
    lines.append("")

    # per-metric mean +/- std (95% CI)
    lines.append("## Final-100-episode metrics (mean +/- std, [95% CI])\n")
    header = "| Metric | " + " | ".join(schemes) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(schemes) + 1))
    arrs: Dict[str, Dict[str, np.ndarray]] = {}
    for metric in METRICS:
        row = [PRETTY[metric]]
        for s in schemes:
            vals = np.array([r[metric] for r in groups[s]], float)
            arrs.setdefault(metric, {})[s] = vals
            mean = vals.mean()
            if vals.size >= 2:
                row.append(f"{mean:.3f} +/- {vals.std(ddof=1):.3f} [±{ci95(vals):.3f}]")
            else:
                row.append(f"{mean:.3f} (n=1)")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # pairwise Welch t-tests
    lines.append("## Pairwise Welch t-tests (two-sided)\n")
    pairs = [(schemes[i], schemes[j]) for i in range(len(schemes)) for j in range(i + 1, len(schemes))]
    if not pairs:
        lines.append("_only one scheme present_\n")
    for metric in METRICS:
        lines.append(f"### {PRETTY[metric]}")
        lines.append("| Comparison | diff | t | df | p | Cohen's d |")
        lines.append("|---|---|---|---|---|---|")
        for a, b in pairs:
            va, vb = arrs[metric][a], arrs[metric][b]
            r = welch_ttest(va, vb)
            diff = va.mean() - vb.mean()
            sig = ""
            if r["p"] == r["p"]:  # not nan
                sig = " ***" if r["p"] < 0.001 else " **" if r["p"] < 0.01 else " *" if r["p"] < 0.05 else ""
            p_str = "n/a" if r["p"] != r["p"] else f"{r['p']:.4f}{sig}"
            t_str = "n/a" if r["t"] != r["t"] else f"{r['t']:.3f}"
            df_str = "n/a" if r["df"] != r["df"] else f"{r['df']:.1f}"
            d_str = "n/a" if r["cohen_d"] != r["cohen_d"] else f"{r['cohen_d']:.2f}"
            lines.append(f"| {a} vs {b} | {diff:+.3f} | {t_str} | {df_str} | {p_str} | {d_str} |")
        lines.append("")
    lines.append("_significance: * p<0.05, ** p<0.01, *** p<0.001. n/a = need >=2 seeds per group._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs" / "results"))
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--schemes", nargs="+", default=["selfish", "cooperative", "mixed"])
    ap.add_argument("--out", default=None, help="markdown output path (default: <results-root>/../analysis/)")
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    groups = discover(root, args.episodes, args.schemes)
    total = sum(len(v) for v in groups.values())
    if total == 0:
        print(f"No runs found under {root} matching run_{args.episodes}_<scheme>[_seedN]. "
              f"Run the sweep first, or point --results-root at the frozen results.")
        return

    report = build_report(groups)
    print(report)

    # default output always lands inside the git-ignored workspace, regardless of
    # which results-root was analysed (so pointing at the frozen tree never writes to it).
    here = Path(__file__).resolve().parent
    default_out = here / ".." / "runs" / "analysis" / f"aggregate_{args.episodes}.md"
    out_path = Path(args.out) if args.out else default_out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    # also dump the raw grouped values for downstream plotting
    raw = {s: {m: [r[m] for r in groups[s]] for m in METRICS} | {"seeds": [r["_seed"] for r in groups[s]]}
           for s in groups if groups[s]}
    (out_path.parent / f"aggregate_{args.episodes}.json").write_text(json.dumps(raw, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
