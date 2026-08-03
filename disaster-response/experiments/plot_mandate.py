"""Figures for the disaster mandate sweep.

Left  : severe vs minor save rate against alpha - the H1 figure. If an individual mandate
        suppresses cooperation, the severe curve should fall with alpha while the minor
        curve does not.
Right : collective outcome (lives saved) and joint rescues against alpha.

Usage: python plot_mandate.py --episodes 8000
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


def discover(root: Path, episodes: int) -> Dict[float, List[dict]]:
    groups: Dict[float, List[dict]] = {}
    for d in sorted(root.glob(f"disaster_{episodes}_a*")):
        s = d / "summary.json"
        m = re.match(rf"disaster_{episodes}_a([0-9.]+)_seed(\d+)(.*)$", d.name)
        if s.is_file() and m:
            groups.setdefault(float(m.group(1)), []).append(json.loads(s.read_text()))
    return groups


def _stats(groups, alphas, key):
    mean = np.array([np.mean([r[key] for r in groups[a]]) for a in alphas])
    sd = np.array([
        np.std([r[key] for r in groups[a]], ddof=1) if len(groups[a]) > 1 else 0.0
        for a in alphas
    ])
    return mean, sd


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-root", default=str(here / ".." / "runs"))
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--out", default=str(here / ".." / "figures" / "mandate_sweep.png"))
    args = ap.parse_args()

    groups = discover(Path(args.results_root).resolve(), args.episodes)
    if not groups:
        print("no runs found")
        return
    alphas = sorted(groups)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for key, colour, label in (
        ("mean_severe_save_rate", "tab:red", "Severe victims (need 2 responders)"),
        ("mean_minor_save_rate", "tab:blue", "Minor victims (need 1)"),
    ):
        m, s = _stats(groups, alphas, key)
        ax1.plot(alphas, m, "-o", color=colour, label=label, linewidth=1.9)
        ax1.fill_between(alphas, m - s, m + s, color=colour, alpha=0.16)
    ax1.set_xlabel(r"$\alpha$   (0 = collective credit,  1 = individual credit)")
    ax1.set_ylabel("Save rate")
    ax1.set_title("H1: does individual credit suppress cooperative rescues?")
    ax1.set_xticks(alphas)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    m, s = _stats(groups, alphas, "mean_lives_saved")
    ax2.plot(alphas, m, "-o", color="tab:green", label="Lives saved", linewidth=1.9)
    ax2.fill_between(alphas, m - s, m + s, color="tab:green", alpha=0.16)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Lives saved")
    ax2.set_title("Collective outcome")
    ax2.set_xticks(alphas)
    ax2.grid(alpha=0.25)

    ax2b = ax2.twinx()
    mj, sj = _stats(groups, alphas, "mean_joint_rescues")
    ax2b.plot(alphas, mj, "--s", color="tab:purple", label="Joint rescues", linewidth=1.5)
    ax2b.set_ylabel("Joint rescues", color="tab:purple")
    ax2b.tick_params(axis="y", colors="tab:purple")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")

    n = min(len(v) for v in groups.values())
    fig.suptitle(f"Mandate sweep ({args.episodes} episodes, n={n} seeds; shaded = ±1 s.d.)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    for a in alphas:
        sv = np.mean([r["mean_severe_save_rate"] for r in groups[a]])
        mn = np.mean([r["mean_minor_save_rate"] for r in groups[a]])
        ls = np.mean([r["mean_lives_saved"] for r in groups[a]])
        print(f"alpha={a}: severe={sv:.3f} minor={mn:.3f} lives={ls:.2f} gap={mn-sv:+.3f}")


if __name__ == "__main__":
    main()
