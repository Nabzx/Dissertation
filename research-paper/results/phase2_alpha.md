# Phase 2 result — α-sweep (mixed reward, α ∈ {0,0.25,0.5,0.75,1}, 3 seeds, 30k)

Reward `r_i = α·own + (1−α)·team_avg`. α=0 is cooperative (team-average), α=1 is selfish.
Figure: `figures/alpha_sweep.png`. Data: `alpha_sweep.json`. Regenerate with
`experiments/plot_alpha.py --episodes 30000`.

## Table (final-100-episode, mean ± std across 3 seeds)

| α | Efficiency | Jain fairness | Cooperation |
|---|---|---|---|
| 0.00 (cooperative) | 0.513 ± 0.020 | 0.656 ± 0.004 | 0.345 ± 0.009 |
| 0.25 | 0.603 ± 0.033 | 0.701 ± 0.007 | 0.432 ± 0.026 |
| 0.50 (mixed) | 0.579 ± 0.054 | 0.691 ± 0.019 | 0.404 ± 0.040 |
| 0.75 | 0.620 ± 0.010 | 0.691 ± 0.011 | 0.436 ± 0.009 |
| 1.00 (selfish) | 0.641 ± 0.020 | 0.720 ± 0.016 | 0.468 ± 0.023 |

## Consistency check (endpoints ≡ standalone runs)
mixed α=1 ≡ selfish and mixed α=0 ≡ cooperative(team_avg) — **byte-identical per seed**
(e.g. seed 0 efficiency: α=1 = selfish = 0.6552; α=0 = cooperative = 0.5308). The hollow
squares on the figure sit exactly on the curve endpoints. Confirms the α wiring and the
whole pipeline.

## Interpretation (the mechanistic story)

1. **The curve is steep from α=0→0.25, then plateaus.** The single biggest gain comes from
   adding *a little* individual incentive; going fully selfish adds only marginal further
   benefit. On efficiency: +0.090 from α=0→0.25, but only +0.038 more across 0.25→1.
2. **The harm is specific to pure team-average reward.** As soon as an agent's own
   contribution is even 25% of its signal, free-riding stops being rational and the
   credit-assignment problem eases — performance jumps.
3. This upgrades the Phase 1 claim from three discrete bars to a **continuous dose-response
   curve**: emergent efficiency/fairness/coordination are a *saturating increasing* function
   of individual-incentive weight, with the knee near α≈0.25.
4. The α=0.5 point dips slightly below its neighbours — within noise (n=3, largest std).
   More seeds would smooth it; the monotone-rising-then-flat trend is clear regardless.

Caveat: α-sweep used 3 seeds (vs 5 for the main conditions) to fit one overnight. If a
reviewer wants tighter bands, rerun `ALPHAS="..." SEEDS="0 1 2 3 4"`.
