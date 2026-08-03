# Phase 1 result — multi-seed (5 seeds, 30k episodes, team_avg cooperative)

This is the paper's core quantitative evidence. Final-100-episode metrics, mean ± std
across seeds 0–4. Full table + Welch t-tests in `runs/analysis/aggregate_30000.md`
(regenerate with `experiments/aggregate.py --episodes 30000`).

## Headline table (mean ± std [95% CI])

| Metric | selfish | cooperative | mixed |
|---|---|---|---|
| Efficiency | 0.627 ± 0.032 [±0.040] | **0.502 ± 0.027** [±0.033] | 0.584 ± 0.045 [±0.056] |
| Jain fairness | 0.716 ± 0.013 [±0.016] | **0.656 ± 0.005** [±0.007] | 0.692 ± 0.016 [±0.020] |
| Cooperation | 0.455 ± 0.028 [±0.035] | **0.338 ± 0.015** [±0.019] | 0.410 ± 0.035 [±0.043] |
| Reward | 15.67 ± 0.81 | 12.56 ± 0.67 | 14.60 ± 1.12 |

## Significance (Welch t-test, two-sided; Cohen's d)

| Comparison | Efficiency | Fairness | Cooperation |
|---|---|---|---|
| selfish vs cooperative | p=0.0002 ***, d=4.2 | p=0.0001 ***, d=6.2 | p=0.0002 ***, d=5.1 |
| cooperative vs mixed | p=0.011 *, d=−2.2 | p=0.006 **, d=−3.0 | p=0.007 **, d=−2.7 |
| selfish vs mixed | p=0.124 (n.s.), d=1.1 | p=0.032 *, d=1.7 | p=0.059 (n.s.), d=1.4 |

## Interpretation (the claim to make in the paper)

1. **Explicitly cooperative (team-average) reward is significantly and substantially worse
   than schemes that preserve individual incentive** — on efficiency, fairness AND
   coordination. selfish > cooperative holds at p<0.001 with *very large* effect sizes
   (d = 4–6). This is the counterintuitive headline, and it is robust across seeds.
2. **cooperative < mixed** is also significant (p<0.05–0.01). So the harm comes specifically
   from *removing* the individual signal, and even a 50/50 blend recovers most of it.
3. **selfish vs mixed is a small, mostly non-significant difference** (efficiency/cooperation
   n.s.; fairness marginally significant). Honest framing: selfish trends best but is
   statistically close to mixed. Do NOT over-claim a strict selfish > mixed > cooperative
   ordering — the defensible claim is *{selfish, mixed} ≫ cooperative*.
4. Cooperative also has the **lowest variance** (very consistent collapse), consistent with
   the credit-assignment / free-riding mechanism the paper argues.

This nuance is a strength: it's a cleaner, more precise claim than the single-seed
dissertation ("selfish beats all"), and it survives statistical scrutiny.
