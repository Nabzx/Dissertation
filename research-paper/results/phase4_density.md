# Resource-density sweep — the effect is scarcity-modulated

Selfish vs cooperative(team_avg) at 15 / 25 / 40 resources, 3 seeds each (25 = the main
5-seed condition). Efficiency is normalised by each run's own resource count, so densities
are directly comparable. Figure: `figures/density.png`.

## Results

| Metric | | 15 (scarce) | 25 (default) | 40 (abundant) |
|---|---|---|---|---|
| Efficiency | selfish | 0.619 ± 0.048 | 0.627 ± 0.032 | 0.648 ± 0.027 |
| | cooperative | 0.297 ± 0.049 | 0.502 ± 0.027 | 0.570 ± 0.020 |
| | **gap** | **+0.322** | **+0.125** | **+0.078** |
| Jain fairness | gap | +0.127 | +0.060 | **+0.009** |
| Cooperation | gap | +0.239 | +0.117 | +0.060 |

Significance (efficiency, selfish vs cooperative): scarce p=0.0012 (d=6.6);
default p=0.0002 (d=4.2); abundant p=0.0187 (d=3.3). **Significant at every density**,
but the magnitude falls monotonically as resources become abundant.

## Interpretation

1. **Selfish reward is robust to scarcity.** Efficiency barely moves across a 2.7× change in
   resource count (0.619 → 0.648). Individually-motivated agents collect roughly the same
   proportion of whatever is available.
2. **Cooperative reward is fragile.** Efficiency nearly doubles from scarce to abundant
   (0.297 → 0.570). The team-average signal only works when resources are plentiful enough
   that inefficiency is not punished.
3. **Scarcity is the mechanism's amplifier.** The gap grows monotonically as resources become
   scarce, on all three metrics. Under abundance the fairness gap essentially vanishes
   (+0.009). This is exactly what the social-dilemma account predicts: when there is enough
   for everyone, free-riding is cheap; when resources are contested, free-riding is ruinous.
4. **This is a boundary condition, not a weakening.** The result generalises (significant at
   every density tested) *and* we can now say precisely when it matters most. A paper that
   states the conditions under which its effect is strong or weak is more credible than one
   claiming a universal effect.

## Effect on the paper
- Answers the "single environment configuration" threat to validity.
- Adds a second dose–response axis (scarcity) alongside the α-sweep, both pointing at the
  same mechanism.
- New claim: *the harm of team-average reward scales with resource scarcity; it is largely
  benign under abundance.*

Caveat: 3 seeds at the non-default densities; the r15/r40 confidence intervals are
correspondingly wider, though all comparisons remain significant.
