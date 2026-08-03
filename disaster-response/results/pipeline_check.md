# Pipeline check (Phase 1b–1d plumbing)

A 60-episode × 2-seed × 2-α sweep, run purely to verify the chain
`run_alpha.sh → train_disaster → aggregate.py → plot_mandate.py` works end to end.
**These are not results** — 60 episodes is nowhere near convergence.

## Verified working
- Sweep launcher, parallel jobs, per-run CSV + summary.json
- Aggregation with Welch t-tests and 95% CIs (reusing the paper branch's stats)
- Mandate figure generation

## One observation worth recording
At 60 episodes, α=0 (collective credit) is *worse* than α=1 (individual): 1.70 vs 5.10 lives
saved. That is **not** evidence against H1 — it is the early-learning effect already
established in the gridworld paper, where collective reward has a noisier gradient and
therefore learns more slowly.

This matters for interpreting the real sweep, because two effects push in **opposite
directions**:

| Effect | Favours | When it dominates |
|---|---|---|
| Credit-assignment noise (collective reward learns slowly) | high α | early training |
| Triage distortion — H1 (individual credit discourages joint rescues) | low α | after convergence |

So the sweep must run long enough to converge before H1 can be tested at all. If we read the
severe-save-rate curve too early we will measure learning speed and mistake it for triage
behaviour. Convergence should be confirmed from the learning curves before interpreting the
final-window numbers.
