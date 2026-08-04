# Convergence probe — PPO clears the bar

Single run, α=1, 6000 episodes, 40×40 / 6 responders / 20 victims / 300 steps.
Figure: `figures/learning_disaster_6000_a1_seed0_probe.png`.

## The ladder at this configuration

| Policy | lives saved /20 | share of ceiling |
|---|---|---|
| Random | 0.50 | 3% |
| Greedy (decentralised, uncoordinated) | 6.02 | 30% |
| **PPO (6000 episodes)** | **13.90** | **70%** |
| Coordinated (privileged ceiling) | 19.97 | 100% |

**PPO overtakes greedy at episode ~491** and finishes at 2.3× greedy. The coordination claim
is therefore supportable: learned policies substantially exceed a competent uncoordinated
heuristic, while still leaving headroom below the privileged ceiling.

## Convergence

Curves flatten around **episode 4000–5000**:

| Metric | final | flattens |
|---|---|---|
| Lives saved | 13.9 | ~4000 |
| Severe save rate | 0.87 | ~4000 |
| Minor save rate | 0.93 | ~1500 |
| Policy entropy | 0.41 | still declining (annealing runs to the end) |

Minor victims are learned much faster than severe ones — unsurprising, since a minor rescue
needs one agent and a severe rescue needs a rendezvous. **Coordination is the slow part of
learning**, which is itself worth reporting.

**Sweep sizing: 8000 episodes** — comfortably past the plateau with margin.

## Why the sweep uses a different configuration

This probe ran on the `base` config, where the coordinated ceiling is 0.99 — everyone can be
saved, so triage never binds and H1 cannot express itself. The sweep uses `harder`
(50×50, 6 responders, 40 victims, 300 steps; ceiling 0.80) for the reasons in
`config_selection.md`. Absolute numbers will therefore be lower than the 13.9 here; what
matters is the severe-vs-minor gap *across* α.
