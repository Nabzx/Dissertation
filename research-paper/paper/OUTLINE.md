# Paper outline (~8 pages)

Working title (sharper than the dissertation title, foregrounds the finding):
**"Selfishness as an Accident of Cooperation: How Reward Structure Shapes Emergent
Coordination in a Multi-Agent Survival Gridworld"**
(alt: "When Selfish Reward Beats Cooperative Reward in Multi-Agent Reinforcement Learning")

## Target venue (primary)
**ALA — Adaptive and Learning Agents workshop @ AAMAS.** Best fit: MARL-focused,
welcomes empirical emergent-behaviour studies, student-friendly, typically non-archival
(so it doesn't burn a future full-paper submission). Format: AAMAS/Springer style,
~8 pages.

Backups: NeurIPS *Cooperative AI* workshop; AAMAS *Blue Sky / Student* track; an
ICLR/ICML workshop on emergent behaviour or social dilemmas. See `../notes/venues.md`.

## The one-sentence contribution
A controlled study in which reward structure is the *only* independent variable shows that
an explicitly cooperative (team-average) reward produces *less* fair, efficient and
coordinated behaviour than a selfish reward — because team-average reward makes free-riding
individually rational (the credit-assignment / tragedy-of-commons mechanism), demonstrated
across seeds and as a continuous function of the selfish–cooperative mixing weight α.

## Section plan (2-column, page budget approximate)

1. **Introduction (0.75 pg)** — cooperation from self-interest; the counterintuitive result
   as the hook; explicit contribution bullets.
2. **Related work (0.75 pg)** — sequential social dilemmas (Leibo et al. 2017), MAPPO
   (Yu et al. 2022), CTDE, Jain fairness, tragedy of the commons / Axelrod. Position the
   reward-as-controlled-variable design as the gap.
3. **Environment & Dec-POMDP (1 pg)** — 25×25 octagonal gridworld, 4 agents, 5×5 partial
   obs, 2% respawn, 250-step episodes. The formal tuple. (Drop the phantom action space,
   `ISSUES.md` #6.)
4. **Method (1.25 pg)** — from-scratch shared-weight PPO (2×64 MLP, GAE, clipped objective,
   entropy); the three reward regimes + the α continuum; the corrected reward equation
   (`ISSUES.md` #1); seeding & truncation-bootstrap protocol (`ISSUES.md` #2, #3).
5. **Experimental setup (0.5 pg)** — seeds, episodes, hardware, metrics (efficiency, Jain
   fairness, the new non-circular cooperation/free-riding metric), significance testing.
6. **Results (2 pg)** — (a) multi-seed learning curves with CI bands; (b) final-metric table
   with mean±std + significance; (c) the α-sweep curve; (d) qualitative trajectory/heatmap
   evidence of territorial separation. Lead with the mechanism, not the bars.
7. **Discussion (0.75 pg)** — credit assignment, correlated equilibrium, when the result
   would/ wouldn't generalise; threats to validity (shared weights, one map).
8. **Conclusion + future work (0.25 pg)** — centralised critic, scaling, richer comms.

## Figures to (re)build
- F1: Environment schematic (reuse the octagonal-arena render).
- F2: Multi-seed learning curves, mean ± CI, per condition (efficiency + fairness).
- F3: **α-sweep curve** — the new centrepiece.
- F4: Final-metrics table with significance stars.
- F5: Trajectory / heatmap qualitative panel (reuse + regenerate from multi-seed runs).

## Claims that MUST be backed by a logged multi-seed run before they go in
- "selfish > cooperative on efficiency / fairness / cooperation"
- "cooperative collapses early and recovers ~5× slower"
- any specific number (16.1, 0.644, 0.716, 0.468, etc. — all currently single-seed).
