# Roadmap: dissertation → workshop paper

Goal: a ~8-page paper with a defensible, statistically-supported central claim, ready to
submit to a MARL-focused workshop. Ordered by dependency — do not skip Phase 0/1.

---

## Phase 0 — Freeze, fork, reproduce (integrity)

Make the pipeline trustworthy before we scale it up.

- [x] **Fork the code.** Working on git branch `paper`; `main` stays frozen. _(2026-07-28)_
- [~] **Resolve the reward-equation mismatch** (`ISSUES.md` #1). Code now supports both via
      `--cooperative-variant plus_own|team_avg` (default preserves frozen behaviour). Still
      need the E1 run to confirm which reproduces the 50k numbers, then standardise.
- [x] **Seed everything** (`ISSUES.md` #3): `set_global_seeds()` seeds Python/NumPy/PyTorch;
      `--seed` flag; disjoint per-seed episode streams. Verified deterministic. _(2026-07-28)_
- [x] **Fix truncation bootstrapping** (`ISSUES.md` #2): per-step done = termination only;
      truncation bootstraps each agent from `V(s_T)`. _(2026-07-28)_
- [ ] **Reproduce the headline ordering** (selfish > mixed > cooperative) — run `run_repro.sh`
      (E1). Sanity gate before the full multi-seed sweep. Launcher built + smoke-tested.

## Phase 1 — Statistical rigour (non-negotiable for publication)

The paper itself names single-seed as its most significant limitation (§5.5, §6.3.1).
Reviewers will reject a headline claim from n=1. This phase converts anecdote → result.

- [x] Run **5 seeds** × 3 conditions at 30k episodes (15 runs). _(2026-07-29)_
- [x] Report **mean ± std / 95% CI** on the metrics — see `results/phase1_multiseed.md`.
- [ ] Plot learning curves with **confidence bands** across seeds (not a single trace). ← NEXT
- [x] **Significance test** — Welch t-tests + Cohen's d in `aggregate.py`. _(2026-07-29)_
- [x] **Story decided:** *{selfish, mixed} ≫ cooperative*, p<0.001, d=4–6 for
      selfish>cooperative. selfish vs mixed is small/mostly n.s. — claim the honest version,
      not a strict 3-way ordering. _(2026-07-29)_

## Phase 2 — Deepen the contribution (pick 1–2, highest leverage first)

A single gridworld with 3 discrete conditions is thin for a standalone paper. Add depth:

- [ ] **α-sweep (highest leverage, low cost).** Mixed reward already has an `alpha`; sweep
      α ∈ {0, 0.25, 0.5, 0.75, 1.0} (0 = cooperative, 1 = selfish). Turns 3 discrete bars
      into a **curve** of cooperation/fairness/efficiency vs. selfishness. Far more
      compelling and directly tests the mechanism.
- [ ] **A non-circular cooperation metric** (`ISSUES.md` #4). Current score = efficiency ×
      fairness (circular). Add an independent measure: e.g. a free-riding index (Gini of
      per-agent contribution), counterfactual contribution, or territory-overlap.
- [ ] **Generalisation sweep** (optional): vary resource density and/or agent count to show
      the finding isn't specific to 25 resources / 4 agents / one map.
- [ ] **Stronger baseline** (optional, higher cost): a centralised-critic learner (MAPPO)
      or independent (non-shared-weight) policies, to show the effect isn't an artifact of
      shared-weight learners (paper flags this in §5.5, §6.3.2–6.3.3).

## Phase 3 — Write the paper (~8 pages)

- [ ] Draft in `paper/` following `paper/OUTLINE.md`, in the target venue's template.
- [ ] Rebuild figures from Phase 1/2 data (confidence bands, α-curve).
- [ ] **Related work**: position against sequential social dilemmas (Leibo et al. 2017),
      MAPPO (Yu et al. 2022), Jain fairness, tragedy of the commons. Replace weak
      dissertation citations (Medium / ResearchGate figures — `ISSUES.md` #6).
- [ ] Internal consistency pass: every number in the text traces to a logged run.

## Phase 4 — Polish & submit

- [ ] Reproducibility appendix: seeds, configs, released code, hardware, wall-clock.
- [ ] Supervisor review pass with Dr Afzal / Prof Anjum before submission.
- [ ] Format to venue page limit and submit.

---

## Definition of done (what "publishable" means here)

1. Central claim holds across ≥5 seeds with reported variance and a significance test.
2. No internal inconsistency between paper equations and the code that produced the numbers.
3. At least one metric that isn't circular, and at least one axis of generalisation
   (α-sweep counts) beyond the original 3 discrete conditions.
4. Positioned against real MARL literature, not blog posts.
