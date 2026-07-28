# Roadmap: dissertation → workshop paper

Goal: a ~8-page paper with a defensible, statistically-supported central claim, ready to
submit to a MARL-focused workshop. Ordered by dependency — do not skip Phase 0/1.

---

## Phase 0 — Freeze, fork, reproduce (integrity)

Make the pipeline trustworthy before we scale it up.

- [ ] **Fork the code.** Create a working copy we're allowed to edit. Recommended:
      a git branch `paper` in the repo, *plus* this folder for artifacts. (A branch keeps
      the original files untouched on `main` while giving us a mutable tree. Alternatively
      copy `agents/ env/ train/ analysis/` into `research-paper/code/` — heavier, but
      fully isolated. Decision pending — see `notes/decisions.md`.)
- [ ] **Resolve the reward-equation mismatch** (`ISSUES.md` #1). Decide whether the
      published finding used `team_avg + 0.1*own` (what the code does) or the pure team
      average (what the paper says). Re-run to confirm which reproduces the 50k numbers,
      then make code and paper agree.
- [ ] **Seed everything** (`ISSUES.md` #3): NumPy *and* PyTorch weight init *and* Python
      `random`, from a single `--seed`. This is the prerequisite for Phase 1.
- [ ] **Fix truncation bootstrapping** (`ISSUES.md` #2): 250-step truncations should
      bootstrap from the value estimate, not be treated as terminal.
- [ ] **Reproduce the headline ordering** (selfish > mixed > cooperative) once on the
      forked code to confirm nothing broke. Sanity gate before spending compute.

## Phase 1 — Statistical rigour (non-negotiable for publication)

The paper itself names single-seed as its most significant limitation (§5.5, §6.3.1).
Reviewers will reject a headline claim from n=1. This phase converts anecdote → result.

- [ ] Run **≥5 seeds** (target 8–10) × 3 conditions. ~15–30 runs. Each 50k-episode run is
      CPU-only; budget wall-clock and consider dropping to 20–30k episodes if the plateau
      is reached earlier (justify from the curves).
- [ ] Report **mean ± std / 95% CI** on the four metrics over the final-100-episode window.
- [ ] Plot learning curves with **confidence bands** across seeds (not a single trace).
- [ ] **Significance test** the pairwise gaps (Welch's t-test or Mann–Whitney U on the
      per-seed final metrics). State effect sizes, not just p-values.
- [ ] Decide the story: does "selfish > cooperative" survive across seeds? If it narrows
      or flips, that is *still* a paper — just a more honest one.

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
