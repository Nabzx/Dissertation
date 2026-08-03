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
- [x] Plot learning curves with **confidence bands** across seeds — `experiments/plot_curves.py`
      → `results/figures/learning_curves.png`. _(2026-07-30)_
- [x] **Significance test** — Welch t-tests + Cohen's d in `aggregate.py`. _(2026-07-29)_
- [x] **Story decided:** *{selfish, mixed} ≫ cooperative*, p<0.001, d=4–6 for
      selfish>cooperative. selfish vs mixed is small/mostly n.s. — claim the honest version,
      not a strict 3-way ordering. _(2026-07-29)_

## Phase 2 — Deepen the contribution (pick 1–2, highest leverage first)

A single gridworld with 3 discrete conditions is thin for a standalone paper. Add depth:

- [x] **α-sweep DONE** _(2026-07-30)_: 5 α × 3 seeds run; curve built
      (`figures/alpha_sweep.png`, `results/phase2_alpha.md`). Steep rise α=0→0.25 then
      plateau; knee at α≈0.25; endpoints byte-match standalone selfish/cooperative.
- [x] **A non-circular cooperation metric DONE** _(2026-07-30, resolves `ISSUES.md` #4)_:
      free-rider fraction + contribution Gini (`experiments/freerider.py`,
      `results/freerider.md`). Cooperative free-rides significantly MORE than selfish
      (0.307 vs 0.251, p=0.002, d=3.0) and mixed — direct evidence for the mechanism.
      Added to paper as Table 2.
- [ ] **Generalisation sweep** (optional): vary resource density and/or agent count to show
      the finding isn't specific to 25 resources / 4 agents / one map.
- [~] **Independent-policies ablation WIRED** _(2026-07-30)_: `agents/independent_ppo.py`
      (one net per agent), `--independent` flag, `run_ablation.sh`. Verified: nets diverge,
      shared-weight path unchanged (no regression). RUN PENDING (~5h) — then insert results
      into the paper's threats-to-validity section.
- [ ] **MAPPO centralised-critic baseline** (optional, higher cost) — bank for a full-paper
      version.

## Phase 3 — Write the paper (~8 pages)

- [x] Draft in `paper/` — full first draft `paper/main.tex` (ACM sigconf) + `references.bib`,
      all sections, Table 1 + Figs 2–4 with real numbers. _(2026-07-30)_ Venue: ALA@AAMAS
      (8pg, non-archival, double-blind), targeting ALA 2027 (~Feb 2027 deadline).
- [x] Rebuild figures from Phase 1/2 data (confidence bands, α-curve). _(2026-07-30)_
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
