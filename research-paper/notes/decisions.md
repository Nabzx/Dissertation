# Decisions log

Running log of choices that affect the paper. Newest first.

## Open decisions (need Nabil's input)
- **Episode budget for multi-seed runs:** E1 ran fine at 30k; curves plateau well before.
  Leaning 30k for the sweep (~8h/night). — _confirm_
- **Second Phase 2 axis:** α-sweep chosen. Optional companion (independent cooperation
  metric vs. generalisation sweep vs. MAPPO baseline) — _undecided_

## Made
- 2026-07-29 — **E1 reproduction PASSED.** Fixed code (30k, seeded, truncation-bootstrapped)
  reproduces selfish > mixed > cooperative on efficiency (0.655 > 0.631 > 0.531), fairness
  (0.714 > 0.684 > 0.652) and cooperation (0.478 > 0.437 > 0.351). Ordering holds; gap is
  sharper than frozen.
- 2026-07-29 — **Reward variant DECIDED: `team_avg`** (resolves `ISSUES.md` #1). It matches
  the dissertation eq 2.7, is the α=0 endpoint of the reward spectrum
  `r_i = α·own + (1−α)·team_avg` (selfish α=1, mixed α=0.5, cooperative α=0), and gives a
  cleaner/sharper result. `plus_own` is off-spectrum and dropped from the paper.
  `run_sweep.sh` default set to `team_avg`.
- 2026-07-29 — **Bug fixed:** `run_name` omitted the cooperative variant, so E1b (team_avg)
  overwrote E1a (plus_own). run_name now includes the variant tag. plus_own cooperative E1
  data was lost but is irrelevant (off-spectrum, unused).
- 2026-07-28 — Created this workspace; agreed nothing outside `research-paper/` is edited.
- 2026-07-28 — **Fork strategy: git branch `paper`** (main stays frozen).
- 2026-07-28 — **Phase 2 depth: α-sweep** confirmed as the primary added contribution.
- 2026-07-28 — **Phase 0 code fixes implemented on branch `paper`** (uncommitted):
  - Seeding: `set_global_seeds()` seeds Python/NumPy/**PyTorch**; `--seed` flag; disjoint
    per-seed episode streams (`seed*num_episodes + episode`; seed 0 reproduces original).
  - Truncation bootstrap: per-step `done` = termination only; on truncation each agent's GAE
    tail bootstraps from `V(s_T)` via new `PPOAgent.get_value()`; `update()` accepts
    per-trajectory `last_value`/`last_done`.
  - Reward variant made configurable (`--cooperative-variant plus_own|team_avg`).
  - Verified: smoke test passes (determinism holds, seeds diverge, all schemes finite).
- 2026-07-28 — **Calibration on M1:** ~0.26 s/episode → 30k run ≈ 2.1 h, 50k ≈ 3.5 h.
  Full 3 schemes × 5 seeds × 30k ≈ 8 h at JOBS=4 (fits one overnight). All local/free.
- 2026-07-28 — **Built unattended launchers** `experiments/run_sweep.sh` (parallel
  multi-seed) + `run_repro.sh` (E1). Outputs → `research-paper/runs/` (git-ignored via
  `research-paper/.gitignore`). Full pipeline smoke-tested end-to-end (training + auto
  analysis plots). **Decision: 30k episodes, JOBS=4, single-thread per job.**
- 2026-07-28 — Commit style confirmed: short lowercase student-style, ~10 per push,
  no Claude co-author trailer.
