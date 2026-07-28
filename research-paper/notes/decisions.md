# Decisions log

Running log of choices that affect the paper. Newest first.

## Open decisions (need Nabil's input)
- **Reward variant for the paper** (`ISSUES.md` #1): reproduce which of `team_avg+0.1*own`
  (`plus_own`) vs pure `team_avg` matches the frozen 50k numbers, then standardise. Code now
  supports both via `--cooperative-variant`; default `plus_own` preserves frozen behaviour.
  — _pending E1 reproduction run_
- **Episode budget for multi-seed runs:** keep 50k or drop to 20–30k if plateau is earlier?
  Depends on wall-clock per run. — _undecided (measure during E1)_
- **Second Phase 2 axis:** α-sweep chosen. Optional companion (independent cooperation
  metric vs. generalisation sweep vs. MAPPO baseline) — _undecided_

## Made
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
