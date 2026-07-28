# Experiments

Specs and configs for the new runs that back the paper. Actual training code lives on the
fork (Phase 0), not here — this folder holds run specs, seeds, and pointers to output.

## Planned batches

### E1 — Reproduction (Phase 0)
- 3 conditions (selfish/cooperative/mixed), 1 seed, on the fixed fork.
- Purpose: confirm the headline ordering survives the reward/seeding/truncation fixes.

### E2 — Multi-seed (Phase 1) — the core evidence
- 3 conditions × {≥5 seeds}. Log final-100 metrics per seed for mean±std + significance.
- Seeds: fixed list (e.g. 0,1,2,3,4[,5,6,7]) applied to Python/NumPy/torch together.

### E3 — α-sweep (Phase 2) — the centrepiece figure
- Mixed reward, α ∈ {0, 0.25, 0.5, 0.75, 1.0} × {≥3 seeds each}.
- α=0 should match cooperative, α=1 should match selfish — a built-in consistency check.

### E4 (optional) — generalisation
- Vary `num_resources` (density) and/or `num_agents`; 1 sweep axis, few seeds.

## Launchers

Two unattended scripts. Both run the frozen code via `PYTHONPATH` and write everything
into `research-paper/runs/` (git-ignored) — nothing outside `research-paper/` is touched.

- `run_sweep.sh` — the multi-seed sweep. Env-var config, parallel jobs, resilient to a
  single run failing. Examples:
  ```bash
  DRYRUN=1 ./run_sweep.sh                          # print the plan only
  ./run_sweep.sh                                   # 30k eps, 5 seeds, 3 schemes, 4 jobs
  EPISODES=50000 SEEDS="0 1 2 3 4" ./run_sweep.sh
  JOBS=2 ./run_sweep.sh                            # lighter on the laptop
  ```
- `run_repro.sh` — E1 gate: seed 0, all three schemes, plus a cooperative `team_avg` run
  to compare both cooperative variants against the frozen `results/run_50000_*`.
  ```bash
  EPISODES=30000 ./run_repro.sh
  ```

**Calibration (M1, 2026-07-28):** ~0.26 s/episode. So ~2.1 h per 30k run; the full
3 schemes × 5 seeds × 30k sweep is ~8 h wall-clock at `JOBS=4` — one overnight. Jobs run
single-threaded so 4 pack onto the M1's 4 performance cores. All local, no paid compute.

## Logging contract
Every run records: config (all hyperparams + seed + reward variant), per-episode CSV,
final-100 summary JSON. No number enters the paper unless it traces to one of these.
