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
- `run_alpha.sh` — Phase 2 α-sweep: `mixed` reward across α ∈ {0,0.25,0.5,0.75,1.0} × seeds.
  α=1 == selfish, α=0 == cooperative(team_avg) (endpoints double as consistency checks —
  verified exact). Dirs: `run_<EP>_mixed_a<α>_seed<N>`. ~8h for 5α × 3 seeds at JOBS=4.
  ```bash
  ./run_alpha.sh                        # full sweep
  ALPHAS="0.25 0.75" ./run_alpha.sh     # only the mid-points not already covered
  ```

**Calibration (M1, 2026-07-28):** ~0.26 s/episode. So ~2.1 h per 30k run; the full
3 schemes × 5 seeds × 30k sweep is ~8 h wall-clock at `JOBS=4` — one overnight. Jobs run
single-threaded so 4 pack onto the M1's 4 performance cores. All local, no paid compute.

- `run_ablation.sh` — independent-policies ablation (`--independent`: one network per agent
  instead of shared weights). Answers the main threat to validity. Dirs suffixed `_indep`;
  `aggregate.py` treats them as separate conditions (`selfish_indep`, `cooperative_indep`).
  ~5h for 2 schemes × 3 seeds.
  ```bash
  ./run_ablation.sh
  python aggregate.py --episodes 30000 --schemes selfish cooperative selfish_indep cooperative_indep
  ```

## Analysis

- `aggregate.py` — after a sweep, groups runs by scheme across seeds and reports the
  headline table (mean ± std, 95% CI) plus pairwise **Welch t-tests** and Cohen's d.
  Pure NumPy (no scipy). Writes markdown + json into `../runs/analysis/`.
  ```bash
  python aggregate.py --episodes 30000                       # workspace runs
  python aggregate.py --results-root ../../results --episodes 50000   # frozen data
  ```
  Validated against the frozen results: it reproduces the dissertation's 50k numbers
  exactly (n=1 → p-values n/a until the multi-seed sweep provides ≥2 seeds).
- `freerider.py` — non-circular free-riding metrics (free-rider fraction, contribution Gini)
  with Welch tests. No training needed — runs on existing data. Writes `../results/freerider.md`.
  ```bash
  python freerider.py --episodes 30000
  ```
- `plot_curves.py` — multi-seed learning curves with ±1 s.d. bands (paper Figure 2).
  Streams the per-seed CSVs, computes per-episode efficiency & Jain fairness, averages
  across seeds. Writes `../results/figures/learning_curves.png`.
  ```bash
  python plot_curves.py --episodes 30000
  ```

## Logging contract
Every run records: config (all hyperparams + seed + reward variant), per-episode CSV,
final-100 summary JSON. No number enters the paper unless it traces to one of these.
