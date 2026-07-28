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

## Logging contract
Every run records: config (all hyperparams + seed + reward variant), per-episode CSV,
final-100 summary JSON. No number enters the paper unless it traces to one of these.
